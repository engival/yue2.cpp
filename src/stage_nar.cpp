// yue2-nar — YuE2-3B NAR acoustic flow matching (semantic tokens -> VAE latent) on ggml.
// See SPEC_NAR.md and src/STATUS_NAR.md.
//
// Two GGUFs are loaded: the stage-2 AR file (qwen3 arch: token_embd, the AR half
// of every layer, output_norm) and the NAR file (the nar_* half plus vae2llm,
// llm2vae, the timestep embedder and the latent position table). Phase A runs the
// AR prefill once per chunk and keeps the post-RoPE K and the raw V of all 28
// layers; phase B evaluates the flow-matching velocity 2*steps times over a
// midpoint ODE integrated backwards from t = 1 to t = 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "stage_nar.hpp"

#include "common/device.hpp"
#include "common/gguf_kv.hpp"
#include "common/noise.hpp"
#include "common/util.hpp"

#include "npy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

// Private to this translation unit: stage_ar/stage_nar/stage_vae each have
// their own Config / Model / Builder / Graph, and `yue2` links all three.
namespace
{

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// protocol.py:7-12
static const int32_t MUSIC_END    = 151852;
static const int32_t CODEC_OFFSET = 151853;
static const int32_t CODEC_SIZE   = 32768;
static const int64_t CONTEXT      = 24576;

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

struct Config
{
	int     n_layer    = 28;
	int     n_embd     = 2048;
	int     n_ff       = 6144;
	int     n_head     = 16;
	int     n_head_kv  = 8;
	int     head_dim   = 128;
	float   eps        = 1e-6f;
	float   rope_base  = 1e6f;
	int     latent_dim = 64;
	int64_t max_frames = 24576;
	float   t_shift    = 1.0f;
	int     time_freq  = 256;
};

struct Model
{
	Config cfg;
	ggml_context *        ctx  = nullptr;
	ggml_backend_buffer_t abuf = nullptr;   // AR weights
	ggml_backend_buffer_t nbuf = nullptr;   // NAR weights
	std::map<std::string, ggml_tensor *> tensors;

	ggml_tensor * get(const std::string & name) const
	{
		auto it = tensors.find(name);
		if (it == tensors.end())
		{
			die("missing tensor '%s' in the GGUF pair", name.c_str());
		}
		return it->second;
	}
};

// Loads every tensor of one GGUF into model.tensors. With widen_f16 the F16
// weights are converted to F32 at load: ggml's CPU mul_mat feeds an F16 src0
// through vec_dot_f16, which rounds the *activations* to F16 as well (~1.7e-4
// relative per matmul). That costs ~25 dB against the f32 torch golden, so the
// CPU acceptance runs widen. See src/STATUS_NAR.md (deviation 1).
static ggml_backend_buffer_t load_gguf(const char * path, const char * arch, ggml_backend_t backend,
                                       Model & model, bool widen_f16, gguf_context ** gc_out)
{
	gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
	gguf_context * gc = gguf_init_from_file(path, gp);
	if (gc == nullptr)
	{
		die("cannot open GGUF '%s'", path);
	}

	const std::string have = kv_str(gc, "general.architecture");
	if (!have.empty() && have != arch)
	{
		die("GGUF '%s' has general.architecture '%s', expected '%s'", path, have.c_str(), arch);
	}

	const int64_t n_tensors = gguf_get_n_tensors(gc);
	if (n_tensors == 0)
	{
		die("GGUF '%s' contains no tensors", path);
	}

	for (int64_t i = 0; i < n_tensors; i++)
	{
		const char *    name = gguf_get_tensor_name(gc, i);
		const int64_t * ne   = gguf_get_tensor_ne(gc, i);
		ggml_type       type = gguf_get_tensor_type(gc, i);
		if (widen_f16 && type == GGML_TYPE_F16)
		{
			type = GGML_TYPE_F32;
		}
		ggml_tensor * t = ggml_new_tensor_4d(model.ctx, type, ne[0], ne[1], ne[2], ne[3]);
		ggml_set_name(t, name);
		if (model.tensors.count(name) != 0)
		{
			die("tensor '%s' appears in both GGUFs", name);
		}
		model.tensors[name] = t;
	}

	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
	if (buf == nullptr)
	{
		die("failed to allocate the weight buffer for '%s'", path);
	}

	FILE * f = fopen(path, "rb");
	if (f == nullptr)
	{
		die("cannot reopen GGUF '%s'", path);
	}
	const size_t data_offset = gguf_get_data_offset(gc);

	std::vector<uint8_t> raw;
	std::vector<float>   conv;
	for (int64_t i = 0; i < n_tensors; i++)
	{
		const char *  name   = gguf_get_tensor_name(gc, i);
		const size_t  nbytes = gguf_get_tensor_size(gc, i);
		const ggml_type type = gguf_get_tensor_type(gc, i);
		ggml_tensor * t = model.tensors[name];
		const bool    widened = type == GGML_TYPE_F16 && t->type == GGML_TYPE_F32;
		if (!widened && ggml_nbytes(t) != nbytes)
		{
			die("tensor '%s' size mismatch (%zu vs %zu)", name, ggml_nbytes(t), nbytes);
		}
		raw.resize(nbytes);
		if (fseek(f, (long) (data_offset + gguf_get_tensor_offset(gc, i)), SEEK_SET) != 0 ||
		    fread(raw.data(), 1, nbytes, f) != nbytes)
		{
			die("truncated tensor data for '%s' in '%s'", name, path);
		}
		if (widened)
		{
			conv.resize((size_t) ggml_nelements(t));
			ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), conv.data(), ggml_nelements(t));
			ggml_backend_tensor_set(t, conv.data(), 0, conv.size() * sizeof(float));
		} else {
			ggml_backend_tensor_set(t, raw.data(), 0, nbytes);
		}
	}
	fclose(f);

	*gc_out = gc;
	return buf;
}

// ---------------------------------------------------------------------------
// per-chunk KV cache (outside the compute graph, so gallocr never touches it)
// ---------------------------------------------------------------------------

struct KV
{
	ggml_context *        ctx  = nullptr;
	ggml_backend_buffer_t buf  = nullptr;
	std::vector<ggml_tensor *> k;    // [head_dim, smax, n_head_kv]
	std::vector<ggml_tensor *> v;    // [smax, head_dim, n_head_kv], or [head_dim, smax, n_head_kv] with fa
	bool fa = false;
	// F32 (exact) or F16. F16 halves the cache and, more importantly, puts the two
	// attention matmuls on ggml-vulkan's f16 (coopmat) pipelines instead of the
	// much slower F32 x F32 one -- see src/STATUS_NAR_PERF.md.
	ggml_type kv_type = GGML_TYPE_F32;
	ggml_tensor * o_scratch = nullptr;
	// Velocity-graph inputs live here, not in the gallocr arena: the RoPE and
	// latent position ids are written once per chunk, and gallocr would recycle
	// a graph-owned leaf's memory the moment its last consumer is done.
	ggml_tensor * in_x   = nullptr;
	ggml_tensor * in_t   = nullptr;
	ggml_tensor * in_pos = nullptr;
	ggml_tensor * in_lat = nullptr;
	int64_t smax = 0;

	void free_all()
	{
		if (buf != nullptr)
		{
			ggml_backend_buffer_free(buf);
			buf = nullptr;
		}
		if (ctx != nullptr)
		{
			ggml_free(ctx);
			ctx = nullptr;
		}
		k.clear();
		v.clear();
		o_scratch = nullptr;
		in_x = in_t = in_pos = in_lat = nullptr;
		kv_type = GGML_TYPE_F32;
		smax = 0;
	}
};

static void kv_alloc(KV & kv, const Config & c, ggml_backend_t backend, bool fa,
                     ggml_type kv_type, int64_t smax, int64_t N, int64_t n_scratch)
{
	kv.free_all();
	kv.smax    = smax;
	kv.fa      = fa;
	kv.kv_type = kv_type;

	ggml_init_params ip = {
		/*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (2 * c.n_layer + 16),
		/*.mem_buffer =*/ nullptr,
		/*.no_alloc   =*/ true,
	};
	kv.ctx = ggml_init(ip);

	for (int l = 0; l < c.n_layer; l++)
	{
		kv.k.push_back(ggml_new_tensor_3d(kv.ctx, kv_type, c.head_dim, smax, c.n_head_kv));
		// ggml_flash_attn_ext wants V in the same layout as K; the soft_max path
		// needs it transposed so it can be the src0 of the second mul_mat.
		kv.v.push_back(fa
			? ggml_new_tensor_3d(kv.ctx, kv_type, c.head_dim, smax, c.n_head_kv)
			: ggml_new_tensor_3d(kv.ctx, kv_type, smax, c.head_dim, c.n_head_kv));
	}
	if (n_scratch > 0)
	{
		kv.o_scratch = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F32, c.head_dim, n_scratch, c.n_head);
	}

	kv.in_x   = ggml_new_tensor_2d(kv.ctx, GGML_TYPE_F32, c.latent_dim, N);
	kv.in_t   = ggml_new_tensor_1d(kv.ctx, GGML_TYPE_F32, c.time_freq);
	kv.in_pos = ggml_new_tensor_1d(kv.ctx, GGML_TYPE_I32, N);
	kv.in_lat = ggml_new_tensor_1d(kv.ctx, GGML_TYPE_I32, N);
	ggml_set_name(kv.in_x,   "x_nar");
	ggml_set_name(kv.in_t,   "t_emb");
	ggml_set_name(kv.in_pos, "nar_pos");
	ggml_set_name(kv.in_lat, "latent_pos");

	kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
	if (kv.buf == nullptr)
	{
		die("failed to allocate the %lld-token KV cache", (long long) smax);
	}
	// Attention always reads the whole cache (see block()), so the rows a prefill
	// block has not written yet must be finite, not uninitialised.
	ggml_backend_buffer_clear(kv.buf, 0);
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

struct Builder
{
	ggml_context * ctx;
	ggml_cgraph *  gf;
	const Model *  model;

	ggml_tensor * w(const std::string & n) const
	{
		return model->get(n);
	}

	ggml_tensor * rms(ggml_tensor * x, const std::string & wn) const
	{
		return ggml_mul(ctx, ggml_rms_norm(ctx, x, model->cfg.eps), w(wn));
	}

	ggml_tensor * mm(const std::string & wn, ggml_tensor * x) const
	{
		return ggml_mul_mat(ctx, w(wn), x);
	}
};

static std::string bn(int l, const char * pfx, const char * suffix)
{
	char buf[128];
	snprintf(buf, sizeof(buf), "blk.%d.%s%s", l, pfx, suffix);
	return buf;
}

// One transformer block, shared by the AR prefill (pfx "") and the NAR velocity
// (pfx "nar_"). x: [n_embd, T]. The block's K/V are written into kv at rows
// [off, off+T) and attention runs over rows [0, off+T). mask == NULL is the NAR
// reading (SPEC_NAR.md 1.6): no mask at all, AR keys included.
static ggml_tensor * block(const Builder & b, KV & kv, int l, const char * pfx,
                           ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask,
                           int64_t off, int64_t query_chunk)
{
	const Config & c = b.model->cfg;
	ggml_context * ctx = b.ctx;

	const int64_t T  = x->ne[1];
	const int64_t hd = c.head_dim;

	ggml_tensor * h = b.rms(x, bn(l, pfx, "attn_norm.weight"));

	ggml_tensor * q = ggml_reshape_3d(ctx, b.mm(bn(l, pfx, "attn_q.weight"), h), hd, c.n_head,    T);
	ggml_tensor * k = ggml_reshape_3d(ctx, b.mm(bn(l, pfx, "attn_k.weight"), h), hd, c.n_head_kv, T);
	ggml_tensor * v = ggml_reshape_3d(ctx, b.mm(bn(l, pfx, "attn_v.weight"), h), hd, c.n_head_kv, T);

	// q_norm / k_norm are RMSNorm over head_dim and run BEFORE RoPE
	// (modeling_yue2.py:185-186).
	q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.eps), b.w(bn(l, pfx, "attn_q_norm.weight")));
	k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.eps), b.w(bn(l, pfx, "attn_k_norm.weight")));

	q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
	                  c.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
	k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
	                  c.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

	// store: K as [hd, smax, n_head_kv], V transposed as [smax, hd, n_head_kv]
	ggml_tensor * kc = kv.k[l];
	ggml_tensor * vc = kv.v[l];
	ggml_build_forward_expand(b.gf, ggml_cpy(ctx,
		ggml_permute(ctx, k, 0, 2, 1, 3),
		ggml_view_3d(ctx, kc, hd, T, c.n_head_kv, kc->nb[1], kc->nb[2], (size_t) off * kc->nb[1])));
	ggml_build_forward_expand(b.gf, kv.fa
		? ggml_cpy(ctx, ggml_permute(ctx, v, 0, 2, 1, 3),
			ggml_view_3d(ctx, vc, hd, T, c.n_head_kv, vc->nb[1], vc->nb[2], (size_t) off * vc->nb[1]))
		: ggml_cpy(ctx, ggml_permute(ctx, v, 1, 2, 0, 3),
			ggml_view_3d(ctx, vc, T, hd, c.n_head_kv, vc->nb[1], vc->nb[2], (size_t) off * vc->nb[0])));

	// Deliberately NOT a [0, S) prefix view: ggml-vulkan's mul_mat reads an F32
	// src0 whose ne2 stride is not tight (a partial view of the cache) as zero for
	// every kv head past the first, which silently destroys the prefill from layer
	// 1 on. Attending over the full cache keeps src0 contiguous; the rows this
	// block has not written yet are zeroed at alloc and masked out below.
	// See src/STATUS_NAR.md (deviation 2).
	ggml_tensor * kview = kc;
	ggml_tensor * vview = vc;

	// [hd, T, n_head]; mul_mat broadcasts ne2 (16 q-heads over 8 kv-heads) for GQA.
	ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);

	const float scale = 1.0f / sqrtf((float) hd);
	// Spread the query rows evenly over ceil(T / query_chunk) tiles instead of
	// taking full tiles plus a remainder: a 15-row tail tile ran at 0.3 TFLOP/s on
	// the Arc and cost 4 % of the whole evaluation (src/STATUS_NAR_PERF.md).
	const int64_t tiles = query_chunk > 0 ? (T + query_chunk - 1) / query_chunk : 1;
	const int64_t tile  = (T + tiles - 1) / tiles;

	ggml_tensor * o = nullptr;
	if (kv.fa)
	{
		// [hd, n_head, T] straight out, exactly the o_proj input layout.
		o = ggml_flash_attn_ext(ctx, qp, kview, vview, mask, scale, 0.0f, 0.0f);
		x = ggml_add(ctx, x, b.mm(bn(l, pfx, "attn_output.weight"), ggml_reshape_2d(ctx, o, c.n_embd, T)));
	} else if (tile >= T) {
		ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kview, ggml_cont(ctx, qp)), mask, scale, 0.0f);
		o = ggml_mul_mat(ctx, vview, kq);
	} else {
		// Query tiling is a memory measure only: softmax is per query row, so the
		// result is identical (bit-identical on CPU). Tiles land in a persistent
		// scratch tensor; concatenating them would be quadratic in copies.
		GGML_ASSERT(kv.o_scratch != nullptr && kv.o_scratch->ne[1] >= T);
		ggml_tensor * os = kv.o_scratch;
		for (int64_t start = 0; start < T; start += tile)
		{
			const int64_t n = std::min(tile, T - start);
			ggml_tensor * qt = ggml_cont(ctx,
				ggml_view_3d(ctx, qp, hd, n, c.n_head, qp->nb[1], qp->nb[2], (size_t) start * qp->nb[1]));
			ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kview, qt), mask, scale, 0.0f);
			ggml_build_forward_expand(b.gf, ggml_cpy(ctx, ggml_mul_mat(ctx, vview, kq),
				ggml_view_3d(ctx, os, hd, n, c.n_head, os->nb[1], os->nb[2], (size_t) start * os->nb[1])));
		}
		o = ggml_view_3d(ctx, os, hd, T, c.n_head, os->nb[1], os->nb[2], 0);
	}

	if (!kv.fa)
	{
		// [hd, T, n_head] -> [hd, n_head, T] -> [n_embd, T]: head index slower than
		// the head dimension, which is what torch's h.flatten(1) produces.
		o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), c.n_embd, T);
		x = ggml_add(ctx, x, b.mm(bn(l, pfx, "attn_output.weight"), o));
	}

	ggml_tensor * h2 = b.rms(x, bn(l, pfx, "ffn_norm.weight"));
	ggml_tensor * up = ggml_mul(ctx,
		ggml_silu(ctx, b.mm(bn(l, pfx, "ffn_gate.weight"), h2)),
		b.mm(bn(l, pfx, "ffn_up.weight"), h2));
	return ggml_add(ctx, x, b.mm(bn(l, pfx, "ffn_down.weight"), up));
}

struct Graph
{
	ggml_context * ctx = nullptr;
	ggml_cgraph *  gf  = nullptr;
	std::vector<uint8_t> buf;

	ggml_tensor * in_a   = nullptr;   // prefill: ids | velocity: x_nar
	ggml_tensor * in_b   = nullptr;   // prefill: pos | velocity: t_emb256
	ggml_tensor * mask   = nullptr;   // prefill only
	ggml_tensor * in_pos = nullptr;   // velocity: RoPE positions
	ggml_tensor * in_lat = nullptr;   // velocity: latent position ids
	ggml_tensor * out  = nullptr;
	ggml_tensor * dbg_x_in = nullptr;
	ggml_tensor * dbg_x_l0 = nullptr;

	void init(size_t nodes)
	{
		buf.resize(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false));
		ggml_init_params ip = { buf.size(), buf.data(), /*.no_alloc =*/ true };
		ctx = ggml_init(ip);
		gf  = ggml_new_graph_custom(ctx, nodes, false);
	}

	void free_all()
	{
		if (ctx != nullptr)
		{
			ggml_free(ctx);
			ctx = nullptr;
		}
		gf = nullptr;
	}
};

// Phase A: one block of the AR prefill, tokens [off, off+B).
static void build_prefill(const Model & m, KV & kv, int64_t off, int64_t B, Graph & g)
{
	const Config & c = m.cfg;
	g.init((size_t) (256 + 96 * c.n_layer));

	Builder b { g.ctx, g.gf, &m };

	g.in_a = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, B);
	ggml_set_name(g.in_a, "ids");
	ggml_set_input(g.in_a);
	g.in_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, B);
	ggml_set_name(g.in_b, "pos");
	ggml_set_input(g.in_b);
	// ggml_flash_attn_ext only takes an F16 mask, and both the CPU and the Vulkan
	// kernels want its row count padded to a multiple of GGML_KQ_MASK_PAD.
	const int64_t mrows = kv.fa ? GGML_PAD(B, 64) : B;
	g.mask = ggml_new_tensor_2d(g.ctx, kv.fa ? GGML_TYPE_F16 : GGML_TYPE_F32, kv.smax, mrows);
	ggml_set_name(g.mask, "mask");
	ggml_set_input(g.mask);

	ggml_tensor * x = ggml_get_rows(g.ctx, b.w("token_embd.weight"), g.in_a);
	for (int l = 0; l < c.n_layer; l++)
	{
		x = block(b, kv, l, "", x, g.in_b, g.mask, off, 0);
	}
	// The prefill's final hidden state is never used (nar.py:132-149 keeps only
	// the per-layer K/V); x is expanded so the last layer is not pruned.
	ggml_set_output(x);
	g.out = x;
	ggml_build_forward_expand(g.gf, x);
}

// Phase B: one velocity evaluation. Rebuilt once per chunk, re-run 2*steps times.
static void build_velocity(const Model & m, KV & kv, int64_t ar_length, int64_t N,
                           int64_t query_chunk, bool dump, Graph & g)
{
	const Config & c = m.cfg;
	const int64_t tiles = !kv.fa && query_chunk > 0 ? (N + query_chunk - 1) / query_chunk : 1;
	g.init((size_t) (512 + (int64_t) c.n_layer * (96 + 8 * tiles)));

	Builder b { g.ctx, g.gf, &m };

	g.in_a   = kv.in_x;
	g.in_b   = kv.in_t;
	g.in_pos = kv.in_pos;
	g.in_lat = kv.in_lat;
	ggml_tensor * pos = kv.in_pos;
	ggml_tensor * lat = kv.in_lat;
	GGML_ASSERT(g.in_a->ne[1] == N);

	// x = vae2llm(x_nar) + time_embedder(t) + latent_pos_embed(0..N-1)
	ggml_tensor * x = ggml_add(g.ctx, b.mm("nar.vae2llm.weight", g.in_a), b.w("nar.vae2llm.bias"));
	ggml_tensor * te = ggml_add(g.ctx, b.mm("nar.time_embd.0.weight", g.in_b), b.w("nar.time_embd.0.bias"));
	te = ggml_add(g.ctx, b.mm("nar.time_embd.1.weight", ggml_silu(g.ctx, te)), b.w("nar.time_embd.1.bias"));
	x = ggml_add(g.ctx, x, te);
	x = ggml_add(g.ctx, x, ggml_get_rows(g.ctx, b.w("nar.latent_pos_embd.weight"), lat));

	if (dump)
	{
		ggml_set_name(x, "x_in");
		ggml_set_output(x);
		g.dbg_x_in = x;
	}

	for (int l = 0; l < c.n_layer; l++)
	{
		x = block(b, kv, l, "nar_", x, pos, nullptr, ar_length, kv.fa ? 0 : query_chunk);
		if (dump && l == 0)
		{
			ggml_set_name(x, "x_l0");
			ggml_set_output(x);
			g.dbg_x_l0 = x;
		}
	}

	// llm2vae(model.norm(x))[1:-1]   (nar.py:167; output_norm is the shared backbone norm)
	ggml_tensor * out = ggml_add(g.ctx, b.mm("nar.llm2vae.weight", b.rms(x, "output_norm.weight")),
	                             b.w("nar.llm2vae.bias"));
	out = ggml_cont(g.ctx, ggml_view_2d(g.ctx, out, c.latent_dim, N - 2, out->nb[1], out->nb[1]));
	ggml_set_name(out, "velocity");
	ggml_set_output(out);
	g.out = out;
	ggml_build_forward_expand(g.gf, out);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s --ar yue2-ar-f16.gguf -m yue2-nar-f16.gguf\n"
		"           (--artifacts DIR | --prefix prefix.npy --codec semantic.npy)\n"
		"           (--noise noise.npy | --seed N) -o latent.npy\n"
		"           [--steps 32] [--context 24576] [--query-chunk 1024] [--prefill-block 512]\n"
		"           [--gpu N] [--cpu] [--threads N]\n"
		"           [--frames N] [--flash-attn] [--kv-f16] [--dump-dir DIR] [--dump-kv-all]\n"
		"           [--weights f16|f32] [--vk-f16-matmul]\n", argv0);
}

// modeling_yue2.py:603-606 with timestep_shift from the GGUF.
static float shift_t(double raw, float shift)
{
	const float sig = 1.0f / (1.0f + expf(-(float) raw));
	return shift * sig / (1.0f + (shift - 1.0f) * sig);
}

// modeling_yue2.py:326-330, in f32 on the host.
static void timestep_embedding(float t, int size, std::vector<float> & out)
{
	const int   half = size / 2;
	const float lg   = logf(10000.0f);
	out.resize((size_t) size);
	for (int i = 0; i < half; i++)
	{
		const float f = expf(-lg * (float) i / (float) half);
		out[(size_t) i]        = cosf(t * f);
		out[(size_t) half + i] = sinf(t * f);
	}
}

static void dump_npy(const std::string & path, const std::vector<int64_t> & shape, const float * data)
{
	const std::string err = npy::save(path.c_str(), shape, data);
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
	printf("dump:    %s\n", path.c_str());
}

// The [2, ar_length, 8, 128] layout the golden uses: stack(post-RoPE K, raw V).
static void dump_ar_kv(const KV & kv, const Config & c, int l, int64_t ar_length, const std::string & path)
{
	std::vector<float> kh((size_t) ggml_nelements(kv.k[l]));
	std::vector<float> vh((size_t) ggml_nelements(kv.v[l]));
	if (kv.kv_type == GGML_TYPE_F16)
	{
		std::vector<ggml_fp16_t> raw(std::max(kh.size(), vh.size()));
		ggml_backend_tensor_get(kv.k[l], raw.data(), 0, ggml_nbytes(kv.k[l]));
		ggml_fp16_to_fp32_row(raw.data(), kh.data(), (int64_t) kh.size());
		ggml_backend_tensor_get(kv.v[l], raw.data(), 0, ggml_nbytes(kv.v[l]));
		ggml_fp16_to_fp32_row(raw.data(), vh.data(), (int64_t) vh.size());
	} else {
		ggml_backend_tensor_get(kv.k[l], kh.data(), 0, ggml_nbytes(kv.k[l]));
		ggml_backend_tensor_get(kv.v[l], vh.data(), 0, ggml_nbytes(kv.v[l]));
	}

	const int64_t hd   = c.head_dim;
	const int64_t nh   = c.n_head_kv;
	const int64_t smax = kv.smax;
	std::vector<float> out((size_t) 2 * ar_length * nh * hd);
	for (int64_t t = 0; t < ar_length; t++)
	{
		for (int64_t j = 0; j < nh; j++)
		{
			for (int64_t d = 0; d < hd; d++)
			{
				const size_t o = (size_t) ((t * nh + j) * hd + d);
				out[o] = kh[(size_t) (j * smax * hd + t * hd + d)];
				out[o + (size_t) ar_length * nh * hd] = kv.fa
					? vh[(size_t) (j * smax * hd + t * hd + d)]
					: vh[(size_t) (j * hd * smax + d * smax + t)];
			}
		}
	}
	dump_npy(path, { 2, ar_length, nh, hd }, out.data());
}

// Minimal "seed": N extraction from request.json, for reporting only.
static bool json_int(const std::string & path, const char * key, long long & out)
{
	FILE * f = fopen(path.c_str(), "rb");
	if (f == nullptr)
	{
		return false;
	}
	std::string s;
	char buf[4096];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
	{
		s.append(buf, n);
	}
	fclose(f);
	const std::string needle = std::string("\"") + key + "\"";
	size_t p = s.find(needle);
	if (p == std::string::npos)
	{
		return false;
	}
	p = s.find(':', p + needle.size());
	if (p == std::string::npos)
	{
		return false;
	}
	out = strtoll(s.c_str() + p + 1, nullptr, 10);
	return true;
}

} // namespace

NarParams parse_nar_args(const char * argv0, int argc, char ** argv)
{
	NarParams p;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--ar")
		{
			p.ar_model = need(argc, argv, i);
		} else if (a == "-m" || a == "--model") {
			p.model = need(argc, argv, i);
		} else if (a == "--artifacts") {
			p.artifacts = need(argc, argv, i);
		} else if (a == "--prefix") {
			p.prefix_path = need(argc, argv, i);
		} else if (a == "--codec") {
			p.codec_path = need(argc, argv, i);
		} else if (a == "--noise") {
			p.noise_path = need(argc, argv, i);
		} else if (a == "--seed") {
			p.have_seed = true;
			p.seed      = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "-o" || a == "--output") {
			p.output = need(argc, argv, i);
		} else if (a == "--dump-dir") {
			p.dump_dir = need(argc, argv, i);
		} else if (a == "--device") {
			p.device = need(argc, argv, i);
		} else if (a == "--cpu") {
			p.device = "cpu";
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--threads" || a == "-t") {
			p.threads = atoi(need(argc, argv, i));
		} else if (a == "--steps") {
			p.steps = atoi(need(argc, argv, i));
		} else if (a == "--context") {
			p.context = atoll(need(argc, argv, i));
		} else if (a == "--query-chunk") {
			p.query_chunk = atoll(need(argc, argv, i));
		} else if (a == "--prefill-block") {
			p.prefill_block = atoll(need(argc, argv, i));
		} else if (a == "--frames") {
			p.frames = atoll(need(argc, argv, i));
		} else if (a == "--weights") {
			const std::string v = need(argc, argv, i);
			if (v == "f32")
			{
				p.widen_f16 = true;
			} else if (v != "f16") {
				die("--weights must be f16 or f32");
			}
		} else if (a == "--flash-attn" || a == "-fa") {
			p.flash_attn = true;
		} else if (a == "--kv-f16") {
			p.kv_f16 = true;
		} else if (a == "--dump-kv-all") {
			p.dump_kv_all = true;
		} else if (a == "--vk-f16-matmul") {
			p.vk_f16_matmul = true;
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument '%s'", argv[i]);
		}
	}

	if (p.ar_model.empty() || p.model.empty())
	{
		usage(argv0);
		die("--ar and -m are required");
	}
	if (!p.artifacts.empty())
	{
		if (p.prefix_path.empty())
		{
			p.prefix_path = p.artifacts + "/prefix.npy";
		}
		if (p.codec_path.empty())
		{
			p.codec_path = p.artifacts + "/semantic.npy";
		}
		if (p.output.empty())
		{
			p.output = p.artifacts + "/latent.npy";
		}
		// The request's own seed drives the noise when neither --noise nor
		// --seed was given, so an artifacts dir on its own is a complete run.
		long long seed = 0;
		if (!p.have_seed && p.noise_path.empty() && json_int(p.artifacts + "/request.json", "seed", seed) && seed >= 0)
		{
			p.have_seed = true;
			p.seed      = (uint64_t) seed;
			printf("request: seed %lld (noise is generated from it)\n", seed);
		}
	}
	if (p.prefix_path.empty() || p.codec_path.empty())
	{
		usage(argv0);
		die("--artifacts or both of --prefix/--codec are required");
	}
	if (p.noise_path.empty() && !p.have_seed)
	{
		die("one of --noise FILE or --seed N is required");
	}
	if (p.output.empty())
	{
		die("-o is required");
	}
	if (p.steps < 1)
	{
		die("--steps must be >= 1");
	}
	if (p.context < 1 || p.context > CONTEXT)
	{
		die("--context must be in 1..%lld", (long long) CONTEXT);
	}
	if (p.prefill_block < 1)
	{
		die("--prefill-block must be >= 1");
	}
	return p;
}

int run_nar(const NarParams & p)
{
	ggml_time_init();

	// backend ---------------------------------------------------------------
	// Every attention and MLP projection here is a mul_mat, and the ODE
	// integrates 64 evaluations of a 28-layer transformer, so fp16 operand
	// staging compounds rather than cancels. vulkan_want_exact_f32() selects the
	// genuinely-F32 matmul pipelines.
	if (p.device == "vulkan" && !p.vk_f16_matmul)
	{
		vulkan_want_exact_f32();
	}

	ggml_backend_t backend = init_compute_backend(p.device, p.gpu, p.threads);
	printf("backend: %s (%s)\n",
	       ggml_backend_name(backend),
	       ggml_backend_dev_description(ggml_backend_get_device(backend)));

	// model -----------------------------------------------------------------
	Model model;
	{
		ggml_init_params ip = {
			/*.mem_size   =*/ ggml_tensor_overhead() * 1024,
			/*.mem_buffer =*/ nullptr,
			/*.no_alloc   =*/ true,
		};
		model.ctx = ggml_init(ip);
	}

	const int64_t t_load0 = now_us();
	gguf_context * gc_ar  = nullptr;
	gguf_context * gc_nar = nullptr;
	model.abuf = load_gguf(p.ar_model.c_str(), "qwen3",    backend, model, p.widen_f16, &gc_ar);
	model.nbuf = load_gguf(p.model.c_str(),    "yue2-nar", backend, model, p.widen_f16, &gc_nar);

	const std::string sha_ar  = kv_str(gc_ar,  "yue2.source_sha256");
	const std::string sha_nar = kv_str(gc_nar, "yue2.source_sha256");
	if (sha_ar.empty() || sha_nar.empty())
	{
		die("both GGUFs must carry yue2.source_sha256 (ar '%s', nar '%s')", sha_ar.c_str(), sha_nar.c_str());
	}
	if (sha_ar != sha_nar)
	{
		die("GGUF pair mismatch: ar yue2.source_sha256 %s != nar %s", sha_ar.c_str(), sha_nar.c_str());
	}

	Config & c = model.cfg;
	c.n_layer    = kv_i32(gc_nar, "yue2nar.block_count",                        c.n_layer);
	c.n_embd     = kv_i32(gc_nar, "yue2nar.embedding_length",                   c.n_embd);
	c.n_ff       = kv_i32(gc_nar, "yue2nar.feed_forward_length",                c.n_ff);
	c.n_head     = kv_i32(gc_nar, "yue2nar.attention.head_count",               c.n_head);
	c.n_head_kv  = kv_i32(gc_nar, "yue2nar.attention.head_count_kv",            c.n_head_kv);
	c.head_dim   = kv_i32(gc_nar, "yue2nar.attention.key_length",               c.head_dim);
	c.eps        = kv_f32(gc_nar, "yue2nar.attention.layer_norm_rms_epsilon",   c.eps);
	c.rope_base  = kv_f32(gc_nar, "yue2nar.rope.freq_base",                     c.rope_base);
	c.latent_dim = kv_i32(gc_nar, "yue2nar.latent_dim",                         c.latent_dim);
	c.max_frames = kv_i32(gc_nar, "yue2nar.max_latent_frames",       (int) c.max_frames);
	c.t_shift    = kv_f32(gc_nar, "yue2nar.timestep_shift",                     c.t_shift);
	c.time_freq  = kv_i32(gc_nar, "yue2nar.time_embd_frequency_size",           c.time_freq);

	const int64_t vocab = model.get("token_embd.weight")->ne[1];
	printf("load:    %.3f s (%zu tensors, %.1f MiB + %.1f MiB)\n",
	       (now_us() - t_load0) / 1e6, model.tensors.size(),
	       ggml_backend_buffer_get_size(model.abuf) / 1024.0 / 1024.0,
	       ggml_backend_buffer_get_size(model.nbuf) / 1024.0 / 1024.0);
	printf("config:  %d layers, n_embd %d, heads %d/%d x %d, eps %g, rope %g, shift %g, vocab %lld\n",
	       c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim,
	       (double) c.eps, (double) c.rope_base, (double) c.t_shift, (long long) vocab);
	if (c.n_head % c.n_head_kv != 0)
	{
		die("n_head %d is not a multiple of n_head_kv %d", c.n_head, c.n_head_kv);
	}

	gguf_free(gc_ar);
	gguf_free(gc_nar);

	// inputs ----------------------------------------------------------------
	// `yue2 song` passes the AR's tokens straight over; everyone else reads the
	// two .npy files the AR stage wrote.
	npy::ArrayI32 prefix, codec;
	std::string   err;
	if (p.prefix_in != nullptr && p.codec_in != nullptr)
	{
		prefix.data = *p.prefix_in;
		codec.data  = *p.codec_in;
		prefix.shape.push_back((int64_t) prefix.data.size());
		codec.shape.push_back((int64_t) codec.data.size());
	} else {
		err = npy::load_i32(p.prefix_path.c_str(), prefix);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
		err = npy::load_i32(p.codec_path.c_str(), codec);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
	}
	if (prefix.shape.size() != 1 || codec.shape.size() != 1)
	{
		die("--prefix and --codec must be 1-D int32 arrays");
	}

	int64_t T = (int64_t) codec.data.size();
	if (p.frames > 0)
	{
		if (p.frames > T)
		{
			die("--frames %lld exceeds the %lld codec frames", (long long) p.frames, (long long) T);
		}
		T = p.frames;
	}
	if (T < 1)
	{
		die("empty codec");
	}
	const int64_t P = (int64_t) prefix.data.size();
	if (P < 1)
	{
		die("empty prefix");
	}
	for (int64_t i = 0; i < P; i++)
	{
		if (prefix.data[(size_t) i] < 0 || prefix.data[(size_t) i] >= vocab)
		{
			die("prefix[%lld] = %d outside [0, %lld)", (long long) i, prefix.data[(size_t) i], (long long) vocab);
		}
	}
	for (int64_t i = 0; i < T; i++)
	{
		if (codec.data[(size_t) i] < 0 || codec.data[(size_t) i] >= CODEC_SIZE)
		{
			die("codec[%lld] = %d outside [0, %d)", (long long) i, codec.data[(size_t) i], CODEC_SIZE);
		}
	}

	npy::Array noise;
	if (p.noise_in != nullptr)
	{
		noise.data  = *p.noise_in;
		noise.shape = { (int64_t) noise.data.size() / c.latent_dim, c.latent_dim };
	} else if (!p.noise_path.empty()) {
		err = npy::load(p.noise_path.c_str(), noise);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
	} else {
		noise.data  = noise::gaussian(p.seed, T, c.latent_dim);
		noise.shape = { T, c.latent_dim };
		printf("noise:   mt19937_64(%llu) + Box-Muller, [%lld, %d]\n",
		       (unsigned long long) p.seed, (long long) T, c.latent_dim);
	}
	if (noise.shape.size() != 2 || noise.shape[0] < T || noise.shape[1] != c.latent_dim)
	{
		die("noise must be float32 [>=%lld, %d]", (long long) T, c.latent_dim);
	}
	for (int64_t i = 0; i < T * c.latent_dim; i++)
	{
		if (!std::isfinite(noise.data[(size_t) i]))
		{
			die("--noise contains non-finite values");
		}
	}

	if (!p.dump_dir.empty())
	{
		mkdir(p.dump_dir.c_str(), 0755);
	}

	// chunking (protocol.py:142-145) ----------------------------------------
	const int64_t size = std::min((p.context - P - 3) / 2, CONTEXT);
	if (size < 1)
	{
		die("prefix of %lld tokens leaves no acoustic context at --context %lld",
		    (long long) P, (long long) p.context);
	}
	std::vector<std::pair<int64_t, int64_t>> ranges;
	for (int64_t a = 0; a < T; a += size)
	{
		ranges.emplace_back(a, std::min(a + size, T));
	}
	printf("chunks:  %zu, size %lld (prefix %lld, frames %lld, steps %d)\n",
	       ranges.size(), (long long) size, (long long) P, (long long) T, p.steps);

	std::vector<float> out((size_t) T * c.latent_dim, 0.0f);
	const int64_t t_all0 = now_us();

	for (size_t ci = 0; ci < ranges.size(); ci++)
	{
		const int64_t a  = ranges[ci].first;
		const int64_t bb = ranges[ci].second;
		const int64_t Tc = bb - a;
		const int64_t ar_length  = P + Tc + 1;
		const int64_t N          = Tc + 2;
		const int64_t S          = ar_length + N;
		if (S > CONTEXT)
		{
			die("chunk %zu needs %lld tokens, over the %lld context", ci, (long long) S, (long long) CONTEXT);
		}

		std::vector<int32_t> ids;
		ids.reserve((size_t) ar_length);
		ids.insert(ids.end(), prefix.data.begin(), prefix.data.begin() + (size_t) P);
		for (int64_t i = a; i < bb; i++)
		{
			ids.push_back(codec.data[(size_t) i] + CODEC_OFFSET);
		}
		ids.push_back(MUSIC_END);

		printf("chunk %zu/%zu: frames [%lld, %lld), ar_length %lld, nar_length %lld, S %lld\n",
		       ci + 1, ranges.size(), (long long) a, (long long) bb,
		       (long long) ar_length, (long long) N, (long long) S);

		KV kv;
		kv_alloc(kv, c, backend, p.flash_attn, p.kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, S, N,
		         !p.flash_attn && p.query_chunk > 0 && p.query_chunk < N ? N : 0);
		printf("         kv %.1f MiB\n", ggml_backend_buffer_get_size(kv.buf) / 1024.0 / 1024.0);

		// ---- phase A: AR prefill ----
		const int64_t t_pre0 = now_us();
		{
			ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
			std::vector<int32_t>     pos;
			std::vector<float>       mask;
			std::vector<ggml_fp16_t> mask16;
			for (int64_t off = 0; off < ar_length; off += p.prefill_block)
			{
				const int64_t B = std::min(p.prefill_block, ar_length - off);
				Graph g;
				build_prefill(model, kv, off, B, g);
				if (!ggml_gallocr_alloc_graph(ga, g.gf))
				{
					die("failed to allocate the prefill graph (block at %lld)", (long long) off);
				}
				pos.resize((size_t) B);
				for (int64_t i = 0; i < B; i++)
				{
					pos[(size_t) i] = (int32_t) (off + i);
				}
				const int64_t mrows = g.mask->ne[1];
				mask.assign((size_t) S * mrows, -INFINITY);
				for (int64_t i = 0; i < B; i++)
				{
					for (int64_t j = 0; j < S; j++)
					{
						mask[(size_t) (i * S + j)] = j <= off + i ? 0.0f : -INFINITY;
					}
				}
				ggml_backend_tensor_set(g.in_a, &ids[(size_t) off], 0, (size_t) B * sizeof(int32_t));
				ggml_backend_tensor_set(g.in_b, pos.data(), 0, (size_t) B * sizeof(int32_t));
				if (g.mask->type == GGML_TYPE_F16)
				{
					mask16.resize(mask.size());
					ggml_fp32_to_fp16_row(mask.data(), mask16.data(), (int64_t) mask.size());
					ggml_backend_tensor_set(g.mask, mask16.data(), 0, mask16.size() * sizeof(ggml_fp16_t));
				} else {
					ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * sizeof(float));
				}
				if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS)
				{
					die("prefill graph compute failed");
				}
				g.free_all();
			}
			ggml_gallocr_free(ga);
		}
		const double pre_s = (now_us() - t_pre0) / 1e6;
		printf("         prefill %.3f s (%.1f tok/s)\n", pre_s, ar_length / pre_s);

		if (!p.dump_dir.empty() && ci == 0)
		{
			if (p.dump_kv_all)
			{
				for (int l = 0; l < c.n_layer; l++)
				{
					dump_ar_kv(kv, c, l, ar_length,
						p.dump_dir + "/nar_ar_kv_l" + std::to_string(l) + ".npy");
				}
			} else {
				dump_ar_kv(kv, c, 0,             ar_length, p.dump_dir + "/nar_ar_kv_l0.npy");
				dump_ar_kv(kv, c, c.n_layer - 1, ar_length, p.dump_dir + "/nar_ar_kv_l27.npy");
			}
		}

		// ---- phase B: the ODE ----
		const bool dump = !p.dump_dir.empty() && ci == 0;
		Graph g;
		build_velocity(model, kv, ar_length, N, p.query_chunk, dump, g);
		ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
		if (!ggml_gallocr_alloc_graph(ga, g.gf))
		{
			die("failed to allocate the velocity graph");
		}
		printf("         velocity graph: %d nodes, %.1f MiB\n",
		       ggml_graph_n_nodes(g.gf), ggml_gallocr_get_buffer_size(ga, 0) / 1024.0 / 1024.0);

		{
			std::vector<int32_t> pos((size_t) N), lat((size_t) N);
			for (int64_t i = 0; i < N; i++)
			{
				pos[(size_t) i] = (int32_t) (ar_length + i);
				lat[(size_t) i] = (int32_t) std::min(i, c.max_frames - 1);
			}
			ggml_backend_tensor_set(g.in_pos, pos.data(), 0, (size_t) N * sizeof(int32_t));
			ggml_backend_tensor_set(g.in_lat, lat.data(), 0, (size_t) N * sizeof(int32_t));
		}

		const int64_t D = c.latent_dim;
		std::vector<float> state((size_t) Tc * D);
		memcpy(state.data(), &noise.data[(size_t) a * D], (size_t) Tc * D * sizeof(float));

		std::vector<float> x_nar((size_t) N * D, 0.0f);
		std::vector<float> vel((size_t) Tc * D);
		std::vector<float> mid((size_t) Tc * D);
		std::vector<float> temb;

		int    n_eval = 0;
		double ode_us = 0.0;

		auto velocity = [&](const std::vector<float> & st, double raw, std::vector<float> & vout)
		{
			memset(x_nar.data(), 0, x_nar.size() * sizeof(float));
			memcpy(&x_nar[(size_t) D], st.data(), (size_t) Tc * D * sizeof(float));
			timestep_embedding(shift_t(raw, c.t_shift), c.time_freq, temb);

			ggml_backend_tensor_set(g.in_a, x_nar.data(), 0, x_nar.size() * sizeof(float));
			ggml_backend_tensor_set(g.in_b, temb.data(), 0, temb.size() * sizeof(float));

			const int64_t t0 = now_us();
			if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS)
			{
				die("velocity graph compute failed");
			}
			ode_us += (double) (now_us() - t0);
			n_eval++;

			vout.resize((size_t) Tc * D);
			ggml_backend_tensor_get(g.out, vout.data(), 0, (size_t) Tc * D * sizeof(float));

			if (dump && n_eval == 1)
			{
				std::vector<float> h;
				if (g.dbg_x_in != nullptr)
				{
					h.resize((size_t) ggml_nelements(g.dbg_x_in));
					ggml_backend_tensor_get(g.dbg_x_in, h.data(), 0, ggml_nbytes(g.dbg_x_in));
					dump_npy(p.dump_dir + "/nar_x_in_step0.npy", { N, c.n_embd }, h.data());
				}
				if (g.dbg_x_l0 != nullptr)
				{
					h.resize((size_t) ggml_nelements(g.dbg_x_l0));
					ggml_backend_tensor_get(g.dbg_x_l0, h.data(), 0, ggml_nbytes(g.dbg_x_l0));
					dump_npy(p.dump_dir + "/nar_x_l0_step0.npy", { N, c.n_embd }, h.data());
				}
				dump_npy(p.dump_dir + "/nar_v_step0.npy", { Tc, D }, vout.data());
			}
		};

		// nar.py:169-198 — midpoint (RK2), integrating backwards from t = 1.
		const double dt = 1.0 / p.steps;
		std::vector<double> raws;
		for (int k = 0; k < p.steps; k++)
		{
			const double t    = 1.0 - k * dt;
			const double raw  = std::max(-20.0, std::min(20.0, log(t / (1.0 - t))));
			const double raw2 = std::max(-20.0, std::min(20.0, log((t - dt / 2) / (1.0 - (t - dt / 2)))));
			raws.push_back(raw);
			raws.push_back(raw2);

			velocity(state, raw, vel);
			for (size_t i = 0; i < state.size(); i++)
			{
				mid[i] = state[i] - vel[i] * (float) (dt / 2);
			}
			velocity(mid, raw2, vel);
			for (size_t i = 0; i < state.size(); i++)
			{
				state[i] = state[i] - vel[i] * (float) dt;
			}
		}

		for (size_t i = 0; i < state.size(); i++)
		{
			if (!std::isfinite(state[i]))
			{
				die("chunk %zu produced non-finite latents", ci);
			}
		}
		memcpy(&out[(size_t) a * D], state.data(), (size_t) Tc * D * sizeof(float));

		printf("         ode %.3f s, %d evals, %.3f s/eval, %.3f s/step\n",
		       ode_us / 1e6, n_eval, ode_us / 1e6 / n_eval, ode_us / 1e6 / p.steps);

		if (dump)
		{
			std::vector<float> rr(raws.begin(), raws.end());
			dump_npy(p.dump_dir + "/nar_tshift_f32.npy", { (int64_t) raws.size() }, rr.data());
		}

		ggml_gallocr_free(ga);
		g.free_all();
		kv.free_all();
	}

	printf("total:   %.3f s for %lld frames\n", (now_us() - t_all0) / 1e6, (long long) T);

	float lo = out[0];
	float hi = out[0];
	for (float x : out)
	{
		lo = std::min(lo, x);
		hi = std::max(hi, x);
	}
	printf("range:   [%.4f, %.4f]\n", (double) lo, (double) hi);

	err = npy::save(p.output.c_str(), { T, (int64_t) c.latent_dim }, out.data());
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
	printf("wrote:   %s [%lld, %d]\n", p.output.c_str(), (long long) T, c.latent_dim);

	ggml_free(model.ctx);
	ggml_backend_buffer_free(model.abuf);
	ggml_backend_buffer_free(model.nbuf);
	ggml_backend_free(backend);
	return 0;
}
