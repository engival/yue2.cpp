// yue2-vae — YuE2 Oobleck audio VAE decoder on ggml.
// See SPEC.md (sections 1, 3, 4) and src/STATUS.md.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "stage_vae.hpp"

#include "common/device.hpp"
#include "common/flac.hpp"
#include "common/gguf_kv.hpp"
#include "common/util.hpp"

#include "npy.hpp"
#include "wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// Everything below is private to this translation unit: the three stages each
// have their own Model / Builder / Graph, and `yue2` links all three.
namespace
{

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Debug probes (--probe-dir): keep named intermediates live so they can be read
// back and diffed against another backend. Off by default; costs memory (it also
// blocks op fusion, since a fused chain's inner nodes may not be graph outputs).
static bool        g_probe = false;
static std::string g_probe_dir;

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

struct Config
{
	int channels           = 64;
	std::vector<int> c_mults = {1, 2, 4, 8, 16, 32};
	std::vector<int> strides = {2, 2, 4, 4, 5, 6};
	int latent_dim         = 64;
	int out_channels       = 2;
	int sample_rate        = 48000;
	int downsampling_ratio = 1920;
	int core_frames        = 1024;
	int halo_frames        = 16;
};

struct Model
{
	Config cfg;
	ggml_context *        wctx   = nullptr;   // weight metadata
	ggml_backend_buffer_t wbuf   = nullptr;   // weight storage
	std::map<std::string, ggml_tensor *> tensors;

	ggml_tensor * get(const std::string & name) const
	{
		auto it = tensors.find(name);
		if (it == tensors.end())
		{
			die("missing tensor '%s' in the GGUF", name.c_str());
		}
		return it->second;
	}

	ggml_tensor * opt(const std::string & name) const
	{
		auto it = tensors.find(name);
		return it == tensors.end() ? nullptr : it->second;
	}
};

static bool ends_with(const std::string & s, const char * suf)
{
	const size_t n = strlen(suf);
	return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static void load_model(const char * path, ggml_backend_t backend, Model & model)
{
	gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
	gguf_context * gc = gguf_init_from_file(path, gp);
	if (gc == nullptr)
	{
		die("cannot open GGUF '%s'", path);
	}

	const int64_t arch_id = gguf_find_key(gc, "general.architecture");
	if (arch_id >= 0 && strcmp(gguf_get_val_str(gc, arch_id), "yue2-vae") != 0)
	{
		die("GGUF general.architecture is '%s', expected 'yue2-vae'", gguf_get_val_str(gc, arch_id));
	}

	Config & c = model.cfg;
	c.channels           = kv_i32(gc, "yue2vae.channels",           c.channels);
	c.c_mults            = kv_i32_array(gc, "yue2vae.c_mults",      c.c_mults);
	c.strides            = kv_i32_array(gc, "yue2vae.strides",      c.strides);
	c.latent_dim         = kv_i32(gc, "yue2vae.latent_dim",         c.latent_dim);
	c.out_channels       = kv_i32(gc, "yue2vae.out_channels",       c.out_channels);
	c.sample_rate        = kv_i32(gc, "yue2vae.sample_rate",        c.sample_rate);
	c.downsampling_ratio = kv_i32(gc, "yue2vae.downsampling_ratio", c.downsampling_ratio);
	c.core_frames        = kv_i32(gc, "yue2vae.decode_core_frames", c.core_frames);
	c.halo_frames        = kv_i32(gc, "yue2vae.decode_halo_frames", c.halo_frames);

	int ratio = 1;
	for (int s : c.strides)
	{
		ratio *= s;
	}
	if (ratio != c.downsampling_ratio)
	{
		die("strides product %d != downsampling_ratio %d", ratio, c.downsampling_ratio);
	}

	const int64_t n_tensors = gguf_get_n_tensors(gc);
	if (n_tensors == 0)
	{
		die("GGUF contains no tensors");
	}

	ggml_init_params ip = {
		/*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_tensors + 8),
		/*.mem_buffer =*/ nullptr,
		/*.no_alloc   =*/ true,
	};
	model.wctx = ggml_init(ip);

	// Everything lives as F32 on the backend: the graph feeds conv weights to
	// mul_mat as src1, which must be F32, and the Vulkan conv_transpose_1d
	// kernel is F32-only. F16 tensors in the file are widened at load.
	for (int64_t i = 0; i < n_tensors; i++)
	{
		const char * name = gguf_get_tensor_name(gc, i);
		const int64_t * ne = gguf_get_tensor_ne(gc, i);
		ggml_tensor * t = ggml_new_tensor_4d(model.wctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		ggml_set_name(t, name);
		model.tensors[name] = t;
	}

	model.wbuf = ggml_backend_alloc_ctx_tensors(model.wctx, backend);
	if (model.wbuf == nullptr)
	{
		die("failed to allocate weight buffer");
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
		const std::string name = gguf_get_tensor_name(gc, i);
		const ggml_type type   = gguf_get_tensor_type(gc, i);
		const size_t   nbytes  = gguf_get_tensor_size(gc, i);
		ggml_tensor * t = model.tensors[name];
		const int64_t nelem = ggml_nelements(t);

		raw.resize(nbytes);
		if (fseek(f, (long) (data_offset + gguf_get_tensor_offset(gc, i)), SEEK_SET) != 0 ||
		    fread(raw.data(), 1, nbytes, f) != nbytes)
		{
			die("truncated tensor data for '%s'", name.c_str());
		}

		float * src = nullptr;
		if (type == GGML_TYPE_F32)
		{
			src = (float *) raw.data();
		} else if (type == GGML_TYPE_F16) {
			conv.resize((size_t) nelem);
			ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), conv.data(), nelem);
			src = conv.data();
		} else {
			die("tensor '%s' has unsupported type %s", name.c_str(), ggml_type_name(type));
		}

		// SnakeBeta beta is stored already exponentiated; the graph multiplies
		// by its reciprocal, so fold 1/(beta + 1e-9) in here.
		if (ends_with(name, ".beta"))
		{
			if (src != conv.data())
			{
				conv.assign(src, src + nelem);
				src = conv.data();
			}
			for (int64_t k = 0; k < nelem; k++)
			{
				src[k] = 1.0f / (src[k] + 1e-9f);
			}
		}

		ggml_backend_tensor_set(t, src, 0, (size_t) nelem * sizeof(float));
	}
	fclose(f);
	gguf_free(gc);
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

struct Builder
{
	ggml_context * ctx;
	const Model *  model;
	ggml_type      im2col_type;
	std::vector<ggml_tensor *> * probes = nullptr;

	ggml_tensor * probe(ggml_tensor * t, const std::string & name)
	{
		// Skip the very large intermediates (im2col of the wide layers) so a
		// probe run still fits in VRAM.
		if (probes != nullptr && ggml_nelements(t) <= 16 * 1024 * 1024)
		{
			ggml_set_name(t, name.c_str());
			ggml_set_output(t);
			probes->push_back(t);
		}
		return t;
	}

	// x: [L, Cin, 1]; weight: [K, Cin, Cout]
	ggml_tensor * conv1d(ggml_tensor * x, const std::string & prefix, int pad, int dil)
	{
		ggml_tensor * w = model->get(prefix + ".weight");
		ggml_tensor * im2col = ggml_im2col(ctx, w, x, 1, 0, pad, 0, dil, 0, false, im2col_type);
		probe(im2col, prefix + ".im2col");
		ggml_tensor * r = ggml_mul_mat(ctx,
			ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
			ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
		probe(r, prefix + ".mm");
		r = ggml_reshape_3d(ctx, r, im2col->ne[1], w->ne[2], im2col->ne[2]);
		ggml_tensor * b = model->opt(prefix + ".bias");
		if (b != nullptr)
		{
			r = ggml_add(ctx, r, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
		}
		return r;
	}

	// torch ConvTranspose1d(stride=s, kernel=2s, padding=p): ggml has no
	// padding, so run it unpadded and crop p samples off each end.
	ggml_tensor * conv_transpose1d(ggml_tensor * x, const std::string & prefix, int stride, int pad)
	{
		ggml_tensor * w = model->get(prefix + ".weight");
		ggml_tensor * r = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
		if (pad > 0)
		{
			const int64_t len = r->ne[0] - 2 * pad;
			GGML_ASSERT(len > 0);
			r = ggml_cont(ctx, ggml_view_2d(ctx, r, len, r->ne[1], r->nb[1], (size_t) pad * r->nb[0]));
		}
		ggml_tensor * b = model->opt(prefix + ".bias");
		if (b != nullptr)
		{
			r = ggml_add(ctx, r, ggml_reshape_3d(ctx, b, 1, b->ne[0], 1));
		}
		return r;
	}

	// x + sin(x * alpha)^2 * inv_beta   (inv_beta folded at load)
	ggml_tensor * snake(ggml_tensor * x, const std::string & prefix)
	{
		ggml_tensor * a  = model->get(prefix + ".alpha");
		ggml_tensor * ib = model->get(prefix + ".beta");
		ggml_tensor * t  = probe(ggml_mul(ctx, x, ggml_reshape_3d(ctx, a, 1, a->ne[0], 1)), prefix + ".sinarg");
		ggml_tensor * s  = probe(ggml_sin(ctx, t), prefix + ".sin");
		s = ggml_sqr(ctx, s);
		s = ggml_mul(ctx, s, ggml_reshape_3d(ctx, ib, 1, ib->ne[0], 1));
		return probe(ggml_add(ctx, x, s), prefix + ".out");
	}

	ggml_tensor * residual_unit(ggml_tensor * x, const std::string & prefix, int dil)
	{
		ggml_tensor * h = snake(x, prefix + ".layers.0");
		h = conv1d(h, prefix + ".layers.1", 3 * dil, dil);
		h = snake(h, prefix + ".layers.2");
		h = conv1d(h, prefix + ".layers.3", 0, 1);
		return ggml_add(ctx, x, h);
	}

	ggml_tensor * decoder_block(ggml_tensor * x, const std::string & prefix, int stride)
	{
		const int pad = (stride + 1) / 2;   // ceil(stride / 2)
		x = snake(x, prefix + ".layers.0");
		x = conv_transpose1d(x, prefix + ".layers.1", stride, pad);
		x = residual_unit(x, prefix + ".layers.2", 1);
		x = residual_unit(x, prefix + ".layers.3", 3);
		x = residual_unit(x, prefix + ".layers.4", 9);
		return x;
	}
};

struct Graph
{
	ggml_context * ctx   = nullptr;
	ggml_cgraph *  gf    = nullptr;
	ggml_tensor *  input = nullptr;
	ggml_tensor *  out   = nullptr;
	std::vector<ggml_tensor *> probes;
	std::vector<uint8_t> buf;
};

static const size_t GRAPH_NODES = 4096;

static void build_graph(const Model & model, int64_t frames, ggml_type im2col_type, Graph & g)
{
	g.buf.resize(ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false));
	ggml_init_params ip = { g.buf.size(), g.buf.data(), /*.no_alloc =*/ true };
	g.ctx = ggml_init(ip);
	g.gf  = ggml_new_graph_custom(g.ctx, GRAPH_NODES, false);

	Builder b { g.ctx, &model, im2col_type };
	if (g_probe)
	{
		b.probes = &g.probes;
	}
	const Config & c = model.cfg;

	g.input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, frames, c.latent_dim, 1);
	ggml_set_name(g.input, "latent");
	ggml_set_input(g.input);

	ggml_tensor * x = b.conv1d(g.input, "decoder.layers.0", 3, 1);

	const int depth = (int) c.strides.size();   // 6 decoder blocks
	for (int i = 0; i < depth; i++)
	{
		const int stride = c.strides[depth - 1 - i];
		x = b.decoder_block(x, "decoder.layers." + std::to_string(i + 1), stride);
	}

	x = b.snake(x, "decoder.layers." + std::to_string(depth + 1));
	x = b.conv1d(x, "decoder.layers." + std::to_string(depth + 2), 3, 1);

	ggml_set_output(x);
	ggml_set_name(x, "audio");
	g.out = x;
	ggml_build_forward_expand(g.gf, x);
}

static int64_t output_length(const Config & c, int64_t frames)
{
	return (int64_t) c.downsampling_ratio * frames - 64;
}

// ---------------------------------------------------------------------------
// runner
// ---------------------------------------------------------------------------

// Read every probed tensor back and print max |value|; with a non-empty
// --probe-dir also write each one as <dir>/<name>.npy ([ne1, ne0]).
static void dump_probes(const Graph & g)
{
	if (g.probes.empty())
	{
		return;
	}
	std::vector<float> host;
	for (ggml_tensor * t : g.probes)
	{
		host.resize((size_t) ggml_nelements(t));
		ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));
		float mx = 0.0f;
		for (float v : host)
		{
			mx = std::max(mx, std::fabs(v));
		}
		printf("probe %-52s ne=[%6lld,%5lld] max|v|=%.6g\n",
		       ggml_get_name(t), (long long) t->ne[0], (long long) t->ne[1], mx);
		if (!g_probe_dir.empty())
		{
			const std::string path = g_probe_dir + "/" + ggml_get_name(t) + ".npy";
			const std::string err  = npy::save(path.c_str(), { t->ne[1], t->ne[0] }, host.data());
			if (!err.empty())
			{
				die("%s", err.c_str());
			}
		}
	}
}

struct Runner
{
	const Model *  model;
	ggml_backend_t backend;
	ggml_type      im2col_type;
	std::map<int64_t, ggml_gallocr_t> allocs;

	~Runner()
	{
		for (auto & kv : allocs)
		{
			ggml_gallocr_free(kv.second);
		}
	}

	// latent: [frames, latent_dim] in ggml layout (time fastest).
	// out:    [1920*frames - 64, out_channels], channel-major.
	void decode(const float * latent, int64_t frames, std::vector<float> & out, int64_t * us_compute)
	{
		Graph g;
		build_graph(*model, frames, im2col_type, g);

		auto it = allocs.find(frames);
		if (it == allocs.end())
		{
			ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
			if (!ggml_gallocr_reserve(ga, g.gf))
			{
				die("failed to reserve compute buffer for %lld frames", (long long) frames);
			}
			it = allocs.emplace(frames, ga).first;
		}
		if (!ggml_gallocr_alloc_graph(it->second, g.gf))
		{
			die("failed to allocate the compute graph");
		}

		ggml_backend_tensor_set(g.input, latent, 0, (size_t) frames * model->cfg.latent_dim * sizeof(float));

		const int64_t t0 = now_us();
		if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS)
		{
			die("graph compute failed");
		}
		const int64_t t1 = now_us();
		if (us_compute != nullptr)
		{
			*us_compute = t1 - t0;
		}

		dump_probes(g);

		const int64_t expect = output_length(model->cfg, frames);
		if (g.out->ne[0] != expect || g.out->ne[1] != model->cfg.out_channels)
		{
			die("unexpected output shape [%lld, %lld], expected [%lld, %d]",
			    (long long) g.out->ne[0], (long long) g.out->ne[1],
			    (long long) expect, model->cfg.out_channels);
		}

		out.resize((size_t) ggml_nelements(g.out));
		ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
		ggml_free(g.ctx);
	}
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char * argv0)
{
	fprintf(stderr,
		"usage: %s -m vae.gguf -i latent.npy -o out.wav|out.flac\n"
		"           [--device cpu|vulkan] [--gpu N] [--threads N]\n"
		"           [--core-frames 256] [--halo-frames 16] [--im2col f32|f16]\n"
		"           [--frames N] [--npy out.npy] [--full] [--vk-f16-matmul]\n"
		"           [--probe] [--probe-dir DIR]   # debug: dump intermediates\n", argv0);
}

} // namespace

VaeParams parse_vae_args(const char * argv0, int argc, char ** argv)
{
	VaeParams p;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "-m" || a == "--model")
		{
			p.model = need(argc, argv, i);
		} else if (a == "-i" || a == "--input") {
			p.input = need(argc, argv, i);
		} else if (a == "-o" || a == "--output") {
			p.output = need(argc, argv, i);
		} else if (a == "--npy") {
			p.npy_out = need(argc, argv, i);
		} else if (a == "--device") {
			p.device = need(argc, argv, i);
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--threads" || a == "-t") {
			p.threads = atoi(need(argc, argv, i));
		} else if (a == "--core-frames") {
			p.core_frames = atoi(need(argc, argv, i));
		} else if (a == "--halo-frames") {
			p.halo_frames = atoi(need(argc, argv, i));
		} else if (a == "--vk-f16-matmul") {
			p.vk_f16_matmul = true;
		} else if (a == "--probe") {
			g_probe = true;
		} else if (a == "--probe-dir") {
			g_probe     = true;
			g_probe_dir = need(argc, argv, i);
		} else if (a == "--frames") {
			p.frames = atoll(need(argc, argv, i));
		} else if (a == "--im2col") {
			const std::string v = need(argc, argv, i);
			if (v == "f32")
			{
				p.im2col_type = GGML_TYPE_F32;
			} else if (v == "f16") {
				p.im2col_type = GGML_TYPE_F16;
			} else {
				die("--im2col must be f32 or f16");
			}
		} else if (a == "--full") {
			p.full = true;
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument '%s'", argv[i]);
		}
	}

	if (p.model.empty() || p.input.empty())
	{
		usage(argv0);
		die("-m and -i are required");
	}
	if (p.output.empty() && p.npy_out.empty())
	{
		usage(argv0);
		die("at least one of -o / --npy is required");
	}
	if (p.core_frames < 1)
	{
		die("--core-frames must be >= 1");
	}
	if (p.halo_frames < 16)
	{
		die("--halo-frames must be >= 16 for this decoder");
	}
	return p;
}

int run_vae(const VaeParams & p)
{
	ggml_time_init();

	// backend ---------------------------------------------------------------
	// Every Conv1d here is im2col + mul_mat, and fp16 operand staging silently
	// costs ~1.7e-4 relative error per conv, which the decoder's ~40 SnakeBeta
	// non-linearities amplify into a ~65 dB noise floor with occasional ~85 ms
	// bursts. vulkan_want_exact_f32() selects the genuinely-F32 pipelines.
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
	int64_t t0 = now_us();
	load_model(p.model.c_str(), backend, model);
	printf("load:    %.3f s (%zu tensors, %.1f MiB)\n",
	       (now_us() - t0) / 1e6, model.tensors.size(),
	       ggml_backend_buffer_get_size(model.wbuf) / 1024.0 / 1024.0);

	// latent ----------------------------------------------------------------
	npy::Array in;
	const std::string err = npy::load(p.input.c_str(), in);
	if (!err.empty())
	{
		die("%s", err.c_str());
	}

	const int C = model.cfg.latent_dim;
	int64_t frames = 0;
	std::vector<float> latent;   // ggml layout: [T, C], time fastest
	if (in.shape.size() == 2 && in.shape[1] == C)
	{
		// [T, C] C-order -> transpose
		frames = in.shape[0];
		latent.resize((size_t) frames * C);
		for (int64_t t = 0; t < frames; t++)
		{
			for (int c = 0; c < C; c++)
			{
				latent[(size_t) c * frames + t] = in.data[(size_t) t * C + c];
			}
		}
		printf("latent:  %s shape [%lld, %d] (interpreted as [T, C])\n",
		       p.input.c_str(), (long long) frames, C);
	} else if ((in.shape.size() == 2 && in.shape[0] == C) ||
	           (in.shape.size() == 3 && in.shape[0] == 1 && in.shape[1] == C)) {
		frames = in.shape.back();
		latent.assign(in.data.begin(), in.data.end());
		printf("latent:  %s shape [%s%d, %lld] (interpreted as [C, T])\n",
		       p.input.c_str(), in.shape.size() == 3 ? "1, " : "", C, (long long) frames);
	} else {
		std::string s;
		for (size_t i = 0; i < in.shape.size(); i++)
		{
			s += (i ? ", " : "") + std::to_string(in.shape[i]);
		}
		die("latent shape [%s] is not [T, %d], [%d, T] or [1, %d, T]", s.c_str(), C, C, C);
	}

	if (p.frames > 0)
	{
		if (p.frames > frames)
		{
			die("--frames %lld exceeds the %lld frames in the latent",
			    (long long) p.frames, (long long) frames);
		}
		// latent is [C, T] with time fastest; cropping T means a per-channel copy
		std::vector<float> cropped((size_t) p.frames * C);
		for (int c = 0; c < C; c++)
		{
			memcpy(&cropped[(size_t) c * p.frames], &latent[(size_t) c * frames], (size_t) p.frames * sizeof(float));
		}
		latent.swap(cropped);
		frames = p.frames;
		printf("frames:  limited to %lld\n", (long long) frames);
	}
	if (frames < 1)
	{
		die("empty latent");
	}

	// decode ----------------------------------------------------------------
	Runner runner { &model, backend, p.im2col_type, {} };
	const int64_t total = output_length(model.cfg, frames);
	std::vector<float> audio((size_t) total * model.cfg.out_channels, 0.0f);

	const int64_t t_dec0 = now_us();
	if (p.full)
	{
		int64_t us = 0;
		std::vector<float> tile;
		runner.decode(latent.data(), frames, tile, &us);
		audio.swap(tile);
		printf("decode:  untiled, %lld frames, %.3f s\n", (long long) frames, us / 1e6);
	} else {
		const int64_t core  = p.core_frames;
		const int64_t halo  = p.halo_frames;
		const int64_t ratio = model.cfg.downsampling_ratio;
		const int64_t tiles = (frames + core - 1) / core;
		int64_t index = 0;
		std::vector<float> tile;
		std::vector<float> sub;
		for (int64_t start = 0; start < frames; start += core)
		{
			const int64_t end   = std::min(frames, start + core);
			const int64_t left  = std::max((int64_t) 0, start - halo);
			const int64_t right = std::min(frames, end + halo);
			const int64_t n     = right - left;

			sub.resize((size_t) n * C);
			for (int c = 0; c < C; c++)
			{
				memcpy(&sub[(size_t) c * n], &latent[(size_t) c * frames + left], (size_t) n * sizeof(float));
			}

			int64_t us = 0;
			runner.decode(sub.data(), n, tile, &us);

			const int64_t tile_len   = output_length(model.cfg, n);
			const int64_t out_start  = start * ratio;
			const int64_t out_end    = std::min(end * ratio, total);
			const int64_t crop_start = (start - left) * ratio;
			const int64_t len        = out_end - out_start;
			if (crop_start + len > tile_len)
			{
				die("tile did not cover its requested output core");
			}
			for (int c = 0; c < model.cfg.out_channels; c++)
			{
				memcpy(&audio[(size_t) c * total + out_start],
				       &tile[(size_t) c * tile_len + crop_start],
				       (size_t) len * sizeof(float));
			}
			index++;
			printf("tile %lld/%lld: frames [%lld, %lld) ctx %lld, %.3f s\n",
			       (long long) index, (long long) tiles,
			       (long long) start, (long long) end, (long long) n, us / 1e6);
		}
	}
	printf("total:   %.3f s for %lld samples/ch (%.2f s audio)\n",
	       (now_us() - t_dec0) / 1e6, (long long) total, (double) total / model.cfg.sample_rate);

	// output ----------------------------------------------------------------
	if (!p.npy_out.empty())
	{
		const std::string e = npy::save(p.npy_out.c_str(),
			{ (int64_t) model.cfg.out_channels, total }, audio.data());
		if (!e.empty())
		{
			die("%s", e.c_str());
		}
		printf("wrote:   %s [%d, %lld]\n", p.npy_out.c_str(), model.cfg.out_channels, (long long) total);
	}
	if (!p.output.empty())
	{
		// Encode to a sibling temp name and rename on success, so a mid-stream
		// encoder error cannot leave a truncated song that looks finished.
		// Character devices (-o /dev/null) are written in place: renaming over
		// one would replace the device node.
		const bool atomic = !std::filesystem::exists(p.output) ||
		                    std::filesystem::is_regular_file(p.output);
		const std::string path = atomic ? p.output + ".partial" : p.output;

		// .flac goes through libFLAC as 24-bit PCM (what soundfile wrote for the
		// reference pipeline); anything else stays float WAV.
		const std::string e = ends_with(p.output, ".flac")
			? flac::save_f32_24(path.c_str(), audio.data(),
				model.cfg.out_channels, total, model.cfg.sample_rate)
			: wav::save_f32(path.c_str(), audio.data(),
				model.cfg.out_channels, total, model.cfg.sample_rate);
		if (!e.empty())
		{
			if (atomic)
			{
				std::error_code rm;
				std::filesystem::remove(path, rm);
			}
			die("%s", e.c_str());
		}
		if (atomic)
		{
			std::error_code mv;
			std::filesystem::rename(path, p.output, mv);
			if (mv)
			{
				die("cannot rename %s to %s: %s", path.c_str(), p.output.c_str(),
				    mv.message().c_str());
			}
		}
		printf("wrote:   %s\n", p.output.c_str());
	}

	ggml_free(model.wctx);
	ggml_backend_buffer_free(model.wbuf);
	ggml_backend_free(backend);
	return 0;
}
