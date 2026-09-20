// `--nar-lora FILE[:S]` — a LoRA folded into the NAR weights at load time.
// SPEC_LORA.md; the plain (unfused) safetensors layout only. The merge runs on
// one core between reading a GGUF tensor and uploading it, so there is no graph
// change and no per-step cost.
#pragma once

#include "ggml.h"

#include <nlohmann/json.hpp>

#include "common/util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace lora
{

// One --nar-lora argument, after FILE[:STRENGTH] has been split.
struct Spec
{
	std::string file;
	float       strength = 1.0f;
};

// SPEC_LORA §2: split on the last ':' and only when what follows parses fully as
// a float, so a path that happens to contain a colon still works.
inline Spec parse_arg(const std::string & text)
{
	Spec s;
	s.file = text;
	const size_t colon = text.rfind(':');
	if (colon != std::string::npos && colon + 1 < text.size())
	{
		const char * tail = text.c_str() + colon + 1;
		char *       endp = nullptr;
		const float  v    = strtof(tail, &endp);
		if (endp != tail && *endp == '\0')
		{
			s.file     = text.substr(0, colon);
			s.strength = v;
		}
	}
	if (!std::isfinite(s.strength))
	{
		die("--nar-lora strength must be finite (got \"%s\")", text.c_str());
	}
	if (s.file.empty())
	{
		die("--nar-lora needs a file name (got \"%s\")", text.c_str());
	}
	return s;
}

// ---------------------------------------------------------------------------
// the adapter key -> GGUF tensor name map (mirrors convert/convert_nar.py)
// ---------------------------------------------------------------------------

struct NamePair
{
	const char * lora;
	const char * gguf;
};

static const NamePair LAYER_MAP[] = {
	{ "nar_self_attn.q_proj", "nar_attn_q"      },
	{ "nar_self_attn.k_proj", "nar_attn_k"      },
	{ "nar_self_attn.v_proj", "nar_attn_v"      },
	{ "nar_self_attn.o_proj", "nar_attn_output" },
	{ "nar_mlp.gate_proj",    "nar_ffn_gate"    },
	{ "nar_mlp.up_proj",      "nar_ffn_up"      },
	{ "nar_mlp.down_proj",    "nar_ffn_down"    },
};

static const NamePair FULL_MAP[] = {
	{ "vae2llm.weight", "nar.vae2llm.weight" },
	{ "vae2llm.bias",   "nar.vae2llm.bias"   },
	{ "llm2vae.weight", "nar.llm2vae.weight" },
	{ "llm2vae.bias",   "nar.llm2vae.bias"   },
};

// `layers.{i}.<proj>.lora_{A,B}` or one of the four full replacements. `part` is
// 'A', 'B' or 'W' (replacement) on success.
inline bool map_key(const std::string & key, std::string & gguf, char & part)
{
	for (const NamePair & p : FULL_MAP)
	{
		if (key == p.lora)
		{
			gguf = p.gguf;
			part = 'W';
			return true;
		}
	}
	if (key.compare(0, 7, "layers.") != 0)
	{
		return false;
	}
	const size_t dot = key.find('.', 7);
	if (dot == std::string::npos || dot == 7)
	{
		return false;
	}
	const std::string index = key.substr(7, dot - 7);
	if (index.find_first_not_of("0123456789") != std::string::npos)
	{
		return false;
	}
	const std::string rest = key.substr(dot + 1);
	if (rest.size() < 8)
	{
		return false;
	}
	const std::string tail = rest.substr(rest.size() - 7);
	if (tail != ".lora_A" && tail != ".lora_B")
	{
		return false;
	}
	part = tail[6];
	const std::string proj = rest.substr(0, rest.size() - 7);
	for (const NamePair & p : LAYER_MAP)
	{
		if (proj == p.lora)
		{
			gguf = "blk." + index + "." + p.gguf + ".weight";
			return true;
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

struct Entry
{
	std::string          dtype;
	std::vector<int64_t> shape;
	uint64_t             begin = 0;
	uint64_t             end   = 0;
};

struct File
{
	std::string  path;
	float        strength   = 1.0f;
	FILE *       fp         = nullptr;
	uint64_t     data_start = 0;
	int          rank_min   = 0;
	int          rank_max   = 0;
	int          n_patched  = 0;
	int          n_replaced = 0;
	double       seconds    = 0.0;
	std::map<std::string, Entry> entries;
};

// One merge step against one GGUF tensor. A LoRA pair carries both factors; a
// replacement carries the new weight alone. `replacement` is not `b == nullptr`:
// an adapter that ships a lora_A without its lora_B has to be told apart from a
// replacement, and that is the whole point of the flag.
struct Op
{
	size_t        adapter     = 0;
	bool          replacement = false;
	const Entry * a           = nullptr;   // lora_A [r, in], or the new weight
	const Entry * b           = nullptr;   // lora_B [out, r]
};

class Merge
{
public:
	Merge()                         = default;
	Merge(const Merge &)            = delete;   // it owns open FILE handles
	Merge & operator=(const Merge &) = delete;

	~Merge()
	{
		for (File & f : files)
		{
			if (f.fp != nullptr)
			{
				fclose(f.fp);
			}
		}
	}

	bool empty() const
	{
		return plan.empty();
	}

	bool targets(const std::string & name) const
	{
		return plan.count(name) != 0;
	}

	// Opens every adapter, maps its keys onto GGUF tensor names and checks
	// everything that can be checked without the model (SPEC_LORA §3.4).
	void load(const std::vector<Spec> & specs)
	{
		files.resize(specs.size());
		for (size_t i = 0; i < specs.size(); i++)
		{
			read_header(files[i], specs[i]);
			build_plan(i);
		}
	}

	// The model half of §3.4: every key must land on a tensor that exists, is
	// F16 or F32, and has the shape the factors imply.
	void bind(const std::map<std::string, ggml_tensor *> & tensors)
	{
		for (const auto & kv : plan)
		{
			auto it = tensors.find(kv.first);
			if (it == tensors.end())
			{
				die("--nar-lora: '%s' names GGUF tensor '%s', which this model does not have",
				    files[kv.second[0].adapter].path.c_str(), kv.first.c_str());
			}
			const ggml_tensor * t = it->second;
			if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32)
			{
				die("--nar-lora: GGUF tensor '%s' is %s; only F16 and F32 targets can be merged",
				    kv.first.c_str(), ggml_type_name(t->type));
			}
			for (const Op & op : kv.second)
			{
				check_shape(kv.first, op, t->ne[0], t->ne[1]);
			}
		}
	}

	// `w` holds the GGUF tensor widened to F32, `in` x `out` row-major (element
	// (o, i) at o*in + i, which is both torch's [out, in] and ggml's ne=[in,out]).
	void apply(const std::string & name, float * w, int64_t in, int64_t out)
	{
		const std::vector<Op> & ops = plan.at(name);
		// A replacement is a delta against the bytes the GGUF held, so the ones
		// that follow it in the stack must still see the original.
		std::vector<float> base;
		for (const Op & op : ops)
		{
			if (op.replacement)
			{
				base.assign(w, w + (size_t) (in * out));
				break;
			}
		}
		for (const Op & op : ops)
		{
			File &       f  = files[op.adapter];
			const double t0 = now_seconds();
			if (op.replacement)
			{
				read_f32(f, *op.a, buf_a);
				for (size_t e = 0; e < buf_a.size(); e++)
				{
					w[e] += f.strength * (buf_a[e] - base[e]);
				}
			} else {
				read_f32(f, *op.a, buf_a);
				read_f32(f, *op.b, buf_b);
				const int64_t r = op.a->shape[0];
				for (int64_t o = 0; o < out; o++)
				{
					float * row = w + (size_t) o * (size_t) in;
					for (int64_t k = 0; k < r; k++)
					{
						const float  s = f.strength * buf_b[(size_t) (o * r + k)];
						const float * av = buf_a.data() + (size_t) k * (size_t) in;
						for (int64_t i = 0; i < in; i++)
						{
							row[i] += s * av[i];
						}
					}
				}
			}
			f.seconds += now_seconds() - t0;
		}
	}

	// One line per adapter, once the load loop is done.
	void report() const
	{
		for (const File & f : files)
		{
			char rank[32];
			if (f.rank_min == f.rank_max)
			{
				snprintf(rank, sizeof(rank), "rank %d", f.rank_max);
			} else {
				snprintf(rank, sizeof(rank), "rank %d-%d", f.rank_min, f.rank_max);
			}
			printf("lora:    %s (%s, strength %g): %d patched, %d replaced, merge %.3f s\n",
			       f.path.c_str(), f.rank_max == 0 ? "no lora pairs" : rank,
			       (double) f.strength, f.n_patched, f.n_replaced, f.seconds);
		}
	}

private:
	std::vector<File>                     files;
	std::map<std::string, std::vector<Op>> plan;
	std::vector<float>                    buf_a;
	std::vector<float>                    buf_b;

	static size_t dtype_size(const std::string & dtype)
	{
		if (dtype == "F32")
		{
			return 4;
		}
		if (dtype == "F16" || dtype == "BF16")
		{
			return 2;
		}
		return 0;
	}

	static void read_header(File & f, const Spec & spec)
	{
		f.path     = spec.file;
		f.strength = spec.strength;
		f.fp       = fopen(f.path.c_str(), "rb");
		if (f.fp == nullptr)
		{
			die("--nar-lora: cannot open '%s'", f.path.c_str());
		}
		if (fseek(f.fp, 0, SEEK_END) != 0)
		{
			die("--nar-lora: cannot seek in '%s'", f.path.c_str());
		}
		const uint64_t size = (uint64_t) ftell(f.fp);
		rewind(f.fp);

		uint64_t header_len = 0;
		if (size < 8 || fread(&header_len, 1, 8, f.fp) != 8)
		{
			die("--nar-lora: '%s' is not a safetensors file (too short for the header length)",
			    f.path.c_str());
		}
		if (header_len == 0 || header_len > size - 8)
		{
			die("--nar-lora: '%s' is not a safetensors file (header length %llu does not fit in %llu bytes)",
			    f.path.c_str(), (unsigned long long) header_len, (unsigned long long) size);
		}
		std::string text((size_t) header_len, '\0');
		if (fread(&text[0], 1, (size_t) header_len, f.fp) != (size_t) header_len)
		{
			die("--nar-lora: '%s' is truncated inside its header", f.path.c_str());
		}
		f.data_start = 8 + header_len;

		const nlohmann::json header = nlohmann::json::parse(text, nullptr, false);
		if (header.is_discarded() || !header.is_object())
		{
			die("--nar-lora: the safetensors header of '%s' is not a JSON object", f.path.c_str());
		}
		for (auto it = header.begin(); it != header.end(); ++it)
		{
			if (it.key() == "__metadata__")
			{
				continue;
			}
			const nlohmann::json & v = it.value();
			if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets"))
			{
				die("--nar-lora: entry '%s' of '%s' is not a safetensors tensor record",
				    it.key().c_str(), f.path.c_str());
			}
			Entry e;
			// The right keys with the wrong JSON types throw rather than return.
			try
			{
				e.dtype = v["dtype"].get<std::string>();
				e.shape = v["shape"].get<std::vector<int64_t>>();
				e.begin = v["data_offsets"].at(0).get<uint64_t>();
				e.end   = v["data_offsets"].at(1).get<uint64_t>();
			} catch (const nlohmann::json::exception &) {
				die("--nar-lora: entry '%s' of '%s' is not a safetensors tensor record",
				    it.key().c_str(), f.path.c_str());
			}
			const size_t width = dtype_size(e.dtype);
			if (width == 0)
			{
				die("--nar-lora: '%s' in '%s' has dtype %s; only F32, F16 and BF16 are supported",
				    it.key().c_str(), f.path.c_str(), e.dtype.c_str());
			}
			int64_t n = 1;
			for (int64_t d : e.shape)
			{
				if (d < 1)
				{
					die("--nar-lora: '%s' in '%s' has a non-positive dimension",
					    it.key().c_str(), f.path.c_str());
				}
				n *= d;
			}
			if (e.end < e.begin || e.end - e.begin != (uint64_t) n * width)
			{
				die("--nar-lora: '%s' in '%s' has data_offsets that do not match its shape",
				    it.key().c_str(), f.path.c_str());
			}
			if (f.data_start + e.end > size)
			{
				die("--nar-lora: '%s' is shorter than its header claims ('%s' ends at %llu of %llu bytes)",
				    f.path.c_str(), it.key().c_str(),
				    (unsigned long long) (f.data_start + e.end), (unsigned long long) size);
			}
			f.entries[it.key()] = e;
		}
		if (f.entries.empty())
		{
			die("--nar-lora: '%s' contains no tensors", f.path.c_str());
		}
	}

	// Turns one adapter's keys into merge ops, and rejects any key that has
	// nowhere to go or has lost its partner.
	void build_plan(size_t index)
	{
		File & f = files[index];
		// GGUF name -> the op being assembled for this adapter.
		std::map<std::string, Op> pairs;
		for (const auto & kv : f.entries)
		{
			std::string gguf;
			char        part = 0;
			if (!map_key(kv.first, gguf, part))
			{
				if (kv.first.compare(0, 16, "diffusion_model.") == 0)
				{
					die("--nar-lora: '%s' is the fused ComfyUI layout (key '%s'); "
					    "use the plain adapter file instead", f.path.c_str(), kv.first.c_str());
				}
				die("--nar-lora: key '%s' in '%s' maps to no NAR tensor",
				    kv.first.c_str(), f.path.c_str());
			}
			Op & op = pairs[gguf];
			op.adapter = index;
			if (part == 'W')
			{
				op.a           = &kv.second;
				op.replacement = true;
			} else if (part == 'A') {
				op.a = &kv.second;
			} else {
				op.b = &kv.second;
			}
		}
		for (auto & kv : pairs)
		{
			Op & op = kv.second;
			if (op.replacement)
			{
				f.n_replaced++;
			} else {
				if (op.a == nullptr)
				{
					die("--nar-lora: '%s' has a lora_B for '%s' but no lora_A",
					    f.path.c_str(), kv.first.c_str());
				}
				if (op.b == nullptr)
				{
					die("--nar-lora: '%s' has a lora_A for '%s' but no lora_B",
					    f.path.c_str(), kv.first.c_str());
				}
				if (op.a->shape.size() != 2 || op.b->shape.size() != 2)
				{
					die("--nar-lora: the factors for '%s' in '%s' must both be 2-D",
					    kv.first.c_str(), f.path.c_str());
				}
				const int r = (int) op.a->shape[0];
				if (op.b->shape[1] != op.a->shape[0])
				{
					die("--nar-lora: rank mismatch for '%s' in '%s' (lora_A %lld, lora_B %lld)",
					    kv.first.c_str(), f.path.c_str(),
					    (long long) op.a->shape[0], (long long) op.b->shape[1]);
				}
				f.rank_min = f.rank_min == 0 ? r : std::min(f.rank_min, r);
				f.rank_max = std::max(f.rank_max, r);
				f.n_patched++;
			}
			plan[kv.first].push_back(op);
		}
	}

	// `ne0` x `ne1` is the GGUF tensor's shape, i.e. [in, out].
	void check_shape(const std::string & name, const Op & op, int64_t ne0, int64_t ne1) const
	{
		const File & f = files[op.adapter];
		if (op.replacement)
		{
			const bool ok = op.a->shape.size() == 1
				? op.a->shape[0] == ne0 && ne1 == 1
				: op.a->shape.size() == 2 && op.a->shape[1] == ne0 && op.a->shape[0] == ne1;
			if (!ok)
			{
				die("--nar-lora: the replacement for '%s' in '%s' has shape %s, not the base's [%lld, %lld]",
				    name.c_str(), f.path.c_str(), shape_str(op.a->shape).c_str(),
				    (long long) ne1, (long long) ne0);
			}
			return;
		}
		if (op.a->shape[1] != ne0 || op.b->shape[0] != ne1)
		{
			die("--nar-lora: '%s' in '%s' is a %lld x %lld adapter, but the base is %lld x %lld",
			    name.c_str(), f.path.c_str(),
			    (long long) op.b->shape[0], (long long) op.a->shape[1],
			    (long long) ne1, (long long) ne0);
		}
	}

	static std::string shape_str(const std::vector<int64_t> & shape)
	{
		std::string s = "[";
		for (size_t i = 0; i < shape.size(); i++)
		{
			s += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
		}
		return s + "]";
	}

	// Reads one adapter tensor and widens it to F32. The file stays on disk:
	// only the tensor being merged is ever in memory.
	static void read_f32(const File & f, const Entry & e, std::vector<float> & out)
	{
		int64_t n = 1;
		for (int64_t d : e.shape)
		{
			n *= d;
		}
		out.resize((size_t) n);
		const size_t         nbytes = (size_t) (e.end - e.begin);
		std::vector<uint8_t> raw(nbytes);
		if (fseek(f.fp, (long) (f.data_start + e.begin), SEEK_SET) != 0 ||
		    fread(raw.data(), 1, nbytes, f.fp) != nbytes)
		{
			die("--nar-lora: cannot read tensor data from '%s'", f.path.c_str());
		}
		if (e.dtype == "F32")
		{
			memcpy(out.data(), raw.data(), nbytes);
		} else if (e.dtype == "F16") {
			ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), n);
		} else {
			ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.data(), n);
		}
		for (int64_t i = 0; i < n; i++)
		{
			if (!std::isfinite(out[(size_t) i]))
			{
				die("--nar-lora: '%s' holds a non-finite value", f.path.c_str());
			}
		}
	}
};

} // namespace lora
