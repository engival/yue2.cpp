// `yue2 convert` — safetensors → GGUF for the AR, the NAR and the VAE, in C++.
// SPEC_CONVERT.md. Each converter ports its Python twin in convert/ (tensor
// maps, KV keys and order, which tensors stay F32, the asserts on config.json)
// and writes the same bytes: the acceptance is a byte compare against the
// Python-made GGUFs.
//
// Compiled with -ffp-contract=off (CMakeLists.txt): numpy never fuses a
// multiply into an add, so neither may the weight-norm fold or the exp port.

#include "convert.hpp"

#include "common/util.hpp"

#include "ggml.h"
#include "gguf.h"

extern "C" {
#include <hash/sha256/sha256.h>
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace
{

const char * const REPO_3B  = "m-a-p/YuE2-3B";
const char * const REPO_VAE = "m-a-p/YuE2-Vae";

// ---- finding the snapshots (convert/common.py resolve_snapshot) ------------

const char * env_value(const char * name)
{
	const char * v = getenv(name);
	return v != nullptr && *v != '\0' ? v : nullptr;
}

// huggingface_hub expands a leading ~ in its path variables; so do we.
fs::path expand_home(const char * text)
{
	if (text[0] == '~' && (text[1] == '/' || text[1] == '\\' || text[1] == '\0'))
	{
		const char * home = env_value("HOME") ? env_value("HOME") : env_value("USERPROFILE");
		if (home != nullptr)
		{
			return fs::path(home) / (text[1] == '\0' ? "" : text + 2);
		}
	}
	return text;
}

// huggingface_hub's constants.py order: HF_HUB_CACHE, the legacy
// HUGGINGFACE_HUB_CACHE, HF_HOME/hub, XDG_CACHE_HOME/huggingface/hub, then
// ~/.cache/huggingface/hub (%USERPROFILE% when HOME is unset).
fs::path hub_dir()
{
	for (const char * var : { "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE" })
	{
		if (env_value(var) != nullptr)
		{
			return expand_home(env_value(var));
		}
	}
	if (env_value("HF_HOME") != nullptr)
	{
		return expand_home(env_value("HF_HOME")) / "hub";
	}
	if (env_value("XDG_CACHE_HOME") != nullptr)
	{
		return expand_home(env_value("XDG_CACHE_HOME")) / "huggingface" / "hub";
	}
	const char * home = env_value("HOME") ? env_value("HOME") : env_value("USERPROFILE");
	if (home == nullptr)
	{
		die("cannot locate the Hugging Face cache: none of HF_HUB_CACHE, HF_HOME, XDG_CACHE_HOME, "
		    "HOME or USERPROFILE is set; pass --src-3b / --src-vae");
	}
	return fs::path(home) / ".cache" / "huggingface" / "hub";
}

std::string trim(const std::string & s)
{
	const size_t a = s.find_first_not_of(" \t\r\n");
	if (a == std::string::npos)
	{
		return "";
	}
	return s.substr(a, s.find_last_not_of(" \t\r\n") + 1 - a);
}

// The snapshot dir of `repo` holding every file in `required`, or "" with
// `err` saying what to do about it. `flag` names the override (--src-3b).
std::string find_snapshot(const std::string & given, const char * repo, const char * flag,
	const std::vector<const char *> & required, std::string & err)
{
	const std::string fetch = std::string("run `hf download ") + repo + "` (or `yue2 convert " + flag
		+ " DIR` for a copy elsewhere)";
	fs::path         snap;
	std::error_code  ec;
	if (!given.empty())
	{
		snap = given;
	} else {
		const fs::path hub = hub_dir();
		std::string    dir_name = std::string("models--") + repo;   // models--m-a-p--YuE2-3B
		dir_name.replace(dir_name.find('/'), 1, "--");
		const fs::path repo_dir = hub / dir_name;
		std::ifstream  ref(repo_dir / "refs" / "main");
		if (ref)
		{
			std::stringstream ss;
			ss << ref.rdbuf();
			snap = repo_dir / "snapshots" / trim(ss.str());
		} else {
			std::vector<std::string> found;
			for (const fs::directory_entry & d : fs::directory_iterator(repo_dir / "snapshots", ec))
			{
				if (d.is_directory(ec))
				{
					found.push_back(d.path().filename().string());
				}
			}
			if (found.empty())
			{
				err = std::string(repo) + " is not in the Hugging Face cache (" + hub.string() + "); " + fetch;
				return "";
			}
			if (found.size() > 1)
			{
				std::sort(found.begin(), found.end());
				err = std::string(repo) + " has several snapshots and no refs/main in " + repo_dir.string() + ":";
				for (const std::string & f : found)
				{
					err += " " + f;
				}
				err += "; pass " + std::string(flag) + " DIR";
				return "";
			}
			snap = repo_dir / "snapshots" / found[0];
		}
	}
	for (const char * f : required)
	{
		// exists() follows the snapshot's symlinks into blobs/.
		if (!fs::exists(snap / f, ec))
		{
			err = (snap / f).string() + " is missing; " + fetch;
			return "";
		}
	}
	return snap.string();
}

// ---- SHA-256 of a multi-GB file (llama.cpp's vendored sha256, streamed) ---

// Runs on a background thread: reports by throwing, wait_sha() dies for it.
std::string sha256_of(const fs::path & path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
	{
		throw std::runtime_error("cannot open " + path.string() + " to hash it");
	}
	sha256_t ctx;
	sha256_init(&ctx);
	std::vector<char> buf(1 << 20);
	while (in)
	{
		in.read(buf.data(), (std::streamsize) buf.size());
		sha256_update(&ctx, (const unsigned char *) buf.data(), (size_t) in.gcount());
	}
	if (in.bad())
	{
		throw std::runtime_error("read error while hashing " + path.string());
	}
	unsigned char digest[SHA256_DIGEST_SIZE];
	sha256_final(&ctx, digest);
	static const char hex[] = "0123456789abcdef";
	std::string out;
	for (unsigned char b : digest)
	{
		out += hex[b >> 4];
		out += hex[b & 15];
	}
	return out;
}

std::string wait_sha(const std::shared_future<std::string> & sha)
{
	try
	{
		return sha.get();
	}
	catch (const std::exception & e)
	{
		die("%s", e.what());
	}
}

// ---- safetensors reader ----------------------------------------------------

struct StEntry
{
	std::string          dtype;
	std::vector<int64_t> shape;
	uint64_t             begin = 0;
	int64_t              numel = 1;
	size_t               width = 0;
};

// Header in memory, data read on demand: a 7 GB checkpoint never is.
class SafeTensors
{
public:
	explicit SafeTensors(const fs::path & path) : path_(path.string()), in_(path, std::ios::binary)
	{
		if (!in_)
		{
			die("cannot open %s", path_.c_str());
		}
		const uint64_t size = (uint64_t) fs::file_size(path);
		uint64_t header_len = 0;
		in_.read((char *) &header_len, 8);
		// A real header is tens of KB; a garbage length must not become a huge allocation.
		if (!in_ || size < 8 || header_len == 0 || header_len > size - 8 || header_len > ((uint64_t) 256 << 20))
		{
			die("%s is not a safetensors file", path_.c_str());
		}
		std::string text((size_t) header_len, '\0');
		in_.read(&text[0], (std::streamsize) header_len);
		if (!in_)
		{
			die("%s is truncated inside its header", path_.c_str());
		}
		data_start_ = 8 + header_len;

		const nlohmann::json header = nlohmann::json::parse(text, nullptr, false);
		if (header.is_discarded() || !header.is_object())
		{
			die("the safetensors header of %s is not a JSON object", path_.c_str());
		}
		for (auto it = header.begin(); it != header.end(); ++it)
		{
			if (it.key() == "__metadata__")
			{
				continue;
			}
			StEntry  e;
			uint64_t end = 0;
			try
			{
				e.dtype = it.value().at("dtype").get<std::string>();
				e.shape = it.value().at("shape").get<std::vector<int64_t>>();
				e.begin = it.value().at("data_offsets").at(0).get<uint64_t>();
				end     = it.value().at("data_offsets").at(1).get<uint64_t>();
			}
			catch (const nlohmann::json::exception &)
			{
				die("entry '%s' of %s is not a safetensors tensor record", it.key().c_str(), path_.c_str());
			}
			e.width = e.dtype == "F32" ? 4 : (e.dtype == "F16" || e.dtype == "BF16") ? 2 : 0;
			for (int64_t d : e.shape)
			{
				if (d < 0)
				{
					die("'%s' in %s has a negative dimension", it.key().c_str(), path_.c_str());
				}
				e.numel *= d;
			}
			// An unknown dtype is fine until it is read — read_f32 names it then.
			if (end < e.begin || data_start_ + end > size
				|| (e.width != 0 && end - e.begin != (uint64_t) e.numel * e.width))
			{
				die("'%s' in %s: data_offsets do not fit its %s shape or the file",
				    it.key().c_str(), path_.c_str(), e.dtype.c_str());
			}
			entries_[it.key()] = e;
		}
	}

	bool has(const std::string & name) const
	{
		return entries_.count(name) != 0;
	}

	const StEntry & at(const std::string & name) const
	{
		const auto it = entries_.find(name);
		if (it == entries_.end())
		{
			die("%s has no tensor '%s'", path_.c_str(), name.c_str());
		}
		return it->second;
	}

	const std::map<std::string, StEntry> & entries() const
	{
		return entries_;
	}

	// Elements [first, first + n) of `name`, as F32 (exact for every dtype here).
	void read_f32(const std::string & name, int64_t first, int64_t n, std::vector<float> & out)
	{
		const StEntry & e = at(name);
		if (e.width == 0)
		{
			die("'%s' in %s has dtype %s; only F32, F16 and BF16 are supported",
			    name.c_str(), path_.c_str(), e.dtype.c_str());
		}
		out.resize((size_t) n);
		raw_.resize((size_t) n * e.width);
		in_.seekg((std::streamoff) (data_start_ + e.begin + (uint64_t) first * e.width));
		in_.read((char *) raw_.data(), (std::streamsize) raw_.size());
		if (!in_)
		{
			die("short read of '%s' from %s", name.c_str(), path_.c_str());
		}
		if (e.dtype == "BF16")
		{
			ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw_.data(), out.data(), n);
		} else if (e.dtype == "F16") {
			ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw_.data(), out.data(), n);
		} else {
			memcpy(out.data(), raw_.data(), raw_.size());
		}
	}

	void read_f32(const std::string & name, std::vector<float> & out)
	{
		read_f32(name, 0, at(name).numel, out);
	}

private:
	std::string                    path_;
	std::ifstream                  in_;
	uint64_t                       data_start_ = 0;
	std::map<std::string, StEntry> entries_;
	std::vector<uint8_t>           raw_;
};

// ---- GGUF out: tensor infos first, data streamed, metadata written last ----

struct Sink
{
	std::ofstream & f;
	uint64_t        bytes = 0;

	void write(const void * data, size_t n)
	{
		f.write((const char *) data, (std::streamsize) n);
		bytes += n;
	}
};

struct OutTensor
{
	std::string                 name;
	ggml_type                   type;
	std::vector<int64_t>        shape;   // row-major, as the safetensors header has it
	std::function<void(Sink &)> write;
};

struct Plan
{
	std::vector<OutTensor> tensors;
	// Every KV in file order; `sha` is the checkpoint's SHA-256.
	std::function<void(gguf_context *, const std::string & sha)> set_kv;
};

// The .tmp being written, removed by the atexit handler when die() exits.
fs::path g_tmp;

void remove_tmp_at_exit()
{
	std::error_code ec;
	if (!g_tmp.empty())
	{
		fs::remove(g_tmp, ec);
	}
}

// The source SHA-256 is a KV, and hashing 7 GB takes as long as converting
// it — so the data goes first, behind a placeholder of the metadata's exact
// size, and the real metadata is written over it once `final_sha` returns.
// Each writer has its own NAME.tmp.<random>: two `yue2 song` first runs that
// both convert never share a file, and the later rename replaces the earlier.
void write_gguf(const fs::path & out, const Plan & plan, const std::function<std::string()> & final_sha)
{
	ggml_init_params ip = {};
	ip.mem_size   = ggml_tensor_overhead() * plan.tensors.size();
	ip.no_alloc   = true;
	ggml_context * infos = ggml_init(ip);
	std::vector<ggml_tensor *> ts;
	for (const OutTensor & t : plan.tensors)
	{
		int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
		const int n_dims = (int) t.shape.size();
		if (n_dims < 1 || n_dims > GGML_MAX_DIMS)
		{
			die("%s: %d dimensions", t.name.c_str(), n_dims);
		}
		for (int d = 0; d < n_dims; d++)
		{
			ne[d] = t.shape[n_dims - 1 - d];
		}
		if (ne[0] % ggml_blck_size(t.type) != 0)
		{
			die("%s: row of %lld does not split into %s blocks", t.name.c_str(), (long long) ne[0],
			    ggml_type_name(t.type));
		}
		ggml_tensor * g = ggml_new_tensor(infos, t.type, n_dims, ne);
		ggml_set_name(g, t.name.c_str());
		ts.push_back(g);
	}
	const auto make = [&](const std::string & sha_hex)
	{
		gguf_context * gc = gguf_init_empty();
		plan.set_kv(gc, sha_hex);
		for (ggml_tensor * g : ts)
		{
			gguf_add_tensor(gc, g);
		}
		return gc;
	};

	gguf_context * draft     = make(std::string(64, '0'));
	const size_t   meta_size = gguf_get_meta_size(draft);
	const size_t   align     = gguf_get_alignment(draft);

	static const bool cleanup_registered = std::atexit(remove_tmp_at_exit) == 0;
	(void) cleanup_registered;
	char tag[32];
	snprintf(tag, sizeof(tag), ".tmp.%08x%08x", std::random_device{}(),
	         (unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
	fs::path tmp = out;
	tmp += tag;
	g_tmp = tmp;
	std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
	if (!f)
	{
		die("cannot write %s — pass `yue2 convert --out DIR` to write somewhere else", tmp.string().c_str());
	}
	std::vector<char> meta(meta_size, 0);
	f.write(meta.data(), (std::streamsize) meta.size());
	const char zeros[64] = {};
	GGML_ASSERT(align <= sizeof(zeros));
	for (size_t i = 0; i < plan.tensors.size(); i++)
	{
		Sink s{ f };
		plan.tensors[i].write(s);
		if (s.bytes != gguf_get_tensor_size(draft, (int64_t) i))
		{
			die("%s: wrote %llu bytes, its GGUF info says %zu", plan.tensors[i].name.c_str(),
			    (unsigned long long) s.bytes, gguf_get_tensor_size(draft, (int64_t) i));
		}
		f.write(zeros, (std::streamsize) (GGML_PAD(s.bytes, align) - s.bytes));
	}
	gguf_free(draft);

	gguf_context * final_meta = make(final_sha());
	if (gguf_get_meta_size(final_meta) != meta_size)
	{
		die("%s: metadata size changed with the source hash", out.string().c_str());
	}
	gguf_get_meta_data(final_meta, meta.data());
	gguf_free(final_meta);
	ggml_free(infos);
	f.seekp(0);
	f.write(meta.data(), (std::streamsize) meta.size());
	f.close();
	if (f.fail())
	{
		die("cannot write %s (disk full?)", tmp.string().c_str());
	}
	std::error_code ec;
	fs::rename(tmp, out, ec);   // replaces an existing file, POSIX and Windows alike
	if (ec)
	{
		die("cannot rename %s to %s: %s", tmp.string().c_str(), out.string().c_str(), ec.message().c_str());
	}
	g_tmp.clear();
}

// One safetensors tensor out as `type`, a slab of rows at a time. Q8_0 rounds
// through F16 first: yue2-ar-q8_0.gguf has always been llama-quantize of the
// F16 GGUF (F16 → F32 → quantize_q8_0), and this is that, without the file.
void write_rows(SafeTensors & st, const std::string & name, ggml_type type, Sink & out)
{
	const StEntry & e = st.at(name);
	if (e.numel == 0)
	{
		die("'%s' is empty (a zero dimension) — not the YuE2 checkpoint this converter knows", name.c_str());
	}
	const int64_t n_per_row = e.shape.size() > 1 ? e.numel / e.shape[0] : e.numel;
	const int64_t n_rows    = e.numel / n_per_row;
	const int64_t slab      = std::max<int64_t>(1, ((int64_t) 1 << 23) / n_per_row);
	std::vector<float>       f32;
	std::vector<ggml_fp16_t> f16;
	std::vector<ggml_bf16_t> bf16;
	std::vector<uint8_t>     q8;
	for (int64_t r0 = 0; r0 < n_rows; r0 += slab)
	{
		const int64_t rows = std::min(slab, n_rows - r0);
		const int64_t n    = rows * n_per_row;
		st.read_f32(name, r0 * n_per_row, n, f32);
		switch (type)
		{
			case GGML_TYPE_F32:
				out.write(f32.data(), (size_t) n * sizeof(float));
				break;
			case GGML_TYPE_F16:
				f16.resize((size_t) n);
				ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
				out.write(f16.data(), (size_t) n * sizeof(ggml_fp16_t));
				break;
			case GGML_TYPE_BF16:
				bf16.resize((size_t) n);
				ggml_fp32_to_bf16_row_ref(f32.data(), bf16.data(), n);
				out.write(bf16.data(), (size_t) n * sizeof(ggml_bf16_t));
				break;
			case GGML_TYPE_Q8_0:
				f16.resize((size_t) n);
				ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
				ggml_fp16_to_fp32_row(f16.data(), f32.data(), n);
				q8.resize(ggml_row_size(GGML_TYPE_Q8_0, n_per_row) * (size_t) rows);
				out.write(q8.data(), ggml_quantize_chunk(GGML_TYPE_Q8_0, f32.data(), q8.data(), 0, rows, n_per_row, nullptr));
				break;
			default:
				die("write_rows: no path to %s", ggml_type_name(type));
		}
	}
}

// ---- config.json -----------------------------------------------------------

nlohmann::json read_json(const fs::path & path)
{
	std::ifstream in(path);
	const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
	if (j.is_discarded() || !j.is_object())
	{
		die("%s is not a JSON object", path.string().c_str());
	}
	return j;
}

const nlohmann::json & cfg_at(const nlohmann::json & cfg, const char * key, const fs::path & path)
{
	if (!cfg.contains(key))
	{
		die("%s has no \"%s\" — not the YuE2 checkpoint this converter knows", path.string().c_str(), key);
	}
	return cfg.at(key);
}

// The Python converters assert these; a different checkpoint fails here.
void require_num(const nlohmann::json & cfg, const char * key, double want, const fs::path & path)
{
	const nlohmann::json & v = cfg_at(cfg, key, path);
	if (!v.is_number() || v.get<double>() != want)
	{
		die("%s: \"%s\" is %s, expected %g — not the YuE2 checkpoint this converter knows",
		    path.string().c_str(), key, v.dump().c_str(), want);
	}
}

int cfg_int(const nlohmann::json & cfg, const char * key, const fs::path & path)
{
	const nlohmann::json & v = cfg_at(cfg, key, path);
	if (!v.is_number_integer())
	{
		die("%s: \"%s\" is %s, expected an integer", path.string().c_str(), key, v.dump().c_str());
	}
	return v.get<int>();
}

void require_qwen3_shape(const nlohmann::json & cfg, const fs::path & path)
{
	require_num(cfg, "num_hidden_layers", 28, path);
	require_num(cfg, "hidden_size", 2048, path);
	require_num(cfg, "num_attention_heads", 16, path);
	require_num(cfg, "num_key_value_heads", 8, path);
	require_num(cfg, "head_dim", 128, path);
	require_num(cfg, "intermediate_size", 6144, path);
	require_num(cfg, "rms_norm_eps", 1e-6, path);
	require_num(cfg, "rope_theta", 1000000, path);
	require_num(cfg, "max_position_embeddings", 24576, path);
}

const int N_LAYERS = 28;

// ---- AR: qwen3 + the tiktoken vocabulary (convert_ar.py) -------------------

const int32_t EOD          = 151643;
const int32_t N_RANKS      = 151643;
const int32_t CODEC_OFFSET = 151853;
const int32_t CODEC_SIZE   = 32768;
const int32_t VOCAB_SIZE   = 184704;

// gguf.TokenType
const int32_t TOKEN_NORMAL       = 1;
const int32_t TOKEN_CONTROL      = 3;
const int32_t TOKEN_USER_DEFINED = 4;
const int32_t TOKEN_UNUSED       = 5;

// transformers' bytes_to_unicode(), each byte's code point as UTF-8.
std::array<std::string, 256> byte_encoder()
{
	std::array<bool, 256> printable = {};
	for (int b = '!'; b <= '~'; b++)
	{
		printable[b] = true;
	}
	for (int b = 0xA1; b <= 0xAC; b++)
	{
		printable[b] = true;
	}
	for (int b = 0xAE; b <= 0xFF; b++)
	{
		printable[b] = true;
	}
	std::array<std::string, 256> enc;
	int n = 0;
	for (int b = 0; b < 256; b++)
	{
		const int cp = printable[b] ? b : 256 + n++;
		if (cp < 0x80)
		{
			enc[b] = std::string(1, (char) cp);
		} else {
			enc[b] = { (char) (0xC0 | (cp >> 6)), (char) (0x80 | (cp & 0x3F)) };
		}
	}
	return enc;
}

std::string base64_decode(const std::string & s, bool & ok)
{
	std::string out;
	uint32_t    acc  = 0;
	int         bits = 0;
	ok = true;
	for (char c : s)
	{
		int v;
		if (c >= 'A' && c <= 'Z')
		{
			v = c - 'A';
		} else if (c >= 'a' && c <= 'z') {
			v = c - 'a' + 26;
		} else if (c >= '0' && c <= '9') {
			v = c - '0' + 52;
		} else if (c == '+') {
			v = 62;
		} else if (c == '/') {
			v = 63;
		} else if (c == '=') {
			break;
		} else {
			ok = false;
			return "";
		}
		acc = (acc << 6) | (uint32_t) v;
		bits += 6;
		if (bits >= 8)
		{
			bits -= 8;
			out += (char) ((acc >> bits) & 0xFF);
		}
	}
	return out;
}

// llama.cpp conversion/qwen.py QwenModel.bpe(), MIT (NOTICE.md): replay the
// merges of `token` that rank below max_rank.
std::vector<std::string> bpe(const std::unordered_map<std::string, int32_t> & ranks,
	const std::string & token, int32_t max_rank)
{
	std::vector<std::string> parts;
	for (char c : token)
	{
		parts.emplace_back(1, c);
	}
	while (true)
	{
		size_t  min_idx  = 0;
		int32_t min_rank = -1;
		for (size_t i = 0; i + 1 < parts.size(); i++)
		{
			const auto it = ranks.find(parts[i] + parts[i + 1]);
			if (it != ranks.end() && (min_rank < 0 || it->second < min_rank))
			{
				min_idx  = i;
				min_rank = it->second;
			}
		}
		if (min_rank < 0 || min_rank >= max_rank)
		{
			break;
		}
		parts[min_idx] += parts[min_idx + 1];
		parts.erase(parts.begin() + (std::ptrdiff_t) min_idx + 1);
	}
	return parts;
}

struct Vocab
{
	std::vector<std::string> tokens;
	std::vector<int32_t>     types;
	std::vector<std::string> merges;
};

// convert_ar.py build_vocab(): tiktoken ranks → gpt2-style tokens + merges,
// then the specials, <music>, </music>, the codec tokens and the pads.
Vocab build_vocab(const fs::path & tiktoken_path)
{
	std::ifstream in(tiktoken_path, std::ios::binary);
	if (!in)
	{
		die("cannot open %s", tiktoken_path.string().c_str());
	}
	std::vector<std::pair<std::string, int32_t>> in_order;   // file order is the merges' order
	std::unordered_map<std::string, int32_t>     ranks;
	std::string line;
	while (std::getline(in, line))
	{
		if (!line.empty() && line.back() == '\r')
		{
			line.pop_back();
		}
		std::istringstream fields(line);
		std::string        b64, rank_text, extra;
		if (!(fields >> b64))
		{
			continue;
		}
		bool ok = false;
		const std::string token = (fields >> rank_text) && !(fields >> extra) ? base64_decode(b64, ok) : "";
		char * endp = nullptr;
		const long rank = ok ? strtol(rank_text.c_str(), &endp, 10) : -1;
		if (!ok || token.empty() || *endp != '\0' || rank < 0 || rank >= N_RANKS)
		{
			die("%s: bad line \"%s\"", tiktoken_path.string().c_str(), line.c_str());
		}
		if (!ranks.emplace(token, (int32_t) rank).second)
		{
			die("%s: token on line \"%s\" appears twice", tiktoken_path.string().c_str(), line.c_str());
		}
		in_order.emplace_back(token, (int32_t) rank);
	}
	if ((int32_t) ranks.size() != N_RANKS)
	{
		die("Expected %d tiktoken ranks, got %zu", N_RANKS, ranks.size());
	}

	const std::array<std::string, 256> enc = byte_encoder();
	const auto to_string = [&](const std::string & bytes)
	{
		std::string s;
		for (char c : bytes)
		{
			s += enc[(unsigned char) c];
		}
		return s;
	};

	Vocab v;
	v.tokens.assign(VOCAB_SIZE, "");
	v.types.assign(VOCAB_SIZE, 0);
	for (const auto & [token, rank] : in_order)
	{
		if (v.types[rank] != 0)
		{
			die("%s: rank %d appears twice", tiktoken_path.string().c_str(), rank);
		}
		v.tokens[rank] = to_string(token);
		v.types[rank]  = TOKEN_NORMAL;
		if (token.size() == 1)
		{
			continue;
		}
		const std::vector<std::string> merged = bpe(ranks, token, rank);
		if (merged.size() != 2)
		{
			die("%s: rank %d does not split into one merge", tiktoken_path.string().c_str(), rank);
		}
		v.merges.push_back(to_string(merged[0]) + " " + to_string(merged[1]));
	}

	// YuE2TextTokenizer's 208 specials, extra_196/197 renamed to <abc>/</abc>.
	std::vector<std::string> specials = { "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>" };
	for (int i = 0; i < 200; i++)
	{
		specials.push_back("<extra_" + std::to_string(i) + ">");
	}
	specials[204] = "<abc>";
	specials[205] = "</abc>";
	specials.push_back("<music>");    // MUSIC_START = 151851
	specials.push_back("</music>");   // MUSIC_END   = 151852
	for (size_t i = 0; i < specials.size(); i++)
	{
		v.tokens[EOD + i] = specials[i];
		v.types[EOD + i]  = TOKEN_CONTROL;
	}
	for (int32_t c = 0; c < CODEC_SIZE; c++)
	{
		v.tokens[CODEC_OFFSET + c] = "<codec_" + std::to_string(c) + ">";
		v.types[CODEC_OFFSET + c]  = TOKEN_USER_DEFINED;
	}
	for (int32_t i = CODEC_OFFSET + CODEC_SIZE; i < VOCAB_SIZE; i++)
	{
		v.tokens[i] = "<pad_" + std::to_string(i - CODEC_OFFSET - CODEC_SIZE) + ">";
		v.types[i]  = TOKEN_UNUSED;
	}
	return v;
}

void set_arr_str(gguf_context * gc, const char * key, const std::vector<std::string> & strs)
{
	std::vector<const char *> ptrs;
	for (const std::string & s : strs)
	{
		ptrs.push_back(s.c_str());
	}
	gguf_set_arr_str(gc, key, ptrs.data(), ptrs.size());
}

// llama_model_loader::weight_name_comparer — the order llama-quantize writes.
bool quantize_order(const OutTensor & a, const OutTensor & b)
{
	int a_layer = -1;
	int b_layer = -1;
	sscanf(a.name.c_str(), "blk.%d.", &a_layer);
	sscanf(b.name.c_str(), "blk.%d.", &b_layer);
	if (a_layer != b_layer)
	{
		return a_layer < b_layer;
	}
	return a.name < b.name;
}

void convert_ar(const fs::path & snap, const fs::path & out, const std::string & type,
	std::shared_future<std::string> sha)
{
	const fs::path cfg_path = snap / "config.json";
	const nlohmann::json cfg = read_json(cfg_path);
	require_qwen3_shape(cfg, cfg_path);
	require_num(cfg, "vocab_size", VOCAB_SIZE, cfg_path);

	const auto  vocab = std::make_shared<Vocab>(build_vocab(snap / "qwen.tiktoken"));
	SafeTensors st(snap / "model.safetensors");

	std::vector<std::pair<std::string, std::string>> tmap = {
		{ "model.embed_tokens.weight", "token_embd.weight" },
		{ "model.norm.weight",         "output_norm.weight" },
		{ "lm_head.weight",            "output.weight" },
	};
	const char * const per_layer[][2] = {
		{ "input_layernorm.weight",          "attn_norm.weight" },
		{ "self_attn.q_proj.weight",         "attn_q.weight" },
		{ "self_attn.k_proj.weight",         "attn_k.weight" },
		{ "self_attn.v_proj.weight",         "attn_v.weight" },
		{ "self_attn.o_proj.weight",         "attn_output.weight" },
		{ "self_attn.q_norm.weight",         "attn_q_norm.weight" },
		{ "self_attn.k_norm.weight",         "attn_k_norm.weight" },
		{ "post_attention_layernorm.weight", "ffn_norm.weight" },
		{ "mlp.gate_proj.weight",            "ffn_gate.weight" },
		{ "mlp.up_proj.weight",              "ffn_up.weight" },
		{ "mlp.down_proj.weight",            "ffn_down.weight" },
	};
	for (int n = 0; n < N_LAYERS; n++)
	{
		for (const auto & m : per_layer)
		{
			tmap.emplace_back("model.layers." + std::to_string(n) + "." + m[0],
			                  "blk." + std::to_string(n) + "." + m[1]);
		}
	}

	const ggml_type qtype = type == "q8_0" ? GGML_TYPE_Q8_0 : type == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
	Plan plan;
	for (const auto & [src, dst] : tmap)
	{
		const StEntry & e      = st.at(src);
		const bool      as_f32 = e.shape.size() <= 1
			|| (dst.size() > 12 && dst.compare(dst.size() - 12, 12, "_norm.weight") == 0);
		const ggml_type t      = as_f32 ? GGML_TYPE_F32 : qtype;
		const std::string name = src;
		plan.tensors.push_back({ dst, t, e.shape, [&st, name, t](Sink & s) { write_rows(st, name, t, s); } });
	}
	if (qtype == GGML_TYPE_Q8_0)
	{
		std::stable_sort(plan.tensors.begin(), plan.tensors.end(), quantize_order);
	}

	// LlamaFileType: MOSTLY_F16 = 1, MOSTLY_Q8_0 = 7, MOSTLY_BF16 = 32. A
	// llama-quantize output has the source's KVs, then these two at the end.
	const uint32_t file_type = qtype == GGML_TYPE_Q8_0 ? 7 : qtype == GGML_TYPE_BF16 ? 32 : 1;
	plan.set_kv = [vocab, qtype, file_type](gguf_context * gc, const std::string & sha_hex)
	{
		gguf_set_val_str(gc, "general.architecture", "qwen3");
		gguf_set_val_str(gc, "general.name", "YuE2-3B AR");
		gguf_set_val_u32(gc, "qwen3.context_length", 24576);
		gguf_set_val_u32(gc, "qwen3.embedding_length", 2048);
		gguf_set_val_u32(gc, "qwen3.block_count", N_LAYERS);
		gguf_set_val_u32(gc, "qwen3.feed_forward_length", 6144);
		gguf_set_val_u32(gc, "qwen3.attention.head_count", 16);
		gguf_set_val_u32(gc, "qwen3.attention.head_count_kv", 8);
		gguf_set_val_u32(gc, "qwen3.attention.key_length", 128);
		gguf_set_val_u32(gc, "qwen3.attention.value_length", 128);
		gguf_set_val_f32(gc, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
		gguf_set_val_f32(gc, "qwen3.rope.freq_base", 1000000.0f);
		if (qtype != GGML_TYPE_Q8_0)
		{
			gguf_set_val_u32(gc, "general.file_type", file_type);
		}
		gguf_set_val_str(gc, "yue2.source_sha256", sha_hex.c_str());
		gguf_set_val_str(gc, "tokenizer.ggml.model", "gpt2");
		gguf_set_val_str(gc, "tokenizer.ggml.pre", "qwen2");
		set_arr_str(gc, "tokenizer.ggml.tokens", vocab->tokens);
		gguf_set_arr_data(gc, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, vocab->types.data(), vocab->types.size());
		set_arr_str(gc, "tokenizer.ggml.merges", vocab->merges);
		gguf_set_val_u32(gc, "tokenizer.ggml.bos_token_id", EOD);
		gguf_set_val_u32(gc, "tokenizer.ggml.eos_token_id", EOD);
		gguf_set_val_bool(gc, "tokenizer.ggml.add_bos_token", false);
		if (qtype == GGML_TYPE_Q8_0)
		{
			gguf_set_val_u32(gc, "general.quantization_version", GGML_QNT_VERSION);
			gguf_set_val_u32(gc, "general.file_type", file_type);
		}
	};
	write_gguf(out, plan, [&] { return wait_sha(sha); });
}

// ---- NAR (convert_nar.py) --------------------------------------------------

const int HIDDEN            = 2048;
const int MAX_LATENT_FRAMES = 24576;

// SPEC_NAR §2.2: the stored (BF16-rounded) latent_pos_embed.pe must be the
// sinusoid of AudioPositionEmbedding, or the layout reading is wrong.
void check_pe(SafeTensors & st)
{
	const StEntry & e = st.at("latent_pos_embed.pe");
	if (e.shape != std::vector<int64_t>{ MAX_LATENT_FRAMES, HIDDEN })
	{
		die("latent_pos_embed.pe is not [%d, %d]", MAX_LATENT_FRAMES, HIDDEN);
	}
	std::vector<double> div_term(HIDDEN / 2);
	for (int k = 0; k < HIDDEN / 2; k++)
	{
		div_term[k] = std::exp((double) (2 * k) * (-std::log(10000.0) / HIDDEN));
	}
	const int64_t      slab = 1024;
	std::vector<float> pe;
	float              max_delta = 0.0f;
	for (int64_t r0 = 0; r0 < MAX_LATENT_FRAMES; r0 += slab)
	{
		st.read_f32("latent_pos_embed.pe", r0 * HIDDEN, slab * HIDDEN, pe);
		for (int64_t r = 0; r < slab; r++)
		{
			for (int k = 0; k < HIDDEN / 2; k++)
			{
				const double a = (double) (r0 + r) * div_term[k];
				max_delta = std::max(max_delta, std::fabs(pe[r * HIDDEN + 2 * k] - (float) std::sin(a)));
				max_delta = std::max(max_delta, std::fabs(pe[r * HIDDEN + 2 * k + 1] - (float) std::cos(a)));
			}
		}
	}
	if (!(max_delta < 8e-3f))
	{
		die("latent_pos_embed.pe does not match the sinusoid formula: max|delta|=%.3e", max_delta);
	}
	printf("convert: latent_pos_embed.pe vs formula: max|delta|=%.3e (must be < 8e-3)\n", max_delta);
}

void convert_nar(const fs::path & snap, const fs::path & out, const std::string & type,
	std::shared_future<std::string> sha, const fs::path & ar_gguf)
{
	const fs::path cfg_path = snap / "config.json";
	const nlohmann::json cfg = read_json(cfg_path);
	require_qwen3_shape(cfg, cfg_path);
	require_num(cfg, "latent_dim", 64, cfg_path);
	require_num(cfg, "max_latent_frames", MAX_LATENT_FRAMES, cfg_path);
	require_num(cfg, "timestep_shift", 1.0, cfg_path);

	// convert_nar.py's read_ar_sha256 cross-check: `yue2 nar` refuses a pair
	// whose yue2.source_sha256 differ, so catch it here rather than at load.
	// The AR's value is read now; the source hash is compared once it is in,
	// after the data, so the hash and the write still overlap.
	std::string     ar_sha;
	std::error_code ec;
	if (fs::exists(ar_gguf, ec))
	{
		gguf_init_params gp = { true, nullptr };
		gguf_context *   gc = gguf_init_from_file(ar_gguf.string().c_str(), gp);
		const int64_t    id = gc ? gguf_find_key(gc, "yue2.source_sha256") : -1;
		if (id >= 0 && gguf_get_kv_type(gc, id) == GGUF_TYPE_STRING)
		{
			ar_sha = gguf_get_val_str(gc, id);
		}
		if (gc)
		{
			gguf_free(gc);
		}
	}

	SafeTensors st(snap / "model.safetensors");
	check_pe(st);

	enum Kind { NORM, MAT, PE };
	std::vector<std::tuple<std::string, std::string, Kind>> tmap;
	const std::tuple<const char *, const char *, Kind> per_layer[] = {
		{ "nar_input_layernorm.weight",    "nar_attn_norm.weight",   NORM },
		{ "nar_self_attn.q_proj.weight",   "nar_attn_q.weight",      MAT },
		{ "nar_self_attn.k_proj.weight",   "nar_attn_k.weight",      MAT },
		{ "nar_self_attn.v_proj.weight",   "nar_attn_v.weight",      MAT },
		{ "nar_self_attn.o_proj.weight",   "nar_attn_output.weight", MAT },
		{ "nar_self_attn.q_norm.weight",   "nar_attn_q_norm.weight", NORM },
		{ "nar_self_attn.k_norm.weight",   "nar_attn_k_norm.weight", NORM },
		{ "nar_pre_mlp_layernorm.weight",  "nar_ffn_norm.weight",    NORM },
		{ "nar_mlp.gate_proj.weight",      "nar_ffn_gate.weight",    MAT },
		{ "nar_mlp.up_proj.weight",        "nar_ffn_up.weight",      MAT },
		{ "nar_mlp.down_proj.weight",      "nar_ffn_down.weight",    MAT },
	};
	for (int n = 0; n < N_LAYERS; n++)
	{
		for (const auto & [src, dst, kind] : per_layer)
		{
			tmap.emplace_back("model.layers." + std::to_string(n) + "." + src,
			                  "blk." + std::to_string(n) + "." + dst, kind);
		}
	}
	// The NAR-only auxiliaries stay F32, all but the positional table.
	tmap.emplace_back("vae2llm.weight",             "nar.vae2llm.weight",       NORM);
	tmap.emplace_back("vae2llm.bias",               "nar.vae2llm.bias",         NORM);
	tmap.emplace_back("llm2vae.weight",             "nar.llm2vae.weight",       NORM);
	tmap.emplace_back("llm2vae.bias",               "nar.llm2vae.bias",         NORM);
	tmap.emplace_back("time_embedder.mlp.0.weight", "nar.time_embd.0.weight",   NORM);
	tmap.emplace_back("time_embedder.mlp.0.bias",   "nar.time_embd.0.bias",     NORM);
	tmap.emplace_back("time_embedder.mlp.2.weight", "nar.time_embd.1.weight",   NORM);
	tmap.emplace_back("time_embedder.mlp.2.bias",   "nar.time_embd.1.bias",     NORM);
	tmap.emplace_back("latent_pos_embed.pe",        "nar.latent_pos_embd.weight", PE);

	const ggml_type qtype = type == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
	Plan plan;
	for (const auto & [src, dst, kind] : tmap)
	{
		// PE is F16 whatever --nar-type says: exact for a BF16-rounded value.
		const ggml_type t = kind == NORM ? GGML_TYPE_F32 : kind == PE ? GGML_TYPE_F16 : qtype;
		const std::string name = src;
		plan.tensors.push_back({ dst, t, st.at(src).shape, [&st, name, t](Sink & s) { write_rows(st, name, t, s); } });
	}
	plan.set_kv = [](gguf_context * gc, const std::string & sha_hex)
	{
		gguf_set_val_str(gc, "general.architecture", "yue2-nar");
		gguf_set_val_str(gc, "general.name", "YuE2-3B NAR");
		gguf_set_val_u32(gc, "yue2nar.block_count", N_LAYERS);
		gguf_set_val_u32(gc, "yue2nar.embedding_length", HIDDEN);
		gguf_set_val_u32(gc, "yue2nar.feed_forward_length", 6144);
		gguf_set_val_u32(gc, "yue2nar.attention.head_count", 16);
		gguf_set_val_u32(gc, "yue2nar.attention.head_count_kv", 8);
		gguf_set_val_u32(gc, "yue2nar.attention.key_length", 128);
		gguf_set_val_u32(gc, "yue2nar.attention.value_length", 128);
		gguf_set_val_f32(gc, "yue2nar.attention.layer_norm_rms_epsilon", 1e-6f);
		gguf_set_val_f32(gc, "yue2nar.rope.freq_base", 1000000.0f);
		gguf_set_val_u32(gc, "yue2nar.context_length", 24576);
		gguf_set_val_u32(gc, "yue2nar.latent_dim", 64);
		gguf_set_val_u32(gc, "yue2nar.max_latent_frames", MAX_LATENT_FRAMES);
		gguf_set_val_f32(gc, "yue2nar.timestep_shift", 1.0f);
		gguf_set_val_u32(gc, "yue2nar.time_embd_frequency_size", 256);
		gguf_set_val_u32(gc, "yue2nar.ode_steps", 32);
		gguf_set_val_str(gc, "yue2nar.ode_method", "midpoint");
		gguf_set_val_str(gc, "yue2.source_sha256", sha_hex.c_str());
	};
	const std::string source = (snap / "model.safetensors").string();
	write_gguf(out, plan, [&]
	{
		const std::string sha_hex = wait_sha(sha);
		if (!ar_sha.empty() && ar_sha != sha_hex)
		{
			die("source_sha256 mismatch: %s has %s, %s is %s — AR and NAR GGUFs must come from the same "
			    "model.safetensors; remake the AR with `yue2 convert --force --only ar`",
			    ar_gguf.string().c_str(), ar_sha.c_str(), source.c_str(), sha_hex.c_str());
		}
		return sha_hex;
	});
}

// ---- VAE (convert_vae.py) --------------------------------------------------

// numpy's pairwise summation (loops_utils.h.src pairwise_sum), which is what
// np.linalg.norm's add.reduce runs over a contiguous row.
double pairwise_sum(const double * a, int64_t n)
{
	if (n < 8)
	{
		double res = 0.0;
		for (int64_t i = 0; i < n; i++)
		{
			res += a[i];
		}
		return res;
	}
	if (n <= 128)
	{
		double  r[8];
		int64_t i;
		for (int j = 0; j < 8; j++)
		{
			r[j] = a[j];
		}
		for (i = 8; i < n - (n % 8); i += 8)
		{
			for (int j = 0; j < 8; j++)
			{
				r[j] += a[i + j];
			}
		}
		double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
		for (; i < n; i++)
		{
			res += a[i];
		}
		return res;
	}
	int64_t n2 = n / 2;
	n2 -= n2 % 8;
	return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// weight_norm(dim=0) folded: w = g * v / ||v||, the norm over every dim but 0,
// in double and narrowed once — fold_weight_norm() in convert_vae.py.
std::vector<float> fold_weight_norm(const std::vector<float> & g, const std::vector<float> & v)
{
	const size_t        rows = g.size();
	const size_t        m    = v.size() / rows;
	std::vector<float>  w(v.size());
	std::vector<double> sq(m);
	for (size_t c = 0; c < rows; c++)
	{
		const float * vc = v.data() + c * m;
		for (size_t i = 0; i < m; i++)
		{
			sq[i] = (double) vc[i] * (double) vc[i];
		}
		const double norm = std::sqrt(pairwise_sum(sq.data(), (int64_t) m));
		for (size_t i = 0; i < m; i++)
		{
			w[c * m + i] = (float) ((double) g[c] * (double) vc[i] / norm);
		}
	}
	return w;
}

// numpy's float32 exp on AVX2/AVX512F (loops_exponent_log simd_exp_FLOAT):
// Cody-Waite reduction and a 5/2 rational minimax, not libm's expf — the
// Python VAE converter's exp(alpha) is this, and differs from a correctly
// rounded expf in ~40 % of the checkpoint's values.
float numpy_expf(float x)
{
	if (std::isnan(x))
	{
		return x;
	}
	if (x >= 88.72283935546875f)
	{
		return INFINITY;
	}
	if (x <= -103.97208404541015625f)
	{
		return 0.0f;
	}
	const float magic = 0x1.800000p+23f;
	float q = x * 1.442695040888963407359924681001892137f;
	q = (q + magic) - magic;
	float r = std::fma(q, -6.93145752e-1f, x);
	r = std::fma(q, -1.42860677e-6f, r);
	r = std::fma(q, 0.0f, r);
	float num = std::fma(5.082762527590693718096e-04f, r, 6.757896990527504603057e-03f);
	num = std::fma(num, r, 5.114512081637298353406e-02f);
	num = std::fma(num, r, 2.473615434895520810817e-01f);
	num = std::fma(num, r, 7.257664613233124478488e-01f);
	num = std::fma(num, r, 9.999999999980870924916e-01f);
	float den = std::fma(2.159509375685829852307e-02f, r, -2.742335390411667452936e-01f);
	den = std::fma(den, r, 1.0f);
	return std::ldexp(num / den, (int) q);
}

struct VaeModule
{
	enum Kind { CONV, CONV_T, SNAKE } kind;
	std::string prefix;   // under "decoder."
	bool        bias;
};

// OobleckDecoder's modules in named_modules() order, from its constructor:
// conv, then per stride a DecoderBlock (snake, transposed conv, three
// ResidualUnits of snake/conv7/snake/conv1), then snake and a bias-free conv.
std::vector<VaeModule> decoder_modules(size_t n_blocks)
{
	std::vector<VaeModule> mods = { { VaeModule::CONV, "layers.0", true } };
	for (size_t b = 1; b <= n_blocks; b++)
	{
		const std::string blk = "layers." + std::to_string(b) + ".layers.";
		mods.push_back({ VaeModule::SNAKE, blk + "0", false });
		mods.push_back({ VaeModule::CONV_T, blk + "1", true });
		for (int r = 2; r <= 4; r++)
		{
			const std::string ru = blk + std::to_string(r) + ".layers.";
			mods.push_back({ VaeModule::SNAKE, ru + "0", false });
			mods.push_back({ VaeModule::CONV, ru + "1", true });
			mods.push_back({ VaeModule::SNAKE, ru + "2", false });
			mods.push_back({ VaeModule::CONV, ru + "3", true });
		}
	}
	mods.push_back({ VaeModule::SNAKE, "layers." + std::to_string(n_blocks + 1), false });
	mods.push_back({ VaeModule::CONV, "layers." + std::to_string(n_blocks + 2), false });
	return mods;
}

void convert_vae(const fs::path & snap, const fs::path & out, const std::string & type,
	std::shared_future<std::string> sha)
{
	const fs::path cfg_path = snap / "config.json";
	const nlohmann::json cfg = read_json(cfg_path);
	const nlohmann::json & dcfg = cfg_at(cfg, "decoder_config", cfg_path);
	std::vector<int> c_mults, strides;
	bool             released = false;
	try
	{
		c_mults = cfg_at(dcfg, "c_mults", cfg_path).get<std::vector<int>>();
		strides = cfg_at(dcfg, "strides", cfg_path).get<std::vector<int>>();
		// OobleckDecoder raises on the filter/antialias/nearest options, and the
		// C++ decoder has no final Tanh (OobleckDecoder's default is to have one).
		released = dcfg.value("use_snake", false) && dcfg.value("snake_type", std::string("vanilla")) == "vanilla"
			&& !dcfg.value("use_filter", false) && !dcfg.value("antialias_activation", false)
			&& !dcfg.value("use_nearest_upsample", false) && !dcfg.value("final_tanh", true);
	}
	catch (const nlohmann::json::exception &)
	{
		die("%s: decoder_config has values of the wrong JSON type", cfg_path.string().c_str());
	}
	if (c_mults.empty() || c_mults.size() != strides.size())
	{
		die("%s: decoder_config c_mults and strides differ in length", cfg_path.string().c_str());
	}
	if (!released)
	{
		die("%s: decoder_config is not the released decoder (vanilla snake, no filter, no final tanh)",
		    cfg_path.string().c_str());
	}
	const int channels     = cfg_int(dcfg, "channels", cfg_path);
	const int latent_dim   = cfg_int(dcfg, "latent_dim", cfg_path);
	const int out_channels = cfg_int(dcfg, "out_channels", cfg_path);
	const int sample_rate  = cfg_int(cfg, "sample_rate", cfg_path);
	const int down_ratio   = cfg_int(cfg, "downsampling_ratio", cfg_path);
	const int core_frames  = cfg_int(cfg, "decode_core_frames", cfg_path);
	const int halo_frames  = cfg_int(cfg, "decode_halo_frames", cfg_path);

	SafeTensors st(snap / "model.safetensors");
	const std::vector<VaeModule> mods = decoder_modules(c_mults.size());

	// load_state_dict(strict=True): the decoder.* keys are exactly these.
	std::set<std::string> expected;
	for (const VaeModule & m : mods)
	{
		const std::string p = "decoder." + m.prefix;
		if (m.kind == VaeModule::SNAKE)
		{
			expected.insert({ p + ".alpha", p + ".beta" });
		} else {
			expected.insert({ p + ".weight_g", p + ".weight_v" });
			if (m.bias)
			{
				expected.insert(p + ".bias");
			}
		}
	}
	for (const auto & [name, e] : st.entries())
	{
		if (name.compare(0, 8, "decoder.") == 0 && expected.erase(name) == 0)
		{
			die("%s: unexpected decoder tensor %s", (snap / "model.safetensors").string().c_str(), name.c_str());
		}
	}
	if (!expected.empty())
	{
		die("%s: missing decoder tensor %s", (snap / "model.safetensors").string().c_str(), expected.begin()->c_str());
	}

	Plan plan;
	for (const VaeModule & m : mods)
	{
		const std::string p = "decoder." + m.prefix;
		if (m.kind == VaeModule::SNAKE)
		{
			for (const char * which : { ".alpha", ".beta" })
			{
				const std::string name = p + which;
				plan.tensors.push_back({ name, GGML_TYPE_F32, st.at(name).shape, [&st, name](Sink & s)
				{
					std::vector<float> x;
					st.read_f32(name, x);
					for (float & y : x)
					{
						y = numpy_expf(y);
					}
					s.write(x.data(), x.size() * sizeof(float));
				} });
			}
			continue;
		}
		const StEntry & v = st.at(p + ".weight_v");
		if (v.shape.size() != 3 || v.numel == 0 || st.at(p + ".weight_g").numel != v.shape[0])
		{
			die("%s: weight_g/weight_v are not a [C,1,1] / [C,X,K] pair", p.c_str());
		}
		// --vae-type f16 narrows the Conv1d weights only; ConvTranspose1d stays F32.
		const ggml_type t = type == "f16" && m.kind == VaeModule::CONV ? GGML_TYPE_F16 : GGML_TYPE_F32;
		plan.tensors.push_back({ p + ".weight", t, v.shape, [&st, p, t](Sink & s)
		{
			std::vector<float> g, vv;
			st.read_f32(p + ".weight_g", g);
			st.read_f32(p + ".weight_v", vv);
			const std::vector<float> w = fold_weight_norm(g, vv);
			if (t == GGML_TYPE_F16)
			{
				std::vector<ggml_fp16_t> h(w.size());
				ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t) w.size());
				s.write(h.data(), h.size() * sizeof(ggml_fp16_t));
			} else {
				s.write(w.data(), w.size() * sizeof(float));
			}
		} });
		if (m.bias)
		{
			const std::string name = p + ".bias";
			plan.tensors.push_back({ name, GGML_TYPE_F32, st.at(name).shape,
				[&st, name](Sink & s) { write_rows(st, name, GGML_TYPE_F32, s); } });
		}
	}

	plan.set_kv = [=](gguf_context * gc, const std::string & sha_hex)
	{
		gguf_set_val_str(gc, "general.architecture", "yue2-vae");
		gguf_set_val_str(gc, "general.name", "YuE2-Vae decoder");
		gguf_set_val_i32(gc, "yue2vae.channels", channels);
		gguf_set_arr_data(gc, "yue2vae.c_mults", GGUF_TYPE_INT32, c_mults.data(), c_mults.size());
		gguf_set_arr_data(gc, "yue2vae.strides", GGUF_TYPE_INT32, strides.data(), strides.size());
		gguf_set_val_i32(gc, "yue2vae.latent_dim", latent_dim);
		gguf_set_val_i32(gc, "yue2vae.out_channels", out_channels);
		gguf_set_val_i32(gc, "yue2vae.sample_rate", sample_rate);
		gguf_set_val_i32(gc, "yue2vae.downsampling_ratio", down_ratio);
		gguf_set_val_i32(gc, "yue2vae.decode_core_frames", core_frames);
		gguf_set_val_i32(gc, "yue2vae.decode_halo_frames", halo_frames);
		gguf_set_val_bool(gc, "yue2vae.snake_folded", true);
		gguf_set_val_str(gc, "yue2vae.source_sha256", sha_hex.c_str());
	};
	write_gguf(out, plan, [&] { return wait_sha(sha); });
}

// ---- driver ----------------------------------------------------------------

std::shared_future<std::string> hash_in_background(const fs::path & path)
{
	return std::async(std::launch::async, [path] { return sha256_of(path); }).share();
}

// `announce` is the auto-convert path: one line naming the sources first.
int convert_impl(const ConvertParams & p, bool announce)
{
	const double   t0  = now_seconds();
	const fs::path out = p.out;
	std::error_code ec;
	fs::create_directories(out, ec);

	const fs::path ar_out  = out / ("yue2-ar-" + p.ar_type + ".gguf");
	const fs::path nar_out = out / ("yue2-nar-" + p.nar_type + ".gguf");
	const fs::path vae_out = out / ("yue2-vae-" + p.vae_type + ".gguf");
	const auto todo = [&](bool wanted, const fs::path & path)
	{
		if (wanted && !p.force && fs::exists(path, ec))
		{
			printf("convert: %s exists, skipped (--force to redo)\n", path.string().c_str());
			return false;
		}
		return wanted;
	};
	const bool do_ar  = todo(p.ar, ar_out);
	const bool do_nar = todo(p.nar, nar_out);
	const bool do_vae = todo(p.vae, vae_out);

	std::string err;
	std::string snap_3b, snap_vae;
	if (do_ar || do_nar)
	{
		snap_3b = find_snapshot(p.src_3b, REPO_3B, "--src-3b", { "config.json", "model.safetensors", "qwen.tiktoken" }, err);
		if (snap_3b.empty())
		{
			die("%s", err.c_str());
		}
	}
	if (do_vae)
	{
		snap_vae = find_snapshot(p.src_vae, REPO_VAE, "--src-vae", { "config.json", "model.safetensors" }, err);
		if (snap_vae.empty())
		{
			die("%s", err.c_str());
		}
	}
	if (announce)
	{
		// Output sizes of the default types: 2.3 + 2.9 + 0.27 GB.
		const double gb = (do_ar ? 2.3 : 0.0) + (do_nar ? 2.9 : 0.0) + (do_vae ? 0.3 : 0.0);
		const std::string src = snap_3b.empty() ? snap_vae : snap_vae.empty() ? snap_3b : snap_3b + " + " + snap_vae;
		printf("yue2: converting YuE2 weights from %s (one-time, ~%.1f GB)...\n", src.c_str(), gb);
	}

	std::shared_future<std::string> sha_3b;
	if (do_ar || do_nar)
	{
		sha_3b = hash_in_background(fs::path(snap_3b) / "model.safetensors");
	}
	if (do_ar)
	{
		const double t = now_seconds();
		convert_ar(snap_3b, ar_out, p.ar_type, sha_3b);
		printf("convert: wrote %s in %.1f s\n", ar_out.string().c_str(), now_seconds() - t);
	}
	if (do_nar)
	{
		const double t = now_seconds();
		convert_nar(snap_3b, nar_out, p.nar_type, sha_3b, ar_out);
		printf("convert: wrote %s in %.1f s\n", nar_out.string().c_str(), now_seconds() - t);
	}
	if (do_vae)
	{
		const double t = now_seconds();
		convert_vae(snap_vae, vae_out, p.vae_type, hash_in_background(fs::path(snap_vae) / "model.safetensors"));
		printf("convert: wrote %s in %.1f s\n", vae_out.string().c_str(), now_seconds() - t);
	}
	if (do_ar || do_nar || do_vae)
	{
		printf("convert: done in %.1f s\n", now_seconds() - t0);
	}
	return 0;
}

void usage(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s [--src-3b DIR] [--src-vae DIR] [--out DIR]\n"
	        "        [--ar-type q8_0|f16|bf16] [--nar-type f16|f32] [--vae-type f32|f16]\n"
	        "        [--only ar|nar|vae] [--force]\n"
	        "\n"
	        "m-a-p/YuE2-3B and m-a-p/YuE2-Vae (Hugging Face cache, or --src-*) to\n"
	        "yue2-ar-TYPE.gguf, yue2-nar-TYPE.gguf and yue2-vae-TYPE.gguf in --out\n"
	        "(default: next to this binary, where `yue2 song` looks). Existing files are\n"
	        "skipped unless --force.\n",
	        argv0);
}

} // namespace

ConvertParams parse_convert_args(const char * argv0, int argc, char ** argv)
{
	ConvertParams p;
	const auto pick = [](const char * flag, const char * value, std::initializer_list<const char *> allowed)
	{
		for (const char * a : allowed)
		{
			if (strcmp(a, value) == 0)
			{
				return std::string(value);
			}
		}
		die("%s: unknown type '%s'", flag, value);
	};
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--src-3b")
		{
			p.src_3b = need(argc, argv, i);
		} else if (a == "--src-vae") {
			p.src_vae = need(argc, argv, i);
		} else if (a == "--out" || a == "-o") {
			p.out = need(argc, argv, i);
		} else if (a == "--ar-type") {
			p.ar_type = pick("--ar-type", need(argc, argv, i), { "q8_0", "f16", "bf16" });
		} else if (a == "--nar-type") {
			p.nar_type = pick("--nar-type", need(argc, argv, i), { "f16", "f32" });
		} else if (a == "--vae-type") {
			p.vae_type = pick("--vae-type", need(argc, argv, i), { "f32", "f16" });
		} else if (a == "--only") {
			const std::string only = pick("--only", need(argc, argv, i), { "ar", "nar", "vae" });
			p.ar  = only == "ar";
			p.nar = only == "nar";
			p.vae = only == "vae";
		} else if (a == "--force") {
			p.force = true;
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument '%s'", argv[i]);
		}
	}
	return p;
}

int run_convert(const ConvertParams & p)
{
	return convert_impl(p, false);
}

void convert_missing_defaults(const std::string & out_dir, bool ar, bool nar, bool vae)
{
	ConvertParams p;
	p.out = out_dir;
	p.ar  = ar;
	p.nar = nar;
	p.vae = vae;
	convert_impl(p, true);
}
