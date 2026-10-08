// yue2 song — AR (libllama) -> NAR (raw ggml) -> VAE, one request in, one FLAC
// out. See SPEC_SINGLE.md; the stages themselves are unchanged, this is the
// plumbing that used to live in yue2_gen.py.

#include "stage_song.hpp"

#include "convert.hpp"
#include "stage_ar.hpp"
#include "stage_nar.hpp"
#include "stage_vae.hpp"

#include "common/device.hpp"
#include "common/fileio.hpp"
#include "common/flac.hpp"
#include "common/noise.hpp"
#include "common/util.hpp"

#include "npy.hpp"

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

const int SAMPLE_RATE  = 48000;
const int DOWNSAMPLING = 1920;
const int LATENT_DIM   = 64;

// The artifacts directory a run without --artifacts makes for itself. die()
// goes through exit(), which runs atexit handlers, so a failure anywhere in the
// run still takes the directory with it instead of leaving several GB of debris
// in /tmp for a batch driver to accumulate.
std::vector<std::string> g_temp_artifacts;

void remove_temp_artifacts()
{
	std::error_code ec;
	for (size_t i = 0; i < g_temp_artifacts.size(); i++)
	{
		std::filesystem::remove_all(g_temp_artifacts[i], ec);
	}
	g_temp_artifacts.clear();
}

// The per-job directory a run without `artifacts` makes for itself. Anchored in
// a literal prefix so the remove_all above can never be handed anything else.
std::string temp_artifacts_dir(size_t job)
{
#ifdef _WIN32
	const unsigned long pid = GetCurrentProcessId();
#else
	const long pid = (long) getpid();
#endif
	const std::filesystem::path path = std::filesystem::temp_directory_path() /
		("yue2song." + std::to_string(pid) + "." + std::to_string(job));
	const std::string dir = path.string();
	if (path.filename().string().rfind("yue2song.", 0) != 0 || path.parent_path().empty())
	{
		die("refusing to use '%s' as the temporary artifacts directory", dir.c_str());
	}
	if (g_temp_artifacts.empty())
	{
		atexit(remove_temp_artifacts);
	}
	g_temp_artifacts.push_back(dir);
	return dir;
}

// The FLAC's Vorbis comments for one request: its "tags" block as given, the
// words that are sung (a "lyrics_file" read as the AR stage reads it), the
// encoder. Nothing else of the request reaches the file unless it is named in
// "tags". A request the AR stage will refuse anyway (unreadable, not JSON, a
// missing lyrics file) yields no tags and no error here.
std::string request_tags(const std::string & request_path, const std::string & base_dir,
	flac::Tags & tags)
{
	std::string body;
	if (!read_file(request_path, body).empty())
	{
		return "";
	}
	json root = json::parse(body, nullptr, false);
	if (!root.is_object() || !resolve_request_files(root, base_dir).empty())
	{
		return "";
	}
	if (root.contains("tags"))
	{
		if (!root["tags"].is_object())
		{
			return "\"tags\" must be an object of NAME: \"text\"";
		}
		for (const auto & item : root["tags"].items())
		{
			if (!item.value().is_string())
			{
				return "\"tags\": \"" + item.key() + "\" must be a string";
			}
			const std::string err = flac::check_tag(item.key(), item.value().get<std::string>());
			if (!err.empty())
			{
				return "\"tags\": " + err;
			}
			tags.emplace_back(item.key(), item.value().get<std::string>());
		}
	}
	if (root.contains("lyrics") && root["lyrics"].is_string() &&
	    std::none_of(tags.begin(), tags.end(), [](const auto & t) { return t.first == "LYRICS"; }))
	{
		tags.emplace_back("LYRICS", root["lyrics"].get<std::string>());
	}
	tags.emplace_back("ENCODER", "yue2.cpp");
	return "";
}

// --opus-bitrate, the same range for `song` and for `batch`.
int parse_opus_bitrate(const char * value)
{
	const int kbps = atoi(value);
	if (kbps < opus::BITRATE_MIN || kbps > opus::BITRATE_MAX)
	{
		die("--opus-bitrate must be between %d and %d kbit/s", opus::BITRATE_MIN, opus::BITRATE_MAX);
	}
	return kbps;
}

void usage(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s --request R.json --out X.flac|X.opus [--artifacts DIR] [--seed N]\n"
	        "        [--ar AR.gguf] [--nar NAR.gguf] [--vae VAE.gguf]\n"
	        "        [--gpu N] [--cpu] [--nar-f32] [--noise FILE] [--steps 16]\n"
	        "        [--nar-lora LORA.safetensors[:S]]\n"
	        "        [--guidance-trace] [--no-tags] [--opus-bitrate 160]\n"
	        "        [--draft HEAD.gguf [--draft-k 2] [--draft-lambda 1] [--draft-q shaped|raw]\n"
	        "         [--draft-window N] [--draft-trace]]\n", argv0);
}

void usage_batch(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s --jobs jobs.json [--parallel N] [--summary FILE]\n"
	        "        [--ar AR.gguf] [--nar NAR.gguf] [--vae VAE.gguf] [--seed N]\n"
	        "        [--gpu N] [--cpu] [--nar-f32] [--steps 16]\n"
	        "        [--nar-lora LORA.safetensors[:S]]\n"
	        "        [--threads N] [--greedy] [--max-abc N] [--max-semantic N]\n"
	        "        [--continue-on-error] [--guidance-trace] [--no-tags]\n"
	        "        [--opus-bitrate 160]\n"
	        "        [--draft HEAD.gguf [--draft-k 2] [--draft-lambda 1] [--draft-q shaped|raw]\n"
	        "         [--draft-window N] [--draft-trace]]   --draft needs --parallel 1\n"
	        "\n"
	        "jobs.json is an array of { \"request\", \"out\", \"artifacts\", \"seed\", \"noise\" };\n"
	        "request and out are required and paths are relative to the working directory.\n",
	        argv0);
}

// This executable's own path — where resolve_gguf() looks for the default GGUFs.
std::filesystem::path exe_path()
{
#ifdef _WIN32
	wchar_t buf[32768];
	const DWORD n = GetModuleFileNameW(nullptr, buf, sizeof(buf) / sizeof(buf[0]));
	if (n == 0 || n >= sizeof(buf) / sizeof(buf[0]))
	{
		die("cannot get this executable's path (GetModuleFileNameW)");
	}
	return std::filesystem::path(std::wstring(buf, n));
#else
	std::error_code ec;
	std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
	if (ec)
	{
		die("cannot read /proc/self/exe: %s", ec.message().c_str());
	}
	return p;
#endif
}

// A default GGUF's path, or "" when it is in none of the places resolve_gguf()
// looks: next to the executable, one level up, the working directory.
std::string find_default_gguf(const char * name)
{
	const std::filesystem::path here = exe_path().parent_path();
	const std::string tried[] = { (here / name).string(), (here / ".." / name).string(), name };
	for (const std::string & candidate : tried)
	{
		if (std::filesystem::exists(candidate))
		{
			return std::filesystem::weakly_canonical(candidate).string();
		}
	}
	return "";
}

// The GGUF defaults: next to the executable, then one level up (a build
// directory's parent is the repo root, where the hardlinks live), then the
// working directory. An explicitly given path is checked here too: a typo in
// --vae would otherwise only surface after the AR and the NAR, two minutes on
// the Arc and half an hour on the CPU later.
std::string resolve_gguf(const std::string & given, const char * name)
{
	if (!given.empty())
	{
		if (!std::filesystem::exists(given))
		{
			die("%s does not exist", given.c_str());
		}
		return given;
	}
	const std::string found = find_default_gguf(name);
	if (found.empty())
	{
		const std::filesystem::path here = exe_path().parent_path();
		die("cannot find %s — looked in %s, %s and the working directory",
		    name, here.c_str(), (here / "..").c_str());
	}
	return found;
}

// SPEC_SINGLE.md §2.7 / STATUS_NAR_PERF.md §8: the fast NAR flags differ per
// card, and the choice is by device name, not index.
void pick_nar_flags(const BatchParams & p, int64_t prefix_len, int64_t frames,
	NarParams & nar, std::string & card)
{
	if (p.device != "vulkan")
	{
		card = "CPU";
		printf("nar:     cpu path, default flags\n");
		return;
	}
	card = ggml_backend_dev_description(vulkan_device(p.gpu));
	nar.kv_f16        = true;
	nar.vk_f16_matmul = !p.nar_f32;
	if (card.find("Intel") != std::string::npos && !nar.vk_f16_matmul)
	{
		// Without coopmat (--nar-f32), ggml-vulkan's flash attention falls back
		// to its scalar kernel on Battlemage (2.6 TFLOP/s); one unfused query
		// tile wins — when it fits. Untiled,
		// the materialized scores are one S*N*n_head float buffer: 2.6 GB for a
		// 164 s song, and past Vulkan's per-allocation ceiling by ~5000 frames,
		// where the allocation fails outright. Keep the default tiling there.
		const int64_t N     = frames + 2;
		const int64_t S     = prefix_len + 2 * frames + 3;
		const int64_t bytes = S * N * 16 * 4;
		if (bytes < (int64_t) 3 << 30)
		{
			nar.query_chunk = 8192;
		} else {
			printf("nar:     untiled scores would need %.1f GiB in one buffer; "
			       "keeping --query-chunk %lld\n",
			       bytes / 1073741824.0, (long long) nar.query_chunk);
		}
	} else {
		nar.flash_attn = true;
	}
	const std::string attention = nar.flash_attn
		? std::string("--flash-attn")
		: "--query-chunk " + std::to_string(nar.query_chunk);
	printf("nar:     %s -> --kv-f16 %s%s\n", card.c_str(), attention.c_str(),
	       nar.vk_f16_matmul ? " --vk-f16-matmul" : "");
}

// `song` and `batch` run every stage in one process, and ggml-vulkan fixes
// matmul operand staging at Vulkan device init — so the VAE decodes at whatever
// precision the rest of the run is using and cannot be asked for another
// (SPEC_SINGLE.md §2.2). Refuse the flags that ask for one, in the parser,
// before a single model is loaded.
[[noreturn]] void die_vae_precision(const std::string & flag)
{
	die("%s: song/batch decode the VAE in-process at the NAR's Vulkan precision; "
	    "for the exact-F32 decode run the stage on its own: "
	    "yue2 vae -m yue2-vae-f32.gguf -i ARTIFACTS/latent.npy -o OUT.flac", flag.c_str());
}

json json_timing(const GenStats & st)
{
	json out = json::object();
	out["seconds"]         = st.seconds;
	out["prefill_seconds"] = st.prefill_seconds;
	out["ttft_seconds"]    = st.ttft_seconds;
	out["output_tokens"]   = st.output_tokens;
	out["content_tokens"]  = st.content_tokens;
	out["output_tps"]      = st.output_tps;
	out["prefix_tokens"]   = st.prefix_tokens;
	out["cfg_branches"]           = st.cfg_branches;
	out["guided_steps"]           = st.guided_steps;
	out["branch_prefill_seconds"] = st.branch_prefill_seconds;
	out["kept_frames"]            = st.kept_frames;
	out["keep_prefill_seconds"]   = st.keep_prefill_seconds;
	out["section_cuts"]           = st.section_cuts;
	out["section_prefill_seconds"] = st.section_prefill_seconds;
	out["execution"]       = "eager";
	out["attention"]       = "llama.cpp";
	return out;
}

// What the score template's holes cost, for a job that had one. SPEC_TEMPLATE §5.
json json_template(const TemplateStats & t)
{
	json out = json::object();
	out["holes"]          = t.holes;
	out["retries"]        = t.retries;
	out["rest_filled"]    = t.rest_filled;
	out["primer_tokens"]  = t.primer_tokens;
	out["given_tokens"]   = t.given_tokens;
	out["sampled_tokens"] = t.sampled_tokens;
	out["offlength_bars"] = t.offlength_bars;
	out["continued_tokens"] = t.continued_tokens;
	out["chord_lines"]      = t.chord_lines;
	return out;
}

// config.json, minus the fields that only meant something under torch (see
// src/STATUS_SINGLE.md for the list of drops).
void write_config(const std::string & dir, const BatchParams & p, const ArResult & ar,
	const std::string & draft_sha)
{
	json gen = json::object();
	gen["ode_steps"]  = p.steps;
	gen["ode_method"] = "midpoint";
	gen["context"]    = 24576;
	gen["version"]    = "yue2-native-v1";

	json cfg = json::object();
	cfg["generation"]      = gen;
	cfg["cot"]             = ar.cot;
	cfg["cfg_scale"]       = ar.cfg_scale;
	cfg["negative_style"]  = ar.negative_style.empty() ? json(nullptr) : json(ar.negative_style);
	cfg["negative_lyrics"] = ar.negative_lyrics;
	cfg["cfg_score"]       = ar.cfg_score;
	cfg["score_tempo"]     = ar.score_tempo != 0 ? json(ar.score_tempo) : json(nullptr);
	cfg["seed"]            = ar.seed;
	cfg["backend"]         = "yue2.cpp";
	cfg["ar_gguf"]         = p.ar_model;
	cfg["nar_gguf"]        = p.nar_model;
	cfg["vae_gguf"]        = p.vae_model;
	cfg["device"]          = p.device == "vulkan" ? "vulkan:" + std::to_string(p.gpu) : "cpu";
	cfg["card"]            = ar.card;
	cfg["nar_precision"]   = p.nar_f32 ? "f32" : "f16-staged";
	// SPEC_LORA §2: only present when adapters were merged, so a run without
	// --nar-lora writes exactly the config.json it always did.
	if (!p.nar_lora.empty())
	{
		json adapters = json::array();
		for (const lora::Spec & s : p.nar_lora)
		{
			json one = json::object();
			one["file"]     = s.file;
			one["strength"] = s.strength;
			one["sha256"]   = sha256_file_hex(s.file);
			adapters.push_back(one);
		}
		cfg["nar_lora"] = adapters;
	}
	// SPEC_DRAFT §5: only with --draft, so a run without it writes exactly the
	// config.json it always did.
	if (!p.draft.file.empty())
	{
		json draft = json::object();
		draft["file"]   = std::filesystem::path(p.draft.file).filename().string();
		draft["sha256"] = draft_sha;
		draft["type"]   = ar.draft_type;
		draft["k"]      = p.draft.k;
		draft["lambda"] = p.draft.lambda;
		draft["q"]      = p.draft.q_raw ? "raw" : "shaped";
		draft["window"] = p.draft.window;
		if (p.draft.extract_only)
		{
			draft["extract_only"] = true;
		}
		cfg["draft"] = draft;
	}
	cfg["vae_decode"]      = "halo_crop";
	cfg["vae_core_frames"] = 256;
	cfg["vae_halo_frames"] = 16;
	write_file_or_die(dir + "config.json", dump_py(cfg));
}

void write_result(const std::string & dir, const ArResult & ar, const std::string & card,
	int64_t frames, double nar_seconds, double vae_seconds, double e2e_seconds)
{
	json semantic = json::object();
	semantic["seconds"]       = ar.semantic.seconds;
	semantic["output_tokens"] = ar.semantic.output_tokens;
	semantic["attention"]     = "llama.cpp";
	// SPEC_DRAFT §5, only when a draft head was loaded.
	if (!ar.draft_type.empty())
	{
		const GenStats & st = ar.semantic;
		semantic["draft_rounds"]       = st.draft_rounds;
		semantic["draft_proposed"]     = st.draft_proposed;
		semantic["draft_accepted"]     = st.draft_accepted;
		semantic["tested_per_depth"]   = st.tested_per_depth;
		semantic["accepted_per_depth"] = st.accepted_per_depth;
		semantic["draft_seconds"]      = st.draft_seconds;
		semantic["verify_seconds"]     = st.verify_seconds;
	}

	json timing = json::object();
	timing["abc"]         = json_timing(ar.abc);
	timing["semantic"]    = semantic;
	timing["nar_seconds"] = nar_seconds;
	timing["vae_seconds"] = vae_seconds;
	timing["e2e_seconds"] = e2e_seconds;
	timing["card"]        = card;

	json truncated = json::object();
	truncated["abc"]      = ar.abc.truncated;
	truncated["semantic"] = ar.semantic.truncated;

	// Sorted, so two runs of the same request give the same result.json layout
	// (directory_iterator order is whatever the filesystem hands back).
	std::vector<std::string> names;
	for (const std::filesystem::directory_entry & e : std::filesystem::directory_iterator(dir))
	{
		if (e.is_regular_file() && e.path().filename() != "result.json")
		{
			names.push_back(e.path().filename().string());
		}
	}
	std::sort(names.begin(), names.end());

	json artifacts = json::object();
	for (const std::string & name : names)
	{
		json entry = json::object();
		entry["sha256"] = sha256_file_hex(dir + name);
		entry["bytes"]  = (uint64_t) std::filesystem::file_size(dir + name);
		artifacts[name] = entry;
	}

	// How the song was decoded: seed alone does not reproduce it, the batch
	// composition is part of the result (SPEC_BATCH §6).
	json batch = json::object();
	batch["parallel"] = ar.parallel;
	batch["slot"]     = ar.slot;
	batch["jobs"]     = ar.batch_jobs;

	json out = json::object();
	out["status"]        = "complete";
	out["truncated"]     = truncated;
	if (ar.is_template)
	{
		out["template"] = json_template(ar.tpl);
	}
	out["sample_rate"]   = SAMPLE_RATE;
	out["audio_seconds"] = (double) (DOWNSAMPLING * frames - 64) / SAMPLE_RATE;
	out["timing"]        = timing;
	out["batch"]         = batch;
	out["artifacts"]     = artifacts;
	write_file_or_die(dir + "result.json", dump_py(out));
}

} // namespace

std::string gguf_home_dir()
{
	return exe_path().parent_path().string();
}

SongParams parse_song_args(const char * argv0, int argc, char ** argv)
{
	SongParams p;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--request")
		{
			p.request_path = need(argc, argv, i);
		} else if (a == "--out" || a == "-o") {
			p.out = need(argc, argv, i);
		} else if (a == "--artifacts") {
			p.artifacts = need(argc, argv, i);
		} else if (a == "--ar") {
			p.ar_model = need(argc, argv, i);
		} else if (a == "--nar") {
			p.nar_model = need(argc, argv, i);
		} else if (a == "--vae") {
			p.vae_model = need(argc, argv, i);
		} else if (a == "--noise") {
			p.noise_path = need(argc, argv, i);
		} else if (a == "--device") {
			p.device = need(argc, argv, i);
		} else if (a == "--cpu") {
			p.device = "cpu";
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--steps") {
			p.steps = atoi(need(argc, argv, i));
		} else if (a == "--nar-f32") {
			p.nar_f32 = true;
		} else if (a == "--nar-lora") {
			p.nar_lora.push_back(lora::parse_arg(need(argc, argv, i)));
		} else if (a == "--guidance-trace") {
			p.guidance_trace = true;
		} else if (a == "--no-tags") {
			p.no_tags = true;
		} else if (a == "--opus-bitrate") {
			p.opus_bitrate = parse_opus_bitrate(need(argc, argv, i));
		} else if (a == "--seed") {
			p.has_seed = true;
			p.seed     = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "--vk-f16-matmul" || a == "--no-vk-f16-matmul") {
			die_vae_precision(a);
		} else if (parse_draft_arg(a, argc, argv, i, p.draft)) {
			continue;
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument %s", a.c_str());
		}
	}
	check_draft_args(p.draft);
	if (p.request_path.empty() || p.out.empty())
	{
		usage(argv0);
		die("--request and --out are required");
	}
	// An Opus-less build refuses the output name here, before any model loads.
	const std::string opus_err = opus::check_support(p.out);
	if (!opus_err.empty())
	{
		die("%s", opus_err.c_str());
	}
	if (p.steps < 1)
	{
		die("--steps must be >= 1");
	}
	return p;
}

BatchParams parse_batch_args(const char * argv0, int argc, char ** argv)
{
	BatchParams p;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--jobs")
		{
			p.jobs_path = need(argc, argv, i);
		} else if (a == "--summary") {
			p.summary = need(argc, argv, i);
		} else if (a == "--ar") {
			p.ar_model = need(argc, argv, i);
		} else if (a == "--nar") {
			p.nar_model = need(argc, argv, i);
		} else if (a == "--vae") {
			p.vae_model = need(argc, argv, i);
		} else if (a == "--device") {
			p.device = need(argc, argv, i);
		} else if (a == "--cpu") {
			p.device = "cpu";
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--steps") {
			p.steps = atoi(need(argc, argv, i));
		} else if (a == "--parallel") {
			p.parallel = atoi(need(argc, argv, i));
		} else if (a == "--threads") {
			p.threads = atoi(need(argc, argv, i));
		} else if (a == "--greedy") {
			p.greedy = true;
		} else if (a == "--max-abc") {
			p.max_abc = atoi(need(argc, argv, i));
			if (p.max_abc < 1)
			{
				die("--max-abc must be >= 1");
			}
		} else if (a == "--max-semantic") {
			p.max_semantic = atoi(need(argc, argv, i));
			if (p.max_semantic < 1)
			{
				die("--max-semantic must be >= 1");
			}
		} else if (a == "--nar-f32") {
			p.nar_f32 = true;
		} else if (a == "--nar-lora") {
			p.nar_lora.push_back(lora::parse_arg(need(argc, argv, i)));
		} else if (a == "--continue-on-error") {
			p.continue_on_error = true;
		} else if (a == "--guidance-trace") {
			p.guidance_trace = true;
		} else if (a == "--no-tags") {
			p.no_tags = true;
		} else if (a == "--opus-bitrate") {
			p.opus_bitrate = parse_opus_bitrate(need(argc, argv, i));
		} else if (a == "--seed") {
			p.has_seed = true;
			p.seed     = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "--vk-f16-matmul" || a == "--no-vk-f16-matmul") {
			die_vae_precision(a);
		} else if (parse_draft_arg(a, argc, argv, i, p.draft)) {
			continue;
		} else if (a == "-h" || a == "--help") {
			usage_batch(argv0);
			exit(0);
		} else {
			usage_batch(argv0);
			die("unknown argument %s", a.c_str());
		}
	}
	check_draft_args(p.draft);
	if (p.jobs_path.empty())
	{
		usage_batch(argv0);
		die("--jobs jobs.json is required");
	}
	if (p.steps < 1)
	{
		die("--steps must be >= 1");
	}
	if (p.parallel < 1)
	{
		die("--parallel must be >= 1");
	}
	return p;
}

// SPEC_BATCH §4.5: the AR decodes every job in one context and frees it, then
// the NAR and the VAE run per job in file order — both are compute-bound and
// gain nothing from batching, and freeing first keeps the peak at
// max(AR batch, one NAR) instead of their sum.
int run_batch(const BatchParams & given, std::vector<ArJob> jobs)
{
	ggml_time_init();
	const double t_batch0 = now_seconds();

	// The jobs file has just been read and nothing has been loaded yet: an
	// Opus-less build turns an .opus output away here, not after the render.
	for (const ArJob & job : jobs)
	{
		const std::string err = opus::check_support(job.out);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
	}

	// ggml-vulkan reads the exactness switch once per device init, and the AR
	// inits the device first — so --nar-f32 has to be set here, not inside
	// run_nar, or the NAR would silently keep the fp16-staged pipelines. It
	// covers the whole run, VAE included: one process, one precision.
	if (given.nar_f32 && given.device == "vulkan")
	{
		vulkan_want_exact_f32();
	}

	// SPEC_CONVERT §1: a default GGUF that is nowhere to be found is made from
	// the Hugging Face download, once, next to the binary — but never one the
	// command line named.
	const bool convert_ar  = given.ar_model.empty()  && find_default_gguf("yue2-ar-q8_0.gguf").empty();
	const bool convert_nar = given.nar_model.empty() && find_default_gguf("yue2-nar-f16.gguf").empty();
	const bool convert_vae = given.vae_model.empty() && find_default_gguf("yue2-vae-f32.gguf").empty();
	if (convert_ar || convert_nar || convert_vae)
	{
		convert_missing_defaults(gguf_home_dir(), convert_ar, convert_nar, convert_vae);
	}

	BatchParams p = given;
	p.ar_model    = resolve_gguf(p.ar_model,  "yue2-ar-q8_0.gguf");
	p.nar_model   = resolve_gguf(p.nar_model, "yue2-nar-f16.gguf");
	p.vae_model   = resolve_gguf(p.vae_model, "yue2-vae-f32.gguf");

	// The latent always goes to the artifacts directory: the VAE stage reads it
	// from there, so a job without `artifacts` still needs one and cleans it up.
	std::vector<bool> temp_dir(jobs.size(), false);
	std::error_code   ec;
	for (size_t k = 0; k < jobs.size(); k++)
	{
		if (jobs[k].artifacts.empty())
		{
			jobs[k].artifacts = temp_artifacts_dir(k);
			temp_dir[k]       = true;
		}
		if (p.has_seed && !jobs[k].has_seed)
		{
			jobs[k].has_seed = true;
			jobs[k].seed     = p.seed;
		}
		// The trace belongs beside the artifacts it explains, so a job whose
		// directory is the temporary one this run removes traces nothing.
		jobs[k].trace = p.guidance_trace && !temp_dir[k];
		if (p.guidance_trace && temp_dir[k])
		{
			printf("--guidance-trace: job %zu has no artifacts directory to write "
			       "guidance_trace.npy into; nothing traced\n", k + 1);
		}

		const std::filesystem::path out_parent = std::filesystem::path(jobs[k].out).parent_path();
		if (!out_parent.empty())
		{
			std::filesystem::create_directories(out_parent, ec);
		}
		// A stale output from a previous run must not survive a failure of this
		// one: a driver that only checks for the file would take it for fresh.
		if (std::filesystem::is_regular_file(jobs[k].out))
		{
			std::filesystem::remove(jobs[k].out, ec);
		}
	}

	// ---- AR, all jobs in one context ---------------------------------------

	ArBatchParams ar_params;
	ar_params.model             = p.ar_model;
	ar_params.device            = p.device;
	ar_params.gpu               = p.gpu;
	ar_params.threads           = p.threads;
	ar_params.parallel          = p.parallel;
	ar_params.greedy            = p.greedy;
	ar_params.max_abc           = p.max_abc;
	ar_params.max_semantic      = p.max_semantic;
	ar_params.continue_on_error = p.continue_on_error;
	ar_params.draft             = p.draft;

	// A bad "tags" block is a request error, so it is one before anything renders.
	std::vector<flac::Tags> tags(jobs.size());
	for (size_t k = 0; k < jobs.size() && !p.no_tags; k++)
	{
		const std::string err = request_tags(jobs[k].request_path, jobs[k].base_dir, tags[k]);
		if (!err.empty())
		{
			die("%s: %s", jobs[k].request_path.c_str(), err.c_str());
		}
	}

	// The requests as given, read now: a request that sits in its own artifacts
	// dir as request.json is overwritten by the one the AR stage normalises.
	std::vector<std::string> as_given(jobs.size());
	for (size_t k = 0; k < jobs.size(); k++)
	{
		const std::string err = read_file(jobs[k].request_path, as_given[k]);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
		// Text named by "lyrics_file" & co. is kept as read, so the artifacts
		// never point back at a file that may have changed since.
		json root = json::parse(as_given[k], nullptr, false);
		int  baked = 0;
		if (resolve_request_files(root, jobs[k].base_dir, &baked).empty() && baked > 0)
		{
			as_given[k] = dump_py(root);
		}
	}

	std::vector<ArResult> ar(jobs.size());
	run_ar_batch(ar_params, jobs, ar);

	// ---- NAR + VAE, per job, in file order ---------------------------------

	json summary  = json::array();
	int  failed   = 0;
	// The head's, for every config.json: hashed once, after the AR is done with it.
	const std::string draft_sha = p.draft.file.empty() ? "" : sha256_file_hex(p.draft.file);
	for (size_t k = 0; k < jobs.size(); k++)
	{
		const ArJob & job = jobs[k];
		const std::string tag = jobs.size() > 1
			? "[" + std::to_string(k + 1) + "/" + std::to_string(jobs.size()) + " " +
			  std::filesystem::path(job.out).filename().string() + "] "
			: std::string();

		json entry     = json::object();
		entry["out"]   = job.out;
		if (!ar[k].ok)
		{
			entry["status"] = "error";
			entry["error"]  = ar[k].error;
			summary.push_back(entry);
			failed++;
			continue;
		}

		const std::string dir = job.artifacts + "/";

		// The request as given, next to the request.json the AR stage normalises.
		write_file_or_die(dir + "ar_request.json", as_given[k]);

		const int64_t frames = (int64_t) ar[k].codes.size();
		if (frames < 1)
		{
			die("%sthe AR stage produced no semantic tokens", tag.c_str());
		}

		// ---- noise ---------------------------------------------------------

		std::vector<float> noise_data;
		if (job.noise_path.empty())
		{
			noise_data = noise::gaussian(ar[k].seed, frames, LATENT_DIM);
			printf("%snoise:   mt19937_64(%llu) + Box-Muller, [%lld, %d]\n", tag.c_str(),
			       (unsigned long long) ar[k].seed, (long long) frames, LATENT_DIM);
		} else {
			npy::Array loaded;
			const std::string err = npy::load(job.noise_path.c_str(), loaded);
			if (!err.empty())
			{
				die("%s%s", tag.c_str(), err.c_str());
			}
			if (loaded.shape.size() != 2 || loaded.shape[0] < frames ||
			    loaded.shape[1] != LATENT_DIM)
			{
				die("%s--noise must be float32 [>=%lld, %d]",
				    tag.c_str(), (long long) frames, LATENT_DIM);
			}
			loaded.data.resize((size_t) (frames * LATENT_DIM));
			noise_data.swap(loaded.data);
			printf("%snoise:   %s\n", tag.c_str(), job.noise_path.c_str());
		}
		{
			const std::string err = npy::save((dir + "nar_noise.npy").c_str(),
				{ frames, (int64_t) LATENT_DIM }, noise_data.data());
			if (!err.empty())
			{
				die("%s%s", tag.c_str(), err.c_str());
			}
		}

		// ---- NAR -----------------------------------------------------------

		NarParams nar;
		nar.ar_model  = p.ar_model;
		nar.model     = p.nar_model;
		nar.output    = dir + "latent.npy";
		nar.device    = p.device;
		nar.gpu       = p.gpu;
		nar.steps     = p.steps;
		nar.nar_lora  = p.nar_lora;
		nar.prefix_in = &ar[k].prefix_sem;
		nar.codec_in  = &ar[k].codes;
		nar.noise_in  = &noise_data;

		std::string card;
		pick_nar_flags(p, (int64_t) ar[k].prefix_sem.size(), frames, nar, card);

		const double t_nar0 = now_seconds();
		run_nar(nar);
		const double nar_seconds = now_seconds() - t_nar0;

		// ---- VAE -----------------------------------------------------------
		// In this process, on the device the NAR just used, at the same matmul
		// precision: the fp16-staged decode is 58.5 dB from the exact-F32 one
		// and a listening test could not separate them, while the exact path
		// needs a Vulkan device init the NAR cannot share
		// (docs/vulkan_burst_investigation.md). `yue2 vae` remains the exact
		// reference decoder.
		VaeParams vae;
		vae.model         = p.vae_model;
		vae.input         = nar.output;
		vae.output        = job.out;
		vae.device        = p.device;
		vae.gpu           = p.gpu;
		vae.threads       = p.threads;
		vae.vk_f16_matmul = nar.vk_f16_matmul;
		vae.opus_bitrate  = p.opus_bitrate;
		if (!p.no_tags)
		{
			// Finds the artifacts directory a FLAC came from without saying what is in it.
			vae.tags = tags[k];
			vae.tags.emplace_back("YUE2_ID", sha256_file_hex(dir + "semantic.npy").substr(0, 16));
		}

		const double t_vae0 = now_seconds();
		run_vae(vae);
		const double vae_seconds = now_seconds() - t_vae0;

		// ---- artifacts -----------------------------------------------------

		const double e2e_seconds = now_seconds() - t_batch0;
		write_config(dir, p, ar[k], draft_sha);
		write_result(dir, ar[k], card, frames, nar_seconds, vae_seconds, e2e_seconds);

		printf("%stotal:   abc %.1f s, semantic %.1f s, nar %.1f s, vae %.1f s\n", tag.c_str(),
		       ar[k].abc.seconds, ar[k].semantic.seconds, nar_seconds, vae_seconds);
		printf("%s[gen] %.1fs\n", tag.c_str(), e2e_seconds);
		printf("%s[done] %s\n", tag.c_str(), job.out.c_str());

		if (temp_dir[k])
		{
			std::filesystem::remove_all(job.artifacts, ec);
		}

		entry["status"]        = "ok";
		entry["audio_seconds"] = (double) (DOWNSAMPLING * frames - 64) / SAMPLE_RATE;
		entry["e2e_seconds"]   = e2e_seconds;
		if (ar[k].is_template)
		{
			entry["template"] = json_template(ar[k].tpl);
		}
		summary.push_back(entry);
	}

	if (!p.summary.empty())
	{
		write_file_or_die(p.summary, dump_py(summary));
	}
	if (jobs.size() > 1)
	{
		printf("batch:   %zu/%zu jobs rendered in %.1f s\n",
		       jobs.size() - (size_t) failed, jobs.size(), now_seconds() - t_batch0);
	}
	remove_temp_artifacts();
	return failed == 0 ? 0 : 1;
}

int run_song(const SongParams & p)
{
	BatchParams bp;
	bp.ar_model  = p.ar_model;
	bp.nar_model = p.nar_model;
	bp.vae_model = p.vae_model;
	bp.device    = p.device;
	bp.gpu       = p.gpu;
	bp.steps     = p.steps;
	bp.parallel  = 1;          // the reproducible path, SPEC_BATCH §6
	bp.nar_f32   = p.nar_f32;
	bp.nar_lora  = p.nar_lora;
	bp.guidance_trace = p.guidance_trace;
	bp.no_tags   = p.no_tags;
	bp.opus_bitrate = p.opus_bitrate;
	bp.draft     = p.draft;

	ArJob job;
	job.request_path = p.request_path;
	// SPEC_KEEP §2: a relative "semantic_keep" path is beside the request file.
	job.base_dir     = std::filesystem::path(p.request_path).parent_path().string();
	job.out          = p.out;
	job.artifacts    = p.artifacts;
	job.noise_path   = p.noise_path;
	job.has_seed     = p.has_seed;
	job.seed         = p.seed;

	return run_batch(bp, std::vector<ArJob>(1, job));
}
