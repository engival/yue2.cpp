// yue2 song — AR (libllama) -> NAR (raw ggml) -> VAE, one request in, one FLAC
// out. See SPEC_SINGLE.md; the stages themselves are unchanged, this is the
// plumbing that used to live in yue2_gen.py.

#include "stage_song.hpp"

#include "stage_ar.hpp"
#include "stage_nar.hpp"
#include "stage_vae.hpp"

#include "common/device.hpp"
#include "common/fileio.hpp"
#include "common/noise.hpp"
#include "common/util.hpp"

#include "npy.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// in /tmp for a batch driver to accumulate. The forked VAE child leaves through
// execv() or _exit(), neither of which runs atexit, so it never removes it.
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
	const std::string dir = (std::filesystem::temp_directory_path() /
		("yue2song." + std::to_string(getpid()) + "." + std::to_string(job))).string();
	if (dir.find("/yue2song.") == std::string::npos)
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

void usage(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s --request R.json --out X.flac [--artifacts DIR] [--seed N]\n"
	        "        [--ar AR.gguf] [--nar NAR.gguf] [--vae VAE.gguf]\n"
	        "        [--device cpu|vulkan] [--gpu N] [--nar-f32] [--noise FILE] [--steps 32]\n", argv0);
}

void usage_batch(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s --jobs jobs.json [--parallel N] [--summary FILE]\n"
	        "        [--ar AR.gguf] [--nar NAR.gguf] [--vae VAE.gguf] [--seed N]\n"
	        "        [--device cpu|vulkan] [--gpu N] [--nar-f32] [--steps 32]\n"
	        "        [--threads N] [--greedy] [--max-abc N] [--max-semantic N]\n"
	        "        [--continue-on-error]\n"
	        "\n"
	        "jobs.json is an array of { \"request\", \"out\", \"artifacts\", \"seed\", \"noise\" };\n"
	        "request and out are required and paths are relative to the working directory.\n",
	        argv0);
}

// Directory holding this executable, via /proc/self/exe — the same path the
// VAE child is started from.
std::string exe_path()
{
	char    buf[4096];
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (n <= 0)
	{
		die("cannot read /proc/self/exe");
	}
	buf[n] = '\0';
	return buf;
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
	const std::filesystem::path here = std::filesystem::path(exe_path()).parent_path();
	const std::string tried[] = { (here / name).string(), (here / ".." / name).string(), name };
	for (const std::string & candidate : tried)
	{
		if (std::filesystem::exists(candidate))
		{
			return std::filesystem::weakly_canonical(candidate).string();
		}
	}
	die("cannot find %s — looked in %s, %s and the working directory",
	    name, here.c_str(), (here / "..").c_str());
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
	if (card.find("Intel") != std::string::npos)
	{
		// ggml-vulkan's flash-attention kernel is untuned for Battlemage
		// (2.6 TFLOP/s); one unfused query tile wins — when it fits. Untiled,
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

// The VAE needs true-F32 matmuls and the NAR's fast path needs the opposite;
// ggml-vulkan reads that switch once per device init, so the VAE runs as a
// child of this process rather than in it (SPEC_SINGLE.md §2.2). It loads in
// ~1 s, so residency would buy nothing anyway.
double run_vae_child(const BatchParams & p, const std::string & out,
	const std::string & latent_path, std::string & err)
{
	const std::string exe = exe_path();
	const std::string gpu = std::to_string(p.gpu);
	err.clear();

	std::vector<std::string> args = { "yue2", "vae",
		"-m", p.vae_model, "-i", latent_path, "-o", out,
		"--device", p.device, "--gpu", gpu };

	std::vector<char *> argv;
	for (std::string & a : args)
	{
		argv.push_back(&a[0]);
	}
	argv.push_back(nullptr);

	printf("vae:     %s vae (child process)\n", exe.c_str());
	fflush(stdout);

	const double t0  = now_seconds();
	const pid_t  pid = fork();
	if (pid < 0)
	{
		die("fork for the VAE child failed: %s", strerror(errno));
	}
	if (pid == 0)
	{
		// The child sets GGML_VK_DISABLE_F16 / _COOPMAT itself, exactly as a
		// hand-run `yue2 vae` does — which is what the equivalence check needs.
		execv(exe.c_str(), argv.data());
		fprintf(stderr, "error: cannot exec %s: %s\n", exe.c_str(), strerror(errno));
		_exit(127);
	}

	int status = 0;
	if (waitpid(pid, &status, 0) != pid)
	{
		die("waitpid on the VAE child failed: %s", strerror(errno));
	}
	char note[128];
	if (WIFSIGNALED(status))
	{
		snprintf(note, sizeof(note), "the VAE child was killed by signal %d (%s)",
		         WTERMSIG(status), strsignal(WTERMSIG(status)));
		err = note;
	} else if (!WIFEXITED(status)) {
		snprintf(note, sizeof(note), "the VAE child did not exit normally (wait status 0x%x)",
		         (unsigned) status);
		err = note;
	} else if (WEXITSTATUS(status) != 0) {
		snprintf(note, sizeof(note), "the VAE child exited %d", WEXITSTATUS(status));
		err = note;
	}
	return now_seconds() - t0;
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
	out["cfg_branches"]    = 1;
	out["execution"]       = "eager";
	out["attention"]       = "llama.cpp";
	return out;
}

// config.json, minus the fields that only meant something under torch (see
// src/STATUS_SINGLE.md for the list of drops).
void write_config(const std::string & dir, const BatchParams & p, const ArResult & ar)
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
	cfg["seed"]            = ar.seed;
	cfg["backend"]         = "yue2.cpp";
	cfg["ar_gguf"]         = p.ar_model;
	cfg["nar_gguf"]        = p.nar_model;
	cfg["vae_gguf"]        = p.vae_model;
	cfg["device"]          = p.device == "vulkan" ? "vulkan:" + std::to_string(p.gpu) : "cpu";
	cfg["card"]            = ar.card;
	cfg["nar_precision"]   = p.nar_f32 ? "f32" : "f16-staged";
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
	out["sample_rate"]   = SAMPLE_RATE;
	out["audio_seconds"] = (double) (DOWNSAMPLING * frames - 64) / SAMPLE_RATE;
	out["timing"]        = timing;
	out["batch"]         = batch;
	out["artifacts"]     = artifacts;
	write_file_or_die(dir + "result.json", dump_py(out));
}

} // namespace

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
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--steps") {
			p.steps = atoi(need(argc, argv, i));
		} else if (a == "--nar-f32") {
			p.nar_f32 = true;
		} else if (a == "--seed") {
			p.has_seed = true;
			p.seed     = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument %s", a.c_str());
		}
	}
	if (p.request_path.empty() || p.out.empty())
	{
		usage(argv0);
		die("--request and --out are required");
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
		} else if (a == "--continue-on-error") {
			p.continue_on_error = true;
		} else if (a == "--seed") {
			p.has_seed = true;
			p.seed     = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "-h" || a == "--help") {
			usage_batch(argv0);
			exit(0);
		} else {
			usage_batch(argv0);
			die("unknown argument %s", a.c_str());
		}
	}
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

	// The VAE stage re-executes this binary (§2.2), so check now rather than
	// three minutes from now that it is still on disk: a rebuild mid-run leaves
	// /proc/self/exe reading "<path> (deleted)" and execv would fail at the end.
	if (!std::filesystem::exists(exe_path()))
	{
		die("this executable is no longer on disk (%s); the VAE stage re-executes it "
		    "— rerun the rebuilt binary", exe_path().c_str());
	}

	// ggml-vulkan reads the exactness switch once per device init, and the AR
	// inits the device first — so --nar-f32 has to be set here, not inside
	// run_nar, or the NAR would silently keep the fp16-staged pipelines.
	if (given.nar_f32 && given.device == "vulkan")
	{
		vulkan_want_exact_f32();
	}

	BatchParams p = given;
	p.ar_model    = resolve_gguf(p.ar_model,  "yue2-ar-q8_0.gguf");
	p.nar_model   = resolve_gguf(p.nar_model, "yue2-nar-f16.gguf");
	p.vae_model   = resolve_gguf(p.vae_model, "yue2-vae-f32.gguf");

	// The latent always goes to the artifacts directory: the VAE child reads it
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

	std::vector<ArResult> ar(jobs.size());
	run_ar_batch(ar_params, jobs, ar);

	// ---- NAR + VAE, per job, in file order ---------------------------------

	json summary  = json::array();
	int  failed   = 0;
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
		{
			std::string body;
			const std::string err = read_file(job.request_path, body);
			if (!err.empty())
			{
				die("%s", err.c_str());
			}
			write_file_or_die(dir + "ar_request.json", body);
		}

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
		nar.prefix_in = &ar[k].prefix_sem;
		nar.codec_in  = &ar[k].codes;
		nar.noise_in  = &noise_data;

		std::string card;
		pick_nar_flags(p, (int64_t) ar[k].prefix_sem.size(), frames, nar, card);

		const double t_nar0 = now_seconds();
		run_nar(nar);
		const double nar_seconds = now_seconds() - t_nar0;

		// ---- VAE (child) ---------------------------------------------------

		std::string  vae_err;
		const double vae_seconds = run_vae_child(p, job.out, nar.output, vae_err);
		if (!vae_err.empty())
		{
			if (!p.continue_on_error)
			{
				die("%s%s", tag.c_str(), vae_err.c_str());
			}
			fprintf(stderr, "error: %s%s\n", tag.c_str(), vae_err.c_str());
			entry["status"] = "error";
			entry["error"]  = vae_err;
			summary.push_back(entry);
			failed++;
			continue;
		}

		// ---- artifacts -----------------------------------------------------

		const double e2e_seconds = now_seconds() - t_batch0;
		write_config(dir, p, ar[k]);
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

	ArJob job;
	job.request_path = p.request_path;
	job.out          = p.out;
	job.artifacts    = p.artifacts;
	job.noise_path   = p.noise_path;
	job.has_seed     = p.has_seed;
	job.seed         = p.seed;

	return run_batch(bp, std::vector<ArJob>(1, job));
}
