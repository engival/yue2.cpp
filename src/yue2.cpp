// yue2 — one binary for the whole pipeline. `yue2 song` renders a request to
// audio in a single process; `yue2 ar|nar|vae` are the individual stages with
// the flags yue2-ar / yue2-nar / yue2-vae take. See SPEC_SINGLE.md.

#include "stage_ar.hpp"
#include "stage_nar.hpp"
#include "stage_song.hpp"
#include "stage_vae.hpp"

#include "common/noise.hpp"
#include "common/util.hpp"

#include "npy.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void usage()
{
	fprintf(stderr,
	        "usage: yue2 song  --request R.json --out X.flac [--artifacts DIR] ...\n"
	        "       yue2 batch --jobs jobs.json [--parallel N] [--summary FILE] ...\n"
	        "       yue2 ar    -m AR.gguf --request R.json --artifacts DIR ...\n"
	        "       yue2 nar   --ar AR.gguf -m NAR.gguf --artifacts DIR -o latent.npy ...\n"
	        "       yue2 vae   -m VAE.gguf -i latent.npy -o out.flac ...\n"
	        "       yue2 noise --seed N --frames T -o noise.npy\n"
	        "\n"
	        "`yue2 <stage> --help` prints that stage's own flags.\n");
}

// The NAR's noise generator on its own, so the golden in tests/golden/ can be
// regenerated and compared byte for byte (SPEC_SINGLE.md §2.4).
static int run_noise(int argc, char ** argv)
{
	uint64_t    seed   = 0;
	int64_t     frames = 0;
	std::string out;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--seed")
		{
			seed = strtoull(need(argc, argv, i), nullptr, 10);
		} else if (a == "--frames") {
			frames = atoll(need(argc, argv, i));
		} else if (a == "-o" || a == "--output") {
			out = need(argc, argv, i);
		} else {
			die("unknown argument '%s'", argv[i]);
		}
	}
	if (frames < 1 || out.empty())
	{
		die("yue2 noise needs --frames T and -o FILE");
	}

	const std::vector<float> data = noise::gaussian(seed, frames, 64);
	const std::string        err  = npy::save(out.c_str(), { frames, 64 }, data.data());
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
	printf("wrote:   %s [%lld, 64] from mt19937_64(%llu)\n",
	       out.c_str(), (long long) frames, (unsigned long long) seed);
	return 0;
}

int main(int argc, char ** argv)
{
	// A driver that runs `yue2 batch` through a pipe sees a fully buffered
	// stdout, i.e. nothing for minutes. Line-buffer it once, here, rather than
	// scattering fflush() through the stages.
	setvbuf(stdout, nullptr, _IOLBF, 0);

	if (argc < 2)
	{
		usage();
		return 1;
	}

	// The stage parsers read argv[1..]; handing them a window that starts at the
	// subcommand puts the first real flag at index 1 again.
	const std::string cmd  = argv[1];
	const int         sub  = argc - 1;
	char ** const     rest = argv + 1;

	if (cmd == "song")
	{
		return run_song(parse_song_args("yue2 song", sub, rest));
	}
	if (cmd == "batch")
	{
		const BatchParams p = parse_batch_args("yue2 batch", sub, rest);
		std::vector<ArJob> jobs;
		const std::string  err = load_jobs_file(p.jobs_path, true, jobs);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
		return run_batch(p, jobs);
	}
	if (cmd == "ar")
	{
		return run_ar(parse_ar_args("yue2 ar", sub, rest), nullptr);
	}
	if (cmd == "nar")
	{
		return run_nar(parse_nar_args("yue2 nar", sub, rest));
	}
	if (cmd == "vae")
	{
		return run_vae(parse_vae_args("yue2 vae", sub, rest));
	}
	if (cmd == "noise")
	{
		return run_noise(sub, rest);
	}

	usage();
	fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
	return 1;
}
