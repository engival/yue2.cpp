// yue2-sampler-diff — SPEC_SAMPLER.md §4.2(b): drive the stage-6 sparse
// `sample_step` and the frozen stage-5 `sample_step_ref` over the same
// synthetic logits and assert they return the same token, every time.
//
// It includes stage_ar.cpp rather than linking it: both samplers live in that
// file's anonymous namespace, which is exactly where a same-translation-unit
// test can reach them, and it guarantees the code under test is the code that
// ships (no second copy to drift).
//
//	build_sampler/yue2-sampler-diff [--cases N] [--seed N] [--block N] [-v]
//	                                [--only ref|new]      # microbenchmark one side
//	                                [--shipped]           # only the two real configs
//
// A "block" is one freshly generated logits buffer; the cases inside it perturb
// it, so N cases cost N small edits and N/block full buffers. Exit 0 = every
// case agreed.

#include "stage_ar.cpp"

#include <cinttypes>

namespace
{

// xorshift128+, so the case stream is reproducible from --seed and independent
// of whatever the samplers do to their own std::mt19937_64.
struct Rand
{
	uint64_t a;
	uint64_t b;

	explicit Rand(uint64_t seed)
	{
		a = seed * 0x9e3779b97f4a7c15ull + 0x243f6a8885a308d3ull;
		b = seed ^ 0xbf58476d1ce4e5b9ull;
		for (int i = 0; i < 8; i++)
		{
			next();
		}
	}

	uint64_t next()
	{
		uint64_t x = a;
		const uint64_t y = b;
		a = y;
		x ^= x << 23;
		b = x ^ y ^ (x >> 17) ^ (y >> 26);
		return b + y;
	}

	// [0, n)
	int below(int n)
	{
		return n <= 0 ? 0 : (int) (next() % (uint64_t) n);
	}

	double unit()
	{
		return (double) (next() >> 11) * (1.0 / 9007199254740992.0);
	}

	// A rough normal, good enough to look like a logit row.
	float normal(float sigma)
	{
		double s = 0;
		for (int i = 0; i < 6; i++)
		{
			s += unit();
		}
		return (float) ((s - 3.0) * 1.4142135623730951 * (double) sigma);
	}
};

// How a block's logits are shaped. `quantised` is the interesting one: rounding
// to a coarse grid makes exact ties at the top-k boundary the common case
// instead of a once-in-a-run accident.
enum Shape { SHAPE_NORMAL, SHAPE_QUANTISED, SHAPE_COARSE, SHAPE_FLAT, SHAPE_SPIKY, SHAPE_N };

static const char * shape_name(int shape)
{
	switch (shape)
	{
		case SHAPE_NORMAL:    return "normal";
		case SHAPE_QUANTISED: return "quantised";
		case SHAPE_COARSE:    return "coarse";
		case SHAPE_FLAT:      return "flat";
		default:              return "spiky";
	}
}

static void fill_block(Rand & rnd, int shape, std::vector<float> & logits)
{
	const float flat = rnd.normal(4.0f);
	for (size_t i = 0; i < logits.size(); i++)
	{
		float v = rnd.normal(3.0f);
		switch (shape)
		{
			case SHAPE_QUANTISED: v = std::floor(v * 4.0f) / 4.0f;    break;
			case SHAPE_COARSE:    v = std::floor(v);                  break;
			case SHAPE_FLAT:      v = flat;                           break;
			case SHAPE_SPIKY:     v = rnd.below(64) == 0 ? v + 30.0f : v - 30.0f; break;
			default:                                                  break;
		}
		logits[i] = v;
	}

	// Infinities on both sides: -inf must drop out of the candidate set on both
	// paths, +inf must dominate the heap. NaN is deliberately never generated
	// (SPEC_SAMPLER §4.2b) — the stage-5 code excluded it and nothing produces it.
	for (int i = 0; i < 16; i++)
	{
		logits[(size_t) rnd.below((int) logits.size())] =
			rnd.below(2) == 0 ? -std::numeric_limits<float>::infinity()
			                  :  std::numeric_limits<float>::infinity();
	}
}

// One id the case is allowed to touch: biased hard towards the two allowed sets
// and towards the end tokens, which is where the two implementations differ
// structurally (bounds vs mask).
static int poke_id(Rand & rnd, bool phase_abc)
{
	switch (rnd.below(8))
	{
		case 0:  return ABC_END;
		case 1:  return MUSIC_END;
		case 2:  return EOD;
		case 3:  return CODEC_OFFSET;
		case 4:  return CODEC_OFFSET + CODEC_SIZE - 1;
		case 5:  return rnd.below(VOCAB_SIZE);
		default: return phase_abc ? rnd.below(EOD)
		                          : CODEC_OFFSET + rnd.below(CODEC_SIZE);
	}
}

} // namespace

int main(int argc, char ** argv)
{
	long long cases   = 100000;
	int       block   = 100;
	uint64_t  seed    = 20260912;
	bool      verbose = false;
	bool      run_ref = true;
	bool      run_new = true;
	bool      shipped = false;

	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "--cases")
		{
			cases = atoll(need(argc, argv, i));
		} else if (a == "--block") {
			block = atoi(need(argc, argv, i));
		} else if (a == "--seed") {
			seed = strtoull(need(argc, argv, i), nullptr, 10);
		} else if (a == "--only") {
			// Microbenchmark: run one sampler, not both. The case stream is
			// identical either way, so the two wall times are comparable.
			const std::string which = need(argc, argv, i);
			run_ref = which == "ref";
			run_new = which == "new";
			if (!run_ref && !run_new)
			{
				die("--only takes ref or new");
			}
		} else if (a == "--shipped") {
			// Only the two configs a song actually decodes with, in the ratio a
			// song actually decodes them (abc is ~28 % of a full-length song),
			// so the timing means "per real token" rather than "per grid point".
			shipped = true;
		} else if (a == "-v" || a == "--verbose") {
			verbose = true;
		} else {
			die("usage: %s [--cases N] [--block N] [--seed N] [--only ref|new] [--shipped] [-v]",
			    argv[0]);
		}
	}
	if (cases < 1 || block < 1)
	{
		die("--cases and --block must be >= 1");
	}

	Rand               rnd(seed);
	std::vector<float> logits((size_t) VOCAB_SIZE, 0.0f);
	SampleScratch      scratch;

	// The two shipped configs are the ones that matter; the rest of the grid
	// walks top_k / top_p / penalty / window around them.
	const double temps[4]   = {0.0, 0.7, 1.0, 1.3};
	const int    topks[6]   = {1, 2, 3, 30, 100, 400};
	const double topps[5]   = {0.0, 0.5, 0.9, 0.95, 1.0};
	const double pens[4]    = {1.0, 1.005, 1.2, 2.0};
	const int    windows[4] = {0, 1, 50, 100};

	const double t0 = now_seconds();

	long long mismatches = 0;

	// Hand-built boundary cases first: a flat floor, exactly top_k ids raised to
	// one value, the `end` token raised to that same value (so k+1 sit exactly on
	// the threshold), and one id an ulp under it. Every temperature, both phases,
	// `end` masked and unmasked.
	if (run_ref && run_new)
	{
		const int fixed_topks[3] = {1, 30, 100};
		for (int ti = 0; ti < 4; ti++)
		for (int ki = 0; ki < 3; ki++)
		for (int phase = 0; phase < 2; phase++)
		for (int masked = 0; masked < 2; masked++)
		{
			const bool abc = phase == 1;
			std::fill(logits.begin(), logits.end(), 0.0f);
			const int base = abc ? 0 : CODEC_OFFSET;
			for (int k = 0; k < fixed_topks[ki]; k++)
			{
				logits[(size_t) (base + 7 * k)] = 1.0f;
			}
			logits[(size_t) (abc ? ABC_END : MUSIC_END)] = 1.0f;
			logits[(size_t) (base + 3)] = std::nextafter(1.0f, 0.0f);

			Sampling s = abc ? sampling_abc() : sampling_semantic();
			s.temperature = temps[ti];
			s.top_k       = fixed_topks[ki];
			s.min_tokens  = 5;

			const std::vector<llama_token> history;
			std::mt19937_64                rng_ref(99);
			std::mt19937_64                rng_new(99);
			const llama_token ref = sample_step_ref(logits.data(), VOCAB_SIZE, s, history,
			                                        masked ? 4 : 5, abc, false, rng_ref);
			const llama_token got = sample_step(logits.data(), VOCAB_SIZE, s, history,
			                                    masked ? 4 : 5, abc, false, rng_new, &scratch);
			if (ref != got || !(rng_ref == rng_new))
			{
				mismatches++;
				fprintf(stderr, "MISMATCH fixed case: ref %d new %d — temp %.1f top_k %d "
				        "phase %s end %s\n", (int) ref, (int) got, s.temperature, s.top_k,
				        abc ? "abc" : "semantic", masked ? "masked" : "open");
			}
		}
		printf("sampler-diff: 48 fixed boundary cases, %lld mismatches\n", mismatches);
	}
	long long by_shape[SHAPE_N] = {0};
	int       shape = 0;

	for (long long c = 0; c < cases; c++)
	{
		if (c % block == 0)
		{
			shape = (int) ((c / block) % SHAPE_N);
			fill_block(rnd, shape, logits);
		} else {
			// Perturb: a handful of ordinary edits, then deliberate ties made by
			// copying one entry's value onto others — the top-k boundary case.
			for (int k = 0; k < 24; k++)
			{
				logits[(size_t) poke_id(rnd, (c & 1) != 0)] = rnd.normal(3.0f);
			}
			const bool  abc = (c & 1) != 0;
			const float tie = logits[(size_t) poke_id(rnd, abc)];
			for (int k = 0, n = 1 + rnd.below(200); k < n; k++)
			{
				logits[(size_t) poke_id(rnd, abc)] = tie;
			}
			// One ulp under the tie: distinct as logits, and for temperature > 1
			// often the same float after the division — the boundary collapse
			// SPEC_SAMPLER §2 warns about. Both scans must see the same value.
			const float under = std::nextafter(tie, -std::numeric_limits<float>::infinity());
			for (int k = 0, n = rnd.below(8); k < n; k++)
			{
				logits[(size_t) poke_id(rnd, abc)] = under;
			}
			// The end token is the one id both implementations special-case.
			logits[(size_t) (abc ? ABC_END : MUSIC_END)] =
				rnd.below(3) == 0 ? tie : rnd.normal(3.0f);
		}
		by_shape[shape]++;

		const bool shipped_abc = shipped && rnd.below(100) < 28;

		Sampling s;
		if (shipped)
		{
			s = shipped_abc ? sampling_abc() : sampling_semantic();
		}
		s.temperature        = shipped ? s.temperature : temps[rnd.below(4)];
		s.top_p              = shipped ? s.top_p : topps[rnd.below(5)];
		s.top_k              = shipped ? s.top_k : topks[rnd.below(6)];
		s.repetition_penalty = shipped ? s.repetition_penalty : pens[rnd.below(4)];
		s.penalty_window     = shipped ? s.penalty_window : windows[rnd.below(4)];
		s.min_tokens         = shipped ? 0 : rnd.below(4);
		s.max_tokens         = 9000;

		const bool phase_abc  = shipped ? shipped_abc : rnd.below(2) == 0;
		const bool legacy_off = rnd.below(2) == 0;

		// min_tokens boundary: step lands on min_tokens-1 / min_tokens / above,
		// so `end` is masked and unmasked about equally often.
		const int step = s.min_tokens - 1 + rnd.below(3);

		std::vector<llama_token> history;
		for (int k = 0, n = rnd.below(160); k < n; k++)
		{
			// Repeats (so freq > 1 and alpha != penalty), out-of-range ids (the
			// stage-5 code skipped them), and the end tokens.
			history.push_back(!history.empty() && rnd.below(3) == 0
			                  ? history[(size_t) rnd.below((int) history.size())]
			                  : (rnd.below(16) == 0 ? rnd.below(4 * VOCAB_SIZE) - VOCAB_SIZE
			                                        : poke_id(rnd, phase_abc)));
		}

		const uint64_t  rng_seed = rnd.next();
		std::mt19937_64 rng_ref(rng_seed);
		std::mt19937_64 rng_new(rng_seed);

		const llama_token ref = run_ref
			? sample_step_ref(logits.data(), VOCAB_SIZE, s, history, step,
			                  phase_abc, legacy_off, rng_ref)
			: 0;
		const llama_token got = run_new
			? sample_step(logits.data(), VOCAB_SIZE, s, history, step,
			              phase_abc, legacy_off, rng_new, &scratch)
			: 0;

		if ((run_ref && run_new) && (ref != got || !(rng_ref == rng_new)))
		{
			mismatches++;
			fprintf(stderr, "MISMATCH case %lld (%s): ref %d new %d%s — "
			        "temp %.3f top_p %.3f top_k %d pen %.4f window %d min_tokens %d "
			        "step %d phase %s legacy_off %d history %zu\n",
			        c, shape_name(shape), (int) ref, (int) got,
			        rng_ref == rng_new ? "" : " (rng state differs)",
			        s.temperature, s.top_p, s.top_k, s.repetition_penalty,
			        s.penalty_window, s.min_tokens, step, phase_abc ? "abc" : "semantic",
			        legacy_off ? 1 : 0, history.size());
			if (mismatches >= 20)
			{
				fprintf(stderr, "too many mismatches, stopping\n");
				break;
			}
		}
		if (verbose && c % 10000 == 0)
		{
			printf("  %lld / %lld cases\n", c, cases);
			fflush(stdout);
		}
	}

	printf("sampler-diff: %lld cases, seed %" PRIu64 ", block %d — ", cases, seed, block);
	for (int i = 0; i < SHAPE_N; i++)
	{
		printf("%s %lld%s", shape_name(i), by_shape[i], i + 1 < SHAPE_N ? ", " : "\n");
	}
	printf("sampler-diff: %.2f s wall for %s\n", now_seconds() - t0,
	       run_ref && run_new ? "both samplers" : run_ref ? "the stage-5 sampler only"
	                                                      : "the stage-6 sampler only");
	if (run_ref && run_new)
	{
		printf("sampler-diff: %lld mismatches\n", mismatches);
	}
	return mismatches == 0 ? 0 : 1;
}
