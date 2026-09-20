// yue2-guidance — SPEC_GUIDANCE.md §5.8: table tests for the weight curves, for
// the "needed at or after t" rule that decides when a branch is decoded, and for
// every request error of §2.3/§2.4 — plus SPEC_KEEP §5.7's file-free half, the
// "semantic_keep" rules that are decided without opening the file. None of it
// touches llama, a model or a device, so this runs anywhere.
//
// Like tests/bars.cpp it *includes* stage_ar.cpp rather than linking it: the
// functions under test live in that file's anonymous namespace, and including
// it guarantees the code under test is the code that ships.
//
//	build_guidance/yue2-guidance [-v]
//
// Exit 0 = every case matched.

#include "stage_ar.cpp"

namespace
{

int  failures = 0;
int  checked  = 0;
bool verbose  = false;

void check(bool ok, const char * what, const std::string & got, const std::string & want)
{
	checked++;
	if (ok)
	{
		if (verbose)
		{
			printf("ok   %-34s %s\n", what, got.c_str());
		}
		return;
	}
	failures++;
	printf("FAIL %-34s got %s, wanted %s\n", what, got.c_str(), want.c_str());
}

// ------------------------------------------------------------ the curves ---

std::vector<Keyframe> curve(const std::vector<std::pair<int, double>> & kf)
{
	std::vector<Keyframe> out;
	for (size_t i = 0; i < kf.size(); i++)
	{
		Keyframe k;
		k.offset = kf[i].first;
		k.weight = kf[i].second;
		out.push_back(k);
	}
	return out;
}

struct CurveCase
{
	std::vector<std::pair<int, double>> kf;
	int                                 off;
	double                              weight;
	const char *                        why;
};

const CurveCase CURVE_CASES[] =
{
	// --- one keyframe holds everywhere ------------------------------------
	{ {{0, 5}},                     0,    5,   "a single keyframe at 0" },
	{ {{0, 5}},                  9999,    5,   "and it holds forever" },
	{ {{40, 5}},                    0,    5,   "the first weight holds before the first offset" },

	// --- the README's presets ---------------------------------------------
	{ {{0, 5}},                    12,    5,   "the band changes at a section" },
	{ {{0, 11}, {60, 11}, {85, 3}}, 0,   11,   "a new lead voice: the plateau" },
	{ {{0, 11}, {60, 11}, {85, 3}},60,   11,   "still the plateau at its end" },
	{ {{0, 11}, {60, 11}, {85, 3}},85,    3,   "the ramp's end" },
	{ {{0, 11}, {60, 11}, {85, 3}},70,    7.8, "linear between 60 and 85" },
	{ {{0, 11}, {60, 11}, {85, 3}},999,   3,   "the last weight holds after the last" },
	{ {{0, 0}, {750, 5}},         375,    2.5, "gradual colouring, half way" },

	// --- steps and edges ---------------------------------------------------
	{ {{10, 1}, {10, 5}},           9,    1,   "before a step, the first weight" },
	{ {{10, 1}, {10, 5}},          10,    5,   "at a step, the later keyframe wins" },
	{ {{10, 1}, {10, 5}},          11,    5,   "after it, the later weight holds" },
	{ {{0, 1}, {10, 1}, {10, 5}},   5,    1,   "a step after a plateau" },
	{ {{0, 1}, {10, 1}, {10, 5}},  10,    5,   "and the step still wins at its offset" },
	{ {{0, 4}, {8, -4}},            4,    0,   "a curve may cross zero" },
	{ {{0, -2}},                    3,   -2,   "and be negative throughout" },
};

struct NeededCase
{
	std::vector<std::pair<int, double>> kf;
	int                                 off;
	bool                                needed;
	const char *                        why;
};

const NeededCase NEEDED_CASES[] =
{
	{ {{0, 5}},                     0,   true,  "a constant non-zero curve is always needed" },
	{ {{0, 5}},                 99999,   true,  "including past its last keyframe" },
	{ {{0, 0}},                     0,   false, "a constant zero curve is never needed" },
	{ {{0, 11}, {85, 0}},           0,   true,  "a ramp down to zero, at its start" },
	{ {{0, 11}, {85, 0}},          84,   true,  "one frame before it lands" },
	{ {{0, 11}, {85, 0}},          85,   false, "at zero, with nothing after it" },
	{ {{0, 11}, {85, 0}},          86,   false, "and after it" },
	{ {{0, 0}, {750, 5}},           0,   true,  "a ramp up from zero" },
	{ {{0, 0}, {750, 0}, {900, 4}}, 10,  true,  "a zero stretch with a push behind it" },
	{ {{0, 4}, {10, 0}, {20, 0}},  10,   false, "zero from here to the end" },
	{ {{0, 4}, {10, 0}, {20, 0}},   9,   true,  "but not one frame earlier" },
};

void run_curves()
{
	for (size_t i = 0; i < sizeof(CURVE_CASES) / sizeof(CURVE_CASES[0]); i++)
	{
		const CurveCase & c   = CURVE_CASES[i];
		const double      got = curve_at(curve(c.kf), c.off);
		check(std::fabs(got - c.weight) < 1e-9, "curve_at", strf("%g", got),
		      strf("%g (%s)", c.weight, c.why));
	}
	for (size_t i = 0; i < sizeof(NEEDED_CASES) / sizeof(NEEDED_CASES[0]); i++)
	{
		const NeededCase & c   = NEEDED_CASES[i];
		const bool         got = curve_needed_from(curve(c.kf), c.off);
		check(got == c.needed, "curve_needed_from", got ? "needed" : "not needed",
		      strf("%s (%s)", c.needed ? "needed" : "not needed", c.why));
	}
}

// ------------------------------------------------- the --guidance-trace math ---

// softmax / TV / argmax on hand-sized rows: the three pure functions behind a
// row of guidance_trace.npy, with the expectations worked out on paper.
void run_trace_math()
{
	std::vector<double> p;
	std::vector<double> q;

	// Two equal logits: half the mass each, and log(sum exp) = log 2.
	{
		const float         row[] = { 0.0f, 0.0f };
		const double        logz  = softmax_row(row, 2, p);
		check(std::fabs(p[0] - 0.5) < 1e-12 && std::fabs(p[1] - 0.5) < 1e-12 &&
		      std::fabs(logz - std::log(2.0)) < 1e-12,
		      "softmax_row", strf("%g %g, logZ %g", p[0], p[1], logz),
		      strf("0.5 0.5, logZ %g (two equal logits)", std::log(2.0)));
		// log p_0 = row[0] - logZ, which is what column 7 reports.
		check(std::fabs((row[0] - logz) + std::log(2.0)) < 1e-12, "softmax_row logp",
		      strf("%g", row[0] - logz), strf("%g (a token of probability 0.5)", -std::log(2.0)));
	}

	// The same two logits a thousand nats up: the maximum is subtracted, so
	// nothing overflows and the distribution is unchanged.
	{
		const float  row[] = { 1000.0f, 1000.0f };
		const double logz  = softmax_row(row, 2, q);
		check(std::fabs(q[0] - 0.5) < 1e-12 && std::fabs(logz - (1000.0 + std::log(2.0))) < 1e-9,
		      "softmax_row (offset)", strf("%g, logZ %g", q[0], logz),
		      strf("0.5, logZ %g (max-subtracted)", 1000.0 + std::log(2.0)));
		check(std::fabs(tv_distance(p, q)) < 1e-12, "tv_distance (same)",
		      strf("%g", tv_distance(p, q)), "0 (a constant shift is no distance at all)");
	}

	// exp(1) : 1 : 1 — the shares are e/(e+2), 1/(e+2), 1/(e+2).
	{
		const float  row[] = { 1.0f, 0.0f, 0.0f };
		const double logz  = softmax_row(row, 3, p);
		const double want  = std::exp(1.0) / (std::exp(1.0) + 2.0);
		check(std::fabs(p[0] - want) < 1e-12 &&
		      std::fabs(logz - (1.0 + std::log(1.0 + 2.0 * std::exp(-1.0)))) < 1e-12,
		      "softmax_row (skewed)", strf("%g, logZ %g", p[0], logz),
		      strf("%g, logZ %g", want, 1.0 + std::log(1.0 + 2.0 * std::exp(-1.0))));
	}

	// Two distributions sharing no mass are a full total variation apart, and
	// 0.5 vs 0.25 over two ids is a quarter.
	{
		const float one[]  = { 0.0f, -1000.0f };
		const float other[] = { -1000.0f, 0.0f };
		softmax_row(one, 2, p);
		softmax_row(other, 2, q);
		check(std::fabs(tv_distance(p, q) - 1.0) < 1e-9, "tv_distance (disjoint)",
		      strf("%g", tv_distance(p, q)), "1 (no shared mass)");

		const float even[]   = { 0.0f, 0.0f };
		const float lopsided[] = { 0.0f, (float) std::log(3.0) };
		softmax_row(even, 2, p);
		softmax_row(lopsided, 2, q);
		// 1e-6: log 3 is only a float here, which is the precision the real rows
		// come in at too.
		check(std::fabs(tv_distance(p, q) - 0.25) < 1e-6, "tv_distance (0.5 vs 0.25)",
		      strf("%g", tv_distance(p, q)), "0.25 (half of |0.5-0.25| twice)");
	}

	// argmax: the first of a tie, and a negative row has one too.
	{
		const float tie[]  = { 2.0f, 2.0f, 1.0f };
		const float down[] = { -9.0f, -3.0f, -7.0f };
		check(argmax_row(tie, 3) == 0 && argmax_row(down, 3) == 1, "argmax_row",
		      strf("%d %d", argmax_row(tie, 3), argmax_row(down, 3)),
		      "0 1 (the first of a tie; the least negative)");
	}
}

// ------------------------------------------------------- the request form ---

// The smallest request the parser accepts, with `extra` spliced in.
std::string request_with(const std::string & extra)
{
	return "{\"style\": \"slow waltz, upright bass\", \"lyrics\": \"[Verse]\\nrain on the tin\\n\""
	       + (extra.empty() ? std::string() : ", " + extra) + "}";
}

// "semantic_keep" belongs to one exact score, so every case below that is not
// about the missing score carries one (SPEC_KEEP §2).
#define KEEP_ABC "\"abc\": \"X:1\\nM:3/4\\nK:C\\n\""

struct RequestCase
{
	const char * extra;
	const char * err;     // a substring of the expected error, nullptr = must be accepted
	const char * why;
};

const RequestCase REQUEST_CASES[] =
{
	// --- accepted ----------------------------------------------------------
	{ "",                                                             nullptr,
	  "no guidance at all" },
	{ "\"cfg_scale\": 3",                                             nullptr,
	  "plain classifier-free guidance" },
	{ "\"cfg_scale\": 1.0, \"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"", nullptr,
	  "cfg_scale 1 is not guidance, so a template is fine" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}]", nullptr,
	  "a blank push from the start" },
	{ "\"guidance\": [{\"frame\": 3400, \"style\": \"brass band\", "
	  "\"against\": {\"previous\": [[0, 11], [60, 11], [85, 3]], \"blank\": [[0, 2]]}}]", nullptr,
	  "the worked example of §2.3" },
	{ "\"guidance\": [{\"frame\": 100, \"against\": {\"blank\": [[0, 0], [750, 5]]}}], "
	  "\"cfg_scale\": 1.0",                                           nullptr,
	  "an explicit cfg_scale of 1 beside a guidance block" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}, "
	  "{\"frame\": 1, \"style\": \"brass band\", \"against\": {\"previous\": [[0, 5]]}}]", nullptr,
	  "two entries, the second cutting to new tags" },

	// --- the exclusions of §2.4 --------------------------------------------
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], \"cfg_scale\": 3",
	  "not both",              "guidance and a cfg_scale that is not 1" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], "
	  "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "not supported yet",     "guidance with a score template" },
	{ "\"cfg_scale\": 3, \"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "not supported yet",     "cfg_scale with a score template" },

	// --- the entry rules of §2.3 -------------------------------------------
	{ "\"guidance\": [{\"frame\": 10, \"against\": {\"previous\": [[0, 5]]}}]",
	  "needs a \"style\"",     "a previous branch with no new prefix to leave behind" },
	{ "\"guidance\": [{\"frame\": 0, \"style\": \"brass band\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]",
	  "at frame 0 is the request's own", "nothing was in force before frame 0" },
	{ "\"guidance\": [{\"frame\": 0, \"style\": \"brass band\"}]",
	  "at frame 0 is the request's own", "a cut with nothing to cut away from" },
	{ "\"guidance\": [{\"frame\": 10, \"style\": \"slow waltz, upright bass\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]",
	  "already in force",      "the entry's style is the request's own" },
	{ "\"guidance\": [{\"frame\": 10, \"against\": {\"blank\": [[0, 2]]}}, "
	  "{\"frame\": 10, \"against\": {\"blank\": [[0, 3]]}}]",
	  "strictly increasing",   "two entries at one frame" },
	{ "\"guidance\": [{\"frame\": 20, \"against\": {\"blank\": [[0, 2]]}}, "
	  "{\"frame\": 10, \"against\": {\"blank\": [[0, 3]]}}]",
	  "strictly increasing",   "entries out of order" },

	// --- the curve rules of §2.3 -------------------------------------------
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": []}}]",
	  "non-empty list",        "an empty curve" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[10, 2], [5, 2]]}}]",
	  "must not decrease",     "offsets going backwards" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[-1, 2]]}}]",
	  "offset must be in",     "a negative offset" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 21]]}}]",
	  "weight must be finite", "a weight past the ceiling" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2, 3]]}}]",
	  "must be [offset, weight]", "a keyframe that is not a pair" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [0, 2]}}]",
	  "must be [offset, weight]", "a flat list instead of keyframes" },

	// --- shape and strictness ----------------------------------------------
	{ "\"guidance\": []",                    "non-empty list of entries", "an empty block" },
	{ "\"guidance\": {\"frame\": 0}",        "non-empty list of entries", "an object, not a list" },
	{ "\"guidance\": [{\"against\": {\"blank\": [[0, 2]]}}]",
	  "needs a \"frame\"",     "an entry with no frame" },
	{ "\"guidance\": [{\"frame\": -1, \"against\": {\"blank\": [[0, 2]]}}]",
	  "\"frame\" must be an integer", "a negative frame" },
	{ "\"guidance\": [{\"frame\": 0, \"weight\": 2}]",
	  "unknown key \"weight\"", "an unknown key inside an entry" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"silence\": [[0, 2]]}}]",
	  "unknown branch \"silence\"", "an unknown branch inside against" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {}}]",
	  "non-empty object",      "an empty against" },
	{ "\"guidance\": [{\"frame\": 0, \"style\": 7, \"against\": {\"blank\": [[0, 2]]}}]",
	  "\"style\" must be a string", "a style that is not text" },

	// --- "semantic_keep", SPEC_KEEP §2 -------------------------------------
	// Only the rules that need no file: the ones that do (the codes, the
	// length, the codec range) are checked where load_keep_codes reads it.
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}", nullptr,
	  "the worked example of SPEC_KEEP §2" },
	{ KEEP_ABC ", \"cfg_scale\": 3, "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}", nullptr,
	  "a plain cfg_scale: its blank branch is born at step N" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}, "
	  "\"guidance\": [{\"frame\": 100, \"style\": \"brass band\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]", nullptr,
	  "a cut at frame == N, the normal case" },
	{ "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  "needs the score its codes were sung to", "kept codes with no score to sing them to" },
	{ "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\", "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  "with \"abc_template\" is not supported", "kept codes and a written score" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}, "
	  "\"guidance\": [{\"frame\": 99, \"against\": {\"blank\": [[0, 2]]}}]",
	  "guidance frame 99 is inside the kept 100 frames", "an entry under the cut" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\"}",
	  "needs a \"frames\"",    "no implicit \"all of it\"" },
	{ KEEP_ABC ", \"semantic_keep\": {\"frames\": 100}",
	  "needs a \"file\"",      "frames with nothing to take them from" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 0}",
	  "\"frames\" must be an integer", "keeping nothing" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 1.5}",
	  "\"frames\" must be an integer", "a fractional frame count" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"\", \"frames\": 100}",
	  "\"file\" must be a path", "an empty path" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100, "
	  "\"from\": 0}",
	  "unknown key \"from\"",  "an unknown key inside semantic_keep" },
	{ KEEP_ABC ", \"semantic_keep\": \"r/semantic.npy\"",
	  "must be an object",     "the block given as a bare path" },
};

void run_requests()
{
	for (size_t i = 0; i < sizeof(REQUEST_CASES) / sizeof(REQUEST_CASES[0]); i++)
	{
		const RequestCase & c    = REQUEST_CASES[i];
		const std::string   what = strf("request(case %zu)", i + 1);
		Request             req;
		std::string         err;
		try
		{
			err = parse_request_json(json::parse(request_with(c.extra)), "R", req);
		}
		catch (const std::exception & e)
		{
			check(false, what.c_str(), e.what(), "valid JSON in the case itself");
			continue;
		}
		if (err.empty())
		{
			err = validate_request(req);
		}

		if (c.err == nullptr)
		{
			check(err.empty(), what.c_str(), err.empty() ? "accepted" : err, strf("accepted (%s)", c.why));
			continue;
		}
		check(err.find(c.err) != std::string::npos, what.c_str(),
		      err.empty() ? "accepted" : err, strf("an error containing \"%s\" (%s)", c.err, c.why));
	}
}

// ------------------------------------------------------------ the plan ------

struct PlanCase
{
	const char * extra;
	size_t       entries;
	double       weight0;   // the blank weight at frame 0, when there is one
	double       scalar;    // what request.json records as cfg_scale
	const char * why;
};

const PlanCase PLAN_CASES[] =
{
	{ "",                       0, 0,    1.0,  "no guidance: no branch, no plan" },
	{ "\"cfg_scale\": 1.0",     0, 0,    1.0,  "cfg_scale 1 is the unguided path" },
	{ "\"cfg_scale\": 3",       1, 2,    3.0,  "cfg_scale c is the blank branch at c - 1" },
	{ "\"cot\": \"off\"",       1, 0.01, 1.01, "cot=off defaults to 1.01, as the reference does" },
	{ "\"cot\": \"off\", \"cfg_scale\": 1.0", 0, 0, 1.0,
	  "and asking for 1.0 turns it off again" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}]", 1, 2, 1.0,
	  "a guidance block leaves cfg_scale at 1" },
};

void run_plans()
{
	for (size_t i = 0; i < sizeof(PLAN_CASES) / sizeof(PLAN_CASES[0]); i++)
	{
		const PlanCase &  c    = PLAN_CASES[i];
		const std::string what = strf("guidance_plan(case %zu)", i + 1);
		Request           req;
		std::string       err  = parse_request_json(json::parse(request_with(c.extra)), "R", req);
		if (err.empty())
		{
			err = validate_request(req);
		}
		if (!err.empty())
		{
			check(false, what.c_str(), err, "an accepted request");
			continue;
		}

		const std::vector<GuidanceEntry> plan   = guidance_plan(req);
		const double                     scalar = req.has_guidance ? 1.0 : cfg_scalar(req);
		if (plan.size() != c.entries)
		{
			check(false, what.c_str(), strf("%zu entries", plan.size()),
			      strf("%zu (%s)", c.entries, c.why));
			continue;
		}
		const double w = plan.empty() ? 0 : curve_at(plan[0].curve[BRANCH_BLANK], 0);
		check(std::fabs(w - c.weight0) < 1e-9 && std::fabs(scalar - c.scalar) < 1e-9,
		      what.c_str(), strf("weight %g, cfg_scale %g", w, scalar),
		      strf("weight %g, cfg_scale %g (%s)", c.weight0, c.scalar, c.why));
		check(is_guided(req) == (c.entries > 0), what.c_str(),
		      is_guided(req) ? "guided" : "unguided",
		      strf("%s (%s)", c.entries > 0 ? "guided" : "unguided", c.why));
	}
}

} // namespace

int main(int argc, char ** argv)
{
	for (int i = 1; i < argc; i++)
	{
		verbose = verbose || std::string(argv[i]) == "-v";
	}

	run_curves();
	run_trace_math();
	run_requests();
	run_plans();

	printf("%s: %d cases, %d failures\n", failures == 0 ? "PASS" : "FAIL", checked, failures);
	return failures == 0 ? 0 : 1;
}
