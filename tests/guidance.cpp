// yue2-guidance — SPEC_GUIDANCE.md §5.8: table tests for the weight curves, for
// the "needed at or after t" rule that decides when a branch is decoded, and for
// every request error of §2.3/§2.4 — plus SPEC_KEEP §5.7's file-free half, the
// "semantic_keep" rules that are decided without opening the file, and
// SPEC_NEGATIVE §2's "negative_style" and §7's "negative_lyrics" errors and
// prefix text, §8's "cfg_score" rules and the span its blend covers, and
// §9/§10's header watch, header check and "score_tempo" rules. None of it
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

// -------------------------------------------------------- the score clock ---
// SPEC_SECTIONS §6 found the planner's label vocabulary to be a closed set of
// bare `% name` lines, and §4 turns the bar a label sits on into a semantic
// frame. The scores below are that shape, cut down to what the clock reads.

// A header, a labelled section of `body` on the Vocal voice with an Ins line
// under it, and a second label. `% intro` is line 9, the body line 11, `% verse`
// line 14 — the numbers the cases below expect.
std::string score_of(const char * meter, const char * tempo, const char * body)
{
	return std::string("X:1\nT:\nM:") + meter + "\nL:1/16\nQ:" + tempo +
	       "\nV: Vocal clef=treble\nV: Ins clef=treble\nK:C\n"
	       "% intro\nV: Vocal\n" + body + "\nV: Ins\nZ9|\n"
	       "% verse\nV: Vocal\nC16|\n";
}

struct ClockCase
{
	std::string  score;
	const char * section;
	int          nth;
	int          lead;
	bool         found;
	int          line;
	int          bar;
	double       seconds;
	int          frame;
	const char * why;
};

void run_clock()
{
	const ClockCase CASES[] =
	{
		{ score_of("4/4", "1/4=120", "z16|z16|"), "verse", 1, 35, true, 14, 2, 4.0,   65,
		  "4/4 at Q=120 is 50 frames a bar" },
		{ score_of("4/4", "1/4=120", "z16|z16|"), "intro", 1, 35, true,  9, 0, 0.0,    1,
		  "the first label sits at bar 0, and the lead clamps to frame 1" },
		{ score_of("3/4", "1/4=120", "z12|z12|"), "verse", 1, 35, true, 14, 2, 3.0,   40,
		  "3/4 at Q=120 is 37.5 frames a bar" },
		{ score_of("6/8", "1/4=120", "z12|z12|"), "verse", 1, 35, true, 14, 2, 3.0,   40,
		  "6/8 counted in the quarters Q: names" },
		{ score_of("4/4", "1/4=90",  "z16|z16|"), "verse", 1, 35, true, 14, 2, 8.0 / 1.5, 98,
		  "Q=90 is 66.67 frames a bar, and the frame is rounded once" },
		{ score_of("4/4", "1/4=120", "z16|[M:3/4]z12|z12|"), "verse", 1, 35, true, 14, 3, 5.0, 90,
		  "an inline [M:3/4] changes the bars behind it" },
		{ score_of("4/4", "1/4=120", "z16|z16|"), "verse", 1, 250, true, 14, 2, 4.0,   1,
		  "a lead longer than the song so far clamps to frame 1" },
		{ score_of("4/4", "1/4=120", "z16|z16|"), "verse", 2, 35, false, 0, 0, 0.0,    0,
		  "a second verse the score never writes" },
		{ score_of("4/4", "1/4=120", "z16|z16|"), "outro", 1, 35, false, 0, 0, 0.0,    0,
		  "a label the score never writes" },
		{ score_of("4/4", "1/4=120", "Z4|"),      "verse", 1, 35, true, 14, 4, 8.0,  165,
		  "a multi-measure rest is four bars" },
		{ std::string("X:1\nM:4/4\nL:1/16\nQ:1/4=120\nK:C\n%%yue2-gen bars=4\n% verse\nz16|\n"),
		  "verse", 1, 0, true, 7, 0, 0.0, 1,
		  "%%yue2-gen is a directive, not a label" },
		{ std::string("X:1\nM:4/4\nL:1/16\nQ:1/4=120\nK:C\nz16|\n% verse\nz16|\n"),
		  "verse", 1, 0, true, 7, 1, 2.0, 50,
		  "a score with no V: line at all counts its body lines" },
	};

	for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++)
	{
		const ClockCase &         c = CASES[i];
		std::vector<SectionEntry> es(1);
		es[0].section = c.section;
		es[0].nth     = c.nth;
		es[0].lead    = c.lead;
		sections_locate(c.score, es);
		sections_frames(es);

		const bool ok = es[0].found == c.found &&
		                (!c.found || (es[0].line == c.line && es[0].bar == c.bar &&
		                              std::fabs(es[0].seconds - c.seconds) < 1e-9 &&
		                              es[0].frame == c.frame));
		check(ok, strf("clock(case %zu)", i + 1).c_str(),
		      es[0].found ? strf("line %d, bar %d, %.4f s, frame %d",
		                         es[0].line, es[0].bar, es[0].seconds, es[0].frame)
		                  : std::string("not found"),
		      c.found ? strf("line %d, bar %d, %.4f s, frame %d (%s)",
		                     c.line, c.bar, c.seconds, c.frame, c.why)
		              : strf("not found (%s)", c.why));
	}
}

// Several entries over one score: the nth counted over every label line of that
// name, the strictly-increasing clamp, and an entry the score orders the other
// way round going unfound rather than moving the song (§2, §4).
void run_sections_multi()
{
	// intro, verse 1, interlude, verse 2, chorus 1 — the shape §6's survey found,
	// with the planner's own interlude between the verses. One bar each, 4/4 at
	// Q=120, so every label sits two seconds after the one before it.
	const std::string score =
		"X:1\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal\nV: Ins\nK:C\n"
		"% intro\nV: Vocal\nz16|\nV: Ins\nZ1|\n"
		"% verse\nV: Vocal\nz16|\nV: Ins\nZ1|\n"
		"% interlude\nV: Vocal\nz16|\nV: Ins\nZ1|\n"
		"% verse\nV: Vocal\nz16|\nV: Ins\nZ1|\n"
		"% chorus\nV: Vocal\nz16|\n";

	{
		// The second verse is the *fourth* label: per-name counting steps over
		// the interlude the planner put between them.
		std::vector<SectionEntry> es(2);
		es[0].section = "verse";
		es[0].nth     = 2;
		es[0].lead    = 0;
		es[1].section = "chorus";
		es[1].lead    = 0;
		sections_locate(score, es);
		sections_frames(es);
		check(es[0].found && es[0].bar == 3 && es[0].frame == 150 &&
		      es[1].found && es[1].bar == 4 && es[1].frame == 200,
		      "sections(per-name nth)",
		      strf("verse 2 bar %d frame %d, chorus 1 bar %d frame %d",
		           es[0].bar, es[0].frame, es[1].bar, es[1].frame),
		      "verse 2 bar 3 frame 150, chorus 1 bar 4 frame 200");
	}
	{
		// Both leads are long enough to land on or before the entry ahead: the
		// frames are pushed apart rather than allowed to collide.
		std::vector<SectionEntry> es(2);
		es[0].section = "verse";
		es[0].nth     = 2;
		es[0].lead    = 150;
		es[1].section = "chorus";
		es[1].lead    = 200;
		sections_locate(score, es);
		sections_frames(es);
		check(es[0].frame == 1 && es[1].frame == 2, "sections(increasing clamp)",
		      strf("%d %d", es[0].frame, es[1].frame),
		      "1 2 (clamped to >= 1, then strictly increasing)");
	}
	{
		// The score plays the verse before the chorus, so an entry list that asks
		// for the chorus first leaves the verse unfound.
		std::vector<SectionEntry> es(2);
		es[0].section = "chorus";
		es[1].section = "verse";
		sections_locate(score, es);
		sections_frames(es);
		check(es[0].found && !es[1].found, "sections(order is the score's)",
		      strf("chorus %s, verse %s", es[0].found ? "found" : "not found",
		           es[1].found ? "found" : "not found"),
		      "chorus found, verse not found");
	}
	{
		// A plan is one guidance entry per section entry, compacted to the found
		// ones by sections_compile; the plain-swap test is what decides whether
		// the song needs a second KV stream at all (§4).
		std::vector<SectionEntry> es(1);
		es[0].section = "verse";
		es[0].style   = "brass band";
		const std::vector<GuidanceEntry> plan = sections_plan(es);
		check(plan.size() == 1 && plan[0].has_style && plan[0].style == "brass band" &&
		      !plan[0].has[BRANCH_PREVIOUS] && !plan[0].has[BRANCH_BLANK] &&
		      sections_plain_swap(es), "sections(plan of a plain swap)",
		      strf("%zu entries, plain swap %d", plan.size(), (int) sections_plain_swap(es)),
		      "1 entry carrying the style, plain swap 1");
		es[0].has[BRANCH_BLANK] = true;
		check(!sections_plain_swap(es), "sections(plan with an against)",
		      sections_plain_swap(es) ? "plain swap" : "needs a branch",
		      "needs a branch");
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

	// --- SPEC_SECTIONS §2 --------------------------------------------------
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}]",
	  nullptr,                 "the plain swap of §2" },
	{ "\"sections\": [{\"section\": \"chorus\", \"style\": \"brass band\", "
	  "\"lead_frames\": 0, \"against\": {\"previous\": [[0, 4], [250, 2]], "
	  "\"blank\": [[0, 2]]}}]",
	  nullptr,                 "the worked example of §2, with curves" },
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}, "
	  "{\"section\": \"verse\", \"nth\": 4, \"style\": \"marching band\"}]",
	  nullptr,                 "two entries on the same label, in order" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"cfg_scale\": 1.0",    nullptr,
	  "an explicit cfg_scale of 1 beside a sections block" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"guidance\": [{\"frame\": 10, \"against\": {\"blank\": [[0, 2]]}}]",
	  "not both",              "sections beside the frames it compiles into" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], \"cfg_scale\": 3",
	  "not in \"cfg_scale\"",  "sections beside a cfg_scale that is not 1" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "not supported yet",     "sections with a score template" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"cot\": \"off\"",
	  "cot=off has none",      "sections with no score phase to name labels in" },
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 4, \"style\": \"brass band\"}, "
	  "{\"section\": \"verse\", \"nth\": 2, \"style\": \"marching band\"}]",
	  "cannot come after",     "two entries on one label, out of order" },
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}, "
	  "{\"section\": \"verse\", \"nth\": 2, \"style\": \"marching band\"}]",
	  "cannot come after",     "two entries on the same label occurrence" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"slow waltz, upright bass\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]",
	  "already in force",      "the entry's style is the request's own" },
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 1, \"style\": \"brass band\"}, "
	  "{\"section\": \"chorus\", \"style\": \"brass band\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]",
	  "already in force",      "the second entry repeats the first entry's tags" },
	{ "\"sections\": [{\"section\": \"verse\"}]",
	  "needs a \"style\"",     "an entry with no tags to change to" },
	{ "\"sections\": [{\"style\": \"brass band\"}]",
	  "needs a \"section\"",   "an entry that names no label" },
	{ "\"sections\": [{\"section\": \"\", \"style\": \"brass band\"}]",
	  "must be a label name",  "an empty label" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"\"}]",
	  "must be the tag string","empty tags" },
	{ "\"sections\": [{\"section\": \"verse\", \"nth\": 0, \"style\": \"brass band\"}]",
	  "must be an integer in [1", "nth below 1" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\", "
	  "\"lead_frames\": 251}]",
	  "must be an integer in [0", "a lead past the 250-frame ceiling" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\", "
	  "\"lead_frames\": -1}]",
	  "must be an integer in [0", "a negative lead" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\", \"frame\": 100}]",
	  "unknown key \"frame\"", "a guidance key inside a sections entry" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\", "
	  "\"against\": {\"bank\": [[0, 2]]}}]",
	  "unknown branch",        "a misspelt branch kind" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\", \"against\": {}}]",
	  "non-empty object",      "an empty against" },
	{ "\"sections\": []",
	  "non-empty list",        "an empty sections block" },
	{ "\"sections\": {\"section\": \"verse\", \"style\": \"brass band\"}",
	  "non-empty list",        "the block given as one entry" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  nullptr,                 "sections beside a semantic_keep (the frames are the score's)" },

	// --- "negative_style", SPEC_NEGATIVE §2 --------------------------------
	{ "\"cfg_scale\": 3, \"negative_style\": \"children's song\"", nullptr,
	  "the worked example of SPEC_NEGATIVE §2" },
	{ "\"cot\": \"off\", \"negative_style\": \"children's song\"", nullptr,
	  "cot=off's 1.01 default counts as a cfg_scale" },
	{ "\"cfg_scale\": 0.5, \"negative_style\": \"children's song\"", nullptr,
	  "a cfg_scale below 1 carries a weight too (pulled towards the style)" },
	{ "\"negative_style\": null",                                     nullptr,
	  "null is today's blank branch" },
	{ KEEP_ABC ", \"cfg_scale\": 3, \"negative_style\": \"children's song\", "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}", nullptr,
	  "a negative branch born at step N" },
	{ "\"negative_style\": \"children's song\"",
	  "does nothing at cfg_scale 1", "no cfg_scale: the branch would weigh 0" },
	{ "\"cfg_scale\": 1.0, \"negative_style\": \"children's song\"",
	  "does nothing at cfg_scale 1", "an explicit cfg_scale of 1" },
	{ "\"cot\": \"off\", \"cfg_scale\": 1.0, \"negative_style\": \"children's song\"",
	  "does nothing at cfg_scale 1", "cot=off with its 1.01 turned off again" },
	{ "\"cfg_scale\": 3, \"negative_style\": \"\"",
	  "must be non-empty text", "an empty negative style" },
	{ "\"cfg_scale\": 3, \"negative_style\": \" \\n\\t \"",
	  "must be non-empty text", "whitespace only" },
	{ "\"cfg_scale\": 3, \"negative_style\": 7",
	  "\"negative_style\" must be a string", "a negative style that is not text" },
	{ "\"cfg_scale\": 3, \"negative_style\": [\"children's song\"]",
	  "\"negative_style\" must be a string", "a list of tags instead of the text" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], "
	  "\"negative_style\": \"children's song\"",
	  "\"negative_style\" with \"guidance\" is not supported", "beside a guidance block" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"negative_style\": \"children's song\"",
	  "\"negative_style\" with \"sections\" is not supported", "beside a sections block" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}], "
	  "\"negative_style\": \"children's song\"",
	  "\"negative_style\" with \"handover\" is not supported", "beside a handover" },
	{ "\"cfg_scale\": 3, \"negative_style\": \"children's song\", "
	  "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "not supported yet",     "a template is already out with cfg_scale" },

	// --- "negative_lyrics", SPEC_NEGATIVE §7.2 -----------------------------
	{ "\"cfg_scale\": 3, \"negative_style\": \"children's song\", \"negative_lyrics\": true",
	  nullptr,                 "the first example of SPEC_NEGATIVE §7.2" },
	{ "\"cfg_scale\": 3, \"negative_lyrics\": true", nullptr,
	  "no negative style: empty tags, the same lyrics" },
	{ "\"cot\": \"off\", \"negative_lyrics\": true", nullptr,
	  "cot=off's 1.01 default counts here too" },
	{ KEEP_ABC ", \"cfg_scale\": 3, \"negative_lyrics\": true, "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}", nullptr,
	  "a lyric-carrying negative branch born at step N" },
	{ "\"negative_lyrics\": false",                                   nullptr,
	  "false is stage 11 exactly, even with no cfg_scale" },
	{ "\"negative_lyrics\": null",                                    nullptr,
	  "null is absent" },
	{ "\"cfg_scale\": 3, \"negative_style\": \"children's song\", \"negative_lyrics\": false",
	  nullptr,                 "false beside a negative style is the stage-11 branch" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], "
	  "\"negative_lyrics\": false",
	  nullptr,                 "false beside a guidance block changes nothing" },
	{ "\"negative_lyrics\": true",
	  "\"negative_lyrics\" does nothing at cfg_scale 1", "no cfg_scale: the branch would weigh 0" },
	{ "\"cfg_scale\": 1.0, \"negative_lyrics\": true",
	  "\"negative_lyrics\" does nothing at cfg_scale 1", "an explicit cfg_scale of 1" },
	{ "\"cot\": \"off\", \"cfg_scale\": 1.0, \"negative_lyrics\": true",
	  "\"negative_lyrics\" does nothing at cfg_scale 1", "cot=off with its 1.01 turned off again" },
	{ "\"cfg_scale\": 3, \"negative_lyrics\": \"true\"",
	  "\"negative_lyrics\" must be true, false or null", "a string that reads true" },
	{ "\"cfg_scale\": 3, \"negative_lyrics\": 1",
	  "\"negative_lyrics\" must be true, false or null", "a number for a flag" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], "
	  "\"negative_lyrics\": true",
	  "\"negative_lyrics\" with \"guidance\" is not supported", "beside a guidance block" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"negative_lyrics\": true",
	  "\"negative_lyrics\" with \"sections\" is not supported", "beside a sections block" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}], "
	  "\"negative_lyrics\": true",
	  "\"negative_lyrics\" with \"handover\" is not supported", "beside a handover" },

	// --- "cfg_score", SPEC_NEGATIVE §8.2 -----------------------------------
	{ "\"cfg_score\": 2, \"negative_style\": \"children's song\", \"negative_lyrics\": true, "
	  "\"cfg_scale\": 3",       nullptr, "the worked example of §8.2" },
	{ "\"cfg_score\": 2, \"negative_style\": \"children's song\", \"negative_lyrics\": true",
	  nullptr,                 "cfg_scale 1: the score is guided, the semantic phase is not" },
	{ "\"cfg_score\": 1.5, \"negative_lyrics\": true", nullptr,
	  "no negative style: empty tags, the same lyrics, cfg_scale 1" },
	{ "\"cfg_score\": 0.5, \"negative_lyrics\": true", nullptr,
	  "below 1 pulls the score towards the negative" },
	{ "\"cfg_score\": 20, \"negative_lyrics\": true", nullptr,  "the ceiling itself" },
	{ "\"cfg_score\": 1",                                          nullptr,
	  "1 is the unguided score phase, with no negative needed" },
	{ "\"cfg_score\": 1.0, \"abc\": \"X:1\\nK:C\\n\"",           nullptr,
	  "1 beside a given score changes nothing" },
	{ "\"cfg_score\": null, \"negative_style\": \"children's song\"",
	  "does nothing at cfg_scale 1 and cfg_score 1", "null is absent, so the negative weighs 0" },
	{ "\"cfg_score\": \"2\", \"negative_lyrics\": true",
	  "\"cfg_score\" must be a number", "a string for a scale" },
	{ "\"cfg_score\": [2], \"negative_lyrics\": true",
	  "\"cfg_score\" must be a number", "a list for a scale" },
	{ "\"cfg_score\": 0, \"negative_lyrics\": true",
	  "cfg_score must be finite and in (0, 20]", "zero" },
	{ "\"cfg_score\": -1, \"negative_lyrics\": true",
	  "cfg_score must be finite and in (0, 20]", "a negative scale" },
	{ "\"cfg_score\": 20.5, \"negative_lyrics\": true",
	  "cfg_score must be finite and in (0, 20]", "past the ceiling" },
	{ "\"cfg_score\": 2",
	  "\"cfg_score\" needs \"negative_lyrics\": true", "no negative at all" },
	{ "\"cfg_score\": 2, \"cfg_scale\": 3, \"negative_style\": \"children's song\"",
	  "\"cfg_score\" needs \"negative_lyrics\": true", "a negative without the lyrics" },
	{ "\"cfg_score\": 2, \"negative_style\": \"children's song\", \"negative_lyrics\": false, "
	  "\"cfg_scale\": 3",
	  "\"cfg_score\" needs \"negative_lyrics\": true", "an explicit false" },
	{ "\"cfg_score\": 2, \"negative_lyrics\": true, \"abc\": \"X:1\\nK:C\\n\"",
	  "\"cfg_score\" with \"abc\"", "a given score: no score phase" },
	{ "\"cfg_score\": 2, \"negative_lyrics\": true, "
	  "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "\"cfg_score\" with \"abc_template\" is not supported", "a score template" },
	{ "\"cfg_score\": 2, \"negative_lyrics\": true, \"cot\": \"off\"",
	  "\"cfg_score\" guides the score phase, and cot=off has none", "cot=off" },
	{ KEEP_ABC ", \"cfg_score\": 2, \"negative_lyrics\": true, "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  "\"cfg_score\" with \"semantic_keep\"", "kept codes sing a given score" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}], "
	  "\"cfg_score\": 2, \"negative_lyrics\": true",
	  "\"cfg_score\" with \"guidance\" is not supported", "beside a guidance block" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"brass band\"}], "
	  "\"cfg_score\": 2, \"negative_lyrics\": true",
	  "\"cfg_score\" with \"sections\" is not supported", "beside a sections block" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}], "
	  "\"cfg_score\": 2, \"negative_lyrics\": true",
	  "\"cfg_score\" with \"handover\" is not supported", "beside a handover" },

	// --- "score_tempo", SPEC_NEGATIVE §10.2 ---------------------------------
	{ "\"score_tempo\": 90",                    nullptr, "the worked example of §10.2" },
	{ "\"score_tempo\": 92.5",                  nullptr, "a tempo need not be whole" },
	{ "\"score_tempo\": 20",                    nullptr, "the floor itself" },
	{ "\"score_tempo\": 300",                   nullptr, "the ceiling itself" },
	{ "\"score_tempo\": null",                  nullptr, "null is absent" },
	{ "\"score_tempo\": 140, \"cfg_score\": 2, \"negative_lyrics\": true", nullptr,
	  "with a guided score phase" },
	{ "\"score_tempo\": 60, \"cot\": \"melody\"", nullptr, "cot=melody writes a header too" },
	{ "\"score_tempo\": \"90\"",
	  "\"score_tempo\" must be a number or null", "a string for a tempo" },
	{ "\"score_tempo\": true",
	  "\"score_tempo\" must be a number or null", "a flag for a tempo" },
	{ "\"score_tempo\": 0",
	  "score_tempo must be in [20, 300] quarter notes per minute", "zero is not absent" },
	{ "\"score_tempo\": 19.9",
	  "score_tempo must be in [20, 300] quarter notes per minute", "under the floor" },
	{ "\"score_tempo\": 301",
	  "score_tempo must be in [20, 300] quarter notes per minute", "past the ceiling" },
	{ "\"score_tempo\": 90, \"abc\": \"X:1\\nK:C\\n\"",
	  "\"score_tempo\" with \"abc\"", "a given score has its own Q:" },
	{ "\"score_tempo\": 90, "
	  "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"",
	  "\"score_tempo\" with \"abc_template\"", "a template gives the header" },
	{ "\"score_tempo\": 90, \"cot\": \"off\"",
	  "\"score_tempo\" forces the score's Q: line, and cot=off writes no score", "cot=off" },
	{ KEEP_ABC ", \"score_tempo\": 90, "
	  "\"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  "\"score_tempo\" with \"semantic_keep\"", "kept codes sing a given score" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 2, \"style\": \"brass band\"}], "
	  "\"score_tempo\": 90",
	  "\"score_tempo\" with \"handover\"", "a handover reuses its base take's score" },
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

// ------------------------------------------------------- --draft's guard ---
// SPEC_DRAFT §1: every request and flag a draft head cannot speculate for is an
// error, and the ones it can are not.

const RequestCase DRAFT_CASES[] =
{
	{ "",                                              nullptr, "an unguided song" },
	{ "\"cfg_scale\": 1.0",                            nullptr, "cfg_scale 1 is unguided" },
	{ "\"cot\": \"off\", \"cfg_scale\": 1.0",          nullptr, "cot=off with cfg_scale 1" },
	{ "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\"", nullptr, "a score template" },
	{ "\"sections\": [{\"section\": \"Chorus\", \"style\": \"brass band\"}]", nullptr,
	  "a plain-swap sections block" },
	{ "\"cfg_scale\": 3",                              "does not support guidance", "cfg_scale" },
	{ "\"cot\": \"off\"",                              "does not support guidance",
	  "cot=off alone defaults to cfg_scale 1.01" },
	{ "\"guidance\": [{\"frame\": 0, \"against\": {\"blank\": [[0, 2]]}}]",
	  "does not support guidance", "a guidance block" },
	{ "\"cfg_score\": 1.5, \"negative_lyrics\": true", "does not support guidance", "cfg_score" },
	{ "\"sections\": [{\"section\": \"Chorus\", \"style\": \"brass band\", "
	  "\"against\": {\"previous\": [[0, 5]]}}]",       "does not support guidance",
	  "sections with an against" },
	{ KEEP_ABC ", \"semantic_keep\": {\"file\": \"r/semantic.npy\", \"frames\": 100}",
	  "does not support \"semantic_keep\"", "semantic_keep" },
	{ KEEP_ABC ", \"handover\": [{\"section\": \"chorus\"}]",
	  "does not support \"handover\"", "handover" },
};

struct DraftFlagCase
{
	bool         greedy;
	bool         verify;
	bool         prefix_only;
	int          parallel;
	const char * err;
};

const DraftFlagCase DRAFT_FLAG_CASES[] =
{
	{ false, false, false, 1, nullptr },
	{ true,  false, false, 1, "--greedy" },
	{ false, true,  false, 1, "--verify-sampler" },
	{ false, false, true,  1, "--prefix-only" },
	{ false, false, false, 2, "--parallel 1" },
};

void run_draft_guard()
{
	for (size_t i = 0; i < sizeof(DRAFT_CASES) / sizeof(DRAFT_CASES[0]); i++)
	{
		const RequestCase & c    = DRAFT_CASES[i];
		const std::string   what = strf("draft request(case %zu)", i + 1);
		Request             req;
		std::string         err = parse_request_json(json::parse(request_with(c.extra)), "R", req);
		if (err.empty())
		{
			err = validate_request(req);
		}
		if (!err.empty())
		{
			check(false, what.c_str(), err, strf("a valid request (%s)", c.why));
			continue;
		}
		err = draft_request_error(req);
		if (c.err == nullptr)
		{
			check(err.empty(), what.c_str(), err.empty() ? "accepted" : err, strf("accepted (%s)", c.why));
			continue;
		}
		check(err.find(c.err) != std::string::npos, what.c_str(),
		      err.empty() ? "accepted" : err, strf("an error containing \"%s\" (%s)", c.err, c.why));
	}
	for (size_t i = 0; i < sizeof(DRAFT_FLAG_CASES) / sizeof(DRAFT_FLAG_CASES[0]); i++)
	{
		const DraftFlagCase & c    = DRAFT_FLAG_CASES[i];
		const std::string     what = strf("draft flags(case %zu)", i + 1);
		const std::string     err  = draft_params_error(c.greedy, c.verify, c.prefix_only, c.parallel);
		if (c.err == nullptr)
		{
			check(err.empty(), what.c_str(), err.empty() ? "accepted" : err, "accepted");
			continue;
		}
		check(err.find(c.err) != std::string::npos, what.c_str(),
		      err.empty() ? "accepted" : err, strf("an error containing \"%s\"", c.err));
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
	{ "\"cfg_scale\": 3, \"negative_style\": \"children's song\"", 1, 2, 3.0,
	  "a negative style changes the branch's prefix, not its plan" },
	{ "\"cot\": \"off\", \"negative_style\": \"children's song\"", 1, 0.01, 1.01,
	  "and rides cot=off's 1.01 the same way" },
	{ "\"cfg_scale\": 3, \"negative_lyrics\": true", 1, 2, 3.0,
	  "negative_lyrics changes the branch's prefix, not its plan" },
	{ "\"negative_lyrics\": false", 0, 0, 1.0,
	  "and false alone asks for no branch at all" },
	{ "\"cfg_score\": 2, \"negative_lyrics\": true, \"cfg_scale\": 3", 1, 2, 3.0,
	  "cfg_score leaves the semantic plan as cfg_scale's" },
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

// ------------------------------------------------- the negative prefix text ---

// SPEC_NEGATIVE §3: the negative branch is the positive recipe with the tags
// swapped and the lyrics empty; everything else (cot, score) is the request's.
// The tokens need the vocab, so the model-side half is in STATUS_NEGATIVE.md.
// An accepted request built from `extra`, or false with the failure counted.
bool accepted_request(const char * what, const std::string & extra, Request & req)
{
	std::string err = parse_request_json(json::parse(request_with(extra)), "R", req);
	if (err.empty())
	{
		err = validate_request(req);
	}
	if (!err.empty())
	{
		check(false, what, err, "an accepted request");
		return false;
	}
	return true;
}

void run_negative()
{
	Request req;
	if (accepted_request("negative_request", "\"cfg_scale\": 3, \"cot\": \"melody\", "
	                     "\"negative_style\": \"children's song\"", req))
	{
		const Request     neg  = negative_request(req);
		const std::string want = std::string(instruction("melody")) +
		                         "\n[Tags]\nchildren's song\n[Lyrics]\n\n";
		check(neg.text() == want, "negative_request text", neg.text(), want);
		check(neg.cot == req.cot && neg.cfg_scale == req.cfg_scale && req.lyrics != "" &&
		      req.style == "slow waltz, upright bass", "negative_request copy",
		      "cot " + neg.cot + ", style " + req.style, "the request itself untouched");
		check(negative_label(req) == "negative style \"children's song\"", "negative_label(style)",
		      negative_label(req), "the stage-11 wording");
	}

	// SPEC_NEGATIVE §7.2: "negative_lyrics" keeps the song's lyrics, so the text
	// is the positive one with the tags swapped...
	Request lyr;
	if (accepted_request("negative_request(lyrics)", "\"cfg_scale\": 3, \"cot\": \"melody\", "
	                     "\"negative_style\": \"children's song\", \"negative_lyrics\": true", lyr))
	{
		const std::string want = std::string(instruction("melody")) +
		                         "\n[Tags]\nchildren's song\n[Lyrics]\n[Verse]\nrain on the tin\n\n";
		check(negative_request(lyr).text() == want, "negative_request text(lyrics)",
		      negative_request(lyr).text(), want);
		check(lyr.replaces_blank(), "replaces_blank(lyrics)", "blank", "a negative branch");
		check(negative_label(lyr) == "negative style \"children's song\" with the song's lyrics",
		      "negative_label(lyrics)", negative_label(lyr), "the style, with the lyrics");
	}

	// ... with no negative style, empty tags — still the whole text() recipe,
	// not the blank's bare instruction ...
	Request tags;
	if (accepted_request("negative_request(no style)", "\"cfg_scale\": 3, \"negative_lyrics\": true", tags))
	{
		const std::string want = std::string(instruction("full")) +
		                         "\n[Tags]\n\n[Lyrics]\n[Verse]\nrain on the tin\n\n";
		check(negative_request(tags).text() == want, "negative_request text(no style)",
		      negative_request(tags).text(), want);
		check(negative_label(tags) == "an empty style with the song's lyrics",
		      "negative_label(no style)", negative_label(tags), "an empty style");
	}

	// ... and with the song's own style as the negative, it is the song itself
	// (SPEC_NEGATIVE §7.4 item 2's model-side check rests on this).
	Request self;
	if (accepted_request("negative_request(self)", "\"cfg_scale\": 3, \"negative_style\": "
	                     "\"slow waltz, upright bass\", \"negative_lyrics\": true", self))
	{
		check(negative_request(self).text() == self.text(), "negative_request text(self)",
		      negative_request(self).text(), self.text());
	}

	// SPEC_NEGATIVE §8.2: at cfg_scale 1 a cfg_score song has no semantic plan,
	// yet it is guided — the score phase has a branch — so it runs at
	// --parallel 1 and the context is sized for the branch.
	Request score;
	if (accepted_request("cfg_score(cfg_scale 1)", "\"cfg_score\": 2, \"negative_lyrics\": true", score))
	{
		check(score_guided(score) && is_guided(score) && guidance_plan(score).empty(),
		      "cfg_score(cfg_scale 1)", strf("score %d, guided %d, plan %zu",
		      (int) score_guided(score), (int) is_guided(score), guidance_plan(score).size()),
		      "score 1, guided 1, plan 0");
	}
	Request unscored;
	if (accepted_request("cfg_score(1)", "\"cfg_score\": 1", unscored))
	{
		check(!score_guided(unscored) && !is_guided(unscored), "cfg_score(1)",
		      is_guided(unscored) ? "guided" : "unguided", "unguided");
	}

	// false, or absent, is stage 11: no branch of its own without a style.
	Request off;
	if (accepted_request("negative_lyrics(false)", "\"cfg_scale\": 3, \"negative_lyrics\": false", off))
	{
		check(!off.replaces_blank(), "replaces_blank(false)",
		      off.replaces_blank() ? "a negative branch" : "blank", "blank");
	}
}

// ----------------------------------------------------------- the row spans ---

// SPEC_NEGATIVE §8.2: a branch row carries exactly the ids the phase's sampler
// visits, packed. The abc span must be sample_step's abc segments ([0, EOD) and
// ABC_END); the semantic one the stage-7 run [MUSIC_END, codec end), in order,
// since the guided semantic path is byte-identical only if it is.
void run_spans()
{
	check(SPAN_ABC.total() == EOD + 1 && SPAN_ABC.id(0) == 0 && SPAN_ABC.id(EOD - 1) == EOD - 1 &&
	      SPAN_ABC.id(EOD) == ABC_END, "span(abc)",
	      strf("%d ids, last %d", SPAN_ABC.total(), SPAN_ABC.id(SPAN_ABC.total() - 1)),
	      "[0, EOD) then ABC_END");
	bool contiguous = SPAN_SEM.total() == CODEC_OFFSET + CODEC_SIZE - MUSIC_END;
	for (int i = 0; contiguous && i < SPAN_SEM.total(); i++)
	{
		contiguous = SPAN_SEM.id(i) == MUSIC_END + i && SPAN_SEM.index(MUSIC_END + i) == i;
	}
	check(contiguous, "span(semantic)", contiguous ? "contiguous" : "not the stage-7 run",
	      "MUSIC_END + i at every i");

	const int outside[] = { EOD, ABC_START, ABC_END + 1, MUSIC_START, -1, VOCAB_SIZE };
	bool      none      = true;
	for (size_t i = 0; i < sizeof(outside) / sizeof(outside[0]); i++)
	{
		none = none && SPAN_ABC.index(outside[i]) == -1;
	}
	check(none && SPAN_ABC.index(ABC_END) == EOD && SPAN_ABC.index(0) == 0 &&
	      SPAN_SEM.index(ABC_END) == -1, "span.index", none ? "outside ids are -1" : "an outside id mapped",
	      "-1 off the span, ABC_END packed at EOD");

	// pack_row: the two ranges end to end.
	std::vector<float> full((size_t) VOCAB_SIZE);
	for (size_t i = 0; i < full.size(); i++)
	{
		full[i] = (float) i;
	}
	std::vector<float> packed((size_t) SPAN_ABC.total());
	pack_row(full.data(), SPAN_ABC, packed.data());
	check(packed[0] == 0 && packed[(size_t) EOD - 1] == (float) (EOD - 1) &&
	      packed[(size_t) EOD] == (float) ABC_END, "pack_row(abc)",
	      strf("%g %g %g", packed[0], packed[(size_t) EOD - 1], packed[(size_t) EOD]),
	      strf("0 %d %d", EOD - 1, ABC_END));
}

// ------------------------------------------------------------ the header ---

// Every way of cutting `text` into pieces: `cuts` is a bit mask over the
// len - 1 places between two characters.
std::vector<std::string> split_by(const std::string & text, unsigned cuts)
{
	std::vector<std::string> out(1);
	for (size_t i = 0; i < text.size(); i++)
	{
		out.back() += text[i];
		if (i + 1 < text.size() && (cuts >> i) & 1)
		{
			out.emplace_back();
		}
	}
	return out;
}

// Feeds `head` whole and unforced, then `tail` in pieces, as tokens would
// arrive. Returns the offset into head + tail where the watch closed the header
// or cut, or npos; `piece` is set to the piece that did it.
size_t watch_at(HeaderWatch & w, const std::string & head, const std::vector<std::string> & tail,
	bool force, size_t & piece)
{
	size_t at = header_scan(w, head, false);
	if (w.cut != CUT_NONE || w.state != HEADER_OPEN)
	{
		piece = 0;
		return at;
	}
	size_t base = head.size();
	for (size_t i = 0; i < tail.size(); i++)
	{
		at = header_scan(w, tail[i], force);
		if (w.cut != CUT_NONE || w.state != HEADER_OPEN)
		{
			piece = i + 1;
			return base + at;
		}
		base += tail[i].size();
	}
	return std::string::npos;
}

// SPEC_NEGATIVE §9.2 / §10.2: the watch decides on text, never on how the text
// was cut into tokens. §9.4 item 4 and §10.3 item 5.
void run_header()
{
	const std::string head = "X:1\nT:\nM:6/8\nL:1/32\nQ:1/4=78\nV: Vocal clef=treble\n"
	                         "V: Ins clef=treble";
	const std::string key  = "\nK:Eb\n";

	// The header closes on the newline that ends the K: line, whatever piece
	// carries it, with or without a tempo to force (the Q: line is behind it).
	for (int force = 0; force < 2; force++)
	{
		int bad = 0;
		for (unsigned cuts = 0; cuts < (1u << (key.size() - 1)); cuts++)
		{
			const std::vector<std::string> tail = split_by(key, cuts);
			HeaderWatch w;
			size_t      piece = 0;
			const size_t at   = watch_at(w, head, tail, force != 0, piece);
			bad += at == head.size() + key.size() && piece == tail.size() && w.state == HEADER_KEY &&
			       w.cut == CUT_NONE ? 0 : 1;
		}
		check(bad == 0, force ? "header(K: splits, forced)" : "header(K: splits)",
		      strf("%d of %u splits wrong", bad, 1u << (key.size() - 1)), "every split closes on the last newline");
	}

	// The Q: line of a score_tempo job is cut right after its `Q:`, in the piece
	// that completes it; nothing is cut or closed before.
	{
		const std::string pre  = "X:1\nT:\nM:6/8\nL:1/32";
		const std::string text = "\nQ:1/4=78\n";
		int               bad  = 0;
		for (unsigned cuts = 0; cuts < (1u << (text.size() - 1)); cuts++)
		{
			const std::vector<std::string> tail = split_by(text, cuts);
			HeaderWatch w;
			size_t      piece = 0;
			const size_t at   = watch_at(w, pre, tail, true, piece);
			bad += at == pre.size() + 3 && w.cut == CUT_TEMPO && w.tempo && w.state == HEADER_OPEN ? 0 : 1;
		}
		check(bad == 0, "header(Q: splits)", strf("%d of %u splits wrong", bad,
		      1u << (text.size() - 1)), "every split cuts after \"Q:\"");
	}

	// No Q: line: the cut goes in front of the K: line, or of the line that
	// closes the header without one; unforced, the same text cuts nothing.
	struct InsertCase
	{
		const char * text;
		size_t       at;
		HeaderState  unforced;
		const char * why;
	};
	const InsertCase inserts[] = {
		{ "X:1\nM:6/8\nL:1/32\nV: Vocal\nK:Eb\n",  26, HEADER_KEY,  "before K:" },
		{ "X:1\nM:6/8\nL:1/32\n% intro\n",          17, HEADER_BODY, "before a % line" },
		{ "X:1\nM:6/8\n|z4|\n",                      10, HEADER_BODY, "before a bar line" },
		{ "X:1\nM:6/8\nabc\nV: Vocal\n",            14, HEADER_BODY, "before V: after a loose line" },
	};
	for (size_t i = 0; i < sizeof(inserts) / sizeof(inserts[0]); i++)
	{
		const InsertCase & c = inserts[i];
		HeaderWatch        f;
		const size_t       at = header_scan(f, c.text, true);
		HeaderWatch        u;
		header_scan(u, c.text, false);
		check(at == c.at && f.cut == CUT_INSERT && u.cut == CUT_NONE && u.state == c.unforced,
		      strf("header(insert %zu)", i + 1).c_str(),
		      strf("at %zu, cut %d, unforced %d", at, (int) f.cut, (int) u.state),
		      strf("at %zu, a CUT_INSERT, unforced %d (%s)", c.at, (int) c.unforced, c.why));
	}

	// The §9.2 fallback, unforced: where guidance starts without a K: line.
	struct BodyCase
	{
		const char * text;
		size_t       at;          // npos = the header is still open
		HeaderState  state;
		const char * why;
	};
	const BodyCase bodies[] = {
		{ "X:1\nT:\n% intro\nV: Vocal\n",               8,  HEADER_BODY, "a comment line" },
		{ "X:1\nT:\nz4|z4|\n",                           10, HEADER_BODY, "a bar line mid-line" },
		{ "X:1\nT:\nfoo\nV: Vocal\n",                   12, HEADER_BODY, "V: after a loose line" },
		{ "X:1\nT:\nV: Vocal clef=treble\nV: Vocal\n",  std::string::npos, HEADER_OPEN,
		  "V: lines alone are the header's" },
		{ "X:1\nT:\nM/8\nL:1/32\n",                     std::string::npos, HEADER_OPEN,
		  "a broken field is loose, not a body" },
		{ "X:1\nT:\n\nL:1/32\nV: x\n",                 std::string::npos, HEADER_OPEN,
		  "a blank line is not loose" },
		{ "X:1\nM:C|\nK:C\n",                            13, HEADER_KEY,
		  "a | in a field is not a bar line" },
	};
	for (size_t i = 0; i < sizeof(bodies) / sizeof(bodies[0]); i++)
	{
		const BodyCase & c  = bodies[i];
		HeaderWatch      w;
		const size_t     at = header_scan(w, c.text, false);
		const size_t     got = w.state == HEADER_OPEN ? std::string::npos : at;
		check(got == c.at && w.state == c.state, strf("header(body %zu)", i + 1).c_str(),
		      strf("at %zu, state %d", got, (int) w.state),
		      strf("at %zu, state %d (%s)", c.at, (int) c.state, c.why));
	}

	// §9.3: the fields a score is timed by.
	struct MissingCase
	{
		const char * abc;
		const char * missing;
	};
	const MissingCase missing[] = {
		{ "X:1\nT:\nM:6/8\nL:1/32\nQ:1/4=78\nV: Vocal\nK:Eb\n% intro\n", "" },
		{ "X:1\nT:\nL:1/32\nQ:1/4=78\nK:Eb\n",                           "M:" },
		{ "X:1\nT:\nM/8\nL:1/32\nQ:1/4=60\nK:Eb\n",                     "M:" },
		{ "X:1\nM:C|\nL: 1/8 \nK:C\n",                                    "" },
		{ "X:1\nM:none\nL:1/8\nK:Am\n",                                   "" },
		{ "X:1\nM:6/\nL:1/8x\nK:\n",                                      "M:, L:, K:" },
		{ "X:1\nM:3/4\nL:1/8\nV: Vocal\nz4|\nK:C\n",                    "K:" },
		{ "X:1\nM:3/4\nK:C\nL:1/8\n",                                     "L:" },
		{ "",                                                               "M:, L:, K:" },
	};
	for (size_t i = 0; i < sizeof(missing) / sizeof(missing[0]); i++)
	{
		const std::string got = score_header_missing(missing[i].abc);
		check(got == missing[i].missing, strf("score_header_missing(%zu)", i + 1).c_str(),
		      "\"" + got + "\"", strf("\"%s\"", missing[i].missing));
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
	run_clock();
	run_sections_multi();
	run_requests();
	run_draft_guard();
	run_plans();
	run_negative();
	run_spans();
	run_header();

	printf("%s: %d cases, %d failures\n", failures == 0 ? "PASS" : "FAIL", checked, failures);
	return failures == 0 ? 0 : 1;
}
