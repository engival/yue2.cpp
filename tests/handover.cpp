// yue2-handover — SPEC_HANDOVER.md §7: table tests for the offset measurement
// of §4, for the forced history a leg is built from (§5), for the cut resolver
// of §2 and for every request error the block can raise. No llama, no model, no
// device: two synthetic streams and a score in a string are all it needs.
//
// Like tests/guidance.cpp it *includes* stage_ar.cpp rather than linking it:
// the functions under test live in that file's anonymous namespace, and
// including it guarantees the code under test is the code that ships.
//
//	build_handover/yue2-handover [-v]
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
			printf("ok   %-40s %s\n", what, got.c_str());
		}
		return;
	}
	failures++;
	printf("FAIL %-40s got %s, wanted %s\n", what, got.c_str(), want.c_str());
}

// ------------------------------------------------------------ the streams ---

// A stream of codec indices nobody could match by accident: 32768 values, so
// two independent ones agree on about three frames in a hundred thousand. The
// generator is a plain LCG, so every case below is the same on every machine.
std::vector<int32_t> stream(uint64_t seed, size_t n)
{
	std::vector<int32_t> out;
	out.reserve(n);
	for (size_t i = 0; i < n; i++)
	{
		seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
		out.push_back((int32_t) ((seed >> 33) % CODEC_SIZE));
	}
	return out;
}

// `take` made to agree with `song` at lag `k` — take[t - k] == song[t] — on one
// frame in `every`, over [from, to). That is the 2-3 % two takes of one score
// share at the lag one runs ahead of the other (§1).
void plant(const std::vector<int32_t> & song, std::vector<int32_t> & take, int k,
	int from, int to, int every)
{
	for (int t = from; t < to; t += every)
	{
		const long long u = (long long) t - k;
		if (u >= 0 && u < (long long) take.size() && t < (int) song.size())
		{
			take[(size_t) u] = song[(size_t) t];
		}
	}
}

void run_offsets()
{
	const std::vector<int32_t> song = stream(1, 4000);

	// --- the lag a take runs ahead is recovered, either way round ------------
	const int lags[] = { 11, -7, 0, 34, 100, -100 };
	for (size_t i = 0; i < sizeof(lags) / sizeof(lags[0]); i++)
	{
		std::vector<int32_t> take = stream(2 + i, 4000);
		plant(song, take, lags[i], 150, 3500, 32);
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == lags[i] && fit.confident, "offset: planted lag",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits),
		      strf("%+d, confident (3 %% of the frames planted)", lags[i]));
	}

	// --- the density two takes of one score really share: 2.5 % of a window --
	{
		std::vector<int32_t> take = stream(41, 4000);
		plant(song, take, 13, 3415 - HANDOVER_SPAN, 3415, 40);
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == 13 && fit.confident, "offset: 2.5 % over 750 frames",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits), "+13, confident");
	}

	// --- two streams that share nothing, over many seeds ---------------------
	// A z-score alone is fooled by one coincidental hit at one lag, so this is
	// the case the absolute hit count of §4 is there for: none of these may come
	// back confident.
	{
		int confident = 0;
		int hits      = 0;
		for (uint64_t seed = 100; seed < 130; seed++)
		{
			const OffsetFit fit = handover_offset(song, stream(seed, 4000), 3415);
			confident += fit.confident ? 1 : 0;
			hits       = std::max(hits, fit.hits);
		}
		check(confident == 0, "offset: 30 unrelated streams",
		      strf("%d confident, at most %d frames agreeing", confident, hits),
		      "none confident");
	}

	// --- §4's fallback: a lag nobody believes is dropped for 0 ---------------
	// The peak the retry scan found is what `measured` has to carry, and the
	// offset the leg is played at is 0 whatever that peak was.
	{
		int zeroed = 0;
		int kept   = 0;
		for (uint64_t seed = 100; seed < 130; seed++)
		{
			const std::vector<int32_t> take = stream(seed, 4000);
			const OffsetFit fit  = handover_offset(song, take, 3415);
			const OffsetFit scan = offset_scan(song, take, HANDOVER_LO, 3415);
			zeroed += fit.offset == 0 && !fit.confident ? 1 : 0;
			kept   += fit.measured == scan.offset ? 1 : 0;
		}
		check(zeroed == 30 && kept == 30, "offset: an unbelieved fit falls back to 0",
		      strf("%d at offset 0, %d keeping the peak in measured", zeroed, kept),
		      "all 30, and measured is the peak the retry scan found");
	}
	{
		// A believed lag is untouched by the fallback: offset and measured agree.
		std::vector<int32_t> take = stream(2, 4000);
		plant(song, take, 11, 150, 3500, 32);
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == 11 && fit.measured == 11 && fit.confident,
		      "offset: a believed lag is not dropped",
		      strf("%+d (measured %+d)", fit.offset, fit.measured), "+11, measured +11");
	}

	// --- one coincidental hit is a huge z and still not a lag ---------------
	{
		std::vector<int32_t> take = stream(77, 4000);
		plant(song, take, 40, 3000, 3001, 1);          // exactly one frame agrees
		const OffsetFit fit = offset_scan(song, take, 3415 - HANDOVER_SPAN, 3415);
		check(!fit.confident, "offset: one planted frame",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits),
		      strf("not confident (under %d agreeing frames)", HANDOVER_MIN_HITS));
	}

	// --- the window is flat, the song before it is not (§4's retry) ---------
	{
		std::vector<int32_t> take = stream(7, 4000);
		plant(song, take, 5, 150, 1000, 16);
		const OffsetFit near = offset_scan(song, take, 3415 - HANDOVER_SPAN, 3415);
		check(!near.confident, "offset: the last 750 frames are flat",
		      strf("%+d (z %.1f, %d hits)", near.offset, near.z, near.hits), "not confident");
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == 5 && fit.confident, "offset: retried from frame 150",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits), "+5, confident");
	}

	// --- ties go to the smallest |k|, and at equal |k| to the negative one ---
	{
		std::vector<int32_t> take = stream(11, 4000);
		plant(song, take,  3, 150, 3400, 40);
		plant(song, take, -9, 150, 3400, 40);
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == 3, "offset: two lags, equally planted",
		      strf("%+d (z %.1f)", fit.offset, fit.z), "+3, the smaller |k|");
	}
	{
		std::vector<int32_t> take = stream(13, 4000);
		plant(song, take,  6, 150, 3400, 40);
		plant(song, take, -6, 150, 3400, 40);
		const OffsetFit fit = handover_offset(song, take, 3415);
		check(fit.offset == -6, "offset: two lags at the same distance",
		      strf("%+d (z %.1f)", fit.offset, fit.z), "-6, the negative one");
	}

	// --- a cut so early that no frame is compared ---------------------------
	{
		std::vector<int32_t> take = stream(3, 4000);
		plant(song, take, 12, 0, 3500, 32);
		const OffsetFit fit = handover_offset(song, take, 100);
		check(fit.offset == 0 && !fit.confident && fit.z == 0, "offset: a cut before frame 150",
		      strf("%+d (z %.1f)", fit.offset, fit.z), "0, z 0, not confident");
	}

	// --- a take that stops short: the frames past its end are skipped -------
	{
		std::vector<int32_t> take = stream(5, 4000);
		plant(song, take, 8, 150, 3500, 32);
		take.resize(3000);
		const OffsetFit fit = handover_offset(song, take, 2900);
		check(fit.offset == 8 && fit.confident, "offset: a take that ends early",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits), "+8, confident");
	}

	// --- two identical streams: every lag matches nothing but lag 0 ---------
	{
		const OffsetFit fit = handover_offset(song, song, 2000);
		check(fit.offset == 0 && fit.confident, "offset: a take of itself",
		      strf("%+d (z %.1f, %d hits)", fit.offset, fit.z, fit.hits), "0, confident");
	}
}

// -------------------------------------------------------------- the legs ---

struct KeepCase
{
	int          cut;
	int          x;
	int          off;
	size_t       take_len;
	size_t       song_len;
	const char * err;     // a substring of the expected error, nullptr = it builds
	const char * why;
};

const KeepCase KEEP_CASES[] =
{
	{ 3415, 125,  11, 6400, 6425, nullptr, "the reference cut" },
	{  100,  25,  10, 4000, 4000, nullptr, "a short one" },
	{  100,  25, -10, 4000, 4000, nullptr, "a take that runs behind the song" },
	{  126, 125,   0, 4000, 4000, nullptr, "one frame of the take survives" },
	{  125, 125,   0, 4000, 4000, "leaves nothing before it", "the take is gone" },
	{  100, 125,   0, 4000, 4000, "leaves nothing before it", "the intrusion is longer than the song" },
	{ 3415, 125,  11, 3000, 6425, "past its", "the take is too short for the cut" },
	{ 3415, 125,  11, 6400, 3000, "the song ended", "the song ends before the cut" },
};

void run_keeps()
{
	for (size_t i = 0; i < sizeof(KEEP_CASES) / sizeof(KEEP_CASES[0]); i++)
	{
		const KeepCase & c    = KEEP_CASES[i];
		const std::string what = strf("keep(case %zu)", i + 1);
		const std::vector<int32_t> song = stream(1, c.song_len);
		const std::vector<int32_t> take = stream(2, c.take_len);

		std::vector<int32_t> keep;
		const std::string    err = handover_keep(song, take, c.cut, c.x, c.off, keep);
		if (c.err != nullptr)
		{
			check(err.find(c.err) != std::string::npos, what.c_str(),
			      err.empty() ? "built" : err, strf("an error containing \"%s\" (%s)",
			      c.err, c.why));
			continue;
		}
		if (!err.empty())
		{
			check(false, what.c_str(), err, strf("a history (%s)", c.why));
			continue;
		}
		// §5: the take up to x frames before the cut in its own clock, then the
		// last x frames of the song — c - off codes all told.
		bool same = keep.size() == (size_t) (c.cut - c.off);
		for (int t = 0; same && t < c.cut - c.x - c.off; t++)
		{
			same = keep[(size_t) t] == take[(size_t) t];
		}
		for (int t = 0; same && t < c.x; t++)
		{
			same = keep[keep.size() - (size_t) c.x + (size_t) t] ==
			       song[(size_t) (c.cut - c.x + t)];
		}
		check(same, what.c_str(), strf("%zu codes", keep.size()),
		      strf("%d codes, the take then the song (%s)", c.cut - c.off, c.why));
	}
}

// -------------------------------------------------------------- the cuts ---

// 4/4 at Q:1/4=120 is a two-second bar, so a bar line is 50 frames.
const char * SCORE =
	"X:1\n"
	"T:\n"
	"M:4/4\n"
	"L:1/8\n"
	"Q:1/4=120\n"
	"V: Vocal clef=treble\n"
	"V: Ins clef=treble\n"
	"K:C\n"
	"% intro\n"
	"V: Vocal\n"
	"abcd|efga|\n"
	"V: Ins\n"
	"abcd|efga|\n"
	"% verse\n"
	"V: Vocal\n"
	"abcd|efga|abcd|efga|\n"
	"% interlude\n"
	"V: Vocal\n"
	"abcd|efga|\n"
	"% verse\n"
	"V: Vocal\n"
	"abcd|efga|\n";

struct CutCase
{
	const char * section;   // nullptr = a "frame" entry
	int          nth;
	int          lead;
	int          frame;
	bool         found;
	int          cut;
	const char * why;
};

const CutCase CUT_CASES[] =
{
	// 2 bars of intro = 4 s = frame 100, less the 35-frame lead.
	{ "verse",     1, SECTIONS_LEAD, 0, true,   65, "the first verse, at the default lead" },
	{ "verse",     1,             0, 0, true,  100, "and without one" },
	// 2 + 4 + 2 bars before the second verse = 16 s = frame 400.
	{ "verse",     2, SECTIONS_LEAD, 0, true,  365, "per-name nth steps over the interlude" },
	{ "interlude", 1, SECTIONS_LEAD, 0, true,  265, "the interlude itself" },
	{ "intro",     1, SECTIONS_LEAD, 0, true,    1, "bar 0 clamps to frame 1" },
	{ "verse",     3, SECTIONS_LEAD, 0, false,   0, "an nth the score never reaches" },
	{ "chorus",    1, SECTIONS_LEAD, 0, false,   0, "a label the score never writes" },
	{ nullptr,     1, SECTIONS_LEAD, 2315, true, 2315, "a frame is the base take's own clock" },
};

void run_cuts()
{
	for (size_t i = 0; i < sizeof(CUT_CASES) / sizeof(CUT_CASES[0]); i++)
	{
		const CutCase &   c    = CUT_CASES[i];
		const std::string what = strf("cut(case %zu)", i + 1);

		std::vector<HandoverEntry> es(1);
		if (c.section == nullptr)
		{
			es[0].has_frame = true;
			es[0].frame     = c.frame;
		} else {
			es[0].section = c.section;
			es[0].nth     = c.nth;
			es[0].lead    = c.lead;
		}
		handover_locate(SCORE, es);
		check(es[0].found == c.found && (!c.found || es[0].cut == c.cut), what.c_str(),
		      es[0].found ? strf("frame %d (bar %d, %.2f s)", es[0].cut, es[0].bar,
		                         es[0].bar_seconds)
		                  : std::string("not found"),
		      c.found ? strf("frame %d (%s)", c.cut, c.why) : strf("not found (%s)", c.why));
	}

	// Several entries at once: the order is the song's, and the two forms mix.
	{
		std::vector<HandoverEntry> es(3);
		es[0].section = "verse";
		es[1].has_frame = true;
		es[1].frame     = 200;
		es[2].section = "verse";
		es[2].nth     = 2;
		handover_locate(SCORE, es);
		check(es[0].cut == 65 && es[1].cut == 200 && es[2].cut == 365 &&
		      handover_cuts_check(es).empty(), "cuts: labels and frames mixed",
		      strf("%d, %d, %d", es[0].cut, es[1].cut, es[2].cut), "65, 200, 365, accepted");
	}
	{
		std::vector<HandoverEntry> es(2);
		es[0].section   = "verse";
		es[0].nth       = 2;
		es[1].has_frame = true;
		es[1].frame     = 100;
		handover_locate(SCORE, es);
		const std::string err = handover_cuts_check(es);
		check(err.find("not after") != std::string::npos, "cuts: the second is before the first",
		      err.empty() ? "accepted" : err, "an error naming the order");
	}
	{
		// Stage 8's resolver matches the entries in the order they are given and
		// stops at the first one it cannot find, so a label the score never
		// writes takes the entries behind it with it. Both are then reported as
		// unreached rather than being an error (§2), and the song is simply the
		// base take from there on.
		std::vector<HandoverEntry> es(2);
		es[0].section = "chorus";
		es[1].section = "verse";
		es[1].nth     = 2;
		handover_locate(SCORE, es);
		check(!es[0].found && !es[1].found && handover_cuts_check(es).empty(),
		      "cuts: an unwritten label stops the ones behind it",
		      strf("%s, %s", es[0].found ? "found" : "not found",
		           es[1].found ? "found" : "not found"), "neither found, accepted");
	}
}

// --------------------------------------------------------- before the legs ---

// The checks of §5 that are made once every take is in memory and before the
// first leg is decoded — the ones that stop a skipped entry from silently
// ending the song where its predecessor's leg was cut short.
HandoverEntry entry(int cut, int x, bool auto_off, int off)
{
	HandoverEntry h;
	h.has_frame = true;
	h.frame     = cut;
	h.found     = true;
	h.cut       = cut;
	h.x         = x;
	h.auto_off  = auto_off;
	h.offset    = off;
	return h;
}

std::vector<HandoverTake> one_take(size_t frames)
{
	std::vector<HandoverTake> takes(1);
	takes[0].label = "take_0";
	takes[0].codes = stream(1, frames);
	return takes;
}

void run_preflight()
{
	const ArBatchParams p;      // no --max-semantic: the protocol's own cap
	const int           cap = semantic_cap(p);

	{
		std::vector<HandoverEntry> es(1, entry(3415, 125, true, 0));
		check(handover_preflight(es, one_take(6400), p, cap).empty(),
		      "preflight: a cut a take reaches", "accepted", "accepted");
	}
	{
		// The take ends before the cut at every lag the scan could pick.
		std::vector<HandoverEntry> es(1, entry(3415, 125, true, 0));
		const std::string err = handover_preflight(es, one_take(3000), p, cap);
		check(err.find("frames, and the cut") != std::string::npos,
		      "preflight: a take that is too short", err.empty() ? "accepted" : err,
		      "an error naming the take's length");
	}
	{
		// It would fit at a large enough offset, so only the run can tell.
		std::vector<HandoverEntry> es(1, entry(3415, 125, true, 0));
		check(handover_preflight(es, one_take(3350), p, cap).empty(),
		      "preflight: a take the best lag would reach", "accepted",
		      "accepted (the leg itself is the backstop)");
	}
	{
		// The same take with the offset given: now it is arithmetic.
		std::vector<HandoverEntry> es(1, entry(3415, 125, false, 12));
		const std::string err = handover_preflight(es, one_take(3350), p, cap);
		check(err.find("frames, and the cut") != std::string::npos,
		      "preflight: a given offset past the take's end",
		      err.empty() ? "accepted" : err, "an error");
	}
	{
		// A cut with less than the intrusion before it needs no take at all.
		std::vector<HandoverEntry> es(1, entry(100, 125, true, 0));
		std::vector<HandoverTake>  takes(1);
		const std::string          err = handover_preflight(es, takes, p, cap);
		check(err.find("less than the") != std::string::npos,
		      "preflight: the intrusion does not fit", err.empty() ? "accepted" : err,
		      "an error, before any take is rendered");
	}
	{
		// Forced frames against the cap the leg would run under.
		ArBatchParams small = p;
		small.max_semantic  = 600;
		std::vector<HandoverEntry> es(1, entry(3415, 125, true, 0));
		const std::string err = handover_preflight(es, one_take(6400), small,
		                                           semantic_cap(small));
		check(err.find("leave nothing to sample") != std::string::npos,
		      "preflight: --max-semantic under the forced frames",
		      err.empty() ? "accepted" : err, "an error naming the cap");
	}
	{
		// An entry the score never wrote is not checked: it is reported, not played.
		std::vector<HandoverEntry> es(1);
		es[0].section = "chorus";
		check(handover_preflight(es, one_take(100), p, cap).empty(),
		      "preflight: an unresolved entry", "accepted", "accepted");
	}
}

// ---------------------------------------------------------- the requests ---

std::string request_with(const std::string & extra)
{
	return "{\"style\": \"slow waltz, upright bass\", \"lyrics\": \"[Verse]\\nrain on the tin\\n\""
	       ", \"abc\": \"X:1\\nM:3/4\\nK:C\\n\""
	       + (extra.empty() ? std::string() : ", " + extra) + "}";
}

#define HAND_ONE "\"handover\": [{\"section\": \"chorus\"}]"

struct RequestCase
{
	const char * extra;
	const char * err;     // a substring of the expected error, nullptr = must be accepted
	const char * why;
};

const RequestCase REQUEST_CASES[] =
{
	// --- accepted -----------------------------------------------------------
	{ HAND_ONE,                                                       nullptr,
	  "back to the request's own style at the first chorus" },
	{ "\"handover\": [{\"frame\": 3415, \"style\": \"roots reggae\", \"seconds\": 1}]", nullptr,
	  "a hard cut at a frame" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 3, \"take\": \"takes/reggae\","
	  " \"offset\": -12, \"lead_frames\": 0}]",                       nullptr,
	  "an earlier render, at a measured offset" },
	{ "\"handover\": [{\"frame\": 2315, \"style\": \"a\"}, {\"frame\": 3415, \"style\": \"b\"}]",
	  nullptr, "two cuts in order" },
	{ "\"base_take\": \"out/take\", " HAND_ONE,                       nullptr,
	  "an earlier render as the base" },
	{ "\"cfg_scale\": 1.0, " HAND_ONE,                                nullptr,
	  "cfg_scale 1 is no guidance at all" },

	// --- "at": a time in the base take's audio ------------------------------
	{ "\"handover\": [{\"at\": 78.6, \"style\": \"roots reggae\"}]",  nullptr,
	  "seconds as a number" },
	{ "\"handover\": [{\"at\": \"78.6\"}]",                           nullptr,
	  "and as a string" },
	{ "\"handover\": [{\"at\": \"1:18.6\"}]",                         nullptr,
	  "minutes and seconds" },
	{ "\"handover\": [{\"at\": \"1:18\"}]",                           nullptr,
	  "without the fraction" },
	{ "\"handover\": [{\"at\": 79}]",                                 nullptr,
	  "a whole number of seconds" },
	{ "\"handover\": [{\"at\": \"1:00\"}, {\"frame\": 2000}]",        nullptr,
	  "an \"at\" and a \"frame\" are one timeline, and these are in order" },
	{ "\"handover\": [{\"at\": \"2:00\"}, {\"frame\": 2000}]",        "not after",
	  "and out of order they are caught before anything renders" },
	{ "\"handover\": [{\"at\": 0.02}]",                               "less than the",
	  "0.02 s is frame 1, which has nothing before it" },
	{ "\"handover\": [{\"at\": -1}]",                                 "\"at\"",
	  "a time before the song" },
	{ "\"handover\": [{\"at\": \"-5\"}]",                             "\"at\"",
	  "a sign is not part of a clock" },
	{ "\"handover\": [{\"at\": 0}]",                                  "\"at\"",
	  "frame 0 is before the first sampled step" },
	{ "\"handover\": [{\"at\": 100000}]",                             "\"at\"",
	  "a time past the longest song" },
	{ "\"handover\": [{\"at\": 1e30}]",                               "\"at\"",
	  "a finite time whose frame no integer holds" },
	{ "\"handover\": [{\"at\": \"1:75\"}]",                           "\"at\"",
	  "75 seconds is not a clock" },
	{ "\"handover\": [{\"at\": \"1:2:3\"}]",                          "\"at\"",
	  "hours are not a form" },
	{ "\"handover\": [{\"at\": \"\"}]",                               "\"at\"",
	  "an empty time" },
	{ "\"handover\": [{\"at\": \"abc\"}]",                            "\"at\"",
	  "not a number at all" },
	{ "\"handover\": [{\"at\": \"78.6s\"}]",                          "\"at\"",
	  "trailing junk" },
	{ "\"handover\": [{\"at\": \" 78.6\"}]",                          "\"at\"",
	  "nor leading space" },
	{ "\"handover\": [{\"at\": \"1:.6\"}]",                           "\"at\"",
	  "a point needs a digit on both sides" },
	{ "\"handover\": [{\"at\": true}]",                               "\"at\"",
	  "a JSON type that is neither number nor string" },
	{ "\"handover\": [{\"at\": [78.6]}]",                             "\"at\"",
	  "nor a list of one" },
	{ "\"handover\": [{\"at\": 78.6, \"frame\": 2000}]",              "not both",
	  "a time and a frame are two ways to say where" },
	{ "\"handover\": [{\"section\": \"verse\", \"at\": 78.6}]",       "not both",
	  "and so are a label and a time" },
	{ "\"handover\": [{\"at\": 78.6, \"nth\": 2}]",                   "belong to a \"section\"",
	  "nth says which label, and a time names none" },
	{ "\"handover\": [{\"at\": 78.6, \"lead_frames\": 0}]",           "belong to a \"section\"",
	  "the lead is a label's too" },
	{ "\"handover\": [{\"frame\": 2000, \"nth\": 2}]",                "belong to a \"section\"",
	  "nor does a frame have an nth" },

	// --- one mechanism per request ------------------------------------------
	{ "\"guidance\": [{\"frame\": 400, \"style\": \"x\"}], " HAND_ONE, "\"guidance\"",
	  "handover beside guidance" },
	{ "\"sections\": [{\"section\": \"verse\", \"style\": \"x\"}], " HAND_ONE, "\"sections\"",
	  "handover beside sections" },
	{ "\"cfg_scale\": 3, " HAND_ONE,                                  "cfg_scale",
	  "handover beside a scalar push" },
	{ "\"semantic_keep\": {\"file\": \"s.npy\", \"frames\": 64}, " HAND_ONE, "semantic_keep",
	  "handover beside a keep of its own" },
	{ "\"abc_template\": \"X:1\\nK:C\\n%%yue2-gen bars=4\\n\", " HAND_ONE, "abc_template",
	  "handover beside a template" },
	{ "\"base_take\": \"out/take\"",                                  "\"base_take\"",
	  "a base take with nothing to hand over" },

	// --- the block itself ---------------------------------------------------
	{ "\"handover\": []",                                             "non-empty",
	  "an empty list" },
	{ "\"handover\": [3]",                                            "not an object",
	  "an entry that is not an object" },
	{ "\"handover\": [{}]",                                           "needs a \"section\"",
	  "an entry that says nothing" },
	{ "\"handover\": [{\"section\": \"verse\", \"frame\": 100}]",     "not both",
	  "both ways of saying where" },
	{ "\"handover\": [{\"frame\": 100, \"style\": \"a\", \"take\": \"d\"}]", "two ways to name",
	  "both ways of saying what" },
	{ "\"handover\": [{\"frame\": 0}]",                               "\"frame\"",
	  "frame 0 is before the first sampled step" },
	{ "\"handover\": [{\"section\": \"verse\", \"nth\": 0}]",         "\"nth\"",
	  "nth is 1-based" },
	{ "\"handover\": [{\"section\": \"verse\", \"lead_frames\": 999}]", "lead_frames",
	  "a lead of 40 seconds" },
	{ "\"handover\": [{\"frame\": 100, \"seconds\": 0}]",             "\"seconds\"",
	  "an intrusion of nothing" },
	{ "\"handover\": [{\"frame\": 100, \"seconds\": 40}]",            "\"seconds\"",
	  "an intrusion longer than the window" },
	{ "\"handover\": [{\"frame\": 100, \"offset\": \"measure\"}]",    "\"offset\"",
	  "the only word is \"auto\"" },
	{ "\"handover\": [{\"frame\": 100, \"offset\": 250}]",            "\"offset\"",
	  "an offset past the lags that are tried" },
	{ "\"handover\": [{\"frame\": 100, \"style\": 4}]",               "\"style\"",
	  "a style that is not tags" },
	{ "\"handover\": [{\"frame\": 100, \"take\": \"\"}]",             "\"take\"",
	  "an empty directory name" },
	{ "\"handover\": [{\"frame\": 100, \"wobble\": 2}]",              "unknown key",
	  "a key nobody knows" },
	{ "\"handover\": [{\"frame\": 3415}, {\"frame\": 2315}]",         "not after",
	  "two frames out of order" },
	{ "\"handover\": [{\"frame\": 3415}, {\"frame\": 3415}]",         "not after",
	  "the same frame twice" },
	{ "\"handover\": [{\"frame\": 100, \"offset\": 0}]",              "less than the",
	  "the default 5 s intrusion does not fit before frame 100" },
	{ "\"handover\": [{\"frame\": 100}]",                              "less than the",
	  "and an auto offset does not excuse it" },
	{ "\"handover\": [{\"frame\": 500}, {\"section\": \"verse\"}, {\"frame\": 400}]",
	  "not after", "a label between two frames does not reset the order" },
	{ "\"handover\": [{\"frame\": 100, \"offset\": 96, \"seconds\": 0.2}]", "leaves nothing",
	  "nor does the take at that offset" },
	{ "\"base_take\": 5, " HAND_ONE,                                  "\"base_take\"",
	  "a base take that is not a directory" },
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
			check(err.empty(), what.c_str(), err.empty() ? "accepted" : err,
			      strf("accepted (%s)", c.why));
			continue;
		}
		check(err.find(c.err) != std::string::npos, what.c_str(),
		      err.empty() ? "accepted" : err,
		      strf("an error containing \"%s\" (%s)", c.err, c.why));
	}

	// The defaults of §2, which no error message shows.
	{
		Request           req;
		const std::string err = parse_request_json(json::parse(request_with(HAND_ONE)), "R", req);
		const HandoverEntry & h = req.handover[0];
		check(err.empty() && h.nth == 1 && h.lead == SECTIONS_LEAD && h.auto_off &&
		      handover_x(h.seconds) == 125 && !h.has_style && h.take.empty(),
		      "request: the defaults",
		      strf("nth %d, lead %d, %s, x %d", h.nth, h.lead, h.auto_off ? "auto" : "fixed",
		           handover_x(h.seconds)),
		      "nth 1, lead 35, auto, x 125");
	}
	{
		// cot=off has no score to name a label in. The rule is here rather than
		// in the table because the table's requests all carry an "abc", and an
		// external score is refused for cot=off before the handover is looked at.
		Request     req;
		std::string err = parse_request_json(json::parse(
			"{\"style\": \"a\", \"lyrics\": \"b\", \"cot\": \"off\", " HAND_ONE "}"),
			"R", req);
		if (err.empty())
		{
			err = validate_request(req);
		}
		check(err.find("cot=off") != std::string::npos, "request: cot=off has no labels",
		      err.empty() ? "accepted" : err, "an error naming cot=off");
	}
	{
		// §2: frame = round(seconds × 25), no lead, and the three string forms
		// mean what the number does. What the request wrote is kept as it wrote
		// it, for handover.json (§6) and for the error messages.
		Request           req;
		const std::string err = parse_request_json(json::parse(request_with(
			"\"handover\": [{\"at\": 78.6}, {\"at\": \"78.6\"}, {\"at\": \"1:18.6\"}, "
			"{\"at\": \"1:18\"}]")), "R", req);
		const HandoverEntry & h = req.handover[0];
		check(err.empty() && h.has_frame && h.has_at && h.frame == 1965 &&
		      req.handover[1].frame == 1965 && req.handover[2].frame == 1965 &&
		      req.handover[3].frame == 1950 && handover_where(h, h.frame) == "78.6 (frame 1965)" &&
		      handover_where(req.handover[2], 1965) == "1:18.6 (frame 1965)",
		      "request: \"at\" to a frame",
		      err.empty() ? strf("%d, %d, %d, %d — %s", h.frame, req.handover[1].frame,
		                         req.handover[2].frame, req.handover[3].frame,
		                         handover_where(h, h.frame).c_str())
		                  : err,
		      "1965, 1965, 1965, 1950, named by the text the request wrote");
	}
	{
		// The intrusion in frames, rounded once (§2).
		check(handover_x(1) == 25 && handover_x(0.2) == 5 && handover_x(4.98) == 125 &&
		      handover_x(30) == 750, "request: seconds to frames",
		      strf("%d %d %d %d", handover_x(1), handover_x(0.2), handover_x(4.98),
		           handover_x(30)), "25 5 125 750");
	}
}

} // namespace

int main(int argc, char ** argv)
{
	for (int i = 1; i < argc; i++)
	{
		verbose = verbose || std::string(argv[i]) == "-v";
	}

	run_offsets();
	run_keeps();
	run_cuts();
	run_preflight();
	run_requests();

	printf("%s: %d cases, %d failures\n", failures == 0 ? "PASS" : "FAIL", checked, failures);
	return failures == 0 ? 0 : 1;
}
