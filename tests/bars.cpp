// yue2-bars — SPEC_TEMPLATE.md §4 and §2: a table test for the bar counter a
// written hole is validated against, and for the template parser's request
// errors. Neither touches llama, a model or a device, so this runs anywhere.
//
// Like tests/sampler_diff.cpp it *includes* stage_ar.cpp rather than linking it:
// both functions live in that file's anonymous namespace, and including it
// guarantees the code under test is the code that ships.
//
//	build_template/yue2-bars [-v]
//
// Exit 0 = every case matched.

#include "stage_ar.cpp"

namespace
{

int failures = 0;
int checked  = 0;
bool verbose = false;

struct BarCase
{
	const char * line;
	int          m_num;
	int          m_den;
	int          l_num;
	int          l_den;
	int          bars;
	int          offlength;
	const char * why;
};

// The meters are the ones the cases are written against: 4/4 with L:1/32 is what
// the reference model emits, 4/4 with L:1/4 keeps the length arithmetic readable.
const BarCase BAR_CASES[] =
{
	// --- what a real two-voice score looks like ---------------------------
	{ "z16z16|z32|z32|z32|",                     4, 4, 1, 32, 4, 0, "four bars of rest" },
	{ "z16\"C\"z16|\"G\"z32|\"Am\"z32|\"F\"z32|",
	                                             4, 4, 1, 32, 4, 0, "chord symbols are not music" },
	{ "Z|c4G4e4G4c4G4e4G4|d4G4f4G4d4G4f4G4|e4G4g4G4e4G4g4G4|",
	                                             4, 4, 1, 32, 4, 0, "a bar of rest then three played" },
	{ "Z4|",                                     4, 4, 1, 32, 4, 0, "Zn is n bars" },
	{ "Z3|z16c4G4e4G4|",                         4, 4, 1, 32, 4, 0, "Zn plus a played bar" },
	{ "Z|",                                      4, 4, 1, 32, 1, 0, "bare Z is one bar" },

	// --- bar lines --------------------------------------------------------
	{ "C4||D4|]",                                4, 4, 1, 4,  2, 0, "|| and |] are one bar line each" },
	{ "|:C4:|",                                  4, 4, 1, 4,  1, 0, "repeat marks are bar lines" },
	{ "[|C4|",                                   4, 4, 1, 4,  1, 0, "[| is a bar line, not a chord" },
	{ "C4::D4|",                                 4, 4, 1, 4,  2, 0, ":: is one bar line" },
	{ "C4|D4",                                   4, 4, 1, 4,  2, 0, "a trailing run still counts" },
	{ "C4|[1 D4|E4:|[2 F4|G4|]",                 4, 4, 1, 4,  5, 0, "[1 and [2 are repeat endings, not chords" },
	{ "C4|[1 D4:|[2 E4||",                       4, 4, 1, 4,  3, 0, "the same, with the endings one bar long" },
	{ "[CE|D4|E4|",                              4, 4, 1, 4,  3, 1, "an unclosed [ may not swallow the bar lines" },
	{ "C4|\"Em|D4|",                             4, 4, 1, 4,  3, 1, "nor may an unclosed chord symbol" },
	{ "C4 !D4|E4|",                              4, 4, 1, 4,  2, 1, "nor an unpaired decoration mark" },
	{ "C4|D4| % end",                            4, 4, 1, 4,  2, 0, "a trailing comment is not a bar" },
	{ "|||",                                     4, 4, 1, 4,  0, 0, "bar lines without music are no bars" },
	{ "",                                        4, 4, 1, 4,  0, 0, "an empty line holds no bars" },
	{ "   ",                                     4, 4, 1, 4,  0, 0, "whitespace holds no bars" },

	// --- what must be stripped before counting ----------------------------
	{ "!p!C4|",                                  4, 4, 1, 4,  1, 0, "!...! is a decoration" },
	{ "[K:G]C4|",                                4, 4, 1, 4,  1, 0, "[K:..] is an inline field" },
	{ "{gag}C4|",                                4, 4, 1, 4,  1, 0, "grace notes carry no time" },
	{ "\"Cmaj7\"C4|",                            4, 4, 1, 4,  1, 0, "a chord symbol is text" },

	// --- note lengths -----------------------------------------------------
	{ "C D E F|",                                4, 4, 1, 4,  1, 0, "four quarters fill 4/4" },
	{ "C D E|",                                  4, 4, 1, 4,  1, 1, "three do not" },
	{ "C2 D2|",                                  4, 4, 1, 8,  1, 1, "four eighths do not fill 4/4" },
	{ "C/2C/2C/C/C2|",                           4, 4, 1, 4,  1, 0, "/2, / and a plain multiplier" },
	{ "C3/2D/2C D|",                             4, 4, 1, 4,  1, 0, "dotted pairs" },
	{ "C//C//C//C//C2 D|",                       4, 4, 1, 4,  1, 0, "// is a quarter of L" },
	{ "C>D E>F|",                                4, 4, 1, 4,  1, 0, "broken rhythm moves time, it does not add any" },
	{ "(3CCC D2|",                               4, 4, 1, 4,  1, 0, "a triplet takes the time of two" },
	{ "(3:2:3CCC D2|",                           4, 4, 1, 4,  1, 0, "the same, spelled out" },
	{ "[CEG]4|",                                 4, 4, 1, 4,  1, 0, "a chord sounds as one note" },
	{ "z4|",                                     4, 4, 1, 4,  1, 0, "a rest is music" },
	{ "x4|",                                     4, 4, 1, 4,  1, 0, "an invisible rest is too" },
	{ "C6|",                                     6, 8, 1, 8,  1, 0, "6/8 against L:1/8" },
	{ "C4|C2|",                                  4, 4, 1, 4,  2, 1, "one bar short, one bar right" },
	{ "C2-C2|",                                  4, 4, 1, 4,  1, 0, "a tie is not a note" },
};

struct BodyCase
{
	const char * line;
	bool         body;
};

const BodyCase BODY_CASES[] =
{
	{ "z32|",                       true  },
	{ "  z32|",                     true  },
	{ "V: Ins",                     false },
	{ "V: Vocal clef=treble",       false },
	{ "K:Em",                       false },
	{ "w: la la la",                false },
	{ "% intro",                    false },
	{ "%%yue2-gen",                 false },
	{ "",                           false },
	{ "\r",                         false },
	{ "X:1",                        false },
};

struct ParseCase
{
	const char * text;
	const char * err;      // nullptr = must parse
	int          segs;     // when it parses: how many segments
	int          holes;
	int          bars0;    // the first hole's bar count
};

const ParseCase PARSE_CASES[] =
{
	{ "X:1\nM:4/4\nL:1/4\nK:C\nC D E F|\n", nullptr, 1, 0, 0 },
	{ "M:4/4\nL:1/4\nK:C\nC D E F|\n%%yue2-gen\n",          nullptr, 2, 1, 1 },
	{ "M:4/4\nL:1/4\nK:C\nC4|D4|\n%%yue2-gen\n",            nullptr, 2, 1, 2 },
	{ "M:4/4\nL:1/4\nK:C\nC4|D4|\n%%yue2-gen bars=7\n",     nullptr, 2, 1, 7 },
	{ "M:4/4\nK:C\n%%yue2-primer-begin\nC4|\n%%yue2-primer-end\nD4|\n%%yue2-gen\n",
	                                                        nullptr, 4, 1, 1 },
	{ "C4|\n  %%yue2-gen\n",                                nullptr, 2, 1, 1 },
	{ "M:4/4\r\nL:1/4\r\nK:C\r\nC4|D4|\r\n%%yue2-gen\r\n",  nullptr, 2, 1, 2 },
	{ "K:C\n%%yue2-gen\n",                  "no body line",       0, 0, 0 },
	{ "M:4/4\nL:1/4\nK:C\n%%yue2-primer-begin\nC4|D4|\n%%yue2-primer-end\n%%yue2-gen\n",
	                                        "no body line",       0, 0, 0 },
	{ "C4|\n%%yue2-genbars=3\n",            "unknown directive",  0, 0, 0 },
	{ "C4|\n%%yue2-gen bars=99999\n",       "between 1 and 9999", 0, 0, 0 },
	{ "C4|\n%%yue2-gen bars=99999999999999999999\n",
	                                        "nothing but bars=",  0, 0, 0 },
	{ "C4|\n%%yue2-gen bars=0\n",           "between 1 and 9999", 0, 0, 0 },
	{ "C4|\n%%yue2-gen bars=-2\n",          "between 1 and 9999", 0, 0, 0 },
	{ "C4|\n%%yue2-gen bars=two\n",         "nothing but bars=",  0, 0, 0 },
	{ "C4|\n%%yue2-gen 4\n",                "nothing but bars=",  0, 0, 0 },
	{ "C4|\n%%yue2-primer-begin\nD4|\n",    "never closed",       0, 0, 0 },
	{ "%%yue2-primer-end\n",                "without a primer",   0, 0, 0 },
	{ "%%yue2-primer-begin\n%%yue2-primer-begin\nC4|\n%%yue2-primer-end\n",
	                                        "inside a primer",    0, 0, 0 },
	{ "C4|\n%%yue2-primer-begin\n%%yue2-gen\n%%yue2-primer-end\n",
	                                        "inside a primer",    0, 0, 0 },
	{ "C4|\n%%yue2-holes\n",                "unknown directive",  0, 0, 0 },
};

void check(bool ok, const char * what, const std::string & got, const std::string & want)
{
	checked++;
	if (ok)
	{
		if (verbose)
		{
			printf("ok   %s\n", what);
		}
		return;
	}
	failures++;
	printf("FAIL %s\n       got  %s\n       want %s\n", what, got.c_str(), want.c_str());
}

} // namespace

int main(int argc, char ** argv)
{
	for (int i = 1; i < argc; i++)
	{
		if (std::string(argv[i]) == "-v")
		{
			verbose = true;
		} else {
			fprintf(stderr, "usage: %s [-v]\n", argv[0]);
			return 2;
		}
	}

	for (size_t i = 0; i < sizeof(BAR_CASES) / sizeof(BAR_CASES[0]); i++)
	{
		const BarCase & c = BAR_CASES[i];
		Meter           m;
		m.m_num = c.m_num;
		m.m_den = c.m_den;
		m.l_num = c.l_num;
		m.l_den = c.l_den;
		const BarCount bc = count_bars(c.line, m);
		check(bc.bars == c.bars && bc.offlength == c.offlength,
		      strf("count_bars(\"%s\", M:%d/%d L:%d/%d) — %s", c.line,
		           c.m_num, c.m_den, c.l_num, c.l_den, c.why).c_str(),
		      strf("%d bars, %d off-length", bc.bars, bc.offlength),
		      strf("%d bars, %d off-length", c.bars, c.offlength));
	}

	for (size_t i = 0; i < sizeof(BODY_CASES) / sizeof(BODY_CASES[0]); i++)
	{
		const BodyCase & c = BODY_CASES[i];
		check(is_body_line(c.line) == c.body,
		      strf("is_body_line(\"%s\")", c.line).c_str(),
		      is_body_line(c.line) ? "body" : "not a body line",
		      c.body ? "body" : "not a body line");
	}

	for (size_t i = 0; i < sizeof(PARSE_CASES) / sizeof(PARSE_CASES[0]); i++)
	{
		const ParseCase &   c   = PARSE_CASES[i];
		std::vector<TplSeg> segs;
		const std::string   err = parse_template(c.text, segs);
		const std::string   what = strf("parse_template(case %zu)", i + 1);

		if (c.err != nullptr)
		{
			check(err.find(c.err) != std::string::npos, what.c_str(),
			      err.empty() ? "parsed" : err, strf("an error containing \"%s\"", c.err));
			continue;
		}
		if (!err.empty())
		{
			check(false, what.c_str(), err, "no error");
			continue;
		}

		int holes = 0;
		int bars0 = 0;
		for (size_t k = 0; k < segs.size(); k++)
		{
			if (segs[k].kind != TPL_HOLE)
			{
				continue;
			}
			bars0 = holes == 0 ? segs[k].bars : bars0;
			holes++;
		}
		check((int) segs.size() == c.segs && holes == c.holes && bars0 == c.bars0, what.c_str(),
		      strf("%zu segments, %d holes, first hole bars=%d", segs.size(), holes, bars0),
		      strf("%d segments, %d holes, first hole bars=%d", c.segs, c.holes, c.bars0));
	}

	printf("%s: %d cases, %d failures\n", failures == 0 ? "PASS" : "FAIL", checked, failures);
	return failures == 0 ? 0 : 1;
}
