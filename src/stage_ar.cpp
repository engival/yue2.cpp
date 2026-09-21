// yue2-ar — YuE2-3B autoregressive stages (ABC plan + semantic codec tokens)
// on libllama. See SPEC_AR.md §1 and §3; the reference semantics live in
// venv_yue2/.../yue2/{protocol,sampling,pipeline}.py.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "stage_ar.hpp"

#include "common/device.hpp"
#include "common/fileio.hpp"
#include "common/util.hpp"

#include "npy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Private to this translation unit: stage_ar/stage_nar/stage_vae each have
// their own Config / Model / Builder / Graph, and `yue2` links all three.
namespace
{

// --------------------------------------------------------------- protocol ---

static const int EOD           = 151643;
static const int ABC_START     = 151847;
static const int ABC_END       = 151848;
static const int MUSIC_START   = 151851;
static const int MUSIC_END     = 151852;
static const int CODEC_OFFSET  = 151853;
static const int CODEC_SIZE    = 32768;
static const int VOCAB_SIZE    = 184704;
static const int CONTEXT       = 24576;

static const char * INSTRUCTION_OFF =
	"Generate music with codec tokens from the given conditions.";
static const char * INSTRUCTION_MELODY =
	"Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.";
static const char * INSTRUCTION_FULL =
	"Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";

static const char * instruction(const std::string & cot)
{
	if (cot == "off")
	{
		return INSTRUCTION_OFF;
	}
	if (cot == "melody")
	{
		return INSTRUCTION_MELODY;
	}
	return INSTRUCTION_FULL;
}

struct Sampling
{
	double temperature;
	double top_p;
	int    top_k;
	double repetition_penalty;
	int    penalty_window;
	int    min_tokens;
	int    max_tokens;
};

// yue2_generation_config.json / protocol.GenerationConfig defaults.
static Sampling sampling_abc()
{
	Sampling s = {0.7, 0.9, 30, 1.005, 100, 32, 4096};
	return s;
}

static Sampling sampling_semantic()
{
	Sampling s = {1.0, 0.95, 100, 1.2, 50, 200, 9000};
	return s;
}

// -------------------------------------------------------------- guidance ---

// SPEC_GUIDANCE.md. The semantic phase can be decoded beside up to two negative
// branches, all three conditioned on the same generated history:
//
//	L = B + sum_i w_i(t) * (B - N_i)
//
// `previous` is the positive prefix that was in force before the entry opened
// it, `blank` is the reference's negative prefix (§2.2). Each carries a weight
// curve in frames after its entry's frame; 25 frames = 1 s of audio.
enum BranchKind { BRANCH_PREVIOUS = 0, BRANCH_BLANK = 1, BRANCH_KINDS = 2 };

static const char * branch_name(int kind)
{
	return kind == BRANCH_PREVIOUS ? "previous" : "blank";
}

// What a keyframe's weight may be, and how far out a keyframe may sit. The
// offset ceiling is only there to keep the arithmetic in int range; a curve that
// reaches past the semantic phase is legal and simply never gets there.
static const double GUIDANCE_MAX_WEIGHT = 20.0;
static const int    GUIDANCE_MAX_OFFSET = 1000000;

struct Keyframe
{
	int    offset = 0;   // frames after the entry's frame
	double weight = 0;
};

struct GuidanceEntry
{
	int                   frame     = 0;
	bool                  has_style = false;
	std::string           style;               // the tags of the new positive prefix
	bool                  has[BRANCH_KINDS]  = { false, false };
	std::vector<Keyframe> curve[BRANCH_KINDS];
	bool                  reached   = false;   // the decode loop got this far
};

// The semantic codec's frame rate, and what SPEC_SECTIONS §2 allows a
// "lead_frames" and an "nth" to be.
static const double SEMANTIC_FPS      = 25.0;
static const int    SECTIONS_LEAD     = 35;
static const int    SECTIONS_MAX_LEAD = 250;
static const int    SECTIONS_MAX_NTH  = 999;

// SPEC_SECTIONS.md: one "sections" entry — the tags a named score label and
// everything after it is written under. The `against` curves are exactly a
// guidance entry's, so an entry compiles into one (sections_plan); the fields
// below the blank line are what resolving it against the score fills in.
struct SectionEntry
{
	std::string           section;                 // the label, as the score writes it after "% "
	int                   nth     = 1;             // which occurrence of it, 1-based
	std::string           style;                   // the tags from that label on
	int                   lead    = SECTIONS_LEAD; // frames before its bar line to swap at
	bool                  has[BRANCH_KINDS] = { false, false };
	std::vector<Keyframe> curve[BRANCH_KINDS];

	bool                  found   = false;   // the label turned up in the score
	int                   line    = 0;       // its 1-based line number in score.abc
	int                   bar     = 0;       // bars of Vocal written before it
	double                seconds = 0;       // where that bar starts, in seconds
	int                   frame   = 0;       // the semantic step the swap is at
};

// SPEC_HANDOVER §2/§4. What a "handover" entry may ask for, and the window the
// offset is measured over: two takes of one score share 2-3 % identical tokens
// at the lag one runs ahead of the other and next to none at every other lag.
static const double HANDOVER_SECONDS     = 5.0;   // the intrusion X, in seconds
static const double HANDOVER_MIN_SECONDS = 0.2;
static const double HANDOVER_MAX_SECONDS = 30.0;
static const int    HANDOVER_RANGE       = 100;   // lags tried either way
static const int    HANDOVER_SPAN        = 750;   // frames of song a rate is measured over
static const int    HANDOVER_LO          = 150;   // and never before this frame
static const double HANDOVER_MIN_Z       = 6.0;   // under this the peak is not believed
static const int    HANDOVER_MIN_HITS    = 6;     // nor under this many agreeing frames
static const int    HANDOVER_TAIL        = 25;    // frames a leg runs past the next cut

// SPEC_HANDOVER.md: one "handover" entry — where the song changes hands, and to
// which take of the same score. The fields below the blank line are filled in
// as the song is rendered, and are what handover.json records (§6).
struct HandoverEntry
{
	bool        has_frame = false;             // "frame": the base take's own timeline
	int         frame     = 0;
	bool        has_at    = false;             // or "at": a time in the base take's audio
	json        at_json;                       // as the request wrote it (§6)
	std::string section;                       // or a label, through stage 8's resolver
	int         nth       = 1;
	int         lead      = SECTIONS_LEAD;
	bool        has_style = false;             // the engine renders this style's take
	std::string style;
	std::string take;                          // or an earlier render's artifacts directory
	double      seconds   = HANDOVER_SECONDS;  // the intrusion X
	bool        auto_off  = true;              // "offset": "auto"
	int         offset    = 0;

	bool        found     = false;   // the label turned up in the base score
	int         line      = 0;       // its 1-based line number there
	int         bar       = 0;       // bars of Vocal written before it
	double      bar_seconds = 0;     // where that bar starts, in seconds
	int         cut       = 0;       // the frame the song changes hands at
	int         x         = 0;       // the intrusion, in frames
	size_t      idx       = 0;       // which take takes over
	double      z         = 0;       // how far the offset's peak stands out
	int         hits      = 0;       // frames of the window that agree at it
	bool        has_z     = false;   // a given offset is not measured
	bool        confident = true;
	int         measured  = 0;       // the lag that was not believed, when it was not
	bool        reached   = false;   // the song got as far as the cut
	int         gave      = 0;       // frames of the final song this leg contributed
	std::string note;                // why it was skipped, "" when it was not
};

// One take: a full render of one score under one style, and the stream a leg is
// forced through. `label` is the `take_<k>` the engine wrote it to, or the
// directory an entry named (SPEC_HANDOVER §2).
struct HandoverTake
{
	std::string          style;
	std::string          label;
	std::string          dir;      // "" for a take the engine renders
	std::string          score;    // its score.abc, held for the §2 comparison
	std::vector<int32_t> codes;    // empty until the engine has rendered it

	// What the log calls it: the last element of the path, as the keep line of
	// SPEC_KEEP §4 shows a file. handover.json carries the whole label.
	std::string name() const
	{
		return std::filesystem::path(label).filename().string();
	}
};

// The curve's weight `off` frames into its entry: linear between keyframes, the
// first weight before the first and the last weight after the last. Two
// keyframes at one offset are a step, and the later one wins at that offset —
// which is what the `off <= kf[i].offset` return gives, since the scan below
// stops on the *last* keyframe at or before `off` (§2.3).
static double curve_at(const std::vector<Keyframe> & kf, int off)
{
	size_t i = 0;
	while (i + 1 < kf.size() && kf[i + 1].offset <= off)
	{
		i++;
	}
	if (off <= kf[i].offset || i + 1 == kf.size())
	{
		return kf[i].weight;
	}
	const double span = kf[i + 1].offset - kf[i].offset;
	return kf[i].weight + (kf[i + 1].weight - kf[i].weight) * ((off - kf[i].offset) / span);
}

// Is the curve non-zero anywhere at or after `off`? A branch whose curve is not
// is dropped and stops being decoded — that is what keeps guidance's cost to the
// transition window (§4.3). The pieces are linear, so a piece whose two ends are
// both zero is zero throughout and the ends are all that has to be looked at.
static bool curve_needed_from(const std::vector<Keyframe> & kf, int off)
{
	if (kf.back().weight != 0)
	{
		return true;
	}
	for (size_t i = 0; i + 1 < kf.size(); i++)
	{
		if (kf[i + 1].offset < off)
		{
			continue;
		}
		if (curve_at(kf, std::max(off, kf[i].offset)) != 0 || kf[i + 1].weight != 0)
		{
			return true;
		}
	}
	return false;
}

struct Request
{
	std::string style;
	std::string lyrics;
	std::string cot        = "full";
	uint64_t    seed       = 831001;
	bool        has_abc    = false;
	std::string abc;
	bool        has_tpl    = false;   // "abc_template": SPEC_TEMPLATE.md
	std::string abc_template;
	bool        has_cfg    = false;
	double      cfg_scale  = 1.0;
	bool        has_guidance = false;             // "guidance": SPEC_GUIDANCE.md
	std::vector<GuidanceEntry> guidance;
	bool        has_keep   = false;               // "semantic_keep": SPEC_KEEP.md
	std::string keep_file;                        // a semantic.npy of an earlier render
	int         keep_frames = 0;                  // N, how many of its leading codes to keep
	bool        has_sections = false;             // "sections": SPEC_SECTIONS.md
	std::vector<SectionEntry> sections;
	bool        has_handover = false;             // "handover": SPEC_HANDOVER.md
	std::vector<HandoverEntry> handover;
	json        handover_json;                    // the block as the request wrote it (§6)
	std::string base_take;                        // an earlier render to use as the base take
	std::string id         = "song";

	std::string text() const
	{
		return std::string(instruction(cot)) + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n";
	}
};

// protocol.SongRequest.guidance: the effective cfg scalar, 1.01 for cot=off.
static double cfg_scalar(const Request & r)
{
	return r.has_cfg ? r.cfg_scale : (r.cot == "off" ? 1.01 : 1.0);
}

// Every cut of a "sections" request is a plain swap when no entry names an
// `against`: nothing is ever pushed against, so the song owns one KV stream all
// the way through and needs no rule about --parallel (SPEC_SECTIONS §4).
static bool sections_plain_swap(const std::vector<SectionEntry> & es)
{
	for (size_t i = 0; i < es.size(); i++)
	{
		if (es[i].has[BRANCH_PREVIOUS] || es[i].has[BRANCH_BLANK])
		{
			return false;
		}
	}
	return true;
}

// Either request form asks for branches beside the positive sequence.
static bool is_guided(const Request & r)
{
	if (r.has_sections)
	{
		return !sections_plain_swap(r.sections);
	}
	return r.has_guidance || cfg_scalar(r) != 1.0;
}

// The entries the decode loop runs, so that both request forms are one
// mechanism: plain `cfg_scale = c` is the blank branch live from semantic step 0
// with the constant weight c - 1 (SPEC_GUIDANCE §2.2).
static std::vector<GuidanceEntry> guidance_plan(const Request & r)
{
	if (r.has_guidance)
	{
		return r.guidance;
	}
	std::vector<GuidanceEntry> out;
	const double               c = cfg_scalar(r);
	if (c != 1.0)
	{
		GuidanceEntry g;
		Keyframe      k;
		k.offset             = 0;
		k.weight             = c - 1.0;
		g.has[BRANCH_BLANK]  = true;
		g.curve[BRANCH_BLANK].push_back(k);
		out.push_back(g);
	}
	return out;
}

// The guidance entries a "sections" request compiles into: one per entry, in
// order, the frame still unknown — the score phase fills it in when it finds the
// label (SPEC_SECTIONS §3). Built before the score is written, so the context is
// sized on every entry's prefix head like any guided song's, and compacted down
// to the entries that were found once it is (sections_compile).
static std::vector<GuidanceEntry> sections_plan(const std::vector<SectionEntry> & es)
{
	std::vector<GuidanceEntry> out;
	for (size_t i = 0; i < es.size(); i++)
	{
		GuidanceEntry g;
		g.frame     = es[i].frame;
		g.has_style = true;
		g.style     = es[i].style;
		for (int k = 0; k < BRANCH_KINDS; k++)
		{
			g.has[k]   = es[i].has[k];
			g.curve[k] = es[i].curve[k];
		}
		out.push_back(g);
	}
	return out;
}

// ------------------------------------------------------------------ utils ---

static void quiet_log(enum ggml_log_level level, const char * text, void * /*user*/)
{
	if (level >= GGML_LOG_LEVEL_WARN)
	{
		fputs(text, stderr);
	}
}

// die() with the message handed back instead of printed: a batch validates every
// request before the model loads, and --continue-on-error turns a bad one into a
// recorded status rather than an exit.
static std::string strf(const char * fmt, ...)
{
	char    buf[2048];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	return buf;
}

// -------------------------------------------------------- score templates ---

// SPEC_TEMPLATE.md. A template is score text in which a `%%yue2-gen` line is a
// hole the model writes, and `%%yue2-primer-begin` / `%%yue2-primer-end` bracket
// lines that are fed as context and then dropped from the emitted score.
//
// Everything in this section is pure text work — no llama, no Request — so the
// bar counter the holes are validated against can be table-tested on its own
// (tests/bars.cpp).

// Attempts at one hole before it is filled with rests.
static const int TEMPLATE_ATTEMPTS = 4;

// The largest `bars=N` a hole may ask for.
static const int TEMPLATE_MAX_BARS = 9999;

// What a `ZN|\n` rest-fill costs, for the context arithmetic below.
static const int TEMPLATE_REST_TOKENS = 6;

// The emitted score is re-tokenized in one go at the end, so BPE merges across
// the segment seams make it a few tokens shorter or longer than the sum of the
// pieces the abc phase counted. The context guard carries this much slack.
static const int TEMPLATE_SEAM_MARGIN = 64;

enum TplKind { TPL_GIVEN, TPL_PRIMER, TPL_HOLE, TPL_CONTINUE, TPL_CHORDS };

// What a forced stretch of score counts for: score text (fed and emitted), a
// primer (fed only), or the out-of-order feed of a `%%yue2-chords` directive,
// which is rolled back before anything is emitted.
enum GiveKind { GIVE_SCORE, GIVE_PRIMER, GIVE_SCRATCH };

// The `M:` and `L:` a bar's length is measured against, with ABC's defaults.
struct Meter
{
	int m_num = 4;
	int m_den = 4;
	int l_num = 1;
	int l_den = 8;
};

struct TplSeg
{
	TplKind     kind = TPL_GIVEN;
	std::string text;            // given / primer: whole lines, newlines kept
	                             // chords: the two lines below it (`B`), newlines kept
	std::string head;            // chords: the voice header line above it (`A`)
	int         bars = 0;        // hole / chords: the bar count its line must have
	std::string above;           // hole / chords: the body line the budget scales from
	Meter       meter;           // hole / chords: the meter in force where it sits
};

struct BarCount
{
	int bars      = 0;   // bars holding at least one note or rest
	int offlength = 0;   // of those, how many do not hold M:/L: note units
};

static long long gcd_ll(long long a, long long b)
{
	while (b != 0)
	{
		const long long t = a % b;
		a = b;
		b = t;
	}
	return a < 0 ? -a : a;
}

static void add_frac(long long & num, long long & den, long long n, long long d)
{
	num = num * d + n * den;
	den = den * d;
	const long long g = gcd_ll(num, den);
	if (g > 1)
	{
		num /= g;
		den /= g;
	}
}

static bool is_note_letter(char c)
{
	return (c >= 'a' && c <= 'g') || (c >= 'A' && c <= 'G');
}

// A body line is a line that carries music: not a comment, and not a `K:`-style
// field — which is also what rules out `w:` lyrics and the `V:` voice switches.
static bool is_body_line(const std::string & line)
{
	const size_t i = line.find_first_not_of(" \t\r");
	if (i == std::string::npos || line[i] == '%')
	{
		return false;
	}
	return !(i + 1 < line.size() && isalpha((unsigned char) line[i]) && line[i + 1] == ':');
}

// Removes what the bar counter must not see: `"..."` chord symbols, `!...!`
// decorations and `[K:...]` inline fields — the letter-colon test is what keeps
// a `[CEG]` chord. An unquoted `%` ends the line: the rest is a comment.
//
// Both delimiters are only honoured when they close. An unpaired `"` or `!`
// used to swallow everything after it, which silently turned a bar into none.
// A decoration is a short word, so `!` closes before the next space or bar line
// or it is not a decoration at all.
static std::string strip_line(const std::string & line)
{
	std::string out;
	for (size_t i = 0; i < line.size(); i++)
	{
		const char c = line[i];
		if (c == '%')
		{
			break;
		}
		if (c == '"')
		{
			const size_t end = line.find('"', i + 1);
			if (end != std::string::npos)
			{
				i = end;
				continue;
			}
		} else if (c == '!') {
			const size_t end = line.find_first_of("! |", i + 1);
			if (end != std::string::npos && line[end] == '!')
			{
				i = end;
				continue;
			}
		} else if (c == '[' && i + 2 < line.size() &&
		           isalpha((unsigned char) line[i + 1]) && line[i + 2] == ':') {
			const size_t end = line.find(']', i + 1);
			if (end != std::string::npos)
			{
				i = end;
				continue;
			}
		}
		out.push_back(c);
	}
	return out;
}

// The length of the bar line at `i`, or 0. `|`, `||`, `|]`, `[|`, `|:`, `:|` and
// `::` each count as one bar line (SPEC_TEMPLATE §4).
static size_t bar_line_at(const std::string & s, size_t i)
{
	const char c = s[i];
	const char d = i + 1 < s.size() ? s[i + 1] : '\0';
	if (c == '|')
	{
		return d == '|' || d == ']' || d == ':' ? 2 : 1;
	}
	if ((c == '[' || c == ':') && d == '|')
	{
		return 2;
	}
	return c == ':' && d == ':' ? 2 : 0;
}

// ABC note length after the pitch: `2`, `/2`, `/`, `//`, `3/2`.
static void parse_length(const std::string & s, size_t & i, long long & num, long long & den)
{
	num = 1;
	den = 1;
	long long v   = 0;
	bool      got = false;
	while (i < s.size() && isdigit((unsigned char) s[i]) && v < 100000)
	{
		v   = v * 10 + (s[i++] - '0');
		got = true;
	}
	if (got && v > 0)
	{
		num = v;
	}
	if (i < s.size() && s[i] == '/')
	{
		long long halves = 0;
		while (i < s.size() && s[i] == '/')
		{
			i++;
			halves++;
		}
		long long d  = 0;
		bool      gd = false;
		while (i < s.size() && isdigit((unsigned char) s[i]) && d < 100000)
		{
			d  = d * 10 + (s[i++] - '0');
			gd = true;
		}
		den = gd && d > 0 ? d : (1LL << std::min<long long>(halves, 20));
	}
}

// SPEC_TEMPLATE §4: a bar is a maximal run between bar lines that holds at least
// one note or rest; `Z` is one bar and `Zn` is n; a trailing run with no closing
// bar line still counts. Bar *length* is measured beside the count but is never
// a reason to reject a line — the reference model's own scores are not always
// exact and the semantic stage follows the score loosely.
//
// Broken rhythm (`a>b`) needs no handling: it moves duration from one note to
// its neighbour and leaves the bar's total alone.
static BarCount count_bars(const std::string & line, const Meter & m)
{
	const std::string s = strip_line(line);

	// note units per bar, as a multiple of L:
	const long long tgt_num = (long long) m.m_num * m.l_den;
	const long long tgt_den = (long long) m.m_den * m.l_num;

	BarCount  out;
	long long num      = 0;   // units of the bar being scanned
	long long den      = 1;
	int       extra    = 0;   // bars a multi-measure rest adds to it
	bool      sounded  = false;
	int       tup_left = 0;
	long long tup_num  = 1;
	long long tup_den  = 1;

	auto take_note = [&](size_t & j)
	{
		long long n = 1;
		long long d = 1;
		parse_length(s, j, n, d);
		if (tup_left > 0)
		{
			n *= tup_num;
			d *= tup_den;
			tup_left--;
		}
		add_frac(num, den, n, d);
		sounded = true;
	};

	auto close_bar = [&]()
	{
		if (!sounded)
		{
			return;
		}
		out.bars += 1 + extra;
		if (extra == 0 && num != 0 && num * tgt_den != tgt_num * den)
		{
			out.offlength++;
		}
		num     = 0;
		den     = 1;
		extra   = 0;
		sounded = false;
	};

	size_t i = 0;
	while (i < s.size())
	{
		const size_t bl = bar_line_at(s, i);
		if (bl > 0)
		{
			close_bar();
			i += bl;
			continue;
		}
		const char c = s[i];
		if (c == '{')
		{
			// grace notes carry no time of their own
			const size_t end = s.find('}', i + 1);
			i = end == std::string::npos ? s.size() : end + 1;
			continue;
		}
		if (c == '(' && i + 1 < s.size() && isdigit((unsigned char) s[i + 1]))
		{
			// (p[:q[:r]] — p notes in the time of q, over the next r notes.
			long long v[3] = {0, 0, 0};
			int       n    = 0;
			i++;
			while (true)
			{
				while (i < s.size() && isdigit((unsigned char) s[i]) && v[n] < 1000)
				{
					v[n] = v[n] * 10 + (s[i++] - '0');
				}
				if (n == 2 || i >= s.size() || s[i] != ':')
				{
					break;
				}
				i++;
				n++;
			}
			if (v[0] > 0)
			{
				tup_den  = v[0];
				tup_num  = v[1] > 0 ? v[1] : (v[0] == 2 || v[0] == 4 || v[0] == 8 ? 3 : 2);
				tup_left = (int) (v[2] > 0 ? v[2] : v[0]);
			}
			continue;
		}
		if (c == 'Z')
		{
			i++;
			long long n   = 0;
			bool      got = false;
			while (i < s.size() && isdigit((unsigned char) s[i]) && n < 100000)
			{
				n   = n * 10 + (s[i++] - '0');
				got = true;
			}
			extra  += (int) ((got && n > 0 ? n : 1) - 1);
			sounded = true;
			continue;
		}
		if (c == '[')
		{
			// `[1` / `[2` are repeat endings, not chords, and the bars inside
			// them are bars. A real chord closes inside its own bar; a `[` that
			// does not is an ordinary character, so an unclosed one cannot
			// swallow the bar lines after it.
			if (i + 1 < s.size() && isdigit((unsigned char) s[i + 1]))
			{
				i++;
				while (i < s.size() && isdigit((unsigned char) s[i]))
				{
					i++;
				}
				continue;
			}
			const size_t end = s.find_first_of("]|", i + 1);
			if (end == std::string::npos || s[end] != ']')
			{
				i++;
				continue;
			}
			// a chord sounds as one note; its length follows the `]`
			i = end + 1;
			take_note(i);
			continue;
		}
		if (is_note_letter(c) || c == 'z' || c == 'x')
		{
			i++;
			while (i < s.size() && (s[i] == ',' || s[i] == '\''))
			{
				i++;
			}
			take_note(i);
			continue;
		}
		i++;
	}
	close_bar();
	return out;
}

// `M:` and `L:` as a header field line gives them. `C` is 4/4 and `C|` is 2/2.
static void read_meter_field(const std::string & line, Meter & m)
{
	std::string v = line.substr(2);
	const size_t a = v.find_first_not_of(" \t");
	const size_t b = v.find_last_not_of(" \t\r");
	v = a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
	if (v.empty())
	{
		return;
	}
	int num = 0;
	int den = 0;
	if (v[0] == 'C')
	{
		num = v.size() > 1 && v[1] == '|' ? 2 : 4;
		den = num;
	} else if (sscanf(v.c_str(), "%d/%d", &num, &den) != 2 || num <= 0 || den <= 0) {
		return;
	}
	if (line[0] == 'M')
	{
		m.m_num = num;
		m.m_den = den;
	} else {
		m.l_num = num;
		m.l_den = den;
	}
}

static bool starts_with(const std::string & s, const char * prefix)
{
	const size_t n = strlen(prefix);
	return s.size() >= n && s.compare(0, n, prefix) == 0;
}

// `%%yue2-gen` and `%%yue2-chords` take nothing, or `bars=N` with real
// whitespace between the two — `%%yue2-genbars=3` is a typo, not a directive
// with an argument. `bars` comes back 0 when the directive carries no argument
// (the caller's default applies); the return is "" or the reason it is not one.
static std::string parse_bars_arg(const std::string & trimmed, const char * name, size_t lineno,
	int & bars)
{
	bars = 0;

	const std::string args = trimmed.substr(strlen(name));
	if (!args.empty() && args.find_first_of(" \t") != 0)
	{
		return strf("line %zu: unknown directive \"%s\"", lineno, trimmed.c_str());
	}
	const size_t arg = args.find_first_not_of(" \t");
	if (arg == std::string::npos)
	{
		return "";
	}
	const std::string value = args.substr(arg);
	if (!starts_with(value, "bars="))
	{
		return strf("line %zu: %s takes nothing but bars=N (got \"%s\")",
		            lineno, name, value.c_str());
	}
	// strtol, not sscanf: %d on an overflowing literal is undefined.
	errno = 0;
	const char * digits = value.c_str() + strlen("bars=");
	char *       endp   = nullptr;
	const long   n      = strtol(digits, &endp, 10);
	if (errno != 0 || endp == digits || *endp != '\0')
	{
		return strf("line %zu: %s takes nothing but bars=N (got \"%s\")",
		            lineno, name, value.c_str());
	}
	if (n < 1 || n > TEMPLATE_MAX_BARS)
	{
		return strf("line %zu: bars=%ld — a hole holds between 1 and %d bars",
		            lineno, n, TEMPLATE_MAX_BARS);
	}
	bars = (int) n;
	return "";
}

// A line as the *tests* see it. What is fed to the model is always the line
// exactly as the template wrote it, `\r` and all.
static std::string trim_line(const std::string & line)
{
	const size_t head = line.find_first_not_of(" \t");
	const size_t tail = line.find_last_not_of(" \t\r");
	return head == std::string::npos ? std::string() : line.substr(head, tail - head + 1);
}

// A `V:` voice header, the only line a `%%yue2-chords` directive may sit between.
static bool is_voice_line(const std::string & trimmed)
{
	return starts_with(trimmed, "V:");
}

// Splits a template into the ordered segments the abc phase walks. Returns "" or
// the reason the template is not usable; every request error of SPEC_TEMPLATE §2
// is reported here, before anything loads.
static std::string parse_template(const std::string & text, std::vector<TplSeg> & segs)
{
	segs.clear();

	Meter       meter;
	std::string above;        // nearest given body line above the next hole
	bool        in_primer = false;
	size_t      lineno    = 0;
	size_t      i         = 0;
	while (i < text.size())
	{
		const size_t      nl    = text.find('\n', i);
		const size_t      stop  = nl == std::string::npos ? text.size() : nl;
		const std::string whole = text.substr(i, (nl == std::string::npos ? text.size() : nl + 1) - i);
		std::string       line  = text.substr(i, stop - i);
		i = nl == std::string::npos ? text.size() : nl + 1;
		lineno++;

		const std::string trimmed = trim_line(line);
		// Past %%yue2-continue the score is the model's: nothing may follow it but
		// blank lines (§2).
		if (!segs.empty() && segs.back().kind == TPL_CONTINUE)
		{
			if (!trimmed.empty())
			{
				return strf("line %zu: %%%%yue2-continue must be the template's last line", lineno);
			}
			continue;
		}
		if (!starts_with(trimmed, "%%yue2-"))
		{
			// A primer is context, not score: it may not decide a hole's meter or
			// its default bar count (SPEC_TEMPLATE §2 — the block is dropped, so a
			// default taken from it would refer to a line the score never holds).
			if (!in_primer)
			{
				if (starts_with(trimmed, "M:") || starts_with(trimmed, "L:"))
				{
					read_meter_field(trimmed, meter);
				}
				if (is_body_line(trimmed))
				{
					above = trimmed;
				}
			}
			const TplKind kind = in_primer ? TPL_PRIMER : TPL_GIVEN;
			if (segs.empty() || segs.back().kind != kind)
			{
				TplSeg seg;
				seg.kind = kind;
				segs.push_back(seg);
			}
			segs.back().text += whole;
			continue;
		}

		if (trimmed == "%%yue2-primer-begin")
		{
			if (in_primer)
			{
				return strf("line %zu: %%%%yue2-primer-begin inside a primer block", lineno);
			}
			in_primer = true;
			continue;
		}
		if (trimmed == "%%yue2-primer-end")
		{
			if (!in_primer)
			{
				return strf("line %zu: %%%%yue2-primer-end without a primer block", lineno);
			}
			in_primer = false;
			continue;
		}
		if (trimmed == "%%yue2-continue")
		{
			if (in_primer)
			{
				return strf("line %zu: %%%%yue2-continue inside a primer block", lineno);
			}
			TplSeg seg;
			seg.kind = TPL_CONTINUE;
			segs.push_back(seg);
			continue;
		}
		if (starts_with(trimmed, "%%yue2-chords"))
		{
			if (in_primer)
			{
				return strf("line %zu: %%%%yue2-chords inside a primer block — a primer "
				            "holds given lines only", lineno);
			}

			TplSeg seg;
			seg.kind  = TPL_CHORDS;
			seg.meter = meter;
			const std::string err = parse_bars_arg(trimmed, "%%yue2-chords", lineno, seg.bars);
			if (!err.empty())
			{
				return err;
			}

			// `A`, the voice header above it, is peeled off the given segment it
			// ends: the decode checkpoints there, and that has to be a segment
			// boundary (SPEC_TEMPLATE §2).
			if (segs.empty() || segs.back().kind != TPL_GIVEN)
			{
				return strf("line %zu: %%%%yue2-chords must sit directly under a given "
				            "V: voice header", lineno);
			}
			std::string & prev  = segs.back().text;
			size_t        start = 0;
			if (prev.size() >= 2)
			{
				const size_t cut = prev.rfind('\n', prev.size() - 2);
				start = cut == std::string::npos ? 0 : cut + 1;
			}
			seg.head = prev.substr(start);
			if (!is_voice_line(trim_line(seg.head)))
			{
				return strf("line %zu: %%%%yue2-chords must sit directly under a given "
				            "V: voice header (the line above it is \"%s\")",
				            lineno, trim_line(seg.head).c_str());
			}
			prev.erase(start);
			if (prev.empty())
			{
				segs.pop_back();
			}

			// `B`, the other voice's header and its body line, immediately below.
			std::string body;
			for (int k = 0; k < 2; k++)
			{
				if (i >= text.size())
				{
					return strf("line %zu: %%%%yue2-chords needs the other voice's V: "
					            "header and its body line below it", lineno);
				}
				const size_t      bnl   = text.find('\n', i);
				const size_t      bstop = bnl == std::string::npos ? text.size() : bnl;
				const std::string bwhole = text.substr(i, (bnl == std::string::npos
				                                           ? text.size() : bnl + 1) - i);
				const std::string bline = trim_line(text.substr(i, bstop - i));
				i = bnl == std::string::npos ? text.size() : bnl + 1;
				lineno++;
				if (starts_with(bline, "%%yue2-"))
				{
					return strf("line %zu: %%%%yue2-chords needs the other voice's V: "
					            "header and its body line below it, not \"%s\"",
					            lineno, bline.c_str());
				}
				if (k == 0 ? !is_voice_line(bline) : !is_body_line(bline))
				{
					return strf("line %zu: %%%%yue2-chords needs the other voice's V: "
					            "header and its body line below it (got \"%s\")",
					            lineno, bline.c_str());
				}
				seg.text += bwhole;
				if (k == 1)
				{
					body = bline;
				}
			}
			seg.above = body;
			above     = body;
			if (seg.bars == 0)
			{
				seg.bars = count_bars(body, meter).bars;
				if (seg.bars < 1)
				{
					return strf("line %zu: the body line below holds no bars; give "
					            "%%%%yue2-chords an explicit bars=N", lineno);
				}
			}
			segs.push_back(seg);
			continue;
		}
		if (!starts_with(trimmed, "%%yue2-gen"))
		{
			return strf("line %zu: unknown directive \"%s\"", lineno, trimmed.c_str());
		}
		if (in_primer)
		{
			return strf("line %zu: %%%%yue2-gen inside a primer block — a primer holds "
			            "given lines only", lineno);
		}

		TplSeg seg;
		seg.kind  = TPL_HOLE;
		seg.above = above;
		seg.meter = meter;

		const std::string err = parse_bars_arg(trimmed, "%%yue2-gen", lineno, seg.bars);
		if (!err.empty())
		{
			return err;
		}
		if (seg.bars == 0)
		{
			if (above.empty())
			{
				return strf("line %zu: %%%%yue2-gen needs bars=N — there is no body line "
				            "above it to take the bar count from (a primer block's lines "
				            "do not count: they never reach the score)", lineno);
			}
			seg.bars = count_bars(above, meter).bars;
			if (seg.bars < 1)
			{
				return strf("line %zu: the body line above holds no bars; give %%%%yue2-gen "
				            "an explicit bars=N", lineno);
			}
		}
		segs.push_back(seg);
	}

	if (in_primer)
	{
		return "the last %%yue2-primer-begin was never closed by %%yue2-primer-end";
	}
	return "";
}

// --------------------------------------------------------- score sections ---

// SPEC_SECTIONS.md. A planner-written score marks its sections with a comment
// line of its own — `% intro`, `% verse`, `% chorus`, `% interlude`, `% bridge`,
// `% outro` — and a "sections" entry names one of them. What the semantic phase
// needs is *when* that label is, so the clock below walks the score line by line
// and keeps the bar count and the elapsed seconds of the `V: Vocal` voice, which
// is the one that carries the song's bars.
//
// Pure text work, like the template parser above it: the same clock runs over a
// score as the model writes it (Runner::sections_token) and over a score the
// request gave in "abc" (sections_locate), and it is table-tested on its own
// (tests/guidance.cpp).
struct ScoreClock
{
	Meter     meter;                  // M:/L: in force, inline [M:…] included
	long long q_num   = 1;            // the note Q: counts, …
	long long q_den   = 4;            // … as a fraction of a whole note, …
	double    q_bpm   = 120;          // … and how many of them go by in a minute
	bool      vocal   = true;         // the voice the body lines belong to
	int       line    = 0;            // completed lines, the last one included
	int       bars    = 0;            // bars of Vocal completed
	double    seconds = 0;            // where the next bar starts
	std::unordered_map<std::string, int> seen;   // label lines, by name
};

// Seconds of one bar of `m` at the tempo the clock carries: a bar is m_num/m_den
// of a whole note, and `Q:` says how many q_num/q_den notes go by in a minute.
static double bar_seconds(const ScoreClock & c, const Meter & m)
{
	if (c.q_bpm <= 0 || c.q_num <= 0 || c.q_den <= 0 || m.m_den <= 0)
	{
		return 0;
	}
	return 60.0 * ((double) m.m_num * c.q_den) / ((double) m.m_den * (double) c.q_num) / c.q_bpm;
}

// `Q:` as the planner writes it (`Q:1/4=120`), plus ABC's other forms: a bare
// `Q:120` counts unit note lengths, and a quoted label is not a tempo.
static void read_tempo_field(const std::string & line, ScoreClock & c)
{
	std::string  v = line.substr(2);
	const size_t q = v.find('"');
	if (q != std::string::npos)
	{
		v.erase(q);
	}
	long long num = 0;
	long long den = 0;
	double    bpm = 0;
	if (sscanf(v.c_str(), " %lld/%lld = %lf", &num, &den, &bpm) == 3)
	{
		if (num > 0 && den > 0 && bpm > 0)
		{
			c.q_num = num;
			c.q_den = den;
			c.q_bpm = bpm;
		}
		return;
	}
	if (sscanf(v.c_str(), " %lf", &bpm) == 1 && bpm > 0)
	{
		c.q_num = c.meter.l_num;
		c.q_den = c.meter.l_den;
		c.q_bpm = bpm;
	}
}

// The bars one body line holds and what they take, honouring an inline `[M:…]`:
// the line is cut at the bar line the change follows and each piece counted
// under the meter in force over it. A change that does not sit at a bar line
// takes effect at the next one — the bar it is inside keeps the length it
// started with — which is also what keeps a cut from counting one bar twice.
static int body_line_clock(const std::string & line, ScoreClock & c, double & seconds)
{
	int    bars  = 0;
	size_t start = 0;   // the piece being measured starts here
	size_t open  = 0;   // and the bar inside it starts here
	seconds = 0;

	auto take = [&](const std::string & piece)
	{
		const BarCount bc = count_bars(piece, c.meter);
		bars    += bc.bars;
		seconds += bc.bars * bar_seconds(c, c.meter);
	};

	for (size_t i = 0; i < line.size(); i++)
	{
		const char ch = line[i];
		if (ch == '%')
		{
			break;
		}
		// The two delimiters count_bars honours, so that a `|` inside a chord
		// symbol or a decoration is not taken for a bar line here either.
		if (ch == '"' || ch == '!')
		{
			const size_t end = ch == '"' ? line.find('"', i + 1)
			                             : line.find_first_of("! |", i + 1);
			if (end != std::string::npos && line[end] == ch)
			{
				i = end;
			}
			continue;
		}
		const size_t bl = bar_line_at(line, i);
		if (bl > 0)
		{
			i    += bl - 1;
			open  = i + 1;
			continue;
		}
		if (ch != '[' || i + 3 >= line.size() || line[i + 1] != 'M' || line[i + 2] != ':')
		{
			continue;
		}
		const size_t end = line.find(']', i + 3);
		if (end == std::string::npos)
		{
			continue;
		}
		take(line.substr(start, open - start));
		start = open;
		read_meter_field("M:" + line.substr(i + 3, end - i - 3), c.meter);
		i = end;
	}
	take(line.substr(start));
	return bars;
}

// One completed score line, without its newline. Returns the label it is —
// `% verse` comes back as "verse" — or "" for anything else.
static std::string clock_line(ScoreClock & c, const std::string & line)
{
	c.line++;
	const size_t i = line.find_first_not_of(" \t\r");
	if (i == std::string::npos)
	{
		return "";
	}
	if (line[i] == '%')
	{
		// `%%…` is a directive, not a label, and a label is a name and nothing
		// else — the vocabulary of §6's survey.
		if (i + 1 < line.size() && line[i + 1] == '%')
		{
			return "";
		}
		const size_t a = line.find_first_not_of(" \t", i + 1);
		const size_t b = line.find_last_not_of(" \t\r");
		if (a == std::string::npos || a > b)
		{
			return "";
		}
		const std::string name = line.substr(a, b - a + 1);
		c.seen[name]++;
		return name;
	}
	if (i + 1 < line.size() && isalpha((unsigned char) line[i]) && line[i + 1] == ':')
	{
		// A field line. The ones that move the clock are the meter, the unit note
		// length, the tempo and the voice switch; a `V:` with fields behind it is
		// the header's own definition and names the same voice.
		if (line[i] == 'M' || line[i] == 'L')
		{
			read_meter_field(line.substr(i), c.meter);
		} else if (line[i] == 'Q') {
			read_tempo_field(line.substr(i), c);
		} else if (line[i] == 'V') {
			const size_t a = line.find_first_not_of(" \t", i + 2);
			const size_t b = a == std::string::npos ? a : line.find_first_of(" \t\r", a);
			c.vocal = a != std::string::npos &&
			          line.substr(a, b == std::string::npos ? b : b - a) == "Vocal";
		}
		return "";
	}
	if (!c.vocal)
	{
		return "";
	}
	double seconds = 0;
	c.bars    += body_line_clock(line, c, seconds);
	c.seconds += seconds;
	return "";
}

// Walks a score that is already written and fills in every entry it finds. The
// entries are matched in the order the request lists them and `nth` is counted
// over every label line of that name from the top of the score, which is what
// makes an entry naming a label *before* the one already matched simply go
// unfound rather than reorder the song (§2).
static void sections_locate(const std::string & score, std::vector<SectionEntry> & es)
{
	ScoreClock c;
	size_t     next = 0;
	size_t     from = 0;
	while (next < es.size())
	{
		const size_t      nl   = score.find('\n', from);
		const std::string name = clock_line(c, score.substr(from, nl == std::string::npos
		                                                          ? nl : nl - from));
		if (name == es[next].section && c.seen[name] == es[next].nth)
		{
			es[next].found   = true;
			es[next].line    = c.line;
			es[next].bar     = c.bars;
			es[next].seconds = c.seconds;
			next++;
		}
		if (nl == std::string::npos)
		{
			break;
		}
		from = nl + 1;
	}
}

// The semantic step each found entry swaps at: `lead_frames` before the bar line
// it names, never before frame 1 and never at or before the entry ahead of it
// (§4).
static void sections_frames(std::vector<SectionEntry> & es)
{
	long long prev = 0;
	for (size_t i = 0; i < es.size(); i++)
	{
		if (!es[i].found)
		{
			continue;
		}
		const long long f = llround(es[i].seconds * SEMANTIC_FPS) - es[i].lead;
		es[i].frame = (int) std::max(std::max(f, (long long) 1), prev + 1);
		prev        = es[i].frame;
	}
}

// SPEC_SECTIONS §4: `semantic_keep` combines, but an entry that lands inside the
// kept frames is the error SPEC_KEEP §2 makes of a guidance frame there. A keep
// request carries its score in "abc", so this is decided before the model loads.
static std::string sections_keep_check(const Request & r, const std::vector<SectionEntry> & es)
{
	if (!r.has_keep)
	{
		return "";
	}
	for (size_t i = 0; i < es.size(); i++)
	{
		if (es[i].found && es[i].frame < r.keep_frames)
		{
			return strf("\"sections\" entry %zu: %% %s %d lands on frame %d, inside the kept "
			            "%d frames", i + 1, es[i].section.c_str(), es[i].nth, es[i].frame,
			            r.keep_frames);
		}
	}
	return "";
}

// -------------------------------------------------------- style handover ---

// SPEC_HANDOVER.md. One score, one full take per style: at a cut the incoming
// style's renderer is forced through its OWN take up to x frames before the cut
// and then through the last x frames of the song so far, and samples on from
// there. Two takes of one score do not run at the same pace, so the incoming
// one is read at its own clock — `off` frames ahead of the song — and that
// offset is measurable from the tokens alone (§4).
//
// Everything in this block is pure integer work on the two streams: no model,
// no audio, and table-tested on its own (tests/handover.cpp).
struct OffsetFit
{
	int    offset    = 0;
	double z         = 0;   // how far the peak stands out of the 201 rates
	int    hits      = 0;   // frames that agree at it, not the share of them
	bool   confident = false;
	int    measured  = 0;   // the peak itself, which a fit that is not believed drops
};

// The frames in [lo, cut) at which the song and the take agree when the take is
// read `k` frames ahead, and how many were compared at all: frames the take
// does not reach are skipped rather than counted as a miss.
static int offset_hits(const std::vector<int32_t> & song, const std::vector<int32_t> & take,
	int lo, int cut, int k, int & counted)
{
	int hit = 0;
	counted = 0;
	for (int t = lo; t < cut; t++)
	{
		const long long u = (long long) t - k;
		if (u < 0 || u >= (long long) take.size())
		{
			continue;
		}
		counted++;
		hit += song[(size_t) t] == take[(size_t) u] ? 1 : 0;
	}
	return hit;
}

// The best lag over -HANDOVER_RANGE…HANDOVER_RANGE and how far it stands out:
// z = (max - mean) / std over every rate tried. Ties go to the smallest |k|,
// and at equal |k| to the negative one — a take that agrees at no lag at all
// agrees at lag 0, and a tie between +k and -k is read as the take running
// behind rather than ahead.
//
// z alone is not enough to believe a lag. Over a short window one coincidental
// hit at one lag and none at the others is a z of 14, so the peak also has to
// hold at least HANDOVER_MIN_HITS frames that agree — an absolute count, which
// noise does not reach and two takes of one score pass several times over
// (2-3 % of a 750-frame window is ~20).
static OffsetFit offset_scan(const std::vector<int32_t> & song, const std::vector<int32_t> & take,
	int lo, int cut)
{
	std::vector<double> rate;
	std::vector<int>    hits;
	rate.reserve((size_t) (2 * HANDOVER_RANGE + 1));
	hits.reserve((size_t) (2 * HANDOVER_RANGE + 1));
	for (int k = -HANDOVER_RANGE; k <= HANDOVER_RANGE; k++)
	{
		int       counted = 0;
		const int hit     = offset_hits(song, take, lo, cut, k, counted);
		hits.push_back(hit);
		rate.push_back(counted > 0 ? (double) hit / (double) counted : 0.0);
	}

	size_t best = 0;
	for (size_t i = 1; i < rate.size(); i++)
	{
		const int k = (int) i - HANDOVER_RANGE;
		const int b = (int) best - HANDOVER_RANGE;
		if (rate[i] > rate[best] ||
		    (rate[i] == rate[best] && (std::abs(k) < std::abs(b) ||
		                               (std::abs(k) == std::abs(b) && k < b))))
		{
			best = i;
		}
	}

	double mean = 0;
	for (size_t i = 0; i < rate.size(); i++)
	{
		mean += rate[i];
	}
	mean /= (double) rate.size();
	double var = 0;
	for (size_t i = 0; i < rate.size(); i++)
	{
		var += (rate[i] - mean) * (rate[i] - mean);
	}
	var /= (double) rate.size();

	OffsetFit fit;
	fit.offset    = (int) best - HANDOVER_RANGE;
	fit.z         = var > 0 ? (rate[best] - mean) / std::sqrt(var) : 0;
	fit.hits      = hits[best];
	fit.confident = fit.z >= HANDOVER_MIN_Z && fit.hits >= HANDOVER_MIN_HITS;
	fit.measured  = fit.offset;
	return fit;
}

// §4: the last HANDOVER_SPAN frames before the cut, and the whole song from
// HANDOVER_LO on when that window is too flat to believe. A fit that is still
// flat is dropped for offset 0 — the song so far is re-indexed onto its own
// clock either way (§5), so 0 is the neutral guess and a lag that was measured
// out of noise is worse than none. The peak stays in `measured` and the entry
// says it was not believed.
static OffsetFit handover_offset(const std::vector<int32_t> & song,
	const std::vector<int32_t> & take, int cut)
{
	const int lo  = std::max(HANDOVER_LO, cut - HANDOVER_SPAN);
	OffsetFit fit = offset_scan(song, take, lo, cut);
	if (!fit.confident && lo > HANDOVER_LO)
	{
		fit = offset_scan(song, take, HANDOVER_LO, cut);
	}
	if (!fit.confident)
	{
		fit.offset = 0;
	}
	return fit;
}

// The intrusion X of §2, in frames.
static int handover_x(double seconds)
{
	return (int) llround(seconds * SEMANTIC_FPS);
}

// The forced history of one leg (§5): the incoming take up to x frames before
// the cut, in its own clock, then the last x frames of the song so far. `cut -
// off` codes all told, which is the leg's "semantic_keep" frame count. Returns
// "" or the reason this entry cannot be played.
static std::string handover_keep(const std::vector<int32_t> & song,
	const std::vector<int32_t> & take, int cut, int x, int off, std::vector<int32_t> & out)
{
	if ((size_t) cut > song.size())
	{
		return strf("the song ended at frame %zu, before the cut at %d", song.size(), cut);
	}
	if (cut - x - off < 1 || cut - x < 1)
	{
		return strf("a cut at frame %d with a %d-frame intrusion at offset %+d leaves "
		            "nothing before it", cut, x, off);
	}
	if ((size_t) (cut - off) > take.size())
	{
		return strf("frame %d of the take (cut %d, offset %+d) is past its %zu frames",
		            cut - off, cut, off, take.size());
	}
	out.assign(take.begin(), take.begin() + (cut - x - off));
	out.insert(out.end(), song.begin() + (cut - x), song.begin() + cut);
	return "";
}

// Where each entry cuts (§2). A "frame" is the base take's own timeline and
// needs no score — nor does an "at", which became a frame as it was parsed; a
// label goes through stage 8's resolver and clock against the base score, the
// entries that name one in the order they are given.
static void handover_locate(const std::string & score, std::vector<HandoverEntry> & es)
{
	std::vector<SectionEntry> secs;
	std::vector<size_t>       which;
	for (size_t i = 0; i < es.size(); i++)
	{
		if (es[i].has_frame)
		{
			es[i].found = true;
			es[i].cut   = es[i].frame;
			continue;
		}
		SectionEntry s;
		s.section = es[i].section;
		s.nth     = es[i].nth;
		s.lead    = es[i].lead;
		secs.push_back(s);
		which.push_back(i);
	}
	sections_locate(score, secs);
	sections_frames(secs);
	for (size_t k = 0; k < secs.size(); k++)
	{
		HandoverEntry & e = es[which[k]];
		e.found       = secs[k].found;
		e.line        = secs[k].line;
		e.bar         = secs[k].bar;
		e.bar_seconds = secs[k].seconds;
		e.cut         = secs[k].frame;
	}
}

// How an error names where an entry cuts (§2): an entry that wrote an "at" is
// named by that text — the time the reader typed — with the frame it came to
// beside it; anything else by the frame alone.
static std::string handover_where(const HandoverEntry & h, int frame)
{
	if (!h.has_at)
	{
		return strf("frame %d", frame);
	}
	const std::string at = h.at_json.is_string() ? h.at_json.get<std::string>()
	                                             : h.at_json.dump();
	return strf("%s (frame %d)", at.c_str(), frame);
}

// §2: "Cuts must be strictly increasing; order in the array = order in the
// song." Across the two forms only the resolved frames can say so, which is why
// this is checked once the score has been walked.
static std::string handover_cuts_check(const std::vector<HandoverEntry> & es)
{
	int         prev = 0;
	size_t      at   = 0;
	std::string where;
	for (size_t i = 0; i < es.size(); i++)
	{
		if (!es[i].found)
		{
			continue;
		}
		if (prev > 0 && es[i].cut <= prev)
		{
			return strf("\"handover\" entry %zu cuts at %s, which is not after entry "
			            "%zu's %s — entries are in the order the song plays them",
			            i + 1, handover_where(es[i], es[i].cut).c_str(), at, where.c_str());
		}
		prev  = es[i].cut;
		at    = i + 1;
		where = handover_where(es[i], es[i].cut);
	}
	return "";
}

// ------------------------------------------------------------- generation ---

// Sampling order for the candidate list: score descending, id ascending on ties.
// Hoisted out of sample_step so the bounded heap, the frozen reference's
// nth_element and the final sort share one definition. As a heap comparator it
// makes a *min*-heap: the front is the smallest score held.
static bool by_score_desc(const std::pair<float, int> & a, const std::pair<float, int> & b)
{
	return a.first > b.first;
}

static bool by_score_then_id(const std::pair<float, int> & a, const std::pair<float, int> & b)
{
	return a.first != b.first ? a.first > b.first : a.second < b.second;
}

// sample_step_ref: the stage-5 sampler, frozen. --verify-sampler and
// yue2-sampler-diff run it beside the one below (SPEC_SAMPLER.md §4).
#include "sampler_ref.hpp"

// Scratch for one sequence's sampler. Every buffer is bounded by top_k or
// penalty_window (a few hundred entries), never by n_vocab — the whole point of
// SPEC_SAMPLER §3 is that a decode step allocates nothing.
struct SampleScratch
{
	std::vector<std::pair<llama_token, float>> pen;    // penalised ids, ascending by id
	std::vector<std::pair<float, int>>         heap;   // <= top_k, min-heap by score
	std::vector<std::pair<float, int>>         cand;   // the top-k survivors
	std::vector<double>                        prob;
};

// Visits every *allowed* id in ascending order and hands the callback its
// penalised — still untempered — score. `seg` is the allowed set as up to two
// ascending, disjoint id ranges; `pen` is sorted by id, so one cursor walks it
// alongside the ranges instead of a lookup per id.
template <typename F>
static void scan_allowed(const float * logits, const int seg[2][2],
	const std::vector<std::pair<llama_token, float>> & pen, F && f)
{
	size_t pi = 0;
	for (int g = 0; g < 2; g++)
	{
		for (int i = seg[g][0]; i < seg[g][1]; i++)
		{
			while (pi < pen.size() && pen[pi].first < i)
			{
				pi++;
			}
			f(i, pi < pen.size() && pen[pi].first == i ? pen[pi].second : logits[i]);
		}
	}
}

// One sampling step, mirroring sampling.distribution() exactly: allowed mask →
// min_tokens end mask → window penalty → temperature → top-k → top-p → sample.
//
// NOTE: do not "simplify" this into llama.cpp's sampler chain. llama's
// repetition penalty is a flat logit<=0 ? *p : /p once per windowed token, not
// YuE2's penalty**freq, and its top-p breaks on cum_sum >= p rather than
// cumsum - p > top_p — both differ from the reference at the boundary.
//
// Sparse since stage 6 (SPEC_SAMPLER.md). The mask is gone: the allowed set is
// at most two id ranges, so it is iterated rather than written. The penalty
// touches at most penalty_window ids, so it is a sorted side list rather than a
// mutated copy of 184 704 floats. Only the top_k largest can ever be drawn, so
// the k-th largest is found with a bounded min-heap rather than nth_element over
// every finite entry. Everything from the sort down is the stage-5 code
// operating on the same small candidate vector.
//
// Why the threshold is still taken on *tempered* values (SPEC_SAMPLER §2): x/t
// for t > 0 is monotone, so the k-th largest could be selected untempered and
// tempered afterwards — but distinct x can round to the same float after the
// division, which would add a tied survivor at the boundary that the stage-5
// code kept. Rather than bound that collapse, both passes divide: the second
// pass is a compare per allowed id and the divide is one instruction beside it,
// and for the semantic phase — the long one, and the only one where temperature
// is 1 — there is no divide at all.
static llama_token sample_step(const float * logits, int n_vocab, const Sampling & s,
	const std::vector<llama_token> & history, int step,
	bool phase_abc, bool legacy_off, std::mt19937_64 & rng,
	SampleScratch * scratch = nullptr, bool mask_end = false)
{
	const float ninf = -std::numeric_limits<float>::infinity();
	const int   end  = phase_abc ? ABC_END : MUSIC_END;

	SampleScratch   local;
	SampleScratch & sc = scratch != nullptr ? *scratch : local;

	// The allowed set, ascending: abc is [0, EOD) then ABC_END, semantic is
	// MUSIC_END then the codec block. `end` is masked out below min_tokens.
	int seg[2][2];
	if (phase_abc)
	{
		seg[0][0] = 0;
		seg[0][1] = std::min(EOD, n_vocab);
		seg[1][0] = std::min(end, n_vocab);
		seg[1][1] = std::min(end + 1, n_vocab);
	} else {
		seg[0][0] = std::min(end, n_vocab);
		seg[0][1] = std::min(end + 1, n_vocab);
		seg[1][0] = std::min(CODEC_OFFSET, n_vocab);
		seg[1][1] = std::min(CODEC_OFFSET + CODEC_SIZE, n_vocab);
	}
	// `mask_end` is the same mask kept up for a whole hole of a score template:
	// a hole writes one line, so it may never end the score (SPEC_TEMPLATE §3).
	if (step < s.min_tokens || mask_end)
	{
		const int e = phase_abc ? 1 : 0;
		seg[e][1] = seg[e][0];
	}

	// window penalty over the last penalty_window generated ids. Outside the
	// allowed set the stage-5 code penalised a -inf that stayed -inf, so those
	// ids are simply never visited here.
	sc.pen.clear();
	if (s.repetition_penalty != 1.0 && !history.empty())
	{
		const size_t from = history.size() > (size_t) s.penalty_window
		                    ? history.size() - (size_t) s.penalty_window : 0;
		std::unordered_map<llama_token, int> freq;
		for (size_t i = from; i < history.size(); i++)
		{
			freq[history[i]]++;
		}
		for (const auto & kv : freq)
		{
			if (kv.first < 0 || kv.first >= n_vocab)
			{
				continue;
			}
			const float alpha = (float) std::pow(s.repetition_penalty, (double) kv.second);
			const float v     = logits[kv.first];
			sc.pen.push_back(std::make_pair(kv.first, v < 0 ? v * alpha : v / alpha));
		}
		std::sort(sc.pen.begin(), sc.pen.end());
	}

	// greedy: argmax over the penalised scores, first index on a tie (torch
	// semantics). A fully masked step returns 0, as it did before.
	if (s.temperature == 0)
	{
		int   best = 0;
		float top  = ninf;
		scan_allowed(logits, seg, sc.pen, [&](int i, float v)
		{
			if (v > top)
			{
				top  = v;
				best = i;
			}
		});
		return (llama_token) best;
	}

	if (s.top_k < 1)
	{
		// The stage-5 code indexed cand.begin() + (top_k - 1) here; this is the
		// same contract stated instead of trusted. After the greedy return, which
		// never looked at top_k.
		die("top_k must be >= 1 (got %d)", s.top_k);
	}

	// float division, like torch's float32 tensor / float scalar
	const float t      = (float) s.temperature;
	const bool  temper = s.temperature != 1;

	// One scan: count the candidates (everything still finite) and keep the
	// top_k largest in a min-heap, whose front is then the k-th largest value.
	sc.heap.clear();
	size_t n_cand = 0;
	scan_allowed(logits, seg, sc.pen, [&](int i, float v)
	{
		const float x = temper ? v / t : v;
		if (!(x > ninf))
		{
			return;
		}
		n_cand++;
		if ((int) sc.heap.size() < s.top_k)
		{
			sc.heap.push_back(std::make_pair(x, i));
			std::push_heap(sc.heap.begin(), sc.heap.end(), by_score_desc);
		} else if (x > sc.heap.front().first) {
			std::pop_heap(sc.heap.begin(), sc.heap.end(), by_score_desc);
			sc.heap.back() = std::make_pair(x, i);
			std::push_heap(sc.heap.begin(), sc.heap.end(), by_score_desc);
		}
	});
	if (n_cand == 0)
	{
		die("every token was masked out (step %d)", step);
	}

	// top-k: keep everything >= the k-th largest value — ties included, so more
	// than k can survive. Below the boundary the heap already *is* the whole
	// candidate set and the second scan is skipped.
	sc.cand.clear();
	if (n_cand <= (size_t) s.top_k)
	{
		sc.cand = sc.heap;
	} else {
		const float threshold = sc.heap.front().first;
		sc.cand.reserve((size_t) s.top_k + 8);
		scan_allowed(logits, seg, sc.pen, [&](int i, float v)
		{
			const float x = temper ? v / t : v;
			if (x > ninf && !(x < threshold))
			{
				sc.cand.push_back(std::make_pair(x, i));
			}
		});
	}

	std::sort(sc.cand.begin(), sc.cand.end(), by_score_then_id);

	// softmax over the surviving scores
	sc.prob.assign(sc.cand.size(), 0.0);
	{
		const double top = sc.cand[0].first;
		double       sum = 0;
		for (size_t i = 0; i < sc.cand.size(); i++)
		{
			sc.prob[i] = std::exp((double) sc.cand[i].first - top);
			sum       += sc.prob[i];
		}
		for (size_t i = 0; i < sc.prob.size(); i++)
		{
			sc.prob[i] /= sum;
		}
	}

	// top-p: drop where cumsum - p > top_p, always keeping the head
	if (s.top_p < 1)
	{
		const size_t keep_head = legacy_off ? 3 : 1;
		double       cum       = 0;
		size_t       n         = sc.cand.size();
		for (size_t i = 0; i < sc.cand.size(); i++)
		{
			cum += sc.prob[i];
			if (i >= keep_head && cum - sc.prob[i] > s.top_p)
			{
				n = i;
				break;
			}
		}
		sc.cand.resize(n);
		sc.prob.resize(n);
		double sum = 0;
		for (size_t i = 0; i < sc.prob.size(); i++)
		{
			sum += sc.prob[i];
		}
		for (size_t i = 0; i < sc.prob.size(); i++)
		{
			sc.prob[i] /= sum;
		}
	}

	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	const double                           r = uniform(rng);
	double                                 c = 0;
	for (size_t i = 0; i < sc.prob.size(); i++)
	{
		c += sc.prob[i];
		if (r < c)
		{
			return (llama_token) sc.cand[i].second;
		}
	}
	return (llama_token) sc.cand.back().second;
}

// LLAMA_BUILD_COMMON is OFF, so this is common/common.cpp's common_batch_add.
// llama_batch_init leaves every member uninitialised (llama.h:954-960) and
// llama_batch_get_one pins seq_id to 0 and pos to NULL — neither can write to a
// slot other than 0, which is the whole point here.
static void batch_add(llama_batch & b, llama_token id, llama_pos pos, int slot, bool want_logits)
{
	const int i = b.n_tokens;
	b.token[i]     = id;
	b.pos[i]       = pos;
	b.n_seq_id[i]  = 1;
	b.seq_id[i][0] = (llama_seq_id) slot;
	b.logits[i]    = want_logits ? 1 : 0;
	b.n_tokens     = i + 1;
}

// Decodes `feed` into one slot in n_batch-sized chunks, so a prefill larger than
// the batch is a loop here instead of llama.cpp's GGML_ASSERT(n_tokens_all <=
// n_batch) abort. Only the last token asks for logits; the returned index is
// that token's position in the final batch, the only index
// llama_get_logits_ith will accept for it (llama.h:1037-1041).
static int decode_feed(llama_context * ctx, llama_batch & b,
	const std::vector<llama_token> & feed, int slot, llama_pos & pos, const char * what)
{
	const size_t n_batch = (size_t) llama_n_batch(ctx);
	size_t       off     = 0;
	int          last    = -1;
	while (off < feed.size())
	{
		const size_t n = std::min(feed.size() - off, n_batch);
		b.n_tokens = 0;
		for (size_t i = 0; i < n; i++)
		{
			batch_add(b, feed[off + i], pos++, slot, off + i + 1 == feed.size());
		}
		last = (int) n - 1;
		const int ret = llama_decode(ctx, b);
		if (ret != 0)
		{
			die("llama_decode returned %d on the %s prefix of slot %d", ret, what, slot);
		}
		off += n;
	}
	return last;
}

// ------------------------------------------------------------- artifacts ---

// ---- --guidance-trace: what the branches did to a step ----------------------
// Pure functions on one row of logits, so tests/guidance.cpp checks them on
// small arrays with no model and no device. They run only when the flag is on.

// One row of guidance_trace.npy. The columns are in README.md.
static const int TRACE_COLUMNS = 8;

// softmax over `n` logits, in double and with the row's maximum subtracted.
// Returns log(sum exp(row)) — so `row[i] - the return value` is log p_i, which
// is where the sampled token's log-probability comes from without going back
// through the probabilities.
static double softmax_row(const float * row, int n, std::vector<double> & out)
{
	out.assign((size_t) n, 0.0);
	double top = row[0];
	for (int i = 1; i < n; i++)
	{
		top = (double) row[i] > top ? (double) row[i] : top;
	}
	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		out[(size_t) i] = std::exp((double) row[i] - top);
		sum += out[(size_t) i];
	}
	for (int i = 0; i < n; i++)
	{
		out[(size_t) i] /= sum;
	}
	return top + std::log(sum);
}

// Total variation: 0.5 * sum |p - q|, 0 for two identical distributions and 1
// for two that share no mass.
static double tv_distance(const std::vector<double> & p, const std::vector<double> & q)
{
	if (p.size() != q.size())
	{
		die("guidance trace: a TV distance between %zu and %zu probabilities",
		    p.size(), q.size());
	}
	double sum = 0;
	for (size_t i = 0; i < p.size(); i++)
	{
		sum += std::fabs(p[i] - q[i]);
	}
	return 0.5 * sum;
}

// The first index holding the row's maximum.
static int argmax_row(const float * row, int n)
{
	int best = 0;
	for (int i = 1; i < n; i++)
	{
		if (row[i] > row[best])
		{
			best = i;
		}
	}
	return best;
}

static json json_request(const Request & r)
{
	json out = json::object();
	out["style"]     = r.style;
	out["lyrics"]    = r.lyrics;
	out["cot"]       = r.cot;
	out["seed"]      = r.seed;
	out["abc"]       = r.has_abc ? json(r.abc) : json(nullptr);
	out["cfg_scale"] = r.has_cfg ? json(r.cfg_scale) : json(nullptr);
	// Deliberately *not* "abc_template": request.json has to stay loadable by
	// the reference's `SongRequest(**request.json)`, which raises on an unknown
	// key. A template job records its template as `template.abc` beside this
	// file, and lands the score it wrote in `abc` — so the artifacts directory
	// is a plain request that reproduces the song (SPEC_TEMPLATE §5).
	//
	// A handover is the exception SPEC_HANDOVER §6 asks for: the block is copied
	// in as the request wrote it, because the song is the legs and nothing else
	// in the directory would reproduce them. Such a request.json is the one form
	// the reference's SongRequest(**request.json) cannot load — it describes a
	// song only this engine renders.
	if (r.has_handover)
	{
		out["handover"] = r.handover_json;
		if (!r.base_take.empty())
		{
			out["base_take"] = r.base_take;
		}
	}
	out["id"]        = r.id;
	return out;
}

static json json_timing(const GenStats & st)
{
	json out = json::object();
	out["seconds"]         = st.seconds;
	out["prefill_seconds"] = st.prefill_seconds;
	out["ttft_seconds"]    = st.ttft_seconds;
	out["output_tokens"]   = st.output_tokens;
	out["content_tokens"]  = st.content_tokens;
	out["output_tps"]      = st.output_tps;
	out["prefix_tokens"]   = st.prefix_tokens;
	// SPEC_GUIDANCE §3: the most sequences this song ever decoded at once, the
	// steps that had more than one, and what prefilling the branches cost.
	out["cfg_branches"]            = st.cfg_branches;
	out["guided_steps"]            = st.guided_steps;
	out["branch_prefill_seconds"]  = st.branch_prefill_seconds;
	// SPEC_KEEP §4: the leading codes this song did not sample, and their cost.
	out["kept_frames"]             = st.kept_frames;
	out["keep_prefill_seconds"]    = st.keep_prefill_seconds;
	// SPEC_SECTIONS §3: the cuts the score phase made, and what they cost.
	out["section_cuts"]             = st.section_cuts;
	out["section_prefill_seconds"]  = st.section_prefill_seconds;
	out["execution"]       = "eager";
	out["attention"]       = "llama.cpp";
	return out;
}

// The `against` curves of one entry, as a guidance.json / sections.json entry
// carries them: nothing at all when the entry opens no branch.
static json json_against(const bool has[BRANCH_KINDS],
	const std::vector<Keyframe> curve[BRANCH_KINDS])
{
	json out = json::object();
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		if (!has[k])
		{
			continue;
		}
		json list = json::array();
		for (size_t c = 0; c < curve[k].size(); c++)
		{
			json kf = json::array();
			kf.push_back(curve[k][c].offset);
			kf.push_back(curve[k][c].weight);
			list.push_back(kf);
		}
		out[branch_name(k)] = list;
	}
	return out;
}

// The guidance plan as the artifacts record it: the request's own block, plus
// whether the decode loop ever got to each entry (SPEC_GUIDANCE §3).
static json json_guidance(const std::vector<GuidanceEntry> & guide)
{
	json out = json::array();
	for (size_t i = 0; i < guide.size(); i++)
	{
		const GuidanceEntry & g = guide[i];
		json                  e = json::object();
		e["frame"] = g.frame;
		if (g.has_style)
		{
			e["style"] = g.style;
		}
		const json against = json_against(g.has, g.curve);
		if (!against.empty())
		{
			e["against"] = against;
		}
		e["reached"] = g.reached;
		out.push_back(e);
	}
	return out;
}

// The sections block as resolved against the score (SPEC_SECTIONS §5).
// `reached` here is "the label turned up in the written score"; whether the
// decode then got as far as the frame it resolved to is guidance.json's own
// `reached`, on the entry this one compiled into.
static json json_sections(const std::vector<SectionEntry> & es)
{
	json out = json::array();
	for (size_t i = 0; i < es.size(); i++)
	{
		const SectionEntry & s = es[i];
		json                 e = json::object();
		e["section"]     = s.section;
		e["nth"]         = s.nth;
		e["style"]       = s.style;
		e["lead_frames"] = s.lead;
		const json against = json_against(s.has, s.curve);
		if (!against.empty())
		{
			e["against"] = against;
		}
		e["reached"] = s.found;
		e["line"]    = s.found ? json(s.line)    : json(nullptr);
		e["bar"]     = s.found ? json(s.bar)     : json(nullptr);
		e["seconds"] = s.found ? json(s.seconds) : json(nullptr);
		e["frame"]   = s.found ? json(s.frame)   : json(nullptr);
		out.push_back(e);
	}
	return out;
}

// The handover as it was played (SPEC_HANDOVER §6): where each entry cut, which
// take took over, how far ahead that take was running and how much of the final
// song it contributed. `take` is the directory an entry named or the `take_<k>`
// the engine rendered; `note` is there only when the entry was skipped.
static json json_handover(const std::vector<HandoverEntry> & es,
	const std::vector<HandoverTake> & takes)
{
	json out = json::array();
	for (size_t i = 0; i < es.size(); i++)
	{
		const HandoverEntry & h = es[i];
		json                  e = json::object();
		e["section"] = h.has_frame ? json(nullptr) : json(h.section);
		e["nth"]     = h.has_frame ? json(nullptr) : json(h.nth);
		e["line"]    = h.has_frame || !h.found ? json(nullptr) : json(h.line);
		e["bar"]     = h.has_frame || !h.found ? json(nullptr) : json(h.bar);
		e["seconds"] = h.has_frame || !h.found ? json(nullptr) : json(h.bar_seconds);
		e["at"]      = h.has_at ? h.at_json : json(nullptr);
		e["frame"]   = h.found ? json(h.cut) : json(nullptr);
		e["take"]    = takes[h.idx].label;
		e["style"]   = takes[h.idx].style;
		e["x"]       = h.x;
		e["offset"]  = h.offset;
		e["z"]       = h.has_z ? json(h.z) : json(nullptr);
		e["hits"]      = h.has_z ? json(h.hits) : json(nullptr);
		e["confident"] = h.confident;
		e["measured"]  = h.confident ? json(nullptr) : json(h.measured);
		e["reached"] = h.reached;
		e["frames"]  = h.gave;
		if (!h.note.empty())
		{
			e["note"] = h.note;
		}
		out.push_back(e);
	}
	return out;
}

static json json_int_array(const std::vector<llama_token> & ids)
{
	json out = json::array();
	for (size_t i = 0; i < ids.size(); i++)
	{
		out.push_back((int) ids[i]);
	}
	return out;
}

static void save_i32_or_die(const std::string & path, const std::vector<int32_t> & v)
{
	const std::vector<int64_t> shape = { (int64_t) v.size() };
	const std::string          err   = npy::save_i32(path.c_str(), shape,
	                                                 v.empty() ? nullptr : v.data());
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
}

struct Artifacts
{
	std::string              dir;
	const Request *          req           = nullptr;
	const GenStats *         st_abc        = nullptr;
	std::vector<llama_token> abc_ids;
	std::vector<llama_token> prefix_sem;
	std::vector<int32_t>     codes;
	std::string              abc_text;
	bool                     have_abc_text = false;
	std::string              template_text;     // "" unless the request was a template
	// null unless the request carried a "guidance" block; a plain `cfg_scale`
	// is in request.json and needs no file of its own (SPEC_GUIDANCE §3).
	const std::vector<GuidanceEntry> * guidance = nullptr;
	// null unless the request carried a "sections" block: the entries as they
	// resolved against the score, which is what says where the tags changed
	// (SPEC_SECTIONS §5). The compiled entries stay in `guidance` above.
	const std::vector<SectionEntry> *  sections = nullptr;
	// null unless the request carried a "semantic_keep": the codes it kept, for
	// the frame count and the digest plan.json records (SPEC_KEEP §4).
	const std::vector<int32_t> *       keep     = nullptr;
	// null unless the request carried a "handover": the entries as they were
	// played, which is what says where the song changed hands (SPEC_HANDOVER §6).
	const json *                       handover = nullptr;
	// null unless --guidance-trace traced this song: TRACE_COLUMNS floats per
	// traced step, written as guidance_trace.npy. A diagnostic, so it is not in
	// plan.json and not in the manifest.
	const std::vector<float> *         trace    = nullptr;
};

static void write_artifacts(const Artifacts & a)
{
	std::error_code ec;
	std::filesystem::create_directories(a.dir, ec);
	if (ec)
	{
		die("cannot create %s: %s", a.dir.c_str(), ec.message().c_str());
	}
	const std::string dir = a.dir + "/";

	if (a.have_abc_text)
	{
		write_file_or_die(dir + "score.abc", a.abc_text);
	}
	// The template as the request gave it, directives and primer and all — the
	// one thing about a template job that request.json cannot carry.
	if (!a.template_text.empty())
	{
		write_file_or_die(dir + "template.abc", a.template_text);
	}
	// Not in request.json, for the same reason "abc_template" is not: it has to
	// stay loadable by the reference's SongRequest(**request.json).
	if (a.guidance != nullptr)
	{
		write_file_or_die(dir + "guidance.json", dump_py(json_guidance(*a.guidance)));
	}
	if (a.sections != nullptr)
	{
		write_file_or_die(dir + "sections.json", dump_py(json_sections(*a.sections)));
	}
	if (a.handover != nullptr)
	{
		write_file_or_die(dir + "handover.json", dump_py(*a.handover));
	}
	if (a.trace != nullptr)
	{
		const std::vector<int64_t> shape = { (int64_t) (a.trace->size() / TRACE_COLUMNS),
		                                     (int64_t) TRACE_COLUMNS };
		const std::string          err   = npy::save((dir + "guidance_trace.npy").c_str(), shape,
		                                             a.trace->empty() ? nullptr : a.trace->data());
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
	}
	save_i32_or_die(dir + "abc_tokens.npy", std::vector<int32_t>(a.abc_ids.begin(), a.abc_ids.end()));
	save_i32_or_die(dir + "prefix.npy",     std::vector<int32_t>(a.prefix_sem.begin(), a.prefix_sem.end()));
	save_i32_or_die(dir + "semantic.npy",   a.codes);

	write_file_or_die(dir + "request.json", dump_py(json_request(*a.req)));

	{
		json plan         = json::object();
		plan["request"]   = json_request(*a.req);
		plan["timing"]    = json_timing(*a.st_abc);
		plan["truncated"] = a.st_abc->truncated;
		plan["prefix"]    = json_int_array(a.prefix_sem);
		plan["abc_ids"]   = json_int_array(a.abc_ids);
		plan["abc"]       = a.have_abc_text ? json(a.abc_text) : json(nullptr);
		if (a.guidance != nullptr)
		{
			plan["guidance"] = json_guidance(*a.guidance);
		}
		if (a.sections != nullptr)
		{
			plan["sections"] = json_sections(*a.sections);
		}
		if (a.handover != nullptr)
		{
			plan["handover"] = *a.handover;
		}
		// What of this song came from an earlier render, and enough of a digest
		// to tell which. The path is deliberately not here: an artifacts
		// directory has to stay relocatable and carry no local paths (§4).
		if (a.keep != nullptr)
		{
			json keep     = json::object();
			keep["frames"] = (int) a.keep->size();
			keep["sha256"] = hash_sha256_hex(a.keep->data(), a.keep->size() * sizeof(int32_t));
			plan["semantic_keep"] = keep;
		}
		write_file_or_die(dir + "plan.json", dump_py(plan));
	}

	{
		std::vector<std::string> names = { "plan.json", "abc_tokens.npy", "prefix.npy" };
		if (a.have_abc_text)
		{
			names.push_back("score.abc");
		}
		if (!a.template_text.empty())
		{
			names.push_back("template.abc");
		}
		if (a.guidance != nullptr)
		{
			names.push_back("guidance.json");
		}
		if (a.sections != nullptr)
		{
			names.push_back("sections.json");
		}
		if (a.handover != nullptr)
		{
			names.push_back("handover.json");
		}
		json manifest = json::object();
		for (size_t i = 0; i < names.size(); i++)
		{
			manifest[names[i]] = sha256_file_hex(dir + names[i]);
		}
		write_file_or_die(dir + "plan_manifest.json", dump_py(manifest));
	}
}

// -------------------------------------------------------------------- main ---

static int parse_positive_arg(const char * name, const char * text)
{
	errno = 0;
	char *      endp  = nullptr;
	const long  value = strtol(text, &endp, 10);
	if (errno != 0 || endp == text || *endp != '\0' || value <= 0 || value > CONTEXT)
	{
		die("%s must be an integer in [1, %d] (got \"%s\")", name, CONTEXT, text);
	}
	return (int) value;
}

// One `[offset, weight]` list. SPEC_GUIDANCE §2.3: non-empty, offsets
// non-decreasing and >= 0, weights finite and within +-GUIDANCE_MAX_WEIGHT.
static std::string parse_curve(const json & v, const char * what, std::vector<Keyframe> & out)
{
	if (!v.is_array() || v.empty())
	{
		return strf("\"%s\" must be a non-empty list of [offset, weight] pairs", what);
	}
	for (size_t i = 0; i < v.size(); i++)
	{
		const json & kf = v[i];
		if (!kf.is_array() || kf.size() != 2 || !kf[0].is_number_integer() || !kf[1].is_number())
		{
			return strf("\"%s\" keyframe %zu must be [offset, weight]", what, i + 1);
		}
		Keyframe k;
		const long long offset = kf[0].get<long long>();
		if (offset < 0 || offset > GUIDANCE_MAX_OFFSET)
		{
			return strf("\"%s\" keyframe %zu: offset must be in [0, %d] frames",
			            what, i + 1, GUIDANCE_MAX_OFFSET);
		}
		k.offset = (int) offset;
		k.weight = kf[1].get<double>();
		if (!std::isfinite(k.weight) || std::fabs(k.weight) > GUIDANCE_MAX_WEIGHT)
		{
			return strf("\"%s\" keyframe %zu: weight must be finite and within +-%g",
			            what, i + 1, GUIDANCE_MAX_WEIGHT);
		}
		if (!out.empty() && k.offset < out.back().offset)
		{
			return strf("\"%s\" keyframe %zu: offsets must not decrease", what, i + 1);
		}
		out.push_back(k);
	}
	return "";
}

// The `against` object of a guidance or a sections entry: one curve per branch
// kind and nothing else (SPEC_GUIDANCE §2.3).
static std::string parse_against(const json & v, bool has[BRANCH_KINDS],
	std::vector<Keyframe> curve[BRANCH_KINDS])
{
	if (!v.is_object() || v.empty())
	{
		return "\"against\" must be a non-empty object (previous, blank)";
	}
	for (json::const_iterator a = v.begin(); a != v.end(); ++a)
	{
		const int kind = a.key() == "previous" ? BRANCH_PREVIOUS
		                 : a.key() == "blank"   ? BRANCH_BLANK : -1;
		if (kind < 0)
		{
			return strf("\"against\": unknown branch \"%s\" (previous, blank)",
			            a.key().c_str());
		}
		const std::string err = parse_curve(a.value(), a.key().c_str(), curve[kind]);
		if (!err.empty())
		{
			return strf("\"against\": %s", err.c_str());
		}
		has[kind] = true;
	}
	return "";
}

// The "guidance" block. New in stage 7, so it is strict: an unknown key inside
// an entry or inside "against" is an error rather than something ignored.
static std::string parse_guidance(const json & v, std::vector<GuidanceEntry> & out)
{
	if (!v.is_array() || v.empty())
	{
		return "must be a non-empty list of entries";
	}
	for (size_t i = 0; i < v.size(); i++)
	{
		const json & e = v[i];
		if (!e.is_object())
		{
			return strf("entry %zu is not an object", i + 1);
		}
		GuidanceEntry g;
		bool          has_frame = false;
		for (json::const_iterator it = e.begin(); it != e.end(); ++it)
		{
			const std::string & key = it.key();
			if (key == "frame")
			{
				if (!it.value().is_number_integer() || it.value().get<long long>() < 0 ||
				    it.value().get<long long>() > CONTEXT)
				{
					return strf("entry %zu: \"frame\" must be an integer in [0, %d]",
					            i + 1, CONTEXT);
				}
				g.frame   = it.value().get<int>();
				has_frame = true;
			} else if (key == "style") {
				if (!it.value().is_string())
				{
					return strf("entry %zu: \"style\" must be a string", i + 1);
				}
				g.has_style = true;
				g.style     = it.value().get<std::string>();
			} else if (key == "against") {
				const std::string err = parse_against(it.value(), g.has, g.curve);
				if (!err.empty())
				{
					return strf("entry %zu: %s", i + 1, err.c_str());
				}
			} else {
				return strf("entry %zu: unknown key \"%s\" (frame, style, against)",
				            i + 1, key.c_str());
			}
		}
		if (!has_frame)
		{
			return strf("entry %zu needs a \"frame\"", i + 1);
		}
		out.push_back(g);
	}
	return "";
}

// The "sections" block (SPEC_SECTIONS §2). Strict like "guidance": `section`
// and `style` are both required, and an unknown key is an error.
static std::string parse_sections(const json & v, std::vector<SectionEntry> & out)
{
	if (!v.is_array() || v.empty())
	{
		return "must be a non-empty list of entries";
	}
	for (size_t i = 0; i < v.size(); i++)
	{
		const json & e = v[i];
		if (!e.is_object())
		{
			return strf("entry %zu is not an object", i + 1);
		}
		SectionEntry s;
		bool         has_style = false;
		for (json::const_iterator it = e.begin(); it != e.end(); ++it)
		{
			const std::string & key = it.key();
			if (key == "section")
			{
				if (!it.value().is_string() || it.value().get<std::string>().empty())
				{
					return strf("entry %zu: \"section\" must be a label name, as the score "
					            "writes it after the \"%% \"", i + 1);
				}
				s.section = it.value().get<std::string>();
			} else if (key == "nth") {
				if (!it.value().is_number_integer() || it.value().get<long long>() < 1 ||
				    it.value().get<long long>() > SECTIONS_MAX_NTH)
				{
					return strf("entry %zu: \"nth\" must be an integer in [1, %d]",
					            i + 1, SECTIONS_MAX_NTH);
				}
				s.nth = it.value().get<int>();
			} else if (key == "style") {
				if (!it.value().is_string() || it.value().get<std::string>().empty())
				{
					return strf("entry %zu: \"style\" must be the tag string that holds from "
					            "this section on", i + 1);
				}
				s.style   = it.value().get<std::string>();
				has_style = true;
			} else if (key == "lead_frames") {
				if (!it.value().is_number_integer() || it.value().get<long long>() < 0 ||
				    it.value().get<long long>() > SECTIONS_MAX_LEAD)
				{
					return strf("entry %zu: \"lead_frames\" must be an integer in [0, %d] "
					            "(25 frames = 1 s)", i + 1, SECTIONS_MAX_LEAD);
				}
				s.lead = it.value().get<int>();
			} else if (key == "against") {
				const std::string err = parse_against(it.value(), s.has, s.curve);
				if (!err.empty())
				{
					return strf("entry %zu: %s", i + 1, err.c_str());
				}
			} else {
				return strf("entry %zu: unknown key \"%s\" (section, nth, style, "
				            "lead_frames, against)", i + 1, key.c_str());
			}
		}
		if (s.section.empty())
		{
			return strf("entry %zu needs a \"section\"", i + 1);
		}
		if (!has_style)
		{
			return strf("entry %zu needs a \"style\": the tags from this section on", i + 1);
		}
		out.push_back(s);
	}
	return "";
}

// Digits and nothing else, at least one of them.
static bool all_digits(const std::string & s)
{
	for (size_t i = 0; i < s.size(); i++)
	{
		if (s[i] < '0' || s[i] > '9')
		{
			return false;
		}
	}
	return !s.empty();
}

// Seconds as a player writes them: "78.6", "1:18.6", "1:18" — at most one
// colon, seconds under 60 when there is one, the fraction optional and a digit
// on both sides of the point when there is one. No sign, no space, nothing
// before or after (§2). Returns false on anything else.
static bool parse_clock(const std::string & s, double & out)
{
	const size_t      colon = s.find(':');
	const std::string mins  = colon == std::string::npos ? std::string() : s.substr(0, colon);
	const std::string secs  = colon == std::string::npos ? s : s.substr(colon + 1);
	const size_t      dot   = secs.find('.');
	// A second colon makes the seconds part "2:3", which is not digits either.
	// The lengths are what keeps the conversions below in range; a cut is at
	// most CONTEXT frames into the song anyway.
	if (secs.size() > 24 || mins.size() > 9 ||
	    (colon != std::string::npos && !all_digits(mins)))
	{
		return false;
	}
	if (dot == std::string::npos ? !all_digits(secs)
	                             : !(all_digits(secs.substr(0, dot)) &&
	                                 all_digits(secs.substr(dot + 1))))
	{
		return false;
	}
	out = std::atof(secs.c_str());
	if (colon == std::string::npos)
	{
		return true;
	}
	if (out >= 60)
	{
		return false;
	}
	out += (double) std::atoi(mins.c_str()) * 60;
	return true;
}

// "at" (§2): a time in the base take's audio, as a number of seconds or as one
// of the string forms. Returns false on anything else — a JSON type that is
// neither, a number that is negative or not finite, or a string that is not a
// clock.
static bool parse_at(const json & v, double & seconds)
{
	if (v.is_number())
	{
		seconds = v.get<double>();
		return std::isfinite(seconds) && seconds >= 0;
	}
	return v.is_string() && parse_clock(v.get<std::string>(), seconds);
}

// The "handover" block (SPEC_HANDOVER §2). Strict like "sections": an entry
// says where the song changes hands, in one of the three forms, and what takes
// over, in one of the three. Unknown keys are errors.
static std::string parse_handover(const json & v, std::vector<HandoverEntry> & out)
{
	if (!v.is_array() || v.empty())
	{
		return "must be a non-empty list of entries";
	}
	for (size_t i = 0; i < v.size(); i++)
	{
		const json & e = v[i];
		if (!e.is_object())
		{
			return strf("entry %zu is not an object", i + 1);
		}
		HandoverEntry h;
		// Which of the three ways of saying where, and whether the two keys
		// that belong to a label were written: both are rules of their own, and
		// a "frame" and an "at" are the same field once parsed.
		bool key_frame = false;
		bool key_label = false;
		for (json::const_iterator it = e.begin(); it != e.end(); ++it)
		{
			const std::string & key = it.key();
			if (key == "frame")
			{
				key_frame = true;
				if (!it.value().is_number_integer() || it.value().get<long long>() < 1 ||
				    it.value().get<long long>() > CONTEXT)
				{
					return strf("entry %zu: \"frame\" must be an integer in [1, %d]",
					            i + 1, CONTEXT);
				}
				h.frame     = it.value().get<int>();
				h.has_frame = true;
			} else if (key == "at") {
				double seconds = 0;
				if (!parse_at(it.value(), seconds))
				{
					return strf("entry %zu: \"at\" must be a time in the base take's audio — "
					            "seconds as a number (78.6) or as a string (\"78.6\", "
					            "\"1:18.6\", \"1:18\")", i + 1);
				}
				// Rounded as a double: a number like 1e30 is a finite time, and
				// no integer holds its frame.
				const double frame = std::round(seconds * SEMANTIC_FPS);
				if (frame < 1 || frame > CONTEXT)
				{
					return strf("entry %zu: \"at\" is frame %.0f, and a cut must be in "
					            "[1, %d] — 25 frames = 1 s", i + 1, frame, CONTEXT);
				}
				h.frame     = (int) frame;
				h.has_frame = true;
				h.has_at    = true;
				h.at_json   = it.value();
			} else if (key == "section") {
				if (!it.value().is_string() || it.value().get<std::string>().empty())
				{
					return strf("entry %zu: \"section\" must be a label name, as the score "
					            "writes it after the \"%% \"", i + 1);
				}
				h.section = it.value().get<std::string>();
			} else if (key == "nth") {
				key_label = true;
				if (!it.value().is_number_integer() || it.value().get<long long>() < 1 ||
				    it.value().get<long long>() > SECTIONS_MAX_NTH)
				{
					return strf("entry %zu: \"nth\" must be an integer in [1, %d]",
					            i + 1, SECTIONS_MAX_NTH);
				}
				h.nth = it.value().get<int>();
			} else if (key == "lead_frames") {
				key_label = true;
				if (!it.value().is_number_integer() || it.value().get<long long>() < 0 ||
				    it.value().get<long long>() > SECTIONS_MAX_LEAD)
				{
					return strf("entry %zu: \"lead_frames\" must be an integer in [0, %d] "
					            "(25 frames = 1 s)", i + 1, SECTIONS_MAX_LEAD);
				}
				h.lead = it.value().get<int>();
			} else if (key == "style") {
				if (!it.value().is_string() || it.value().get<std::string>().empty())
				{
					return strf("entry %zu: \"style\" must be the tag string the engine "
					            "renders this take under", i + 1);
				}
				h.style     = it.value().get<std::string>();
				h.has_style = true;
			} else if (key == "take") {
				if (!it.value().is_string() || it.value().get<std::string>().empty())
				{
					return strf("entry %zu: \"take\" must be the artifacts directory of an "
					            "earlier render of this score", i + 1);
				}
				h.take = it.value().get<std::string>();
			} else if (key == "seconds") {
				if (!it.value().is_number() ||
				    it.value().get<double>() < HANDOVER_MIN_SECONDS ||
				    it.value().get<double>() > HANDOVER_MAX_SECONDS)
				{
					return strf("entry %zu: \"seconds\" must be a number in [%g, %g] — the "
					            "intrusion, 5 s blends and 1 s cuts", i + 1,
					            HANDOVER_MIN_SECONDS, HANDOVER_MAX_SECONDS);
				}
				h.seconds = it.value().get<double>();
			} else if (key == "offset") {
				if (it.value().is_string() && it.value().get<std::string>() == "auto")
				{
					h.auto_off = true;
				} else if (it.value().is_number_integer() &&
				           std::abs(it.value().get<long long>()) <= HANDOVER_RANGE) {
					h.auto_off = false;
					h.offset   = it.value().get<int>();
				} else {
					return strf("entry %zu: \"offset\" must be \"auto\" or an integer in "
					            "[-%d, %d] — frames the incoming take runs ahead of the song",
					            i + 1, HANDOVER_RANGE, HANDOVER_RANGE);
				}
			} else {
				return strf("entry %zu: unknown key \"%s\" (section, nth, lead_frames, frame, "
				            "at, style, take, seconds, offset)", i + 1, key.c_str());
			}
		}
		std::vector<std::string> where;
		if (!h.section.empty())
		{
			where.push_back("\"section\"");
		}
		if (key_frame)
		{
			where.push_back("\"frame\"");
		}
		if (h.has_at)
		{
			where.push_back("\"at\"");
		}
		if (where.size() > 1)
		{
			return strf("entry %zu: %s and %s are two ways to say where the cut is — "
			            "name the label, the frame or the time, not both", i + 1,
			            where[0].c_str(), where[1].c_str());
		}
		if (where.empty())
		{
			return strf("entry %zu needs a \"section\" (with an optional \"nth\"), a "
			            "\"frame\" or an \"at\": where the song changes hands", i + 1);
		}
		if (key_label && h.section.empty())
		{
			return strf("entry %zu: \"nth\" and \"lead_frames\" say which %% label the cut "
			            "is at — they belong to a \"section\", not to a frame or a time",
			            i + 1);
		}
		if (h.has_style && !h.take.empty())
		{
			return strf("entry %zu: \"style\" and \"take\" are two ways to name what takes "
			            "over — the tags to render, or a render to read", i + 1);
		}
		out.push_back(h);
	}
	return "";
}

// The "semantic_keep" block (SPEC_KEEP §2): the start of an earlier render,
// forced as history instead of being sampled. Strict like "guidance" — both
// keys are required and an unknown one is an error rather than something
// ignored. The file itself is read later, once the base directory is known.
static std::string parse_semantic_keep(const json & v, Request & req)
{
	if (!v.is_object())
	{
		return "must be an object with a \"file\" and a \"frames\"";
	}
	bool has_frames = false;
	for (json::const_iterator it = v.begin(); it != v.end(); ++it)
	{
		const std::string & key = it.key();
		if (key == "file")
		{
			if (!it.value().is_string() || it.value().get<std::string>().empty())
			{
				return "\"file\" must be a path to a semantic.npy";
			}
			req.keep_file = it.value().get<std::string>();
		} else if (key == "frames") {
			if (!it.value().is_number_integer() || it.value().get<long long>() < 1 ||
			    it.value().get<long long>() > CONTEXT)
			{
				return strf("\"frames\" must be an integer in [1, %d]", CONTEXT);
			}
			req.keep_frames = it.value().get<int>();
			has_frames      = true;
		} else {
			return strf("unknown key \"%s\" (file, frames)", key.c_str());
		}
	}
	if (req.keep_file.empty())
	{
		return "needs a \"file\"";
	}
	if (!has_frames)
	{
		// Deliberately no implicit "all of it": the last frames of a render are
		// where it ended, and keeping them by accident keeps the ending too.
		return "needs a \"frames\": how many leading codes of the file to keep";
	}
	return "";
}

// The request as JSON, with `where` naming it in the error messages. Split out
// of parse_request so tests/guidance.cpp can drive the request errors of
// SPEC_GUIDANCE §2.3/§2.4 without a file on disk.
static std::string parse_request_json(const json & root, const std::string & where, Request & req)
{
	const char * path = where.c_str();
	if (!root.is_object())
	{
		return strf("%s: expected a JSON object", path);
	}

	// A request cannot pick the VAE's matmul precision: `yue2 song`/`yue2 batch`
	// decode it in the process that ran the NAR, and ggml-vulkan fixes operand
	// staging at device init (SPEC_SINGLE.md §2.2). This runs before any model
	// loads.
	if (root.contains("vk_f16_matmul"))
	{
		return strf("%s: \"vk_f16_matmul\": song/batch decode the VAE in-process at the "
		            "NAR's Vulkan precision; for the exact-F32 decode run the stage on its "
		            "own: yue2 vae -m yue2-vae-f32.gguf -i ARTIFACTS/latent.npy -o OUT.flac",
		            path);
	}

	if (root.contains("style") && root["style"].is_string())
	{
		req.style = root["style"].get<std::string>();
	} else if (root.contains("tags") && root["tags"].is_string()) {
		req.style = root["tags"].get<std::string>();
	} else {
		return strf("%s: needs a \"style\" (or \"tags\") string", path);
	}
	if (root.contains("lyrics") && root["lyrics"].is_string())
	{
		req.lyrics = root["lyrics"].get<std::string>();
	} else {
		return strf("%s: needs a \"lyrics\" string", path);
	}
	if (root.contains("cot") && !root["cot"].is_null())
	{
		if (!root["cot"].is_string())
		{
			return strf("%s: \"cot\" must be a string", path);
		}
		req.cot = root["cot"].get<std::string>();
	}
	if (root.contains("seed") && !root["seed"].is_null())
	{
		// protocol.SongRequest: an integer in [0, 2**63). Read it as an exact
		// uint64 — never through a double, which loses seeds above 2**53.
		const json & v = root["seed"];
		if (!v.is_number_unsigned())
		{
			return strf("%s: \"seed\" must be a non-negative integer below 2**63", path);
		}
		req.seed = v.get<uint64_t>();
	}
	if (root.contains("id") && !root["id"].is_null())
	{
		if (!root["id"].is_string())
		{
			return strf("%s: \"id\" must be a string", path);
		}
		req.id = root["id"].get<std::string>();
	}
	if (root.contains("abc") && !root["abc"].is_null())
	{
		if (!root["abc"].is_string())
		{
			return strf("%s: \"abc\" must be a string or null", path);
		}
		req.has_abc = true;
		req.abc     = root["abc"].get<std::string>();
	}
	if (root.contains("abc_template") && !root["abc_template"].is_null())
	{
		if (!root["abc_template"].is_string())
		{
			return strf("%s: \"abc_template\" must be a string or null", path);
		}
		req.has_tpl      = true;
		req.abc_template = root["abc_template"].get<std::string>();
	}
	if (root.contains("cfg_scale") && !root["cfg_scale"].is_null())
	{
		if (!root["cfg_scale"].is_number())
		{
			return strf("%s: \"cfg_scale\" must be a number or null", path);
		}
		req.has_cfg   = true;
		req.cfg_scale = root["cfg_scale"].get<double>();
	}
	if (root.contains("guidance") && !root["guidance"].is_null())
	{
		const std::string err = parse_guidance(root["guidance"], req.guidance);
		if (!err.empty())
		{
			return strf("%s: \"guidance\": %s", path, err.c_str());
		}
		req.has_guidance = true;
	}
	if (root.contains("sections") && !root["sections"].is_null())
	{
		const std::string err = parse_sections(root["sections"], req.sections);
		if (!err.empty())
		{
			return strf("%s: \"sections\": %s", path, err.c_str());
		}
		req.has_sections = true;
	}
	if (root.contains("handover") && !root["handover"].is_null())
	{
		const std::string err = parse_handover(root["handover"], req.handover);
		if (!err.empty())
		{
			return strf("%s: \"handover\": %s", path, err.c_str());
		}
		req.has_handover  = true;
		req.handover_json = root["handover"];
	}
	if (root.contains("base_take") && !root["base_take"].is_null())
	{
		if (!root["base_take"].is_string() || root["base_take"].get<std::string>().empty())
		{
			return strf("%s: \"base_take\" must be the artifacts directory of an earlier "
			            "render to use as the base take", path);
		}
		req.base_take = root["base_take"].get<std::string>();
	}
	if (root.contains("semantic_keep") && !root["semantic_keep"].is_null())
	{
		const std::string err = parse_semantic_keep(root["semantic_keep"], req);
		if (!err.empty())
		{
			return strf("%s: \"semantic_keep\": %s", path, err.c_str());
		}
		req.has_keep = true;
	}
	return "";
}

// Returns "" or the reason the file is not a usable request. A batch validates
// every job before the model loads (SPEC_BATCH §3.1), so this reports rather
// than exits; the single-request paths turn a non-empty return into die().
static std::string parse_request(const std::string & path, Request & req)
{
	std::string text;
	const std::string err = read_file(path, text);
	if (!err.empty())
	{
		return err;
	}
	json root;
	try
	{
		// Strict: rejects trailing garbage, bad escapes and lone surrogates.
		root = json::parse(text);
	}
	catch (const std::exception & e)
	{
		return strf("%s: %s", path.c_str(), e.what());
	}
	return parse_request_json(root, path, req);
}

// protocol.SongRequest.__post_init__, so anything we accept can still be read
// back by SymbolicPlan.load() -> SongRequest(**request.json).
static std::string validate_request(const Request & r)
{
	if (r.cot != "off" && r.cot != "melody" && r.cot != "full")
	{
		return strf("cot must be off, melody or full (got \"%s\")", r.cot.c_str());
	}
	if (r.seed >= (uint64_t) 1 << 63)
	{
		return "seed must be an integer in [0, 2**63)";
	}

	// id: [A-Za-z0-9][A-Za-z0-9_.-]{0,179}
	bool id_ok = !r.id.empty() && r.id.size() <= 180 && isalnum((unsigned char) r.id[0]);
	for (size_t i = 1; id_ok && i < r.id.size(); i++)
	{
		const unsigned char c = (unsigned char) r.id[i];
		id_ok = isalnum(c) || c == '_' || c == '.' || c == '-';
	}
	if (!id_ok)
	{
		return strf("id must be a filename-safe identifier matching "
		            "[A-Za-z0-9][A-Za-z0-9_.-]{0,179} (got \"%s\")", r.id.c_str());
	}

	if (r.has_abc)
	{
		if (r.cot == "off")
		{
			return "external ABC requires cot=melody or cot=full, not off";
		}
		if (r.abc.find_first_not_of(" \t\n\r\f\v") == std::string::npos)
		{
			return "external ABC requires nonempty text";
		}
	}
	if (r.has_tpl)
	{
		if (r.has_abc)
		{
			return "\"abc\" and \"abc_template\" are mutually exclusive: the first sings a "
			       "score as given, the second lets the model write some of its lines";
		}
		if (r.cot == "off")
		{
			return "\"abc_template\" needs cot=melody or cot=full — cot=off has no abc phase "
			       "to write the holes in";
		}
		if (r.abc_template.find_first_not_of(" \t\n\r\f\v") == std::string::npos)
		{
			return "\"abc_template\" requires nonempty text";
		}
		std::vector<TplSeg> segs;
		const std::string   err = parse_template(r.abc_template, segs);
		if (!err.empty())
		{
			return strf("\"abc_template\": %s", err.c_str());
		}
	}
	if (r.has_cfg && (!std::isfinite(r.cfg_scale) || r.cfg_scale < 0 || r.cfg_scale > 20))
	{
		return strf("cfg_scale must be finite and in [0, 20] (got %g)", r.cfg_scale);
	}

	// SPEC_GUIDANCE §2.3 and §2.4.
	if (r.has_guidance)
	{
		if (r.has_cfg && r.cfg_scale != 1.0)
		{
			return "\"guidance\" and \"cfg_scale\" are two ways to ask for the same "
			       "machinery: give the blend curves in \"guidance\", or a single scalar in "
			       "\"cfg_scale\", not both";
		}
		// The tags in force before the entry being checked.
		std::string style      = r.style;
		int         prev_frame = -1;
		for (size_t i = 0; i < r.guidance.size(); i++)
		{
			const GuidanceEntry & g = r.guidance[i];
			if (g.frame <= prev_frame)
			{
				return strf("\"guidance\" entry %zu: entries must be strictly increasing "
				            "in \"frame\" (got %d after %d)", i + 1, g.frame, prev_frame);
			}
			prev_frame = g.frame;
			if (g.has_style && g.frame == 0)
			{
				// The cut relabels the sequence the song has been decoding and
				// prefills its replacement; at frame 0 there is nothing to
				// relabel, and "these tags from the first frame on" is what the
				// request's own "style" already says. It is also what keeps
				// prefix.npy — the frame-0 positive prefix, and what the NAR
				// reads — the prefix the song actually started from. With this,
				// §2.3's "previous is not allowed at frame 0" needs no rule of
				// its own: it needs a style, and a style cannot be there.
				return strf("\"guidance\" entry %zu: a \"style\" at frame 0 is the request's "
				            "own \"style\"; put it there instead", i + 1);
			}
			if (g.has[BRANCH_PREVIOUS])
			{
				if (!g.has_style)
				{
					return strf("\"guidance\" entry %zu: \"against\".\"previous\" needs a "
					            "\"style\" — without one there is nothing to push against",
					            i + 1);
				}
				if (g.style == style)
				{
					return strf("\"guidance\" entry %zu: \"style\" is the one already in "
					            "force, so \"against\".\"previous\" would push against "
					            "itself", i + 1);
				}
			}
			if (g.has_style)
			{
				style = g.style;
			}
		}
	}
	// SPEC_SECTIONS §2. One mechanism per request: "sections" *is* guidance, so
	// it cannot be given beside the block or the scalar it compiles into.
	if (r.has_sections)
	{
		if (r.has_guidance)
		{
			return "\"sections\" and \"guidance\" are two ways to ask for the same "
			       "machinery: \"sections\" compiles into guidance entries, so give the "
			       "sections or the frames, not both";
		}
		if (r.has_cfg && r.cfg_scale != 1.0)
		{
			return "\"sections\" and \"cfg_scale\" are two ways to ask for the same "
			       "machinery: put the push under an entry's \"against\", not in "
			       "\"cfg_scale\"";
		}
		if (r.has_tpl)
		{
			return "\"sections\" with \"abc_template\" is not supported yet: a template "
			       "re-prefills the slot per hole, and a cut re-prefills it per section";
		}
		if (r.cot == "off")
		{
			return "\"sections\" names labels in a score, and cot=off has none — use "
			       "cot=melody or cot=full";
		}
		// The tags in force before the entry being checked, as SPEC_GUIDANCE's
		// own chain does it.
		std::string style = r.style;
		for (size_t i = 0; i < r.sections.size(); i++)
		{
			const SectionEntry & s = r.sections[i];
			for (size_t k = 0; k < i; k++)
			{
				// Two entries on the same label are the only ordering the request
				// can be held to before the score exists: across labels the order
				// is the score's, and an entry the score puts too early is simply
				// never found (§2).
				if (r.sections[k].section == s.section && r.sections[k].nth >= s.nth)
				{
					return strf("\"sections\" entry %zu: %% %s %d cannot come after "
					            "entry %zu's %% %s %d — entries are in the order the song "
					            "plays them", i + 1, s.section.c_str(), s.nth, k + 1,
					            r.sections[k].section.c_str(), r.sections[k].nth);
				}
			}
			if (s.has[BRANCH_PREVIOUS] && s.style == style)
			{
				return strf("\"sections\" entry %zu: \"style\" is the one already in force, "
				            "so \"against\".\"previous\" would push against itself", i + 1);
			}
			style = s.style;
		}
	}
	// SPEC_HANDOVER §2. One mechanism per request, as for "sections": a handover
	// drives the tags by rendering whole takes and forcing their codes, so it
	// has no use for a branch, a curve or a keep of its own — every one of them
	// would be a second answer to the same question.
	if (r.has_handover)
	{
		if (r.has_guidance || r.has_sections)
		{
			return strf("\"handover\" and %s are two ways to change the tags mid-song: a "
			            "handover swaps the whole history, not the prefix",
			            r.has_guidance ? "\"guidance\"" : "\"sections\"");
		}
		if (r.has_cfg && r.cfg_scale != 1.0)
		{
			return "\"handover\" with \"cfg_scale\": the legs are rendered as plain takes, "
			       "with no branch to push against";
		}
		if (r.has_keep)
		{
			return "\"handover\" with \"semantic_keep\": a handover forces the history of "
			       "every leg itself, so the keep would be overwritten";
		}
		if (r.has_tpl)
		{
			return "\"handover\" with \"abc_template\" is not supported yet: every take has "
			       "to sing one score, and a template writes some of its lines per render";
		}
		if (r.cot == "off")
		{
			return "\"handover\" needs the score every take sings, and cot=off has none — "
			       "use cot=melody or cot=full";
		}
		// What can be held to §2's "cuts must be strictly increasing" before the
		// score exists: the entries that name a frame, which are in order among
		// themselves whatever the labels between them resolve to. The rest is
		// checked once the base score has been walked (handover_cuts_check).
		int         prev = 0;
		size_t      at   = 0;
		std::string where;
		for (size_t i = 0; i < r.handover.size(); i++)
		{
			const HandoverEntry & h = r.handover[i];
			if (!h.has_frame)
			{
				continue;
			}
			if (prev > 0 && h.frame <= prev)
			{
				return strf("\"handover\" entry %zu: %s is not after entry %zu's "
				            "%s — entries are in the order the song plays them",
				            i + 1, handover_where(h, h.frame).c_str(), at, where.c_str());
			}
			prev  = h.frame;
			at    = i + 1;
			where = handover_where(h, h.frame);
			// The intrusion has to fit before the cut. Against the song that is
			// arithmetic on the frame alone; against the take it needs the
			// offset, which only a given one is (§5).
			const int x = handover_x(h.seconds);
			if (h.frame - x < 1)
			{
				return strf("\"handover\" entry %zu: a cut at %s has less than the "
				            "%d-frame intrusion before it", i + 1,
				            handover_where(h, h.frame).c_str(), x);
			}
			if (!h.auto_off && h.frame - x - h.offset < 1)
			{
				return strf("\"handover\" entry %zu: a cut at %s with a %d-frame "
				            "intrusion at offset %+d leaves nothing of the take before it",
				            i + 1, handover_where(h, h.frame).c_str(), x, h.offset);
			}
		}
	} else if (!r.base_take.empty()) {
		return "\"base_take\" is the take a \"handover\" hands over from; without one there "
		       "is nothing for it to do";
	}
	// SPEC_KEEP §2. Everything here is decided without opening the file; the
	// rules that need its contents are checked where it is read, still before
	// the model loads.
	if (r.has_keep)
	{
		if (r.has_tpl)
		{
			return "\"semantic_keep\" with \"abc_template\" is not supported: the kept codes "
			       "were sung to one exact score, and a template writes some of its lines";
		}
		if (!r.has_abc)
		{
			return "\"semantic_keep\" needs the score its codes were sung to, in \"abc\" — a "
			       "freshly written score would not match them";
		}
		for (size_t i = 0; i < r.guidance.size(); i++)
		{
			// frame == N is the normal case: the cut is the first sampled step.
			// A plain cfg_scale needs no rule of its own — its blank branch is
			// simply born at step N.
			if (r.guidance[i].frame < r.keep_frames)
			{
				return strf("\"guidance\" entry %zu: guidance frame %d is inside the kept "
				            "%d frames", i + 1, r.guidance[i].frame, r.keep_frames);
			}
		}
	}
	if (r.has_tpl && (r.has_guidance || cfg_scalar(r) != 1.0))
	{
		return strf("%s with \"abc_template\" is not supported yet: the template's abc phase "
		            "re-prefills the slot, and the branches would have to follow it",
		            r.has_guidance ? "\"guidance\"" : "classifier-free guidance");
	}
	return "";
}

// The request as one job's decode loop needs it: parsed, the command-line
// overrides applied, validated. `guidance` comes back as the effective cfg
// scalar the artifacts record — 1.0 whenever the request drives the branches
// through its own "guidance" block, which guidance.json then carries
// (SPEC_GUIDANCE §3).
static std::string prepare_request(const std::string & path, const std::string & cot_override,
	bool has_seed, uint64_t seed, Request & req, double & guidance)
{
	std::string err = parse_request(path, req);
	if (!err.empty())
	{
		return err;
	}
	if (!cot_override.empty())
	{
		req.cot = cot_override;
	}
	if (has_seed)
	{
		req.seed = seed;
	}
	err = validate_request(req);
	if (!err.empty())
	{
		return strf("%s: %s", path.c_str(), err.c_str());
	}
	guidance = req.has_guidance ? 1.0 : cfg_scalar(req);
	return "";
}

// Where a relative path inside a request resolves from: the directory of the
// file the request was read out of (SPEC_KEEP §2).
static std::string dir_of(const std::string & path)
{
	return std::filesystem::path(path).parent_path().string();
}

// A path a request names, resolved against the file the request was read out
// of — the rule "semantic_keep" follows and "handover" follows with it
// (SPEC_KEEP §2, SPEC_HANDOVER §2).
static std::string request_relative(const std::string & base, const std::string & rel)
{
	std::filesystem::path path = rel;
	if (path.is_relative() && !base.empty())
	{
		path = std::filesystem::path(base) / path;
	}
	return path.string();
}

// The kept codes of SPEC_KEEP §2, through the NAR's `--codec` reader: a 1-D
// int32 .npy, every value a codec index. `max_steps` is the semantic phase's
// own cap, which kept steps count against like any other (§3), so a request
// that keeps all of it would have nothing left to sample. Runs at validation
// time, before the model loads; `name` comes back as the basename, which is all
// the artifacts and the log line ever show of the path.
static std::string load_keep_codes(const Request & r, const std::string & base, int max_steps,
	std::vector<int32_t> & out, std::string & name)
{
	const std::filesystem::path path = request_relative(base, r.keep_file);
	name = path.filename().string();

	npy::ArrayI32     codec;
	const std::string err = npy::load_i32(path.string().c_str(), codec);
	if (!err.empty())
	{
		return strf("\"semantic_keep\": %s", err.c_str());
	}
	if (codec.shape.size() != 1)
	{
		return strf("\"semantic_keep\": %s must be a 1-D int32 array, as the NAR's --codec is",
		            path.string().c_str());
	}
	if ((size_t) r.keep_frames > codec.data.size())
	{
		return strf("\"semantic_keep\": \"frames\" %d exceeds the %zu codes in %s",
		            r.keep_frames, codec.data.size(), name.c_str());
	}
	if (r.keep_frames >= max_steps)
	{
		return strf("\"semantic_keep\": \"frames\" %d leaves nothing to sample under the "
		            "%d-step semantic cap", r.keep_frames, max_steps);
	}
	out.assign(codec.data.begin(), codec.data.begin() + r.keep_frames);
	for (size_t i = 0; i < out.size(); i++)
	{
		// A `semantic.npy` the AR stage wrote holds codes, never token ids and
		// never an end marker — generate() drops MUSIC_END — so this one check
		// is also the SPEC's "the end marker cannot be kept".
		if (out[i] < 0 || out[i] >= CODEC_SIZE)
		{
			return strf("\"semantic_keep\": %s[%zu] = %d is not a codec index in [0, %d)",
			            name.c_str(), i, (int) out[i], CODEC_SIZE);
		}
	}
	return "";
}

// llama_tokenize's size-then-fill idiom, with both return values checked.
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
	const char * what)
{
	std::vector<llama_token> ids;
	const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, false);
	if (n < 0)
	{
		die("tokenizing the %s failed", what);
	}
	ids.resize((size_t) n);
	if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
	                   ids.data(), (int32_t) ids.size(), false, false) < 0)
	{
		die("tokenizing the %s failed", what);
	}
	return ids;
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & ids)
{
	std::vector<char> buf(ids.size() * 8 + 64);
	int               n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(),
	                                       buf.data(), (int32_t) buf.size(), false, false);
	if (n < 0)
	{
		buf.resize((size_t) (-n) + 1);
		n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(),
		                     buf.data(), (int32_t) buf.size(), false, false);
	}
	if (n < 0)
	{
		die("detokenizing the abc ids failed");
	}
	return std::string(buf.data(), (size_t) n);
}

// protocol.token_prefixes without abc ids: [EOD] + tokenize(text) + [ABC_START].
// The abc phase's whole prefix, and the head of every semantic one — including
// the head of a guidance entry that changes the tags (SPEC_GUIDANCE §4.4).
static std::vector<llama_token> prefix_head(const llama_vocab * vocab, const std::string & text)
{
	std::vector<llama_token>       out(1, (llama_token) EOD);
	const std::vector<llama_token> ids = tokenize(vocab, text, "request text");
	out.insert(out.end(), ids.begin(), ids.end());
	out.push_back(ABC_START);
	return out;
}

// protocol.token_prefixes with abc ids: the semantic phase's prefix. A cot=off
// job has no abc ids, which is that branch of the reference verbatim
// (base + [ABC_START, ABC_END, MUSIC_START]).
static std::vector<llama_token> semantic_prefix(const std::vector<llama_token> & head,
	const std::vector<llama_token> & abc_ids)
{
	std::vector<llama_token> out = head;
	out.insert(out.end(), abc_ids.begin(), abc_ids.end());
	out.push_back(ABC_END);
	out.push_back(MUSIC_START);
	return out;
}

// protocol.negative_prefix: the instruction with no tags and no lyrics, then the
// positive branch's exact score. The instruction is tokenized on its own, so its
// last token is not the one the positive prefix holds — where ".\n" merges — and
// the positive prefix may not be sliced to get this (SPEC_GUIDANCE §2.2).
static std::vector<llama_token> blank_prefix(const llama_vocab * vocab, const Request & r,
	const std::vector<llama_token> & abc_ids)
{
	std::vector<llama_token>       out(1, (llama_token) EOD);
	const std::vector<llama_token> ids = tokenize(vocab, instruction(r.cot), "negative instruction");
	out.insert(out.end(), ids.begin(), ids.end());
	if (r.cot == "off")
	{
		out.push_back(MUSIC_START);
		return out;
	}
	out.push_back(ABC_START);
	return semantic_prefix(out, abc_ids);
}


// `vocab_only` loads the tokenizer and no weights, on no device (--prefix-only).
static llama_model * load_model(const std::string & path, const std::string & device, int gpu,
	std::string & backend_name, ggml_backend_dev_t * devices, bool vocab_only = false)
{
	llama_model_params mparams = llama_model_default_params();
	backend_name = "CPU";
	if (vocab_only)
	{
		mparams.vocab_only   = true;
		mparams.devices      = devices;
		mparams.n_gpu_layers = 0;
		backend_name         = "none (tokenizer only)";
	} else if (device == "cpu") {
		// An empty device list keeps llama on the CPU backend and skips Vulkan
		// instance creation entirely.
		mparams.devices      = devices;
		mparams.n_gpu_layers = 0;
	} else if (device == "vulkan") {
		devices[0]           = vulkan_device(gpu);
		mparams.devices      = devices;
		mparams.n_gpu_layers = 999;
		backend_name         = std::string("Vulkan") + std::to_string(gpu) + " (" +
		                       ggml_backend_dev_description(devices[0]) + ")";
	} else {
		die("--device must be cpu or vulkan");
	}

	const double  t_load = now_seconds();
	llama_model * model  = llama_model_load_from_file(path.c_str(), mparams);
	if (model == nullptr)
	{
		die("cannot load %s", path.c_str());
	}
	printf("backend: %s\n", backend_name.c_str());
	printf("model:   %s (%.2f s)\n", path.c_str(), now_seconds() - t_load);
	return model;
}

// ----------------------------------------------------------- batch decode ---

enum Phase { PHASE_ABC, PHASE_SEM, PHASE_DONE };

// How often the decode loop prints its one progress line.
static const double PROGRESS_SECONDS = 10.0;

// Everything about one song that outlives the slot it is decoded in.
struct JobState
{
	Request                  req;
	std::vector<llama_token> prefix_abc;
	std::vector<llama_token> abc_ids;
	std::vector<llama_token> prefix_sem;
	std::string              abc_text;
	bool                     have_abc_text = false;
	bool                     do_abc        = false;
	double                   guidance      = 1.0;

	// SPEC_TEMPLATE.md, all unused unless the request carries an "abc_template".
	bool                     is_template   = false;
	std::vector<TplSeg>      segs;
	std::vector<int>         budget;       // per segment, the hole's sampled-token ceiling
	// Suffix sums over the segments, indexed 0..segs.size(): what is still to be
	// fed, what of that still reaches the score, and how many holes are left.
	// They are what tells a hole how much context it may take (§3).
	std::vector<int>         tail_fed;
	std::vector<int>         tail_given;
	std::vector<int>         tail_holes;
	std::string              tpl_text;     // the emitted score, grown segment by segment
	size_t                   emitted      = 0;   // tokens of it, counted piece by piece
	TemplateStats            tpl;

	// SPEC_GUIDANCE.md, empty unless the request asked for branches. `guide` is
	// the normalised plan (a plain `cfg_scale` is one entry at frame 0), `heads`
	// the prefix head of every entry that changes the tags — tokenized once, at
	// sizing time, so the cut itself is a splice.
	std::vector<GuidanceEntry>            guide;
	std::vector<std::vector<llama_token>> heads;
	int                                   max_branches = 1;   // live sequences at once
	int                                   guided_steps = 0;
	double                                branch_seconds = 0; // prefilling them

	// --guidance-trace: TRACE_COLUMNS floats per semantic step that had a live
	// branch, written as guidance_trace.npy. Empty unless the flag is on.
	std::vector<float>                    trace;
	double                                trace_seconds = 0;

	// SPEC_KEEP.md, empty unless the request carried a "semantic_keep": the N
	// leading codes of an earlier render, read at validation time, forced as
	// history instead of being sampled.
	std::vector<int32_t>     keep_codes;
	std::string              keep_name;     // its basename, all the log line shows
	double                   keep_seconds = 0;

	// SPEC_SECTIONS.md, empty unless the request carried a "sections": the
	// entries as they resolve against the score, which `guide` above is compiled
	// from once they have (sections_compile). `plain_swap` says no entry opens a
	// branch, so the song owns one KV stream and may decode beside others (§4).
	std::vector<SectionEntry> sections;
	bool                      plain_swap  = false;
	double                    sec_seconds = 0;   // re-prefilling the score phase at a cut

	// SPEC_HANDOVER §5: a leg's codes are handed back in memory and spliced into
	// the song, so it writes no artifacts directory of its own.
	bool                     no_files = false;

	GenStats                 st_abc;
	GenStats                 st_sem;
	bool                     ok    = true;   // false once the job is rejected
	std::string              tag;            // "[k/N name] ", empty for a single job
};

// One shadow sequence of a guided song: the same generated history under a
// different prefix. SPEC_GUIDANCE §4.1.
struct Branch
{
	int                      slot    = -1;   // llama seq_id, >= 1
	int                      kind    = BRANCH_BLANK;
	llama_pos                pos     = 0;    // tokens this slot holds
	int                      i_batch = -1;   // row in the batch last submitted
	bool                     live    = false;
	double                   weight  = 0;    // w_i at the step being sampled
	std::vector<llama_token> prefix;         // what it is (re-)prefilled from
	std::vector<float>       row;            // its logits over [MUSIC_END, codec end)
};

// One KV stream of the shared context, and the phase it is in. SPEC_BATCH §4.3.
struct Seq
{
	int                      slot     = 0;   // == the llama seq_id
	int                      home     = 0;   // the slot it starts every job in
	int                      job      = -1;  // -1 = idle
	Phase                    phase    = PHASE_DONE;
	std::vector<llama_token> history;        // the current phase's output and penalty window
	int                      step     = 0;
	std::mt19937_64          rng;
	SampleScratch            scratch;      // the sampler's buffers, reused every step
	llama_pos                pos      = 0;   // tokens this slot holds in the cache
	llama_token              next     = 0;   // fed by the next lockstep batch
	llama_token              sampled  = 0;   // sampled from this step's logits, not applied yet
	int                      i_batch  = -1;  // index in the batch last submitted
	llama_token              last_fed = 0;   // last token a feed put into the cache
	double                   t_phase0 = 0;

	// The template walk (SPEC_TEMPLATE §3), untouched by a job without one.
	size_t                   seg       = 0;  // the segment being fed or sampled
	bool                     in_hole   = false;
	bool                     tpl_cont  = false;   // past %%yue2-continue: sampling freely to ABC_END
	bool                     tpl_full  = false;   // the job's sampled-token cap is spent
	int                      attempt   = 0;  // 1..TEMPLATE_ATTEMPTS within the hole
	llama_pos                hole_pos  = 0;  // where the hole's tokens start
	size_t                   hole_hist = 0;  // history size there
	int                      hole_step = 0;
	llama_token              hole_prev = 0;  // the token before it, re-decoded on a rollback
	std::vector<llama_token> hole_ids;       // sampled and decoded inside the hole
	std::string              carry;          // the closing token's text, fed with the next segment

	// %%yue2-chords: where the slot stood before the out-of-order feed, so the
	// whole of it can be undone and the three lines fed in template order (§3).
	llama_pos                chk_pos   = 0;
	size_t                   chk_hist  = 0;
	int                      chk_step  = 0;

	// Guidance (SPEC_GUIDANCE §4.1), all unused unless the job carries a plan.
	// The primary's slot moves when an entry changes the positive prefix, so
	// `slot` above is not `home` for the whole song.
	bool                     guided    = false;
	bool                     traced    = false; // --guidance-trace, and this song is guided
	size_t                   g_next    = 0;   // the next entry of the plan to take effect
	int                      g_entry   = -1;  // the entry in force, -1 before the first
	Branch                   branch[BRANCH_KINDS];
	std::vector<float>       blend;           // n_vocab floats: the blended row

	// The sections walk (SPEC_SECTIONS §3), unused unless the job writes its own
	// score and carries a "sections". `line_ids` are the tokens of the line being
	// written; `line_pre` is the text a token left behind when it closed one.
	ScoreClock               clock;
	std::vector<llama_token> line_ids;
	std::string              line_pre;
	size_t                   sec_next  = 0;   // the next entry to look for
	int                      sec_cuts  = 0;   // cuts this score phase made
};

// The resolved sections become the guidance plan: a frame each from the bar the
// score phase recorded, and only the entries whose label turned up. `heads` is
// compacted with them, so a cut still finds its prefix head where the plan says
// (SPEC_SECTIONS §4).
static void sections_compile(JobState & js)
{
	sections_frames(js.sections);

	std::vector<GuidanceEntry>            guide;
	std::vector<std::vector<llama_token>> heads;
	for (size_t i = 0; i < js.sections.size(); i++)
	{
		if (!js.sections[i].found)
		{
			continue;
		}
		guide.push_back(js.guide[i]);
		guide.back().frame = js.sections[i].frame;
		heads.push_back(js.heads[i]);
	}
	js.guide = guide;
	js.heads = heads;
}

// The shared decode loop: one llama_decode per step carrying one token for each
// active slot, then one state-machine step per slot. SPEC_BATCH §4.4.
struct Runner
{
	llama_context *            ctx     = nullptr;
	const llama_vocab *        vocab   = nullptr;
	llama_memory_t             mem     = nullptr;
	llama_batch                batch   = {};
	int                        n_vocab = 0;
	Sampling                   s_abc   = sampling_abc();
	Sampling                   s_sem   = sampling_semantic();
	uint32_t                   n_ctx_seq = 0;
	int                        parallel  = 1;
	int                        n_streams = 1;   // KV streams: parallel, or 3 when guided
	std::string                card;
	const std::vector<ArJob> * jobs    = nullptr;
	std::vector<JobState> *    states  = nullptr;
	std::vector<ArResult> *    results = nullptr;
	std::vector<Seq>           seqs;
	size_t                     queued  = 0;   // next unstarted job
	int                        active  = 0;

	// §4.7 item 3: an idle slot between two active ones splits the step into two
	// ubatches. Counted here so the drain phase can be reported, not fixed.
	int    steps_whole = 0;
	int    steps_holed = 0;
	double time_whole  = 0;
	double time_holed  = 0;
	int    decodes     = 0;

	// --verify-sampler: every step also runs the frozen stage-5 sampler on a
	// copy of the sequence's RNG and dies on the first disagreement.
	bool      verify   = false;
	long long verified = 0;

	// --guidance-trace: the two distributions a traced step compares, kept here
	// so that tracing a step allocates nothing after the first one.
	std::vector<double> trace_p;
	std::vector<double> trace_q;

	// Batch-level progress, so a driver watching a pipe sees the loop is alive.
	// Printed from run() at most every PROGRESS_SECONDS, never per step.
	long long sampled     = 0;   // tokens drawn since the loop started
	double    t_loop0     = 0;
	double    t_progress  = 0;

	void        enter(Seq & q, int job);
	void        feed_tokens(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	                        const char * what);
	void        feed(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	                 const char * what);
	llama_token sample(Seq & q, const float * logits);
	void        apply(Seq & q, llama_token token);

	// Guidance (SPEC_GUIDANCE §4.2, §4.3). Everything below returns at once for
	// a job with no plan.
	int           free_slot(const Seq & q) const;
	llama_pos     prefill_branch(Seq & q, int slot, const std::vector<llama_token> & prefix,
	                             const char * what, int & i_batch);
	void          branch_drop(Seq & q, Branch & b, const char * why);
	void          take_entry(Seq & q, size_t idx);
	void          guidance_enter(Seq & q);
	void          guidance_step(Seq & q);
	void          guidance_clear(Seq & q);
	void          fetch_row(Branch & b);
	const float * blend_row(Seq & q, const float * primary);
	void          trace_step(Seq & q, const float * primary, llama_token token);

	// SPEC_KEEP §3: the semantic phase entered on an earlier render's codes.
	void        keep_enter(Seq & q);

	// SPEC_SECTIONS §3: the score phase, watching the lines go by for the labels
	// the request named and re-prefilling the slot at each one.
	void        sections_token(Seq & q, llama_token token);
	void        sections_cut(Seq & q, size_t idx);

	// The template walk. give() feeds one forced stretch of score; the rest is
	// the hole state machine. SPEC_TEMPLATE §3.
	void        give(Seq & q, const std::string & text, GiveKind kind, const char * what);
	bool        abc_fits(const Seq & q, size_t extra) const;
	bool        sem_fits(const Seq & q, size_t extra) const;
	void        template_step(Seq & q);
	void        open_hole(Seq & q);
	void        open_chords(Seq & q);
	void        finish_chords(Seq & q, const std::string & line);
	void        template_token(Seq & q, llama_token token);
	void        fail_hole(Seq & q, const std::string & why);
	void        rollback_hole(Seq & q);
	void        rest_fill(Seq & q);
	void        template_done(Seq & q);
	void        start_continue(Seq & q);
	void        continue_token(Seq & q, llama_token token);
	void        finish_continue(Seq & q, const char * stopped_by);

	void        finish_job(Seq & q);
	void        progress();
	void        run();
};

llama_token Runner::sample(Seq & q, const float * logits)
{
	if (logits == nullptr)
	{
		die("llama_get_logits_ith(%d) returned NULL for slot %d", q.i_batch, q.slot);
	}
	JobState &       js  = (*states)[q.job];
	const bool       abc = q.phase == PHASE_ABC;
	const Sampling & s   = abc ? s_abc : s_sem;
	GenStats &       st  = abc ? js.st_abc : js.st_sem;

	const bool legacy_off = abc ? false : js.req.cot == "off";

	// The reference runs first, on a copy of the RNG, so the sampler proper is
	// still the one that advances the real stream — the sequence is exactly what
	// it would have been without the flag. Both must also consume the same
	// number of draws, which is what the state comparison checks.
	std::mt19937_64 rng_ref;
	llama_token     ref = 0;
	if (verify)
	{
		rng_ref = q.rng;
		ref     = sample_step_ref(logits, n_vocab, s, q.history, q.step, abc, legacy_off,
		                          rng_ref, q.in_hole);
	}

	const llama_token token = sample_step(logits, n_vocab, s, q.history, q.step,
	                                      abc, legacy_off, q.rng, &q.scratch, q.in_hole);
	if (verify)
	{
		if (token != ref)
		{
			die("--verify-sampler: slot %d %s step %d sampled %d, the stage-5 sampler "
			    "sampled %d", q.slot, abc ? "abc" : "semantic", q.step,
			    (int) token, (int) ref);
		}
		if (!(rng_ref == q.rng))
		{
			die("--verify-sampler: slot %d %s step %d — the two samplers drew a "
			    "different number of times", q.slot, abc ? "abc" : "semantic", q.step);
		}
		verified++;
	}

	sampled++;
	if (st.ttft_seconds == 0)
	{
		st.ttft_seconds = now_seconds() - q.t_phase0;
	}
	return token;
}

// The ids the semantic sampler can visit: MUSIC_END and the codec block, which
// sit next to each other, so one contiguous run of floats is the whole of what a
// branch row has to carry and the whole of what the blend writes (§2.1).
static const int SEM_ROW_FIRST = MUSIC_END;
static const int SEM_ROW_LEN   = CODEC_OFFSET + CODEC_SIZE - MUSIC_END;

// The lowest KV stream this song is not already using. Live slots are kept
// contiguous from 0 that way, which is what `split_equal` wants (SPEC_BATCH
// §4.7 item 3); a guided song runs at parallel 1, so no other Seq competes.
int Runner::free_slot(const Seq & q) const
{
	for (int s = 0; s < n_streams; s++)
	{
		if (s == q.slot)
		{
			continue;
		}
		bool used = false;
		for (int k = 0; k < BRANCH_KINDS; k++)
		{
			used = used || (q.branch[k].live && q.branch[k].slot == s);
		}
		if (!used)
		{
			return s;
		}
	}
	die("guidance: no free KV stream among %d for slot %d", n_streams, q.slot);
}

// Prefills a sequence with `prefix` followed by every code generated so far but
// the one still pending in `q.next` — the state every sequence of a guided song
// shares (§4.3). Its own llama_decode calls, never the lockstep batch. Returns
// the position the slot then stands at, and hands back the last token's logits
// row index, which is this branch's row for the step about to be sampled *only*
// at the semantic entry, where nothing is pending; mid-song the lockstep decode
// of `q.next` produces it.
llama_pos Runner::prefill_branch(Seq & q, int slot, const std::vector<llama_token> & prefix,
	const char * what, int & i_batch)
{
	JobState &               js   = (*states)[q.job];
	const double             t0   = now_seconds();
	std::vector<llama_token> feed = prefix;
	if (!q.history.empty())
	{
		feed.insert(feed.end(), q.history.begin(), q.history.end() - 1);
	}
	// What the slot will hold at the end: this prefix and, at the most, a whole
	// semantic phase of codes. Sized for in run_ar_batch (§4.5); checked here
	// because that estimate is the only thing between it and a decode error.
	if (prefix.size() + (size_t) s_sem.max_tokens > (size_t) n_ctx_seq)
	{
		die("%sguidance: the %s prefix %zu + max_tokens %d exceeds the allocated "
		    "%u-token context", js.tag.c_str(), what, prefix.size(), s_sem.max_tokens, n_ctx_seq);
	}
	const llama_pos held = llama_memory_seq_pos_max(mem, slot);
	if (held != -1)
	{
		die("%sguidance: slot %d still holds %d tokens before the %s prefix enters it",
		    js.tag.c_str(), slot, (int) held + 1, what);
	}

	llama_pos pos = 0;
	i_batch = decode_feed(ctx, batch, feed, slot, pos, what);
	decodes++;
	if ((size_t) (llama_memory_seq_pos_max(mem, slot) + 1) != feed.size())
	{
		die("%sguidance: the %s prefill left %d tokens in slot %d, it fed %zu",
		    js.tag.c_str(), what, (int) llama_memory_seq_pos_max(mem, slot) + 1, slot, feed.size());
	}
	js.branch_seconds += now_seconds() - t0;
	printf("%sguidance: step %d: %s prefilled into slot %d (%zu + %zu tokens)\n",
	       js.tag.c_str(), q.step, what, slot, prefix.size(), feed.size() - prefix.size());
	return pos;
}

void Runner::branch_drop(Seq & q, Branch & b, const char * why)
{
	llama_memory_seq_rm(mem, b.slot, -1, -1);
	const llama_pos held = llama_memory_seq_pos_max(mem, b.slot);
	if (held != -1)
	{
		die("guidance: slot %d still holds %d tokens after the %s branch was dropped",
		    b.slot, (int) held + 1, branch_name(b.kind));
	}
	printf("%sguidance: step %d: %s branch dropped from slot %d (%s)\n",
	       (*states)[q.job].tag.c_str(), q.step, branch_name(b.kind), b.slot, why);
	b.live = false;
	b.pos  = 0;
	b.slot = -1;
}

// One guidance entry takes effect. A new `style` is the cut (§4.3): the old
// primary sequence already *is* "previous prefix + history", so it is relabelled
// rather than recomputed, and the new prefix is prefilled into a free slot.
void Runner::take_entry(Seq & q, size_t idx)
{
	JobState &      js = (*states)[q.job];
	GuidanceEntry & g  = js.guide[idx];
	g.reached = true;

	if (g.has_style)
	{
		Branch & prev = q.branch[BRANCH_PREVIOUS];
		if (prev.live)
		{
			// Three slots always suffice because the outgoing entry's `previous`
			// goes before this entry's takes its place.
			branch_drop(q, prev, "replaced by the entry that follows it");
		}
		const std::vector<llama_token> np = semantic_prefix(js.heads[idx], js.abc_ids);
		const int                      old_slot = q.slot;
		q.slot = -1;
		if (g.has[BRANCH_PREVIOUS])
		{
			prev.slot   = old_slot;
			prev.pos    = q.pos;
			prev.live   = true;
			prev.prefix.clear();     // it is never re-prefilled: see guidance_step
			printf("%sguidance: step %d: slot %d relabelled as the previous branch\n",
			       js.tag.c_str(), q.step, old_slot);
		} else {
			llama_memory_seq_rm(mem, old_slot, -1, -1);
			if (llama_memory_seq_pos_max(mem, old_slot) != -1)
			{
				die("%sguidance: slot %d still holds tokens after the cut cleared it",
				    js.tag.c_str(), old_slot);
			}
		}
		// A plain swap has no branch to keep out of the way, and the slot it just
		// cleared is the one to use: with one stream per song, free_slot's "lowest
		// stream this song is not using" would hand out a neighbour's (§4).
		const int slot = js.plain_swap ? old_slot : free_slot(q);
		q.pos  = prefill_branch(q, slot, np, "new positive", q.i_batch);
		q.slot = slot;
	}
	q.g_entry = (int) idx;
}

// Called for a guided sequence before the batch that yields the logits of
// semantic step `q.step`: takes every entry whose frame has come, then prefills
// or drops branches so that exactly the ones still needed are live (§4.3).
void Runner::guidance_step(Seq & q)
{
	JobState & js = (*states)[q.job];
	while (q.g_next < js.guide.size() && js.guide[q.g_next].frame <= q.step)
	{
		take_entry(q, q.g_next++);
	}
	if (q.g_entry < 0)
	{
		return;
	}

	const GuidanceEntry & g   = js.guide[(size_t) q.g_entry];
	const int             off = q.step - g.frame;
	int                   live = 1;
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		Branch &   b    = q.branch[k];
		const bool need = g.has[k] && curve_needed_from(g.curve[k], off);
		if (need && !b.live)
		{
			// `previous` is only ever born at a cut, where it is relabelled; and
			// "needed at or after off" only ever goes from true to false, so a
			// branch this entry needs later was live when the entry was taken.
			if (b.prefix.empty())
			{
				die("%sguidance: the %s branch is needed at step %d and has no prefix",
				    js.tag.c_str(), branch_name(k), q.step);
			}
			b.slot = free_slot(q);
			b.pos  = prefill_branch(q, b.slot, b.prefix, branch_name(k), b.i_batch);
			b.live = true;
			fetch_row(b);
		} else if (!need && b.live) {
			branch_drop(q, b, "its curve is zero from here on");
		}
		// A needed branch whose weight happens to be 0 is still decoded: its KV
		// has to stay in step with the history.
		b.weight = need ? curve_at(g.curve[k], off) : 0;
		live    += b.live ? 1 : 0;
	}
	js.max_branches = std::max(js.max_branches, live);
}

// The semantic phase of a guided job is about to start: the branch prefixes are
// known (the score is written), and whatever step 0 needs is prefilled *before*
// the primary's own feed, so that the primary's is the last decode and its
// logits are the live ones when the first token is sampled (§4.3).
void Runner::guidance_enter(Seq & q)
{
	JobState & js = (*states)[q.job];
	if (js.guide.empty())
	{
		return;
	}
	q.guided  = true;
	q.traced  = (*jobs)[q.job].trace;
	q.g_next  = 0;
	q.g_entry = -1;
	q.blend.assign((size_t) n_vocab, 0.0f);
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		q.branch[k]      = Branch();
		q.branch[k].kind = k;
		q.branch[k].row.assign((size_t) SEM_ROW_LEN, 0.0f);
	}
	q.branch[BRANCH_BLANK].prefix = blank_prefix(vocab, js.req, js.abc_ids);
	guidance_step(q);
}

// Every shadow slot goes back, so the next job of a --parallel 1 batch enters a
// clean context (§4.1), and the primary returns to the slot it started in.
void Runner::guidance_clear(Seq & q)
{
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		if (q.branch[k].live)
		{
			branch_drop(q, q.branch[k], "the song is finished");
		}
		q.branch[k] = Branch();
	}
	q.guided  = false;
	q.traced  = false;
	q.g_next  = 0;
	q.g_entry = -1;
	q.blend.clear();
}

// A branch's row for this step, copied out of the context's output buffer: the
// next llama_decode overwrites it, and at the semantic entry the branches were
// decoded before the primary.
void Runner::fetch_row(Branch & b)
{
	const float * logits = llama_get_logits_ith(ctx, b.i_batch);
	if (logits == nullptr)
	{
		die("llama_get_logits_ith(%d) returned NULL for the %s branch in slot %d",
		    b.i_batch, branch_name(b.kind), b.slot);
	}
	memcpy(b.row.data(), logits + SEM_ROW_FIRST, (size_t) SEM_ROW_LEN * sizeof(float));
}

// L = B + sum_i w_i * (B - N_i), in f32 and only over the ids the semantic
// sampler can visit; the rest of the scratch row is never read (§2.1). The raw
// row is handed back untouched when nothing contributes, which is every step of
// an unguided song and every abc step.
const float * Runner::blend_row(Seq & q, const float * primary)
{
	if (!q.guided || primary == nullptr)
	{
		return primary;
	}
	bool live = false;
	bool any  = false;
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		live = live || q.branch[k].live;
		any  = any  || (q.branch[k].live && q.branch[k].weight != 0);
	}
	(*states)[q.job].guided_steps += live ? 1 : 0;
	if (!any)
	{
		return primary;
	}

	for (int i = 0; i < SEM_ROW_LEN; i++)
	{
		const float b   = primary[SEM_ROW_FIRST + i];
		float       acc = b;
		for (int k = 0; k < BRANCH_KINDS; k++)
		{
			if (q.branch[k].live && q.branch[k].weight != 0)
			{
				acc += (float) q.branch[k].weight * (b - q.branch[k].row[(size_t) i]);
			}
		}
		q.blend[(size_t) (SEM_ROW_FIRST + i)] = acc;
	}
	return q.blend.data();
}

// One row of guidance_trace.npy for the step just drawn: how far each branch
// stands from the primary, how far the blend moved the distribution, and how
// surprising the drawn token is to the unguided model. Called after the draw and
// before apply(), on a guided song only, and it touches neither the RNG nor a
// row the sampler reads — the flag cannot change what the song sings.
void Runner::trace_step(Seq & q, const float * primary, llama_token token)
{
	if (primary == nullptr)
	{
		return;
	}
	bool live = false;
	bool any  = false;
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		live = live || q.branch[k].live;
		any  = any  || (q.branch[k].live && q.branch[k].weight != 0);
	}
	if (!live)
	{
		return;
	}

	JobState &    js    = (*states)[q.job];
	const double  t0    = now_seconds();
	const float * b_row = primary + SEM_ROW_FIRST;
	const double  logz  = softmax_row(b_row, SEM_ROW_LEN, trace_p);
	const float   na    = std::numeric_limits<float>::quiet_NaN();

	// Columns 1..4 are two per branch, in BranchKind order.
	static_assert(BRANCH_PREVIOUS == 0 && BRANCH_BLANK == 1 && BRANCH_KINDS == 2,
	              "the trace columns are laid out in BranchKind order");
	float row[TRACE_COLUMNS];
	row[0] = (float) q.step;
	for (int k = 0; k < BRANCH_KINDS; k++)
	{
		const Branch & b = q.branch[k];
		row[1 + k] = b.live ? (float) b.weight : 0.0f;
		if (!b.live)
		{
			row[3 + k] = na;
			continue;
		}
		softmax_row(b.row.data(), SEM_ROW_LEN, trace_q);
		row[3 + k] = (float) tv_distance(trace_p, trace_q);
	}
	// Whether the branch this song pushes away from still wants the same token:
	// the cheapest reading of "the two prefixes have parted".
	const Branch & prev = q.branch[BRANCH_PREVIOUS];
	row[5] = prev.live
		? (argmax_row(b_row, SEM_ROW_LEN) == argmax_row(prev.row.data(), SEM_ROW_LEN) ? 1.0f : 0.0f)
		: na;
	// What the sampler was handed: blend_row leaves the primary's own row when
	// no live branch carries a weight, and that is a distance of 0.
	row[6] = 0.0f;
	if (any)
	{
		softmax_row(q.blend.data() + SEM_ROW_FIRST, SEM_ROW_LEN, trace_q);
		row[6] = (float) tv_distance(trace_p, trace_q);
	}
	const int id = (int) token - SEM_ROW_FIRST;
	row[7] = id >= 0 && id < SEM_ROW_LEN ? (float) ((double) b_row[id] - logz) : na;

	js.trace.insert(js.trace.end(), row, row + TRACE_COLUMNS);
	js.trace_seconds += now_seconds() - t0;
}

void Runner::apply(Seq & q, llama_token token)
{
	if (q.in_hole)
	{
		template_token(q, token);
		return;
	}
	if (q.tpl_cont)
	{
		continue_token(q, token);
		return;
	}

	JobState &       js  = (*states)[q.job];
	const bool       abc = q.phase == PHASE_ABC;
	const Sampling & s   = abc ? s_abc : s_sem;
	GenStats &       st  = abc ? js.st_abc : js.st_sem;

	const bool eos = token == (abc ? ABC_END : MUSIC_END);
	if (!eos)
	{
		// The end token is never part of the output (stage_ar.cpp's generate()).
		q.history.push_back(token);
		q.step++;
		if (q.step < s.max_tokens)
		{
			q.next = token;
			// The score clock, and the cut when the line this token closed is a
			// label an entry named (SPEC_SECTIONS §3). The slot it re-prefills
			// still holds "prefix + history minus the token pending in q.next".
			if (abc && !js.sections.empty())
			{
				sections_token(q, token);
			}
			return;
		}
	}

	st.seconds        = now_seconds() - q.t_phase0;
	st.content_tokens = (int) q.history.size();
	st.output_tokens  = st.content_tokens + (eos ? 1 : 0);
	st.output_tps     = st.seconds > 0 ? st.output_tokens / st.seconds : 0;
	st.truncated      = !eos;

	if (!abc)
	{
		finish_job(q);
		return;
	}

	printf("%sabc:      %d tokens in %.2f s = %.2f tok/s%s\n", js.tag.c_str(),
	       st.output_tokens, st.seconds, st.output_tps, st.truncated ? " (TRUNCATED)" : "");

	js.abc_ids = q.history;
	for (size_t i = 0; i < js.abc_ids.size(); i++)
	{
		if (js.abc_ids[i] < 0 || js.abc_ids[i] >= EOD)
		{
			die("%sabc id %d at %zu leaves the ordinary text vocabulary",
			    js.tag.c_str(), (int) js.abc_ids[i], i);
		}
	}
	js.abc_text      = detokenize(vocab, js.abc_ids);
	js.have_abc_text = true;

	js.prefix_sem = semantic_prefix(js.prefix_abc, js.abc_ids);
	if ((int) js.prefix_sem.size() + s_sem.max_tokens > (int) n_ctx_seq)
	{
		die("%ssemantic prefix %zu + max_tokens %d exceeds the allocated %u-token context",
		    js.tag.c_str(), js.prefix_sem.size(), s_sem.max_tokens, n_ctx_seq);
	}

	// The labels this score turned out to hold become the guidance plan, with a
	// frame each (SPEC_SECTIONS §4).
	if (!js.sections.empty())
	{
		sections_compile(js);
	}

	q.phase    = PHASE_SEM;
	q.history.clear();
	q.step     = 0;
	q.rng.seed(js.req.seed);
	q.t_phase0 = now_seconds();

	// The cache already holds prefix_abc plus every abc id that was decoded; a
	// truncated phase still has its last kept token in hand, so the bridge is
	// two tokens or three. Its own decode call, not the lockstep batch: §4.4.
	//
	// Unless a section cut re-prefilled the slot under other tags: what it holds
	// is then that entry's prefix and not this request's, and the semantic phase
	// starts from the request's own style either way (SPEC_SECTIONS §3). That is
	// a clear and a whole prefill — the hand-over a template makes.
	std::vector<llama_token> bridge;
	if (q.sec_cuts > 0)
	{
		llama_memory_seq_rm(mem, q.slot, -1, -1);
		const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
		if (held != -1)
		{
			die("%sslot %d still holds %d tokens after the score phase cleared it",
			    js.tag.c_str(), q.slot, (int) held + 1);
		}
		q.pos  = 0;
		bridge = js.prefix_sem;
	} else {
		if (!eos)
		{
			bridge.push_back(token);
		}
		bridge.push_back(ABC_END);
		bridge.push_back(MUSIC_START);
	}

	js.st_sem.prefix_tokens = (int) js.prefix_sem.size();

	// Whatever step 0 is guided against goes in before the bridge, so that the
	// primary's decode is the last one and its logits are live (§4.3).
	guidance_enter(q);
	feed(q, bridge, js.prefix_sem.size(), "semantic");
}

// Decodes `tokens` into this slot in its own call(s) and checks that the cache
// then holds exactly `expect_pos` of them. Nothing is sampled: the caller either
// wants the logits (feed) or is forcing score text in (give, rollback_hole).
void Runner::feed_tokens(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	const char * what)
{
	if (tokens.empty())
	{
		die("%s: nothing to feed into slot %d", what, q.slot);
	}
	q.i_batch  = decode_feed(ctx, batch, tokens, q.slot, q.pos, what);
	q.last_fed = tokens.back();
	decodes++;

	// The KV cache must agree with the prefix the artifacts record: a phase that
	// was truncated used to leave the cache one token short here.
	const llama_pos have = llama_memory_seq_pos_max(mem, q.slot) + 1;
	if ((size_t) have != expect_pos)
	{
		die("%s%s prefill left %d tokens in slot %d, the prefix is %zu — "
		    "the cache and prefix.npy disagree",
		    (*states)[q.job].tag.c_str(), what, (int) have, q.slot, expect_pos);
	}
}

void Runner::feed(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	const char * what)
{
	JobState & js = (*states)[q.job];
	feed_tokens(q, tokens, expect_pos, what);

	GenStats & st = q.phase == PHASE_ABC ? js.st_abc : js.st_sem;
	st.prefill_seconds = now_seconds() - q.t_phase0;

	// Immediately: the next llama_decode from any slot overwrites this buffer.
	// The branches were prefilled before this feed, so their rows are in hand
	// (SPEC_GUIDANCE §4.3).
	const float *     primary = llama_get_logits_ith(ctx, q.i_batch);
	const llama_token token   = sample(q, blend_row(q, primary));
	if (q.traced)
	{
		trace_step(q, primary, token);
	}
	apply(q, token);
}

// One forced stretch of score: tokenized, decoded, and pushed into `history` so
// the repetition-penalty window sees it as if the model had written it. The kind
// separates a given line (which reaches score.abc) from a primer line (which
// does not) and from the out-of-order feed of a chords directive (which is
// rolled back, so it counts for nothing) — the text itself is appended by the
// caller that knows which.
void Runner::give(Seq & q, const std::string & text, GiveKind kind, const char * what)
{
	JobState &                     js  = (*states)[q.job];
	const std::vector<llama_token> ids = tokenize(vocab, text, what);

	// Unreachable once the two guards below hold — the given text is counted
	// exactly at sizing time and a hole is stopped before it eats the room the
	// rest of the template needs. It is the backstop that turns a sizing bug
	// into a message instead of a -1 from llama_decode.
	if ((size_t) q.pos + ids.size() > (size_t) n_ctx_seq)
	{
		die("%sthe template's %s needs %zu of the %u tokens this slot has — the "
		    "context estimate was wrong, please report the template",
		    js.tag.c_str(), what, (size_t) q.pos + ids.size(), n_ctx_seq);
	}

	q.history.insert(q.history.end(), ids.begin(), ids.end());
	if (kind == GIVE_SCORE)
	{
		js.tpl.given_tokens += (int) ids.size();
		js.emitted          += ids.size();
	} else if (kind == GIVE_PRIMER) {
		js.tpl.primer_tokens += (int) ids.size();
	}
	feed_tokens(q, ids, (size_t) q.pos + ids.size(), what);
}

// Is there room in the slot for `extra` more tokens of this hole, plus every
// given and primer segment still to be fed and a rest-fill for every hole after
// this one? The abc phase runs in the slot; the guard is what keeps a hole that
// writes until its budget stops it from starving the template behind it.
bool Runner::abc_fits(const Seq & q, size_t extra) const
{
	const JobState & js = (*states)[q.job];
	return (size_t) q.pos + extra + (size_t) js.tail_fed[q.seg + 1] +
	       (size_t) TEMPLATE_REST_TOKENS * js.tail_holes[q.seg + 1] <= (size_t) n_ctx_seq;
}

// The same question for the phase after it: the semantic prefill is the emitted
// score re-tokenized, and it has to leave max_tokens of codec behind it. Without
// this, a template that writes more than the sizing estimate expected reaches
// template_done and dies there — taking the whole batch with it.
bool Runner::sem_fits(const Seq & q, size_t extra) const
{
	const JobState & js   = (*states)[q.job];
	// A chords directive's own three lines are not in `emitted` yet: they reach
	// the score only once the line is written and they are fed in order.
	const size_t     own  = q.seg < js.segs.size() && js.segs[q.seg].kind == TPL_CHORDS
	                        ? (size_t) (js.tail_given[q.seg] - js.tail_given[q.seg + 1]) : 0;
	const size_t     need = js.prefix_abc.size() + js.emitted + extra + own +
	                        (size_t) js.tail_given[q.seg + 1] +
	                        (size_t) TEMPLATE_REST_TOKENS * js.tail_holes[q.seg + 1] +
	                        2 + (size_t) s_sem.max_tokens + (size_t) TEMPLATE_SEAM_MARGIN;
	return need <= (size_t) n_ctx_seq;
}

// Feeds given and primer segments until a hole opens or the template runs out.
// A hole's first token is sampled from the logits of whatever was fed last,
// which is why the feeds happen here and not in the lockstep batch.
void Runner::template_step(Seq & q)
{
	JobState & js = (*states)[q.job];
	while (q.seg < js.segs.size())
	{
		const TplSeg & seg = js.segs[q.seg];
		if (seg.kind == TPL_HOLE)
		{
			open_hole(q);
			return;
		}
		if (seg.kind == TPL_CONTINUE)
		{
			start_continue(q);
			return;
		}
		if (seg.kind == TPL_CHORDS)
		{
			open_chords(q);
			return;
		}
		q.seg++;
		const bool        emit = seg.kind == TPL_GIVEN;
		const std::string text = q.carry + seg.text;
		q.carry.clear();
		if (emit)
		{
			js.tpl_text += seg.text;
		}
		give(q, text, emit ? GIVE_SCORE : GIVE_PRIMER, emit ? "given segment" : "primer segment");
	}
	template_done(q);
}

void Runner::open_hole(Seq & q)
{
	JobState & js = (*states)[q.job];

	// The tail of the line the previous hole closed with: its text is already in
	// score.abc, but the cache has not seen it (§3 — the closing token is never
	// fed as sampled), so it goes in now, re-tokenized.
	if (!q.carry.empty())
	{
		const std::string text = q.carry;
		q.carry.clear();
		give(q, text, GIVE_SCORE, "hole tail");
	}

	q.in_hole   = true;
	q.attempt   = 1;
	q.hole_pos  = q.pos;
	q.hole_hist = q.history.size();
	q.hole_step = q.step;
	q.hole_prev = q.last_fed;
	q.hole_ids.clear();
	js.tpl.holes++;

	if (q.tpl_full)
	{
		rest_fill(q);
		return;
	}
	template_token(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
}

// %%yue2-chords: the line the model writes here sits *above* the one it is
// written from. The slot is checkpointed, the two lines below the directive are
// fed, then the voice header above it and an opening double quote; from there
// the line is sampled exactly as a hole's is. Whatever it comes to, accepted or
// rest-filled, the out-of-order feed is undone and the three lines are fed in
// template order — finish_chords (SPEC_TEMPLATE §3).
void Runner::open_chords(Seq & q)
{
	JobState &     js  = (*states)[q.job];
	const TplSeg & seg = js.segs[q.seg];

	// As open_hole: the head of the last hole's closing token is score text the
	// cache has not seen yet. It goes in before the checkpoint.
	if (!q.carry.empty())
	{
		const std::string text = q.carry;
		q.carry.clear();
		give(q, text, GIVE_SCORE, "hole tail");
	}

	q.chk_pos  = q.pos;
	q.chk_hist = q.history.size();
	q.chk_step = q.step;
	js.tpl.holes++;
	js.tpl.chord_lines++;

	if (q.tpl_full)
	{
		rest_fill(q);
		return;
	}

	give(q, seg.text, GIVE_SCRATCH, "chords lookahead");
	give(q, seg.head, GIVE_SCRATCH, "chords voice");
	give(q, "\"", GIVE_SCRATCH, "chords quote");

	q.in_hole   = true;
	q.attempt   = 1;
	q.hole_pos  = q.pos;
	q.hole_hist = q.history.size();
	q.hole_step = q.step;
	q.hole_prev = q.last_fed;
	q.hole_ids.clear();

	template_token(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
}

// The end of a chords directive, from either side: the accepted line or the
// rest-fill. Everything the out-of-order feed put in the slot goes back, and the
// header, the line and the two lines below it are fed in the order the score
// holds them — so the cache and `history` end up as if the model had written the
// line where it stands. `step` is the caller's business: an accepted line spent
// its draws, a rest-fill spent none.
void Runner::finish_chords(Seq & q, const std::string & line)
{
	JobState &        js   = (*states)[q.job];
	const TplSeg &    seg  = js.segs[q.seg];
	const std::string text = seg.head + line + seg.text;

	llama_memory_seq_rm(mem, q.slot, q.chk_pos, -1);
	q.pos = q.chk_pos;
	q.history.resize(q.chk_hist);
	q.hole_ids.clear();
	q.in_hole = false;
	q.seg++;

	js.tpl_text += text;
	give(q, text, GIVE_SCORE, "chords segment");
	template_step(q);
}

// One token sampled inside a hole. The hole closes on the first `\n`; the token
// that carries it is never fed as sampled, so the part of it before the newline
// rides along with the next segment instead (§3).
void Runner::template_token(Seq & q, llama_token token)
{
	JobState &     js  = (*states)[q.job];
	const TplSeg & seg = js.segs[q.seg];

	js.tpl.sampled_tokens++;
	q.step++;

	// --max-abc caps the job's *sampled* tokens — every draw, the ones a rolled
	// back attempt spent included. `q.step` is restored by a rollback and is the
	// wrong counter for it; `tpl.sampled_tokens` only ever grows.
	if (js.tpl.sampled_tokens >= s_abc.max_tokens)
	{
		q.tpl_full = true;
	}

	// ABC_END is masked for the whole hole, so it can only arrive from a sampler
	// that was told otherwise. Treated as a failed attempt rather than trusted.
	if (token == ABC_END)
	{
		fail_hole(q, "ended the score");
		return;
	}

	q.hole_ids.push_back(token);
	const std::string full = detokenize(vocab, q.hole_ids);
	const size_t      nl   = full.find('\n');
	if (nl == std::string::npos)
	{
		if ((int) q.hole_ids.size() >= js.budget[q.seg])
		{
			fail_hole(q, strf("ran past its %d-token budget", js.budget[q.seg]));
			return;
		}
		if (q.tpl_full)
		{
			fail_hole(q, "the job's sampled-token cap is spent");
			return;
		}
		// One more sampled token is one more cell, and one more token of the
		// score the semantic phase has to prefill. Either ceiling closes the hole
		// and rest-fills it rather than failing a decode or dying in
		// template_done (SPEC_TEMPLATE §3).
		if (!abc_fits(q, q.hole_ids.size() + 1))
		{
			fail_hole(q, "ran the slot out of context");
			return;
		}
		if (!sem_fits(q, q.hole_ids.size() + 1))
		{
			fail_hole(q, "would leave the semantic phase no room");
			return;
		}
		q.history.push_back(token);
		q.next = token;
		return;
	}

	// The accepted line is what was decoded plus the closing token's head, which
	// is re-tokenized into the next feed — one token, near enough for the check.
	if (!sem_fits(q, q.hole_ids.size() + 1))
	{
		fail_hole(q, "would leave the semantic phase no room");
		return;
	}

	q.hole_ids.pop_back();
	const std::string prev = q.hole_ids.empty() ? std::string() : detokenize(vocab, q.hole_ids);
	// A chords line is sampled from an opening double quote that was fed, not
	// drawn: the quote is part of the line, not of what the cache has to undo.
	const std::string head = seg.kind == TPL_CHORDS ? std::string("\"") : std::string();
	const std::string line = head + full.substr(0, nl + 1);
	const std::string body = line.substr(0, line.size() - 1);

	if (!is_body_line(body))
	{
		fail_hole(q, "wrote a field or comment line, not a body line");
		return;
	}
	const BarCount bc = count_bars(body, seg.meter);
	if (bc.bars != seg.bars)
	{
		fail_hole(q, strf("got %d bars", bc.bars));
		return;
	}
	if (seg.kind == TPL_CHORDS)
	{
		// Rests carrying harmony, nothing else: at least one chord symbol, and
		// no note left once the chord symbols are out of the way (§2).
		if (body.find('"') == std::string::npos ||
		    body.find('"', body.find('"') + 1) == std::string::npos)
		{
			fail_hole(q, "wrote no chord symbol");
			return;
		}
		const std::string bare = strip_line(body);
		if (bare.find_first_of("ABCDEFGabcdefg") != std::string::npos)
		{
			fail_hole(q, "wrote notes, not a line of rests with chord symbols");
			return;
		}
		js.tpl.offlength_bars += bc.offlength;
		finish_chords(q, line);
		return;
	}

	js.tpl.offlength_bars += bc.offlength;
	js.tpl_text           += line;
	js.emitted            += q.hole_ids.size();   // the carry is counted by give()
	q.carry   = line.substr(prev.size());
	q.in_hole = false;
	q.seg++;
	template_step(q);
}

void Runner::fail_hole(Seq & q, const std::string & why)
{
	JobState &     js  = (*states)[q.job];
	const TplSeg & seg = js.segs[q.seg];

	printf("%sabc: hole %d (bars=%d): attempt %d, %s\n", js.tag.c_str(),
	       js.tpl.holes, seg.bars, q.attempt, why.c_str());

	if (q.attempt >= TEMPLATE_ATTEMPTS || q.tpl_full)
	{
		rest_fill(q);
		return;
	}
	js.tpl.retries++;
	q.attempt++;
	rollback_hole(q);
	template_token(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
}

// Undoes an attempt: the slot's cells, position, history and step go back to
// where the hole started. The RNG is deliberately *not* restored, so the retry
// draws differently. The logits the next attempt samples from are gone with the
// cells, so the token before the hole is decoded a second time (§3).
void Runner::rollback_hole(Seq & q)
{
	llama_memory_seq_rm(mem, q.slot, q.hole_pos - 1, -1);
	q.pos  = q.hole_pos - 1;
	q.step = q.hole_step;
	q.history.resize(q.hole_hist);
	q.hole_ids.clear();

	const std::vector<llama_token> one(1, q.hole_prev);
	feed_tokens(q, one, (size_t) q.hole_pos, "hole rollback");
}

// The fallback after the last attempt: N bars of rest, fed as a given line.
void Runner::rest_fill(Seq & q)
{
	JobState &     js  = (*states)[q.job];
	const TplSeg & seg = js.segs[q.seg];

	js.tpl.rest_filled++;
	printf("%sabc: hole %d (bars=%d): filled with rests\n", js.tag.c_str(),
	       js.tpl.holes, seg.bars);

	// Only here: the phase really did lose a line to `--max-abc`. A cap that
	// lands on the closing token of the last hole costs the score nothing.
	if (q.tpl_full)
	{
		js.st_abc.truncated = true;
	}

	const std::string fill = seg.bars == 1 ? std::string("Z|\n") : strf("Z%d|\n", seg.bars);
	if (seg.kind == TPL_CHORDS)
	{
		// The whole out-of-order feed goes back, not just the attempt, and the
		// rests go in where the directive stood — with no chord symbol: four
		// attempts at the harmony is where the model's opinion of it ends (§2).
		q.step = q.chk_step;
		finish_chords(q, fill);
		return;
	}

	// Nothing to undo when the hole failed on its very first token, or when the
	// sampled-token cap closed it before it opened — but `step` is restored
	// either way, since a rest-fill spends none of the phase's draws.
	if (!q.hole_ids.empty() || q.pos != q.hole_pos)
	{
		rollback_hole(q);
	}
	q.step = q.hole_step;
	js.tpl_text += fill;
	q.in_hole    = false;
	q.seg++;
	give(q, fill, GIVE_SCORE, "rest fill");
	template_step(q);
}

// %%yue2-continue: the template's last segment. What was given so far is the
// score's beginning and the model writes the rest as a plain score phase would
// — no line checks, ABC_END allowed — so a caller can hand in the opening of a
// score, well-formed or not, and read what the model makes of it (§3).
void Runner::start_continue(Seq & q)
{
	// As open_hole: the head of the last hole's closing token is score text the
	// cache has not seen yet.
	if (!q.carry.empty())
	{
		const std::string text = q.carry;
		q.carry.clear();
		give(q, text, GIVE_SCORE, "hole tail");
	}

	// q.seg stays on the continue segment: abc_fits / sem_fits read the suffix
	// sums at q.seg + 1, which is the end of the template (nothing left to feed).
	q.tpl_cont = true;
	q.hole_ids.clear();
	if (q.tpl_full || q.step >= s_abc.max_tokens)
	{
		finish_continue(q, "the job's sampled-token cap is spent");
		return;
	}
	continue_token(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
}

void Runner::continue_token(Seq & q, llama_token token)
{
	JobState & js = (*states)[q.job];

	if (token == ABC_END)
	{
		finish_continue(q, nullptr);
		return;
	}
	js.tpl.sampled_tokens++;
	q.step++;
	q.hole_ids.push_back(token);
	q.history.push_back(token);

	if (q.step >= s_abc.max_tokens)
	{
		finish_continue(q, "the job's sampled-token cap is spent");
		return;
	}
	if (!abc_fits(q, q.hole_ids.size() + 1))
	{
		finish_continue(q, "the slot is out of context");
		return;
	}
	if (!sem_fits(q, q.hole_ids.size() + 1))
	{
		finish_continue(q, "the semantic phase would have no room");
		return;
	}
	q.next = token;
}

// The free tail is over: it becomes the end of the emitted score and the job
// goes on exactly as a template that ran out of segments (template_done).
void Runner::finish_continue(Seq & q, const char * stopped_by)
{
	JobState & js = (*states)[q.job];

	const std::string text = q.hole_ids.empty() ? std::string() : detokenize(vocab, q.hole_ids);
	js.tpl_text            += text;
	js.emitted             += q.hole_ids.size();
	js.tpl.continued_tokens = (int) q.hole_ids.size();
	if (stopped_by != nullptr)
	{
		js.st_abc.truncated = true;
		printf("%sabc: continue: stopped after %zu tokens, %s\n", js.tag.c_str(),
		       q.hole_ids.size(), stopped_by);
	}
	q.hole_ids.clear();
	q.tpl_cont = false;
	q.seg++;
	template_done(q);
}

// The abc phase of a template job ends when the template runs out. The emitted
// score is re-tokenized in one go — BPE merges across the segment seams differ
// from the per-segment ids — and prefilled into a *cleared* slot, exactly as an
// external "abc" would be. That is what lets the primer be dropped (§3).
void Runner::template_done(Seq & q)
{
	JobState & js = (*states)[q.job];
	GenStats & st = js.st_abc;

	js.abc_ids = tokenize(vocab, js.tpl_text, "template score");
	for (size_t i = 0; i < js.abc_ids.size(); i++)
	{
		if (js.abc_ids[i] < 0 || js.abc_ids[i] >= EOD)
		{
			die("%sabc id %d at %zu leaves the ordinary text vocabulary",
			    js.tag.c_str(), (int) js.abc_ids[i], i);
		}
	}
	js.abc_text      = js.tpl_text;
	js.have_abc_text = true;

	// From here the request *is* a plain external-"abc" request for the score the
	// holes produced, and that is what request.json and plan.json record. Nothing
	// downstream reads `has_abc` any more — validation and the phase choice both
	// happened before the decode — so this only changes what is written.
	js.req.has_abc = true;
	js.req.abc     = js.tpl_text;

	st.seconds        = now_seconds() - q.t_phase0;
	st.content_tokens = (int) js.abc_ids.size();
	st.output_tokens  = st.content_tokens + 1;
	st.output_tps     = st.seconds > 0 ? st.output_tokens / st.seconds : 0;

	printf("%sabc:      %d tokens in %.2f s = %.2f tok/s%s\n", js.tag.c_str(),
	       st.output_tokens, st.seconds, st.output_tps, st.truncated ? " (TRUNCATED)" : "");
	printf("%sabc: template: %d holes, %d sampled tokens, %d retries, %d rest-filled, "
	       "%d given + %d primer tokens, %d off-length bars, %d continued, %d chord lines\n",
	       js.tag.c_str(),
	       js.tpl.holes, js.tpl.sampled_tokens, js.tpl.retries, js.tpl.rest_filled,
	       js.tpl.given_tokens, js.tpl.primer_tokens, js.tpl.offlength_bars,
	       js.tpl.continued_tokens, js.tpl.chord_lines);

	js.prefix_sem = semantic_prefix(js.prefix_abc, js.abc_ids);
	if ((int) js.prefix_sem.size() + s_sem.max_tokens > (int) n_ctx_seq)
	{
		die("%ssemantic prefix %zu + max_tokens %d exceeds the allocated %u-token context",
		    js.tag.c_str(), js.prefix_sem.size(), s_sem.max_tokens, n_ctx_seq);
	}

	llama_memory_seq_rm(mem, q.slot, -1, -1);
	const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
	if (held != -1)
	{
		die("slot %d still holds %d tokens after the template phase cleared it",
		    q.slot, (int) held + 1);
	}

	q.pos      = 0;
	q.phase    = PHASE_SEM;
	q.history.clear();
	q.step     = 0;
	q.rng.seed(js.req.seed);
	q.t_phase0 = now_seconds();

	js.st_sem.prefix_tokens = (int) js.prefix_sem.size();
	feed(q, js.prefix_sem, js.prefix_sem.size(), "semantic");
}

void Runner::enter(Seq & q, int job)
{
	JobState & js = (*states)[job];

	// A guided job before this one moved the primary between streams; every one
	// of them was cleared when it finished, and this one starts at home again.
	q.slot = q.home;

	const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
	if (held != -1)
	{
		die("slot %d still holds %d tokens before job %d enters it",
		    q.slot, (int) held + 1, job + 1);
	}

	q.job      = job;
	q.guided   = false;
	q.phase    = js.do_abc ? PHASE_ABC : PHASE_SEM;
	q.history.clear();
	q.step     = 0;
	q.rng.seed(js.req.seed);
	q.pos      = 0;
	q.i_batch  = -1;
	q.t_phase0 = now_seconds();
	q.seg      = 0;
	q.in_hole  = false;
	q.tpl_cont = false;
	q.tpl_full = false;
	q.hole_ids.clear();
	q.carry.clear();
	q.clock    = ScoreClock();
	q.line_ids.clear();
	q.line_pre.clear();
	q.sec_next = 0;
	q.sec_cuts = 0;
	active++;

	if (js.do_abc)
	{
		// Unreachable: "semantic_keep" needs the score in "abc" (§2), and that
		// is exactly what clears do_abc. The backstop is here because the keep
		// would otherwise be dropped in silence while plan.json still claimed it.
		if (!js.keep_codes.empty())
		{
			die("%skept codes belong to a score this job is about to write itself",
			    js.tag.c_str());
		}
		js.st_abc.prefix_tokens = (int) js.prefix_abc.size();
		if (js.is_template)
		{
			// The prefix is fed but nothing is sampled from it: the template says
			// what comes next, and it is usually a given line (SPEC_TEMPLATE §3).
			feed_tokens(q, js.prefix_abc, js.prefix_abc.size(), "abc");
			js.st_abc.prefill_seconds = now_seconds() - q.t_phase0;
			template_step(q);
			return;
		}
		feed(q, js.prefix_abc, js.prefix_abc.size(), "abc");
		return;
	}
	if (js.req.cot == "off")
	{
		printf("%sabc:      skipped (cot=off)\n", js.tag.c_str());
	} else {
		printf("%sabc:      %zu externally provided tokens\n", js.tag.c_str(), js.abc_ids.size());
	}
	js.st_sem.prefix_tokens = (int) js.prefix_sem.size();
	if (!js.keep_codes.empty())
	{
		keep_enter(q);
		return;
	}
	guidance_enter(q);
	feed(q, js.prefix_sem, js.prefix_sem.size(), "semantic");
}

// SPEC_KEEP §3: steps 0 .. N-1 are not sampled. The slot is prefilled with the
// semantic prefix and every kept code but the last, which is left pending in
// `q.next` — exactly the state a sampled step leaves behind, so the lockstep
// batch feeds it to the primary and to every branch at once and the first row
// it yields is step N's. No row is read here, so the rng makes no draw for a
// kept step; the penalty window is `history`, which holds the kept codes as if
// they had been drawn.
void Runner::keep_enter(Seq & q)
{
	JobState &   js = (*states)[q.job];
	const double t0 = now_seconds();

	q.history.clear();
	q.history.reserve(js.keep_codes.size());
	std::vector<llama_token> tokens = js.prefix_sem;
	for (size_t i = 0; i < js.keep_codes.size(); i++)
	{
		const llama_token id = (llama_token) js.keep_codes[i] + CODEC_OFFSET;
		q.history.push_back(id);
		if (i + 1 < js.keep_codes.size())
		{
			tokens.push_back(id);
		}
	}
	// One call, chunked by the batch size like any long prefix (decode_feed).
	feed_tokens(q, tokens, tokens.size(), "semantic");
	q.step = (int) js.keep_codes.size();
	q.next = q.history.back();

	js.keep_seconds           = now_seconds() - t0;
	js.st_sem.prefill_seconds = now_seconds() - q.t_phase0;
	printf("%skeep: %d frames from %s prefilled (%zu + %d tokens, %.2f s)\n",
	       js.tag.c_str(), q.step, js.keep_name.c_str(), js.prefix_sem.size(), q.step,
	       js.keep_seconds);

	// After the primary, not before it as an unkept song does: an entry at
	// frame N is a cut, and a cut relabels the sequence the song has been
	// decoding, which at this point is the one just prefilled (§3). Nothing is
	// sampled here, so the rows those prefills leave behind are stale by
	// design — run() re-fetches every live branch's row before it samples.
	guidance_enter(q);
}

// One abc token of a job that carries a "sections": the clock is fed whole
// lines, and a completed label line the next entry names is a cut (§3). `nth` is
// counted over every label line of that name, so the intro and interlude labels
// the planner inserts on its own are stepped over rather than counted.
void Runner::sections_token(Seq & q, llama_token token)
{
	JobState & js = (*states)[q.job];
	q.line_ids.push_back(token);

	// As the template's hole does it: the line is the detokenization of the ids
	// drawn since the last newline, plus whatever the token that closed it left
	// behind. One token can close a line and open the next.
	std::string text = q.line_pre + detokenize(vocab, q.line_ids);
	if (text.find('\n') == std::string::npos)
	{
		return;
	}
	for (size_t nl = text.find('\n'); nl != std::string::npos; nl = text.find('\n'))
	{
		const std::string name = clock_line(q.clock, text.substr(0, nl));
		text.erase(0, nl + 1);
		if (q.sec_next >= js.sections.size() || name != js.sections[q.sec_next].section ||
		    q.clock.seen[name] != js.sections[q.sec_next].nth)
		{
			continue;
		}
		SectionEntry & s = js.sections[q.sec_next];
		s.found   = true;
		s.line    = q.clock.line;
		s.bar     = q.clock.bars;
		s.seconds = q.clock.seconds;
		sections_cut(q, q.sec_next++);
	}
	q.line_ids.clear();
	q.line_pre = text;
}

// The cut of SPEC_SECTIONS §3: the score so far is re-prefilled under the
// entry's own tags and sampling goes on, so the label line was written under the
// old tags and everything after it under the new ones. The slot is cleared and
// refilled the way the template hand-over does it, and it keeps the invariant
// every prefill in this file keeps — prefix + `history` minus the token still
// pending in `q.next`. No RNG reset, no extra draw.
void Runner::sections_cut(Seq & q, size_t idx)
{
	JobState &           js = (*states)[q.job];
	const SectionEntry & s  = js.sections[idx];
	const double         t0 = now_seconds();

	std::vector<llama_token> feed = js.heads[idx];
	const size_t             head = feed.size();
	feed.insert(feed.end(), q.history.begin(), q.history.end() - 1);

	// Unreachable once the sizing guard holds: n_ctx_want is built on the longest
	// entry head plus the whole abc cap (§4.5). The backstop turns a sizing bug
	// into a message instead of a -1 from llama_decode.
	if (feed.size() + 1 > (size_t) n_ctx_seq)
	{
		die("%ssections: the cut at line %d needs %zu of the %u tokens this slot has",
		    js.tag.c_str(), s.line, feed.size() + 1, n_ctx_seq);
	}

	llama_memory_seq_rm(mem, q.slot, -1, -1);
	const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
	if (held != -1)
	{
		die("%ssections: slot %d still holds %d tokens after the cut cleared it",
		    js.tag.c_str(), q.slot, (int) held + 1);
	}
	q.pos = 0;
	feed_tokens(q, feed, feed.size(), "sections cut");

	q.sec_cuts++;
	js.sec_seconds += now_seconds() - t0;
	printf("%sabc: line %d: %% %s %d (bar %d, %.2f s): the score goes on under its own "
	       "tags (%zu + %zu tokens, %.2f s)\n", js.tag.c_str(), s.line, s.section.c_str(),
	       s.nth, s.bar, s.seconds, head, feed.size() - head, now_seconds() - t0);
}

void Runner::finish_job(Seq & q)
{
	JobState & js = (*states)[q.job];
	printf("%ssemantic: %d tokens in %.2f s = %.2f tok/s%s\n", js.tag.c_str(),
	       js.st_sem.output_tokens, js.st_sem.seconds, js.st_sem.output_tps,
	       js.st_sem.truncated ? " (TRUNCATED)" : "");

	std::vector<int32_t> codes;
	codes.reserve(q.history.size());
	for (size_t i = 0; i < q.history.size(); i++)
	{
		const int code = (int) q.history[i] - CODEC_OFFSET;
		if (code < 0 || code >= CODEC_SIZE)
		{
			die("%ssemantic id %d at %zu is not a codec token",
			    js.tag.c_str(), (int) q.history[i], i);
		}
		codes.push_back((int32_t) code);
	}

	// What the branches cost this song, on both phases' timing blocks: they
	// describe the song, and the semantic phase is the only one they touch
	// (SPEC_GUIDANCE §3).
	js.st_abc.cfg_branches = js.max_branches;
	js.st_sem.cfg_branches = js.max_branches;
	js.st_abc.guided_steps = js.guided_steps;
	js.st_sem.guided_steps = js.guided_steps;
	js.st_abc.branch_prefill_seconds = js.branch_seconds;
	js.st_sem.branch_prefill_seconds = js.branch_seconds;
	js.st_abc.kept_frames = (int) js.keep_codes.size();
	js.st_sem.kept_frames = (int) js.keep_codes.size();
	js.st_abc.keep_prefill_seconds = js.keep_seconds;
	js.st_sem.keep_prefill_seconds = js.keep_seconds;
	js.st_abc.section_cuts = q.sec_cuts;
	js.st_sem.section_cuts = q.sec_cuts;
	js.st_abc.section_prefill_seconds = js.sec_seconds;
	js.st_sem.section_prefill_seconds = js.sec_seconds;

	if (!js.no_files)
	{
		Artifacts a;
		a.dir           = (*jobs)[q.job].artifacts;
		a.req           = &js.req;
		a.st_abc        = &js.st_abc;
		a.abc_ids       = js.abc_ids;
		a.prefix_sem    = js.prefix_sem;
		a.codes         = codes;
		a.abc_text      = js.abc_text;
		a.have_abc_text = js.have_abc_text;
		a.template_text = js.is_template ? js.req.abc_template : std::string();
		a.guidance      = js.req.has_guidance || js.req.has_sections ? &js.guide : nullptr;
		a.sections      = js.req.has_sections ? &js.sections : nullptr;
		a.keep          = js.keep_codes.empty() ? nullptr : &js.keep_codes;
		a.trace         = (*jobs)[q.job].trace && !js.guide.empty() ? &js.trace : nullptr;
		write_artifacts(a);
		printf("%sartifacts: %s (abc %zu ids, semantic %zu codes)\n", js.tag.c_str(),
		       (*jobs)[q.job].artifacts.c_str(), js.abc_ids.size(), codes.size());
	} else {
		printf("%s%zu codes, handed over in memory\n", js.tag.c_str(), codes.size());
	}
	if ((*jobs)[q.job].trace)
	{
		const size_t rows = js.trace.size() / TRACE_COLUMNS;
		if (js.guide.empty())
		{
			printf("%sguidance trace: this request has no guidance, nothing to trace\n",
			       js.tag.c_str());
		} else {
			printf("%sguidance trace: %zu rows x %d in guidance_trace.npy (%.3f ms/row)\n",
			       js.tag.c_str(), rows, TRACE_COLUMNS,
			       rows > 0 ? 1000 * js.trace_seconds / (double) rows : 0);
		}
	}

	ArResult & r  = (*results)[q.job];
	r.prefix_sem.assign(js.prefix_sem.begin(), js.prefix_sem.end());
	r.codes       = codes;
	r.abc         = js.st_abc;
	r.semantic    = js.st_sem;
	r.seed        = js.req.seed;
	r.cot         = js.req.cot;
	r.cfg_scale   = js.guidance;
	r.card        = card;
	r.ok          = true;
	r.is_template = js.is_template;
	r.tpl         = js.tpl;
	r.parallel    = parallel;
	r.slot        = q.home;
	r.batch_jobs  = (int) jobs->size();

	// The shadow slots first: the next job of a --parallel 1 batch has to enter
	// a context with nothing of this one left in it (§4.1).
	guidance_clear(q);

	// The whole sequence, so this never fails (llama.h:745-747); the
	// postcondition is what the next enter() depends on.
	llama_memory_seq_rm(mem, q.slot, -1, -1);
	const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
	if (held != -1)
	{
		die("slot %d still holds %d tokens after it was cleared", q.slot, (int) held + 1);
	}
	q.pos   = 0;
	q.job   = -1;
	q.phase = PHASE_DONE;
	active--;
}

void Runner::run()
{
	t_loop0    = now_seconds();
	t_progress = t_loop0;

	while (true)
	{
		// Refill from the queue first, so a hole in the slot set only survives
		// the drain phase. enter() can finish a whole job on its own when the
		// token limits are tiny, hence the rescan.
		bool filled = true;
		while (filled)
		{
			filled = false;
			for (size_t i = 0; i < seqs.size(); i++)
			{
				if (seqs[i].job >= 0)
				{
					continue;
				}
				while (queued < jobs->size() && !(*states)[queued].ok)
				{
					queued++;      // rejected by validation; never enters a slot
				}
				if (queued >= jobs->size())
				{
					break;
				}
				enter(seqs[i], (int) queued++);
				filled = true;
				break;
			}
		}
		if (active == 0)
		{
			break;
		}

		// A guided sequence takes its entries and settles which branches are live
		// before the batch is built — a branch prefill is its own decode call,
		// never part of the lockstep step (§4.3).
		for (size_t i = 0; i < seqs.size(); i++)
		{
			if (seqs[i].job >= 0 && seqs[i].guided && seqs[i].phase == PHASE_SEM)
			{
				guidance_step(seqs[i]);
			}
		}

		batch.n_tokens = 0;
		int first = -1;
		int last  = -1;
		for (size_t i = 0; i < seqs.size(); i++)
		{
			Seq & q = seqs[i];
			if (q.job < 0)
			{
				continue;
			}
			q.i_batch = batch.n_tokens;
			batch_add(batch, q.next, q.pos++, q.slot, true);
			first = first < 0 || q.slot < first ? q.slot : first;
			last  = std::max(last, q.slot);
			// The same token into every live shadow branch, each at its own
			// position: one history, several prefixes (§2.1).
			for (int k = 0; k < BRANCH_KINDS; k++)
			{
				Branch & b = q.branch[k];
				if (!b.live)
				{
					continue;
				}
				b.i_batch = batch.n_tokens;
				batch_add(batch, q.next, b.pos++, b.slot, true);
				first = b.slot < first ? b.slot : first;
				last  = std::max(last, b.slot);
			}
		}

		const double t0  = now_seconds();
		const int    ret = llama_decode(ctx, batch);
		if (ret != 0)
		{
			die("llama_decode returned %d on a lockstep batch of %d slots",
			    ret, batch.n_tokens);
		}
		const double dt = now_seconds() - t0;
		decodes++;
		if (last - first + 1 == batch.n_tokens)
		{
			steps_whole++;
			time_whole += dt;
		} else {
			steps_holed++;
			time_holed += dt;
		}

		// Sample every slot before applying any of them: an apply() that bridges
		// a phase decodes, and that invalidates the logits of every other slot.
		for (size_t i = 0; i < seqs.size(); i++)
		{
			if (seqs[i].job < 0)
			{
				continue;
			}
			// Every row of this decode, the branches' included, before any of it
			// is applied (SPEC_BATCH §4.7).
			for (int k = 0; k < BRANCH_KINDS; k++)
			{
				if (seqs[i].branch[k].live)
				{
					fetch_row(seqs[i].branch[k]);
				}
			}
			const float * primary = llama_get_logits_ith(ctx, seqs[i].i_batch);
			seqs[i].sampled       = sample(seqs[i], blend_row(seqs[i], primary));
			if (seqs[i].traced)
			{
				trace_step(seqs[i], primary, seqs[i].sampled);
			}
		}
		for (size_t i = 0; i < seqs.size(); i++)
		{
			if (seqs[i].job >= 0)
			{
				apply(seqs[i], seqs[i].sampled);
			}
		}

		progress();
	}
}

// One batch-level line every PROGRESS_SECONDS of wall time, so a driver reading
// a pipe can see the loop is alive. Never per step, and nothing at all from a
// batch that finishes inside the first interval.
void Runner::progress()
{
	const double now = now_seconds();
	if (now - t_progress < PROGRESS_SECONDS)
	{
		return;
	}
	t_progress = now;

	int abc = 0;
	int sem = 0;
	for (size_t i = 0; i < seqs.size(); i++)
	{
		if (seqs[i].job < 0)
		{
			continue;
		}
		if (seqs[i].phase == PHASE_ABC)
		{
			abc++;
		} else {
			sem++;
		}
	}

	const double elapsed = now - t_loop0;
	printf("ar: %.0f s, %d active (%d abc, %d semantic), %lld tokens, %.0f tok/s\n",
	       elapsed, abc + sem, abc, sem, sampled, elapsed > 0 ? sampled / elapsed : 0);
}

// Runs the ABC prefix only and writes the last-position logits.
static void run_dump_logits(llama_context * ctx, llama_batch & batch, int n_vocab,
	const ArParams & p, const std::vector<llama_token> & feed)
{
	const double t0  = now_seconds();
	llama_pos    pos = 0;
	decode_feed(ctx, batch, feed, 0, pos, "abc");
	const float * logits = llama_get_logits_ith(ctx, -1);

	int   best = 0;
	float top  = logits[0];
	for (int i = 1; i < n_vocab; i++)
	{
		if (logits[i] > top)
		{
			top  = logits[i];
			best = i;
		}
	}
	const std::vector<int64_t> shape = { (int64_t) n_vocab };
	const std::string          err   = npy::save(p.dump_logits.c_str(), shape, logits);
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
	printf("prefill: %.3f s, argmax %d (%.6f)\n", now_seconds() - t0, best, (double) top);
	printf("wrote %s [%d] float32\n", p.dump_logits.c_str(), n_vocab);
}

// `--dump-logits FILE.npy` writes the branch rows beside it as
// `FILE.primary.npy`, `FILE.blank.npy`, ... (SPEC_GUIDANCE §3).
static std::string dump_sibling(const std::string & path, const char * what)
{
	const size_t npy = path.size() >= 4 && path.compare(path.size() - 4, 4, ".npy") == 0
	                   ? path.size() - 4 : path.size();
	return path.substr(0, npy) + "." + what + ".npy";
}

static void save_row_or_die(const std::string & path, int n_vocab, const float * row)
{
	const std::vector<int64_t> shape = { (int64_t) n_vocab };
	const std::string          err   = npy::save(path.c_str(), shape, row);
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
	printf("wrote %s [%d] float32\n", path.c_str(), n_vocab);
}

// `yue2 ar --dump-logits`: the goldens' path, one sequence, unchanged.
static int run_ar_dump(const ArParams & p)
{
	Request req;
	double  guidance = 1.0;
	{
		const std::string err = prepare_request(p.request_path, p.cot, p.has_seed, p.seed,
		                                        req, guidance);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
	}
	if (req.has_keep)
	{
		// The row this flag dumps is the *first* semantic step, and with codes
		// kept that step is N — a prefill of the whole kept stretch, which is
		// what --artifacts is for.
		die("--dump-logits dumps the first semantic step, which \"semantic_keep\" moves to "
		    "frame %d; run the request with --artifacts instead", req.keep_frames);
	}

	llama_log_set(quiet_log, nullptr);
	llama_backend_init();

	ggml_backend_dev_t devices[2] = {nullptr, nullptr};
	std::string        backend_name;
	llama_model *      model = load_model(p.model, p.device, p.gpu, backend_name, devices);

	const llama_vocab * vocab   = llama_model_get_vocab(model);
	const int           n_vocab = llama_vocab_n_tokens(vocab);
	if (n_vocab != VOCAB_SIZE)
	{
		die("vocab is %d, the protocol requires exactly %d — wrong GGUF?", n_vocab, VOCAB_SIZE);
	}

	const std::vector<llama_token> prefix_abc = prefix_head(vocab, req.text());

	// Guided: the row worth dumping is the *blended* one of the first semantic
	// step, so the prefixes are the semantic ones and every live branch is
	// prefilled beside the positive sequence (SPEC_GUIDANCE §3).
	const bool                            guided = is_guided(req);
	const std::vector<GuidanceEntry>      plan   = guidance_plan(req);
	std::vector<std::vector<llama_token>> feeds;
	std::vector<int>                      kinds;   // parallel to feeds, -1 = the positive one
	if (!guided)
	{
		feeds.push_back(prefix_abc);
		kinds.push_back(-1);
	} else {
		if (req.cot != "off" && !req.has_abc)
		{
			die("--dump-logits on a guided request dumps the first *semantic* step, so the "
			    "score has to be in the request's \"abc\" (or cot=off) — writing one is a "
			    "whole decode, which is what --artifacts is for");
		}
		const std::vector<llama_token> abc_ids = req.cot == "off"
		                                         ? std::vector<llama_token>()
		                                         : tokenize(vocab, req.abc, "external abc");
		feeds.push_back(semantic_prefix(prefix_abc, abc_ids));
		kinds.push_back(-1);
		// Only an entry at frame 0 is in force at step 0, and §2.3 keeps
		// "previous" out of one: in practice that is the blank branch alone.
		if (!plan.empty() && plan[0].frame == 0)
		{
			for (int k = 0; k < BRANCH_KINDS; k++)
			{
				if (plan[0].has[k] && curve_needed_from(plan[0].curve[k], 0))
				{
					feeds.push_back(k == BRANCH_BLANK ? blank_prefix(vocab, req, abc_ids)
					                                  : semantic_prefix(prefix_abc, abc_ids));
					kinds.push_back(k);
				}
			}
		}
	}

	size_t longest = 0;
	for (size_t i = 0; i < feeds.size(); i++)
	{
		longest = std::max(longest, feeds[i].size());
	}

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx     = (uint32_t) (longest + 4) * (uint32_t) feeds.size();
	cparams.n_batch   = (uint32_t) std::max<size_t>(longest, 512);
	cparams.n_ubatch  = cparams.n_batch;
	cparams.n_seq_max = (uint32_t) feeds.size();
	cparams.no_perf   = true;
	if (p.threads > 0)
	{
		cparams.n_threads       = p.threads;
		cparams.n_threads_batch = p.threads;
	}

	llama_context * ctx = llama_init_from_model(model, cparams);
	if (ctx == nullptr)
	{
		die("failed to create the llama context");
	}
	printf("context: %u tokens, %s prefix %zu tokens, batch %u\n",
	       cparams.n_ctx, guided ? "semantic" : "abc", feeds[0].size(), llama_n_batch(ctx));

	llama_batch batch = llama_batch_init((int32_t) llama_n_batch(ctx), 0, 1);
	if (!guided)
	{
		run_dump_logits(ctx, batch, n_vocab, p, prefix_abc);
	} else {
		// Branches first, the positive sequence last: each row is copied out as
		// it is produced, since the next llama_decode overwrites the buffer.
		const double                    t0 = now_seconds();
		std::vector<std::vector<float>> rows(feeds.size());
		for (size_t i = feeds.size(); i > 0; i--)
		{
			llama_pos     pos    = 0;
			const int     last   = decode_feed(ctx, batch, feeds[i - 1], (int) i - 1, pos,
			                                   kinds[i - 1] < 0 ? "primary" : branch_name(kinds[i - 1]));
			const float * logits = llama_get_logits_ith(ctx, last);
			if (logits == nullptr)
			{
				die("llama_get_logits_ith(%d) returned NULL for the %s prefix", last,
				    kinds[i - 1] < 0 ? "primary" : branch_name(kinds[i - 1]));
			}
			rows[i - 1].assign(logits, logits + n_vocab);
		}

		// The blend of §2.1 over the ids the semantic sampler can visit; outside
		// them the file keeps the positive row, so one .npy shows both.
		std::vector<float> blended = rows[0];
		for (int i = 0; i < SEM_ROW_LEN; i++)
		{
			const float b   = rows[0][(size_t) (SEM_ROW_FIRST + i)];
			float       acc = b;
			for (size_t k = 1; k < rows.size(); k++)
			{
				const float w = (float) curve_at(plan[0].curve[kinds[k]], 0);
				acc += w * (b - rows[k][(size_t) (SEM_ROW_FIRST + i)]);
			}
			blended[(size_t) (SEM_ROW_FIRST + i)] = acc;
		}
		printf("prefill: %.3f s, %zu branch%s\n", now_seconds() - t0, rows.size() - 1,
		       rows.size() == 2 ? "" : "es");
		save_row_or_die(p.dump_logits, n_vocab, blended.data());
		for (size_t i = 0; i < rows.size(); i++)
		{
			save_row_or_die(dump_sibling(p.dump_logits, i == 0 ? "primary" : branch_name(kinds[i])),
			                n_vocab, rows[i].data());
		}
	}

	llama_batch_free(batch);
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return 0;
}

// The per-phase sampling limits the command line asks for: the caps count kept
// and forced steps like any other (SPEC_KEEP §3), so they are settled before
// anything is validated against them.
static void sampling_caps(const ArBatchParams & p, Sampling & s_abc, Sampling & s_sem)
{
	if (p.max_abc > 0)
	{
		s_abc.max_tokens = p.max_abc;
		s_abc.min_tokens = std::min(s_abc.min_tokens, s_abc.max_tokens);
	}
	if (p.max_semantic > 0)
	{
		s_sem.max_tokens = p.max_semantic;
		s_sem.min_tokens = std::min(s_sem.min_tokens, s_sem.max_tokens);
	}
	if (p.greedy)
	{
		s_abc.temperature = 0;
		s_sem.temperature = 0;
	}
}

// The semantic phase's effective frame cap: what the command line asked for, or
// the protocol's own when it asked for nothing. What a "semantic_keep" and a
// handover leg are held to before anything is decoded.
static int semantic_cap(const ArBatchParams & p)
{
	return p.max_semantic > 0 ? p.max_semantic : sampling_semantic().max_tokens;
}

// Everything after validation: the prefixes, the context sized on them, and the
// decode loop. The model is a parameter and is neither loaded nor freed here,
// so one process can run several generations through one copy of the weights —
// which is what a handover does with its takes and legs (SPEC_HANDOVER §3).
// `states` is already prepared: requests parsed, kept codes read, jobs that
// were rejected marked. Every generation builds its own context, so a run
// through this function is exactly what `yue2 ar` does for the same jobs.
static int ar_decode_jobs(const ArBatchParams & p, const std::vector<ArJob> & jobs,
	std::vector<JobState> & states, std::vector<ArResult> & results, llama_model * model,
	const llama_vocab * vocab, const std::string & card, int parallel, int rejected)
{
	const int n_vocab = llama_vocab_n_tokens(vocab);
	Sampling  s_abc   = sampling_abc();
	Sampling  s_sem   = sampling_semantic();
	sampling_caps(p, s_abc, s_sem);

	// ---- prefixes ----------------------------------------------------------

	uint32_t n_ctx_want = 0;
	size_t   max_feed   = 0;
	for (size_t i = 0; i < jobs.size(); i++)
	{
		JobState & js = states[i];
		if (!js.ok)
		{
			continue;
		}

		const std::string text = js.req.text();
		bool              ascii = true;
		for (size_t k = 0; k < text.size(); k++)
		{
			if ((unsigned char) text[k] >= 0x80)
			{
				ascii = false;
				break;
			}
		}
		if (!ascii)
		{
			fprintf(stderr, "%snote: request text is non-ASCII and is assumed to be NFC "
			                "already (no normalisation is performed)\n", js.tag.c_str());
		}

		js.prefix_abc = prefix_head(vocab, text);

		js.do_abc = js.req.cot != "off" && !js.req.has_abc;

		// A template: the segments, and the token budget of every hole. Both are
		// needed before the context is sized, since the abc phase holds the prefix,
		// every forced token and the hole being written (SPEC_TEMPLATE §3).
		size_t tpl_fed    = 0;   // given + primer tokens
		size_t tpl_emit   = 0;   // of those, the ones that reach score.abc
		size_t tpl_lines  = 0;   // what the holes are expected to write, all told
		size_t tpl_budget = 0;   // the largest single hole budget
		if (js.req.has_tpl)
		{
			js.is_template = true;
			const std::string err = parse_template(js.req.abc_template, js.segs);
			if (!err.empty())
			{
				die("%s%s: \"abc_template\": %s", js.tag.c_str(),
				    jobs[i].request_path.c_str(), err.c_str());
			}
			js.budget.assign(js.segs.size(), 0);
			js.tail_fed.assign(js.segs.size() + 1, 0);
			js.tail_given.assign(js.segs.size() + 1, 0);
			js.tail_holes.assign(js.segs.size() + 1, 0);
			for (size_t k = 0; k < js.segs.size(); k++)
			{
				const TplSeg & seg = js.segs[k];
				if (seg.kind == TPL_CONTINUE)
				{
					// The free tail may run to the phase's own cap, as a plain
					// score would; the guards in continue_token stop it earlier
					// if the slot is smaller than that.
					tpl_lines += (size_t) s_abc.max_tokens;
					continue;
				}
				if (seg.kind != TPL_HOLE && seg.kind != TPL_CHORDS)
				{
					const size_t n = tokenize(vocab, seg.text, "template segment").size();
					tpl_fed += n;
					js.tail_fed[k] = (int) n;
					if (seg.kind == TPL_GIVEN)
					{
						tpl_emit          += n;
						js.tail_given[k]   = (int) n;
					}
					max_feed = std::max(max_feed, n);
					continue;
				}
				js.tail_holes[k] = 1;
				const size_t above = seg.above.empty()
				                     ? 0 : tokenize(vocab, seg.above, "template line").size();
				js.budget[k] = (int) std::min<size_t>(std::max<size_t>(64, 8 * above),
				                                      (size_t) s_abc.max_tokens);
				if (seg.kind == TPL_CHORDS)
				{
					// Its three given lines are fed twice — once out of order to
					// write the line from, once in template order — but never at
					// the same time: the first feed is rolled back. So they are
					// counted once, like any given segment, and the peak is the
					// same as a hole's.
					const size_t n = tokenize(vocab, seg.head, "template segment").size() +
					                 tokenize(vocab, seg.text, "template segment").size();
					tpl_fed          += n;
					tpl_emit         += n;
					js.tail_fed[k]    = (int) n;
					js.tail_given[k]  = (int) n;
					max_feed = std::max(max_feed, n + (size_t) js.budget[k]);
				}
				// The budget is what a hole may *spend*; what it is expected to
				// *keep* is a line about as long as the one above it. Sizing the
				// context on the budgets would ask for an order of magnitude more
				// KV than any real template uses, so the expectation is what the
				// estimate below uses and template_token guards the difference.
				tpl_lines  += std::min<size_t>((size_t) js.budget[k],
				                               std::max<size_t>(64, 2 * above + 16));
				tpl_budget  = std::max(tpl_budget, (size_t) js.budget[k]);
			}
			for (size_t k = js.segs.size(); k > 0; k--)
			{
				js.tail_fed[k - 1]   += js.tail_fed[k];
				js.tail_given[k - 1] += js.tail_given[k];
				js.tail_holes[k - 1] += js.tail_holes[k];
			}
			// A primer works as specified and is kept for that; it just rarely
			// does what it was built for. A hole under a resting vocal line still
			// sees that line's chord symbols, so an intro hole already knows the
			// tune's harmony bar by bar. What a primer adds is the melody, and the
			// model quotes it rather than introducing it.
			if (tpl_fed > tpl_emit)
			{
				fprintf(stderr, "%swarning: the template has a primer block. Shown a section's "
				        "melody ahead of a hole, the model tends to quote it, so an intro primed "
				        "with the verse sounds like an interlude. The chord symbols kept on the "
				        "intro's resting vocal line already tie it to the tune; try without.\n",
				        js.tag.c_str());
			}
		}

		// External ABC: tokenize it here so the semantic prefix matches torch.
		if (js.req.cot != "off" && js.req.has_abc)
		{
			js.abc_ids       = tokenize(vocab, js.req.abc, "external abc");
			js.abc_text      = detokenize(vocab, js.abc_ids);
			js.have_abc_text = true;
			for (size_t k = 0; k < js.abc_ids.size(); k++)
			{
				if (js.abc_ids[k] < 0 || js.abc_ids[k] >= EOD)
				{
					die("%sabc id %d at %zu leaves the ordinary text vocabulary",
					    js.tag.c_str(), (int) js.abc_ids[k], k);
				}
			}
		}

		if (js.do_abc && (int) js.prefix_abc.size() + s_abc.max_tokens > CONTEXT)
		{
			die("%sabc prefix %zu + max_tokens %d exceeds the %d context; "
			    "no implicit truncation",
			    js.tag.c_str(), js.prefix_abc.size(), s_abc.max_tokens, CONTEXT);
		}

		// A job that skips the abc phase enters directly in SEMANTIC, and its
		// whole semantic prefix is one prefill.
		const size_t sem_prefix_len = js.prefix_abc.size() + js.abc_ids.size() + 2;
		if (!js.do_abc)
		{
			js.prefix_sem = semantic_prefix(js.prefix_abc, js.abc_ids);
			max_feed = std::max(max_feed, sem_prefix_len);
		}
		max_feed = std::max(max_feed, js.prefix_abc.size());

		if ((int) sem_prefix_len + s_sem.max_tokens > CONTEXT && !js.do_abc)
		{
			die("%ssemantic prefix %zu + max_tokens %d exceeds the %d context; "
			    "no implicit truncation",
			    js.tag.c_str(), sem_prefix_len, s_sem.max_tokens, CONTEXT);
		}

		// The guidance plan and the prefix head of every branch it can open
		// (SPEC_GUIDANCE §4.4/§4.5). Only the head differs between the positive
		// sequence and a branch — the score and the two closing tokens are the
		// same — so the longest head is what the context has to be sized on.
		js.guide = js.req.has_sections ? sections_plan(js.sections) : guidance_plan(js.req);
		js.heads.assign(js.guide.size(), std::vector<llama_token>());
		size_t head_max = js.prefix_abc.size();
		if (!js.guide.empty())
		{
			// [EOD] + the instruction on its own + [ABC_START], which is what
			// blank_prefix builds; cot=off's is shorter still.
			head_max = std::max(head_max,
			                    tokenize(vocab, instruction(js.req.cot),
			                             "negative instruction").size() + 2);
			for (size_t k = 0; k < js.guide.size(); k++)
			{
				if (!js.guide[k].has_style)
				{
					continue;
				}
				Request alt = js.req;      // only the tags change: same lyrics, same score
				alt.style   = js.guide[k].style;
				js.heads[k] = prefix_head(vocab, alt.text());
				head_max    = std::max(head_max, js.heads[k].size());
			}
			const size_t longest = head_max + js.abc_ids.size() +
			                       (js.do_abc ? (size_t) s_abc.max_tokens : 0) + 2;
			if ((int) longest + s_sem.max_tokens > CONTEXT)
			{
				die("%sguidance: a branch prefix of %zu tokens + max_tokens %d exceeds the "
				    "%d context; shorten the style text of the entries",
				    js.tag.c_str(), longest, s_sem.max_tokens, CONTEXT);
			}
			if (!js.do_abc)
			{
				// As for the positive sequence: a job that skips the abc phase
				// prefills a whole prefix in one go. A job that writes its score
				// prefills the branches through decode_feed's chunking instead,
				// so n_batch stays where the unguided path put it.
				max_feed = std::max(max_feed, longest);
			}
		}

		// A score the request gave is located already, so the plan is final here
		// — before the context exists. One the model writes is compiled at the
		// end of its abc phase instead (Runner::apply).
		if (js.req.has_sections && !js.do_abc)
		{
			sections_compile(js);
		}

		uint32_t want = (uint32_t) (head_max + js.abc_ids.size() +
		                            (js.do_abc ? (size_t) s_abc.max_tokens : 0) + 2 +
		                            (size_t) s_sem.max_tokens + 8);
		if (js.is_template)
		{
			// A template job's two phases are a max, not a sum: the semantic phase
			// starts from a cleared slot (SPEC_TEMPLATE §3). The abc phase holds
			// the prefix, every forced token and one hole at a time; the emitted
			// score is at most the given tokens plus the hole budgets.
			const size_t sem_est = js.prefix_abc.size() + tpl_emit + tpl_lines + 2;
			const size_t abc_est = js.prefix_abc.size() + tpl_fed + tpl_lines + tpl_budget;
			max_feed = std::max(max_feed, sem_est);
			want     = (uint32_t) (std::max(abc_est, sem_est + (size_t) s_sem.max_tokens) + 8);
			if (want > (uint32_t) CONTEXT)
			{
				die("%sthe template needs %u tokens of context, the model has %d",
				    js.tag.c_str(), want, CONTEXT);
			}
		}
		if (want > (uint32_t) CONTEXT)
		{
			want = (uint32_t) CONTEXT;
		}
		n_ctx_want = std::max(n_ctx_want, want);
	}

	// ---- --prefix-only -----------------------------------------------------
	// The NAR's view of a request: [EOD] text <abc> abc </abc><music>. Complete
	// before any decode when the score is given, so a finished song's codes can
	// be re-rendered under other request text (same score, other style).
	if (p.prefix_only)
	{
		for (size_t i = 0; i < jobs.size(); i++)
		{
			const JobState & js = states[i];
			if (!js.ok)
			{
				continue;
			}
			if (js.do_abc)
			{
				die("%s--prefix-only needs the score in the request's \"abc\" (or cot off); "
				    "generating one is a decode", js.tag.c_str());
			}
			std::error_code ec;
			std::filesystem::create_directories(jobs[i].artifacts, ec);
			if (ec)
			{
				die("cannot create %s: %s", jobs[i].artifacts.c_str(), ec.message().c_str());
			}
			save_i32_or_die(jobs[i].artifacts + "/prefix.npy",
			                std::vector<int32_t>(js.prefix_sem.begin(), js.prefix_sem.end()));
			printf("%swrote:   %s/prefix.npy [%zu] (text %zu, abc %zu ids)\n", js.tag.c_str(),
			       jobs[i].artifacts.c_str(), js.prefix_sem.size(), js.prefix_abc.size() - 2, js.abc_ids.size());
		}
		return rejected == 0 ? 0 : 1;
	}

	// ---- context -----------------------------------------------------------

	// A guided song owns three KV streams: the positive sequence and up to two
	// shadow branches (SPEC_GUIDANCE §4.1). `parallel` keeps meaning "songs
	// decoded side by side" — it is what result.json records — so the stream
	// count is its own number, and it is 3 only when a guided job was accepted.
	bool guided = false;
	for (size_t i = 0; i < jobs.size(); i++)
	{
		// A plain-swap "sections" job is not one of them: its cut drops the old
		// sequence and refills the slot it freed, so no branch is ever live and
		// one stream is the whole of what it needs (SPEC_SECTIONS §4).
		guided = guided || (states[i].ok && !states[i].guide.empty() && !states[i].plain_swap);
	}
	const int n_streams = guided ? BRANCH_KINDS + 1 : parallel;

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx     = n_ctx_want * (uint32_t) n_streams;
	cparams.n_batch   = (uint32_t) std::max<size_t>(max_feed, 512);
	cparams.n_ubatch  = cparams.n_batch;
	cparams.n_seq_max = (uint32_t) n_streams;
	cparams.kv_unified = false;     // one KV stream per slot; llama's default
	cparams.no_perf   = true;
	if (p.threads > 0)
	{
		cparams.n_threads       = p.threads;
		cparams.n_threads_batch = p.threads;
	}

	// F16 K and V, both halves, over the padded per-stream context.
	const int    head_dim     = llama_model_n_embd(model) / llama_model_n_head(model);
	const double kv_per_token = (double) llama_model_n_layer(model) *
	                            llama_model_n_head_kv(model) * head_dim * 2 * 2;
	printf("context: %u tokens/slot x %d slot%s%s, batch %u, KV ~%.0f MiB/slot = %.2f GiB\n",
	       n_ctx_want, n_streams, n_streams == 1 ? "" : "s",
	       guided ? " (1 song + up to 2 guidance branches)" : "", cparams.n_batch,
	       kv_per_token * n_ctx_want / 1048576.0,
	       kv_per_token * n_ctx_want * n_streams / 1073741824.0);

	llama_context * ctx = llama_init_from_model(model, cparams);
	if (ctx == nullptr)
	{
		die("failed to create a %u-token context with %d sequences — "
		    "out of VRAM? lower --parallel", cparams.n_ctx, n_streams);
	}
	// libllama pads n_ctx to a multiple of 256 and n_ctx_seq = n_ctx / n_seq_max
	// up again (llama-context.cpp:288-303), so this can only hold — but every
	// position check downstream depends on it.
	const uint32_t n_ctx_seq = llama_n_ctx_seq(ctx);
	if (n_ctx_seq < n_ctx_want)
	{
		die("llama gave %u tokens per sequence, %u were asked for", n_ctx_seq, n_ctx_want);
	}

	// ---- decode ------------------------------------------------------------

	Runner r;
	r.ctx       = ctx;
	r.vocab     = vocab;
	r.mem       = llama_get_memory(ctx);
	r.batch     = llama_batch_init((int32_t) std::max<uint32_t>(cparams.n_batch,
	                                                            (uint32_t) n_streams), 0, 1);
	r.n_vocab   = n_vocab;
	r.s_abc     = s_abc;
	r.s_sem     = s_sem;
	r.n_ctx_seq = n_ctx_seq;
	r.parallel  = parallel;
	r.n_streams = n_streams;
	r.card      = card;
	r.verify    = p.verify_sampler;
	r.jobs      = &jobs;
	r.states    = &states;
	r.results   = &results;
	r.seqs.resize((size_t) parallel);
	for (int i = 0; i < parallel; i++)
	{
		r.seqs[(size_t) i].slot = i;
		r.seqs[(size_t) i].home = i;
	}

	const double t_ar0 = now_seconds();
	r.run();
	const double ar_seconds = now_seconds() - t_ar0;

	int    ok_jobs = 0;
	double tokens  = 0;
	for (size_t i = 0; i < results.size(); i++)
	{
		if (results[i].ok)
		{
			ok_jobs++;
			tokens += results[i].abc.output_tokens + results[i].semantic.output_tokens;
		}
	}
	if (r.verify)
	{
		printf("ar verify: %lld sampling steps matched the stage-5 sampler exactly\n",
		       r.verified);
	}
	if (jobs.size() > 1 || parallel > 1)
	{
		printf("ar batch: %d/%zu jobs, %.0f tokens in %.2f s = %.2f tok/s aggregate\n",
		       ok_jobs, jobs.size(), tokens, ar_seconds,
		       ar_seconds > 0 ? tokens / ar_seconds : 0);
		printf("ar steps: %d full-width, %d with an idle slot in the middle"
		       " (%.2f vs %.2f ms/step), %d decode calls\n",
		       r.steps_whole, r.steps_holed,
		       r.steps_whole > 0 ? 1000 * r.time_whole / r.steps_whole : 0,
		       r.steps_holed > 0 ? 1000 * r.time_holed / r.steps_holed : 0,
		       r.decodes);
	}

	llama_batch_free(r.batch);
	llama_free(ctx);
	return ok_jobs == (int) jobs.size() ? 0 : 1;
}

// ----------------------------------------------------------- the handover ---

// An earlier render's artifacts as §2 reads them: the semantic stream, the
// score it sang, and the tags it sang them under. Every message names the file
// it is about, because a handover reads several directories and a bare "not a
// .npy file" would not say which.
static std::string load_take(const std::string & dir, HandoverTake & take)
{
	std::string err = read_file(dir + "/score.abc", take.score);
	if (!err.empty())
	{
		return err;
	}
	npy::ArrayI32 sem;
	err = npy::load_i32((dir + "/semantic.npy").c_str(), sem);
	if (!err.empty())
	{
		return strf("%s/semantic.npy: %s", dir.c_str(), err.c_str());
	}
	if (sem.shape.size() != 1)
	{
		return strf("%s/semantic.npy must be a 1-D int32 array, as the NAR's --codec is",
		            dir.c_str());
	}
	for (size_t i = 0; i < sem.data.size(); i++)
	{
		if (sem.data[i] < 0 || sem.data[i] >= CODEC_SIZE)
		{
			return strf("%s/semantic.npy[%zu] = %d is not a codec index in [0, %d)",
			            dir.c_str(), i, (int) sem.data[i], CODEC_SIZE);
		}
	}
	take.codes = sem.data;

	std::string text;
	err = read_file(dir + "/request.json", text);
	if (!err.empty())
	{
		return err;
	}
	json root;
	try
	{
		root = json::parse(text);
	}
	catch (const std::exception & e)
	{
		return strf("%s/request.json: %s", dir.c_str(), e.what());
	}
	const char * key = root.contains("style") && root["style"].is_string() ? "style"
	                 : root.contains("tags")  && root["tags"].is_string()  ? "tags" : nullptr;
	if (key == nullptr)
	{
		return strf("%s/request.json has no \"style\": a take's tags are what it hands over",
		            dir.c_str());
	}
	take.style = root[key].get<std::string>();
	return "";
}

// §2: every take has to sing the base score to the byte — alignment and the
// whole premise depend on it. Run as soon as the base score is known.
static std::string takes_score_check(const std::vector<HandoverTake> & takes,
	const std::string & score, const char * base_name)
{
	for (size_t k = 0; k < takes.size(); k++)
	{
		if (takes[k].dir.empty() || takes[k].score == score)
		{
			continue;
		}
		return strf("%s/score.abc is not the base score (%s) — the two takes have to sing "
		            "the same notes, which is what lets one hand over to the other",
		            takes[k].dir.c_str(), base_name);
	}
	return "";
}

// One take or one leg, through the model this process has already loaded. The
// sub-run is a single job with its own context, so what comes back is bit for
// bit what `yue2 ar` would have produced for that request on its own (§3).
// `js` goes in prepared and comes back with what the run made of it — the score
// it wrote, its prefix, its timings.
static ArResult handover_render(const ArBatchParams & p, const ArJob & job,
	llama_model * model, const llama_vocab * vocab, const std::string & card,
	JobState & js, int max_semantic)
{
	ArBatchParams sub = p;
	sub.parallel      = 1;
	sub.max_semantic  = max_semantic;

	std::vector<ArJob>    jobs(1, job);
	std::vector<JobState> states(1);
	std::vector<ArResult> out(1);
	states[0] = js;
	ar_decode_jobs(sub, jobs, states, out, model, vocab, card, 1, 0);
	js = states[0];
	if (!out[0].ok)
	{
		die("%s%s", js.tag.c_str(), out[0].error.empty() ? "the take did not render"
		                                                 : out[0].error.c_str());
	}
	return out[0];
}

// The semantic cap one take or leg decodes under: the request's own, and never
// more than the command line asked for. A leg needs only to reach the next cut
// (§5); the last one runs to its natural end, which `want` 0 asks for.
static int handover_cap(const ArBatchParams & p, int want)
{
	if (want <= 0)
	{
		return p.max_semantic;
	}
	return p.max_semantic > 0 ? std::min(p.max_semantic, want) : want;
}

// How far this entry's leg has to decode: 25 frames past the next resolved cut
// in the leg's own clock, or 0 — its natural end — when nothing follows it.
static int handover_want(const std::vector<HandoverEntry> & es, size_t i, int off)
{
	for (size_t k = i + 1; k < es.size(); k++)
	{
		if (es[k].found)
		{
			return es[k].cut - off + HANDOVER_TAIL;
		}
	}
	return 0;
}

// What can be known about an entry before its leg runs (§5): a cut with less
// than the intrusion before it, a take that does not reach the cut at any lag
// the scan could pick, and a leg whose forced history would leave nothing to
// sample under the cap. An `"offset": "auto"` is not measured yet, so each
// check is made at the offset that would suit it best: only a certain failure
// is one here. Takes the engine has not rendered yet are skipped, so this runs
// once before anything is rendered and again once every take is in memory.
static std::string handover_preflight(const std::vector<HandoverEntry> & es,
	const std::vector<HandoverTake> & takes, const ArBatchParams & p, int cap_default)
{
	for (size_t i = 0; i < es.size(); i++)
	{
		const HandoverEntry & h = es[i];
		if (!h.found)
		{
			continue;
		}
		if (h.cut - h.x < 1)
		{
			return strf("\"handover\" entry %zu: a cut at %s has less than the "
			            "%d-frame intrusion before it", i + 1,
			            handover_where(h, h.cut).c_str(), h.x);
		}
		// Least of the take before the intrusion: the smallest offset it could get.
		const int left = h.cut - h.x + (h.auto_off ? HANDOVER_RANGE : -h.offset);
		if (left < 1)
		{
			return strf("\"handover\" entry %zu: a cut at %s with a %d-frame intrusion "
			            "at offset %+d leaves nothing of the take before it", i + 1,
			            handover_where(h, h.cut).c_str(), h.x, h.offset);
		}
		// The frame of the take the cut lands on at the largest offset the entry
		// could end up with. It is both how far into the take the leg reads and
		// how long its forced history is, so it answers two questions: whether
		// the take is long enough, and whether anything is left to sample under
		// the cap. The cap is the longest this leg could run to.
		const int deep = h.cut - (h.auto_off ? HANDOVER_RANGE : h.offset);
		const int cap  = handover_cap(p, handover_want(es, i,
			h.auto_off ? -HANDOVER_RANGE : h.offset));
		if (deep >= (cap > 0 ? cap : cap_default))
		{
			return strf("\"handover\" entry %zu: its %d forced frames leave nothing to sample "
			            "under the %d-step semantic cap", i + 1, deep,
			            cap > 0 ? cap : cap_default);
		}
		// The only check that needs the take itself, so it is also the only one
		// that waits for a take the engine has still to render.
		const HandoverTake & t = takes[h.idx];
		if (!t.codes.empty() && deep > (int) t.codes.size())
		{
			return strf("\"handover\" entry %zu: %s has %zu frames, and the cut at %s needs "
			            "frame %d of it%s", i + 1, t.name().c_str(), t.codes.size(),
			            handover_where(h, h.cut).c_str(), deep,
			            h.auto_off ? " even at the largest offset the scan can pick" : "");
		}
	}
	return "";
}

// SPEC_HANDOVER: render one take per style from one score, then at each cut run
// the incoming style's renderer over a forced history — its own take up to x
// frames before the cut, then the last x frames of the song so far — and keep
// what it samples from the cut on. The model is loaded once; every take and
// every leg is a single-job decode through it.
//
// Order matters: every file the request names is read, and every check that can
// be made without decoding is made, before anything is rendered (§3). The one
// exception is a request that writes its own score — then the base take has to
// exist before a label can be resolved or a take's score compared.
static int run_handover(const ArBatchParams & p, const ArJob & job, JobState & top,
	ArResult & result, llama_model * model, const llama_vocab * vocab, const std::string & card)
{
	const double                 t0 = now_seconds();
	std::vector<HandoverEntry> & es = top.req.handover;
	std::vector<HandoverTake>    takes;
	double                       spent = 0;   // semantic decode seconds, takes and legs

	// The cap a leg is held to when neither the command line nor the next cut
	// gives one: the same one load_keep_codes holds a "semantic_keep" to.
	const int sem_cap = semantic_cap(p);

	// The base request is the request as it stands without the handover; the
	// takes and the legs are it with another `style`, and it is what the
	// artifacts of the whole run record (§3, §6).
	JobState base;
	base.req              = top.req;
	base.req.has_handover = false;
	base.req.handover.clear();
	base.req.handover_json = json();
	base.req.base_take.clear();
	base.tag              = "[take_0] ";
	if (job.trace)
	{
		printf("--guidance-trace: a handover decodes no guidance branches; nothing traced\n");
	}

	// ---- every file the request names, before anything is rendered ----------

	{
		HandoverTake t;
		t.style = top.req.style;
		t.label = top.req.base_take.empty() ? "take_0" : top.req.base_take;
		t.dir   = top.req.base_take.empty()
		          ? std::string() : request_relative(job.base_dir, top.req.base_take);
		takes.push_back(t);
	}
	std::string          score;   // "" until the base score is known
	std::vector<int32_t> song;
	if (!top.req.base_take.empty())
	{
		const std::string err = load_take(takes[0].dir, takes[0]);
		if (!err.empty())
		{
			die("%s: \"base_take\": %s", job.request_path.c_str(), err.c_str());
		}
		score = takes[0].score;
		song  = takes[0].codes;
		// The tags the base take was rendered under are not necessarily the
		// request's: a leg that hands back to the base renders under the
		// request's `style` over that take's history, which is what §2 means by
		// "back to the request's own style". Worth saying out loud.
		if (takes[0].style != top.req.style)
		{
			fprintf(stderr, "warning: \"base_take\" was rendered under other tags than this "
			        "request's, so a leg that hands back to it sings the request's:\n"
			        "  base_take: %s\n  request:   %s\n", takes[0].style.c_str(),
			        top.req.style.c_str());
		}
		takes[0].style = top.req.style;
	}
	// What the engine would write as score.abc for this request, which is what a
	// take's own score.abc is: the comparison is of scores, not of request text,
	// so the request that produced the base take is never refused over a stray
	// newline (§2).
	if (top.req.has_abc)
	{
		base.abc_ids  = tokenize(vocab, top.req.abc, "external abc");
		base.abc_text = detokenize(vocab, base.abc_ids);
		if (!score.empty() && base.abc_text != score)
		{
			die("%s: \"base_take\": %s/score.abc is not the score the request gives in "
			    "\"abc\" — a handover has one score", job.request_path.c_str(),
			    takes[0].dir.c_str());
		}
		score = base.abc_text;
	}

	// Which take takes over at each entry, and the directories they name.
	for (size_t i = 0; i < es.size(); i++)
	{
		HandoverEntry & h = es[i];
		h.x = handover_x(h.seconds);
		if (!h.take.empty())
		{
			const std::string dir = request_relative(job.base_dir, h.take);
			size_t            at  = takes.size();
			for (size_t k = 0; k < takes.size(); k++)
			{
				at = takes[k].dir == dir ? k : at;
			}
			if (at == takes.size())
			{
				HandoverTake t;
				t.label               = h.take;
				t.dir                 = dir;
				const std::string err = load_take(dir, t);
				if (!err.empty())
				{
					die("%s: \"handover\" entry %zu: %s", job.request_path.c_str(),
					    i + 1, err.c_str());
				}
				takes.push_back(t);
			}
			h.idx = at;
			continue;
		}
		if (!h.has_style || h.style == top.req.style)
		{
			// Back to the request's own style is the base take, which is also
			// what an entry naming that style word for word asks for (§2).
			h.idx = 0;
			continue;
		}
		size_t at = takes.size();
		for (size_t k = 0; k < takes.size(); k++)
		{
			at = takes[k].dir.empty() && takes[k].style == h.style ? k : at;
		}
		if (at == takes.size())
		{
			HandoverTake t;
			t.style = h.style;
			t.label = strf("take_%zu", takes.size());
			takes.push_back(t);
		}
		h.idx = at;
	}

	// ---- the base take: the song up to the first cut ------------------------

	const char * base_name = top.req.base_take.empty() ? "the one this run wrote"
	                                                   : top.req.base_take.c_str();
	if (!score.empty())
	{
		// Everything is knowable: the takes are of one score, and so are the
		// cuts, before a single token is decoded.
		const std::string err = takes_score_check(takes, score, base_name);
		if (!err.empty())
		{
			die("%s: %s", job.request_path.c_str(), err.c_str());
		}
	}
	if (song.empty())
	{
		ArJob bj      = job;
		bj.artifacts  = job.artifacts + "/take_0";
		bj.trace      = false;
		const ArResult r = handover_render(p, bj, model, vocab, card, base,
		                                   handover_cap(p, 0));
		song           = r.codes;
		score          = base.abc_text;
		takes[0].codes = song;             // the base take, which the legs read
		takes[0].score = score;
		spent         += r.semantic.seconds;
		printf("handover: take_0 (the request's own style): %zu frames in %.1f s = "
		       "%.1f tok/s\n", song.size(), r.semantic.seconds, r.semantic.output_tps);
		const std::string err = takes_score_check(takes, score, base_name);
		if (!err.empty())
		{
			die("%s: %s", job.request_path.c_str(), err.c_str());
		}
	} else {
		// The prefix the NAR reads is the base request's, not the take's (§5),
		// so it is built here exactly as a given-score job builds it.
		base.req.abc       = score;
		base.req.has_abc   = true;
		base.abc_ids       = tokenize(vocab, score, "base take score");
		base.abc_text      = detokenize(vocab, base.abc_ids);
		base.have_abc_text = true;
		base.prefix_abc    = prefix_head(vocab, base.req.text());
		base.prefix_sem    = semantic_prefix(base.prefix_abc, base.abc_ids);
		printf("handover: base take %s: %zu frames of an earlier render, %zu abc ids\n",
		       takes[0].label.c_str(), song.size(), base.abc_ids.size());
	}
	base.req.abc     = score;
	base.req.has_abc = true;

	// ---- where the cuts are, and whether the legs can be played -------------

	handover_locate(score, es);
	{
		std::string err = handover_cuts_check(es);
		if (err.empty())
		{
			err = handover_preflight(es, takes, p, sem_cap);
		}
		if (!err.empty())
		{
			die("%s: %s", job.request_path.c_str(), err.c_str());
		}
	}

	// ---- the takes the engine renders itself, in order of first use ---------

	for (size_t k = 1; k < takes.size(); k++)
	{
		if (!takes[k].dir.empty())
		{
			printf("handover: %s: %zu frames of an earlier render\n",
			       takes[k].label.c_str(), takes[k].codes.size());
			continue;
		}
		JobState js;
		js.req       = base.req;       // one score, one seed: only the tags differ
		js.req.style = takes[k].style;
		js.tag       = strf("[%s] ", takes[k].name().c_str());

		ArJob tj     = job;
		tj.artifacts = job.artifacts + "/" + takes[k].label;
		tj.trace     = false;
		const ArResult r = handover_render(p, tj, model, vocab, card, js, handover_cap(p, 0));
		takes[k].codes = r.codes;
		takes[k].score = score;
		spent         += r.semantic.seconds;
		printf("handover: %s: %zu frames in %.1f s = %.1f tok/s\n", takes[k].name().c_str(),
		       r.codes.size(), r.semantic.seconds, r.semantic.output_tps);
	}
	{
		// Every take is in memory now, so the checks that needed one are made
		// before the first leg rather than being found half way through the song.
		const std::string err = handover_preflight(es, takes, p, sem_cap);
		if (!err.empty())
		{
			die("%s: %s", job.request_path.c_str(), err.c_str());
		}
	}

	// ---- the legs -----------------------------------------------------------

	bool   ended = false;    // the song ended before a cut; every later entry too
	bool   lost  = false;    // a label before this one was not found
	size_t prev  = 0;        // the entry the last leg was decoded for, 1-based
	for (size_t i = 0; i < es.size(); i++)
	{
		HandoverEntry & h = es[i];
		if (!h.found)
		{
			h.note = lost ? std::string("not resolved: an earlier entry's label was not found")
			              : strf("the base score has no %% %s %d", h.section.c_str(), h.nth);
			lost   = true;
			printf("handover: entry %zu: %s — skipped\n", i + 1, h.note.c_str());
			continue;
		}
		if (ended || (size_t) h.cut > song.size())
		{
			ended  = true;
			h.note = strf("the song ends at frame %zu, before the cut at %d",
			              song.size(), h.cut);
			printf("handover: entry %zu: %s — skipped\n", i + 1, h.note.c_str());
			continue;
		}
		const HandoverTake & t = takes[h.idx];
		if (h.auto_off)
		{
			const OffsetFit fit = handover_offset(song, t.codes, h.cut);
			h.offset    = fit.offset;
			h.z         = fit.z;
			h.hits      = fit.hits;
			h.has_z     = true;
			h.confident = fit.confident;
			h.measured  = fit.measured;
			if (!fit.confident)
			{
				fprintf(stderr, "warning: \"handover\" entry %zu: no lag of %s stands out at "
				        "frame %d (the best is %+d, z %.1f of %.0f wanted, %d frames agree of "
				        "%d) — handing over at offset 0; the two takes may not be of one "
				        "score\n", i + 1, t.name().c_str(), h.cut, fit.measured, fit.z,
				        HANDOVER_MIN_Z, fit.hits, HANDOVER_MIN_HITS);
			}
		}

		// How far this leg decodes, and what it is forced through. A failure
		// here is an error, not a skip: the leg before this one was cut short to
		// reach this entry, so skipping it would end the song there in silence.
		const int   cap = handover_cap(p, handover_want(es, i, h.offset));
		std::vector<int32_t> keep;
		std::string err = handover_keep(song, t.codes, h.cut, h.x, h.offset, keep);
		if (err.empty() && (int) keep.size() >= (cap > 0 ? cap : sem_cap))
		{
			err = strf("its %zu forced frames leave nothing to sample under the %d-step "
			           "semantic cap", keep.size(), cap > 0 ? cap : sem_cap);
		}
		if (!err.empty())
		{
			if (prev > 0)
			{
				die("%s: \"handover\" entry %zu cannot be played at the offset measured for "
				    "it (%+d): %s — and entry %zu's leg was already cut short to reach it, "
				    "so the song would end there", job.request_path.c_str(), i + 1, h.offset,
				    err.c_str(), prev);
			}
			die("%s: \"handover\" entry %zu cannot be played at the offset measured for it "
			    "(%+d): %s", job.request_path.c_str(), i + 1, h.offset, err.c_str());
		}

		JobState js;
		js.req             = base.req;   // same lyrics, same score, same seed
		js.req.style       = t.style;
		js.req.has_keep    = true;
		js.req.keep_frames = (int) keep.size();
		js.keep_codes      = keep;
		js.keep_name       = t.name();
		js.no_files        = true;       // the codes are handed back in memory (§5)
		js.tag             = strf("[leg %zu] ", i + 1);

		ArJob lj     = job;
		lj.trace     = false;
		const ArResult r = handover_render(p, lj, model, vocab, card, js, cap);
		if (r.codes.size() < keep.size())
		{
			die("%sthe leg came back with %zu frames, fewer than the %zu it was given",
			    js.tag.c_str(), r.codes.size(), keep.size());
		}

		h.reached = true;
		h.gave    = (int) (r.codes.size() - keep.size());
		song.resize((size_t) h.cut);
		song.insert(song.end(), r.codes.begin() + (long) keep.size(), r.codes.end());
		spent += r.semantic.seconds;
		prev   = i + 1;
		printf("handover: leg %zu: frame %d <- %s, offset %+d", i + 1, h.cut, t.name().c_str(),
		       h.offset);
		if (h.has_z && h.confident)
		{
			printf(" (z %.1f, %d frames agree)", h.z, h.hits);
		} else if (h.has_z) {
			printf(" (not believed: best %+d, z %.1f, %d frames agree)", h.measured, h.z,
			       h.hits);
		}
		// The leg's own tok/s counts the forced frames as steps (SPEC_KEEP §3);
		// what this line reports is the frames it sampled, over the same wall
		// time, so a leg and a take can be compared.
		printf(", x %d, %d new frames in %.1f s = %.1f frames/s, song %zu frames\n",
		       h.x, h.gave, r.semantic.seconds,
		       r.semantic.seconds > 0 ? h.gave / r.semantic.seconds : 0, song.size());
	}

	// ---- the song, and what became of every entry ---------------------------

	const json handover = json_handover(es, takes);

	// The abc phase is the base take's; the semantic numbers describe the song
	// that came out of it, over every take and leg this run decoded.
	GenStats st_sem       = base.st_sem;
	st_sem.output_tokens  = (int) song.size();
	st_sem.content_tokens = (int) song.size();
	st_sem.prefix_tokens  = (int) base.prefix_sem.size();
	st_sem.seconds        = spent;
	st_sem.output_tps     = spent > 0 ? song.size() / spent : 0;

	{
		// request.json is the request as it was given, the handover block and
		// the base take included, so the song reproduces from its own artifacts
		// directory (§6).
		Request art       = base.req;
		art.has_handover  = true;
		art.handover_json = top.req.handover_json;
		art.base_take     = top.req.base_take;

		Artifacts a;
		a.dir           = job.artifacts;
		a.req           = &art;
		a.st_abc        = &base.st_abc;
		a.abc_ids       = base.abc_ids;
		a.prefix_sem    = base.prefix_sem;
		a.codes         = song;
		a.abc_text      = base.abc_text;
		a.have_abc_text = true;
		a.handover      = &handover;
		write_artifacts(a);
	}
	printf("handover: %zu take%s, %zu leg%s, %zu frames in %.1f s (%.1f s of decode)\n",
	       takes.size(), takes.size() == 1 ? "" : "s", es.size(), es.size() == 1 ? "" : "s",
	       song.size(), now_seconds() - t0, spent);
	printf("artifacts: %s (abc %zu ids, semantic %zu codes)\n", job.artifacts.c_str(),
	       base.abc_ids.size(), song.size());

	result.prefix_sem.assign(base.prefix_sem.begin(), base.prefix_sem.end());
	result.codes      = song;
	result.abc        = base.st_abc;
	result.semantic   = st_sem;
	result.seed       = base.req.seed;
	result.cot        = base.req.cot;
	result.cfg_scale  = 1.0;
	result.card       = card;
	result.ok         = true;
	result.parallel   = 1;
	result.batch_jobs = 1;
	return 0;
}

static void usage(const char * argv0)
{
	fprintf(stderr,
	        "usage: %s -m MODEL.gguf --request song.json --artifacts DIR\n"
	        "       %s -m MODEL.gguf --requests jobs.json [--parallel N]\n"
	        "        [--seed N] [--cot full|melody|off] [--gpu N] [--cpu]\n"
	        "        [--threads N] [--dump-logits FILE.npy] [--greedy]\n"
	        "        [--max-abc N] [--max-semantic N] [--continue-on-error]\n"
	        "        [--verify-sampler] [--guidance-trace]\n"
	        "        [--prefix-only]   the request carries its \"abc\": write prefix.npy, decode nothing\n", argv0, argv0);
}

} // namespace

std::string load_jobs_file(const std::string & path, bool need_out, std::vector<ArJob> & jobs)
{
	std::string text;
	{
		const std::string err = read_file(path, text);
		if (!err.empty())
		{
			return err;
		}
	}
	json root;
	try
	{
		root = json::parse(text);
	}
	catch (const std::exception & e)
	{
		return strf("%s: %s", path.c_str(), e.what());
	}
	if (!root.is_array() || root.empty())
	{
		return strf("%s: expected a non-empty JSON array of jobs", path.c_str());
	}

	for (size_t i = 0; i < root.size(); i++)
	{
		const json & e = root[i];
		if (!e.is_object())
		{
			return strf("%s: job %zu is not an object", path.c_str(), i + 1);
		}
		ArJob job;
		for (json::const_iterator it = e.begin(); it != e.end(); ++it)
		{
			const std::string & key = it.key();
			if (key == "seed")
			{
				if (!it.value().is_number_unsigned() ||
				    it.value().get<uint64_t>() >= (uint64_t) 1 << 63)
				{
					return strf("%s: job %zu: \"seed\" must be an integer in [0, 2**63)",
					            path.c_str(), i + 1);
				}
				job.has_seed = true;
				job.seed     = it.value().get<uint64_t>();
				continue;
			}
			if (!it.value().is_string())
			{
				return strf("%s: job %zu: \"%s\" must be a string",
				            path.c_str(), i + 1, key.c_str());
			}
			const std::string value = it.value().get<std::string>();
			if (key == "request")
			{
				job.request_path = value;
			} else if (key == "out") {
				job.out = value;
			} else if (key == "artifacts") {
				job.artifacts = value;
			} else if (key == "noise") {
				job.noise_path = value;
			} else {
				return strf("%s: job %zu: unknown key \"%s\" (request, out, artifacts, "
				            "seed, noise)", path.c_str(), i + 1, key.c_str());
			}
		}
		if (job.request_path.empty())
		{
			return strf("%s: job %zu needs a \"request\"", path.c_str(), i + 1);
		}
		// SPEC_KEEP §2: for a batch, a relative path inside a request resolves
		// against the batch file, which is the one the paths around it are
		// written beside.
		job.base_dir = dir_of(path);
		if (need_out && job.out.empty())
		{
			return strf("%s: job %zu needs an \"out\"", path.c_str(), i + 1);
		}
		if (!need_out && job.artifacts.empty())
		{
			return strf("%s: job %zu needs an \"artifacts\" directory", path.c_str(), i + 1);
		}
		jobs.push_back(job);
	}

	// Two jobs writing one path would race in the AR loop and silently overwrite
	// each other's artifacts — a usage error, caught before anything loads.
	for (size_t i = 0; i < jobs.size(); i++)
	{
		for (size_t k = i + 1; k < jobs.size(); k++)
		{
			if (!jobs[i].out.empty() && jobs[i].out == jobs[k].out)
			{
				return strf("%s: jobs %zu and %zu share the output \"%s\"",
				            path.c_str(), i + 1, k + 1, jobs[i].out.c_str());
			}
			if (!jobs[i].artifacts.empty() && jobs[i].artifacts == jobs[k].artifacts)
			{
				return strf("%s: jobs %zu and %zu share the artifacts directory \"%s\"",
				            path.c_str(), i + 1, k + 1, jobs[i].artifacts.c_str());
			}
		}
	}
	return "";
}

ArParams parse_ar_args(const char * argv0, int argc, char ** argv)
{
	ArParams p;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		if (a == "-m" || a == "--model")
		{
			p.model = need(argc, argv, i);
		} else if (a == "--request") {
			p.request_path = need(argc, argv, i);
		} else if (a == "--requests") {
			p.requests = need(argc, argv, i);
		} else if (a == "--artifacts") {
			p.artifacts = need(argc, argv, i);
		} else if (a == "--dump-logits") {
			p.dump_logits = need(argc, argv, i);
		} else if (a == "--device") {
			p.device = need(argc, argv, i);
		} else if (a == "--cpu") {
			p.device = "cpu";
		} else if (a == "--gpu") {
			p.gpu = atoi(need(argc, argv, i));
		} else if (a == "--threads") {
			p.threads = atoi(need(argc, argv, i));
		} else if (a == "--parallel") {
			p.parallel = atoi(need(argc, argv, i));
		} else if (a == "--seed") {
			p.has_seed = true;
			p.seed     = parse_seed_arg("--seed", need(argc, argv, i));
		} else if (a == "--cot") {
			p.cot = need(argc, argv, i);
		} else if (a == "--greedy") {
			p.greedy = true;
		} else if (a == "--continue-on-error") {
			p.continue_on_error = true;
		} else if (a == "--verify-sampler") {
			p.verify_sampler = true;
		} else if (a == "--guidance-trace") {
			p.guidance_trace = true;
		} else if (a == "--prefix-only") {
			p.prefix_only = true;
		} else if (a == "--max-abc") {
			p.max_abc = parse_positive_arg("--max-abc", need(argc, argv, i));
		} else if (a == "--max-semantic") {
			p.max_semantic = parse_positive_arg("--max-semantic", need(argc, argv, i));
		} else if (a == "-h" || a == "--help") {
			usage(argv0);
			exit(0);
		} else {
			usage(argv0);
			die("unknown argument %s", a.c_str());
		}
	}

	if (p.model.empty() || (p.request_path.empty() == p.requests.empty()))
	{
		usage(argv0);
		die("-m and exactly one of --request / --requests are required");
	}
	if (p.requests.empty() && p.artifacts.empty() && p.dump_logits.empty())
	{
		usage(argv0);
		die("give --artifacts DIR, or --dump-logits FILE.npy");
	}
	if (!p.requests.empty() && !p.dump_logits.empty())
	{
		usage(argv0);
		die("--dump-logits is a single-request path; use --request");
	}
	return p;
}

int run_ar_batch(const ArBatchParams & p, const std::vector<ArJob> & jobs,
	std::vector<ArResult> & results)
{
	if (jobs.empty())
	{
		die("run_ar_batch: no jobs");
	}
	if (p.parallel < 1 || p.parallel > 256)
	{
		die("--parallel must be in [1, 256] (llama caps sequences at LLAMA_MAX_SEQ)");
	}

	std::vector<JobState> states(jobs.size());
	results.assign(jobs.size(), ArResult());

	// The semantic limit, before validation rather than after it: the kept
	// frames of SPEC_KEEP count against it, and a keep that leaves nothing to
	// sample is a request error like any other (§2). The abc limit is the decode
	// loop's own business and is settled there.
	const int sem_cap = semantic_cap(p);

	// ---- validation, before anything loads (SPEC_BATCH §3.1) ----------------

	int rejected = 0;
	auto reject = [&](size_t i, const std::string & err)
	{
		if (!p.continue_on_error)
		{
			die("%s", err.c_str());
		}
		fprintf(stderr, "error: %s%s\n", states[i].tag.c_str(), err.c_str());
		states[i].ok     = false;
		results[i].ok    = false;
		results[i].error = err;
		rejected++;
	};

	for (size_t i = 0; i < jobs.size(); i++)
	{
		JobState & js = states[i];
		if (jobs.size() > 1)
		{
			const std::string & shown = jobs[i].out.empty() ? jobs[i].artifacts : jobs[i].out;
			const size_t        slash = shown.find_last_of('/');
			js.tag = strf("[%zu/%zu %s] ", i + 1, jobs.size(),
			              slash == std::string::npos ? shown.c_str() : shown.c_str() + slash + 1);
		}
		const std::string err = prepare_request(jobs[i].request_path, p.cot,
			jobs[i].has_seed, jobs[i].has_seed ? jobs[i].seed : 0, js.req, js.guidance);
		if (!err.empty())
		{
			reject(i, err);
			continue;
		}
		// SPEC_HANDOVER §2: not in a batch for now. A handover is several
		// generations of its own, each with the context to itself, so it has
		// nothing to share with the songs beside it.
		if (js.req.has_handover && jobs.size() > 1)
		{
			reject(i, strf("%s: \"handover\" is not supported in a batch of %zu songs — "
			               "render it on its own", jobs[i].request_path.c_str(), jobs.size()));
			continue;
		}
		// --prefix-only asks for the NAR's view of the request and decodes
		// nothing, which for a handover is the base request's prefix (§5). The
		// base take's score stands in for the "abc" it would otherwise need.
		if (p.prefix_only && js.req.has_handover && !js.req.base_take.empty() && !js.req.has_abc)
		{
			const std::string dir  = request_relative(jobs[i].base_dir, js.req.base_take);
			const std::string read = read_file(dir + "/score.abc", js.req.abc);
			if (!read.empty())
			{
				reject(i, strf("%s: \"base_take\": %s", jobs[i].request_path.c_str(),
				               read.c_str()));
				continue;
			}
			js.req.has_abc = true;
		}
		// The sections a request names (SPEC_SECTIONS §2). A score the request
		// gave is already written, so its labels are located here — which is what
		// makes a section landing inside the kept frames a request error like any
		// other, before anything touches the GPU (§4).
		if (js.req.has_sections)
		{
			js.sections   = js.req.sections;
			js.plain_swap = sections_plain_swap(js.req.sections);
			if (js.req.has_abc)
			{
				sections_locate(js.req.abc, js.sections);
				sections_frames(js.sections);
				const std::string sec_err = sections_keep_check(js.req, js.sections);
				if (!sec_err.empty())
				{
					reject(i, strf("%s: %s", jobs[i].request_path.c_str(), sec_err.c_str()));
					continue;
				}
			}
		}
		// The earlier render's codes (SPEC_KEEP §2). Read here, so a bad file is
		// one more rejected job rather than a death in the middle of a batch,
		// and so nothing has touched the GPU yet.
		if (js.req.has_keep)
		{
			const std::string keep_err = load_keep_codes(js.req, jobs[i].base_dir,
				sem_cap, js.keep_codes, js.keep_name);
			if (!keep_err.empty())
			{
				js.keep_codes.clear();
				reject(i, strf("%s: %s", jobs[i].request_path.c_str(), keep_err.c_str()));
			}
		}
	}

	// SPEC_GUIDANCE §2.4: a guided job needs the whole context to itself, since
	// its shadow branches are the other KV streams. The decode runs at the
	// *clamped* parallel — one job of a `yue2 batch --parallel 4` is still a
	// single song — so that is what the jobs are held to.
	int parallel = std::min<int>(p.parallel, (int) jobs.size() - rejected);
	if (parallel > 1)
	{
		for (size_t i = 0; i < jobs.size(); i++)
		{
			if (states[i].ok && is_guided(states[i].req))
			{
				reject(i, strf("%s: guidance needs --parallel 1 (this batch decodes %d songs "
				               "side by side)", jobs[i].request_path.c_str(), parallel));
			}
		}
		parallel = std::min<int>(parallel, (int) jobs.size() - rejected);
	}

	if (rejected == (int) jobs.size())
	{
		fprintf(stderr, "error: every job was rejected; nothing to decode\n");
		return 1;
	}

	// ---- model -------------------------------------------------------------

	llama_log_set(quiet_log, nullptr);
	llama_backend_init();

	ggml_backend_dev_t devices[2] = {nullptr, nullptr};
	std::string        backend_name;
	llama_model *      model = load_model(p.model, p.device, p.gpu, backend_name, devices, p.prefix_only);

	const llama_vocab * vocab   = llama_model_get_vocab(model);
	const int           n_vocab = llama_vocab_n_tokens(vocab);
	if (n_vocab != VOCAB_SIZE)
	{
		// sample_step indexes MUSIC_END and the codec range directly; a smaller
		// vocab would read and write out of bounds.
		die("vocab is %d, the protocol requires exactly %d — wrong GGUF?", n_vocab, VOCAB_SIZE);
	}

	// SPEC_HANDOVER: a handover is a driver of its own — it renders a take per
	// style and a leg per cut through the model just loaded, and the artifacts
	// it writes are the song those legs add up to.
	const int rc = !p.prefix_only && states[0].ok && states[0].req.has_handover
		? run_handover(p, jobs[0], states[0], results[0], model, vocab, backend_name)
		: ar_decode_jobs(p, jobs, states, results, model, vocab, backend_name,
		                 parallel, rejected);

	llama_model_free(model);
	llama_backend_free();
	return rc;
}

int run_ar(const ArParams & p, ArResult * out)
{
	if (!p.dump_logits.empty())
	{
		if (p.guidance_trace)
		{
			printf("--guidance-trace: --dump-logits decodes one step into no artifacts "
			       "directory; nothing traced\n");
		}
		return run_ar_dump(p);
	}

	ArBatchParams bp;
	bp.model             = p.model;
	bp.device            = p.device;
	bp.gpu               = p.gpu;
	bp.threads           = p.threads;
	bp.greedy            = p.greedy;
	bp.cot               = p.cot;
	bp.max_abc           = p.max_abc;
	bp.max_semantic      = p.max_semantic;
	bp.continue_on_error = p.continue_on_error;
	bp.verify_sampler    = p.verify_sampler;
	bp.prefix_only       = p.prefix_only;

	std::vector<ArJob> jobs;
	if (p.requests.empty())
	{
		ArJob job;
		job.request_path = p.request_path;
		job.base_dir     = dir_of(p.request_path);
		job.artifacts    = p.artifacts;
		job.has_seed     = p.has_seed;
		job.seed         = p.seed;
		job.trace        = p.guidance_trace;
		jobs.push_back(job);
		bp.parallel = 1;
	} else {
		const std::string err = load_jobs_file(p.requests, false, jobs);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
		bp.parallel = p.parallel;
		// A --seed on the command line is the default for jobs that name none;
		// --guidance-trace is for every job of the batch that is guided.
		for (size_t i = 0; i < jobs.size(); i++)
		{
			if (p.has_seed && !jobs[i].has_seed)
			{
				jobs[i].has_seed = true;
				jobs[i].seed     = p.seed;
			}
			jobs[i].trace = p.guidance_trace;
		}
	}

	std::vector<ArResult> results;
	const int             rc = run_ar_batch(bp, jobs, results);
	if (out != nullptr && !results.empty())
	{
		*out = results[0];
	}
	return rc;
}
