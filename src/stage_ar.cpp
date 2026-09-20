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

// Either request form asks for branches beside the positive sequence.
static bool is_guided(const Request & r)
{
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
	out["execution"]       = "eager";
	out["attention"]       = "llama.cpp";
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
		json against = json::object();
		for (int k = 0; k < BRANCH_KINDS; k++)
		{
			if (!g.has[k])
			{
				continue;
			}
			json curve = json::array();
			for (size_t c = 0; c < g.curve[k].size(); c++)
			{
				json kf = json::array();
				kf.push_back(g.curve[k][c].offset);
				kf.push_back(g.curve[k][c].weight);
				curve.push_back(kf);
			}
			against[branch_name(k)] = curve;
		}
		if (!against.empty())
		{
			e["against"] = against;
		}
		e["reached"] = g.reached;
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
	// null unless the request carried a "semantic_keep": the codes it kept, for
	// the frame count and the digest plan.json records (SPEC_KEEP §4).
	const std::vector<int32_t> *       keep     = nullptr;
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
				if (!it.value().is_object() || it.value().empty())
				{
					return strf("entry %zu: \"against\" must be a non-empty object "
					            "(previous, blank)", i + 1);
				}
				for (json::const_iterator a = it.value().begin(); a != it.value().end(); ++a)
				{
					const int kind = a.key() == "previous" ? BRANCH_PREVIOUS
					                 : a.key() == "blank"   ? BRANCH_BLANK : -1;
					if (kind < 0)
					{
						return strf("entry %zu: \"against\": unknown branch \"%s\" "
						            "(previous, blank)", i + 1, a.key().c_str());
					}
					const std::string err = parse_curve(a.value(), a.key().c_str(),
					                                    g.curve[kind]);
					if (!err.empty())
					{
						return strf("entry %zu: \"against\": %s", i + 1, err.c_str());
					}
					g.has[kind] = true;
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

// The kept codes of SPEC_KEEP §2, through the NAR's `--codec` reader: a 1-D
// int32 .npy, every value a codec index. `max_steps` is the semantic phase's
// own cap, which kept steps count against like any other (§3), so a request
// that keeps all of it would have nothing left to sample. Runs at validation
// time, before the model loads; `name` comes back as the basename, which is all
// the artifacts and the log line ever show of the path.
static std::string load_keep_codes(const Request & r, const std::string & base, int max_steps,
	std::vector<int32_t> & out, std::string & name)
{
	std::filesystem::path path = r.keep_file;
	if (path.is_relative() && !base.empty())
	{
		path = std::filesystem::path(base) / path;
	}
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
};

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
		const int slot = free_slot(q);
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

	q.phase    = PHASE_SEM;
	q.history.clear();
	q.step     = 0;
	q.rng.seed(js.req.seed);
	q.t_phase0 = now_seconds();

	// The cache already holds prefix_abc plus every abc id that was decoded; a
	// truncated phase still has its last kept token in hand, so the bridge is
	// two tokens or three. Its own decode call, not the lockstep batch: §4.4.
	std::vector<llama_token> bridge;
	if (!eos)
	{
		bridge.push_back(token);
	}
	bridge.push_back(ABC_END);
	bridge.push_back(MUSIC_START);

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
		a.guidance      = js.req.has_guidance ? &js.guide : nullptr;
		a.keep          = js.keep_codes.empty() ? nullptr : &js.keep_codes;
		a.trace         = (*jobs)[q.job].trace && !js.guide.empty() ? &js.trace : nullptr;
		write_artifacts(a);
	}
	printf("%sartifacts: %s (abc %zu ids, semantic %zu codes)\n", js.tag.c_str(),
	       (*jobs)[q.job].artifacts.c_str(), js.abc_ids.size(), codes.size());
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

	// The per-phase limits, before validation rather than after it: the kept
	// frames of SPEC_KEEP count against the semantic cap, and that is a
	// request error like any other (§2).
	Sampling s_abc = sampling_abc();
	Sampling s_sem = sampling_semantic();
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
		// The earlier render's codes (SPEC_KEEP §2). Read here, so a bad file is
		// one more rejected job rather than a death in the middle of a batch,
		// and so nothing has touched the GPU yet.
		if (js.req.has_keep)
		{
			const std::string keep_err = load_keep_codes(js.req, jobs[i].base_dir,
				s_sem.max_tokens, js.keep_codes, js.keep_name);
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
		js.guide = guidance_plan(js.req);
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
		llama_model_free(model);
		llama_backend_free();
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
		guided = guided || (states[i].ok && !states[i].guide.empty());
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
	r.card      = backend_name;
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
	llama_model_free(model);
	llama_backend_free();
	return ok_jobs == (int) jobs.size() ? 0 : 1;
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
