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
	std::string id         = "song";

	std::string text() const
	{
		return std::string(instruction(cot)) + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n";
	}
};

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

enum TplKind { TPL_GIVEN, TPL_PRIMER, TPL_HOLE };

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
	int         bars = 0;        // hole: the bar count its line must have
	std::string above;           // hole: the body line its token budget scales from
	Meter       meter;           // hole: the meter in force where it sits
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

		// Only the *tests* see a trimmed line; what is fed to the model is the
		// line exactly as the template wrote it, `\r` and all.
		const size_t head = line.find_first_not_of(" \t");
		const size_t tail = line.find_last_not_of(" \t\r");
		const std::string trimmed = head == std::string::npos
		                            ? std::string() : line.substr(head, tail - head + 1);
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

		// `%%yue2-gen` alone, or `%%yue2-gen bars=N` with real whitespace between
		// the two — `%%yue2-genbars=3` is a typo, not a directive with an argument.
		const std::string args = trimmed.substr(strlen("%%yue2-gen"));
		if (!args.empty() && args.find_first_of(" \t") != 0)
		{
			return strf("line %zu: unknown directive \"%s\"", lineno, trimmed.c_str());
		}
		const size_t arg = args.find_first_not_of(" \t");
		if (arg == std::string::npos)
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
		} else {
			const std::string value = args.substr(arg);
			if (!starts_with(value, "bars="))
			{
				return strf("line %zu: %%%%yue2-gen takes nothing but bars=N (got \"%s\")",
				            lineno, value.c_str());
			}
			// strtol, not sscanf: %d on an overflowing literal is undefined.
			errno = 0;
			const char * digits = value.c_str() + strlen("bars=");
			char *       endp   = nullptr;
			const long   bars   = strtol(digits, &endp, 10);
			if (errno != 0 || endp == digits || *endp != '\0')
			{
				return strf("line %zu: %%%%yue2-gen takes nothing but bars=N (got \"%s\")",
				            lineno, value.c_str());
			}
			if (bars < 1 || bars > TEMPLATE_MAX_BARS)
			{
				return strf("line %zu: bars=%ld — a hole holds between 1 and %d bars",
				            lineno, bars, TEMPLATE_MAX_BARS);
			}
			seg.bars = (int) bars;
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
	out["cfg_branches"]    = 1;
	out["execution"]       = "eager";
	out["attention"]       = "llama.cpp";
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

// Returns "" or the reason the file is not a usable request. A batch validates
// every job before the model loads (SPEC_BATCH §3.1), so this reports rather
// than exits; the single-request paths turn a non-empty return into die().
static std::string parse_request(const std::string & path, Request & req)
{
	std::string text;
	std::string err = read_file(path, text);
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
	if (!root.is_object())
	{
		return strf("%s: expected a JSON object", path.c_str());
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
		            path.c_str());
	}

	if (root.contains("style") && root["style"].is_string())
	{
		req.style = root["style"].get<std::string>();
	} else if (root.contains("tags") && root["tags"].is_string()) {
		req.style = root["tags"].get<std::string>();
	} else {
		return strf("%s: needs a \"style\" (or \"tags\") string", path.c_str());
	}
	if (root.contains("lyrics") && root["lyrics"].is_string())
	{
		req.lyrics = root["lyrics"].get<std::string>();
	} else {
		return strf("%s: needs a \"lyrics\" string", path.c_str());
	}
	if (root.contains("cot") && !root["cot"].is_null())
	{
		if (!root["cot"].is_string())
		{
			return strf("%s: \"cot\" must be a string", path.c_str());
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
			return strf("%s: \"seed\" must be a non-negative integer below 2**63", path.c_str());
		}
		req.seed = v.get<uint64_t>();
	}
	if (root.contains("id") && !root["id"].is_null())
	{
		if (!root["id"].is_string())
		{
			return strf("%s: \"id\" must be a string", path.c_str());
		}
		req.id = root["id"].get<std::string>();
	}
	if (root.contains("abc") && !root["abc"].is_null())
	{
		if (!root["abc"].is_string())
		{
			return strf("%s: \"abc\" must be a string or null", path.c_str());
		}
		req.has_abc = true;
		req.abc     = root["abc"].get<std::string>();
	}
	if (root.contains("abc_template") && !root["abc_template"].is_null())
	{
		if (!root["abc_template"].is_string())
		{
			return strf("%s: \"abc_template\" must be a string or null", path.c_str());
		}
		req.has_tpl      = true;
		req.abc_template = root["abc_template"].get<std::string>();
	}
	if (root.contains("cfg_scale") && !root["cfg_scale"].is_null())
	{
		if (!root["cfg_scale"].is_number())
		{
			return strf("%s: \"cfg_scale\" must be a number or null", path.c_str());
		}
		req.has_cfg   = true;
		req.cfg_scale = root["cfg_scale"].get<double>();
	}
	return "";
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
	return "";
}

// The request as one job's decode loop needs it: parsed, the command-line
// overrides applied, validated, and cfg rejected. protocol.SongRequest.guidance
// is 1.01 for cot=off, which yue2.cpp cannot run either (stage 2b+).
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
	guidance = req.has_cfg ? req.cfg_scale : (req.cot == "off" ? 1.01 : 1.0);
	if (guidance != 1.0)
	{
		return strf("%s: cfg_scale %g: classifier-free guidance is not supported (stage 2b+)",
		            path.c_str(), guidance);
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

	GenStats                 st_abc;
	GenStats                 st_sem;
	bool                     ok    = true;   // false once the job is rejected
	std::string              tag;            // "[k/N name] ", empty for a single job
};

// One KV stream of the shared context, and the phase it is in. SPEC_BATCH §4.3.
struct Seq
{
	int                      slot     = 0;   // == the llama seq_id
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
	bool                     tpl_full  = false;   // the job's sampled-token cap is spent
	int                      attempt   = 0;  // 1..TEMPLATE_ATTEMPTS within the hole
	llama_pos                hole_pos  = 0;  // where the hole's tokens start
	size_t                   hole_hist = 0;  // history size there
	int                      hole_step = 0;
	llama_token              hole_prev = 0;  // the token before it, re-decoded on a rollback
	std::vector<llama_token> hole_ids;       // sampled and decoded inside the hole
	std::string              carry;          // the closing token's text, fed with the next segment
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

	// The template walk. give() feeds one forced stretch of score; the rest is
	// the hole state machine. SPEC_TEMPLATE §3.
	void        give(Seq & q, const std::string & text, bool emit, const char * what);
	bool        abc_fits(const Seq & q, size_t extra) const;
	bool        sem_fits(const Seq & q, size_t extra) const;
	void        template_step(Seq & q);
	void        open_hole(Seq & q);
	void        template_token(Seq & q, llama_token token);
	void        fail_hole(Seq & q, const std::string & why);
	void        rollback_hole(Seq & q);
	void        rest_fill(Seq & q);
	void        template_done(Seq & q);

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

void Runner::apply(Seq & q, llama_token token)
{
	if (q.in_hole)
	{
		template_token(q, token);
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

	// protocol.token_prefixes(request, tokenizer, abc_ids)
	js.prefix_sem = js.prefix_abc;
	js.prefix_sem.insert(js.prefix_sem.end(), js.abc_ids.begin(), js.abc_ids.end());
	js.prefix_sem.push_back(ABC_END);
	js.prefix_sem.push_back(MUSIC_START);
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
	apply(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
}

// One forced stretch of score: tokenized, decoded, and pushed into `history` so
// the repetition-penalty window sees it as if the model had written it. `emit`
// separates a given line (which reaches score.abc) from a primer line (which
// does not) — the text itself is appended by the caller that knows which.
void Runner::give(Seq & q, const std::string & text, bool emit, const char * what)
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
	if (emit)
	{
		js.tpl.given_tokens += (int) ids.size();
		js.emitted          += ids.size();
	} else {
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
	const size_t     need = js.prefix_abc.size() + js.emitted + extra +
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
		q.seg++;
		const bool        emit = seg.kind == TPL_GIVEN;
		const std::string text = q.carry + seg.text;
		q.carry.clear();
		if (emit)
		{
			js.tpl_text += seg.text;
		}
		give(q, text, emit, emit ? "given segment" : "primer segment");
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
		give(q, text, true, "hole tail");
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
	const std::string line = full.substr(0, nl + 1);
	const std::string body = line.substr(0, nl);

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

	// Nothing to undo when the hole failed on its very first token, or when the
	// sampled-token cap closed it before it opened — but `step` is restored
	// either way, since a rest-fill spends none of the phase's draws.
	if (!q.hole_ids.empty() || q.pos != q.hole_pos)
	{
		rollback_hole(q);
	}
	q.step = q.hole_step;
	const std::string fill = seg.bars == 1 ? std::string("Z|\n") : strf("Z%d|\n", seg.bars);
	js.tpl_text += fill;
	q.in_hole    = false;
	q.seg++;
	give(q, fill, true, "rest fill");
	template_step(q);
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
	       "%d given + %d primer tokens, %d off-length bars\n", js.tag.c_str(),
	       js.tpl.holes, js.tpl.sampled_tokens, js.tpl.retries, js.tpl.rest_filled,
	       js.tpl.given_tokens, js.tpl.primer_tokens, js.tpl.offlength_bars);

	js.prefix_sem = js.prefix_abc;
	js.prefix_sem.insert(js.prefix_sem.end(), js.abc_ids.begin(), js.abc_ids.end());
	js.prefix_sem.push_back(ABC_END);
	js.prefix_sem.push_back(MUSIC_START);
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

	const llama_pos held = llama_memory_seq_pos_max(mem, q.slot);
	if (held != -1)
	{
		die("slot %d still holds %d tokens before job %d enters it",
		    q.slot, (int) held + 1, job + 1);
	}

	q.job      = job;
	q.phase    = js.do_abc ? PHASE_ABC : PHASE_SEM;
	q.history.clear();
	q.step     = 0;
	q.rng.seed(js.req.seed);
	q.pos      = 0;
	q.i_batch  = -1;
	q.t_phase0 = now_seconds();
	q.seg      = 0;
	q.in_hole  = false;
	q.tpl_full = false;
	q.hole_ids.clear();
	q.carry.clear();
	active++;

	if (js.do_abc)
	{
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
	feed(q, js.prefix_sem, js.prefix_sem.size(), "semantic");
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
		write_artifacts(a);
	}
	printf("%sartifacts: %s (abc %zu ids, semantic %zu codes)\n", js.tag.c_str(),
	       (*jobs)[q.job].artifacts.c_str(), js.abc_ids.size(), codes.size());

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
	r.slot        = q.slot;
	r.batch_jobs  = (int) jobs->size();

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
			first = first < 0 ? q.slot : first;
			last  = q.slot;
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
			if (seqs[i].job >= 0)
			{
				seqs[i].sampled = sample(seqs[i], llama_get_logits_ith(ctx, seqs[i].i_batch));
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

	std::vector<llama_token> prefix_abc;
	prefix_abc.push_back(EOD);
	{
		const std::vector<llama_token> ids = tokenize(vocab, req.text(), "request text");
		prefix_abc.insert(prefix_abc.end(), ids.begin(), ids.end());
	}
	prefix_abc.push_back(ABC_START);

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx     = (uint32_t) prefix_abc.size() + 4;
	cparams.n_batch   = (uint32_t) std::max<size_t>(prefix_abc.size(), 512);
	cparams.n_ubatch  = cparams.n_batch;
	cparams.n_seq_max = 1;
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
	printf("context: %u tokens, abc prefix %zu tokens, batch %u\n",
	       cparams.n_ctx, prefix_abc.size(), llama_n_batch(ctx));

	llama_batch batch = llama_batch_init((int32_t) llama_n_batch(ctx), 0, 1);
	run_dump_logits(ctx, batch, n_vocab, p, prefix_abc);

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
	        "        [--verify-sampler]\n"
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

	// ---- validation, before anything loads (SPEC_BATCH §3.1) ----------------

	int rejected = 0;
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
		if (err.empty())
		{
			continue;
		}
		if (!p.continue_on_error)
		{
			die("%s", err.c_str());
		}
		fprintf(stderr, "error: %s%s\n", js.tag.c_str(), err.c_str());
		js.ok            = false;
		results[i].ok    = false;
		results[i].error = err;
		rejected++;
	}
	if (rejected == (int) jobs.size())
	{
		fprintf(stderr, "error: every job was rejected; nothing to decode\n");
		return 1;
	}

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

		const std::vector<llama_token> ids = tokenize(vocab, text, "request text");
		js.prefix_abc.push_back(EOD);
		js.prefix_abc.insert(js.prefix_abc.end(), ids.begin(), ids.end());
		js.prefix_abc.push_back(ABC_START);

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
				if (seg.kind != TPL_HOLE)
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
			js.prefix_sem = js.prefix_abc;
			js.prefix_sem.insert(js.prefix_sem.end(), js.abc_ids.begin(), js.abc_ids.end());
			js.prefix_sem.push_back(ABC_END);
			js.prefix_sem.push_back(MUSIC_START);
			max_feed = std::max(max_feed, sem_prefix_len);
		}
		max_feed = std::max(max_feed, js.prefix_abc.size());

		if ((int) sem_prefix_len + s_sem.max_tokens > CONTEXT && !js.do_abc)
		{
			die("%ssemantic prefix %zu + max_tokens %d exceeds the %d context; "
			    "no implicit truncation",
			    js.tag.c_str(), sem_prefix_len, s_sem.max_tokens, CONTEXT);
		}

		uint32_t want = (uint32_t) (js.prefix_abc.size() + js.abc_ids.size() +
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

	const int parallel = std::min<int>(p.parallel, (int) jobs.size() - rejected);

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx     = n_ctx_want * (uint32_t) parallel;
	cparams.n_batch   = (uint32_t) std::max<size_t>(max_feed, 512);
	cparams.n_ubatch  = cparams.n_batch;
	cparams.n_seq_max = (uint32_t) parallel;
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
	printf("context: %u tokens/slot x %d slot%s, batch %u, KV ~%.0f MiB/slot = %.2f GiB\n",
	       n_ctx_want, parallel, parallel == 1 ? "" : "s", cparams.n_batch,
	       kv_per_token * n_ctx_want / 1048576.0,
	       kv_per_token * n_ctx_want * parallel / 1073741824.0);

	llama_context * ctx = llama_init_from_model(model, cparams);
	if (ctx == nullptr)
	{
		die("failed to create a %u-token context with %d sequences — "
		    "out of VRAM? lower --parallel", cparams.n_ctx, parallel);
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
	                                                            (uint32_t) parallel), 0, 1);
	r.n_vocab   = n_vocab;
	r.s_abc     = s_abc;
	r.s_sem     = s_sem;
	r.n_ctx_seq = n_ctx_seq;
	r.parallel  = parallel;
	r.card      = backend_name;
	r.verify    = p.verify_sampler;
	r.jobs      = &jobs;
	r.states    = &states;
	r.results   = &results;
	r.seqs.resize((size_t) parallel);
	for (int i = 0; i < parallel; i++)
	{
		r.seqs[(size_t) i].slot = i;
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
		job.artifacts    = p.artifacts;
		job.has_seed     = p.has_seed;
		job.seed         = p.seed;
		jobs.push_back(job);
		bp.parallel = 1;
	} else {
		const std::string err = load_jobs_file(p.requests, false, jobs);
		if (!err.empty())
		{
			die("%s", err.c_str());
		}
		bp.parallel = p.parallel;
		// A --seed on the command line is the default for jobs that name none.
		if (p.has_seed)
		{
			for (size_t i = 0; i < jobs.size(); i++)
			{
				if (!jobs[i].has_seed)
				{
					jobs[i].has_seed = true;
					jobs[i].seed     = p.seed;
				}
			}
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
