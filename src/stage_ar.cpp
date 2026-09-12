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
	SampleScratch * scratch = nullptr)
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
	if (step < s.min_tokens)
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


static llama_model * load_model(const std::string & path, const std::string & device, int gpu,
	std::string & backend_name, ggml_backend_dev_t * devices)
{
	llama_model_params mparams = llama_model_default_params();
	backend_name = "CPU";
	if (device == "cpu")
	{
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
	double                   t_phase0 = 0;
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
	void        feed(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	                 const char * what);
	llama_token sample(Seq & q, const float * logits);
	void        apply(Seq & q, llama_token token);
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
		                          rng_ref);
	}

	const llama_token token = sample_step(logits, n_vocab, s, q.history, q.step,
	                                      abc, legacy_off, q.rng, &q.scratch);
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

void Runner::feed(Seq & q, const std::vector<llama_token> & tokens, size_t expect_pos,
	const char * what)
{
	JobState & js = (*states)[q.job];
	q.i_batch = decode_feed(ctx, batch, tokens, q.slot, q.pos, what);
	decodes++;

	GenStats & st = q.phase == PHASE_ABC ? js.st_abc : js.st_sem;
	st.prefill_seconds = now_seconds() - q.t_phase0;

	// The KV cache must agree with the prefix the artifacts record: a phase that
	// was truncated used to leave the cache one token short here.
	const llama_pos have = llama_memory_seq_pos_max(mem, q.slot) + 1;
	if ((size_t) have != expect_pos)
	{
		die("%s%s prefill left %d tokens in slot %d, the prefix is %zu — "
		    "the cache and prefix.npy disagree",
		    js.tag.c_str(), what, (int) have, q.slot, expect_pos);
	}

	// Immediately: the next llama_decode from any slot overwrites this buffer.
	apply(q, sample(q, llama_get_logits_ith(ctx, q.i_batch)));
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
	active++;

	if (js.do_abc)
	{
		js.st_abc.prefix_tokens = (int) js.prefix_abc.size();
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
	        "        [--seed N] [--cot full|melody|off] [--device cpu|vulkan] [--gpu N]\n"
	        "        [--threads N] [--dump-logits FILE.npy] [--greedy]\n"
	        "        [--max-abc N] [--max-semantic N] [--continue-on-error]\n"
	        "        [--verify-sampler]\n", argv0, argv0);
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
	llama_model *      model = load_model(p.model, p.device, p.gpu, backend_name, devices);

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
		if (want > (uint32_t) CONTEXT)
		{
			want = (uint32_t) CONTEXT;
		}
		n_ctx_want = std::max(n_ctx_want, want);
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
