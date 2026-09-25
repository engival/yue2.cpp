// Stage 2 — ABC plan + semantic codec tokens on libllama. SPEC_AR.md, src/STATUS_AR.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ArParams
{
	std::string model;
	std::string request_path;
	std::string artifacts;
	std::string dump_logits;
	std::string requests;          // --requests jobs.json: a batch instead of one request
	std::string device   = "vulkan";
	int         gpu      = 0;
	int         threads  = 0;
	int         parallel = 4;
	bool        greedy   = false;
	bool        has_seed = false;
	uint64_t    seed     = 0;
	std::string cot;
	int         max_abc      = 0;
	int         max_semantic = 0;
	bool        continue_on_error = false;
	bool        verify_sampler    = false;   // run the frozen stage-5 sampler beside the new one
	bool        prefix_only       = false;   // tokenize request + given abc, write prefix.npy, decode nothing
	bool        guidance_trace    = false;   // --guidance-trace: DIR/guidance_trace.npy, diagnostics only
};

// One song in a batch: exactly the per-song options `yue2 song` takes, minus
// the ones that are per process (device, GGUFs, sampling limits). SPEC_BATCH §3.1.
struct ArJob
{
	std::string request_path;
	std::string base_dir;          // what a relative path inside the request resolves against
	std::string artifacts;
	std::string out;               // the FLAC `yue2 batch` writes; `yue2 ar` ignores it
	std::string noise_path;
	bool        has_seed = false;
	uint64_t    seed     = 0;
	// --guidance-trace, per job because it writes into the job's own artifacts
	// directory: a job that has none (a `yue2 song` without --artifacts) traces
	// nothing. Never a key of a `--jobs` file; the command line sets it.
	bool        trace    = false;
};

// What the AR decode loop needs for a whole batch. Per-job overrides are in ArJob.
struct ArBatchParams
{
	std::string model;
	std::string device   = "vulkan";
	int         gpu      = 0;
	int         threads  = 0;
	int         parallel = 4;
	bool        greedy   = false;
	std::string cot;
	int         max_abc      = 0;
	int         max_semantic = 0;
	bool        continue_on_error = false;
	bool        verify_sampler    = false;   // SPEC_SAMPLER.md §4
	bool        prefix_only       = false;
};

// protocol.SongResult timing, as plan.json / result.json record it.
struct GenStats
{
	int    output_tokens   = 0;   // content + end token, as torch counts them
	int    content_tokens  = 0;
	int    prefix_tokens   = 0;
	double seconds         = 0;
	double prefill_seconds = 0;
	double ttft_seconds    = 0;
	double output_tps      = 0;
	bool   truncated       = false;
	// SPEC_GUIDANCE §3: how many sequences this song decoded at once at the most
	// (1 unguided), how many semantic steps had more than one, and what
	// prefilling the extra ones cost. Song-level, carried on both phases.
	int    cfg_branches    = 1;
	int    guided_steps    = 0;
	double branch_prefill_seconds = 0;
	// SPEC_KEEP §4: the leading codes this song took from an earlier render
	// instead of sampling them, and what prefilling them cost.
	int    kept_frames     = 0;
	double keep_prefill_seconds = 0;
	// SPEC_SECTIONS §3: the cuts the score phase made, and what re-prefilling
	// the score under each entry's own tags cost.
	int    section_cuts    = 0;
	double section_prefill_seconds = 0;
};

// What the abc phase of a score-template job did with its holes. SPEC_TEMPLATE
// §5: it reaches result.json and the batch summary, for template jobs only.
struct TemplateStats
{
	int holes          = 0;
	int retries        = 0;   // attempts that were rolled back and drawn again
	int rest_filled    = 0;   // holes that exhausted their attempts
	int primer_tokens  = 0;   // fed as context, dropped from the emitted score
	int given_tokens   = 0;
	int sampled_tokens = 0;   // every draw, including the ones that were rolled back
	int offlength_bars = 0;   // written bars that do not hold M:/L: note units
	int continued_tokens = 0; // written freely after %%yue2-continue, up to ABC_END
	int chord_lines    = 0;   // of the holes, the ones a %%yue2-chords directive opened
};

// What `yue2 song` needs from the AR stage without re-reading the artifacts.
struct ArResult
{
	std::vector<int32_t> prefix_sem;    // == prefix.npy
	std::vector<int32_t> codes;         // == semantic.npy
	GenStats             abc;
	GenStats             semantic;
	uint64_t             seed = 0;      // the seed this run actually used
	std::string          cot;
	double               cfg_scale = 1.0;
	std::string          negative_style;      // "" unless the request carried one (SPEC_NEGATIVE)
	bool                 negative_lyrics = false; // the negative branch sang the lyrics (§7)
	double               cfg_score = 1.0;     // the score phase's guidance (§8), 1 = unguided
	double               score_tempo = 0;     // the forced Q: (§10), 0 = the model's own
	std::string          card;          // ggml device description
	bool                 ok        = false;   // false until the job's artifacts are on disk
	std::string          error;               // why not, when ok == false
	bool                 is_template = false; // the request carried an "abc_template"
	TemplateStats        tpl;
	int                  parallel  = 1;       // how this song was decoded — see SPEC_BATCH §6
	int                  slot      = 0;
	int                  batch_jobs = 1;
};

ArParams parse_ar_args(const char * argv0, int argc, char ** argv);

// Reads a `--jobs` / `--requests` JSON array into `jobs`. `need_out` picks which
// per-job path is required: the FLAC for `yue2 batch`, the artifacts directory
// for `yue2 ar`. Returns "" or the error message; duplicate `out` or `artifacts`
// paths are an error here, before anything loads.
std::string load_jobs_file(const std::string & path, bool need_out, std::vector<ArJob> & jobs);

// `out` may be null; the artifacts are written either way.
int run_ar(const ArParams & p, ArResult * out);

// The AR stage for a whole batch: one llama context, up to p.parallel songs
// decoded in lockstep. `results` is resized to jobs.size(); a job that was
// rejected has ok == false. Returns 0 when every job succeeded.
int run_ar_batch(const ArBatchParams & p, const std::vector<ArJob> & jobs,
	std::vector<ArResult> & results);
