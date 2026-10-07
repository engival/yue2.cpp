// Stage 4 — request -> FLAC in one process. SPEC_SINGLE.md.
// Stage 5 — several requests, one AR decode loop. SPEC_BATCH.md.
#pragma once

#include "stage_ar.hpp"

#include "common/lora.hpp"
#include "common/opus.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct SongParams
{
	std::string request_path;
	std::string out;
	std::string artifacts;
	std::string ar_model;
	std::string nar_model;
	std::string vae_model;
	std::string noise_path;
	std::string device   = "vulkan";
	int         gpu      = 0;
	int         steps    = 16;   // the reference's ode_steps is 32: --steps 32
	bool        has_seed = false;
	uint64_t    seed     = 0;
	bool        nar_f32  = false;
	bool        guidance_trace = false;   // --guidance-trace, needs --artifacts to land in
	bool        no_tags  = false;   // --no-tags: the FLAC carries no Vorbis comments
	int         opus_bitrate = opus::BITRATE_DEFAULT;   // --opus-bitrate, kbit/s, for an .opus --out
	std::vector<lora::Spec> nar_lora;   // --nar-lora, a property of the model, not of a request
	DraftParams draft;                  // --draft*, SPEC_DRAFT §5
};

// What `yue2 batch` takes for the whole list; everything that differs per song
// is in ArJob. SPEC_BATCH §3.4.
struct BatchParams
{
	std::string jobs_path;
	std::string summary;
	std::string ar_model;
	std::string nar_model;
	std::string vae_model;
	std::string device   = "vulkan";
	int         gpu      = 0;
	int         steps    = 16;
	int         parallel = 4;
	int         threads  = 0;
	bool        greedy   = false;
	int         max_abc      = 0;
	int         max_semantic = 0;
	bool        has_seed = false;   // the default seed for jobs that name none
	uint64_t    seed     = 0;
	bool        nar_f32  = false;
	bool        continue_on_error = false;
	bool        guidance_trace = false;   // per job: only one with its own artifacts traces
	bool        no_tags  = false;
	int         opus_bitrate = opus::BITRATE_DEFAULT;
	std::vector<lora::Spec> nar_lora;
	DraftParams draft;
};

SongParams  parse_song_args (const char * argv0, int argc, char ** argv);
BatchParams parse_batch_args(const char * argv0, int argc, char ** argv);

int run_song (const SongParams & p);
int run_batch(const BatchParams & p, std::vector<ArJob> jobs);

// Where the default GGUFs live and `yue2 convert` writes them: the directory
// of this executable.
std::string gguf_home_dir();
