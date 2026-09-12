// Stage 4 — request -> FLAC in one process. SPEC_SINGLE.md.
// Stage 5 — several requests, one AR decode loop. SPEC_BATCH.md.
#pragma once

#include "stage_ar.hpp"

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
	int         steps    = 32;
	bool        has_seed = false;
	uint64_t    seed     = 0;
	bool        nar_f32  = false;
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
	int         steps    = 32;
	int         parallel = 4;
	int         threads  = 0;
	bool        greedy   = false;
	int         max_abc      = 0;
	int         max_semantic = 0;
	bool        has_seed = false;   // the default seed for jobs that name none
	uint64_t    seed     = 0;
	bool        nar_f32  = false;
	bool        continue_on_error = false;
};

SongParams  parse_song_args (const char * argv0, int argc, char ** argv);
BatchParams parse_batch_args(const char * argv0, int argc, char ** argv);

int run_song (const SongParams & p);
int run_batch(const BatchParams & p, std::vector<ArJob> jobs);
