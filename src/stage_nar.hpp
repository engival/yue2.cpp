// Stage 3 — NAR flow matching (semantic tokens -> [T, 64] latent).
// SPEC_NAR.md, src/STATUS_NAR.md, src/STATUS_NAR_PERF.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct NarParams
{
	std::string ar_model;
	std::string model;
	std::string artifacts;
	std::string prefix_path;
	std::string codec_path;
	std::string noise_path;
	std::string output;
	std::string dump_dir;
	std::string device        = "vulkan";   // the CPU is never a default or a fallback: --cpu asks for it
	int      gpu              = 0;
	int      threads          = 0;
	int      steps            = 32;
	int64_t  context          = 24576;
	int64_t  query_chunk      = 1024;
	int64_t  prefill_block    = 512;
	int64_t  frames           = 0;     // 0 = all
	bool     vk_f16_matmul    = false;
	bool     widen_f16        = false;
	bool     dump_kv_all      = false;
	bool     flash_attn       = false;
	bool     kv_f16           = false;
	bool     have_seed        = false;
	uint64_t seed             = 0;

	// `yue2 song` hands the tokens and the noise over in memory rather than
	// through prefix.npy / semantic.npy / nar_noise.npy (SPEC_SINGLE.md §2.3).
	// The files are still written, by the caller; these just skip the reload.
	const std::vector<int32_t> * prefix_in = nullptr;
	const std::vector<int32_t> * codec_in  = nullptr;
	const std::vector<float>   * noise_in  = nullptr;
};

NarParams parse_nar_args(const char * argv0, int argc, char ** argv);
int       run_nar(const NarParams & p);
