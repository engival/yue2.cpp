// Stage 1 — Oobleck VAE decoder (latent -> 48 kHz stereo). SPEC.md, src/STATUS.md.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct VaeParams
{
	std::string model;
	std::string input;
	std::string output;
	std::string npy_out;
	std::string device    = "vulkan";   // the CPU is never a default or a fallback: --cpu asks for it
	int       gpu         = 0;
	int       threads     = 0;
	int       core_frames = 256;
	int       halo_frames = 16;
	ggml_type im2col_type = GGML_TYPE_F32;
	int64_t   frames      = 0;     // 0 = all
	bool      full        = false;
	bool      vk_f16_matmul = false;
	std::vector<std::pair<std::string, std::string>> tags;   // Vorbis comments for a .flac output; `yue2 song` fills them
};

VaeParams parse_vae_args(const char * argv0, int argc, char ** argv);
int       run_vae(const VaeParams & p);
