// Backend selection, shared by all three stages (--device cpu|vulkan, --gpu N).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "common/util.hpp"

#include <string>

// ggml-vulkan's mul_mm shader stages BOTH operands as float16_t whenever the
// device advertises fp16 (and the KHR_coopmat path is fp16 by construction),
// even for an F32 x F32 matmul. Disabling both selects the genuinely-F32 matmul
// pipelines. Must happen before the backend is initialised, and ggml-vulkan
// reads these once per device init — one process cannot have it both ways, which
// is why `yue2 song` runs the VAE as a child (SPEC_SINGLE.md §2.2). A value
// already in the environment wins (overwrite = 0).
// See docs/vulkan_burst_investigation.md.
inline void vulkan_want_exact_f32()
{
	setenv("GGML_VK_DISABLE_F16", "1", 0);
	setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
}

inline ggml_backend_dev_t vulkan_device(int gpu)
{
	ggml_backend_load_all();
	ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
	if (reg == nullptr)
	{
		die("this binary has no Vulkan backend (build with -DGGML_VULKAN=ON)");
	}
	const size_t n = ggml_backend_reg_dev_count(reg);
	if (gpu < 0 || (size_t) gpu >= n)
	{
		die("--gpu %d out of range, %zu Vulkan device(s)", gpu, n);
	}
	return ggml_backend_reg_dev_get(reg, (size_t) gpu);
}

// The raw-ggml stages' backend. `threads` > 0 is passed on where the backend
// takes it (CPU only in practice).
inline ggml_backend_t init_compute_backend(const std::string & device, int gpu, int threads)
{
	ggml_backend_t backend = nullptr;
	if (device == "cpu")
	{
		// No ggml_backend_load_all() here: with GGML_BACKEND_DL=OFF the CPU
		// backend is registered statically, so this never creates a Vulkan
		// instance on a machine whose driver would rather not be poked.
		backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
	} else if (device == "vulkan") {
		backend = ggml_backend_dev_init(vulkan_device(gpu), nullptr);
	} else {
		die("--device must be cpu or vulkan");
	}
	if (backend == nullptr)
	{
		die("failed to initialise the %s backend", device.c_str());
	}
	if (threads > 0)
	{
		ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
		auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
		if (set_threads != nullptr)
		{
			set_threads(backend, threads);
		}
	}
	return backend;
}
