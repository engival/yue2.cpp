// Standard-normal noise for the NAR, from our own RNG.
//
// Seed compatibility with torch is NOT a goal and was already gone at the AR
// stage (yue2-ar samples with std::mt19937_64, torch used a GPU Philox), so the
// only requirement is that one seed gives the same bytes on every machine,
// forever. std::mt19937_64 is bit-specified by the standard; the distributions
// in <random> are not — std::normal_distribution and std::uniform_real_distribution
// are both explicitly implementation-defined, so neither appears here.
// SPEC_SINGLE.md §2.4.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace noise
{

// A 64-bit draw mapped into the open interval (0, 1): 53 significant bits,
// offset by half an ulp so log(u) below can never see a zero.
inline double unit(uint64_t bits)
{
	return ((double) (bits >> 11) + 0.5) * 0x1.0p-53;
}

// float32[rows, cols], row-major, standard normal, via Box-Muller.
inline std::vector<float> gaussian(uint64_t seed, int64_t rows, int64_t cols)
{
	std::mt19937_64    rng(seed);
	std::vector<float> out((size_t) (rows * cols));
	for (size_t i = 0; i < out.size(); i += 2)
	{
		const double r     = std::sqrt(-2.0 * std::log(unit(rng())));
		const double theta = 6.2831853071795864769 * unit(rng());
		out[i] = (float) (r * std::cos(theta));
		if (i + 1 < out.size())
		{
			out[i + 1] = (float) (r * std::sin(theta));
		}
	}
	return out;
}

} // namespace noise
