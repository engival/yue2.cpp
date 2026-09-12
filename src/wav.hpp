// Minimal 32-bit float RIFF/WAVE writer.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace wav
{

// planar: channels * samples, channel-major. Samples clamped to [-1, 1].
inline std::string save_f32(const char * path, const float * planar, int channels, int64_t samples, int sample_rate)
{
	if (channels <= 0 || samples < 0 || sample_rate <= 0)
	{
		return "wav: bad channel count, length or sample rate";
	}

	// RIFF sizes are 32-bit: refuse rather than write a corrupt header.
	const uint64_t bytes = (uint64_t) samples * (uint64_t) channels * 4u;
	if (bytes + 38 > 0xffffffffull)
	{
		return "wav: > 4 GiB of samples does not fit a RIFF header";
	}

	FILE * f = fopen(path, "wb");
	if (f == nullptr)
	{
		return std::string("cannot write ") + path;
	}

	const uint32_t data_bytes  = (uint32_t) bytes;
	const uint32_t byte_rate   = (uint32_t) sample_rate * (uint32_t) channels * 4u;
	const uint16_t block_align = (uint16_t) (channels * 4);

	bool ok = true;
	auto u32 = [&](uint32_t v) { ok = ok && fwrite(&v, 4, 1, f) == 1; };
	auto u16 = [&](uint16_t v) { ok = ok && fwrite(&v, 2, 1, f) == 1; };
	auto tag = [&](const char * s) { ok = ok && fwrite(s, 1, 4, f) == 4; };

	tag("RIFF");
	u32(36 + 2 + data_bytes);          // fmt chunk is 18 bytes for IEEE float
	tag("WAVE");
	tag("fmt ");
	u32(18);
	u16(3);                            // WAVE_FORMAT_IEEE_FLOAT
	u16((uint16_t) channels);
	u32((uint32_t) sample_rate);
	u32(byte_rate);
	u16(block_align);
	u16(32);
	u16(0);                            // cbSize
	tag("data");
	u32(data_bytes);

	std::vector<float> buf;
	const int64_t chunk = 1 << 14;
	buf.resize((size_t) chunk * channels);
	for (int64_t i = 0; ok && i < samples; i += chunk)
	{
		const int64_t n = (samples - i < chunk) ? samples - i : chunk;
		for (int64_t j = 0; j < n; j++)
		{
			for (int c = 0; c < channels; c++)
			{
				float v = planar[(int64_t) c * samples + i + j];
				if (v > 1.0f)
				{
					v = 1.0f;
				}
				if (v < -1.0f)
				{
					v = -1.0f;
				}
				buf[(size_t) (j * channels + c)] = v;
			}
		}
		ok = ok && fwrite(buf.data(), sizeof(float), (size_t) (n * channels), f) == (size_t) (n * channels);
	}
	if (fclose(f) != 0)
	{
		ok = false;
	}
	if (!ok)
	{
		return std::string("short write on ") + path;
	}
	return "";
}

} // namespace wav
