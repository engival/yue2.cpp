// 24-bit FLAC writer (libFLAC 1.5 stream encoder), the format soundfile's
// PCM_24 produced for the reference pipeline. SPEC_SINGLE.md §2.5.
#pragma once

#include <FLAC/stream_encoder.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace flac
{

// planar: channels * samples, channel-major, as yue2-vae holds its audio.
// Samples are clamped to [-1, 1], scaled and rounded to nearest (round-half-even,
// like llrint and numpy.rint), then clipped into the 24-bit range.
//
// The scale is 2**23, not 2**23 - 1: libsndfile's float -> PCM_24 multiplies by
// 0x800000 and clips the integer, and measuring it (ref/x ratio, exactly
// 8388608.0 on every non-zero sample) is what makes our FLAC bit-identical to
// soundfile's rather than 12 % of samples one LSB away. SPEC_SINGLE.md §2.5 says
// 8388607; see src/STATUS_SINGLE.md for the measurement.
const double SCALE_24 = 8388608.0;
const double MAX_24   =  8388607.0;
const double MIN_24   = -8388608.0;
inline std::string save_f32_24(const char * path, const float * planar, int channels,
	int64_t samples, int sample_rate)
{
	if (channels <= 0 || samples < 0 || sample_rate <= 0)
	{
		return "flac: bad channel count, length or sample rate";
	}

	FLAC__StreamEncoder * enc = FLAC__stream_encoder_new();
	if (enc == nullptr)
	{
		return "flac: cannot create the encoder";
	}

	bool ok = FLAC__stream_encoder_set_verify(enc, false) &&
	          FLAC__stream_encoder_set_compression_level(enc, 5) &&
	          FLAC__stream_encoder_set_channels(enc, (uint32_t) channels) &&
	          FLAC__stream_encoder_set_bits_per_sample(enc, 24) &&
	          FLAC__stream_encoder_set_sample_rate(enc, (uint32_t) sample_rate) &&
	          FLAC__stream_encoder_set_total_samples_estimate(enc, (FLAC__uint64) samples);
	if (!ok)
	{
		FLAC__stream_encoder_delete(enc);
		return "flac: the encoder rejected the stream parameters";
	}

	if (FLAC__stream_encoder_init_file(enc, path, nullptr, nullptr) != FLAC__STREAM_ENCODER_INIT_STATUS_OK)
	{
		FLAC__stream_encoder_delete(enc);
		return std::string("cannot write ") + path;
	}

	const int64_t            chunk = 1 << 14;
	std::vector<FLAC__int32> buf((size_t) (chunk * channels));
	for (int64_t i = 0; ok && i < samples; i += chunk)
	{
		const int64_t n = samples - i < chunk ? samples - i : chunk;
		for (int64_t j = 0; j < n; j++)
		{
			for (int c = 0; c < channels; c++)
			{
				double v = planar[(int64_t) c * samples + i + j];
				if (v > 1.0)
				{
					v = 1.0;
				}
				if (v < -1.0)
				{
					v = -1.0;
				}
				double q = std::llrint(v * SCALE_24);
				if (q > MAX_24)
				{
					q = MAX_24;
				}
				if (q < MIN_24)
				{
					q = MIN_24;
				}
				buf[(size_t) (j * channels + c)] = (FLAC__int32) q;
			}
		}
		ok = FLAC__stream_encoder_process_interleaved(enc, buf.data(), (uint32_t) n);
	}

	ok = FLAC__stream_encoder_finish(enc) && ok;
	FLAC__stream_encoder_delete(enc);
	if (!ok)
	{
		return std::string("short write on ") + path;
	}
	return "";
}

} // namespace flac
