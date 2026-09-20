// Ogg Opus writer (libopusenc), the lossy output `--out X.opus` asks for.
// Optional: a build configured without libopusenc keeps this header, and
// save_f32() then only explains itself — the CLI refuses an .opus output while
// it parses its arguments, long before a model loads. SPEC_SINGLE.md §2.5.
#pragma once

#include "tags.hpp"

#include <cstdint>
#include <string>

#ifdef YUE2_HAVE_OPUS
#include <opusenc.h>

#include <vector>
#endif

namespace opus
{

using Tags = vorbis::Tags;

// Opus is constant-quality-ish VBR around a target rate; 160 kbit/s stereo is
// transparent enough for a render nobody will master again.
const int BITRATE_MIN     = 16;
const int BITRATE_MAX     = 510;
const int BITRATE_DEFAULT = 160;

// The container is chosen by extension, exactly like .flac and .wav.
inline bool wanted(const std::string & path)
{
	return path.size() >= 5 && path.compare(path.size() - 5, 5, ".opus") == 0;
}

// Why an Opus-less build cannot write this file, and what to do about it.
inline std::string unsupported(const std::string & path)
{
	return "this build has no Opus support, so it cannot write " + path +
	       " — install libopusenc (pkg-config libopusenc) and configure again with -DYUE2_OPUS=ON";
}

// "" when this build can write `path`; otherwise unsupported(). A path that is
// not an .opus is always fine, so callers can ask about any output name.
inline std::string check_support(const std::string & path)
{
#ifdef YUE2_HAVE_OPUS
	(void) path;
	return "";
#else
	return wanted(path) ? unsupported(path) : "";
#endif
}

#ifdef YUE2_HAVE_OPUS

// planar: channels * samples, channel-major, as yue2-vae holds its audio;
// libopusenc takes it interleaved, so it is repacked a chunk at a time and
// clamped to [-1, 1] on the way (the encoder's own float range).
//
// Tags are written as given; libopusenc adds a vendor string of its own naming
// the two libraries, so an .opus is never quite tagless.
inline std::string save_f32(const char * path, const float * planar, int channels,
	int64_t samples, int sample_rate, int bitrate, const Tags & tags = {})
{
	if (channels <= 0 || samples < 0 || sample_rate <= 0)
	{
		return "opus: bad channel count, length or sample rate";
	}
	if (bitrate < BITRATE_MIN || bitrate > BITRATE_MAX)
	{
		return "opus: the bitrate must be between " + std::to_string(BITRATE_MIN) + " and " +
		       std::to_string(BITRATE_MAX) + " kbit/s";
	}
	// Opus itself runs at 48 kHz and libopusenc only accepts the input rates it
	// can turn into that; ours is 48 kHz already, and resampling anything else is
	// not this writer's job.
	if (sample_rate != 8000 && sample_rate != 12000 && sample_rate != 16000 &&
	    sample_rate != 24000 && sample_rate != 48000)
	{
		return "opus: " + std::to_string(sample_rate) +
		       " Hz is not one of libopusenc's input rates (8, 12, 16, 24 or 48 kHz)";
	}

	OggOpusComments * comments = ope_comments_create();
	if (comments == nullptr)
	{
		return "opus: cannot create the comment block";
	}
	for (const auto & tag : tags)
	{
		if (ope_comments_add(comments, tag.first.c_str(), tag.second.c_str()) != OPE_OK)
		{
			ope_comments_destroy(comments);
			return "opus: cannot store the tag \"" + tag.first + "\" (name printable ASCII without '=', value UTF-8)";
		}
	}

	// Mapping family 0 is mono and stereo; anything wider is a surround stream.
	int          err = OPE_OK;
	OggOpusEnc * enc = ope_encoder_create_file(path, comments, sample_rate, channels,
		channels > 2 ? 1 : 0, &err);
	if (enc == nullptr)
	{
		ope_comments_destroy(comments);
		return std::string("cannot write ") + path + ": " + ope_strerror(err);
	}
	err = ope_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate * 1000));
	if (err != OPE_OK)
	{
		ope_encoder_destroy(enc);
		ope_comments_destroy(comments);
		return std::string("opus: the encoder rejected ") + std::to_string(bitrate) +
		       " kbit/s: " + ope_strerror(err);
	}

	const int64_t      chunk = 1 << 14;
	std::vector<float> buf((size_t) (chunk * channels));
	for (int64_t i = 0; err == OPE_OK && i < samples; i += chunk)
	{
		const int64_t n = samples - i < chunk ? samples - i : chunk;
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
		err = ope_encoder_write_float(enc, buf.data(), (int) n);
	}
	if (err == OPE_OK)
	{
		err = ope_encoder_drain(enc);
	}
	ope_encoder_destroy(enc);
	ope_comments_destroy(comments);
	if (err != OPE_OK)
	{
		return std::string("short write on ") + path + ": " + ope_strerror(err);
	}
	return "";
}

#else

inline std::string save_f32(const char * path, const float *, int, int64_t, int, int, const Tags & = {})
{
	return unsupported(path);
}

#endif

} // namespace opus
