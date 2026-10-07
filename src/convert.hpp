// `yue2 convert` — m-a-p's safetensors to the three GGUFs, no Python.
// SPEC_CONVERT.md. The Python converters in convert/ are the reference; every
// tensor this writes is byte-for-byte what they (and llama-quantize) write.
#pragma once

#include <string>

struct ConvertParams
{
	std::string src_3b;             // --src-3b: snapshot dir of m-a-p/YuE2-3B
	std::string src_vae;            // --src-vae: snapshot dir of m-a-p/YuE2-Vae
	std::string out;                // --out: empty = where `yue2 song` looks
	std::string ar_type  = "q8_0";  // q8_0 | f16 | bf16
	std::string nar_type = "f16";   // f16 | f32
	std::string vae_type = "f32";   // f32 | f16
	bool        ar       = true;    // --only narrows these three
	bool        nar      = true;
	bool        vae      = true;
	bool        force    = false;
};

ConvertParams parse_convert_args(const char * argv0, int argc, char ** argv);

// Writes the selected GGUFs into p.out (which must be set); skips existing
// ones unless p.force. Dies on any error.
int run_convert(const ConvertParams & p);

// `yue2 song` / `yue2 batch` on first run: the default GGUFs that are missing
// are converted into out_dir from the Hugging Face cache, with one line of
// notice. Dies with the `hf download` to run when a snapshot is not there.
void convert_missing_defaults(const std::string & out_dir, bool ar, bool nar, bool vae);
