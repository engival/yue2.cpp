// Whole-file read/write, SHA-256 and the python-shaped JSON dump the artifacts
// use. Shared by the AR stage (plan.json, plan_manifest.json) and `yue2 song`
// (config.json, result.json).
#pragma once

#include <hash/hash.h>
#include <nlohmann/json.hpp>

#include "common/util.hpp"

#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;

// Reads a whole file; empty error string on success.
inline std::string read_file(const std::string & path, std::string & out)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
	{
		return "cannot open " + path;
	}
	std::ostringstream ss;
	ss << in.rdbuf();
	out = ss.str();
	return "";
}

inline std::string write_file(const std::string & path, const std::string & body)
{
	std::ofstream out(path, std::ios::binary);
	if (!out)
	{
		return "cannot write " + path;
	}
	out.write(body.data(), (std::streamsize) body.size());
	out.close();
	return out.fail() ? "cannot write " + path : "";
}

inline void write_file_or_die(const std::string & path, const std::string & body)
{
	const std::string err = write_file(path, body);
	if (!err.empty())
	{
		die("%s", err.c_str());
	}
}

// SHA-256 of a whole file, hex, using llama.cpp's vendored hash (vendor::hash).
// The artifact files are small enough to read into memory.
inline std::string sha256_file_hex(const std::string & path)
{
	std::string body;
	if (!read_file(path, body).empty())
	{
		return "";
	}
	return hash_sha256_hex(body.data(), body.size());
}

// python json.dumps(value, indent=2, ensure_ascii=False) + "\n".
// error_handler_t::replace mirrors the reference's errors="replace" decode:
// detokenized ABC can end on a partial multibyte sequence when it is truncated,
// and the default strict handler would throw after the whole run.
inline std::string dump_py(const json & value)
{
	return value.dump(2, ' ', false, json::error_handler_t::replace) + "\n";
}
