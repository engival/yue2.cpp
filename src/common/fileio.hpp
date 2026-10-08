// Whole-file read/write, SHA-256 and the python-shaped JSON dump the artifacts
// use. Shared by the AR stage (plan.json, plan_manifest.json) and `yue2 song`
// (config.json, result.json).
#pragma once

#include <hash/hash.h>
#include <nlohmann/json.hpp>

#include "common/util.hpp"

#include <filesystem>
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

// A request's text from files (README "Lyrics and scores from files"):
// "lyrics_file", "abc_file" and "abc_template_file" are read into "lyrics",
// "abc" and "abc_template" and the *_file key is dropped, so everything after
// this — request.json, ar_request.json, the FLAC's LYRICS tag — holds the text
// as read and never the path: a later edit of the file cannot change what an
// artifacts directory says it rendered. A relative path resolves against
// `base_dir`, the rule "semantic_keep" follows. CRLF becomes LF and a UTF-8 BOM
// is dropped (what Windows Notepad may write). `baked` = how many keys were read.
inline std::string resolve_request_files(json & root, const std::string & base_dir, int * baked = nullptr)
{
	static const char * const keys[][2] = {
		{ "lyrics_file",       "lyrics"       },
		{ "abc_file",          "abc"          },
		{ "abc_template_file", "abc_template" },
	};
	if (baked)
	{
		*baked = 0;
	}
	if (!root.is_object())
	{
		return "";
	}
	for (const auto & k : keys)
	{
		if (!root.contains(k[0]))
		{
			continue;
		}
		const std::string key = std::string("\"") + k[0] + "\"";
		if (!root[k[0]].is_string() || root[k[0]].get<std::string>().empty())
		{
			return key + " must be a file name";
		}
		if (root.contains(k[1]) && !root[k[1]].is_null())
		{
			return key + " and \"" + k[1] + "\" are both given; keep one";
		}
		std::filesystem::path path = root[k[0]].get<std::string>();
		if (path.is_relative() && !base_dir.empty())
		{
			path = (std::filesystem::path(base_dir) / path).lexically_normal();
		}
		std::string text;
		if (std::filesystem::is_directory(path) || !read_file(path.string(), text).empty())
		{
			return key + ": cannot open " + path.string();
		}
		if (text.empty())
		{
			return key + ": " + path.string() + " is empty";
		}
		if (text.compare(0, 3, "\xEF\xBB\xBF") == 0)
		{
			text.erase(0, 3);
		}
		std::string lf;
		lf.reserve(text.size());
		for (size_t i = 0; i < text.size(); i++)
		{
			if (text[i] != '\r' || i + 1 >= text.size() || text[i + 1] != '\n')
			{
				lf += text[i];
			}
		}
		try
		{
			(void) json(lf).dump();
		}
		catch (const std::exception &)
		{
			return key + ": " + path.string() + " is not UTF-8 text (save it as UTF-8)";
		}
		// The text takes the file key's place, so ar_request.json reads in the
		// order it was written.
		json out = json::object();
		for (auto it = root.begin(); it != root.end(); ++it)
		{
			if (it.key() == k[0])
			{
				out[k[1]] = lf;
			} else if (it.key() != k[1]) {
				out[it.key()] = it.value();
			}
		}
		root = std::move(out);
		if (baked)
		{
			(*baked)++;
		}
	}
	return "";
}
