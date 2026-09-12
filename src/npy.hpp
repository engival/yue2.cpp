// Minimal .npy reader/writer: header versions 1-3, '<f4' and '<i4', C-order only.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npy
{

template <typename T>
struct ArrayT
{
	std::vector<int64_t> shape;
	std::vector<T>       data;
};

using Array    = ArrayT<float>;
using ArrayI32 = ArrayT<int32_t>;

// numpy dtype string for the element types we support.
template <typename T>
inline const char * descr();

template <>
inline const char * descr<float>()
{
	return "<f4";
}

template <>
inline const char * descr<int32_t>()
{
	return "<i4";
}

// Parses "(4109, 64)" / "(64,)" / "()" into a shape vector.
// Digit runs are bounded so a hostile header cannot throw out_of_range.
inline bool parse_shape(const std::string & s, std::vector<int64_t> & out)
{
	out.clear();
	size_t i = 0;
	while (i < s.size() && s[i] != '(')
	{
		i++;
	}
	if (i == s.size())
	{
		return false;
	}
	i++;
	std::string num;
	for (; i < s.size() && s[i] != ')'; i++)
	{
		if (isdigit((unsigned char) s[i]))
		{
			num += s[i];
			if (num.size() > 18)
			{
				return false;
			}
		} else if (s[i] == ',' || s[i] == ' ') {
			if (!num.empty())
			{
				out.push_back((int64_t) strtoll(num.c_str(), nullptr, 10));
				num.clear();
			}
		} else {
			return false;
		}
	}
	if (!num.empty())
	{
		out.push_back((int64_t) strtoll(num.c_str(), nullptr, 10));
	}
	return i < s.size();
}

// Returns an empty error string on success.
template <typename T>
inline std::string load_t(const char * path, ArrayT<T> & out)
{
	FILE * f = fopen(path, "rb");
	if (f == nullptr)
	{
		return std::string("cannot open ") + path;
	}

	unsigned char magic[8];
	if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0)
	{
		fclose(f);
		return "not a .npy file";
	}

	const int major = magic[6];
	uint32_t header_len = 0;
	if (major == 1)
	{
		uint16_t n = 0;
		if (fread(&n, 1, 2, f) != 2)
		{
			fclose(f);
			return "truncated npy header";
		}
		header_len = n;
	} else if (major == 2 || major == 3) {
		uint32_t n = 0;
		if (fread(&n, 1, 4, f) != 4)
		{
			fclose(f);
			return "truncated npy header";
		}
		header_len = n;
	} else {
		fclose(f);
		return "unsupported npy version";
	}

	std::string header(header_len, '\0');
	if (header_len > 0 && fread(&header[0], 1, header_len, f) != header_len)
	{
		fclose(f);
		return "truncated npy header";
	}

	const std::string want  = descr<T>();
	const std::string tick  = "'descr': '" + want + "'";
	const std::string quote = "\"descr\": \"" + want + "\"";
	if (header.find(tick) == std::string::npos && header.find(quote) == std::string::npos)
	{
		fclose(f);
		return "only little-endian '" + want + "' npy input is supported, got: " + header;
	}
	if (header.find("'fortran_order': False") == std::string::npos &&
	    header.find("\"fortran_order\": false") == std::string::npos)
	{
		fclose(f);
		return "only C-order npy input is supported";
	}

	const size_t sp = header.find("'shape'");
	if (sp == std::string::npos || !parse_shape(header.substr(sp), out.shape))
	{
		fclose(f);
		return "cannot parse npy shape";
	}

	// Element count, with the overflow and file-size checks a hostile header needs.
	size_t n = 1;
	for (int64_t d : out.shape)
	{
		if (d < 0)
		{
			fclose(f);
			return "negative npy dimension";
		}
		if (d != 0 && n > SIZE_MAX / (size_t) d)
		{
			fclose(f);
			return "npy shape overflows";
		}
		n *= (size_t) d;
	}
	if (n > SIZE_MAX / sizeof(T))
	{
		fclose(f);
		return "npy shape overflows";
	}

	const long data_start = ftell(f);
	if (data_start < 0 || fseek(f, 0, SEEK_END) != 0)
	{
		fclose(f);
		return "cannot size the npy file";
	}
	const long file_end = ftell(f);
	if (file_end < data_start || (uint64_t) (file_end - data_start) < (uint64_t) n * sizeof(T))
	{
		fclose(f);
		return "truncated npy data";
	}
	if (fseek(f, data_start, SEEK_SET) != 0)
	{
		fclose(f);
		return "cannot seek the npy file";
	}

	out.data.resize(n);
	if (n > 0 && fread(out.data.data(), sizeof(T), n, f) != n)
	{
		fclose(f);
		return "truncated npy data";
	}
	fclose(f);
	return "";
}

template <typename T>
inline std::string save_t(const char * path, const std::vector<int64_t> & shape, const T * data)
{
	std::string dict = std::string("{'descr': '") + descr<T>() + "', 'fortran_order': False, 'shape': (";
	size_t n = 1;
	for (size_t i = 0; i < shape.size(); i++)
	{
		if (shape[i] < 0)
		{
			return "negative npy dimension";
		}
		dict += std::to_string(shape[i]);
		dict += ",";
		if (shape[i] != 0 && n > SIZE_MAX / (size_t) shape[i])
		{
			return "npy shape overflows";
		}
		n *= (size_t) shape[i];
	}
	dict += "), }";
	// total header (10 bytes preamble + dict + '\n') padded to 64 bytes
	size_t total = 10 + dict.size() + 1;
	size_t pad   = (64 - (total % 64)) % 64;
	dict.append(pad, ' ');
	dict += "\n";
	if (dict.size() > 0xffff)
	{
		return "npy header too long for version 1";
	}

	FILE * f = fopen(path, "wb");
	if (f == nullptr)
	{
		return std::string("cannot write ") + path;
	}

	const uint16_t hl = (uint16_t) dict.size();
	bool           ok = fwrite("\x93NUMPY\x01\x00", 1, 8, f) == 8;
	ok = ok && fwrite(&hl, 1, 2, f) == 2;
	ok = ok && fwrite(dict.data(), 1, dict.size(), f) == dict.size();
	ok = ok && (n == 0 || fwrite(data, sizeof(T), n, f) == n);
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

// ------------------------------------------------------- typed front ends ---

inline std::string load(const char * path, Array & out)
{
	return load_t<float>(path, out);
}

inline std::string save(const char * path, const std::vector<int64_t> & shape, const float * data)
{
	return save_t<float>(path, shape, data);
}

inline std::string load_i32(const char * path, ArrayI32 & out)
{
	return load_t<int32_t>(path, out);
}

inline std::string save_i32(const char * path, const std::vector<int64_t> & shape, const int32_t * data)
{
	return save_t<int32_t>(path, shape, data);
}

} // namespace npy
