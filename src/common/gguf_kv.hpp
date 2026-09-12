// GGUF key/value readers with a fallback, shared by the VAE and NAR loaders.
#pragma once

#include "gguf.h"

#include <string>
#include <vector>

inline int kv_i32(gguf_context * gc, const char * key, int fallback)
{
	const int64_t id = gguf_find_key(gc, key);
	if (id < 0)
	{
		return fallback;
	}
	switch (gguf_get_kv_type(gc, id))
	{
		case GGUF_TYPE_INT32:  return gguf_get_val_i32(gc, id);
		case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(gc, id);
		case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(gc, id);
		default:               return fallback;
	}
}

inline float kv_f32(gguf_context * gc, const char * key, float fallback)
{
	const int64_t id = gguf_find_key(gc, key);
	if (id < 0 || gguf_get_kv_type(gc, id) != GGUF_TYPE_FLOAT32)
	{
		return fallback;
	}
	return gguf_get_val_f32(gc, id);
}

inline std::string kv_str(gguf_context * gc, const char * key)
{
	const int64_t id = gguf_find_key(gc, key);
	if (id < 0 || gguf_get_kv_type(gc, id) != GGUF_TYPE_STRING)
	{
		return "";
	}
	return gguf_get_val_str(gc, id);
}

inline std::vector<int> kv_i32_array(gguf_context * gc, const char * key, const std::vector<int> & fallback)
{
	const int64_t id = gguf_find_key(gc, key);
	if (id < 0 || gguf_get_kv_type(gc, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gc, id) != GGUF_TYPE_INT32)
	{
		return fallback;
	}
	const int32_t * p = (const int32_t *) gguf_get_arr_data(gc, id);
	const size_t n = gguf_get_arr_n(gc, id);
	return std::vector<int>(p, p + n);
}
