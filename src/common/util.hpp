// Tiny helpers the three stages had a copy of each: fatal error, argument
// fetch, clock. Nothing speculative lives here — only code that was literally
// duplicated. See SPEC_SINGLE.md §2.1.
#pragma once

#include "ggml.h"

#include <cctype>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

[[noreturn]] inline void die(const char * fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "error: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(1);
}

// argv[i] is a flag that takes a value; returns it and advances i.
inline const char * need(int argc, char ** argv, int & i)
{
	if (i + 1 >= argc)
	{
		die("%s requires an argument", argv[i]);
	}
	return argv[++i];
}

inline int64_t now_us()
{
	return ggml_time_us();
}

inline double now_seconds()
{
	return (double) ggml_time_us() / 1e6;
}

// protocol.SongRequest's seed: [0, 2**63), exactly, never through a double.
// `what` names the flag in the error. Every --seed goes through this — a bare
// strtoull turns "--seed abc" into a silent seed 0.
inline uint64_t parse_seed_arg(const char * what, const char * text)
{
	if (text[0] == '-' || text[0] == '+' || text[0] == '\0' || isspace((unsigned char) text[0]))
	{
		die("%s must be a non-negative integer below 2**63 (got \"%s\")", what, text);
	}
	errno = 0;
	char *         endp  = nullptr;
	const uint64_t value = strtoull(text, &endp, 10);
	if (errno != 0 || endp == text || *endp != '\0')
	{
		die("%s must be a non-negative integer below 2**63 (got \"%s\")", what, text);
	}
	if (value >= (uint64_t) 1 << 63)
	{
		die("%s must be below 2**63 (got \"%s\")", what, text);
	}
	return value;
}
