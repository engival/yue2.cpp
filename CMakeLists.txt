cmake_minimum_required(VERSION 3.16)
project(yue2.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_VULKAN "Build the ggml Vulkan backend" ON)

# ggml comes from llama.cpp/ggml — one copy in the process, shared by both targets.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

# -Wall/-Wextra from llama.cpp are directory-scoped, so ask for them here too.
if(MSVC)
	set(YUE2_WARN_FLAGS /W4)
else()
	set(YUE2_WARN_FLAGS -Wall -Wextra -Wshadow)
endif()

# libFLAC 1.5 (system) writes the 24-bit output `yue2 song --out X.flac` produces.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLAC REQUIRED IMPORTED_TARGET flac)

# libopusenc is optional: with it, an output named X.opus is encoded as Ogg Opus;
# without it, the stages refuse such a name while parsing their arguments.
option(YUE2_OPUS "Opus output via libopusenc when it is installed" ON)
if(YUE2_OPUS)
	pkg_check_modules(OPUSENC IMPORTED_TARGET libopusenc)
endif()
if(YUE2_OPUS AND OPUSENC_FOUND)
	message(STATUS "yue2: Opus output enabled (libopusenc ${OPUSENC_VERSION})")
else()
	message(STATUS "yue2: Opus output disabled (no libopusenc; FLAC and WAV only)")
endif()

# Everything that compiles src/stage_vae.cpp needs the writer and its flag.
function(yue2_add_opus target)
	if(YUE2_OPUS AND OPUSENC_FOUND)
		target_compile_definitions(${target} PRIVATE YUE2_HAVE_OPUS)
		target_link_libraries(${target} PRIVATE PkgConfig::OPUSENC)
	endif()
endfunction()

# Each stage is one translation unit with its own Model/Builder/Graph behind an
# anonymous namespace, so the per-stage binaries and `yue2` can share the source.
add_executable(yue2-vae src/yue2-vae.cpp src/stage_vae.cpp)
target_include_directories(yue2-vae PRIVATE src)
target_link_libraries(yue2-vae PRIVATE ggml PkgConfig::FLAC)
target_compile_options(yue2-vae PRIVATE ${YUE2_WARN_FLAGS})
yue2_add_opus(yue2-vae)

# vendor::hash gives us llama.cpp's SHA-256 (hash_sha256_hex) and vendor::nlohmann
# the JSON header — both are built unconditionally by llama.cpp/vendor.
add_executable(yue2-ar src/yue2-ar.cpp src/stage_ar.cpp)
target_include_directories(yue2-ar PRIVATE src)
target_link_libraries(yue2-ar PRIVATE llama ggml vendor::hash vendor::nlohmann)
target_compile_options(yue2-ar PRIVATE ${YUE2_WARN_FLAGS})

# Stage 3: the NAR flow-matching half, raw ggml only (SPEC_NAR.md section 4.1).
add_executable(yue2-nar src/yue2-nar.cpp src/stage_nar.cpp)
target_include_directories(yue2-nar PRIVATE src)
target_link_libraries(yue2-nar PRIVATE ggml)
target_compile_options(yue2-nar PRIVATE ${YUE2_WARN_FLAGS})

# Stage 4: one binary for the whole song (SPEC_SINGLE.md). `yue2 song` runs the
# AR and NAR in this process and the VAE as a child of it, because ggml-vulkan's
# F32-exactness switch is read once per device init (SPEC_SINGLE.md section 2.2).
add_executable(yue2 src/yue2.cpp src/stage_song.cpp
	src/stage_ar.cpp src/stage_nar.cpp src/stage_vae.cpp)
target_include_directories(yue2 PRIVATE src)
target_link_libraries(yue2 PRIVATE llama ggml vendor::hash vendor::nlohmann PkgConfig::FLAC)
target_compile_options(yue2 PRIVATE ${YUE2_WARN_FLAGS})
yue2_add_opus(yue2)

# Stage 6: the sparse sampler's differential test (SPEC_SAMPLER.md §4.2). It
# includes src/stage_ar.cpp to reach both samplers in its anonymous namespace,
# so it tests the shipping code rather than a copy. Tests only — off by default,
# never installed: cmake -B build_sampler -DYUE2_BUILD_TESTS=ON
option(YUE2_BUILD_TESTS "Build the tests-only targets (yue2-sampler-diff, yue2-bars)" OFF)
if(YUE2_BUILD_TESTS)
	add_executable(yue2-sampler-diff tests/sampler_diff.cpp)
	target_include_directories(yue2-sampler-diff PRIVATE src)
	target_link_libraries(yue2-sampler-diff PRIVATE llama ggml vendor::hash vendor::nlohmann)
	target_compile_options(yue2-sampler-diff PRIVATE ${YUE2_WARN_FLAGS})

	# Stage 7: the score template's bar counter and parser (SPEC_TEMPLATE.md §4),
	# included the same way and for the same reason.
	add_executable(yue2-bars tests/bars.cpp)
	target_include_directories(yue2-bars PRIVATE src)
	target_link_libraries(yue2-bars PRIVATE llama ggml vendor::hash vendor::nlohmann)
	target_compile_options(yue2-bars PRIVATE ${YUE2_WARN_FLAGS})

	# Stage 7 (guidance): the weight curves, the branch lifecycle rule and every
	# request error of SPEC_GUIDANCE.md §2.3/§2.4 — no model, no device.
	add_executable(yue2-guidance tests/guidance.cpp)
	target_include_directories(yue2-guidance PRIVATE src)
	target_link_libraries(yue2-guidance PRIVATE llama ggml vendor::hash vendor::nlohmann)
	target_compile_options(yue2-guidance PRIVATE ${YUE2_WARN_FLAGS})

	# Stage 9 (handover): the offset measurement of SPEC_HANDOVER.md §4, the
	# forced history of §5 and every request error of §2 — synthetic streams
	# and a score in a string, so no model and no device either.
	add_executable(yue2-handover tests/handover.cpp)
	target_include_directories(yue2-handover PRIVATE src)
	target_link_libraries(yue2-handover PRIVATE llama ggml vendor::hash vendor::nlohmann)
	target_compile_options(yue2-handover PRIVATE ${YUE2_WARN_FLAGS})
endif()
