// Vorbis comments — the tag list every writer here takes. Its own header so
// flac.hpp and opus.hpp can share the type without either one pulling in the
// other's library (and so an Opus-less build still has it).
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace vorbis
{

// NAME=value, the name printable ASCII without '=', the value UTF-8.
using Tags = std::vector<std::pair<std::string, std::string>>;

} // namespace vorbis
