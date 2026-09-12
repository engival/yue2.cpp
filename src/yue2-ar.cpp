// yue2-ar — stage 2 on its own. The code lives in src/stage_ar.cpp, which
// `yue2 ar` calls through the same two functions.
#include "stage_ar.hpp"

int main(int argc, char ** argv)
{
	return run_ar(parse_ar_args(argv[0], argc, argv), nullptr);
}
