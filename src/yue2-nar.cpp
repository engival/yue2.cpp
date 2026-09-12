// yue2-nar — stage 3 on its own. The code lives in src/stage_nar.cpp, which
// `yue2 nar` calls through the same two functions.
#include "stage_nar.hpp"

int main(int argc, char ** argv)
{
	return run_nar(parse_nar_args(argv[0], argc, argv));
}
