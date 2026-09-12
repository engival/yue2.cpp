// yue2-vae — stage 1 on its own. The code lives in src/stage_vae.cpp, which
// `yue2 vae` calls through the same two functions.
#include "stage_vae.hpp"

int main(int argc, char ** argv)
{
	return run_vae(parse_vae_args(argv[0], argc, argv));
}
