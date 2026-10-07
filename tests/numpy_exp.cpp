// SPEC_CONVERT: `yue2 convert` ports numpy's float32 exp so the VAE's
// exp(alpha)/exp(beta) match convert_vae.py bit for bit. This reruns that port
// over a fixed sweep of float32 bit patterns and compares the hash of the
// outputs with numpy's (tests/numpy_exp_ref.py, numpy 2.3.3, AVX512F). CPU
// only, no model. It includes src/convert.cpp to reach numpy_expf() in its
// anonymous namespace, so it tests the shipping code, compiled with the same
// no-contraction flag.

#include "convert.cpp"

int main()
{
	const char * const want_hash  = "7227092ca08f90ff57623921b83b70628acfc1f26fdf02fdec70b5f85408f588";
	const uint64_t     want_count = 4291064;

	sha256_t ctx;
	sha256_init(&ctx);
	uint64_t count = 0;
	for (uint64_t bits = 0; bits < ((uint64_t) 1 << 32); bits += 997)
	{
		const uint32_t u = (uint32_t) bits;
		float          x;
		memcpy(&x, &u, sizeof(x));
		if (std::isnan(x))
		{
			continue;
		}
		const float y = numpy_expf(x);
		sha256_update(&ctx, (const unsigned char *) &y, sizeof(y));   // little-endian hosts only
		count++;
	}
	unsigned char digest[SHA256_DIGEST_SIZE];
	sha256_final(&ctx, digest);
	char got_hash[2 * SHA256_DIGEST_SIZE + 1];
	for (int i = 0; i < SHA256_DIGEST_SIZE; i++)
	{
		snprintf(got_hash + 2 * i, 3, "%02x", digest[i]);
	}
	const bool ok = count == want_count && strcmp(got_hash, want_hash) == 0;
	printf("numpy_exp: %llu values, sha256 %s -> %s\n", (unsigned long long) count, got_hash, ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
