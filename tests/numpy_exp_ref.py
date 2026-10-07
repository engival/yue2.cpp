#!/usr/bin/env python3
"""Reference for tests/numpy_exp.cpp: numpy's float32 exp over a fixed sweep
of float32 bit patterns (every 997th, NaNs skipped), as count + SHA-256 of
the little-endian outputs. Must run on x86-64 with AVX2 or AVX512F: that is the
numpy kernel convert_vae.py ran, and on other CPUs numpy falls back to libm.

	python3 -I tests/numpy_exp_ref.py
"""
import hashlib

import numpy as np

bits = np.arange(0, 1 << 32, 997, dtype=np.uint64).astype(np.uint32)
x = bits.view(np.float32)
x = x[~np.isnan(x)]
with np.errstate(over="ignore"):
	y = np.exp(x)
print(len(x), hashlib.sha256(y.astype("<f4").tobytes()).hexdigest())
