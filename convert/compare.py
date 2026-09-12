#!/usr/bin/env python3
"""Compare two float32 audio arrays (.npy, or .wav/.flac via soundfile).

Used by the C++ team and the verification harness (SPEC.md section 5) to score
a ggml decode against a torch golden file, or a tiled full-song decode against
the reference audio.flac.

Both inputs are aligned to shape [2, samples] (mono is broadcast to stereo;
[samples, 2] is transposed) before comparing. Prints max abs error, RMS error,
and SNR in dB; exits 0 if --min-snr is met (or if no threshold given), else 1.

By default, both inputs are clamped to [-1, 1] before scoring. This matters
because the C++ decoder's `--npy` dump is the raw *unclamped* output (a VAE
can legitimately emit a handful of samples slightly outside +-1), while a
reference render like `audio.flac` was written through a pipeline that
clamps (and then PCM-quantizes) to [-1, 1] before saving. Comparing unclamped
vs. clamped means a few out-of-range samples dominate max-abs-err and RMS,
producing a misleadingly low SNR that has nothing to do with decode
accuracy. Pass --no-clamp to compare the raw values instead (e.g. to
deliberately measure how many/how far samples exceed the [-1, 1] range).

Usage:
	venv_yue2/bin/python compare.py a.npy b.npy [--min-snr 60]
	venv_yue2/bin/python compare.py out.npy audio.flac [--min-snr 20]
	venv_yue2/bin/python compare.py out.npy audio.flac --no-clamp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_audio(path: Path) -> np.ndarray:
	"""Return float32 array, shape as stored (caller normalizes to [2, samples])."""
	suffix = path.suffix.lower()
	if suffix == ".npy":
		arr = np.load(path)
	elif suffix in (".wav", ".flac", ".ogg", ".aiff", ".aif"):
		import soundfile as sf

		arr, sample_rate = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
		print(f"[compare]   {path.name}: soundfile sample_rate={sample_rate}")
	else:
		raise ValueError(f"unsupported file type: {path}")
	return np.asarray(arr, dtype=np.float32)


def to_channels_first_stereo(arr: np.ndarray, label: str) -> np.ndarray:
	"""Normalize to shape [2, samples]."""
	if arr.ndim == 1:
		arr = np.stack([arr, arr], axis=0)
	elif arr.ndim == 2:
		if arr.shape[0] == 2:
			pass  # already [2, samples]
		elif arr.shape[1] == 2:
			arr = arr.T  # [samples, 2] -> [2, samples]
		elif arr.shape[0] == 1:
			arr = np.repeat(arr, 2, axis=0)
		elif arr.shape[1] == 1:
			arr = np.repeat(arr.T, 2, axis=0)
		else:
			raise ValueError(f"{label}: cannot interpret shape {arr.shape} as stereo/mono audio")
	else:
		raise ValueError(f"{label}: expected 1-D or 2-D array, got ndim={arr.ndim} shape={arr.shape}")
	return arr


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("a", help="reference file (.npy/.wav/.flac)")
	ap.add_argument("b", help="candidate file (.npy/.wav/.flac)")
	ap.add_argument("--min-snr", type=float, default=None, help="minimum SNR in dB to pass (exit 0); omit to always exit 0")
	ap.add_argument("--no-clamp", action="store_true", help="compare raw values instead of clamping both inputs to [-1, 1] first (default: clamp)")
	args = ap.parse_args()

	path_a, path_b = Path(args.a), Path(args.b)
	print(f"[compare] a = {path_a}")
	print(f"[compare] b = {path_b}")

	raw_a = load_audio(path_a)
	raw_b = load_audio(path_b)
	a = to_channels_first_stereo(raw_a, "a")
	b = to_channels_first_stereo(raw_b, "b")
	print(f"[compare] a: raw shape={raw_a.shape} -> aligned {a.shape}")
	print(f"[compare] b: raw shape={raw_b.shape} -> aligned {b.shape}")

	if a.shape[1] != b.shape[1]:
		n = min(a.shape[1], b.shape[1])
		print(f"[compare] length mismatch ({a.shape[1]} vs {b.shape[1]}); comparing first {n} samples")
		a, b = a[:, :n], b[:, :n]

	over_a = int(np.count_nonzero(np.abs(a) > 1.0))
	over_b = int(np.count_nonzero(np.abs(b) > 1.0))
	if over_a or over_b:
		print(f"[compare] samples with |value| > 1.0: a={over_a} (peak {np.max(np.abs(a)):.4f}), b={over_b} (peak {np.max(np.abs(b)):.4f})")

	if args.no_clamp:
		print("[compare] --no-clamp: comparing raw (unclamped) values")
	else:
		a = np.clip(a, -1.0, 1.0)
		b = np.clip(b, -1.0, 1.0)
		print("[compare] clamping both inputs to [-1, 1] before scoring (pass --no-clamp to disable)")

	diff = a.astype(np.float64) - b.astype(np.float64)
	max_abs_err = float(np.max(np.abs(diff)))
	rms_err = float(np.sqrt(np.mean(diff ** 2)))

	signal_power = float(np.mean(a.astype(np.float64) ** 2))
	noise_power = float(np.mean(diff ** 2))
	if noise_power == 0.0:
		snr_db = float("inf")
	elif signal_power == 0.0:
		snr_db = float("-inf")
	else:
		snr_db = 10.0 * np.log10(signal_power / noise_power)

	print(f"[compare] max abs error: {max_abs_err:.6e}")
	print(f"[compare] RMS error:     {rms_err:.6e}")
	print(f"[compare] SNR:           {snr_db:.2f} dB")

	if args.min_snr is None:
		return 0
	passed = snr_db >= args.min_snr
	print(f"[compare] threshold --min-snr {args.min_snr:.2f} dB: {'PASS' if passed else 'FAIL'}")
	return 0 if passed else 1


if __name__ == "__main__":
	sys.exit(main())
