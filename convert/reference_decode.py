#!/usr/bin/env python3
"""Torch (CPU-only) reference decode -- golden files for the ggml decoder.

NEVER touches the GPU/ROCm (device is hardcoded "cpu"; see SPEC.md section 6 --
torch-on-GPU has taken the box down before). Loads the released decoder via
YuE2VAE.from_pretrained(..., decoder_only=True, device="cpu", local_files_only=True)
and decodes the first N frames of a latent .npy UNTILED (model.decode), matching
SPEC.md section 5's golden-file contract.

Usage:
	venv_yue2/bin/python reference_decode.py \
		--latent /path/to/songs/ref_song/out/alley_swing_s1/latent.npy \
		--frames 8 48 \
		--name alley_swing_s1

`--src` may be omitted if `m-a-p/YuE2-Vae` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot(). `--latent` is required (this
script has no business knowing where any particular user's songs/ tree is).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import load_module_from_snapshot, resolve_snapshot

REPO_ID = "m-a-p/YuE2-Vae"
DEFAULT_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden"


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help="HF snapshot dir (config.json, model.safetensors, modeling_vae.py) "
	                      f"(default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--latent", required=True, help="latent .npy, shape [T, 64] (read-only input)")
	ap.add_argument("--frames", type=int, nargs="+", default=[8, 48], help="frame counts N to decode (first N frames, untiled)")
	ap.add_argument("--name", default="alley_swing_s1", help="golden filename stem")
	ap.add_argument("--out-dir", default=str(DEFAULT_GOLDEN_DIR), help="output dir for golden .npy files")
	args = ap.parse_args()

	src = resolve_snapshot(args.src, REPO_ID)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"[reference_decode] loading decoder from {src} (device=cpu, decoder_only=True)")
	modeling_vae = load_module_from_snapshot(src, "modeling_vae.py", "yue2_modeling_vae")
	t_load0 = time.time()
	model = modeling_vae.YuE2VAE.from_pretrained(
		str(src), decoder_only=True, device="cpu", local_files_only=True
	)
	t_load = time.time() - t_load0
	print(f"[reference_decode] load: {t_load:.2f}s")
	assert next(model.decoder.parameters()).device.type == "cpu", "decoder must stay on CPU"

	latent_np = np.load(args.latent)  # [T, 64], read-only source under songs/ -- never write there
	print(f"[reference_decode] latent {args.latent}: shape={latent_np.shape} dtype={latent_np.dtype}")
	assert latent_np.ndim == 2 and latent_np.shape[1] == model.config.latent_dim, (
		f"expected [T, {model.config.latent_dim}], got {latent_np.shape}"
	)
	latent_full = torch.from_numpy(latent_np).to(torch.float32).t().unsqueeze(0)  # [1, 64, T]

	for n in args.frames:
		if n > latent_np.shape[0]:
			print(f"[reference_decode] skip N={n}: only {latent_np.shape[0]} frames available", file=sys.stderr)
			continue
		latent_n = latent_full[..., :n]
		t0 = time.time()
		audio = model.decode(latent_n)  # untiled, full graph, SPEC.md section 5
		wall = time.time() - t0

		expected_len = 1920 * n - 64
		got = audio.shape
		assert got == (1, 2, expected_len), f"N={n}: expected [1,2,{expected_len}], got {list(got)}"

		out = audio.squeeze(0).to(torch.float32).numpy()  # [2, samples]
		out_path = out_dir / f"{args.name}_f{n}.npy"
		np.save(out_path, out)
		print(f"[reference_decode] N={n}: decode wall={wall:.2f}s -> {out.shape} "
		      f"(== 1920*{n}-64 = {expected_len})  saved {out_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
