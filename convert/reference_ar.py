#!/usr/bin/env python3
"""SPEC_AR.md section 2 "Goldens": torch CPU float32 reference outputs for the
YuE2-3B AR path. CPU ONLY -- never the GPU (this box's ROCm torch build can
report CUDA/HIP as "available"; we never touch it, and assert the model
stays on cpu after load, regardless of what's available).

Loads YuE2ForCausalLM on CPU in float32, builds the alley_swing_s1 prefix
via `protocol.token_prefixes(request, tokenizer)` for cot="full" with
abc_ids=None (the ABC-phase prefix: [EOD] + text + [ABC_START]), runs one
prefill forward pass with a StaticKVCache, and continues 32 greedy steps
(temperature 0 path of sampling.distribution, ABC-phase allowed mask,
min_tokens=32 so ABC_END stays masked for all 32 steps). Saves:
	tests/golden/ar_prefix_ids.npy         int32  [len(prefix)]
	tests/golden/ar_last_logits_f32.npy    float32 [184704]
	tests/golden/ar_greedy_32.npy          int32  [32]

Usage:
	venv_yue2/bin/python reference_ar.py --songs-dir /path/to/songs/ref_song/out/alley_swing_s1

`--src` may be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot(). `--songs-dir` is required (points
at a request.json under some songs/ tree, which is user-specific input).
Run with the venv_yue2 interpreter directly -- no sys.path hacks needed, the
`yue2` package is already on that venv's site-packages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import resolve_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN_DIR = ROOT / "tests" / "golden"

REPO_ID = "m-a-p/YuE2-3B"

N_GREEDY = 32


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help=f"HF snapshot dir (default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--songs-dir", required=True, help="dir with request.json (a songs/.../out/<name> dir)")
	args = ap.parse_args()

	from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
	from yue2.tokenization_yue2 import YuE2TextTokenizer
	from yue2 import protocol
	from yue2.protocol import SongRequest, Sampling
	from yue2.sampling import distribution

	hf_snapshot = resolve_snapshot(args.src, REPO_ID)
	songs_dir = Path(args.songs_dir)
	GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

	device = torch.device("cpu")
	print(f"[reference_ar] loading model from {hf_snapshot} (float32, CPU)")
	t0 = time.perf_counter()
	model = YuE2ForCausalLM.from_pretrained(
		str(hf_snapshot), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
	).eval()
	model.to(device)
	assert next(model.parameters()).device.type == "cpu", "refusing to run off CPU"
	load_seconds = time.perf_counter() - t0
	print(f"[reference_ar] load: {load_seconds:.2f}s")

	tokenizer = YuE2TextTokenizer(hf_snapshot / "qwen.tiktoken")
	req_json = json.loads((songs_dir / "request.json").read_text())
	request = SongRequest(style=req_json["style"], lyrics=req_json["lyrics"], cot="full")

	prefix = protocol.token_prefixes(request, tokenizer, abc_ids=None)
	print(f"[reference_ar] prefix: {len(prefix)} tokens (cot=full, ABC-phase: [EOD] + text + [ABC_START])")
	np.save(GOLDEN_DIR / "ar_prefix_ids.npy", np.asarray(prefix, dtype=np.int32))

	config = model.config
	cache = StaticKVCache(
		num_layers=config.num_hidden_layers, batch_size=1, num_kv_heads=config.num_key_value_heads,
		max_seq_len=len(prefix) + N_GREEDY, head_dim=config.head_dim, dtype=torch.float32, device=device,
	)

	t0 = time.perf_counter()
	with torch.inference_mode():
		out = model(torch.tensor([prefix], device=device), past_key_values=cache, use_cache=True, logits_to_keep=1)
	prefill_seconds = time.perf_counter() - t0
	last_logits = out.logits[0, -1, :].float().numpy()
	print(f"[reference_ar] prefill forward: {prefill_seconds:.2f}s, logits shape {last_logits.shape}, "
	      f"argmax {int(last_logits.argmax())}")
	np.save(GOLDEN_DIR / "ar_last_logits_f32.npy", last_logits.astype(np.float32))

	# 32 greedy steps, ABC-phase mask, min_tokens=32 -> ABC_END stays masked throughout.
	sampling = Sampling(temperature=0.0, top_p=0.9, top_k=30, repetition_penalty=1.005,
	                     penalty_window=100, min_tokens=N_GREEDY, max_tokens=N_GREEDY)
	history = []
	logits = out.logits[:, -1, :]
	t0 = time.perf_counter()
	with torch.inference_mode():
		for step in range(N_GREEDY):
			scores = distribution(logits, sampling, history, step, "abc")
			next_id = scores.argmax(-1, keepdim=True)
			token = int(next_id.item())
			history.append(token)
			if step + 1 < N_GREEDY:
				logits = model(next_id, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]
	greedy_seconds = time.perf_counter() - t0
	print(f"[reference_ar] {N_GREEDY} greedy steps: {greedy_seconds:.2f}s "
	      f"({N_GREEDY / greedy_seconds:.2f} tok/s)")
	print(f"[reference_ar] greedy tokens: {history}")
	np.save(GOLDEN_DIR / "ar_greedy_32.npy", np.asarray(history, dtype=np.int32))

	print(f"\n[reference_ar] wall times: load={load_seconds:.2f}s prefill={prefill_seconds:.2f}s "
	      f"greedy32={greedy_seconds:.2f}s total={load_seconds + prefill_seconds + greedy_seconds:.2f}s")
	return 0


if __name__ == "__main__":
	sys.exit(main())
