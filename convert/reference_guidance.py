#!/usr/bin/env python3
"""SPEC_GUIDANCE.md §5.3: torch CPU float32 reference outputs for the guided
semantic phase. CPU ONLY -- never the GPU (this box's ROCm torch build can
report CUDA/HIP as "available"; we never touch it, and assert the model stays
on cpu after load, regardless of what's available).

Runs one `cfg_scale` step of the reference arithmetic:

	logits = unconditional + cfg_scale * (conditional - unconditional)

with `protocol.token_prefixes` as the positive branch and
`protocol.negative_prefix` as the negative one, then 32 greedy semantic steps
continued from it (temperature 0 path of `sampling.distribution`, phase
"semantic", min_tokens = 32 so MUSIC_END stays masked throughout). The request
must carry its score in "abc" (or use cot=off): the abc phase is never guided
and writing a score here would be a second, unrelated decode.

Saves, into --out (default tests/out/guidance):
	ref_prefix_ids.npy        int32   [len(prefix)]     the positive prefix
	ref_negative_ids.npy      int32   [len(negative)]   the blank prefix
	ref_logits_primary.npy    float32 [184704]
	ref_logits_blank.npy      float32 [184704]
	ref_logits_blended.npy    float32 [184704]
	ref_greedy_32.npy         int32   [32]              codec ids, not codes
	ref_guidance_meta.json    the request, cfg_scale, shapes and SHA-256s

Usage:
	nice -n 10 venv_yue2/bin/python convert/reference_guidance.py \\
	    --request tests/out/guidance/ext_cfg3.json

`--src` may be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot(). Run with the venv_yue2 interpreter
directly -- no sys.path hacks needed, the `yue2` package is already on that
venv's site-packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import resolve_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REPO_ID = "m-a-p/YuE2-3B"

N_GREEDY = 32


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	h.update(path.read_bytes())
	return h.hexdigest()


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help=f"HF snapshot dir (default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--request", required=True, help="a request.json carrying \"abc\" (or cot=off)")
	ap.add_argument("--out", default=str(ROOT / "tests" / "out" / "guidance"))
	ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (keep <= half the cores)")
	args = ap.parse_args()

	from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
	from yue2.tokenization_yue2 import YuE2TextTokenizer
	from yue2 import protocol
	from yue2.protocol import SongRequest, Sampling, CODEC_OFFSET
	from yue2.sampling import distribution

	torch.set_num_threads(args.threads)
	hf_snapshot = resolve_snapshot(args.src, REPO_ID)
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)

	req_json = json.loads(Path(args.request).read_text())
	request = SongRequest(**{k: v for k, v in req_json.items()
	                         if k in {"style", "lyrics", "cot", "seed", "id", "abc", "cfg_scale"}})
	cfg_scale = request.guidance
	if cfg_scale == 1:
		print("[reference_guidance] the request is unguided; give it a cfg_scale", file=sys.stderr)
		return 1
	if request.cot != "off" and request.abc is None:
		print("[reference_guidance] the request needs its score in \"abc\"", file=sys.stderr)
		return 1

	device = torch.device("cpu")
	print(f"[reference_guidance] loading model from {hf_snapshot} (float32, CPU)")
	t0 = time.perf_counter()
	model = YuE2ForCausalLM.from_pretrained(
		str(hf_snapshot), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
	).eval()
	model.to(device)
	assert next(model.parameters()).device.type == "cpu", "refusing to run off CPU"
	load_seconds = time.perf_counter() - t0
	print(f"[reference_guidance] load: {load_seconds:.2f}s")

	tokenizer = YuE2TextTokenizer(hf_snapshot / "qwen.tiktoken")
	abc_ids = None if request.cot == "off" else tokenizer.encode(request.abc)
	prefix = protocol.token_prefixes(request, tokenizer, abc_ids=abc_ids)
	negative = protocol.negative_prefix(request, tokenizer, abc_ids=abc_ids)
	print(f"[reference_guidance] cfg_scale {cfg_scale}, positive {len(prefix)} tokens, "
	      f"negative {len(negative)} tokens")
	np.save(out_dir / "ref_prefix_ids.npy", np.asarray(prefix, dtype=np.int32))
	np.save(out_dir / "ref_negative_ids.npy", np.asarray(negative, dtype=np.int32))

	config = model.config

	def new_cache(length):
		return StaticKVCache(
			num_layers=config.num_hidden_layers, batch_size=1, num_kv_heads=config.num_key_value_heads,
			max_seq_len=length + N_GREEDY, head_dim=config.head_dim, dtype=torch.float32, device=device,
		)

	positive_cache = new_cache(len(prefix))
	negative_cache = new_cache(len(negative))

	t0 = time.perf_counter()
	with torch.inference_mode():
		conditional = model(torch.tensor([prefix], device=device), past_key_values=positive_cache,
		                    use_cache=True, logits_to_keep=1).logits[:, -1, :]
		unconditional = model(torch.tensor([negative], device=device), past_key_values=negative_cache,
		                      use_cache=True, logits_to_keep=1).logits[:, -1, :]
	prefill_seconds = time.perf_counter() - t0
	print(f"[reference_guidance] prefill: {prefill_seconds:.2f}s")

	# sampling.generate_tokens: the blend, then the unchanged sampler.
	blended = unconditional + cfg_scale * (conditional - unconditional)
	np.save(out_dir / "ref_logits_primary.npy", conditional[0].float().numpy().astype(np.float32))
	np.save(out_dir / "ref_logits_blank.npy", unconditional[0].float().numpy().astype(np.float32))
	np.save(out_dir / "ref_logits_blended.npy", blended[0].float().numpy().astype(np.float32))
	print(f"[reference_guidance] argmax: primary {int(conditional.argmax())}, "
	      f"blank {int(unconditional.argmax())}, blended {int(blended.argmax())}")

	# 32 greedy semantic steps, min_tokens = 32 so MUSIC_END stays masked.
	sampling = Sampling(temperature=0.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
	                    penalty_window=50, min_tokens=N_GREEDY, max_tokens=N_GREEDY)
	history = []
	t0 = time.perf_counter()
	with torch.inference_mode():
		for step in range(N_GREEDY):
			logits = unconditional + cfg_scale * (conditional - unconditional)
			scores = distribution(logits, sampling, history, step, "semantic",
			                      legacy_off=request.cot == "off")
			next_id = scores.argmax(-1, keepdim=True)
			history.append(int(next_id.item()))
			if step + 1 < N_GREEDY:
				conditional = model(next_id, past_key_values=positive_cache, use_cache=True,
				                    logits_to_keep=1).logits[:, -1, :]
				unconditional = model(next_id, past_key_values=negative_cache, use_cache=True,
				                      logits_to_keep=1).logits[:, -1, :]
	greedy_seconds = time.perf_counter() - t0
	print(f"[reference_guidance] {N_GREEDY} greedy steps: {greedy_seconds:.2f}s")
	print(f"[reference_guidance] codes: {[t - CODEC_OFFSET for t in history]}")
	np.save(out_dir / "ref_greedy_32.npy", np.asarray(history, dtype=np.int32))

	names = ["ref_prefix_ids.npy", "ref_negative_ids.npy", "ref_logits_primary.npy",
	         "ref_logits_blank.npy", "ref_logits_blended.npy", "ref_greedy_32.npy"]
	meta = {
		"request": request.to_dict(),
		"cfg_scale": cfg_scale,
		"blend": "unconditional + cfg_scale * (conditional - unconditional)",
		"dtype": "float32", "device": "cpu",
		"prefix_tokens": len(prefix), "negative_tokens": len(negative),
		"greedy_steps": N_GREEDY,
		"seconds": {"load": load_seconds, "prefill": prefill_seconds, "greedy": greedy_seconds},
		"sha256": {name: sha256_file(out_dir / name) for name in names},
	}
	(out_dir / "ref_guidance_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
	print(f"[reference_guidance] wrote {out_dir}/ref_guidance_meta.json")
	return 0


if __name__ == "__main__":
	sys.exit(main())
