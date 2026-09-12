#!/usr/bin/env python3
"""SPEC_NAR.md section 3 "Goldens": torch CPU float32 reference outputs for
the YuE2-3B NAR (acoustic flow matching) path. CPU ONLY -- never the GPU.

Drives the REAL code path -- yue2.nar.song_chunks() -> yue2.nar.CachedNAR ->
.solve(steps) -- with forward hooks / a thin call-counting wrapper around
CachedNAR.velocity to capture intermediates. Never reimplements the math (a
golden that shares a bug with the C++ would be worthless).

Three modes:
	short      (default) SPEC_NAR.md 3.1/3.2: alley_swing_s1[:256]/[:128]
	           prefix/codec, steps=2 and steps=32. -> tests/golden/nar_*.npy
	           + tests/golden/nar_short_meta.json
	multichunk SPEC_NAR.md 3.5: same short prefix, codec[:512], context=1200,
	           steps=2 -> tests/out/nar_multichunk_{cpu,noise}.npy. Cheap,
	           mandatory (only test that exercises chunk_ranges).
	full       SPEC_NAR.md 3.4 (optional, slow: ~40-120 min): the whole
	           alley_swing_s1 chunk, steps=32 -> tests/out/nar_full_cpu_latent.npy
	           + tests/out/nar_noise_alley_swing_s1.npy. Meant to be run with
	           nice, in the background; not on the critical path.

Usage:
	venv_yue2/bin/python reference_nar.py --mode short
	venv_yue2/bin/python reference_nar.py --mode multichunk
	nice venv_yue2/bin/python reference_nar.py --mode full &

`--src` may be omitted if `m-a-p/YuE2-3B` is in the huggingface_hub cache.
`--songs-dir` defaults to the read-only alley_swing_s1 artifacts dir; never
written to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must happen before `import torch` -- a wedged HIP run takes the whole box down.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from common import resolve_snapshot, sha256_of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
OUT_DIR = ROOT / "tests" / "out"

REPO_ID = "m-a-p/YuE2-3B"

MUSIC_START = 151851


def load_model(src: Path):
	from yue2.modeling_yue2 import YuE2ForCausalLM

	device = torch.device("cpu")
	print(f"[reference_nar] loading model from {src} (float32, CPU)")
	t0 = time.perf_counter()
	model = YuE2ForCausalLM.from_pretrained(
		str(src), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
	).eval()
	model.to(device)
	assert next(model.parameters()).device.type == "cpu", "refusing to run off CPU"
	# Saves ~1.5 GB of the ~14.5 GB f32 footprint; lm_head is never used by the NAR path.
	del model.lm_head
	load_seconds = time.perf_counter() - t0
	print(f"[reference_nar] load: {load_seconds:.2f}s (lm_head dropped)")
	return model, load_seconds


def traced_solve(model, chunk, steps):
	"""Run CachedNAR(model, chunk).solve(steps) with hooks capturing the
	SPEC_NAR.md 3.2 intermediates from the FIRST velocity() call only, plus
	the full sequence of `raw_t` values fed to every call. Returns
	(latent, capture_dict, engine) -- caller must engine.close().
	"""
	from yue2 import nar

	engine = nar.CachedNAR(model, chunk)

	capture: dict = {"tshift": []}
	call_count = [0]
	capturing = {"active": False}

	def pre_hook_in(module, inputs):
		if capturing["active"] and "x_in" not in capture:
			capture["x_in"] = inputs[0].detach().clone()

	def pre_hook_l0(module, inputs):
		if capturing["active"] and "x_l0" not in capture:
			capture["x_l0"] = inputs[0].detach().clone()

	handles = [
		engine.model.model.layers[0].nar_input_layernorm.register_forward_pre_hook(pre_hook_in),
		engine.model.model.layers[1].nar_input_layernorm.register_forward_pre_hook(pre_hook_l0),
	]

	original_velocity = engine.velocity

	def traced_velocity(state, raw_t):
		call_count[0] += 1
		first = call_count[0] == 1
		capture["tshift"].append(raw_t)
		if first:
			capturing["active"] = True
		out = original_velocity(state, raw_t)
		if first:
			capture["v_step0"] = out.detach().clone()
			capturing["active"] = False
		return out

	engine.velocity = traced_velocity
	try:
		latent = engine.solve(steps)
	finally:
		for h in handles:
			h.remove()

	return latent, capture, engine


def run_short(model, songs_dir: Path, seed: int) -> None:
	from yue2 import nar

	GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

	prefix_full = np.load(songs_dir / "prefix.npy")
	semantic_full = np.load(songs_dir / "semantic.npy")
	prefix_short = np.concatenate([prefix_full[:255], np.array([MUSIC_START], dtype=prefix_full.dtype)])
	codec_short = semantic_full[:128]
	assert prefix_short[-1] == MUSIC_START
	print(f"[reference_nar/short] prefix_short={len(prefix_short)} codec_short={len(codec_short)} seed={seed}")

	np.save(GOLDEN_DIR / "nar_prefix_ids.npy", prefix_short.astype(np.int32))
	np.save(GOLDEN_DIR / "nar_codec_ids.npy", codec_short.astype(np.int32))

	chunks = nar.song_chunks(prefix_short.tolist(), codec_short.tolist(), seed, context=24576)
	assert len(chunks) == 1, f"expected 1 chunk for the short test, got {len(chunks)}"
	chunk = chunks[0]
	ar_length, nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
	print(f"[reference_nar/short] ar_length={ar_length} nar_length={nar_length} "
	      f"S={ar_length + nar_length}")
	np.save(GOLDEN_DIR / "nar_noise.npy", chunk.noise.numpy().astype(np.float32))

	# steps=2 first (cheap, untraced -- no need to re-run the prefill for it).
	t0 = time.perf_counter()
	latent_s2, _capture2, engine = traced_solve(model, chunk, steps=2)
	s2_seconds = time.perf_counter() - t0
	np.save(GOLDEN_DIR / "nar_latent_s2.npy", latent_s2.numpy().astype(np.float32))
	print(f"[reference_nar/short] steps=2 solve: {s2_seconds:.2f}s")

	ar_kv_l0 = torch.stack(list(engine.cache[0])).numpy().astype(np.float32)
	ar_kv_l27 = torch.stack(list(engine.cache[27])).numpy().astype(np.float32)
	np.save(GOLDEN_DIR / "nar_ar_kv_l0.npy", ar_kv_l0)
	np.save(GOLDEN_DIR / "nar_ar_kv_l27.npy", ar_kv_l27)
	print(f"[reference_nar/short] ar_kv_l0 shape={ar_kv_l0.shape} ar_kv_l27 shape={ar_kv_l27.shape}")

	# steps=32 on the SAME engine (same AR-prefill KV, chunk.noise untouched by solve()).
	t0 = time.perf_counter()
	latent_s32, capture32, _engine = traced_solve(model, chunk, steps=32)
	s32_seconds = time.perf_counter() - t0
	np.save(GOLDEN_DIR / "nar_latent_s32.npy", latent_s32.numpy().astype(np.float32))
	print(f"[reference_nar/short] steps=32 solve: {s32_seconds:.2f}s")

	x_in = capture32["x_in"].squeeze(0).numpy().astype(np.float32)
	x_l0 = capture32["x_l0"].squeeze(0).numpy().astype(np.float32)
	v_step0 = capture32["v_step0"].numpy().astype(np.float32)
	tshift = np.asarray(capture32["tshift"], dtype=np.float64)
	assert x_in.shape == (nar_length, 2048), x_in.shape
	assert x_l0.shape == (nar_length, 2048), x_l0.shape
	assert v_step0.shape == (128, 64), v_step0.shape
	assert tshift.shape == (64,), tshift.shape
	np.save(GOLDEN_DIR / "nar_x_in_step0.npy", x_in)
	np.save(GOLDEN_DIR / "nar_x_l0_step0.npy", x_l0)
	np.save(GOLDEN_DIR / "nar_v_step0.npy", v_step0)
	np.save(GOLDEN_DIR / "nar_tshift.npy", tshift)
	print(f"[reference_nar/short] x_in={x_in.shape} x_l0={x_l0.shape} v_step0={v_step0.shape} "
	      f"tshift[0:3]={tshift[:3]}")

	engine.close()

	return dict(
		prefix_len=len(prefix_short), codec_len=len(codec_short),
		ar_length=ar_length, nar_length=nar_length, seed=seed,
		steps=32, torch_version=torch.__version__,
		wall_seconds=dict(load=None, s2_solve=s2_seconds, s32_solve=s32_seconds),
	)


def run_multichunk(model, songs_dir: Path, seed: int) -> None:
	from yue2 import nar

	OUT_DIR.mkdir(parents=True, exist_ok=True)
	prefix_full = np.load(songs_dir / "prefix.npy")
	semantic_full = np.load(songs_dir / "semantic.npy")
	prefix_short = np.concatenate([prefix_full[:255], np.array([MUSIC_START], dtype=prefix_full.dtype)])
	codec_512 = semantic_full[:512]

	t0 = time.perf_counter()
	chunks = nar.song_chunks(prefix_short.tolist(), codec_512.tolist(), seed, context=1200)
	ranges = [(sum(len(c.noise) for c in chunks[:i]),
	           sum(len(c.noise) for c in chunks[:i]) + len(chunks[i].noise)) for i in range(len(chunks))]
	print(f"[reference_nar/multichunk] {len(chunks)} chunks, ranges={ranges}")
	assert ranges == [(0, 470), (470, 512)], f"unexpected chunking: {ranges}"

	noise_full = torch.cat([c.noise for c in chunks], dim=0)
	np.save(OUT_DIR / "nar_multichunk_noise.npy", noise_full.numpy().astype(np.float32))

	outputs = []
	ar_lengths = []
	for chunk in chunks:
		engine = nar.CachedNAR(model, chunk)
		ar_lengths.append(engine.ar_length)
		try:
			outputs.append(engine.solve(steps=2))
		finally:
			engine.close()
	latent = torch.cat(outputs, dim=0)
	seconds = time.perf_counter() - t0
	assert tuple(latent.shape) == (512, 64), latent.shape
	np.save(OUT_DIR / "nar_multichunk_cpu.npy", latent.numpy().astype(np.float32))
	print(f"[reference_nar/multichunk] {seconds:.2f}s, ar_lengths={ar_lengths}, output shape={latent.shape}")
	return dict(chunk_ranges=ranges, ar_lengths=ar_lengths, context=1200, steps=2, seed=seed)


def run_full(model, songs_dir: Path, seed: int) -> None:
	from yue2 import nar

	OUT_DIR.mkdir(parents=True, exist_ok=True)
	prefix = np.load(songs_dir / "prefix.npy").tolist()
	semantic = np.load(songs_dir / "semantic.npy").tolist()
	print(f"[reference_nar/full] P={len(prefix)} T={len(semantic)} seed={seed} steps=32 -- this is slow "
	      f"(estimate 40-120 min), see SPEC_NAR.md 3.4")

	t0 = time.perf_counter()
	chunks = nar.song_chunks(prefix, semantic, seed, context=24576)
	assert len(chunks) == 1, f"alley_swing_s1 is expected to be a single chunk, got {len(chunks)}"
	chunk = chunks[0]
	np.save(OUT_DIR / "nar_noise_alley_swing_s1.npy", chunk.noise.numpy().astype(np.float32))

	engine = nar.CachedNAR(model, chunk)
	try:
		latent = engine.solve(steps=32)
	finally:
		engine.close()
	seconds = time.perf_counter() - t0
	assert tuple(latent.shape) == (len(semantic), 64), latent.shape
	np.save(OUT_DIR / "nar_full_cpu_latent.npy", latent.numpy().astype(np.float32))
	print(f"[reference_nar/full] done in {seconds:.2f}s ({seconds / 60:.1f} min), shape={latent.shape}")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help=f"HF snapshot dir (default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--songs-dir", required=True,
	                 help="read-only artifacts dir with prefix.npy/semantic.npy/request.json (a yue2 song --artifacts dir)")
	ap.add_argument("--mode", choices=["short", "multichunk", "full"], default="short")
	args = ap.parse_args()

	hf_snapshot = resolve_snapshot(args.src, REPO_ID)
	songs_dir = Path(args.songs_dir)
	req = json.loads((songs_dir / "request.json").read_text())
	seed = req["seed"]

	model, load_seconds = load_model(hf_snapshot)

	t_total0 = time.perf_counter()
	if args.mode == "short":
		meta = run_short(model, songs_dir, seed)
		meta["source_sha256"] = sha256_of(hf_snapshot / "model.safetensors")
		meta["wall_seconds"]["load"] = load_seconds
		meta["wall_seconds"]["total"] = time.perf_counter() - t_total0 + load_seconds
		GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
		(GOLDEN_DIR / "nar_short_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
		print(f"[reference_nar] wrote {GOLDEN_DIR / 'nar_short_meta.json'}")
	elif args.mode == "multichunk":
		meta = run_multichunk(model, songs_dir, seed)
		(OUT_DIR / "nar_multichunk_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
		# Also fold the chunk ranges into the short meta file so the C++ agent
		# can check its chunking without rerunning torch (SPEC_NAR.md 3.5).
		short_meta_path = GOLDEN_DIR / "nar_short_meta.json"
		if short_meta_path.exists():
			short_meta = json.loads(short_meta_path.read_text())
			short_meta["multichunk"] = meta
			short_meta_path.write_text(json.dumps(short_meta, indent=2) + "\n")
			print(f"[reference_nar] folded multichunk meta into {short_meta_path}")
	else:
		run_full(model, songs_dir, seed)

	print(f"[reference_nar] mode={args.mode} load={load_seconds:.2f}s "
	      f"total={time.perf_counter() - t_total0 + load_seconds:.2f}s")
	return 0


if __name__ == "__main__":
	sys.exit(main())
