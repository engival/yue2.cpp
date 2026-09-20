#!/usr/bin/env python3
"""safetensors -> GGUF converter for the YuE2-3B NAR (acoustic flow matching)
subset. See ../SPEC_NAR.md section 2 for the contract this implements.

Only the NAR-half tensors are ported here -- nar_input_layernorm,
nar_self_attn.*, nar_pre_mlp_layernorm, nar_mlp.*, vae2llm, llm2vae,
time_embedder.mlp.{0,2}, latent_pos_embed.pe. The AR half + token_embd +
output_norm live in yue2-ar-*.gguf (convert_ar.py) and are reused unchanged
by src/yue2-nar.cpp; nothing here duplicates them.

Run with the venv_yue2 interpreter (torch CPU + safetensors + numpy + gguf):
	venv_yue2/bin/python convert_nar.py \
		--src /path/to/hf/snapshot/models--m-a-p--YuE2-3B/snapshots/<rev> \
		--out tests/out/yue2-nar-f16.gguf \
		--ar-gguf tests/out/yue2-ar-f16.gguf \
		[--type f16]

`--src` may be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot(). `--ar-gguf` is used only to read
back its `yue2.source_sha256` so this script can assert the two files will
agree (yue2-nar.cpp aborts at load time if they don't); if the AR gguf does
not exist yet this check is skipped with a warning.

`--lora LORA.safetensors[:STRENGTH]` (repeatable) bakes NAR adapters into the
output, with the same merge rule `yue2 --nar-lora` applies at load: the deltas
are summed in F32 against the checkpoint weight and the result is narrowed once.
See ../SPEC_LORA.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from common import print_tensor_table, resolve_snapshot, sha256_of

REPO_ID = "m-a-p/YuE2-3B"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "tests" / "out" / "yue2-nar-f16.gguf")
DEFAULT_AR_GGUF = str(Path(__file__).resolve().parent.parent / "tests" / "out" / "yue2-ar-f16.gguf")

HIDDEN = 2048
MAX_LATENT_FRAMES = 24576


def layer_tensor_map(n_layers: int) -> dict:
	m = {}
	for n in range(n_layers):
		p = f"model.layers.{n}."
		g = f"blk.{n}."
		m[p + "nar_input_layernorm.weight"] = (g + "nar_attn_norm.weight", "norm")
		m[p + "nar_self_attn.q_proj.weight"] = (g + "nar_attn_q.weight", "2d")
		m[p + "nar_self_attn.k_proj.weight"] = (g + "nar_attn_k.weight", "2d")
		m[p + "nar_self_attn.v_proj.weight"] = (g + "nar_attn_v.weight", "2d")
		m[p + "nar_self_attn.o_proj.weight"] = (g + "nar_attn_output.weight", "2d")
		m[p + "nar_self_attn.q_norm.weight"] = (g + "nar_attn_q_norm.weight", "norm")
		m[p + "nar_self_attn.k_norm.weight"] = (g + "nar_attn_k_norm.weight", "norm")
		m[p + "nar_pre_mlp_layernorm.weight"] = (g + "nar_ffn_norm.weight", "norm")
		m[p + "nar_mlp.gate_proj.weight"] = (g + "nar_ffn_gate.weight", "2d")
		m[p + "nar_mlp.up_proj.weight"] = (g + "nar_ffn_up.weight", "2d")
		m[p + "nar_mlp.down_proj.weight"] = (g + "nar_ffn_down.weight", "2d")
	# NAR-only auxiliaries -- all F32 except latent_pos_embed.pe (F16, see below).
	m["vae2llm.weight"] = ("nar.vae2llm.weight", "norm")
	m["vae2llm.bias"] = ("nar.vae2llm.bias", "norm")
	m["llm2vae.weight"] = ("nar.llm2vae.weight", "norm")
	m["llm2vae.bias"] = ("nar.llm2vae.bias", "norm")
	m["time_embedder.mlp.0.weight"] = ("nar.time_embd.0.weight", "norm")
	m["time_embedder.mlp.0.bias"] = ("nar.time_embd.0.bias", "norm")
	m["time_embedder.mlp.2.weight"] = ("nar.time_embd.1.weight", "norm")
	m["time_embedder.mlp.2.bias"] = ("nar.time_embd.1.bias", "norm")
	m["latent_pos_embed.pe"] = ("nar.latent_pos_embd.weight", "pe")
	return m


def parse_lora_arg(text: str) -> tuple[Path, float]:
	"""SPEC_LORA section 2: split on the last ':' and only when what follows is a
	float, so a path that happens to contain a colon still works."""
	head, sep, tail = text.rpartition(":")
	if sep:
		try:
			strength = float(tail)
		except ValueError:
			pass
		else:
			if not math.isfinite(strength):
				raise SystemExit(f"--lora strength must be finite (got {text!r})")
			return Path(head), strength
	return Path(text), 1.0


def lora_plan(lora_args: list[str], n_layers: int) -> dict:
	"""{gguf tensor name: [(path, strength, lora_A, lora_B) or (path, strength,
	W_new, None)]}, in command-line order. Mirrors src/common/lora.hpp."""
	proj_map = {}
	for n in range(n_layers):
		for lora_proj, gguf_proj in (
			("nar_self_attn.q_proj", "nar_attn_q"),
			("nar_self_attn.k_proj", "nar_attn_k"),
			("nar_self_attn.v_proj", "nar_attn_v"),
			("nar_self_attn.o_proj", "nar_attn_output"),
			("nar_mlp.gate_proj", "nar_ffn_gate"),
			("nar_mlp.up_proj", "nar_ffn_up"),
			("nar_mlp.down_proj", "nar_ffn_down"),
		):
			proj_map[f"layers.{n}.{lora_proj}"] = f"blk.{n}.{gguf_proj}.weight"
	full_map = {
		"vae2llm.weight": "nar.vae2llm.weight",
		"vae2llm.bias": "nar.vae2llm.bias",
		"llm2vae.weight": "nar.llm2vae.weight",
		"llm2vae.bias": "nar.llm2vae.bias",
	}

	plan: dict = {}
	for arg in lora_args:
		path, strength = parse_lora_arg(arg)
		pairs: dict = {}
		with safe_open(path, framework="pt", device="cpu") as lf:
			for key in lf.keys():
				if key.startswith("diffusion_model."):
					raise SystemExit(f"--lora {path}: fused ComfyUI layout (key {key!r}); "
					                  f"use the plain adapter file instead")
				if key in full_map:
					plan.setdefault(full_map[key], []).append(
						(path, strength, lf.get_tensor(key).to(torch.float32).numpy(), None))
					continue
				stem, sep, part = key.rpartition(".")
				if not sep or part not in ("lora_A", "lora_B") or stem not in proj_map:
					raise SystemExit(f"--lora {path}: key {key!r} maps to no NAR tensor")
				pairs.setdefault(proj_map[stem], {})[part] = lf.get_tensor(key).to(torch.float32).numpy()
		for gguf_name, both in pairs.items():
			if "lora_A" not in both or "lora_B" not in both:
				raise SystemExit(f"--lora {path}: {gguf_name} has only "
				                  f"{sorted(both)[0]}, not both factors")
			plan.setdefault(gguf_name, []).append((path, strength, both["lora_A"], both["lora_B"]))
	return plan


def apply_lora(base: np.ndarray, ops: list) -> np.ndarray:
	"""W += s*(B @ A) for a pair, W += s*(W_new - W) for a replacement -- every
	delta against `base`, summed in F32 before the single narrowing the caller
	does (SPEC_LORA section 3.3)."""
	out = base.astype(np.float32, copy=True)
	for _, strength, a, b in ops:
		if b is None:
			if a.shape != base.shape:
				raise SystemExit(f"--lora: replacement shape {a.shape} != base {base.shape}")
			out += np.float32(strength) * (a - base)
		else:
			if a.shape[1] != base.shape[1] or b.shape[0] != base.shape[0] or a.shape[0] != b.shape[1]:
				raise SystemExit(f"--lora: factors {b.shape} x {a.shape} do not fit base {base.shape}")
			out += (np.float32(strength) * b) @ a
	return out


def reference_pe(max_frames: int, hidden_size: int) -> np.ndarray:
	"""Reimplementation of AudioPositionEmbedding.__init__ (modeling_yue2.py
	:334-346), for the sanity check against the checkpoint's stored (BF16-
	rounded) buffer -- see SPEC_NAR.md section 2.2."""
	pe = np.zeros((max_frames, hidden_size), dtype=np.float64)
	position = np.arange(0, max_frames, dtype=np.float64)[:, None]
	div_term = np.exp(np.arange(0, hidden_size, 2, dtype=np.float64) * (-math.log(10000.0) / hidden_size))
	pe[:, 0::2] = np.sin(position * div_term)
	pe[:, 1::2] = np.cos(position * div_term)
	return pe


def read_ar_sha256(ar_gguf_path: Path) -> str | None:
	if not ar_gguf_path.exists():
		return None
	import gguf
	reader = gguf.GGUFReader(str(ar_gguf_path))
	for field in reader.fields.values():
		if field.name == "yue2.source_sha256":
			return bytes(field.parts[field.data[0]]).decode("utf-8")
	return None


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help="HF snapshot dir with model.safetensors + config.json "
	                      f"(default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--out", default=DEFAULT_OUT, help="output .gguf path")
	ap.add_argument("--ar-gguf", default=DEFAULT_AR_GGUF,
	                 help="AR gguf to cross-check yue2.source_sha256 against (skipped if missing)")
	ap.add_argument("--type", choices=["f16", "f32"], default="f16",
	                 help="dtype for 2D+ weight matrices and latent_pos_embed.pe; "
	                      "1D norm/bias weights always stay F32")
	ap.add_argument("--lora", action="append", default=[], metavar="FILE[:S]",
	                 help="bake a NAR LoRA in (repeatable, applied in order); see ../SPEC_LORA.md")
	args = ap.parse_args()

	torch.set_num_threads(4)

	import gguf

	src = resolve_snapshot(args.src, REPO_ID)
	st_path = src / "model.safetensors"
	cfg_path = src / "config.json"
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	config = json.loads(cfg_path.read_text())
	n_layers = config["num_hidden_layers"]
	assert n_layers == 28
	assert config["hidden_size"] == HIDDEN
	assert config["num_attention_heads"] == 16
	assert config["num_key_value_heads"] == 8
	assert config["head_dim"] == 128
	assert config["intermediate_size"] == 6144
	assert config["rms_norm_eps"] == 1e-6
	assert config["rope_theta"] == 1000000
	assert config["max_position_embeddings"] == 24576
	assert config["latent_dim"] == 64
	assert config["max_latent_frames"] == MAX_LATENT_FRAMES
	assert config["timestep_shift"] == 1.0

	source_sha256 = sha256_of(st_path)
	print(f"[convert_nar] source: {st_path}")
	print(f"[convert_nar] source sha256: {source_sha256}")

	ar_sha256 = read_ar_sha256(Path(args.ar_gguf))
	if ar_sha256 is None:
		print(f"[convert_nar] WARNING: {args.ar_gguf} not found or has no yue2.source_sha256; "
		      f"skipping cross-check (yue2-nar.cpp will still enforce it at load time)")
	elif ar_sha256 != source_sha256:
		raise SystemExit(f"[convert_nar] source_sha256 mismatch: this run={source_sha256} "
		                  f"vs {args.ar_gguf}={ar_sha256} -- AR and NAR ggufs must come from the "
		                  f"same model.safetensors")
	else:
		print(f"[convert_nar] source_sha256 matches {args.ar_gguf}")

	writer = gguf.GGUFWriter(str(out_path), "yue2-nar")
	writer.add_name("YuE2-3B NAR")

	writer.add_uint32("yue2nar.block_count", n_layers)
	writer.add_uint32("yue2nar.embedding_length", HIDDEN)
	writer.add_uint32("yue2nar.feed_forward_length", 6144)
	writer.add_uint32("yue2nar.attention.head_count", 16)
	writer.add_uint32("yue2nar.attention.head_count_kv", 8)
	writer.add_uint32("yue2nar.attention.key_length", 128)
	writer.add_uint32("yue2nar.attention.value_length", 128)
	writer.add_float32("yue2nar.attention.layer_norm_rms_epsilon", 1e-6)
	writer.add_float32("yue2nar.rope.freq_base", 1000000.0)
	writer.add_uint32("yue2nar.context_length", 24576)
	writer.add_uint32("yue2nar.latent_dim", 64)
	writer.add_uint32("yue2nar.max_latent_frames", MAX_LATENT_FRAMES)
	writer.add_float32("yue2nar.timestep_shift", 1.0)
	writer.add_uint32("yue2nar.time_embd_frequency_size", 256)
	writer.add_uint32("yue2nar.ode_steps", 32)
	writer.add_string("yue2nar.ode_method", "midpoint")
	writer.add_key_value("yue2.source_sha256", source_sha256, gguf.GGUFValueType.STRING)

	# A baked GGUF keeps the checkpoint's source_sha256 -- it is still the same
	# model pair, and yue2-nar.cpp matches it against the AR file. The adapters
	# are recorded separately, as `yue2 song` records them in config.json.
	plan = lora_plan(args.lora, n_layers)
	if plan:
		provenance = []
		for arg in args.lora:
			path, strength = parse_lora_arg(arg)
			provenance.append({"file": path.name, "strength": strength, "sha256": sha256_of(path)})
			print(f"[convert_nar] lora: {path} strength {strength}")
		writer.add_key_value("yue2nar.lora", json.dumps(provenance), gguf.GGUFValueType.STRING)
		print(f"[convert_nar] {len(plan)} tensors take a baked delta")

	tmap = layer_tensor_map(n_layers)
	table = []
	qtype = gguf.GGMLQuantizationType.F16 if args.type == "f16" else gguf.GGMLQuantizationType.F32

	with safe_open(st_path, framework="pt", device="cpu") as f:
		keys = set(f.keys())
		missing = [k for k in tmap if k not in keys]
		if missing:
			raise ValueError(f"Missing expected tensors in safetensors: {missing[:5]}")

		# Sanity-check the stored (BF16-rounded) latent_pos_embed.pe against the
		# formula it should match, per SPEC_NAR.md 2.2, BEFORE writing anything --
		# a failure here means the interleaved sin/cos layout reading is wrong.
		stored_pe = f.get_tensor("latent_pos_embed.pe").to(torch.float32).numpy()
		expected_pe = reference_pe(MAX_LATENT_FRAMES, HIDDEN).astype(np.float32)
		max_delta = float(np.max(np.abs(stored_pe - expected_pe)))
		print(f"[convert_nar] latent_pos_embed.pe vs formula: max|delta|={max_delta:.3e} (must be < 8e-3)")
		assert max_delta < 8e-3, f"latent_pos_embed.pe does not match the sinusoid formula: max|delta|={max_delta}"

		for src_name, (gguf_name, kind) in tmap.items():
			t = f.get_tensor(src_name).to(torch.float32).numpy()
			t = np.ascontiguousarray(t)
			if gguf_name in plan:
				t = np.ascontiguousarray(apply_lora(t, plan.pop(gguf_name)))
			if kind == "norm":
				out = t.astype(np.float32)
				writer.add_tensor(gguf_name, out)
				table.append((gguf_name, list(out.shape), "F32"))
			elif kind == "pe":
				# Always stored F16 (exact for a BF16-rounded value, see 2.2),
				# regardless of --type.
				out = gguf.quants.quantize(t, gguf.GGMLQuantizationType.F16)
				writer.add_tensor(gguf_name, out, raw_dtype=gguf.GGMLQuantizationType.F16)
				table.append((gguf_name, list(t.shape), "F16"))
			else:
				out = gguf.quants.quantize(t, qtype)
				writer.add_tensor(gguf_name, out, raw_dtype=qtype)
				table.append((gguf_name, list(t.shape), qtype.name))

	if plan:
		raise SystemExit(f"--lora: {sorted(plan)[:5]} name tensors this converter does not write")

	n_expected = 11 * n_layers + 9
	assert len(tmap) == n_expected, f"expected {n_expected} tensors, mapped {len(tmap)}"
	print(f"[convert_nar] {len(tmap)} tensors mapped (of {len(keys)} in safetensors; "
	      f"AR-half + shared tensors belong to yue2-ar-*.gguf, not duplicated here)")

	writer.write_header_to_file()
	writer.write_kv_data_to_file()
	writer.write_tensors_to_file(progress=False)
	writer.close()

	print(f"\n[convert_nar] wrote {out_path} ({out_path.stat().st_size} bytes)\n")
	print_tensor_table(table)

	return 0


if __name__ == "__main__":
	sys.exit(main())
