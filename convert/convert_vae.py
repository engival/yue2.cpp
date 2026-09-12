#!/usr/bin/env python3
"""safetensors -> GGUF converter for the YuE2 Oobleck VAE decoder.

See ../SPEC.md sections 1 and 2 for the contract this implements. Run with
the venv_yue2 interpreter (has torch CPU + safetensors + numpy); `gguf` must
be installed into that venv (`pip install gguf`).

Usage:
	venv_yue2/bin/python convert_vae.py \
		--src /path/to/hf/snapshot/models--m-a-p--YuE2-Vae/snapshots/<rev> \
		--out tests/out/yue2-vae-f32.gguf \
		[--type f32|f16]

`--src` may be omitted if `m-a-p/YuE2-Vae` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from common import load_module_from_snapshot, print_tensor_table, resolve_snapshot, sha256_of

REPO_ID = "m-a-p/YuE2-Vae"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "tests" / "out" / "yue2-vae-f32.gguf")


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
	"""torch weight_norm(dim=0): w = g * v / ||v||, norm over every dim but 0.

	Works identically for Conv1d ([Cout,Cin,K], norm over (Cin,K)) and
	ConvTranspose1d ([Cin,Cout,K], norm over (Cout,K)) -- both normalize over
	axes (1, 2) of a 3-D weight, only the *meaning* of the axes differs.
	"""
	norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape(-1, 1, 1)
	return g * v / norm


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help="HF snapshot dir with model.safetensors + config.json "
	                      f"(default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--out", default=DEFAULT_OUT, help="output .gguf path")
	ap.add_argument("--type", choices=["f32", "f16"], default="f32",
	                 help="f16 stores Conv1d (not ConvTranspose1d) weights as F16; everything else stays F32")
	args = ap.parse_args()

	src = resolve_snapshot(args.src, REPO_ID)
	st_path = src / "model.safetensors"
	cfg_path = src / "config.json"
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	config = json.loads(cfg_path.read_text())
	dcfg = config["decoder_config"]
	print(f"[convert_vae] source: {st_path}")
	print(f"[convert_vae] decoder_config: {dcfg}")

	source_sha256 = sha256_of(st_path)
	print(f"[convert_vae] source sha256: {source_sha256}")

	# Read every decoder.* tensor from the safetensors file, raw (weight_g/weight_v
	# split intact) -- we fold weight-norm ourselves and cross-check against torch.
	raw = {}
	with safe_open(st_path, framework="pt", device="cpu") as f:
		for key in f.keys():
			if key.startswith("decoder."):
				raw[key] = f.get_tensor(key)

	# Build the real torch decoder and load those same tensors, so we can read
	# back weight_norm's own folded `.weight` (populated by its forward_pre_hook
	# the first time each submodule runs) and assert our manual fold matches it.
	modeling_vae = load_module_from_snapshot(src, "modeling_vae.py", "yue2_modeling_vae")
	decoder = modeling_vae.OobleckDecoder(**dcfg)
	state = {k[len("decoder."):]: v for k, v in raw.items()}
	decoder.load_state_dict(state, strict=True)
	decoder.eval()
	with torch.inference_mode():
		dummy = torch.zeros(1, dcfg["latent_dim"], 32, dtype=torch.float32)
		decoder(dummy)  # trigger every weight_norm hook so module.weight is folded

	import gguf

	writer = gguf.GGUFWriter(str(out_path), "yue2-vae")  # constructor already calls add_architecture()
	writer.add_name("YuE2-Vae decoder")

	def kv_i32(key, val):
		writer.add_key_value(key, int(val), gguf.GGUFValueType.INT32)

	def kv_i32_array(key, vals):
		writer.add_array(key, [int(v) for v in vals])

	def kv_bool(key, val):
		writer.add_key_value(key, bool(val), gguf.GGUFValueType.BOOL)

	def kv_str(key, val):
		writer.add_string(key, str(val))

	kv_i32("yue2vae.channels", dcfg["channels"])
	kv_i32_array("yue2vae.c_mults", dcfg["c_mults"])
	kv_i32_array("yue2vae.strides", dcfg["strides"])
	kv_i32("yue2vae.latent_dim", dcfg["latent_dim"])
	kv_i32("yue2vae.out_channels", dcfg["out_channels"])
	kv_i32("yue2vae.sample_rate", config["sample_rate"])
	kv_i32("yue2vae.downsampling_ratio", config["downsampling_ratio"])
	kv_i32("yue2vae.decode_core_frames", config["decode_core_frames"])
	kv_i32("yue2vae.decode_halo_frames", config["decode_halo_frames"])
	kv_bool("yue2vae.snake_folded", True)
	kv_str("yue2vae.source_sha256", source_sha256)

	table = []  # (name, shape, dtype) for the printed report
	max_fold_diff = 0.0

	def add_conv(prefix: str, module) -> None:
		nonlocal max_fold_diff
		is_convT = isinstance(module, torch.nn.ConvTranspose1d)
		g = raw[f"decoder.{prefix}.weight_g"].numpy().astype(np.float64)
		v = raw[f"decoder.{prefix}.weight_v"].numpy().astype(np.float64)
		folded = fold_weight_norm(g, v).astype(np.float32)

		torch_w = module.weight.detach().numpy().astype(np.float32)
		diff = float(np.max(np.abs(folded - torch_w)))
		max_fold_diff = max(max_fold_diff, diff)
		assert diff < 1e-6, f"{prefix}: weight-norm fold mismatch {diff}"

		store_f16 = (args.type == "f16") and not is_convT
		out = folded.astype(np.float16) if store_f16 else folded
		name = f"decoder.{prefix}.weight"
		writer.add_tensor(name, np.ascontiguousarray(out))
		table.append((name, list(folded.shape), "F16" if store_f16 else "F32"))

		if module.bias is not None:
			bias = raw[f"decoder.{prefix}.bias"].numpy().astype(np.float32)
			bname = f"decoder.{prefix}.bias"
			writer.add_tensor(bname, np.ascontiguousarray(bias))
			table.append((bname, list(bias.shape), "F32"))

	def add_snake(prefix: str) -> None:
		alpha = raw[f"decoder.{prefix}.alpha"].numpy().astype(np.float32)
		beta = raw[f"decoder.{prefix}.beta"].numpy().astype(np.float32)
		alpha_exp = np.exp(alpha).astype(np.float32)
		beta_exp = np.exp(beta).astype(np.float32)
		aname = f"decoder.{prefix}.alpha"
		bname = f"decoder.{prefix}.beta"
		writer.add_tensor(aname, np.ascontiguousarray(alpha_exp))
		writer.add_tensor(bname, np.ascontiguousarray(beta_exp))
		table.append((aname, list(alpha_exp.shape), "F32"))
		table.append((bname, list(beta_exp.shape), "F32"))

	for name, module in decoder.named_modules():
		if name == "":
			continue
		if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
			add_conv(name, module)
		elif isinstance(module, modeling_vae.SnakeBeta):
			add_snake(name)
		# containers (Sequential/DecoderBlock/ResidualUnit) and Identity/Tanh: skip

	print(f"[convert_vae] {len(raw)} raw source tensors -> {len(table)} GGUF tensors")
	print(f"[convert_vae] max weight-norm fold diff vs torch: {max_fold_diff:.3e}")

	writer.write_header_to_file()
	writer.write_kv_data_to_file()
	writer.write_tensors_to_file(progress=False)
	writer.close()

	print(f"\n[convert_vae] wrote {out_path} ({out_path.stat().st_size} bytes)\n")
	print_tensor_table(table, name_width=60)

	return 0


if __name__ == "__main__":
	sys.exit(main())
