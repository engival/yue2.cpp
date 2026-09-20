#!/usr/bin/env python3
"""SPEC_LORA section 6.3, the weight half: how far a baked GGUF
(convert_nar.py --lora, whose base is the F32 checkpoint) is from what
`--nar-lora` computes at load (whose base is the GGUF's own F16 bytes).

	python3 tests/lora_bake_diff.py STOCK.gguf BAKED.gguf LORA.safetensors[:S]

Counts, per targeted tensor and in total, the F16 values that differ and by how
many F16 steps. numpy + the `gguf` package only -- no torch, no device.
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np


def load_adapter(path: str) -> dict:
	with open(path, "rb") as f:
		(header_len,) = struct.unpack("<Q", f.read(8))
		header = json.loads(f.read(header_len))
		start = 8 + header_len
		out = {}
		for key, rec in header.items():
			if key == "__metadata__":
				continue
			begin, end = rec["data_offsets"]
			f.seek(start + begin)
			raw = np.frombuffer(f.read(end - begin), dtype=np.uint8)
			if rec["dtype"] == "F32":
				a = raw.view(np.float32)
			elif rec["dtype"] == "F16":
				a = raw.view(np.float16).astype(np.float32)
			else:
				a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
			out[key] = a.reshape(rec["shape"])
	return out


def gguf_name(key: str) -> str | None:
	proj = {
		"nar_self_attn.q_proj": "nar_attn_q",
		"nar_self_attn.k_proj": "nar_attn_k",
		"nar_self_attn.v_proj": "nar_attn_v",
		"nar_self_attn.o_proj": "nar_attn_output",
		"nar_mlp.gate_proj": "nar_ffn_gate",
		"nar_mlp.up_proj": "nar_ffn_up",
		"nar_mlp.down_proj": "nar_ffn_down",
	}
	if not key.startswith("layers."):
		return None
	index, _, rest = key[len("layers."):].partition(".")
	stem = rest.rsplit(".", 1)[0]
	return f"blk.{index}.{proj[stem]}.weight" if stem in proj else None


def ulp_order(x: np.ndarray) -> np.ndarray:
	"""F16 is sign-magnitude, so its bits are not monotonic across zero; this is
	the usual remap to a totally ordered integer, where a difference of 1 is one
	F16 step. -0.0 and +0.0 both land on 0, as they must: they are the same
	number, and the two merges reach zero from opposite sides all the time."""
	bits = x.view(np.uint16).astype(np.int64)
	return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)


def main() -> int:
	import gguf

	stock, baked, lora_arg = sys.argv[1], sys.argv[2], sys.argv[3]
	head, sep, tail = lora_arg.rpartition(":")
	try:
		path, strength = (head, float(tail)) if sep else (lora_arg, 1.0)
	except ValueError:
		path, strength = lora_arg, 1.0

	adapter = load_adapter(path)
	pairs: dict = {}
	for key, value in adapter.items():
		name = gguf_name(key)
		if name is not None:
			pairs.setdefault(name, {})[key.rsplit(".", 1)[1]] = value

	stock_r = gguf.GGUFReader(stock)
	baked_r = gguf.GGUFReader(baked)
	stock_t = {t.name: t for t in stock_r.tensors}
	baked_t = {t.name: t for t in baked_r.tensors}

	total, differ, worst = 0, 0, 0
	for name in sorted(pairs):
		both = pairs[name]
		base = np.asarray(stock_t[name].data, dtype=np.float16).ravel()
		want = np.asarray(baked_t[name].data, dtype=np.float16).ravel()
		delta = ((np.float32(strength) * both["lora_B"]) @ both["lora_A"]).ravel()
		runtime = (base.astype(np.float32) + delta).astype(np.float16)
		steps = np.abs(ulp_order(runtime) - ulp_order(want))
		total += base.size
		differ += int(np.count_nonzero(steps))
		worst = max(worst, int(steps.max()))
	print(f"{len(pairs)} tensors, {total} F16 values: {differ} differ "
	      f"({100.0 * differ / total:.4f}%), worst {worst} F16 step(s)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
