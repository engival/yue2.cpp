#!/usr/bin/env python3
"""Writes one malformed adapter per error of SPEC_LORA section 3.4, so every
message can be triggered without hand-editing a 140 MB file.

	python3 tests/lora_bad_safetensors.py tests/out/lora/bad

It writes the files plus `cases.tsv` (path, then a substring the error must
contain) and needs nothing but numpy. The GGUF shapes it plays against are the
NAR's: blk.0.nar_attn_q.weight is [2048, 2048] and nar.vae2llm.weight is
[2048, 64], both in torch [out, in] order.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

DTYPE = {"float32": "F32", "float16": "F16", "int64": "I64"}


def write(path: Path, tensors: dict, *, header_len: int | None = None,
          header_bytes: bytes | None = None, truncate: int = 0) -> None:
	"""A plain safetensors file, with the hooks the malformed cases need."""
	blob = bytearray()
	header: dict = {}
	for name, array in tensors.items():
		raw = array.tobytes()
		header[name] = {
			"dtype": DTYPE[array.dtype.name],
			"shape": list(array.shape),
			"data_offsets": [len(blob), len(blob) + len(raw)],
		}
		blob += raw
	body = header_bytes if header_bytes is not None else json.dumps(header).encode()
	out = struct.pack("<Q", len(body) if header_len is None else header_len) + body + bytes(blob)
	path.write_bytes(out[:len(out) - truncate] if truncate else out)


def main() -> int:
	out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/out/lora/bad")
	out.mkdir(parents=True, exist_ok=True)

	rng = np.random.default_rng(1)
	a = rng.standard_normal((32, 2048), dtype=np.float32) * 0.01
	b = rng.standard_normal((2048, 32), dtype=np.float32) * 0.01
	q = "layers.0.nar_self_attn.q_proj"

	cases: list[tuple[str, str]] = []

	def case(name: str, expect: str, **kwargs) -> None:
		write(out / name, kwargs.pop("tensors", {}), **kwargs)
		cases.append((str(out / name), expect))

	# Container (3.1).
	cases.append((str(out / "does_not_exist.safetensors"), "cannot open"))
	(out / "too_short.safetensors").write_bytes(b"abc")
	cases.append((str(out / "too_short.safetensors"), "not a safetensors file"))
	case("header_len.safetensors", "does not fit",
	     tensors={f"{q}.lora_A": a, f"{q}.lora_B": b}, header_len=1 << 40)
	case("header_json.safetensors", "not a JSON object", header_bytes=b"{not json")
	case("short_data.safetensors", "shorter than its header claims",
	     tensors={f"{q}.lora_A": a, f"{q}.lora_B": b}, truncate=4096)
	case("dtype.safetensors", "only F32, F16 and BF16",
	     tensors={f"{q}.lora_A": np.zeros((32, 2048), dtype=np.int64), f"{q}.lora_B": b})

	# Keys (3.2) and pairing (3.4).
	case("unknown_key.safetensors", "maps to no NAR tensor",
	     tensors={"layers.0.nar_self_attn.z_proj.lora_A": a,
	              "layers.0.nar_self_attn.z_proj.lora_B": b})
	case("comfyui.safetensors", "fused ComfyUI layout",
	     tensors={"diffusion_model.model.layers.0.self_attn.o_proj.lora_down.weight": a,
	              "diffusion_model.model.layers.0.self_attn.o_proj.lora_up.weight": b})
	case("only_a.safetensors", "no lora_B", tensors={f"{q}.lora_A": a})
	case("only_b.safetensors", "no lora_A", tensors={f"{q}.lora_B": b})

	# Shapes (3.4).
	case("rank.safetensors", "rank mismatch",
	     tensors={f"{q}.lora_A": a, f"{q}.lora_B": b[:, :16].copy()})
	case("in_dim.safetensors", "the base is",
	     tensors={f"{q}.lora_A": a[:, :999].copy(), f"{q}.lora_B": b})
	case("out_dim.safetensors", "the base is",
	     tensors={f"{q}.lora_A": a, f"{q}.lora_B": b[:999].copy()})
	case("replacement.safetensors", "not the base's",
	     tensors={"vae2llm.weight": np.zeros((2048, 63), dtype=np.float32)})

	# Values (3.4).
	nan = a.copy()
	nan[0, 0] = np.nan
	case("nonfinite.safetensors", "non-finite value",
	     tensors={f"{q}.lora_A": nan, f"{q}.lora_B": b})

	(out / "cases.tsv").write_text("".join(f"{p}\t{e}\n" for p, e in cases))
	print(f"{len(cases)} cases in {out}/cases.tsv")

	# Not an error: one layer, one projection, no replacements. SPEC_LORA 3.4
	# ends on this -- every key must be consumed, not every tensor covered.
	write(out / "one_tensor.safetensors", {f"{q}.lora_A": a, f"{q}.lora_B": b})
	print(f"valid single-tensor adapter: {out}/one_tensor.safetensors")

	# The one error that needs a model rather than an adapter: a target tensor of
	# quantised type. No quantised NAR GGUF exists, so write a one-tensor stub --
	# the check runs while the tensor list is being read, before anything else
	# about the file matters. Needs the `gguf` package (venv_yue2).
	try:
		import gguf
	except ImportError:
		print("no `gguf` package: skipping the quantised-target stub")
		return 0
	writer = gguf.GGUFWriter(str(out / "quantised-nar.gguf"), "yue2-nar")
	writer.add_tensor("blk.0.nar_attn_q.weight",
	                  gguf.quants.quantize(np.zeros((2048, 2048), dtype=np.float32),
	                                       gguf.GGMLQuantizationType.Q8_0),
	                  raw_dtype=gguf.GGMLQuantizationType.Q8_0)
	writer.write_header_to_file()
	writer.write_kv_data_to_file()
	writer.write_tensors_to_file(progress=False)
	writer.close()
	print(f"quantised-target stub: {out}/quantised-nar.gguf "
	      f"(expect \"only F16 and F32 targets can be merged\")")
	return 0


if __name__ == "__main__":
	sys.exit(main())
