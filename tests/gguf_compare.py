#!/usr/bin/env python3
"""SPEC_CONVERT.md section 4.1: compare two GGUFs — every KV (key, type, value,
order) and every tensor (name, type, shape, bytes, order). Test-only; uses
llama.cpp's gguf-py from the submodule, no install.

	python3 -I tests/gguf_compare.py A.gguf B.gguf [--ignore-kv KEY ...]

Exit 0 when nothing but the ignored KVs differs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402
import numpy as np  # noqa: E402


def kv_table(reader: gguf.GGUFReader) -> dict:
	out = {}
	for field in reader.fields.values():
		if field.name.startswith("GGUF."):
			continue
		types = tuple(t.name for t in field.types)
		if field.types[0] == gguf.GGUFValueType.ARRAY:
			value = [bytes(field.parts[i]) for i in field.data]
		elif field.types[0] == gguf.GGUFValueType.STRING:
			value = bytes(field.parts[field.data[0]])
		else:
			value = bytes(field.parts[field.data[0]])   # raw bytes: bit-exact float compare
		out[field.name] = (types, value)
	return out


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("a")
	ap.add_argument("b")
	ap.add_argument("--ignore-kv", action="append", default=[])
	args = ap.parse_args()

	ra = gguf.GGUFReader(args.a)
	rb = gguf.GGUFReader(args.b)
	bad = 0

	ka, kb = kv_table(ra), kv_table(rb)
	if list(ka) != list(kb):
		print(f"KV order/set differs:\n  a: {list(ka)}\n  b: {list(kb)}")
		bad += 1
	for key in sorted(set(ka) | set(kb)):
		if key in args.ignore_kv:
			if ka.get(key) != kb.get(key):
				print(f"KV {key}: differs (ignored)")
			continue
		if ka.get(key) != kb.get(key):
			show = lambda kv: kv if kv is None or kv[0][0] != "ARRAY" else (kv[0], f"array[{len(kv[1])}]")
			print(f"KV {key}: a={show(ka.get(key))} b={show(kb.get(key))}")
			bad += 1
	print(f"KVs compared: {len(ka)} vs {len(kb)}")

	ta = [(t.name, t.tensor_type.name, list(map(int, t.shape))) for t in ra.tensors]
	tb = [(t.name, t.tensor_type.name, list(map(int, t.shape))) for t in rb.tensors]
	if ta != tb:
		print("tensor table (name/type/shape/order) differs")
		da, db = {t[0]: t for t in ta}, {t[0]: t for t in tb}
		for name in sorted(set(da) | set(db)):
			if da.get(name) != db.get(name):
				print(f"  {name}: a={da.get(name)} b={db.get(name)}")
		bad += 1
	tensors_b = {t.name: t for t in rb.tensors}
	n_diff = 0
	for t in ra.tensors:
		u = tensors_b.get(t.name)
		if u is None:
			continue
		x = np.asarray(t.data).view(np.uint8).reshape(-1)
		y = np.asarray(u.data).view(np.uint8).reshape(-1)
		if x.size != y.size or not np.array_equal(x, y):
			n_eq = int(np.sum(x == y)) if x.size == y.size else -1
			print(f"tensor {t.name}: bytes differ ({n_eq}/{x.size} bytes equal)")
			n_diff += 1
	bad += n_diff
	print(f"tensors compared: {len(ta)} vs {len(tb)}, {n_diff} with different bytes")
	print("RESULT:", "IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
	return 0 if bad == 0 else 1


if __name__ == "__main__":
	sys.exit(main())
