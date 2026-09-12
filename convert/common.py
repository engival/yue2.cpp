"""Shared helpers for the yue2.cpp converters (convert_ar.py, convert_vae.py,
reference_ar.py, reference_decode.py, check_tokenizer.py).

Deliberately small: sha256 of a big file, resolving an HF snapshot dir without
hard-coding anyone's home directory, and printing the tensor table every
converter ends its run with.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def resolve_snapshot(explicit: str | None, repo_id: str) -> Path:
	"""Return the HF snapshot dir to read from.

	If --src was given, use it verbatim. Otherwise look it up in the user's
	own huggingface_hub cache (local_files_only, no network) -- this is the
	same cache `hf download` / `from_pretrained(local_files_only=True)` use,
	so it works for whichever revision the user actually has, on whatever
	machine this runs on.
	"""
	if explicit:
		return Path(explicit)

	try:
		from huggingface_hub import snapshot_download
		from huggingface_hub.errors import LocalEntryNotFoundError
	except ImportError as e:
		raise SystemExit(
			f"--src not given and huggingface_hub is not installed to look up "
			f"{repo_id!r} in the cache ({e}); pass --src explicitly"
		)

	try:
		path = snapshot_download(repo_id, local_files_only=True)
	except LocalEntryNotFoundError:
		raise SystemExit(
			f"--src not given and {repo_id!r} is not in the local huggingface "
			f"cache (~/.cache/huggingface/hub); pass --src /path/to/snapshot "
			f"or `hf download {repo_id}` first"
		)
	return Path(path)


def load_module_from_snapshot(src: Path, module_file: str, module_name: str):
	"""Import a yue2 custom-code module (e.g. modeling_vae.py) directly by
	path from an HF snapshot dir, without depending on the `yue2` package
	being importable from site-packages."""
	import importlib.util

	spec = importlib.util.spec_from_file_location(module_name, src / module_file)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def print_tensor_table(table: list[tuple[str, list[int], str]], name_width: int = 28) -> None:
	"""table: list of (gguf tensor name, shape, dtype-name). Same layout both
	converters printed by hand before this was factored out."""
	shape_width = 22
	print(f"{'name':{name_width}s} {'shape':{shape_width}s} dtype")
	print("-" * (name_width + shape_width + 10))
	for name, shape, dtype in table:
		print(f"{name:{name_width}s} {str(shape):{shape_width}s} {dtype}")
	print("-" * (name_width + shape_width + 10))
	print(f"{len(table)} tensors total")
