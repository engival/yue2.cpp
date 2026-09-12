#!/usr/bin/env python3
"""SPEC_AR.md section 2 acceptance check: llama-tokenize ids == tiktoken ids.

Compares `llama-tokenize -m <gguf> --ids` against
`yue2.tokenization_yue2.YuE2TextTokenizer.encode()` (tiktoken, NFC-normalized)
on three texts: the demo request text (examples/tonight-awake.json ->
SongRequest.text(), Chinese lyrics), the alley_swing_s1 request.json text
(English), and a snippet of ABC notation (score.abc from that dir). Zero
mismatches required.

Usage:
	venv_yue2/bin/python check_tokenizer.py --songs-dir /path/to/songs/ref_song/out/alley_swing_s1 \
		[--gguf tests/out/yue2-ar-f16.gguf] [--llama-tokenize /path/to/llama-tokenize]

`--src` may be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot(). `--llama-tokenize` defaults to
`llama-tokenize` on $PATH (this repo's own CMake build does not produce one --
LLAMA_BUILD_TOOLS is OFF, see README.md "Build a llama-tokenize/-quantize" --
so point this at a separate llama.cpp build, e.g. `~/bin/llama-tokenize`).
Run with the venv_yue2 interpreter directly -- no sys.path hacks needed.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from common import resolve_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_GGUF = str(ROOT / "tests" / "out" / "yue2-ar-f16.gguf")
DEFAULT_LLAMA_TOKENIZE = "llama-tokenize"

REPO_ID = "m-a-p/YuE2-3B"


def load_yue2_tokenizer(hf_snapshot: Path):
	from yue2.tokenization_yue2 import YuE2TextTokenizer
	return YuE2TextTokenizer(hf_snapshot / "qwen.tiktoken")


def build_test_texts(hf_snapshot: Path, songs_dir: Path) -> dict:
	from yue2.protocol import SongRequest

	texts = {}

	demo = json.loads((hf_snapshot / "examples" / "tonight-awake.json").read_text())
	req = SongRequest(style=demo["style"], lyrics=demo["lyrics"], cot=demo.get("cot", "full"))
	texts["demo_request_text_zh"] = req.text()

	alley_req = json.loads((songs_dir / "request.json").read_text())
	req2 = SongRequest(style=alley_req["style"], lyrics=alley_req["lyrics"], cot=alley_req.get("cot", "full"))
	texts["alley_swing_s1_request_text_en"] = req2.text()

	abc_full = (songs_dir / "score.abc").read_text()
	texts["abc_snippet"] = "".join(abc_full.splitlines(keepends=True)[:20])

	return texts


def llama_tokenize_ids(binary: str, gguf: str, text: str) -> list:
	with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
		f.write(text)
		path = f.name
	try:
		out = subprocess.run(
			[binary, "-m", gguf, "-f", path, "--ids", "--no-bos", "--no-parse-special", "--no-escape", "--log-disable"],
			capture_output=True, text=True, check=True,
		)
	finally:
		Path(path).unlink(missing_ok=True)
	# llama-tokenize with --ids prints one bracketed python-literal list per line group;
	# collect every integer that appears in stdout, in order.
	ids = []
	for line in out.stdout.splitlines():
		line = line.strip()
		if not line:
			continue
		if line.startswith("["):
			ids.extend(ast.literal_eval(line))
	return ids


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help=f"HF snapshot dir (default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--songs-dir", required=True, help="dir with request.json + score.abc (a songs/.../out/<name> dir)")
	ap.add_argument("--gguf", default=DEFAULT_GGUF)
	ap.add_argument("--llama-tokenize", default=DEFAULT_LLAMA_TOKENIZE)
	args = ap.parse_args()

	hf_snapshot = resolve_snapshot(args.src, REPO_ID)
	songs_dir = Path(args.songs_dir)

	tok = load_yue2_tokenizer(hf_snapshot)
	texts = build_test_texts(hf_snapshot, songs_dir)

	total_mismatches = 0
	for name, text in texts.items():
		want = tok.encode(text)  # NFC + tiktoken.encode_ordinary, per YuE2TextTokenizer.encode
		got = llama_tokenize_ids(args.llama_tokenize, args.gguf, unicodedata.normalize("NFC", text))
		ok = want == got
		print(f"[check_tokenizer] {name}: {len(want)} tiktoken ids, {len(got)} llama-tokenize ids -> {'PASS' if ok else 'FAIL'}")
		if not ok:
			total_mismatches += 1
			n = min(len(want), len(got))
			first_diff = next((i for i in range(n) if want[i] != got[i]), n)
			print(f"    first differing index: {first_diff}")
			print(f"    tiktoken   [{max(0,first_diff-3)}:{first_diff+5}] = {want[max(0,first_diff-3):first_diff+5]}")
			print(f"    llama-tok  [{max(0,first_diff-3)}:{first_diff+5}] = {got[max(0,first_diff-3):first_diff+5]}")

	print(f"\n[check_tokenizer] {len(texts) - total_mismatches}/{len(texts)} texts match, "
	      f"{total_mismatches} mismatch(es)")
	return 1 if total_mismatches else 0


if __name__ == "__main__":
	sys.exit(main())
