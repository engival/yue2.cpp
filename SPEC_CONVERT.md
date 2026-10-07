# SPEC_CONVERT — safetensors → GGUF inside yue2 (no Python)

Goal: a user with only the `yue2` binary and m-a-p's two Hugging Face downloads
(`m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`) can generate songs. No Python, no torch, no
third-party GGUF repo. The Python converters in `convert/` stay as the
reference implementation and golden source; they are not removed.

## 1. Surface

```
yue2 convert [--src-3b DIR] [--src-vae DIR] [--out DIR]
             [--ar-type q8_0|f16|bf16] [--nar-type f16|f32] [--vae-type f32|f16]
             [--only ar|nar|vae] [--force]
```

- Defaults produce exactly the three files `resolve_gguf()` looks for:
  `yue2-ar-q8_0.gguf`, `yue2-nar-f16.gguf`, `yue2-vae-f32.gguf`.
  Default `--out` = the directory `resolve_gguf()` searches (next to the binary).
  Output names follow the chosen types (`yue2-ar-f16.gguf`, …) like the Python ones.
- Existing outputs are skipped unless `--force`. Write to `NAME.tmp`, rename on
  success — an interrupted run never leaves a truncated GGUF under the real name.
- **Auto-convert:** `yue2 song` / `yue2 batch`, when one of the *default* GGUFs
  is missing and the user did not pass that path explicitly, print one line
  (`yue2: converting YuE2 weights from <src> (one-time, ~N GB)…`) and run the
  conversion for the missing files, then continue. If no source snapshot is
  found, fail with a message saying which `hf download` to run (or `--src-*`).

## 2. Finding the sources

Same rule as `convert/common.py:resolve_snapshot()` (huggingface_hub cache
layout), implemented with `std::filesystem` only:

1. `--src-3b` / `--src-vae` if given (a snapshot dir containing `config.json`).
2. Hub dir = `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`
   (`%USERPROFILE%` when `HOME` is unset — Windows is a target, see §6).
3. `models--m-a-p--YuE2-3B/refs/main` → snapshot hash → `snapshots/<hash>/`.
   If `refs/main` is absent, the single snapshot dir; several and no ref → error
   listing them.

Required files: 3B = `config.json`, `model.safetensors`, `qwen.tiktoken`;
VAE = `config.json`, `model.safetensors`. Snapshot entries are symlinks into
`blobs/` — follow them.

## 3. Conversions (each must match its Python converter)

Read `convert_ar.py`, `convert_nar.py`, `convert_vae.py` — they are the contract
(tensor maps, KV keys and values, which tensors stay F32, asserts on config).
Port every assert on `config.json` values; a different checkpoint must fail
loudly, not convert silently.

- **safetensors reader:** u64 LE header length + JSON header + data; dtypes
  BF16/F16/F32 at least. mmap or buffered reads; stream tensor by tensor — never
  hold the whole 7 GB checkpoint in RAM.
- **AR:** qwen3 tensor names, gpt2-style vocab + merges rebuilt from
  `qwen.tiktoken` (port `bpe()` / `token_bytes_to_string` / `build_vocab`
  exactly, including the special / codec / pad token ranges and token types).
  q8_0 default. **The q8_0 file must be bit-identical in tensor data to the
  current pipeline** (`convert_ar.py --type f16` then `llama-quantize … q8_0`):
  that path rounds through F16 before quantizing and llama-quantize picks
  per-tensor types — reproduce both, or write the F16 GGUF to a temp file and
  call `llama_model_quantize()` (libllama API, already linked) then delete it.
  Either is acceptable; say which in STATUS.
- **NAR:** rename + cast; `latent_pos_embed.pe` F16 always plus the formula
  check (< 8e-3); norms F32. `yue2.source_sha256` and the AR-pairing key
  (`read_ar_sha256`) — check what the runtime reads/validates and produce the
  same values. SHA-256 of a multi-GB file: a small vendored implementation is
  fine (record it in NOTICE.md) — check whether llama.cpp/ggml already ship one first.
  `--lora` merging is out of scope (runtime `--nar-lora` exists).
- **VAE:** weight-norm fold `g * v / ||v||` over every dim but 0, computed in
  **double** then cast (as the Python does), `exp(alpha)`, `exp(beta)`, F32
  default. There is no torch cross-check any more — the golden comparison in §4
  replaces it.
- GGUF writing via ggml's `gguf_*` API (`gguf_init_empty`, `gguf_set_*`,
  `gguf_add_tensor`, `gguf_write_to_file` or meta + streamed data). Never build
  the whole file in RAM.

## 4. Acceptance

1. **Bit-identical tensors** vs the Python-made GGUFs in the repo root
   (`yue2-ar-f16.gguf`, `yue2-ar-q8_0.gguf`, `yue2-nar-f16.gguf`,
   `yue2-vae-f32.gguf`): same tensor names, types, shapes, bytes. Also every KV
   equal except converter-identity fields (name/version strings) — list any
   difference in STATUS with the reason. A test-only compare script may use
   gguf-py (`llama.cpp/gguf-py`, run with `python3 -I`); the converter itself is
   C++ only.
2. Tokenizer: `convert/check_tokenizer.py` passes against the C++ AR GGUF.
3. End-to-end: one `yue2 song` with a fixed seed on **`--gpu 1` (Intel Arc)**
   with the old GGUFs and with the new ones → identical output (it should be,
   given 1.).
4. Auto-convert path exercised once in a temp dir (copy of the binary, empty
   GGUF dir, HF cache present): the song runs after the conversion line.
5. Report wall time and peak RSS of `yue2 convert` (all three).

## 5. Docs

README "Get the weights" → `hf download` both repos, then `yue2 convert` (or let
`yue2 song` do it on first run); the Python converters become an optional
"reference converters" note. Usage text in `src/yue2.cpp` gains `convert`.
NOTICE.md: the tokenizer port (llama.cpp `conversion/qwen.py`, MIT) now also
lives in C++; any vendored SHA-256.

## 6. Constraints

- Portable C++17: `std::filesystem`, no `unistd.h`/POSIX-only calls in the new
  code (Windows builds are next). Don't fix the existing `/proc/self/exe` /
  `unistd.h` uses here — that's the portability task.
- New code in `src/convert.cpp` (+ `.hpp`); wired into `yue2` like the stages.
- Repo style and rules in `CLAUDE.md` apply (tabs, brace placement, warnings clean).
