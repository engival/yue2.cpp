# convert/ — STATUS

All three deliverables (SPEC.md sections 2, 5, 6) are done and have been run.

## 1. `convert_vae.py` — safetensors → GGUF

Reproduce:

```
cd convert       # from the repo root; ../../venv_yue2 is the torch-CPU venv
../../venv_yue2/bin/python convert_vae.py            # -> tests/out/yue2-vae-f32.gguf
../../venv_yue2/bin/python convert_vae.py --type f16 \
	--out ../tests/out/yue2-vae-f16.gguf
```

Result:
- `tests/out/yue2-vae-f32.gguf` — 265,437,856 bytes, 173 tensors, all F32.
- `tests/out/yue2-vae-f16.gguf` — 196,361,120 bytes, 173 tensors (Conv1d weights
  F16, ConvTranspose1d weights + all biases/alpha/beta F32).
- `general.architecture = "yue2-vae"`, `general.name = "YuE2-Vae decoder"`.
- `yue2vae.source_sha256 = 807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346`
  (independently verified against `sha256sum model.safetensors`).
- Weight-norm fold verified against torch's own hook-computed `.weight`
  (built the real `OobleckDecoder`, loaded the raw `weight_g`/`weight_v`
  state dict, ran one dummy forward to trigger every `weight_norm`
  `forward_pre_hook`, then compared): **max fold diff = 2.384e-07** (< 1e-6
  required).
- 217 raw source tensors (`decoder.*`, weight_g/weight_v split, encoder
  skipped entirely — never loaded) fold down to 173 GGUF tensors (2048-2×24
  weight_g/weight_v pairs collapse to 1 weight each, minus `layers.8` having
  no bias since `bias=False` on the final conv).
- Tensor table (name/shape/dtype) prints at the end of every run; verified
  with `gguf.GGUFReader` that `general.*`/`yue2vae.*` keys and tensor `ne`
  round-trip correctly, e.g. `decoder.layers.0.weight` reads back as ggml
  `ne = [7, 64, 2048]` for torch `[2048, 64, 7]` — matches SPEC.md §2's
  layout note with **zero transposes in the converter** (gguf-py reverses
  the numpy shape automatically).

### Deviation from spec
None in output format. One implementation note: rather than parsing
`weight_g`/`weight_v` names, the fold+writer classify each tensor by walking
the real `OobleckDecoder`'s `named_modules()` (isinstance checks against
`nn.Conv1d` / `nn.ConvTranspose1d` / `SnakeBeta`), which is what lets one
formula (`w = g * v / ||v||`, norm over axes `(1,2)`) work uniformly for both
conv kinds and is also how the f16-vs-f32 dtype decision (`ConvTranspose1d`
always F32) is made robustly instead of by name-sniffing.

## 2. `reference_decode.py` — torch CPU golden files

Reproduce:

```
../../venv_yue2/bin/python reference_decode.py
```
(defaults match the acceptance test: `--latent .../alley_swing_s1/latent.npy
--frames 8 48 --name alley_swing_s1`)

Result (`device="cpu"` hardcoded and asserted; never touched the GPU):
- `tests/golden/alley_swing_s1_f8.npy`: shape `(2, 15296)` = `1920*8-64`. ✓.
  Decode wall time 0.14s.
- `tests/golden/alley_swing_s1_f48.npy`: shape `(2, 92096)` = `1920*48-64`. ✓.
  Decode wall time 0.55s.
- Load time (decoder_only, CPU): 0.30s.
- `songs/ref_song/out/alley_swing_s1/` confirmed untouched (`ls -la`
  mtimes all predate this session — read-only respected).

## 3. `compare.py` — numeric comparison tool

Self-test (identical file): SNR = inf dB, PASS at `--min-snr 60`.

Sanity check against the tiled torch reference (`audio.flac`, PCM-quantized
per SPEC.md §5) over the first 92096 samples: max|err| = 6.6e-2, SNR =
31.31 dB — expected, this is *not* an acceptance comparison (untiled vs.
tiled+quantized), just confirms the tool loads/aligns/scores `.npy` vs
`.flac` correctly via `soundfile`.

Usage for the C++/verification team:
```
venv_yue2/bin/python compare.py golden.npy candidate.npy --min-snr 60   # SPEC §5 F32 im2col gate
venv_yue2/bin/python compare.py golden.npy candidate.npy                # F16 im2col: just record numbers
venv_yue2/bin/python compare.py audio.flac full_song.npy                # tiled full-song vs torch render
```
Handles `.npy` (any of `[2,S]`, `[S,2]`, `[1,S]`, `[S,1]`, `[S]`) and
`.wav`/`.flac`/`.ogg`/`.aiff` via `soundfile`; truncates to the shorter
length with a printed warning if lengths differ; exits 1 only when
`--min-snr` is given and not met.

## Environment
- `../venv_yue2/bin/python` (the workspace venv beside this repo) — installed `gguf` 0.19.0 via
  pip (torch 2.12.0+rocm7.2 CPU-mode, safetensors, numpy, soundfile already
  present).
- `modeling_vae.py` is loaded directly from the HF snapshot dir by path
  (`importlib.util.spec_from_file_location`) rather than via `import yue2`,
  so both scripts work even if the `yue2` package isn't on `sys.path`.

## Open questions / handoff notes for src/ (C++)
- None blocking. `decoder.layers.8` has no bias tensor in GGUF (matches
  `bias=False` in torch) — the C++ graph must skip the bias-add for that one
  conv.
- `yue2vae.c_mults` / `yue2vae.strides` are stored as the raw 6-value arrays
  from config.json (not the `[1] + c_mults` 7-value effective list) — SPEC.md
  §2 says "the 6 raw values", confirmed that's what got written.
- F16 gguf only shrinks Conv1d weights (the 7 `layers.*.1.weight` /
  `layers.*.3.weight` inside each ResidualUnit, plus `layers.0.weight` and
  `layers.8.weight`); every ConvTranspose1d weight and all bias/alpha/beta
  tensors stay F32 in both output files, per SPEC.md §2's Vulkan
  conv_transpose_1d F32-only constraint.

## Review fixes (cold code review, 2026-09-11)

Applied to `convert/*.py`, README.md, NOTICE.md, .gitignore, docs/ (M3, M4,
M6, S12, S14 -- src/ and CMakeLists.txt are other agents' area):

- **M6 / S14**: `--src` no longer defaults to an absolute path under the
  author's home directory. New
  `convert/common.py` (`resolve_snapshot`) resolves the snapshot dir from the
  huggingface_hub cache (`local_files_only=True`, no network) when `--src` is
  omitted, otherwise `--src` is used verbatim. Applies to `convert_vae.py`,
  `convert_ar.py`, `reference_decode.py` (also dropped its
  `sys.path.insert(...)` hack, and its previously-defaulted `--latent`, which
  pointed at this user's `songs/`, is now `required=True`).
- `common.py` also factors `sha256_of` and the tensor-table printer
  (`print_tensor_table`) out of `convert_vae.py`/`convert_ar.py`, and
  `load_module_from_snapshot` out of `convert_vae.py`'s
  `load_modeling_vae`/`reference_decode.py`'s `load_yue2_module` (identical
  bodies, now one function).
- **M3**: README's `--snapshot`/`-o` corrected to the flags the scripts
  actually take (`--src`/`--out`); added the `llama-quantize`/`llama-tokenize`
  section explaining `LLAMA_BUILD_TOOLS OFF` in this repo's CMake and pointing
  at building them from a separate llama.cpp checkout instead.
- **M4**: NOTICE.md rewritten -- `ggml/` submodule reference (removed in
  `6380b8f`) replaced with llama.cpp (which now carries ggml) + nlohmann/json
  (vendored inside it) + the BPE code copied from llama.cpp's
  `conversion/qwen.py` into `convert/convert_ar.py`; added the `transformers`
  runtime dependency note.
- **S12**: `.gitignore` gained `out/` (README's own `--artifacts out/song`
  example), `*.npy` (was only ignored under `tests/out/`/`tests/golden/`),
  `*.safetensors`, `venv*/`, `.venv/`, `*.log`.
- Added `convert/requirements.txt` (used by README's install line).
- **Re-verified after refactor**: `convert_vae.py` and `convert_ar.py`
  outputs are byte-identical (sha256) to the pre-refactor
  `tests/out/yue2-vae-f32.gguf` and `tests/out/yue2-ar-f16.gguf`. All CPU
  only, no torch on GPU.
- `check_tokenizer.py` also lost its hardcoded `HF_SNAPSHOT`/`SONGS_DIR`/
  `sys.path` block (now `--src`, `--songs-dir` required; `--llama-tokenize`
  defaults to bare `llama-tokenize` on `$PATH` instead of a hardcoded Vulkan
  build path) -- rerun 3/3 PASS against
  `llama.cpp/build_vulkan/bin/llama-tokenize` on `yue2-ar-f16.gguf`.
