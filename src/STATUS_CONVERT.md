# STATUS — `yue2 convert` (SPEC_CONVERT.md)

**Done; all five acceptance items pass.** All four GGUFs `yue2 convert` makes are
**byte-identical files** (`cmp`) to the Python-made ones in the repo root, KV
order and padding included, so there are no KV differences to explain.

New: `src/convert.cpp` / `src/convert.hpp` (one TU, anonymous namespace, linked
into `yue2` only), `tests/gguf_compare.py` (test-only, gguf-py from the
submodule), `tests/numpy_exp.cpp` + `tests/numpy_exp_ref.py`. Changed: `src/yue2.cpp` (subcommand + usage line),
`src/stage_song.cpp` (`find_default_gguf()`, auto-convert in `run_batch()`,
exported `gguf_home_dir()`), `src/stage_song.hpp`, `CMakeLists.txt`,
`README.md` (Quick start steps 2–3, `yue2-ar`/`yue2-nar` sections), `NOTICE.md`.

## 1. Byte compare vs the Python GGUFs

| file | Python path | tensors | KVs | result |
|---|---|---|---|---|
| `yue2-ar-q8_0.gguf` | `convert_ar.py --type f16` + `llama-quantize Q8_0` | 311 | 23 | **file identical** |
| `yue2-ar-f16.gguf`  | `convert_ar.py` | 311 | 22 | **file identical** |
| `yue2-nar-f16.gguf` | `convert_nar.py` | 317 | 19 | **file identical** |
| `yue2-vae-f32.gguf` | `convert_vae.py` | 173 | 13 | **file identical** |
| `yue2-vae-f16.gguf` (extra) | `convert_vae.py --type f16`, run fresh | 173 | 13 | **file identical** |
| `yue2-ar-bf16.gguf` (extra) | — | 311 | 22 | every BF16 tensor == source bits, F32 == source widened |
| `yue2-nar-f32.gguf` (extra) | — | 317 | 19 | every tensor == source widened; PE F16 == numpy cast |

```
build_convert/yue2 convert --out tests/out/convert [--force]
build_convert/yue2 convert --only ar --ar-type f16 --out tests/out/convert
python3 -I tests/gguf_compare.py yue2-ar-q8_0.gguf tests/out/convert/yue2-ar-q8_0.gguf   # RESULT: IDENTICAL
cmp yue2-ar-q8_0.gguf tests/out/convert/yue2-ar-q8_0.gguf                              # (same for the others)
```

**Q8_0 strategy: direct, no temp file.** Per tensor slab: BF16 → F32 →
`ggml_fp32_to_fp16_row` → `ggml_fp16_to_fp32_row` → `ggml_quantize_chunk(Q8_0)` —
exactly llama-quantize's F16 → F32 → `quantize_q8_0`. The file layout is
llama-quantize's too: tensors in `llama_model_loader::weight_name_comparer`
order (non-`blk` by name, then by layer number, then by name), KVs as in the
F16 file minus `general.file_type`, then `general.quantization_version = 2`,
`general.file_type = 7` at the end. No `llama_model_quantize()` and no 4.3 GB
intermediate.

Two numpy behaviours had to be ported for the VAE to be bit-exact
(`-ffp-contract=off` on `convert.cpp` so the compiler fuses nothing):

| what | why | evidence |
|---|---|---|
| `np.exp` on float32 = numpy's AVX2/AVX512F SIMD exp (Cody-Waite + 5/2 rational), **not** a correctly rounded `expf` | 12,954 of the 32,768 snake `exp(alpha/beta)` values differ by an ulp from `(float) exp((double) x)` | port matches `np.exp` on 4,000,000 random floats in [-110, 90] and N(0, 3), 0 mismatches |
| `np.linalg.norm` row sums = numpy's 8-accumulator pairwise summation | a sequential double sum would differ in the last bit now and then | VAE files byte-identical |

## 2. Tokenizer — pass

`convert/check_tokenizer.py` against the C++ AR GGUFs (q8_0 and f16): **3/3 texts
match, 0 mismatches** (473 / 771 / 280 ids). Run with the venv's interpreter
under `-I` (so the script dir is put on `sys.path` by hand), `--src` given
because `-I` hides the user-site `tqdm` that `huggingface_hub` needs, and the
older snapshot because the current one has no `examples/` dir (same
`qwen.tiktoken` blob):

```
GGML_VK_VISIBLE_DEVICES=1 <venv>/bin/python -I -c "
import sys, runpy; sys.path.insert(0, 'convert')
sys.argv = ['check_tokenizer.py', '--src', '<hub>/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19',
  '--songs-dir', '<a yue2 song --artifacts dir>', '--gguf', 'tests/out/convert/yue2-ar-q8_0.gguf',
  '--llama-tokenize', '<llama.cpp build>/bin/llama-tokenize']
runpy.run_path('convert/check_tokenizer.py', run_name='__main__')"
```

## 3. End to end on the Arc — pass

Request `{"style": "upbeat indie pop, acoustic guitar, light drums, warm female
vocal", "lyrics": "[Verse]\nMorning light across the floor\nCoffee waiting by
the door\n\n[Chorus]\nSing it out, sing it loud\nWe are dancing through the
crowd\n"}`, seed 4242, 67 s of audio, ~30 s per render on `Vulkan1 (BMG G31)`:

```
build_convert/yue2 song --request req.json --out old.flac --artifacts art_old --seed 4242 --gpu 1 \
    --ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf --vae yue2-vae-f32.gguf
build_convert/yue2 song ... --out new.flac --artifacts art_new ... --ar tests/out/convert/yue2-ar-q8_0.gguf ...
```

`old.flac` == `new.flac` (`cmp`). Artifacts: `abc_tokens`, `semantic`, `prefix`,
`latent`, `nar_noise`, `score.abc` identical; the JSONs differ only in GGUF
paths and timings.

## 4. Auto-convert — pass

Binary copied to an empty `tests/out/convert/auto/bin/`, run from
`tests/out/convert/auto/` (no GGUF next to it, one level up, or in the cwd):

```
bin/yue2 song --request ../req.json --out auto.flac --artifacts art --seed 4242 --gpu 1
yue2: converting YuE2 weights from <hub>/models--m-a-p--YuE2-3B/snapshots/29b3558… + <hub>/models--m-a-p--YuE2-Vae/snapshots/9a94e1d… (one-time, ~5.5 GB)...
convert: wrote …/auto/bin/yue2-ar-q8_0.gguf in 25.4 s
convert: wrote …/auto/bin/yue2-nar-f16.gguf in 4.5 s
convert: wrote …/auto/bin/yue2-vae-f32.gguf in 1.9 s
[done] auto.flac
```

`auto.flac` == `old.flac`. Only missing *default* files are converted, never one
named by `--ar/--nar/--vae`. Error paths checked: empty hub (`HF_HUB_CACHE` at an
empty dir) → `m-a-p/YuE2-Vae is not in the Hugging Face cache (…); run `hf
download m-a-p/YuE2-Vae` (or `yue2 convert --src-vae DIR` for a copy elsewhere)`;
two snapshots and no `refs/main` → error listing both; one snapshot and no ref
(`HF_HOME` at a fake hub) → used; existing output → `skipped (--force to redo)`.

## 5. Time and memory (warm page cache — no root to drop it)

| run | wall | user + sys | peak RSS |
|---|---|---|---|
| `yue2 convert --force` (AR q8_0 + NAR f16 + VAE f32) | **32.6 s** | 43.6 + 4.0 s | **317 MB** |
| `--only ar --ar-type f16` | 25.2 s | — | 122 MB |
| `--only vae` | 2.3 s | — | 297 MB |

The AR is bound by the SHA-256 of the 7.3 GB checkpoint (`yue2.source_sha256`):
llama.cpp's vendored C sha256 runs ~290 MB/s on a background thread while the
tensors are written behind a placeholder of the metadata's exact size; the real
metadata is written over it at the end. The NAR (2.9 GB of output) takes 4.6 s
once the hash is known, so the AR's own data work is a few seconds. Peak RSS is
the VAE's whole-tensor fold (largest weight 100 MB F32 + its double row work);
AR/NAR stream 8 M-element slabs.

## Design notes / deviations

- `yue2 convert` with no `--out` writes next to the binary (`gguf_home_dir()`),
  the first place `resolve_gguf()` looks; in this repo that is `build/`, not the
  repo-root hardlinks.
- Outputs go to `NAME.tmp` and are renamed at the end; a killed run leaves only
  the `.tmp`.
- NAR → AR `source_sha256` cross-check (`read_ar_sha256`): against
  `OUT/yue2-ar-<ar-type>.gguf` when it exists; the runtime check in
  `stage_nar.cpp` still guards every load.
- Config asserts ported from all three scripts; the VAE also refuses a
  `decoder_config` that `OobleckDecoder` would reject or that has a final Tanh
  (the C++ decoder has none; `OobleckDecoder` defaults to having one), and the
  decoder.* key set must equal the module list built from `c_mults` (the
  `load_state_dict(strict=True)` check).
- Not ported: the torch fold cross-check in `convert_vae.py` (replaced by §1),
  `convert_nar.py --lora` (out of scope, runtime `--nar-lora` exists).
- Portability: new code is `std::filesystem` / iostreams only. Still Linux-only
  around it: `gguf_home_dir()` sits on the existing `/proc/self/exe`
  (`exe_path()`), left for the portability task. MSVC will warn C4996 on
  `getenv`/`sscanf`.
- The platform-dependent side is the Python reference, not the C++: numpy's
  SIMD exp is used on x86-64 with AVX2/AVX512F only and falls back to libm
  `expf` elsewhere, so `convert_vae.py` on ARM would write different
  `alpha`/`beta` bits; `yue2 convert` writes the x86 numpy values everywhere.

## Review fixes (cold review, re-verified after)

| # | fix | where |
|---|---|---|
| 1 | per-writer temp `NAME.tmp.<random>`; rename replaces (no `remove` first); an `atexit` handler deletes our own temp when `die()` exits | `write_gguf` |
| 2 | cache lookup = huggingface_hub's: `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub` (`USERPROFILE` w/o `HOME`); leading `~` expanded; nothing set → clear die, no cwd-relative search | `hub_dir` |
| 3 | no-contraction flag per compiler: GCC/Clang `-ffp-contract=off`, MSVC `/fp:strict`, clang-cl `/clang:-ffp-contract=off` (plain `if()`: the frontend-variant genex needs CMake 3.30, the project is 3.16). New test `yue2-numpy-exp` (`YUE2_BUILD_TESTS`): port over every 997th float32 bit pattern (4,291,064 values, NaN skipped) vs the SHA-256 of numpy's outputs (`tests/numpy_exp_ref.py`) — **PASS** | CMakeLists.txt, `tests/numpy_exp*.{cpp,py}` |
| 4 | hash thread throws `std::runtime_error`; `wait_sha()` dies on the main thread | `sha256_of`, `wait_sha` |
| 5 | header length capped at 256 MB; unknown dtype skips the size check so read_f32's "only F32, F16 and BF16" fires; empty tensors die before any division | `SafeTensors`, `write_rows`, VAE pair check |
| 6 | NAR reads the AR's `source_sha256` up front, compares after the data (hash and write overlap again: NAR-only 25.4 s, = the hash) | `convert_nar` |
| 7 | `catch` on its own line; `fs::path tmp = out; tmp += …`; `GGML_ASSERT(align <= sizeof zeros)`; unwritable `--out` → hint `yue2 convert --out DIR`; sha mismatch → hint `yue2 convert --force --only ar` | — |

Re-run after the fixes: all four GGUFs (`ar-q8_0`, `ar-f16`, `nar-f16`,
`vae-f32`) `cmp`-identical again; auto-convert in an empty dir on `--gpu 1`
(31.0 s convert, then the song) → FLAC identical to the Python-GGUF render;
two concurrent `--only vae --force` into one dir → identical file, no temp
left; sha-mismatch and unwritable-dir dies leave no temp; `XDG_CACHE_HOME`,
`HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE='~/…'` and no-`HOME` paths each checked.
One full `yue2 convert` rerun took 42.0 s (AR 35.4 s) — the hash read slower
that time, page cache partly cold; peak RSS unchanged (317 MB).

## Open

- Cold-cache convert time not measured (needs root to drop caches); expect it
  to be the read of 7.6 GB from disk.
- `hf download` is still Python (huggingface_hub's CLI); a user without it can
  fetch the files any other way and pass `--src-3b` / `--src-vae`.
