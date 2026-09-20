# STATUS_LORA — stage 10, `--nar-lora FILE[:S]`

What was built against [SPEC_LORA.md](../SPEC_LORA.md), the measured numbers and
the places the implementation reads the SPEC differently.

## What is where

| file | what |
|---|---|
| `src/common/lora.hpp` | the whole feature: `FILE[:S]` parsing, the safetensors reader, the adapter-key → GGUF-name map, the plan, the single-threaded merge, the log line |
| `src/stage_nar.cpp` | `--nar-lora`, and the merge path in `load_gguf()` (widen → deltas → narrow, on targeted tensors only) |
| `src/stage_nar.hpp` | `NarParams::nar_lora` |
| `src/stage_song.cpp`, `.hpp` | `--nar-lora` on `song` / `batch`, and the `nar_lora` block in `config.json` |
| `convert/convert_nar.py` | `--lora FILE[:S]`, the bake (SPEC §5) |
| `tests/lora_bad_safetensors.py` | writes one malformed adapter per §3.4 error, plus `cases.tsv` and a quantised-target GGUF stub |
| `tests/lora_bake_diff.py` | counts the F16 values a baked GGUF and the runtime merge disagree on |
| `CMakeLists.txt` | `yue2-nar` now links `vendor::nlohmann` — for the safetensors header and nothing else |

## Reproducing

```bash
cmake -B build_lora -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
nice -n 19 cmake --build build_lora -j4

# the short NAR run every acceptance item below uses
NAR="--ar yue2-ar-q8_0.gguf -m yue2-nar-f16.gguf \
     --prefix tests/golden/nar_prefix_ids.npy --codec tests/golden/nar_codec_ids.npy \
     --noise tests/golden/nar_noise.npy --steps 2 --gpu 1 --kv-f16"
```

`LORA.safetensors` below is the rank-32, 396-tensor, 28-layer community adapter
(F32; its BF16 twin was run too). It is not in this repo.

## Acceptance

| # | item | result |
|---|---|---|
| 1 | no flag, vs the `build/` binary | **byte-identical** (`cmp` clean, 128 frames × 64) |
| 2 | `--nar-lora LORA.safetensors:0` vs no flag | **byte-identical** |
| 3 | runtime merge vs baked GGUF | 1.0: latent **90.72 dB**, 26 964 / 1 409 286 144 F16 weights differ (0.0019 %), worst **1** F16 step. 0.5: **92.85 dB**, 41 826 (0.0030 %), worst **1** step |
| 4 | stock vs `:1.0`, `--steps 32` | both finite, **20.70 dB** — clearly a different render |
| 5 | `:0.5` twice vs `:1.0` | **91.54 dB** |
| 6 | every §3.4 message | **17 / 17** triggered (15 adapter files + non-finite strength + quantised target) |
| 7 | load cost of the merge | **4.56 – 4.77 s** single-threaded; whole-load 3.3 s → 11.3 – 12.1 s, so ~8 s added. Budget was ≤ 10 s |
| 8 | `YUE2_WARN_FLAGS` clean, README, this file | clean (no warning from `src/`); README §"Trying a NAR LoRA" |

Timings were taken while another session was rendering on both cards, so treat
them as upper bounds. The merge is CPU-only and single-threaded, as the SPEC
requires; nothing was parallelised to reach the budget.

### 1–2, 4–5 — the commands

```bash
nice -n 19 ./build/yue2-nar      $NAR -o tests/out/lora/base_ref.npy
nice -n 19 ./build_lora/yue2-nar $NAR -o tests/out/lora/base_new.npy
cmp tests/out/lora/base_ref.npy tests/out/lora/base_new.npy

nice -n 19 ./build_lora/yue2-nar $NAR --nar-lora LORA.safetensors:0 -o tests/out/lora/s0.npy
cmp tests/out/lora/base_new.npy tests/out/lora/s0.npy

nice -n 19 ./build_lora/yue2-nar $NAR --nar-lora LORA.safetensors:0.5 \
                                      --nar-lora LORA.safetensors:0.5 -o tests/out/lora/stack.npy
```

### 3 — the bake

```bash
convert/convert_nar.py --ar-gguf yue2-ar-q8_0.gguf \
    --out tests/out/lora/yue2-nar-lora-1.0.gguf --lora LORA.safetensors:1.0
nice -n 19 ./build_lora/yue2-nar $NAR -m tests/out/lora/yue2-nar-lora-1.0.gguf -o tests/out/lora/baked1.npy
../venv_yue2/bin/python tests/lora_bake_diff.py \
    yue2-nar-f16.gguf tests/out/lora/yue2-nar-lora-1.0.gguf LORA.safetensors:1.0
```

Each baked GGUF is 2.9 GB; bake one, measure, delete it, then the next.

### 6 — the errors

```bash
../venv_yue2/bin/python tests/lora_bad_safetensors.py tests/out/lora/bad
while IFS=$'\t' read -r f e; do
    msg=$(nice -n 19 ./build_lora/yue2-nar $NAR --nar-lora "$f" -o /dev/null 2>&1 >/dev/null | grep '^error:')
    case "$msg" in *"$e"*) echo "PASS $f";; *) echo "FAIL $f: $msg";; esac
done < tests/out/lora/bad/cases.tsv
```

| case | message |
|---|---|
| missing file | `cannot open '…'` |
| 3 bytes | `is not a safetensors file (too short for the header length)` |
| header length 2^40 | `header length 1099511627776 does not fit in 524517 bytes` |
| `{not json` | `the safetensors header of '…' is not a JSON object` |
| file truncated 4 KiB | `is shorter than its header claims ('…lora_B' ends at 524517 of 520421 bytes)` |
| `I64` tensor | `has dtype I64; only F32, F16 and BF16 are supported` |
| `…z_proj.lora_A` | `key '…' maps to no NAR tensor` |
| `diffusion_model.…` | `is the fused ComfyUI layout (key '…'); use the plain adapter file instead` |
| `lora_A` alone | `has a lora_A for 'blk.0.nar_attn_q.weight' but no lora_B` |
| `lora_B` alone | `… but no lora_A` |
| `B` rank 16, `A` rank 32 | `rank mismatch for '…' (lora_A 32, lora_B 16)` |
| `A` `[32, 999]` | `is a 2048 x 999 adapter, but the base is 2048 x 2048` |
| `B` `[999, 32]` | `is a 999 x 2048 adapter, but the base is 2048 x 2048` |
| `vae2llm.weight [2048, 63]` | `the replacement for '…' has shape [2048, 63], not the base's [2048, 64]` |
| NaN in `lora_A` | `holds a non-finite value` |
| `LORA.safetensors:nan` | `--nar-lora strength must be finite` |
| Q8_0 target tensor | `GGUF tensor 'blk.0.nar_attn_q.weight' is q8_0; only F16 and F32 targets can be merged` |

`tests/out/lora/bad/one_tensor.safetensors` is the matching non-error: one
projection of one layer, no replacements, accepted (`1 patched, 0 replaced`).

### Other paths checked

| path | result |
|---|---|
| BF16 adapter | loads; 63.86 dB against the same adapter in F32 (bf16 has 8 mantissa bits) |
| real fused ComfyUI file | refused with the layout message |
| `--weights f32` + adapter | merges, no narrowing (`196 patched, 4 replaced`) |
| `yue2 batch --nar-lora …:0.5` | merges once per job; `config.json` grows `nar_lora: [{file, strength, sha256}]`, sha matching the adapter file |
| adapter that names 196 + 4 tensors | `196 patched, 4 replaced` — the full 28-layer × 7-projection set plus both replacements |

## Deviations

1. **`yue2 batch` merges once per job, not once per batch.** SPEC §4 says the
   NAR loads once for a batch; it does not today — `run_batch()` calls
   `run_nar()` inside its per-job loop, so the GGUF and the adapters are read
   again for every song. Nothing about the merge makes that worse than it
   already was (~5 s a job), and hoisting the model load out of the loop is a
   change to stage 5, not to this one. Left alone; flagged here.
2. **Acceptance 3's weight count compares the bake against a model of the
   runtime, not against the runtime itself.** The merged weights live in device
   memory and nothing reads them back, so `tests/lora_bake_diff.py` recomputes
   the runtime path in numpy (widen the stock F16 → add `s·B@A` in F32 → narrow)
   and diffs that against the baked GGUF. What pins the C++ merge is items 1, 2
   and 5 (bit-exactness and stacking) plus item 3's latent PSNR, which is a
   genuine two-binary comparison.
3. **The count is tens of thousands, not "a handful".** The two merges differ in
   more than summation order: the bake's base is the F32 checkpoint weight, the
   runtime's base is that weight already rounded to F16 (SPEC §5 asks for "one
   narrowing", which is what makes the bake the better of the two). Any value
   whose delta lands near an F16 rounding boundary can then fall either way.
   Every difference is exactly one F16 step, it touches 0.002–0.003 % of
   1.4 G weights, and the latent still agrees to ≥ 90 dB.
4. **"A key that maps to nothing" is two checks.** A key matching no name
   pattern is rejected while the header is parsed; a key whose GGUF tensor is
   absent from the loaded model is rejected in `bind()`, with its own message.
   Both are before any weight reaches the device.
5. **`convert_nar.py` keeps the checkpoint's `yue2.source_sha256`** in a baked
   GGUF — it is still the same model pair, and `yue2-nar` refuses an AR/NAR pair
   whose hashes differ. The adapters are recorded in a separate
   `yue2nar.lora` string (file name, strength, SHA-256), the same shape
   `config.json` uses.
