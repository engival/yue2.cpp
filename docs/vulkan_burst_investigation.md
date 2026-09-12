# The Vulkan error bursts — root cause and fix

Follow-up to the 2026-09-11 full-catalog regression pass (localized error bursts
in several renders at a 64–69 dB floor) and to the "Deviation" section of
`src/STATUS.md`.

**Result: the Vulkan decode was never doing an F32 matmul.** ggml-vulkan's
`mul_mm` shader stages *both* operands as `float16_t` whenever the device
advertises fp16, and the `KHR_coopmat` path is fp16 by construction — even for
an `F32 x F32` `GGML_OP_MUL_MAT`. Every Conv1d in this decoder is
`im2col + mul_mat`, so every convolution was running with fp16 inputs (fp32
accumulate). Disabling both paths fixes it and costs ~3.5 % of wall time.

| song (full, core 256, Vulkan dev 0) | before | after |
|---|---:|---:|
| `crooner_bigband_s1` vs CPU | 58.54 dB | **107.51 dB** |
| `death_metal_s2` vs CPU | 56.03 dB | **105.53 dB** |
| `alley_swing_s1` vs CPU | 64.30 dB | **111.42 dB** |
| 48-frame golden (`--full --frames 48`) | 69.42 dB | **117.13 dB** |
| dev 1 (Intel Arc), `crooner_bigband_s1` vs CPU | 59.65 dB | **102.73 dB** |
| vk core 64 vs vk core 256 (`alley_swing_s1`) | 2.7e-3 max | **0.0, bit-identical** |

CPU is unchanged and still exact (48-frame golden 121.95 dB).

## How it was found

Repro window. The worst `crooner_bigband_s1` burst (global samples
3897453-3901518, t ~= 81 s) sits in tile 7, frames [1792, 2048). It reproduces
in a **64-frame untiled decode** of frames [1998, 2062) — same sample, same
magnitude — so it is data-driven, not a tiling or graph-size artefact:

```
build_vae/yue2-vae -m yue2-vae-f32.gguf -i win64.npy -o /dev/null \
	--npy win64_<dev>.npy --device <cpu|vulkan> --gpu 0 --full
```

| window | vk vs cpu max | vk vs cpu RMS |
|---|---:|---:|
| tile 7 (288 frames) | 3.854e-02 | 6.77e-04 |
| 64-frame window | 3.229e-02 | 1.21e-03 |
| 64-frame window, **Intel Arc** | 3.250e-02 | 1.21e-03 |

The Arc reproduces the burst at the *same sample index* and to three digits of
the same magnitude. Two unrelated GPUs agreeing that closely rules out
split-k/reduction-order noise, which is per-device.

Per-node diff. `--probe` / `--probe-dir DIR` (added to `src/yue2-vae.cpp` for
this investigation) names and keeps every SnakeBeta input/`sin`/output and every
`im2col`/`mul_mat` result, prints `max |v|`, and optionally writes each as
`.npy`. Dumping the same 64-frame window on both backends and diffing node by
node puts the first error at the **very first convolution**, `decoder.layers.0`
(64 -> 2048, k=7, so K = 448), spread uniformly across all 64 time steps — not
a burst at all. Against a float64 reference computed in numpy:

| | max abs | RMS | relative RMS |
|---|---:|---:|---:|
| `decoder.layers.0` im2col, vk vs cpu | **0** | 0 | — |
| `decoder.layers.0` mul_mat, CPU vs float64 | 2.5e-07 | 2.9e-08 | 9.8e-08 |
| `decoder.layers.0` mul_mat, **Vulkan** vs float64 | 3.748e-04 | 5.464e-05 | **1.70e-04** |
| numpy `A.astype(f16) @ W.astype(f16).T` in f32 vs float64 | 3.748e-04 | 5.464e-05 | 1.70e-04 |

The last two rows are not merely the same order of magnitude: the error vectors
have **correlation 1.0000** and identical max/RMS to four digits. The Vulkan
matmul is bit-for-bit "round both operands to fp16, accumulate in f32".

Why it looks like a burst. 1.7e-4 relative is a flat, boring noise input at the
first layer. The decoder then applies ~40 SnakeBeta non-linearities
(`y = x + sin(a x)^2 / b`, local gain `1 + a sin(2 a x) / b`, which exceeds 1 for
much of the input range). The per-node diff shows that noise growing
monotonically layer by layer — 3.5e-4 after layer 1, 4.8e-2 by `layers.2.0`,
5.2e-1 at `layers.3.0` — and then being re-compressed by later layers. Where the
local gain chain happens to be largest for ~85 ms of signal, the output shows a
burst. So the "uniform 64-69 dB floor" and the "localized bursts" were one
phenomenon, not two: same cause, different local amplification. Loudness
correlation was low (0.15-0.22) because the amplifier is the activation pattern,
not the output level.

## Hypotheses tested and rejected

1. **GLSL `sin()` range reduction** (the leading suspect). Rejected outright:
   `--probe` reports `max |x * alpha|` over an entire decode as **11.05**
   (`decoder.layers.3.layers.0`); every other snake is below 5. GLSL `sin()` is
   accurate to ~1 ulp at those magnitudes on both drivers, and the per-node diff
   shows the `.sin` nodes tracking their already-wrong `.sinarg` inputs rather
   than adding error. No 2*pi range reduction was added — it would be pure cost.
2. **The Vulkan snake fusion** (`snake_pattern` in `ggml-vulkan.cpp` fuses
   MUL/SIN/SQR/MUL/ADD into `snake.comp`). Rejected: a `--probe` run marks the
   inner nodes as graph outputs, which disables the fusion, and the decoded
   output is bit-identical to the fused run (max err 0.032288052 either way).
3. **`--im2col f16` / im2col precision.** Rejected: the F32 im2col output is
   bit-identical between CPU and Vulkan.
4. **Tiling / halo / core size.** Rejected earlier and again here: the burst
   reproduces in an untiled 64-frame decode.

## The fix

`GGML_VK_DISABLE_F16=1` alone changes nothing (the coopmat pipeline is still
fp16). `GGML_VK_DISABLE_COOPMAT=1` alone changes nothing (the scalar fallback is
still fp16-staged, since `FLOAT_TYPE` is `float16_t` whenever `device->fp16`).
**Both** are required:

| env, 64-frame window vs CPU | max abs | RMS |
|---|---:|---:|
| (default) | 3.229e-02 | 1.207e-03 |
| `GGML_VK_DISABLE_F16=1` | 3.229e-02 | 1.027e-03 |
| `GGML_VK_DISABLE_COOPMAT=1` | 3.854e-02 | 1.217e-03 |
| `GGML_VK_DISABLE_DOT2=1` | 3.229e-02 | 1.027e-03 |
| **both F16 and COOPMAT** | **1.319e-04** | **4.259e-06** |

There is no in-graph fix: `ggml_mul_mat_set_prec(GGML_PREC_F32)` only selects the
f32 *accumulator* (`ggml_vk_get_mul_mat_mat_f16acc`), which this path already
had; the operand staging is baked into which shader variant was compiled, and
that is chosen from `device->fp16` at backend-init time. So `yue2-vae` sets both
variables itself, before `ggml_backend_load_all()`, with `setenv(..., 0)` so an
explicit value in the environment still wins. `--vk-f16-matmul` restores the old
fast/lossy behaviour for comparisons.

Cost, full song, Vulkan device 0: `crooner_bigband_s1` 8.40 s -> 8.68 s,
`death_metal_s2` 12.42 s -> 12.85 s, `alley_swing_s1` 7.49 s -> 7.75 s (~3.5 %,
still ~21x realtime). On the Arc it was slightly *faster* (16.64 s -> 15.82 s).

Upstream angle: this is arguably a ggml bug — an `F32 x F32` `MUL_MAT` silently
losing ~10 bits of mantissa. It is invisible for LLM inference (weights are
already quantized) and lethal for a 40-layer audio VAE. A useful upstream ask
would be for `GGML_PREC_F32` to also select f32 operand staging.

## Debug tooling left in place

`src/yue2-vae.cpp` keeps two flags:

- `--probe` — print `name`, `ne`, `max |v|` for every SnakeBeta input / `sin` /
  output and every `im2col` / `mul_mat` result (tensors above 16M elements are
  skipped so a probe run still fits in VRAM).
- `--probe-dir DIR` — the same, plus each tensor written as `DIR/<name>.npy`
  (shape `[ne1, ne0]`). ~1.2 GB for a 64-frame window. Diff two such dirs to
  attribute a backend difference to a node.

Probing marks intermediates as graph outputs, which disables op fusion; on this
graph that was verified not to change the result.
