# src/ — STATUS (C++ ggml decoder)

Stage-1 deliverable per SPEC.md §1, §3, §4, §5. **Everything in the spec is
implemented and verified against the torch goldens.**

## Files

| file | what |
|---|---|
| `src/yue2-vae.cpp` | GGUF loader, decoder graph, tiling, CLI |
| `src/npy.hpp` | minimal `.npy` reader + writer (`<f4`, C-order, v1/v2 header) |
| `src/wav.hpp` | 32-bit float RIFF/WAVE writer (`WAVE_FORMAT_IEEE_FLOAT`, 18-byte fmt) |

`CMakeLists.txt` is unchanged — one target, headers picked up via
`target_include_directories`.

## Build

```
cd <repo root>
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
nice cmake --build build -j 6
```

Clean build, no warnings from `src/`. ggml 0.23.0 / commit 7840aaba, Vulkan
backend enabled (`/usr/bin/glslc`, Vulkan 1.4.313). Both devices enumerate:
device 0 = AMD Radeon RX 7900 XTX (RADV NAVI31), device 1 = Intel Arc BMG-G31.
No ROCm/HIP anywhere in the build.

### GPU pinning

`--gpu N` selects by **Vulkan device index**, which is stable across boots
(PCIe slot order) even though the DRM card0/card1 nodes flip: 0 = AMD,
1 = Intel Arc. `GGML_VK_VISIBLE_DEVICES` also works as usual (it filters and
then renumbers, so the Arc becomes `--gpu 0` under it — verify by the printed
backend name, which is always `VulkanN (<device description>)`).

The Arc was smoke-tested on request and **works**, same code path, no changes:

| device | 48-frame golden | full song (core 256) |
|---|---|---|
| `--gpu 0` AMD 7900 XTX | 117.13 dB, 0.113 s | 7.75 s |
| `--gpu 1` Intel Arc B70 | 109.32 dB, 0.176 s | 13.96 s |

Arc full song vs the bit-exact CPU decode: 93.44 dB, 6.31e-04 max, 3.63e-06 RMS.
AMD is ~1.8× faster; both are real-time by a wide margin. (Before the matmul
fix below these read 69.42 / 69.26 dB golden and 64.92 dB Arc full song.)

## Run

```
build/yue2-vae -m tests/out/yue2-vae-f32.gguf \
	-i ../songs/ref_song/out/alley_swing_s1/latent.npy \
	-o tests/out/song.wav --device vulkan --gpu 0
```

Full CLI is SPEC §4: `-m -i -o [--device cpu|vulkan] [--gpu N]
[--threads N] [--core-frames 256] [--halo-frames 16] [--im2col f32|f16]
[--frames N] [--npy out.npy] [--full]`, plus three additions made during the
Vulkan precision investigation (`docs/vulkan_burst_investigation.md`):

- `--vk-f16-matmul` — restore ggml-vulkan's default fp16-staged matmul. Off;
  see "Vulkan matmul precision" below for why the decoder overrides it.
- `--probe` — print `ne` and `max |v|` for every SnakeBeta input / `sin` /
  output and every `im2col` / `mul_mat` result.
- `--probe-dir DIR` — the same, plus each tensor as `DIR/<name>.npy`. Diff two
  such dirs to attribute a CPU-vs-Vulkan difference to a node.

- `.npy` input accepts `[T, 64]` (transposed on load — this is what
  `yue2_gen.py --artifacts` writes and what the test latent is), `[64, T]`,
  and `[1, 64, T]` (both taken as-is). The line it prints says which shape it
  assumed. Anything else exits 1 with the offending shape in the message.
- `--halo-frames < 16` is rejected; bad GGUF / missing tensor / shape mismatch
  all exit non-zero with a specific message.
- `--npy` dumps raw unclamped `[2, samples]` float32; clamping to [-1, 1]
  happens only in the WAV writer.
- Prints backend + device description, load time, per-tile time, total.

## Accuracy — all targets met

`convert/compare.py` (venv_yue2 python), C++ output vs the torch CPU goldens:

| run | max abs err | SNR | verdict |
|---|---|---|---|
| CPU, F32 gguf, F32 im2col, `--full --frames 8` | 5.08e-08 | **99.90 dB** | PASS (≥ 60) |
| CPU, F32 gguf, F32 im2col, `--full --frames 48` | 1.32e-06 | **122.40 dB** | PASS (≥ 60) |
| Vulkan(0), F32 gguf, F32 im2col, `--full --frames 48` | 2.54e-06 | **117.13 dB** | PASS (≥ 60) |
| Vulkan(0), same, but `--vk-f16-matmul` (old behaviour) | 4.27e-04 | 69.42 dB | recorded |
| CPU, F32 gguf, **F16 im2col**, frames 48 | 5.32e-04 | 68.51 dB | recorded, no threshold |
| CPU, **F16 gguf** (F16 Conv1d weights), F32 im2col, frames 48 | 2.87e-04 | 72.65 dB | recorded |

Both spec criteria for the F32 golden test are met: SNR ≥ 60 dB and
max |err| < 1e-3.

## Tiling — exact, and core-size independent on CPU

CPU, 512 frames, F32:

| comparison | max abs diff |
|---|---|
| tiled core=64 vs tiled core=256 | **0.0 (bit-identical)** |
| tiled core=256 vs `--full` untiled | **0.0 (bit-identical)** |

So the crop arithmetic matches torch `decode_tiled` exactly and halo=16 is
genuinely sufficient — a tiled decode reproduces an untiled one bit for bit.

The same holds at full-song scale: **CPU, 4109 frames, core=64 vs core=256 →
max abs diff 0.0, bit-identical.** SPEC §3's `< 1e-4` requirement is met with
room to spare on the CPU path.

And the CPU full-song decode matches the torch reference render:

| CPU core=256 vs `audio.flac` | value |
|---|---|
| SNR | **82.82 dB** |
| max abs err (all samples) | 4.45e-02 |
| max abs err excluding clipped samples | **4.93e-05** |
| RMS err excluding clipped samples | 4.37e-07 |

The 4.45e-02 headline number is entirely an artefact of the reference file:
`audio.flac` is **Signed 24-bit PCM**, and exactly 4 samples of this song
exceed ±1.0 in the unclamped decoder output (peak 1.0446), so the flac clipped
them. Every other sample agrees to 4.9e-05 / 4.4e-07 RMS.

### Vulkan matmul precision — resolved (was: "the < 1e-4 bound does not hold")

**This section previously recorded a deviation: on Vulkan the core-independence
bound was ~3e-3 peak / ~7e-5 RMS instead of the spec's 1e-4, blamed on
shape-dependent matmul/split-k reduction order. That diagnosis was wrong.**

ggml-vulkan's `mul_mm` shader stages *both* operands as `float16_t` whenever the
device advertises fp16, and the `KHR_coopmat` path is fp16 by construction —
even for an `F32 x F32` `GGML_OP_MUL_MAT`. Every Conv1d here is
`im2col + mul_mat`, so every convolution ran with fp16 inputs (f32 accumulate),
costing 1.70e-04 relative RMS at the very first layer; the decoder's ~40
SnakeBeta non-linearities amplified that into the 64-69 dB output floor and the
localized error bursts seen in the full-catalog regression pass. Proof: the
first conv's Vulkan error against a float64 reference matches a numpy
"round both operands to fp16, accumulate in f32" simulation with correlation
**1.0000** and identical max/RMS.

`GGML_PREC_F32` cannot fix it (it only selects the f32 accumulator, which this
path already used) and neither env switch helps alone. The decoder therefore
sets **both** `GGML_VK_DISABLE_F16=1` and `GGML_VK_DISABLE_COOPMAT=1` itself,
before `ggml_backend_load_all()`, with `setenv(..., 0)` so an explicit value in
the environment still wins. `--vk-f16-matmul` opts back out.

Full song (core 256), Vulkan device 0, against the bit-exact CPU decode:

| song | `--vk-f16-matmul` (old) | default (fixed) |
|---|---:|---:|
| `alley_swing_s1` | 64.30 dB | **111.42 dB** |
| `crooner_bigband_s1` | 58.54 dB | **107.51 dB** |
| `death_metal_s2` | 56.03 dB | **105.53 dB** |

And SPEC §3's core-independence requirement now holds on the GPU too — not just
within 1e-4 but exactly:

| comparison (`alley_swing_s1`, 4109 frames) | max abs diff |
|---|---|
| vk core=64 vs vk core=256 | **0.0 (bit-identical)** |
| vk core=256 vs cpu core=256 | 4.48e-05 (111.42 dB) |

Cost: ~3.5 % wall time on the AMD (7.49 → 7.75 s for `alley_swing_s1`); on the
Arc it is slightly *faster*. Full write-up, per-node numbers and the rejected
hypotheses (GLSL `sin()` range reduction, the Vulkan snake fusion, im2col
precision, tiling): `docs/vulkan_burst_investigation.md`.

## Timings (full song, 4109 latent frames = 164.36 s of 48 kHz stereo)

| device | core | wall | per tile | realtime factor |
|---|---|---|---|---|
| Vulkan 0 (7900 XTX) | 256 | **7.49 s** | ~0.46 s × 17 | **21.9×** |
| Vulkan 0 | 64 | 8.62 s | ~0.13 s × 65 | 19.1× |
| CPU (i9-11900K, 16 threads) | 256 | 156.9 s | ~9.2 s × 17 | 1.05× |
| CPU | 64 | 184.3 s | ~2.8 s × 65 | 0.89× |

Untiled 48-frame decode: CPU 1.354 s, Vulkan 0.113 s (12× faster).
Model load is ~0.15 s either way (253 MiB of F32 weights).

Vulkan is ~21× faster than CPU on the full song (7.5 s vs 157 s). On both
devices the 64-frame core is slower in wall time than 256 — per-tile fixed
overhead and the halo being a larger fraction of each tile.

## Implementation notes / spec conformance

- **Conv1d** is `ggml_im2col(..., is_2D=false, dst_type)` + `ggml_mul_mat`,
  copied from `ggml_conv_1d` in `ggml/src/ggml.c` but with the dst type taken
  from `--im2col` (default F32) instead of the hard-coded F16. Bias is added as
  a `[1, Cout, 1]` reshape.
- **ConvTranspose1d** is `ggml_conv_transpose_1d(ctx, w, x, s, 0, 1)` followed
  by a `ggml_view_2d` + `ggml_cont` that crops `ceil(s/2)` samples off each end,
  which is exactly torch's `padding=ceil(s/2)`. `ggml_cont` is needed because
  the following ops want a contiguous tensor. Bias added after the crop.
- **SnakeBeta** is `x + sin(x·alpha)² · inv_beta`. alpha/beta arrive already
  exponentiated from the converter; the loader additionally folds
  `inv_beta = 1/(beta + 1e-9)` at load time so the graph has no divide.
  (SPEC §3 suggested this; it is done.)
- **All weights are widened to F32 on the backend**, including the F16 Conv1d
  weights from `yue2-vae-f16.gguf`. Reason: in the `ggml_conv_1d` formulation
  the weight matrix is `mul_mat`'s **src1**, which must be F32; the precision
  knob is the im2col dst type (src0), which is what `--im2col` controls.
  Consequence: `--im2col f16` is the fast/low-precision path, and loading the
  f16 GGUF costs the same VRAM as the f32 one (253 MiB) — it only saves disk.
- Graph is rebuilt per tile (cheap, ~550 nodes) but the `ggml_gallocr` is
  **cached by tile length**, so all same-size tiles share one compute buffer
  allocation; only the first tile and the short last tile reserve.
- Output length is asserted to be `1920·T − 64` per decode.
- Layer topology, strides and block/residual structure are read from the GGUF
  metadata (`yue2vae.strides` etc.) with the spec values as fallback, and the
  strides→downsampling_ratio product is checked at load.

## Open questions

1. ~~The Vulkan shape-dependence~~ — done, see "Vulkan matmul precision"
   above. Remaining upstream question: should ggml's `GGML_PREC_F32` also
   select f32 *operand staging* in `mul_mm`, so an `F32 x F32` matmul does not
   silently lose ~10 bits of mantissa? Worth an upstream issue.
2. `core=256 halo=16` peaks at roughly 1 GB of im2col scratch on the last
   64-channel layers (F32). It fits the 7900 XTX comfortably, but a larger core
   (torch's 1024) would not; `--im2col f16` halves it if that is ever needed.
3. This song's decode peaks at 1.0446, i.e. the VAE genuinely outputs samples
   outside ±1 (4 of them here). The WAV writer clamps, per spec, and the torch
   reference `audio.flac` clips them too (24-bit PCM). If that ever matters,
   the `--npy` dump is the unclamped truth.
