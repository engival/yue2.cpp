# STATUS — `yue2-nar` performance on the Intel Arc B70 (Vulkan device 1)

**Result: the full `alley_swing_s1` song goes from 251.8 s to 92.0 s on the Arc
(2.74×) at 32.71 dB against `tests/out/nar_full_cpu_latent.npy` — above the
~29 dB floor, and better than the AMD's accepted fast path (29.27 dB).**

## 1. What changed

Two edits to `src/yue2-nar.cpp`, nothing in the submodule.

1. **`--kv-f16` (new flag, default off).** Stores the per-chunk K/V cache as F16.
   The point is not the halved cache: with an F32 cache the two attention
   matmuls are `F32 × F32`, which ggml-vulkan runs at 8.1 / 9.6 TFLOP/s on the
   Arc even with fp16 enabled, while the F16 weight GEMMs beside them get
   32–47. F16 K/V moves them onto the same coopmat pipeline: **13.9 / 26.9**.
   `ggml_cpy` does the narrowing, so the graph is otherwise untouched.
2. **Balanced query tiles.** `--query-chunk` cut full tiles plus a remainder, so
   `N = 4111` gave four 1024-row tiles and a 15-row tail running at 0.3 TFLOP/s
   — 4 % of every evaluation. The rows are now spread over `ceil(N/query_chunk)`
   equal tiles; identical output (softmax is per-row), and it is why the default
   path is 251.8 s instead of 260.3 s.

`--query-chunk 8192` (one tile, no tiling) is fastest at `N = 4111` and costs
2.8 GB of graph arena — fine on 32 GB, hence a flag rather than a new default.

## 2. Timing — full song, Arc, 4109 frames (164.4 s audio), `--steps 32`

| variant | prefill (6659 tok) | s / eval | s / ODE step | total | SNR vs f32 CPU ref |
|---|---:|---:|---:|---:|---:|
| default, before this work (STATUS_NAR §7) | 8.18 s | 3.94 | 7.88 | 260.3 s | 65.96 dB |
| default (balanced tiles) | 6.90 s | 3.82 | 7.65 | **251.8 s** | 65.96 dB |
| `--flash-attn`, before this work | 6.37 s | 4.92 | 9.84 | 321.6 s | — |
| `--kv-f16 --query-chunk 8192` | 6.80 s | 3.68 | 7.36 | 242.5 s | 62.59 dB |
| `--kv-f16 --vk-f16-matmul` | 3.62 s | 1.518 | 3.04 | 100.8 s | 32.71 dB |
| **`--kv-f16 --vk-f16-matmul --query-chunk 8192`** | 3.63 s | **1.377** | **2.75** | **92.0 s** | **32.71 dB** |
| *(AMD 7900 XTX `--flash-attn --vk-f16-matmul`, coordinator)* | 1.20 s | 0.79 | 1.59 | 52.0 s | 29.27 dB |
| *(torch ROCm bf16, the song itself)* | — | — | — | 46.4 s | 35.01 dB |

Memory, best variant: weights 4131.5 + 2803.5 MiB, KV 1179 MiB (was 2389),
velocity graph 2798.7 MiB → **≈ 10.9 GB of 32 GB**. With the default
`--query-chunk 1024` the graph is 611.7 MiB → ≈ 8.7 GB.

## 3. Per-evaluation profile, `GGML_VK_PERF_LOGGER=1`, `N = 4111`, `KV = 10770`

Arc, best variant (1.376 s/eval) next to the coordinator's AMD run (794 ms/eval):

| op | Arc, `--kv-f16 --vk-f16-matmul --query-chunk 8192` | Arc, `--flash-attn --vk-f16-matmul` | AMD, `--flash-attn --vk-f16-matmul` |
|---|---:|---:|---:|
| QK<sup>T</sup> `mul_mat` (28×) | 365 ms — 13.9 TFLOP/s | — | — |
| `soft_max` (28×) | 377 ms | — | — |
| P·V `mul_mat` (28×) | 188 ms — 26.9 TFLOP/s | — | — |
| `flash_attn_ext` (28×) | — | **3898 ms — 2.6 TFLOP/s** | 527 ms — 19.3 TFLOP/s |
| the four F16 weight GEMMs | 285 ms — 32–47 TFLOP/s | 291 ms — 31–45 TFLOP/s | 208 ms — 54–57 TFLOP/s |
| rms_norm(+rope), add, mul, silu, cpy, cont | 161 ms | 178 ms | 60 ms |
| **total** | **1376 ms** | **4338 ms** | **794 ms** |

Attention is 68 % of the Arc evaluation, same as on the AMD (66 %) — but the Arc
reaches it through materialized scores, because its FA kernel is 7.4× slower
than the AMD's and 2.1× slower than the plain matmul path it replaces.

## 4. What did not help

| tried | result |
|---|---|
| `--flash-attn` on the Arc (lever 2) | 2.6–3.2 TFLOP/s at every shape measured; 4.34 s/eval vs 1.38. Never competitive. |
| coopmat2 FA | **Not available.** ANV advertises `VK_NV_cooperative_matrix2`, but ggml-vulkan prints `matrix cores: KHR_coopmat` for the Arc, i.e. one of the seven `VkPhysicalDeviceCooperativeMatrix2FeaturesNV` bits it requires is unset. The build does have `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT`. So FA falls to scalar/coopmat1. |
| forcing FA to scalar (`GGML_VK_DISABLE_COOPMAT=1 --flash-attn`) | 2.77 vs 2.85 TFLOP/s — coopmat1 is not being picked, or is no better. The FA slowness is the kernel, not the path choice. |
| f16 accumulate for FA | Already active: `op_params[3]` is `GGML_PREC_DEFAULT`, so `f32acc = !device->fp16`, false under `--vk-f16-matmul`. No lever here. |
| `GGML_VK_DISABLE_COOPMAT=1` on the best variant | 4.449 s/eval (3.2× worse). Coopmat is the whole f16 win. |
| `--query-chunk 512` | 1.721 s/eval (slower than 1024 and 8192). |
| bf16 (`VK_KHR_shader_bfloat16`, lever 4) | Not tested: ggml-vulkan only takes the bf16 path for a `GGML_TYPE_BF16` **src0**, so it would need a bf16 NAR GGUF and a converter change, not a runtime flag. |
| per-eval weight widening (lever 3) | None exists — `CONT`+`CPY` total 11 ms of 1376, all of it the Q permute and the K/V writes. |

## 5. Accuracy — every acceptance check

| check | bar | before | after |
|---|---|---|---|
| short golden, `--device cpu --weights f32` (§8 cmd 1) | ≥ 70 dB | 98.12 | **98.12** |
| — `nar_ar_kv_l0` / `l27` / `x_in_step0` / `v_step0` | §5.1 | 105.32 / 104.82 / 139.09 / 107.04 | **identical** |
| short golden, Arc, default F32 path (§8 cmd 2) | ≥ 50 dB | 85.10 | **85.10** |
| multi-chunk, Arc, default (§8 cmd 3) | ≥ 45 dB | 81.74 | **81.74** |
| full song, Arc, default (§8 cmd 4) | ≥ 45 dB | 65.96 | **65.96** |
| full song, Arc, `--kv-f16 --vk-f16-matmul --query-chunk 8192` | ~29 dB floor | — | **32.71** |
| full song, Arc, `--kv-f16 --query-chunk 8192` | — | — | 62.59 |
| short golden, Arc, `--kv-f16` | — | — | 57.04 |
| short golden, Arc, `--kv-f16 --vk-f16-matmul` | — | — | 40.04 |
| short golden, Arc, `--vk-f16-matmul` (no `--kv-f16`) | — | — | 39.89 |
| multi-chunk, Arc, `--kv-f16 --vk-f16-matmul` | — | — | 34.41 |

`--kv-f16` on its own costs ~28 dB on the short golden (85.10 → 57.04) and
3.4 dB on the full song (65.96 → 62.59); the `--vk-f16-matmul` staging is what
takes it to ~33 dB, and it costs the same with or without `--kv-f16` (39.89 vs
40.04 on the short golden). The two together are one operating point.

## 6. The one lever left, and why it is a ggml change

The Arc spends 377 ms/eval in `soft_max` plus ~150 ms writing and re-reading the
`[10770, 4111, 16]` F32 score tensor — the traffic flash attention exists to
remove. On the AMD that trade wins (527 ms of FA for ~930 ms of matmul+softmax);
on the Arc ggml-vulkan's FA kernel does the same work at 2.6–3.2 TFLOP/s against
13.9–26.9, so it loses. Closing on the AMD's 52 s means fixing ggml-vulkan's FA
for Battlemage — `get_fa_tuning_params_scalar` sets `disable_subgroups = true`
and halves `block_rows` for `VK_VENDOR_ID_INTEL`, tuned on Xe1, not BMG — or
getting coopmat2 enabled on ANV. Out of scope here; worth an upstream issue with
the shapes in §3.

## 7. Exact command lines

```sh
V=../venv_yue2/bin/python
A=../songs/ref_song/out/alley_swing_s1   # read-only
snr() { $V -c "
import sys, numpy as np
a=np.load(sys.argv[1]).astype(np.float64); b=np.load(sys.argv[2]).astype(np.float64)
d=a-b; print('snr %.2f dB  max|d| %.4e' % (10*np.log10(np.mean(b**2)/np.mean(d**2)), np.max(np.abs(d))))
" "$1" "$2"; }

cmake --build build_nar -j --target yue2-nar

# THE FAST PATH — full song on the Arc, 92.0 s, 32.71 dB
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix $A/prefix.npy --codec $A/semantic.npy \
	--noise tests/out/nar_noise_alley_swing_s1.npy --steps 32 \
	--kv-f16 --vk-f16-matmul --query-chunk 8192 \
	-o tests/out/nar_full_A_kvf16_f16mm_qc8192.npy --device vulkan --gpu 1
snr tests/out/nar_full_A_kvf16_f16mm_qc8192.npy tests/out/nar_full_cpu_latent.npy

# accuracy path on the Arc, 242.5 s, 62.59 dB   (drop --kv-f16 for 251.8 s / 65.96 dB)
build_nar/yue2-nar ... --steps 32 --kv-f16 --query-chunk 8192 \
	-o tests/out/nar_full_B_kvf16_qc8192.npy --device vulkan --gpu 1

# regressions (all unchanged): STATUS_NAR.md §8 commands 1, 2, 3, 4 verbatim.

# per-op profile of one velocity evaluation (--steps 1 = 2 evals; read the last block)
GGML_VK_PERF_LOGGER=1 build_nar/yue2-nar ... --steps 1 \
	--kv-f16 --vk-f16-matmul --query-chunk 8192 -o /tmp/perf.npy --device vulkan --gpu 1

# FOR THE COORDINATOR — the same best variant on the AMD, device 0, never run here:
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix $A/prefix.npy --codec $A/semantic.npy \
	--noise tests/out/nar_noise_alley_swing_s1.npy --steps 32 \
	--kv-f16 --vk-f16-matmul --query-chunk 8192 \
	-o tests/out/nar_full_vk0_kvf16.npy --device vulkan --gpu 0
# and the same with --flash-attn instead of --query-chunk 8192, to see whether
# F16 K/V also speeds up the AMD's FA kernel (it was 19.3 TFLOP/s on F32 K/V).
```

## 8. Coordinator's AMD comparison (device 0 idle, 2026-09-11, build_nar/yue2-nar)

Full alley_swing_s1, AR F16 + NAR F16, SNR vs `tests/out/nar_full_cpu_latent.npy`:

| card | flags | s / ODE step | total | SNR |
|---|---|---:|---:|---:|
| AMD 7900 XTX | `--kv-f16 --vk-f16-matmul --flash-attn` | 1.50 | **49.2 s** | 29.27 dB |
| AMD 7900 XTX | `--kv-f16 --vk-f16-matmul --query-chunk 8192` | 1.92 | 63.4 s | 24.00 dB |
| AMD 7900 XTX | `--vk-f16-matmul --flash-attn` (previous default) | 1.58 | 52.0 s | 29.27 dB |
| Intel Arc B70 | `--kv-f16 --vk-f16-matmul --query-chunk 8192` | 2.88 | 92.0 s | 32.71 dB |
| torch ROCm bf16 | — | — | 46.4 s | 35.01 dB |

F16 K/V is a 5 % win on the AMD's flash-attention path at identical accuracy, so
it becomes the AMD default too. The unfused path on the AMD is both slower and
less accurate (24 dB) than on the Arc, so the two cards keep different flag
sets: AMD = `--flash-attn`, Arc = `--query-chunk 8192`. The remaining Arc lever
is ggml-vulkan's Intel FA tuning (§5), a submodule change.
