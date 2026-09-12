# STATUS — stage 3, `src/yue2-nar.cpp` (NAR flow matching)

**Done and passing.** The NAR half of YuE2-3B (AR prefill + midpoint-ODE flow
matching, semantic tokens → `[T, 64]` VAE latent) runs in raw ggml, on CPU and on
Vulkan, and matches the torch CPU goldens with margin on every SPEC_NAR §5.1
check. The full `alley_swing_s1` song reproduces the torch CPU reference at
**65.96 dB** and decodes to audio that is sample-aligned with `audio.flac`.

**Not done: the §5.4 timing table.** Device 0 was in use by the user (two GPU
resets during the window), so every device-0 number below is contaminated and is
recorded as such, not as timing. The coordinator owns re-running §5.4 on an idle
card. Commands are in §7.

Files touched: `src/yue2-nar.cpp` (new), the one `add_executable(yue2-nar …)`
block in `CMakeLists.txt`, this file. Nothing else. Built only in `build_nar/`.

---

## 1. Build

```
cmake -B build_nar -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build_nar -j --target yue2-nar
```

Links `ggml` only (no libllama), per SPEC_NAR §4.1. Clean at `-Wall -Wextra
-Wshadow`; the only warning is llama.cpp's own `ggml-backend.h`
(`ggml_backend_graph_copy` hides its constructor), the same one `yue2-vae` and
`yue2-ar` emit.

`GGML_VULKAN_CHECK_RESULTS=ON` was used temporarily in `build_nar/` to find the
bug in §4.2 and has been turned back **OFF**; the cache is otherwise unchanged.

## 2. Regression: the other two targets still build and pass

Both were rebuilt **in `build_nar/`** after the CMake change and re-run against
their own goldens:

| check | result | matches `src/STATUS*.md` |
|---|---|---|
| `yue2-vae` CPU F32 `--full --frames 48` vs `tests/golden/alley_swing_s1_f48.npy` | max 1.50e-06, **121.95 dB**, PASS (≥ 60) | yes (121.95) |
| `yue2-ar` CPU F16 `--dump-logits` vs `tests/golden/ar_last_logits_f32.npy` | argmax **55 = 55**, max\|Δ\| **0.00167** | yes (55 / 0.00167) |

## 3. Acceptance — SPEC_NAR §5.1, short golden (385 AR tokens, N = 130)

SNR is `10·log10(mean(golden²)/mean((ours−golden)²))` over the whole tensor —
`convert/compare.py` only accepts audio-shaped arrays, so these used a 5-line
numpy equivalent (§7).

**CPU (`--device cpu --weights f32`, see deviation 1):**

| check | required | measured | max\|Δ\| |
|---|---|---|---|
| `nar_ar_kv_l0` | ≥ 100 dB | **105.32** | 1.18e-04 |
| `nar_ar_kv_l27` | ≥ 85 dB | **104.82** | 9.98e-04 |
| `nar_x_in_step0` | ≥ 95 dB | **139.09** | 1.43e-06 |
| `nar_x_l0_step0` | — | 132.23 | 4.58e-05 |
| `nar_v_step0` | ≥ 85 dB | **107.04** | 5.03e-05 |
| `nar_latent_s2` | ≥ 80 dB | **97.29** | 1.95e-04 |
| **`nar_latent_s32`** | **≥ 70 dB, max < 5e-3** | **98.12** | **1.58e-04** |

**Vulkan device 1 (Intel Arc B70), F16 weights, default (non-FA) path:**

| check | required | measured | max\|Δ\| |
|---|---|---|---|
| `nar_ar_kv_l0` | — | 99.65 | 3.34e-04 |
| `nar_ar_kv_l27` | — | 97.67 | 1.84e-03 |
| `nar_x_in_step0` | — | 140.67 | 1.19e-06 |
| `nar_x_l0_step0` | — | 117.41 | 3.13e-04 |
| `nar_v_step0` | — | 97.95 | 1.30e-04 |
| **`nar_latent_s32`** | **≥ 50 dB, max < 5e-2** | **85.10** | **3.86e-04** |

Vulkan device 0 was not run for acceptance (see the header).

**Independent checks (§5.1, last paragraph):** output finite everywhere; range
`[-5.2956, 3.7706]` (inside ±8; the artifact's is `[-5.69, 4.59]`); two identical
Arc runs are **bit-identical** (`cmp` on the two `.npy` files).

**Timestep schedule:** the 64 `raw` values match `tests/golden/nar_tshift.npy` to
max 1.32e-07 (we dump them as f32 — `nar_tshift_f32.npy` — the golden is f64).

## 4. The two real bugs found (both would have been silent)

### 4.1 gallocr recycles a graph-owned input between evaluations

`nar_pos` / `latent_pos` are written once per chunk and read by every one of the
64 velocity evaluations. As ordinary graph leaves, `ggml_gallocr` frees their
block the moment their last consumer runs and hands it to a later node, so
evaluation 2 read whatever the previous evaluation left there — an out-of-range
`get_rows` index and a `GGML_ABORT`. It only aborted in *some* configurations
(`--dump-dir` changed the allocation pattern enough to hide it), which is exactly
the kind of thing that would have shipped.

Fix: `x_nar`, `t_emb`, `nar_pos` and `latent_pos` are allocated in the persistent
per-chunk `KV` buffer, not in the gallocr arena.

### 4.2 ggml-vulkan reads a strided F32 `src0` as zero past the first batch

The first design followed SPEC_NAR §4.4 literally and attended over a *prefix
view* of the KV cache, `ggml_view_3d(k_cache, 128, S, 8, …)` with `S < smax`, so
`nb[2]` is not tight. On Vulkan (verified on the Arc) the resulting
`ggml_mul_mat` returns **zero for every kv head after the first**: the AR prefill
was correct at layer 0 and garbage from layer 1 on (layer-27 KV came out at
**−1.57 dB**), while CPU was fine. `GGML_VULKAN_CHECK_RESULTS=1` named the node
(`MUL_MAT`, src0 = the strided view, `First error: result=0 correct=-27.3162
i2=2`). Not the mask (a finite −1e30 changed nothing) and not the block size
(128 and 64 behaved identically).

Fix: attention always reads the **whole** cache tensor, which is contiguous, and
the rows a prefill block has not written yet are zeroed at allocation and masked
out (the prefill mask is now `[smax, B]` instead of `[off+B, B]`). The NAR phase
attends over the full `ar_length + N` anyway, so it is unaffected. With the fix,
`GGML_VULKAN_CHECK_RESULTS` reports the whole prefill clean (worst node
`avg_err` ≈ 1.7e-05, all matmuls) and layer-27 KV is 97.67 dB.

Side effect: the prefill's attention cost goes from `O(ar_length²/2)` to
`O(ar_length · smax)`. Measured prefill on the Arc for the full song: 8.2 s for
6659 tokens (814 tok/s) — not a bottleneck. This is worth reporting upstream; it
looks like a genuine ggml-vulkan bug for `F32 × F32` `MUL_MAT` with a
non-tightly-strided `src0`.

## 5. Deviations from SPEC_NAR

1. **`--weights f16|f32` (new flag, default `f16`).** SPEC §4.3 assumes "F16
   weights, F32 activations", but ggml's **CPU** `mul_mat` with an F16 `src0`
   goes through `vec_dot_f16`, which rounds the *activations* to F16 too (the
   same ~1.7e-4 relative error `docs/vulkan_burst_investigation.md` documents for
   Vulkan). Measured cost on CPU, 32-step short golden: **65.31 dB** with F16
   weights versus **98.12 dB** with `--weights f32` — i.e. the spec's own 70 dB
   CPU bar is unreachable with F16 weights on CPU, and layer-0 prefill KV reads
   77.00 dB instead of 105.32. `--weights f32` widens F16 → F32 at load (lossless;
   F16 *is* the BF16 checkpoint). **All CPU acceptance numbers above use
   `--weights f32`; all Vulkan numbers use the F16 default**, where the
   `GGML_VK_DISABLE_F16`/`COOPMAT` block already gives a genuinely-F32 matmul with
   F16 operands read from memory.
2. **Attention reads the whole KV cache, not a `[0, S)` view** (§4.2 above).
   SPEC §4.4's `ggml_view_3d(ar_k[l], 0..off+B)` is not safe on Vulkan.
3. **`--flash-attn` (new flag, default off).** `ggml_flash_attn_ext` over the same
   K/V (V stored un-transposed in that mode). Agrees with the default path to
   **100.86 dB** on the full song and hits 92.54 dB / 97.70 dB on the short
   golden, and cuts the velocity graph's compute buffer from **745 MiB to 257
   MiB** at `N = 4111`. On the Arc it is *slower* (4.92 vs 3.94 s/eval), so it is
   opt-in; whether it wins on RADV is an open question (§6).
4. **`--seed` is not implemented**, as §1.2 permits: it errors with
   `seed-driven noise not implemented; pass --noise`. `--noise` is required.
5. **`--dump-kv-all`** (new debug flag) dumps all 28 layers' prefill KV, not just
   0 and 27. That is how §4.2 was localized; kept, like `yue2-vae`'s `--probe`.
6. **`nar_tshift` is dumped as f32** (`nar_tshift_f32.npy`), because `npy.hpp`
   only writes `<f4`/`<i4`. Compared against the f64 golden after a cast.
7. **`--artifacts DIR` reads `request.json` only to print the seed**, since
   seed-driven noise is not implemented; `--noise` still has to be passed.
8. `nar_cond_end` is not implemented and cannot be set, per §1.1.
9. The full-song **CPU** run of `yue2-nar` was **not** done (§5.3 wants ≥ 65 dB
   there). At ~1.7 s/eval for N = 130 it would be several hours for N = 4111, and
   §5.4 explicitly allows skipping the CPU full song. The short-golden CPU
   numbers plus the Arc full-song result (65.96 dB vs the torch CPU golden) cover
   the same ground.

## 6. Full song, chunking, end-to-end

**Multi-chunk (§5.3, the only test that covers chunking).** Short prefix (256
ids), first 512 codec frames, `--context 1200 --steps 2`. We produce exactly the
ranges and AR lengths the converter recorded in `nar_short_meta.json`:
`(0,470) ar_length 727` and `(470,512) ar_length 299`, output `[512, 64]`.

| run | required | vs `tests/out/nar_multichunk_cpu.npy` |
|---|---|---|
| CPU `--weights f32` | ≥ 60 dB | **88.63 dB**, max 9.12e-04 |
| Vulkan dev 1 (Arc) | ≥ 45 dB | **81.74 dB**, max 2.78e-03 |

**Full song `alley_swing_s1`** (P = 2549, T = 4109, one chunk, ar_length 6659,
N = 4111, S = 10770), Arc, F16, `--steps 32`. Output `[4109, 64]` float32,
range `[-5.6592, 4.6295]` (artifact: `[-5.69, 4.59]`).

| comparison | required | measured |
|---|---|---|
| vs `tests/out/nar_full_cpu_latent.npy` (torch CPU f32, §3.4) | ≥ 45 dB | **65.96 dB**, max 2.80e-02 |
| same, `--flash-attn` | — | 66.04 dB |
| vs `songs/…/latent.npy` (bf16 ROCm) | report only | **34.98 dB** |
| torch CPU golden vs `songs/…/latent.npy` | report only | **35.00 dB** |

That last row is the point: the torch f32 CPU reference is *itself* 35.00 dB from
the bf16 artifact, and we are 34.98 dB from it. The whole gap to `latent.npy` is
the bf16/ROCm run (§3.6), none of it is ours.

**§5.2, latent → VAE → audio** (128 frames, `build/yue2-vae … --device cpu
--full`, both latents through identical settings, compared to the decode of the
golden latent):

| run | required | audio SNR |
|---|---|---|
| CPU | — | **91.55 dB** |
| Vulkan dev 1 (Arc) | ≥ 45 dB | **79.28 dB** |

**Full song audio vs `audio.flac`** (decoded with `yue2-vae`, Vulkan dev 1):
25.41 dB overall — expected, since the latent itself is a bf16-vs-f32 comparison.
I cannot listen, so I checked structure objectively instead: over 328 half-second
windows, **waveform correlation min 0.960 / median 0.999**, 0.5 s envelope
correlation **0.99999**, best cross-correlation lag **0 samples** in a 30–40 s
window, worst window SNR 11.0 dB. Same performance, sample-aligned, no structural
divergence. **A human listen is still worth doing once** —
`tests/out/nar_full_vk1.wav`.

**Report-only accuracy costs** (short golden, 32 steps):

| variant | SNR vs golden |
|---|---|
| CPU, F16 weights (no `--weights f32`) | 65.31 dB |
| Arc, `--ar tests/out/yue2-ar-q8_0.gguf` (Q8_0 AR half) | **30.25 dB** |

Q8_0 on the AR half is expensive here — the AR K/V feeds all 64 velocity
evaluations unchanged, so its error does not average out. Use F16 for the AR half.

## 7. Timing — NOT the §5.4 table

**Coordinator's §5.4 run, device 0 idle (2026-09-11 19:35, `build_nar/yue2-nar`,
full `alley_swing_s1`, 4109 frames = 164.36 s audio, AR F16 + NAR F16):**

| variant | prefill (6659 tok) | s / ODE step | total | vs torch-CPU f32 golden | audio vs song FLAC |
|---|---:|---:|---:|---:|---:|
| default (F32 matmul) | 3.16 s | 3.77 | **123.9 s** | 52.05 dB | 25.1 dB |
| `--flash-attn` | 2.56 s | 3.55 | **116.1 s** | 52.04 dB | — |
| `--flash-attn --vk-f16-matmul` | 1.20 s | 1.59 | **52.0 s** | 29.27 dB | 22.4 dB |
| torch ROCm bf16 (the song itself) | — | — | 46.4 s | 35.01 dB | (reference) |

So the true-F32 path is 2.7× slower than torch and the fp16-staged path is
within 12 % of it. The torch reference is itself only 35 dB from the f32
golden (bf16), so 29 dB is the same class of error, not an outlier; whether
it is audible is a listening test (`out/alley_swing_s1_nar_{f32,f16}.flac`).
The §5.4 "must beat 46.4 s" target is **not met** by either path yet.

Arc (device 1), full song, 4109 frames = 164.36 s of audio:

| variant | prefill (6659 tok) | s / ODE step | total |
|---|---|---|---|
| F16, default path | 8.18 s (814 tok/s) | 7.88 | **260.3 s** |
| F16, `--flash-attn` | 6.37 s | 9.84 | 321.6 s |

Short golden (N = 130) on the Arc: 0.341 s/step, total 11.2 s. CPU
(`--weights f32`): 3.30 s/step, 110.8 s; torch CPU f32 for the same thing was
57.0 s of solve (`nar_short_meta.json`), i.e. we are ~2× slower than torch on
CPU — unsurprising, torch has MKL sgemm and we widen every weight.

**Device-0 numbers are contaminated — do not use them.** During the 23:12–23:21Z
window the user was rendering on device 0 and the card took two resets (two of my
runs died with "context is lost"; the user's VAE was marked guilty). For the
record only, and explicitly *not* as timing: F16 default 124.9 s total,
`--vk-f16-matmul` 69.5 s (and **27.81 dB** against the f32 path — the fp16
staging costs far too much accuracy here to be a real option), `--query-chunk
256` 123.9 s; `--weights f32` and `--query-chunk 4096` both OOM'd the 24 GB card.

Even at face value those miss the "must beat 46.4 s" target by ~2.7×. Honest
reading of why, for whoever re-times it: torch on ROCm ran **bf16 with a fused
flash-attention kernel**, while our default is f32 arithmetic with materialized
`[S, N, 16]` scores. Per velocity evaluation at `N = 4111, S = 10770` the GEMMs
are ~11.6 TFLOP and the attention another ~10.1 TFLOP (SPEC §5.4's "~23 TFLOP of
attention" underestimates this by ~30×; it is ~650 TFLOP over the 64
evaluations). So ~1390 TFLOP total, and 46.4 s implies ~30 TFLOP/s sustained in
f32. The levers, in order: `--flash-attn` (removes the score materialization —
already implemented, needs a RADV measurement), then bf16 or F16 weights with an
f32 accumulator if the accuracy budget allows. I did not chase any of this,
per SPEC §5.4's "a fine place to stop and report".

Memory, Arc, full song: weights 4131 MiB (AR F16) + 2804 MiB (NAR F16), KV 2389
MiB, velocity graph 745 MiB (257 MiB with `--flash-attn`).

## 8. Exact commands

```sh
V=../venv_yue2/bin/python
A=../songs/ref_song/out/alley_swing_s1   # read-only

# SNR helper (convert/compare.py is audio-only)
snr() { $V -c "
import sys, numpy as np
a=np.load(sys.argv[1]).astype(np.float64); b=np.load(sys.argv[2]).astype(np.float64)
d=a-b; print('snr %.2f dB  max|d| %.4e' % (10*np.log10(np.mean(b**2)/np.mean(d**2)), np.max(np.abs(d))))
" "$1" "$2"; }

# 1. short golden, CPU (the primary bar)
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec tests/golden/nar_codec_ids.npy \
	--noise tests/golden/nar_noise.npy --steps 32 --weights f32 \
	-o tests/out/nar_latent_s32_cpu.npy --dump-dir tests/out/nar_dump_cpu --device cpu
snr tests/out/nar_latent_s32_cpu.npy tests/golden/nar_latent_s32.npy
for f in nar_ar_kv_l0 nar_ar_kv_l27 nar_x_in_step0 nar_x_l0_step0 nar_v_step0; do
	snr tests/out/nar_dump_cpu/$f.npy tests/golden/$f.npy; done

# 2. short golden, Arc (swap --gpu 0 for the device-0 acceptance run)
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec tests/golden/nar_codec_ids.npy \
	--noise tests/golden/nar_noise.npy --steps 32 \
	-o tests/out/nar_latent_s32_vk1.npy --dump-dir tests/out/nar_dump_vk1 --device vulkan --gpu 1

# 3. multi-chunk (the only chunking test)
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec $A/semantic.npy --frames 512 \
	--noise tests/out/nar_multichunk_noise.npy --steps 2 --context 1200 \
	-o tests/out/nar_multichunk_vk1.npy --device vulkan --gpu 1
snr tests/out/nar_multichunk_vk1.npy tests/out/nar_multichunk_cpu.npy

# 4. full song (this is the §5.4 timing run; use an IDLE card)
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix $A/prefix.npy --codec $A/semantic.npy \
	--noise tests/out/nar_noise_alley_swing_s1.npy --steps 32 \
	-o tests/out/nar_full_vk1.npy --device vulkan --gpu 1
snr tests/out/nar_full_vk1.npy tests/out/nar_full_cpu_latent.npy

# 5. end-to-end audio
build_nar/yue2-vae -m tests/out/yue2-vae-f32.gguf -i tests/out/nar_latent_s32_cpu.npy \
	-o tests/out/nar_s32_cpu.wav --npy tests/out/nar_s32_cpu_audio.npy --device cpu --full
$V convert/compare.py tests/out/nar_s32_cpu_audio.npy tests/out/nar_s32_golden_s32_audio.npy

# 6. regressions
build_nar/yue2-vae -m tests/out/yue2-vae-f32.gguf -i $A/latent.npy -o /dev/null \
	--npy /tmp/vae_f48.npy --device cpu --full --frames 48
$V convert/compare.py /tmp/vae_f48.npy tests/golden/alley_swing_s1_f48.npy --min-snr 60
build_nar/yue2-ar -m tests/out/yue2-ar-f16.gguf --request tests/out/alley_swing_s1_request.json \
	--artifacts /tmp/ar_regress --dump-logits /tmp/ar_logits.npy --device cpu
```

## 9. Open questions

1. **Does `--flash-attn` win on RADV (device 0)?** It loses on the Arc but that
   is Intel's scalar FA path. If it wins, it should probably become the default:
   it agrees with the soft_max path to 100.86 dB and uses a third of the memory.
2. **The §5.4 timing table is unmeasured** on an idle card, and the target looks
   unreachable without either flash attention or reduced precision (§7).
3. **The ggml-vulkan strided-`src0` `MUL_MAT` bug (§4.2)** deserves a minimal
   repro and an upstream issue. I worked around it rather than reducing it; the
   workaround costs the prefill some attention work and nothing else.
4. **SPEC §4.3's "F16 stored, F32 activations" is not what ggml does on CPU**
   (deviation 1). If stage 4 wants one binary that is accurate on CPU *and* small
   on GPU, it needs the same `--weights` switch, or a bf16 GGUF.
5. **`--query-chunk` no longer does much** on the default path (256 vs 1024
   measured within 3 % on the contaminated device-0 runs) and does nothing at all
   under `--flash-attn`. It is kept as a VRAM lever; SPEC §4.5's memory table
   should be re-derived once the FA default is decided.
6. **`--vk-f16-matmul` is not a usable speed option here** (27.81 dB against the
   f32 path on the full song), unlike in the VAE where it was merely lossy. Worth
   noting in `docs/vulkan_burst_investigation.md` if that doc is ever revised.
