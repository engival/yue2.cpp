# STATUS — stage 4, one `yue2` binary (SPEC_SINGLE.md)

**Done; all six acceptance items pass.** Every existing golden still passes
through `yue2 ar|nar|vae` and through the unchanged `yue2-ar`/`yue2-nar`/`yue2-vae`.

Each `main()` became `parse_<stage>_args()` + `run_<stage>()` in
`src/stage_{ar,nar,vae}.cpp`; the old targets are one-line mains over them and
`src/yue2.cpp` dispatches `argv[1]`. Code was moved, not rewritten — with rename
detection the stage files show 282/239/165 changed lines of 1121/1203/738. Each
stage body sits in an anonymous namespace: all three define
`Config`/`Model`/`Builder`/`Graph`, `yue2` links all three, and at global scope
their inline member functions would have collided silently. `src/common/` holds
only what was literally duplicated, plus `noise.hpp` and `flac.hpp`.

## 1. Goldens through the new subcommands

| check | bar | measured | STATUS ref |
|---|---|---|---|
| VAE 48-frame, CPU F32 | ≥ 60 dB | **121.95 dB**, max 1.50e-06 | 121.95 |
| AR `--dump-logits`, CPU F16 | argmax 55 | **55**, max\|Δ\| 0.00167 | 55 / 0.00167 |
| AR `--greedy --max-abc 32` | identical | **IDENTICAL (32/32)** | identical |
| NAR short golden, CPU `--weights f32` | ≥ 70 dB | **98.12 dB** | 98.12 |
| — `nar_ar_kv_l0`/`l27`/`x_in`/`x_l0`/`v_step0` | — | 105.32 / 104.82 / 139.09 / 132.23 / 107.04 | identical |
| NAR short golden, Arc | ≥ 50 dB | **85.10 dB** | 85.10 |
| — same five dumps | — | 99.65 / 97.67 / 140.67 / 117.41 / 97.95 | identical |
| NAR multi-chunk, Arc | ≥ 45 dB | **81.74 dB**, ranges (0,470)/(470,512) | 81.74 |

`yue2 vae` and `yue2-vae` wrote byte-identical `.npy` from the same input.

## 2. Equivalence — bit-identical

`yue2 song` vs `yue2 ar` → `yue2 nar` → `yue2 vae` by hand (same request, same
`--noise`, Arc, same flags): `prefix.npy`, `semantic.npy`, `abc_tokens.npy`,
`score.abc`, **`latent.npy`** and the output FLAC all **IDENTICAL** (`cmp`),
re-verified against the final binary. Those four AR artifacts also match the
reference `e2e_arc` run's SHA-256s exactly.

## 3. FLAC — bit-identical PCM; the spec's scale factor is wrong

Reference latent → `yue2 vae` → `.flac` vs `e2e_arc/audio.flac`: 15 018 112
samples, **max \|Δ\| 0 LSB, 100.0000 % exactly equal**.

| scale | max \|Δ\| | exactly equal |
|---|---|---|
| 8388607 (SPEC §2.5) | 1 LSB | 88.00 % |
| **8388608 + clip** (what libsndfile does; `ref/x` is exactly 8388608.0 on every non-zero sample) | **0 LSB** | **100.00 %** |

## 4. Noise

`tests/golden/noise_seed831001_16x64.npy`, sha256 `d68db01c…aefb26`, recorded in
`tests/golden/noise_meta.json`; byte-identical on re-run and after a rebuild.
128 000 draws: mean 0.0033, sd 1.00049, skew 0.0097, excess kurtosis −0.0031,
KS vs N(0,1) p = 0.49. `std::mt19937_64` + Box–Muller over a hand-rolled 53-bit
open-interval mapping; no `<random>` distribution (all implementation-defined).
libm's `sqrt/log/cos/sin` are the only cross-platform dependency left. The `.npy`
is untracked, like every existing golden (`.gitignore` has `*.npy`).

## 5. Q8_0 prefix — one `--ar` file serves both stages

Full song, Arc fast path, NAR prefill from each AR GGUF, vs `nar_full_cpu_latent.npy`:

| AR half for the NAR prefill | SNR | weights |
|---|---|---|
| F16 (STATUS_NAR_PERF §2) | 32.71 dB | 4131.5 MiB |
| **Q8_0** | **33.44 dB** | **2195.1 MiB** |

Above the 29 dB bar and marginally *better* than F16 — so no `--ar-prefill`, and
`yue2-ar-f16.gguf` is no longer needed by `song` (1.9 GB less VRAM). STATUS_NAR
§6's "Q8_0 costs ~55 dB" was the short golden on the exact-F32 path; on the fast
path the ~33 dB fp16-staging error already dominates it.

## 6. Timing — Arc (device 1), warm

Default path, 3911 frames = 156.44 s of audio, 32 steps:

| stage | seconds | detail |
|---|---:|---|
| AR load | 1.10 | Q8_0, libllama, Vulkan1 |
| abc | 14.95 | 1949 tokens, 130.3 tok/s |
| semantic | 36.84 | 3912 tokens, 106.2 tok/s |
| NAR load | 3.13 | AR 2195.1 MiB + NAR 2803.5 MiB |
| NAR prefill | 3.40 | 6517 tokens, 1919 tok/s |
| NAR ODE | 81.20 | 64 evals, 1.269 s/eval |
| VAE (child) | 13.83 | load 0.19 + 13.24 decode, 16 tiles |
| **e2e** | **154.7** | re-runs 155.7 / 158.7 |
| *`yue2_gen.py --gpu 1` (docs/YUE2.md)* | *242.3* | *incl. 86.7 s of first-run shader compile inside the AR prefill* |
| `--nar-f32` (a **different** song, dev. 3): 5243 frames / 209.7 s audio | 455.7 | abc 20.4, semantic 52.7, NAR 362.7 (5.48 s/eval), VAE 18.6 |
| `--device cpu`, short request, `--steps 1`: 6301 frames / 252.0 s audio | 1902.7 | abc 91.2 (14.6 t/s), semantic 856.6, NAR 800.4 (prefill 296.6 + 2 evals), VAE 153.6, `.wav` out |

Not an arithmetic win: the stages are the same code, and the reference's 242.3 s
contained an 86.7 s one-time pipeline compile — warm against warm both land at
~155 s. Stage 4 removes python, torch, the temp files and the subprocess
round-trips, and drops the F16 AR half.

Peak VRAM, Arc. The device column is the drop in ggml's free-memory query
(`llama-cli --list-devices`, 23 samples/s) below a 29 354 MiB-free baseline;
approximate, and ~10 % under the sum of the buffers ggml reports.

| stage | device peak | ggml buffers |
|---|---:|---|
| AR | 3457 MiB | Q8_0 weights ≈ 2195 + KV (13 761 ctx) + compute |
| NAR | 7918 MiB | 2195.1 + 2803.5 weights, KV 1141.8, velocity graph 2582.7 = 8723 |
| VAE (child) | 3531 MiB | weights 253.1 + im2col arena (288-frame tiles) |

## Deviations and drops

| # | what | why |
|---|---|---|
| 1 | FLAC scale 8388608, not the spec's 8388607 | §3 — measured; it is what makes the PCM exact |
| 2 | the AR's `yue2-ar: error: ` is now the shared `error: ` | one `die()` in `common/util.hpp` |
| 3 | **`--nar-f32` changes the song, not just the NAR** | one Vulkan device init per process, so `GGML_VK_DISABLE_F16`/`_COOPMAT` must be set before the *AR* loads or the NAR silently keeps fp16 (it was set inside `run_nar`; fixed in `run_song`). The AR's logits then shift and sampling diverges — 5243 frames instead of 3911 for the same request and seed. Exempting the AR would need it in a child process too |
| 4 | the auto untiled query chunk is now size-aware | untiled scores are one `S·N·16·4` buffer: 2.6 GiB at 3911 frames, **4.3 GiB at 5243**, where the Vulkan allocation fails outright. `song` falls back to the default 1024 tiling above a 3 GiB estimate. A pre-existing `yue2-nar` limit that automatic flag selection made reachable |
| 5 | `run_song` refuses to start if `/proc/self/exe` is gone | a rebuild mid-run would otherwise fail at `execv` three minutes later (observed) |
| 6 | progress lines stay on stdout | `yue2_gen.py` had them on stderr only because it redirected each subprocess; `[gen]`/`[done]` are on stdout as before |
| 7 | `ar_request.json` is the `--request` file verbatim | `yue2_gen.py` rewrote a `{style, lyrics, cot, seed}`; `request.json` beside it is already the canonical form |
| 8 | `config.json` drops `overrides`, `cfg_negative`, `backend: torch-eager`, `quantization`, `model_dtype`, `vae_dtype`, `device: cuda:0`, `memory_budget_gib`, `offload_ar`, `runtime_sha256`, `decoder_release`, `validation_status`, `generation.abc`/`.semantic` | torch-only, or `yue2-ar` constants already in `plan.json`. Adds `seed`, the three GGUF paths, `device`, `card`, `nar_precision` |
| 9 | `result.json` drops `identity` and `weights` | safetensors hashes of an HF checkout this binary never opens. Adds `timing.card`, sorts `artifacts` |
| 10 | `yue2 noise --seed N --frames T -o FILE` added | acceptance 4 needs a command that regenerates the golden |
| 11 | `yue2 nar --seed N` works now | with `--artifacts` and no `--noise` the seed comes from `request.json`; STATUS_NAR deviation 4 closed |
| 12 | the per-op ggml patch stays out of scope | ggml-vulkan honours `GGML_PREC_F32` only for flash-attention, not for `mul_mat` pipeline selection, so the VAE child needs a submodule change to go away |

## Exact commands

```sh
cmake -B build_single -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build_single -j

V=../venv_yue2/bin/python
A=../songs/ref_song/out/alley_swing_s1     # read-only
E=<scratchpad>/e2e_arc                                          # reference artifacts
snr() { $V -c "
import sys, numpy as np
a=np.load(sys.argv[1]).astype(np.float64); b=np.load(sys.argv[2]).astype(np.float64)
d=a-b; print('snr %.2f dB  max|d| %.4e' % (10*np.log10(np.mean(b**2)/np.mean(d**2)), np.max(np.abs(d))))
" "$1" "$2"; }

# --- 1. goldens through the subcommands ---
build_single/yue2 vae -m tests/out/yue2-vae-f32.gguf -i $A/latent.npy -o /dev/null \
	--npy tests/out/single_vae_f48.npy --device cpu --full --frames 48
$V convert/compare.py tests/out/single_vae_f48.npy tests/golden/alley_swing_s1_f48.npy --min-snr 60

build_single/yue2 ar -m tests/out/yue2-ar-f16.gguf --request tests/out/alley_swing_s1_request.json \
	--dump-logits tests/out/single_ar_logits.npy --device cpu --threads 12
build_single/yue2 ar -m tests/out/yue2-ar-f16.gguf --request tests/out/alley_swing_s1_request.json \
	--artifacts tests/out/single_ar_greedy32 --greedy --max-abc 32 --max-semantic 8 \
	--device cpu --threads 12      # abc_tokens.npy == tests/golden/ar_greedy_32.npy

build_single/yue2 nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec tests/golden/nar_codec_ids.npy \
	--noise tests/golden/nar_noise.npy --steps 32 --weights f32 \
	-o tests/out/single_nar_s32_cpu.npy --dump-dir tests/out/single_nar_dump_cpu --device cpu
build_single/yue2 nar ... --device vulkan --gpu 1            # same, Arc (85.10 dB)
build_single/yue2 nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec $A/semantic.npy --frames 512 \
	--noise tests/out/nar_multichunk_noise.npy --steps 2 --context 1200 \
	-o tests/out/single_nar_multichunk_vk1.npy --device vulkan --gpu 1

# --- 2. equivalence (bit-identical latent.npy) ---
build_single/yue2 song --request tests/out/alley_swing_s1_request.json \
	--out tests/out/single_song_arc.flac --artifacts tests/out/single_song_arc \
	--noise tests/out/nar_noise_alley_swing_s1.npy --device vulkan --gpu 1
build_single/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/alley_swing_s1_request.json \
	--artifacts tests/out/single_hand_arc --device vulkan --gpu 1
build_single/yue2 nar --ar yue2-ar-q8_0.gguf -m yue2-nar-f16.gguf \
	--artifacts tests/out/single_hand_arc --noise tests/out/nar_noise_alley_swing_s1.npy \
	-o tests/out/single_hand_arc/latent.npy --steps 32 \
	--kv-f16 --vk-f16-matmul --query-chunk 8192 --device vulkan --gpu 1
build_single/yue2 vae -m yue2-vae-f32.gguf -i tests/out/single_hand_arc/latent.npy \
	-o tests/out/single_hand_arc/song.flac --device vulkan --gpu 1
cmp tests/out/single_song_arc/latent.npy tests/out/single_hand_arc/latent.npy

# --- 3. FLAC ---
build_single/yue2 vae -m tests/out/yue2-vae-f32.gguf -i $E/latent.npy \
	-o tests/out/single_flac_check.flac --device vulkan --gpu 1
$V -c "
import soundfile as sf, numpy as np
a,_ = sf.read('tests/out/single_flac_check.flac', dtype='int32'); a = (a>>8).astype(np.int64)
b,_ = sf.read('$E/audio.flac', dtype='int32'); b = (b>>8).astype(np.int64)
d = np.abs(a-b); print('max %d LSB, exact %.4f %%' % (d.max(), 100.0*(d==0).mean()))"

# --- 4. noise golden ---
build_single/yue2 noise --seed 831001 --frames 16 -o tests/golden/noise_seed831001_16x64.npy
sha256sum tests/golden/noise_seed831001_16x64.npy    # == tests/golden/noise_meta.json

# --- 5. Q8_0 prefix ---
build_single/yue2 nar --ar tests/out/yue2-ar-q8_0.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix $A/prefix.npy --codec $A/semantic.npy \
	--noise tests/out/nar_noise_alley_swing_s1.npy --steps 32 \
	--kv-f16 --vk-f16-matmul --query-chunk 8192 \
	-o tests/out/single_nar_full_q8ar.npy --device vulkan --gpu 1
snr tests/out/single_nar_full_q8ar.npy tests/out/nar_full_cpu_latent.npy    # 33.44 dB

# --- 6. timing / VRAM: the acceptance-2 run, with a free-VRAM poller alongside ---
while true; do ~/bin/llama-cli --list-devices 2>/dev/null | grep Vulkan1; done > vram.log &
# the exact path, which is a different song (deviation 3):
build_single/yue2 song --request tests/out/alley_swing_s1_request.json \
	--out tests/out/single_song_f32.flac --artifacts tests/out/single_song_f32 \
	--nar-f32 --device vulkan --gpu 1
# CPU orchestration, short request, 1 ODE step:
build_single/yue2 song --request tests/out/single_cpu_smoke_request.json \
	--out tests/out/single_song_cpu.wav --artifacts tests/out/single_song_cpu \
	--steps 1 --device cpu
```

## Open questions

1. **Is the fp16-staged NAR audibly different?** 33 dB is the same error class as
   torch's own bf16 run (35 dB), and nobody has A/B'd by ear. Because `--nar-f32`
   also changes the AR (dev. 3), a real A/B means running the NAR twice off one
   fixed token set, not two `song` runs.
2. **`GGML_PREC_F32` for Vulkan `mul_mat`** (dev. 12) would fold the VAE child
   back into the parent and let `--nar-f32` leave the AR alone. Upstream.
3. **`song` has no `--max-abc`/`--max-semantic`**, so the CPU end-to-end run
   (which does pass, §6) costs 32 minutes; there is no cheap version of it.
4. **The caller's batch script still calls `yue2_gen.py`.** The flags line up
   (`--request`, `--out`, `--artifacts`); switching it over is not done here.

## Review fixes

Robustness only; no numbers moved.

| # | fix | verification |
|---|---|---|
| 1 | `resolve_gguf` existence-checks an explicitly given `--ar`/`--nar`/`--vae` | a bad `--vae` now fails in **4 ms**, not after the AR+NAR |
| 2 | temp artifacts dir removed via `atexit` (so `die()` takes it too); stale `--out` unlinked at the start of `run_song`; the VAE encodes to `X.partial` and renames on success | a failing run leaves no `/tmp/yue2song.*` and no stale output; `-o /dev/null` is exempted (renaming over a device node would replace it) and still works |
| 3 | one `parse_seed_arg()` in `common/util.hpp` for all three `--seed` | `abc` / `-1` / `12x` / `2**63` each rejected by name in `song`, `nar`, `ar` |
| 4 | wait status decoded with `WIFSIGNALED`/`WTERMSIG`/`WEXITSTATUS` | — |

Re-run after: noise golden byte-identical (`d68db01c…`), VAE 48-frame 121.95 dB,
and one `yue2 song` on the Arc (155.3 s) whose `latent.npy` and FLAC are
**identical** to `tests/out/single_song_arc/`.

## Addendum 2026-09-13 — the VAE fork is gone

`song` and `batch` decode the VAE **in this process**, on the device and at the
precision the NAR just ran at (fp16-staged by default). The listening test in
`docs/vulkan_burst_investigation.md` (addendum) settled that the exact-F32
decode is a numeric reference, not a render default; exact decoding now lives
only in the standalone `yue2 vae`, whose own default is unchanged.

Superseding the table above: deviation 5 is withdrawn (nothing re-executes
`/proc/self/exe` any more — it is only how `resolve_gguf` finds the GGUFs next
to the binary), and deviation 12 now reads as "the per-op ggml patch would let
`song` offer the exact VAE again", not "would remove the child". Deviation 3
still stands and gets stronger: `--nar-f32` is per-process and now covers the
VAE too.

New rule — **mismatched precision aborts, it never downgrades.** `song`/`batch`
reject `--vk-f16-matmul` / `--no-vk-f16-matmul` and the `request.json` key
`vk_f16_matmul`, exit 1 before any model loads, with an error naming `yue2 vae`.

Consequence for `batch`: a VAE failure is now fatal even under
`--continue-on-error`, exactly like the NAR, because `run_vae` reports by
`die()` rather than an exit code (SPEC_BATCH §4.6, updated).

| check, Arc B70 (`--gpu 1`), 191.5 s song | result |
|---|---|
| `yue2 song` end to end | 195.1 s (abc 8.5, semantic 46.9, NAR 120.4, VAE 17.8) |
| song FLAC vs standalone `yue2 vae --vk-f16-matmul` on the same `latent.npy` | **byte-identical** |
| standalone `yue2 vae` exact vs `--vk-f16-matmul` | 67.08 dB, max abs 3.49e-03 |
| VAE stage, exact vs fp16-staged | 16.2 s vs 17.1 s (exact ~6 % faster on this card) |
| `yue2 noise --seed 831001 --frames 16` vs golden | byte-identical |

| check, `--device cpu` | result |
|---|---|
| `yue2 batch --max-abc 32 --max-semantic 160` end to end | 128.2 s (abc 2.6, semantic 10.6, NAR 110.0, VAE 4.5) |
| that FLAC vs standalone `yue2 vae --device cpu` on its `latent.npy` | **byte-identical** |

CPU numerics are unchanged by the fork removal: there is no fp16 operand
staging on the CPU backend, so the in-process decode is the same arithmetic the
child was doing.

`tests/regress.sh` still scores the standalone exact decoder and is unaffected.
It grew `YUE2_REGRESS_DEVICE` / `YUE2_REGRESS_GPU` (defaults `vulkan` / `0`, the
old hard-coded pair) and now exits 1 with a message instead of 0 when the song
set is missing — an empty run used to read as "everything passed". Against a
CPU-exact reference render of one 191.5 s latent: `--gpu 1` **109.42 dB**,
`--device cpu` **128.57 dB**.
