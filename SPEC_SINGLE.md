# SPEC_SINGLE — stage 4: one `yue2` binary, no Python at runtime

Contract for the C++ agent. Read `README.md`, `src/STATUS_AR.md`, `src/STATUS_NAR.md`,
`src/STATUS_NAR_PERF.md` and the three current mains first. Nothing in this spec
changes any math; it is plumbing, and every existing golden must keep passing.

## 1. Goal

`build/yue2 song --request song.json --out song.flac --artifacts DIR` renders a
whole song in one process: AR (libllama) → NAR (raw ggml) → VAE, with the AR
weights loaded once and reused for the NAR's prefix prefill. `yue2_gen.py`
becomes unnecessary; a caller's batch script will call the binary directly with
the same three options (`--request`, `--out`, `--artifacts`).

The three stages stay individually callable (`yue2 ar|nar|vae`, same flags as
today's `yue2-ar`/`yue2-nar`/`yue2-vae`) so a multi-stage pipeline with npy files
in between keeps working, and so the existing goldens keep meaning something.

## 2. Program design (decided — do not re-litigate, but report if it breaks)

### 2.1 Layout

- `src/stage_ar.cpp/.hpp`, `src/stage_nar.cpp/.hpp`, `src/stage_vae.cpp/.hpp`:
  each current `main()` becomes `int run_<stage>(const <Stage>Params &)` plus a
  `parse_<stage>_args(argc, argv, <Stage>Params &)`. Move code; do not rewrite it.
  The `song` path calls the same `run_*` functions the subcommands call, with
  in-memory handoff where §2.3 says so and files otherwise.
- `src/yue2.cpp`: `main()` dispatches on `argv[1]` ∈ {`ar`, `nar`, `vae`, `song`}.
- `src/common/`: whatever the three stages already duplicate (device selection,
  backend init, npy, timing print, JSON) — dedupe only where the code is
  literally the same; no speculative abstraction.
- CMake: new target `yue2` (links `llama ggml vendor::hash vendor::nlohmann FLAC::FLAC`).
  Keep `yue2-ar`, `yue2-nar`, `yue2-vae` building as one-line mains that call
  the matching `run_*` so README, STATUS files and tests stay valid. Build dir
  for your work: `build_single/`.

### 2.2 The Vulkan exactness trap (one process, one precision)

`yue2-vae` needs true-F32 matmuls: it sets `GGML_VK_DISABLE_F16=1` and
`GGML_VK_DISABLE_COOPMAT=1` before backend init (see
`docs/vulkan_burst_investigation.md`). `yue2-nar`'s fast path needs the opposite
(fp16 staging on, coopmat on). ggml-vulkan reads those variables **once per
device init** (`ggml-vulkan.cpp` ~L6584/L7513) and the only per-op precision hint
it honors is `GGML_PREC_F32` for flash-attention (~L11221), not for `mul_mat`.
So one Vulkan context cannot serve both.

Original decision (superseded): `song` forked itself and re-executed
`{"yue2","vae",...}` so the VAE could have the exact env.

**Decision, 2026-09-13 — the fork is gone.** A listening test settled the
question the fork was protecting: the fp16-staged decode is 58.5 dB from the
exact one over a whole song (~40 dB at the worst burst), the residual is
broadband and uncorrelated with the music, and no listener separated the two
(`docs/vulkan_burst_investigation.md`, addendum). On the AMD the exact path is
not even faster (9.5 s vs 9.2 s). So:

- `song` and `batch` run AR → NAR → VAE in **one** process, at **one** Vulkan
  precision: fp16-staged by default, exact throughout under `--nar-f32` (which
  is a per-process switch and always was — it shifts AR sampling too, so it
  yields a *different song*, not a more precise one).
- Exact-F32 decoding stays available as the standalone `yue2 vae`, which keeps
  its own default (exact unless `--vk-f16-matmul`). That is the route the
  goldens, `tests/regress.sh` and any numeric check take.
- **Mismatched modes abort, they do not downgrade.** `song`/`batch` reject
  `--vk-f16-matmul` / `--no-vk-f16-matmul` (the `vae`/`nar` flag) and the
  `request.json` key `vk_f16_matmul` — exit 1 before any model loads, with an
  error naming `yue2 vae`. Any other precision spelling is an unknown argument.

A per-op ggml patch (honor `GGML_PREC_F32` in Vulkan mul_mat pipeline selection)
would let one process hold both precisions and make the rule unnecessary. It is
a submodule change: still out of scope, see `docs/ROADMAP.md`.

### 2.3 Model residency and handoff in `song`

- AR weights: load once via libllama from `--ar AR.gguf`. Q8_0 is today's decode
  default; F16 is what the NAR prefill was validated against.
- NAR prefill needs the AR weights in raw ggml (per-layer K/V — libllama can't
  expose them, see STATUS_NAR §1). Two loads of the same file are acceptable if
  that's what the existing NAR loader does; measure and report VRAM. Do **not**
  attempt to pull K/V out of libllama's state buffer.
- **Q8_0 prefix question (measure, don't guess):** run the NAR prefill from the
  Q8_0 AR GGUF and compare the full-song latent against
  `tests/out/nar_full_cpu_latent.npy`. If it stays ≥ 29 dB on the Arc fast path
  (F16 got 32.7 dB there), a single `--ar` file serves both stages and
  `yue2-ar-f16.gguf` is no longer needed by `song`. If it drops below, add
  `--ar-prefill F16.gguf` and load both. Report the number either way.
- Prefix/semantic tokens: in memory (vectors), also written to the artifacts dir.
- Noise → NAR: in memory, also written (`nar_noise.npy`).
- Latent → VAE: `DIR/latent.npy`, which the artifacts must contain regardless.
- Free the NAR and AR contexts before the VAE stage.

### 2.4 Noise (own RNG — seed compatibility with torch is NOT a goal)

Seed compatibility with torch is already gone at the AR stage (yue2-ar samples
with `std::mt19937_64`, torch used GPU Philox), so bit-matching torch's `randn`
buys nothing. Implement `std::mt19937_64(seed)` + Box–Muller producing
`float32[n_frames, 64]` row-major, standard normal, in a small
`src/common/noise.hpp` with **no** `std::normal_distribution` (its algorithm is
implementation-defined; ours must be identical on every machine). Same seed →
identical bytes forever; that is the whole requirement. `--noise FILE` keeps
overriding it (goldens use it). Commit a golden: `tests/golden/noise_seed831001_16x64.npy`
generated by the implementation, plus its SHA-256 in `tests/golden/noise_meta.json`.

### 2.5 FLAC output (libFLAC 1.5, system)

`--out X.flac` → 48 kHz, 2 ch, 24-bit, via libFLAC's stream encoder. Match what
`soundfile`/libsndfile `PCM_24` produced: clamp to [-1, 1], scale by 8388607,
round to nearest. `--out X.wav` keeps the existing WAV writer. Acceptance: decode
both our FLAC and the reference `audio.flac` from the same latent (see §4) and
compare integer samples: max |Δ| ≤ 1 LSB, ≥ 99.9 % exactly equal.

### 2.6 Artifacts and logging

`--artifacts DIR` writes the same file set `yue2_gen.py` leaves today:
`request.json, ar_request.json, plan.json, plan_manifest.json, score.abc,
abc_tokens.npy, prefix.npy, semantic.npy, nar_noise.npy, latent.npy, config.json,
result.json`. `result.json` carries the timing block (`abc`, `semantic`,
`nar_seconds`, `vae_seconds`, `e2e_seconds`, plus a `card` field with the ggml
device name). Take the current shapes from a recent `yue2_gen.py` artifacts
directory; where a Python-only field has no meaning, drop it and
list the drop in the status file. Keep today's stderr progress lines (`abc: N
tokens in ...`, `ode ...`, `total: ...`) and end with `[gen] Ns` + `[done] path`.

### 2.7 CLI of `song`

```
yue2 song --request R.json --out X.flac [--artifacts DIR] [--seed N]
          [--ar AR.gguf] [--nar NAR.gguf] [--vae VAE.gguf]
          [--device cpu|vulkan] [--gpu N] [--nar-f32] [--noise FILE] [--steps 32]
```
Defaults for the three GGUFs: `yue2-ar-q8_0.gguf`, `yue2-nar-f16.gguf`,
`yue2-vae-f32.gguf` next to the executable, then the repo root (hardlinks live
there). `--gpu 0` → NAR uses `--kv-f16 --vk-f16-matmul --flash-attn`; `--gpu 1`
(Intel Arc) → `--kv-f16 --vk-f16-matmul --query-chunk 8192` — the measured
per-card choice from STATUS_NAR_PERF §8; pick by ggml device **name** (contains
"Intel"), not by index, and print which flags were chosen. `--nar-f32` drops
`--vk-f16-matmul`. `--seed` overrides the request's seed for AR sampling and
noise alike.

## 3. Rules

- **GPU: only `--gpu 1` (Intel Arc) and `--device cpu`. Never device 0.**
- `songs/`, `~/.cache/huggingface`, `tests/golden/` read-only (except the new
  noise golden). Outputs to `tests/out/`.
- Tabs; braces on their own line except `} else {`; every variable earns its
  existence; `YUE2_WARN_FLAGS` clean.
- No torch. No submodule edits.
- Report in `src/STATUS_SINGLE.md` (≤ 300 words + tables + exact commands).
  Update `README.md` (build, `yue2 song`, the one-process precision note). Do not commit.

## 4. Acceptance

1. All existing goldens pass through the subcommands: STATUS_AR §3 (prefix +
   greedy-32 bit-identical, CPU), STATUS_NAR §8 commands 1–3 (CPU ≥ 70 dB,
   Arc, multi-chunk), the VAE 48-frame check (STATUS_NAR §8 command 6).
2. **Equivalence:** `yue2 song` on
   `tests/out/alley_swing_s1_request.json` with `--noise tests/out/nar_noise_alley_swing_s1.npy`
   on the Arc produces `latent.npy` bit-identical to running `yue2 ar` → `yue2 nar`
   → `yue2 vae` by hand with the same flags (same process = same numbers; if it
   isn't bit-identical, find out why before reporting).
3. FLAC: §2.5 bar against the `audio.flac` in the e2e_arc artifacts dir above
   (same latent: feed that dir's `latent.npy` through `yue2 vae` and compare).
4. Noise: golden reproduces byte-for-byte on CPU and after a rebuild.
5. Q8_0 prefix: the §2.3 measurement, reported.
6. Timing table for the full reference song on the Arc: load, abc, semantic,
   NAR, VAE, total, peak VRAM per stage; next to the `yue2_gen.py --gpu 1`
   numbers (242 s cold / see docs/YUE2.md).
