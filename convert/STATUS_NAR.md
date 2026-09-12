# convert/ — STATUS_NAR (stage 3: NAR converter + goldens)

All of SPEC_NAR.md §2 and §3 are done and verified except the optional §3.4
full-song golden, which is running in the background (see "Full song" below).

## 1. `convert_nar.py` — safetensors → `yue2-nar-f16.gguf` (SPEC_NAR.md §2)

Reproduce:

```
cd convert       # from the repo root; ../../venv_yue2 is the torch-CPU venv
../../venv_yue2/bin/python convert_nar.py
# -> tests/out/yue2-nar-f16.gguf, ~22s wall
```

Result:
- `tests/out/yue2-nar-f16.gguf` — 2,939,691,584 bytes, **317 tensors**
  (`11 × 28 + 9`, matches §2.2 exactly). Tensor table printed at end of run
  matches the §2.2 table name-for-name, shape-for-shape, dtype-for-dtype
  (F16 for all 2D weights + `latent_pos_embd.weight`, F32 for every norm/bias
  and the four `vae2llm`/`llm2vae`/`time_embd` linears).
- `general.architecture = "yue2-nar"`, `general.name = "YuE2-3B NAR"`.
- `yue2.source_sha256 = 1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9`
  — **verified equal** to `tests/out/yue2-ar-f16.gguf`'s value (read back both
  with `gguf.GGUFReader` and diffed the strings) and to
  `sha256sum model.safetensors`. `convert_nar.py` also does this cross-check
  itself at conversion time (`--ar-gguf`, defaults to the sibling AR file) and
  aborts on mismatch.
- `latent_pos_embed.pe` sanity check against the closed-form sinusoid
  (`modeling_yue2.py:334-346`, reimplemented as `reference_pe()` in the
  converter, float64, then compared to the stored BF16-rounded buffer cast to
  float32): **max|Δ| = 3.247e-03**, under the required 8e-3 — confirms the
  interleaved sin(even)/cos(odd) layout reading is right, not transposed or
  concatenated. Stored (not recomputed) as F16 in the GGUF per spec, since F16
  is exact for a BF16 source value.
- All `yue2nar.*` metadata keys written per §2.3 (`block_count 28`,
  `embedding_length 2048`, `feed_forward_length 6144`, head counts 16/8,
  `key_length`/`value_length` 128, rms eps 1e-6, rope base 1e6,
  `context_length 24576`, `latent_dim 64`, `max_latent_frames 24576`,
  `timestep_shift 1.0`, `time_embd_frequency_size 256`, `ode_steps 32`,
  `ode_method "midpoint"`).
- No transposes: gguf-py's shape reversal is relied on exactly as in
  `convert_ar.py`; the printed "torch shape" column in the tensor table
  matches §2.2's table directly.

### Deviation from spec
None. One minor addition: `--ar-gguf` is a real cross-check against the file
on disk (not just a documented invariant) — if `yue2-ar-f16.gguf` doesn't
exist yet the check is skipped with a warning rather than failing, since the
NAR GGUF can legitimately be built first or independently.

## 2. `reference_nar.py` — torch CPU float32 goldens (SPEC_NAR.md §3)

Drives the **real** code path for every golden: `yue2.nar.song_chunks()` →
`yue2.nar.CachedNAR(model, chunk)` → `.solve(steps)`. No reimplementation of
any NAR math. Intermediates are captured via:
- `engine.cache[l]` (already the real post-prefill K/V, no hook needed) for
  `nar_ar_kv_l{0,27}`.
- A thin counting wrapper around the *bound* `engine.velocity` method
  (`engine.velocity = traced_velocity`, calls the original unchanged) to
  record every `raw_t` fed in (→ `nar_tshift.npy`) and the first call's
  returned velocity (→ `nar_v_step0.npy`).
- Two `register_forward_pre_hook`s, on `layers[0].nar_input_layernorm` and
  `layers[1].nar_input_layernorm`, active only during that same first call,
  to capture `x_in` (state entering layer 0, i.e. after
  `vae2llm + time_emb + pos_emb`) and `x_l0` (state entering layer 1, i.e.
  after NAR layer 0) — exactly the tensors §3.2 names, taken from the actual
  forward pass.

`CUDA_VISIBLE_DEVICES=""` / `HIP_VISIBLE_DEVICES=""` are set at the top of
the script (before `import torch`), and `next(model.parameters()).device.type
== "cpu"` is asserted right after load. `model.lm_head` is deleted after load
(never used by the NAR path; saves ~1.5 GB).

### Mode `short` (§3.1/3.2) — mandatory, run

```
../../venv_yue2/bin/python reference_nar.py --mode short
```

Wall time: **69.5s total** (load 1.6s, steps=2 solve 6.4s, steps=32 solve
57.0s) — well inside the 5-minute budget, in fact inside the "1-2 min"
estimate for the 32-step run alone.

Built from `alley_swing_s1` exactly per §3.1: `prefix_short = prefix.npy[:255]
+ [151851]` (256 ids, confirmed ends in MUSIC_START), `codec_short =
semantic.npy[:128]`, `seed=1`, `context=24576` → one chunk (confirmed:
`nar.song_chunks(...)` returned exactly 1 `Chunk`), giving `ar_length=385`,
`nar_length=130`, `S=515` — matches §3.1's arithmetic exactly.

All 11 files under `tests/golden/` written, shapes/dtypes verified against
§3.2's table, all finite:

| file | shape | dtype | status |
|---|---|---|---|
| `nar_prefix_ids.npy` | (256,) | int32 | OK |
| `nar_codec_ids.npy` | (128,) | int32 | OK |
| `nar_noise.npy` | (128,64) | float32 | OK |
| `nar_ar_kv_l0.npy` | (2,385,8,128) | float32 | OK |
| `nar_ar_kv_l27.npy` | (2,385,8,128) | float32 | OK |
| `nar_x_in_step0.npy` | (130,2048) | float32 | OK |
| `nar_x_l0_step0.npy` | (130,2048) | float32 | OK |
| `nar_v_step0.npy` | (128,64) | float32 | OK |
| `nar_latent_s2.npy` | (128,64) | float32 | OK |
| `nar_latent_s32.npy` | (128,64) | float32 | OK |
| `nar_tshift.npy` | (64,) float64 | OK |

`nar_tshift[0] == 20.0` exactly (the `t=1 → logit=+inf → clamp(20)` case
predicted in §1.3) — a cheap independent confirmation the capture is wired to
the right call. `tests/golden/nar_short_meta.json` written with
`{prefix_len, codec_len, ar_length, nar_length, seed, steps, source_sha256,
torch_version, wall_seconds}` as specified.

Implementation note: `steps=2` and `steps=32` are solved on the **same**
`CachedNAR` instance (one prefill, `engine.cache` read once for both KV
goldens) rather than two separate instances — `solve()` only reads
`self.chunk.noise` and never mutates it, so this is equivalent to the spec's
two-step-and-32-step framing but saves a redundant 385-token prefill. Noted
here since it's not spelled out in SPEC_NAR.md.

### Mode `multichunk` (§3.5) — mandatory, run

```
../../venv_yue2/bin/python reference_nar.py --mode multichunk
```

Wall time: **22.5s** (well under a minute, as estimated). Uses the same
256-id short prefix, `semantic.npy[:512]`, `context=1200`, `steps=2`. Real
`chunk_ranges` returned: **`[(0,470), (470,512)]`** — matches §3.5's
prediction exactly. `ar_lengths = [727, 299]` (i.e. `256 + 470 + 1` and
`256 + 42 + 1`). Output shape `(512, 64)`, finite.

Written:
- `tests/out/nar_multichunk_cpu.npy` — the stitched `[512,64]` latent.
- `tests/out/nar_multichunk_noise.npy` — the full `[512,64]` noise draw
  (concatenation of both chunks' `chunk.noise` views, i.e. the single
  seed-1 draw for `T=512` before slicing — bit-identical to what the C++
  agent's own `song_chunks`-equivalent chunking should produce).
- `tests/out/nar_multichunk_meta.json`, and the same object is **folded into
  `tests/golden/nar_short_meta.json`** under the `"multichunk"` key so the
  C++ agent has the chunk ranges and both `ar_length` values without
  re-running torch, per §3.5's instruction.

### Mode `full` (§3.4) — optional, RUNNING IN BACKGROUND, not waited on

Launched (this session, niced, non-blocking):

```
cd convert       # from the repo root
nohup nice -n 19 ../../venv_yue2/bin/python reference_nar.py --mode full \
	> ../tests/out/nar_full_cpu.log 2>&1 &
```

PID at launch time: 992 (parent shell backgrounding via `disown`). Estimated
40-120 minutes per §3.3/3.4 (full `alley_swing_s1`: P=2549, T=4109, steps=32).

**How to check on it:**

```
tail -f tests/out/nar_full_cpu.log
ps aux | grep reference_nar
ls -la tests/out/nar_full_cpu_latent.npy \
       tests/out/nar_noise_alley_swing_s1.npy
```

When done, the log's last line is `[reference_nar] mode=full load=...s
total=...s` and both `.npy` files exist:
`tests/out/nar_noise_alley_swing_s1.npy` (`[4109,64]` float32, the seed=1
draw for the whole song) and `tests/out/nar_full_cpu_latent.npy` (`[4109,64]`
float32, the steps=32 CPU latent) — these are `tests/out/` scratch, not
checked-in goldens, per §3.4 (`.gitignore` already has `*.npy`).

This run was **not** waited on before writing this STATUS or reporting to the
coordinator, per the task instructions; the C++ agent's critical-path inputs
(the GGUF and the §3.2/§3.5 files) were already complete before it was
started.

## 3. Deviations / open questions

- None from the spec's required outputs. The only additions are the
  `--ar-gguf` cross-check in `convert_nar.py` and folding `multichunk` into
  `nar_short_meta.json` (both explicitly invited by the spec's wording, not
  contradicting it).
- `reference_nar.py`'s three modes are one script (matching the existing
  `reference_ar.py`/`reference_decode.py` split-by-concern style, just
  collapsed into `--mode` since §3.1/§3.2/§3.5 share essentially all their
  setup code and only §3.4 is materially different in cost).
- Not yet independently confirmed: whether the C++ agent's own `chunk_ranges`
  implementation agrees with `(0,470),(470,512)` for the same inputs — that
  check lives on their side against `nar_multichunk_cpu.npy`/`_meta.json`
  here.
- `torch_dtype` deprecation warning from `from_pretrained` (transformers
  4.57.6 wants `dtype=`) is harmless noise, left as-is rather than papering
  over a warning in someone else's library call.
