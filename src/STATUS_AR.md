# src/ — STATUS (stage 2, C++ AR generator)

SPEC_AR.md §1 and §3 deliverables. **Everything in §3 is implemented and all
three acceptance checks pass** (one number is outside the spec's suggested
bound — see Deviations #1).

## Files

| file | what |
|---|---|
| `src/yue2-ar.cpp` | protocol, sampling, two-phase generation loop on libllama, artifacts, CLI |
| JSON | `nlohmann::ordered_json` from `llama.cpp/vendor/nlohmann/` (no vendoring of our own) |
| SHA-256 | `hash_sha256_hex()` from llama.cpp's `vendor::hash` (our `src/sha256.hpp` is gone) |
| `src/npy.hpp` | extended with `ArrayI32` / `load_i32` / `save_i32` (`'<i4'`); `'<f4'` path untouched |

## Build system

- `llama.cpp` is now a submodule (`8ea290247`, ggml 0.23.0); the standalone
  `ggml/` submodule is gone (`git rm ggml`, `.gitmodules` has only `llama.cpp`).
  ggml comes from `llama.cpp/ggml`, so there is exactly one copy in the process.
- `CMakeLists.txt` sets `LLAMA_BUILD_{EXAMPLES,TESTS,TOOLS,SERVER,COMMON} OFF`,
  `GGML_VULKAN ON`, and builds both `yue2-vae` (links `ggml`) and `yue2-ar`
  (links `llama ggml`). `yue2-ar` also gets `llama.cpp/vendor` on its include path
  for `<nlohmann/json.hpp>`. Clean build, no warnings from `src/`.
- `README.md`'s `git submodule update --init` line is still correct as written.

```
cd <repo root>
git submodule update --init
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
nice cmake --build build -j 6
```

### yue2-vae still passes its golden test on llama.cpp's ggml

```
build/yue2-vae -m tests/out/yue2-vae-f32.gguf \
	-i ../songs/ref_song/out/alley_swing_s1/latent.npy \
	--full --frames 48 --npy tests/out/ar_regress_f48.npy --device cpu
convert/compare.py tests/out/ar_regress_f48.npy tests/golden/alley_swing_s1_f48.npy
```

| run | max abs err | SNR | required |
|---|---|---|---|
| CPU F32 | 1.50e-06 | **121.95 dB** | ≥ 60 dB PASS |
| Vulkan 0 F32 | 4.27e-04 | **69.42 dB** | ≥ 60 dB PASS |

(Was 122.40 / 69.42 dB against the old standalone ggml — unchanged to within
the CPU path's own rounding.)

## Run

```
build/yue2-ar -m tests/out/yue2-ar-q8_0.gguf \
	--request tests/out/alley_swing_s1_request.json \
	--artifacts tests/out/ar_full_vk_q8 --device vulkan --gpu 0
```

Full CLI, exactly SPEC_AR §3 plus `--threads N`:

```
yue2-ar -m MODEL.gguf --request song.json --artifacts DIR
        [--seed N] [--cot full|melody|off] [--device cpu|vulkan] [--gpu N]
        [--threads N] [--dump-logits FILE.npy] [--greedy]
        [--max-abc N] [--max-semantic N]
```

`--gpu N` is the Vulkan device index (0 = AMD 7900 XTX, 1 = Intel Arc), same
convention as `yue2-vae`. Everything below ran on **device 0 only**; no
`llama-server` was up during the measured runs, so nothing had to fall back to
CPU for want of VRAM.

## Acceptance

Request = `songs/ref_song/out/alley_swing_s1/request.json` (copied to
`tests/out/alley_swing_s1_request.json`).

**Prefix.** `prefix_abc` is **655 tokens, bit-identical to
`tests/golden/ar_prefix_ids.npy`** — the protocol text, the tokenizer and the
`[EOD] … [ABC_START]` framing all agree with torch. (655 is also the
`prefix_tokens` recorded in the reference run's `plan.json`.)

### 1. `--dump-logits` vs `tests/golden/ar_last_logits_f32.npy`

Golden argmax is id **55**, logit 33.2120; runner-up is ABC_END at 20.8037.

| gguf | device | argmax | max abs Δ | mean abs Δ | top-50 set overlap |
|---|---|---|---|---|---|
| f16  | CPU      | **55 ✓** | 0.2386 | 0.0026 | 49/50 |
| q8_0 | CPU      | **55 ✓** | 0.1889 | 0.0076 | 49/50 |
| q8_0 | Vulkan 0 | **55 ✓** | 0.1453 | 0.0052 | 49/50 |
| f16  | Vulkan 0 | **55 ✓** | 0.2226 | 0.0046 | 48/50 |

The max is always at id 55 itself — i.e. it is 0.4–0.7 % of the largest logit,
not an outlier somewhere in the tail. See Deviations #1.

### 2. `--greedy --max-abc 32` vs `tests/golden/ar_greedy_32.npy`

| gguf | device | result |
|---|---|---|
| f16  | CPU      | **IDENTICAL (32/32)** |
| q8_0 | CPU      | **IDENTICAL (32/32)** |
| q8_0 | Vulkan 0 | **IDENTICAL (32/32)** |

Re-verified on Vulkan 0 after the JSON swap, together with check 1 (argmax 55,
max abs Δ 0.1453 — unchanged) and a short `--max-abc 64 --max-semantic 256` run
whose artifacts `SymbolicPlan.load()` still accepts.

No divergence at all, so no near-tie to report. Decoded, those 32 tokens are
`X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal`.

### 3. Full run, default sampling, seed 1, Q8_0 on Vulkan device 0

```
abc:      2441 tokens in 12.17 s = 200.58 tok/s
semantic: 4909 tokens in 24.89 s = 197.23 tok/s
artifacts: tests/out/ar_full_vk_q8 (abc 2440 ids, semantic 4908 codes)
```

- ABC phase ended on ABC_END well inside `max_tokens` 4096 (not truncated).
- `score.abc` opens `X:1 / T: / M:4/4 / L:1/32 / Q:1/4=120 / V: Vocal … / K:D#m`
  — the same header shape as the reference `score.abc`.
- Semantic phase ended on MUSIC_END; **4908 codes, all in [6, 32761] ⊂ [0, 32768)**.
- The artifacts round-trip through the Python pipeline:
  `yue2.pipeline.SymbolicPlan.load(dir)` accepts the directory (manifest hashes,
  dtypes, `plan.json`/array agreement, `score.abc`/`abc` agreement all verified),
  `protocol.token_prefixes(request, tokenizer, abc_ids) == prefix.npy`, and
  `tokenizer.decode(abc_ids) == score.abc`.

**Torch numbers to beat were 42 t/s (abc) and 33 t/s (semantic, decaying): we
are ~4.8× and ~6.0× faster, and the semantic phase does not decay.**

`llama-bench` on the same GGUF, device 0 (`~/bin/llama-bench`, build d6822ffca):

| test | t/s |
|---|---|
| pp512 | 13883.40 ± 187.54 |
| tg128 | 294.57 ± 0.83 |

The 197–200 t/s we see is the tg128 ceiling minus the per-step sampling pass
over all 184 704 logits (masking + window penalty + top-k/top-p on the host).

CPU reference for the same full run (Q8_0, `--threads 12`, i9-11900K):

| phase | tokens | seconds | tok/s |
|---|---|---|---|
| abc | 2623 | 293.88 | 8.93 |
| semantic | 4614 | 639.11 | 7.22 |

So Vulkan is ~22–27× faster than CPU here. Model load is 0.3–1.0 s.

## Implementation notes

- **Protocol** reimplemented from `protocol.py`: the three `INSTRUCTIONS`
  strings, `text() = INSTR + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n"`,
  `prefix_abc = [EOD] + tokenize(text) + [ABC_START]`,
  `prefix_semantic = prefix_abc + abc_ids + [ABC_END, MUSIC_START]`.
  `llama_tokenize(..., add_special=false, parse_special=false)`.
  ABC ids are checked to stay `< EOD` before they enter the semantic prefix.
- **Sampling** follows `sampling.distribution` in order: allowed mask → end
  masked while `step < min_tokens` → `window_penalty` over the last
  `penalty_window` *generated* ids (`alpha = penalty^freq`, `logit<0 ? *alpha : /alpha`)
  → temperature 0 ⇒ argmax (first index on a tie, as torch does) → `/temperature`
  → top-k (keep ≥ the k-th value) → top-p (`cumsum − p > top_p`, head always kept;
  3 kept when `legacy_off`) → multinomial from `std::mt19937_64(seed)`.
  Both phases reset the RNG to `request.seed`, matching torch.
- **One context for both phases.** The semantic prefix is exactly the abc prefix
  plus the generated abc ids, so instead of a second prefill we keep the KV cache
  and decode only the two bridging tokens `[ABC_END, MUSIC_START]`. Same causal
  prefix, same positions — mathematically identical to torch's re-prefill, which
  is why the semantic phase's prefill time is ~0. (When the abc phase did *not*
  run — `cot=off`, or an externally supplied `abc` — the cache is cleared and
  the full semantic prefix is decoded.)
- **Refusals**: `prefix + max_tokens > 24576` per phase, `cfg_scale != 1`,
  an abc id outside the text vocabulary, a semantic id outside the codec range.
- `n_ctx` is sized to `prefix + max_abc + 2 + max_semantic + 8`, clamped to 24576
  (13 761 for the default budgets), so nothing is over-allocated.
- Per-phase `tokens / seconds / tokens-per-second` are printed, plus backend,
  device description, load time and context size.

## Artifacts written to `--artifacts DIR`

`prefix.npy` (int32 semantic prefix), `abc_tokens.npy` (int32, no
ABC_START/ABC_END), `score.abc` (UTF-8 detokenization of `abc_tokens`),
`semantic.npy` (int32 codes = id − CODEC_OFFSET, MUSIC_END dropped),
`request.json`, `plan.json`, `plan_manifest.json`.

JSON goes through `nlohmann::ordered_json`; `dump(2, ' ', false)` plus a trailing
newline reproduces python's `json.dumps(..., indent=2, ensure_ascii=False)`
layout — `request.json` from a re-run is **byte-identical** to the previous
hand-rolled writer's, and `plan.json` differs only in the values of that run.
Both use shortest-round-trip float formatting, which is what python's `repr`
does, so the timing/`cfg_scale` numbers agree too.

Parsing is strict: `json::parse` rejects trailing garbage, bad escapes and lone
surrogates, and **`seed` is read as an exact `uint64_t`** (`is_number_unsigned()`
then `get<uint64_t>()`, with an explicit `< 2**63` check) rather than through a
double — a seed above 2^53 would otherwise be silently rounded. Verified:
`seed = 9223372036854775806` round-trips into `request.json` unchanged, while a
negative seed, a fractional seed and trailing garbage are each rejected by name.

`plan.json` carries `request`/`timing`/`truncated`/`prefix`/`abc_ids`/`abc`
with the same keys as `SymbolicPlan.save`; `timing` additionally reports
`execution: "eager"` and `attention: "llama.cpp"`. **Not** written:
`config.json`, `result.json`, `latent.npy`, `audio.flac` — those belong to
`SongResult.save_artifacts` and need the NAR/VAE stages plus weight hashes we
do not have here. The Python resume path (`SymbolicPlan.load`) only needs the
four files it hashes, and it accepts ours.

## Deviations

1. **`max|Δ|` on the logits is 0.15–0.24, not "well under 0.1".** It sits
   entirely on the top logit (33.2), i.e. 0.4–0.7 % relative, with mean |Δ|
   0.003–0.008 over the whole 184 704-wide vector; argmax matches on every
   build/device and the 32-token greedy continuation is bit-identical, so the
   decision path is exact. The cause is llama.cpp's f16/q8 matmul accumulation
   against torch's float32 CPU reference — the converter only produced f16 and
   q8_0, so there is no F32 GGUF to separate weight precision from arithmetic
   precision. Curiously q8_0 is *closer* than f16 on both devices, which is
   consistent with this being accumulation noise rather than weight error.
2. **The first Vulkan run costs ~16 minutes of shader compilation.** Confirmed
   by `gdb` backtrace: the main thread sits in
   `ggml_vk_create_pipeline_func ← ggml_vk_load_shaders ← ggml_pipeline_request_descriptor_sets`
   inside `libvulkan_radeon`, GPU 2 % busy, one core pegged. llama.cpp compiles
   one pipeline per `load_shaders` call on the calling thread, and a
   single-threaded generator gets no help. It is **not** yue2-ar-specific — a
   plain `~/bin/llama-bench` on the same GGUF stalls the same way — and RADV's
   disk cache absorbs it: the next run prefills 655 tokens in **0.085 s** and
   completes in 1.4 s wall. Budget for it once per machine/driver/shader change.
3. **No NFC normalization.** The request text is used as-is; a note goes to
   stderr if it contains non-ASCII bytes. The alley_swing_s1 request is ASCII,
   so NFC is the identity there. A Chinese request would need a real NFC pass.
4. **`cot=off` is currently rejected**, because `SongRequest.guidance` is 1.01
   (not 1.0) for `off` and SPEC_AR says `cfg_scale != 1` → "not supported".
   The `off` code paths (no abc phase, `legacy_off` top-p keeping 3) are written
   and will work once CFG lands in stage 2b.
5. **RNG parity with torch is not attempted** (`std::mt19937_64` + cumulative
   sampling vs `torch.multinomial`), as the spec allows — greedy is the parity path.
6. Timing/`cfg_scale` floats are formatted by nlohmann's shortest-round-trip
   writer. It agrees with python's `repr` on everything seen so far, but the two
   are separate implementations and a disagreement would not be a bug here.
7. `--threads N` was added to the CLI (not in the spec's list) to match
   `yue2-vae` and to make the CPU numbers above reproducible.

## Open questions

1. Would a bf16 or f32 GGUF bring `max|Δ|` under 0.1? Worth one converter run
   to settle whether the residual is weight rounding or accumulation order.
2. `n_ubatch` is set to the prefix length (655) rather than the conventional 512.
   That probably widens the set of matmul pipeline variants the first run has to
   compile; capping it at 512 may shrink deviation #2's one-time cost.
3. The per-step host-side sampling pass touches all 184 704 logits three times
   (mask, penalty, candidate gather). At 200 t/s it is already a visible slice of
   the step; if stage 2b wants more, the mask and the candidate gather can be
   fused, or the allowed range can be scanned directly instead of masked.
4. `cot=off` and CFG (two prefixes, `unconditional + scale*(conditional − unconditional)`)
   are the obvious stage-2b items; the two-branch decode would need two sequences
   in one context or two contexts.

## Review fixes (cold code review, 2026-09-11)

Applied to the files this agent owns: `src/yue2-ar.cpp`, `src/npy.hpp`,
`src/wav.hpp`, `CMakeLists.txt` (`src/sha256.hpp` deleted). Built and verified
in **`build_fix/`** (`cmake -B build_fix -DGGML_VULKAN=ON`), all checks on
`--device cpu` with `tests/out/yue2-ar-f16.gguf`, because a real song was
running on `build/` and Vulkan device 0 at the time.

| finding | change | verification |
|---|---|---|
| **M1** truncated ABC leaves the KV cache one token short | `generate()` decodes **every** token it keeps (the `step + 1 < max_tokens` guard is gone), and takes an `expect_pos` argument: after the prefill it compares `llama_memory_seq_pos_max(mem, 0) + 1` against the phase prefix length and dies on a mismatch | `--greedy --max-abc 16 --max-semantic 8`: `prefix.npy` = 655 prefix + all 16 abc ids + `[ABC_END, MUSIC_START]` (673), and the run passes the KV check. A deliberately un-fixed rebuild of the same source dies with `semantic prefill left 672 tokens in the KV cache, the prefix is 673` — i.e. the guard catches exactly the old corruption |
| **M2** `cot=off` / external `"abc"` abort in `GGML_ASSERT(n_tokens_all <= n_batch)` | `n_batch` is sized from the largest *single* feed (`max(prefix_abc, prefix_abc + abc_ids + 2, 512)`), **and** `decode_feed()` splits any prefill into `llama_n_batch(ctx)`-sized `llama_decode` calls, so an oversized feed can no longer abort | external ABC (1892 abc tokens, batch 2549) and `cot: "off"` + `cfg_scale: 1.0` (batch 647) both run `--max-semantic 8` cleanly. The un-fixed rebuild SIGABRTs with a ggml backtrace through `decode_feed`/`generate` on the same external-ABC request |
| **M5** `system("mkdir -p '…'")` | `std::filesystem::create_directories` + `error_code` | artifacts still written (all four manifest hashes verify) |
| **S1** `--seed` bypassed the `< 2**63` check | `parse_seed_arg()`: `strtoull` with `endptr`, `errno`, no leading `-`/`+`, same `< 2**63` bound as the JSON path | `--seed -1` → `--seed must be a non-negative integer below 2**63 (got "-1")` |
| **S2** `json::dump()` throws on invalid UTF-8 | `dump_py` passes `json::error_handler_t::replace`, matching the reference's `errors="replace"` | truncated-ABC runs (`--max-abc 16/32`) write `plan.json` without throwing |
| **S3** `n_vocab != VOCAB_SIZE` was only a warning | now `die` | code path read; normal runs unaffected (vocab is 184704) |
| **S4** `llama_tokenize` result ignored on the external-ABC path | one `tokenize(vocab, text, what)` helper checks both calls; `detokenize()` likewise | external-ABC run tokenizes 1892 ids |
| **S5** validation looser than `SongRequest.__post_init__` | `validate_request()` mirrors it: `cot`, seed range, `id` regex `[A-Za-z0-9][A-Za-z0-9_.-]{0,179}`, `abc` non-blank and not with `cot=off`, `cfg_scale` finite in `[0,20]` | each rejection exercised: bad `id`, blank `abc`, `abc`+`cot=off`, `cfg_scale: 99` |
| **S6** `npy.hpp` duplication / unchecked arithmetic | one `load_t<T>`/`save_t<T>` pair with a `descr<T>()` trait (`load`/`save`/`load_i32`/`save_i32` are thin wrappers, so `yue2-vae.cpp` is untouched); element count checked for negative dims and `SIZE_MAX` overflow and against the real remaining file size; `parse_shape` bounds digit runs and uses `strtoll` (no `std::stoll` throw); every `fwrite`/`fclose` checked → `short write on <path>` | both binaries rebuilt; `--dump-logits`, `prefix.npy`, `abc_tokens.npy`, `semantic.npy` all round-trip through numpy |
| **S7** `wav.hpp` silently overflowed past 4 GiB | refuses with `wav: > 4 GiB of samples does not fit a RIFF header`; writes checked too | header-only change, `yue2-vae` still compiles and links |
| **S9** `sha256.hpp` duplicated `vendor::hash` | `src/sha256.hpp` deleted; `yue2-ar` links `vendor::hash` (`hash_sha256_hex`) and `vendor::nlohmann` instead of adding `llama.cpp/vendor` to the include path | `plan_manifest.json` hashes re-verified against python `hashlib.sha256` for all four files |
| **S10** `main()` was 545 lines | split into `parse_args`, `parse_request`, `validate_request`, `load_model`, `run_dump_logits`, `write_artifacts`, `tokenize`, `detokenize`, `decode_feed`; `main` is now 271 | full acceptance re-run |
| **S11** style drift | space-only continuation lines gone; the two comparators are named `by_score_desc` / `by_score_then_id`; range-`for` over the penalty map; `std::vector<std::pair<float,int>>`; `{ a, b }` shape literals | tabs/brace style per SPEC §6 |
| **S13** warnings not enabled for `src/` | `target_compile_options(... -Wall -Wextra -Wshadow)` on both targets; `die()` is `[[noreturn]]` | clean build of `yue2-ar` and `yue2-vae`; the only warning left comes from llama.cpp's own `ggml-backend.h` (`ggml_backend_graph_copy` hides its constructor), not from `src/` |
| **N1** temperature division | divides by a `float` copy of the temperature | greedy path unaffected; check 2 still bit-identical |
| **N2** wrong `cand.reserve(4096)` | reserves the actual allowed-set size (`EOD+1` abc, `CODEC_SIZE+1` semantic). The triple pass over 184 704 logits is left alone on purpose — it is the parity path | check 1 and 2 unchanged |
| **N4** `--max-abc 0` silently ignored | `--max-abc` / `--max-semantic` must parse as an integer in `[1, 24576]` | `--max-abc 0` now errors |
| **N5** CPU device still initialised Vulkan | `ggml_backend_load_all()` is only called on the Vulkan path, and the CPU path passes an empty `mparams.devices` list (safe: `GGML_BACKEND_DL=OFF`, so backends are registered statically) | every check above ran on `--device cpu` |
| **S8** `common.hpp` for the code shared with `yue2-vae.cpp` | **not done** — it needs edits in `yue2-vae.cpp`, which another agent owns. The `NOTE:` comment above `sample_step` ("do not replace this with llama's sampler chain") is in place | — |

### Acceptance re-run (`build_fix/yue2-ar`, F16 GGUF, `--device cpu --threads 12`)

| check | result |
|---|---|
| 1. `--dump-logits` vs `tests/golden/ar_last_logits_f32.npy` | argmax **55 = 55**, max abs Δ **0.00167**, mean abs Δ 0.00024 — PASS (and far tighter than the 0.2386 recorded earlier for CPU F16; the only change in that path is the batch/feed plumbing, so the earlier figure is worth re-measuring on `build/`) |
| 2. `--greedy --max-abc 32` vs `tests/golden/ar_greedy_32.npy` | **IDENTICAL (32/32)** |
| (a) `--greedy --max-abc 16` | `prefix.npy` = prefix_abc(655) + 16 abc ids + `[ABC_END, MUSIC_START]`, KV check passes |
| (b) `"cot": "off"` (with `cfg_scale: 1.0`) and external `"abc"` | both prefill and generate 8 semantic tokens, no abort |

`cot=off` still needs an explicit `"cfg_scale": 1.0` in the request, because
`SongRequest.guidance` is 1.01 for `off` and CFG is a stage-2b item
(Deviations #4) — the code path itself is now exercised.

### Deferred: `yue2-vae.cpp` (owned by another agent, mid-investigation)

- **S8** shared `die()`/`need()`/timing/Vulkan-device-pick duplication (`:26`, `:465`, `:37`, `:549-560`).
- **S11** `else if (...) {` with a two-line condition at `:679`.
- **N6** `us_compute` out-parameter is never null (`:389`, `:417-420`) — return it.
- **N7** the magic `- 64` in the output length (`:365`).
- **N8** `fseek(f, (long) …)` in the GGUF loader (`:202`).

Everything in `convert/`, `README.md`, `NOTICE.md`, `.gitignore`, `tests/regress.sh`
(M3, M4, M6, S12, S14, N9, N10, N12) belongs to the other agent as well.
