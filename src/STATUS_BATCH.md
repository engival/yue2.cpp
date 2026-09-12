# STATUS_BATCH — stage 5: multi-song AR decode

What was built against `SPEC_BATCH.md`, what it measures, and where it differs.
Every number is the Intel Arc B70 (Vulkan device 1), `yue2-ar-q8_0.gguf`, built
in `build_batch/`. Device 0 (AMD 7900 XTX) was not touched — the column is left
for the coordinator.

```sh
cmake -B build_batch -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
nice -n 10 cmake --build build_batch -j8
```

## 0. What it is

`stage_ar.cpp`'s `generate()` is gone. In its place a per-sequence state machine
(`Seq`, `JobState`, `Runner`) driven by one shared decode loop: one
`llama_decode` per step carrying exactly one token for each active slot, then one
`sample()` pass over all slots (pure, so no slot's logits are invalidated) and
one `apply()` pass (which may prefill, and therefore decode). `sample_step`,
`Sampling`, the prefix construction and `write_artifacts` are used unchanged.

- `run_ar_batch(ArBatchParams, jobs, results)` is the loop; `run_ar` is a
  one-job, `parallel 1` wrapper around it, and `--dump-logits` keeps its own
  single-sequence path.
- `run_batch(BatchParams, jobs)` is `run_song`'s body over a list: AR for every
  job, free the context, then NAR + VAE per job in file order. `run_song` is a
  one-job call into it.
- `yue2 batch --jobs jobs.json [--parallel N] [--summary FILE]
  [--continue-on-error]`, and `yue2 ar --requests jobs.json [--parallel N]`.

**`kv_unified = false` works on this box's Vulkan** — the first thing tested
(§5 acceptance 2, below). No fallback was needed and none is in the code.

## 1. Aggregate AR throughput vs `--parallel`

Identical requests (`tests/out/batch/r1.json`, seed 831001), sampled, full
length, AR only. "AR wall" is the decode loop, model load excluded. GPU peak is
the drop in ggml's free-memory query below a 29 354 MiB-free baseline, polled as
in `STATUS_SINGLE.md` §6. "CPU/token" is the whole process's `user+sys` divided
by the sampled tokens — `sample_step` is most of it.

| parallel | songs | tokens | AR wall (s) | aggregate tok/s | per-slot tok/s | ms/step | CPU (s) | CPU/token (ms) | GPU peak (MiB) | AMD (dev 0) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 7 154 | 67.3 | 106.4 | 106.4 | 9.40 | 28.6 | 3.99 | 3 459 | 222 tok/s, 3 842 MiB |
| 2 | 2 | 12 274 | 75.9 | 161.6 | 80.8 | 11.51 | 32.7 | 2.66 | 4 818 | 326.5 tok/s (1.47×), 5 343 MiB |
| 4 | 4 | 27 808 | 131.9 | 210.9 | 52.7 | 17.36 | 53.6 | 1.93 | 7 539 | 486.1 tok/s (2.19×), 8 377 MiB |
| 8 | 8 | 55 616 | 197.4 | 281.7 | 35.2 | 24.27 | 96.9 | 1.74 | 12 983 | 650.9 tok/s (2.93×), 14 585 MiB |

Scaling: **1.52× at N=2, 1.98× at N=4, 2.65× at N=8.** The §1 hypothesis
("near-linear to N=4, still rising at N=8") is *not* confirmed on the Arc: it is
still rising at 8, but a step costs real time per added slot (9.4 → 24.3 ms), so
the single-stream AR was never purely launch-bound on this card. Token counts
differ per row because a different batch shape gives a different song (§3).

AMD column (coordinator, device 0, `tests/out/batch/sweep_dev0.sh`, log
`sweep_dev0.log`): same requests and seed; sampled songs differ in length per
run, so tokens/time are not comparable across the two cards row by row — the
aggregate tok/s and the scaling factor are. N=1 on the AMD has no `ar batch:`
line (single-job path): 5 566 tokens in 25.05 s. CPU/token on the AMD:
2.42 / 1.67 / 1.12 / 0.85 ms at N = 1/2/4/8 — at N=1 the process's own CPU is
already ~54 % of the 4.5 ms step, so `sample_step` caps single-stream speed on
the fast card as well as the batch. GPU peak is baseline free − min free.

`tests/regress.sh` on device 0 with the rebuilt `build/yue2`: 48/48, worst
106.39 dB.

Two things worth the coordinator's eye:

- **CPU sampling is becoming the bottleneck.** At `parallel 8` a step is 24.27 ms
  of wall for 8 tokens = 3.03 ms/token, of which 1.74 ms is process CPU.
  `sample_step` allocates and scans a 184 704-float vector per call. §4.4 said
  measure first: measured. A per-`Seq` scratch buffer plus an early mask-aware
  scan is the obvious next move, and it is worth more than any GPU change at
  N ≥ 4.
- The per-slot `tok/s` printed in `plan.json` is wall-clock for a slot that is
  sharing the card, so it drops with N by construction. The aggregate is the
  number that means anything.

### Drain phase (§4.7 item 3)

Four *different* full-length requests, `--parallel 4`, AR only — so no refill is
possible and slots fall idle at different lengths:

```sh
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --requests tests/out/batch/drain.json \
	--parallel 4 --device vulkan --gpu 1
```

| steps | ms/step |
|---|---:|
| 6 157 full-width (no interior hole) | 14.69 |
| 2 111 with an idle slot between two active ones | 27.65 |

27 460 tokens in 159.4 s = 172.3 tok/s aggregate; songs of 3 930 / 4 704 / 5 362 /
5 910 semantic tokens. A holed step costs **1.88×** a full-width one, which is
the two-graph-evaluation tax `split_equal` imposes when the live seq ids are not
consecutive. Read it as an upper bound, not a clean measurement: holed steps only
happen late in the run, where contexts are longest, and they carry fewer tokens.
Not fixed in this stage, as instructed; `docs/ROADMAP.md` carries the
`llama_memory_seq_cp` compaction idea.

## 2. End to end, four songs — the headline

The same four requests (different styles and lyrics, seeds 830001–830004), full
length, rendered twice: once at `--parallel 4`, once at `--parallel 1`, which is
literally four `yue2 song` runs back to back through the same code.

```sh
build_batch/yue2 batch --jobs tests/out/batch/e2e_p4.json --parallel 4 \
	--device vulkan --gpu 1 --summary tests/out/batch/e2e_p4_summary.json
build_batch/yue2 batch --jobs tests/out/batch/e2e_p1.json --parallel 1 \
	--device vulkan --gpu 1 --summary tests/out/batch/e2e_p1_summary.json
```

| | AR (s) | NAR (s) | VAE (s) | wall (s) | frames | audio (s) | **s per s of audio** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--parallel 1` (= 4 × `yue2 song`) | 266.9 | 567.3 | 71.9 | **907.6** | 20 482 | 819.3 | **1.108** |
| `--parallel 4` | 172.0 | 542.2 | 70.3 | **786.3** | 19 902 | 795.8 | **0.988** |

**13.4 % off the wall clock, 11 % once normalised for the audio produced.** All
of it comes from the AR: 266.9 s → 172.0 s, a 1.55× speed-up (105.8 → 159.6
tok/s aggregate). That is well short of the 1.98× the identical-request sweep
gives at `parallel 4`, and the reason is the drain phase — these four songs are
3 873–5 909 frames long, so the batch spends its last minutes running two or
three slots, some of the time with an interior hole (2 111 of 8 268 steps, §1).

The normalisation matters because the two arms do not render the same songs
(§3): the same seeds produce different lengths under different batch shapes.
Per-song NAR time tracks frames almost exactly in both arms, which is the
expected behaviour of a stage that is not batched.

Peak VRAM is the same in both arms (21 625 vs 21 843 MiB free at the low-water
mark): the AR context is freed before the first NAR, so the NAR's 7.9 GiB is the
peak either way, and `--parallel 4`'s 7.5 GiB AR batch sits just under it. That
is the §4.5 design working as intended.

## 3. Determinism (§6)

Same request, same seed (831001), sampled, `parallel 1` vs slot 0 of `parallel 4`:

| | abc tokens | first differing abc token | semantic tokens | first differing semantic token |
|---|---:|---:|---:|---:|
| parallel 1 | 1 988 | — | 5 164 | — |
| parallel 4, slot 0 | 1 903 | **118** | 5 047 | **1** |

So sampled decode diverges early and the song is different — exactly what §6
predicts. Within one batch it is the opposite: **all four (and all eight) slots
of an identical-request batch produce byte-identical `semantic.npy`**, so the
batch itself is deterministic, it is only *different from* single-stream decode.

Greedy is where the mechanism is visible. Same request ×4, `--greedy
--max-semantic 512`, `parallel 4` vs a `parallel 1` greedy run: the four slots
agree byte for byte, and they first differ from the single run at **abc token
796** (batch 23, single 19). Logits at that step, dumped from both runs through a
throwaway instrumentation patch (not committed):

| | argmax | raw logit[19] | raw logit[23] | penalised score[19] | penalised score[23] | winner's margin |
|---|---|---:|---:|---:|---:|---:|
| parallel 1 | 19 | 22.684732 | 20.011839 | 20.025442 | 20.011839 | 1.36e-2 |
| parallel 4 | 23 | 22.688530 | 20.030407 | 20.028795 | 20.030407 | 1.61e-3 |

Token 19 occurs 25 times in the 100-token penalty window, so the 1.005^25
repetition penalty pulls its raw 2.67 lead down to a **1.4e-2 tie** with token
23 — while the batch-vs-single logit noise at that position is up to **6.9e-2**
(max over the vocabulary). A genuine near-tie, not a bug: the noise is an order
of magnitude larger than the gap the argmax is deciding. Per §5 acceptance 2,
that is a pass.

## 4. Peak VRAM vs the §4.2 estimate

112 KiB per token per sequence (28 layers × 8 kv heads × 128 × K+V × 2 B) — the
binary prints it before allocating. At the padded 13 761-token per-slot context
that is 1 505 MiB per slot.

| parallel | estimate: 2 195 weights + KV | measured peak | ratio |
|---:|---:|---:|---:|
| 1 | 3 700 MiB | 3 459 MiB | 0.93 |
| 4 | 8 215 MiB | 7 539 MiB | 0.92 |
| 8 | 14 235 MiB | 12 983 MiB | 0.91 |

The poller reads ~10 % under the sum of the buffers ggml reports, the same offset
`STATUS_SINGLE.md` §6 records, so the estimate is good to within the measurement.
`parallel 8` (13.0 GiB) fits the Arc easily and would fit the 24 GB AMD; the
default stays 4 because that is where the AR peak (7.5 GiB) sits next to the
NAR's 7.9 GiB.

## 5. Acceptance (SPEC_BATCH §5)

All on Vulkan device 1 unless stated.

| # | what | result |
|---|---|---|
| 1 | batch of one == today's binary | **pass**, byte-identical |
| 2 | greedy ×4 `parallel 4` agree with each other; vs greedy single | **pass** — four slots identical; diverge from single at a measured near-tie (§3) |
| 3 | four different requests, sampled, `parallel 2` (refill) | **pass** |
| 4 | refilled slot == an independent single run | **pass** (with an external-ABC job; see deviation 4) |
| 5 | cot=off / external-abc jobs in a mixed batch | **partial** — external-ABC passes; cot=off is unreachable in yue2.cpp at all (deviation 3) |
| 6 | `--continue-on-error` | **pass** |
| — | `tests/regress.sh` | **pass**, 48/48 songs, SNR 89.3–113.5 dB (run on device 1, see below) |

### 1 — batch of one is today's binary

```sh
R=tests/out/alley_swing_s1_request.json
for B in build build_batch; do $B/yue2 ar -m tests/out/yue2-ar-f16.gguf --request $R \
	--artifacts tests/out/batch/g_$B --greedy --max-abc 32 --max-semantic 8 \
	--device cpu --threads 8; done
build/yue2       ar -m tests/out/yue2-ar-f16.gguf --request $R --dump-logits tests/out/batch/dump_old.npy --device cpu --threads 8
build_batch/yue2 ar -m tests/out/yue2-ar-f16.gguf --request $R --dump-logits tests/out/batch/dump_new.npy --device cpu --threads 8
build/yue2       ar -m yue2-ar-q8_0.gguf --request $R --artifacts tests/out/batch/full_old --device vulkan --gpu 1
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --request $R --artifacts tests/out/batch/full_new --device vulkan --gpu 1
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --requests tests/out/batch/one.json --parallel 1 --device vulkan --gpu 1
build/yue2       song --request tests/out/batch/r1.json --out tests/out/batch/song_old.flac --artifacts tests/out/batch/song_old --seed 831001 --device vulkan --gpu 1
build_batch/yue2 song --request tests/out/batch/r1.json --out tests/out/batch/song_new.flac --artifacts tests/out/batch/song_new --seed 831001 --device vulkan --gpu 1
```

- `abc_tokens.npy` == `tests/golden/ar_greedy_32.npy` for both binaries;
  `prefix.npy` / `semantic.npy` sha256 equal.
- `dump_old.npy` and `dump_new.npy` are `cmp`-identical.
- The full sampled song (1 948 abc ids, 3 911 codes) has the same
  `abc_tokens.npy` / `prefix.npy` / `semantic.npy` sha256 from all three of:
  old binary `--request`, new binary `--request`, new binary `--requests`
  (`parallel 1`).
- `yue2 song`: `latent.npy` and the 206.6 s FLAC are `cmp`-identical between the
  two binaries (220.7 s vs 221.1 s end to end).

### 2 — greedy batch

```sh
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --requests tests/out/batch/acc2.json --parallel 4 \
	--greedy --max-semantic 512 --device vulkan --gpu 1
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/batch/r1.json \
	--artifacts tests/out/batch/acc2_single --greedy --max-semantic 512 --device vulkan --gpu 1
```

Four slots: identical `abc_tokens.npy` and `semantic.npy`. 10 956 tokens in
37.1 s = 295.6 tok/s aggregate against 129.98 tok/s for the single greedy run.
Divergence from single: §3.

### 3 / 4 / 5 — mixed phases, refill, external ABC

```sh
build_batch/yue2 batch --jobs tests/out/batch/mixed.json --parallel 2 --max-semantic 600 \
	--device vulkan --gpu 1 --summary tests/out/batch/mixed_summary.json
build_batch/yue2 batch --jobs tests/out/batch/ext.json --parallel 2 --max-semantic 300 \
	--device vulkan --gpu 1 --summary tests/out/batch/ext_summary.json
build_batch/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/batch/r_ext.json \
	--artifacts tests/out/batch/solo_ext --seed 43 --max-semantic 300 --device vulkan --gpu 1
```

`mixed.json` is four different requests (prefixes 445–657 tokens) at
`--parallel 2`, so jobs 3 and 4 refill slots 0 and 1 — `result.json` records
`"batch": {"parallel": 2, "slot": 0|1, "jobs": 4}` accordingly. Every
`expect_pos` check passed (they are `die()`s; the run exited 0), every job wrote
the full artifact set, and each FLAC is exactly `1920·T − 64` = 1 151 936 samples
for T = 600.

`ext.json` puts an **external-ABC** job third at `--parallel 2`, so it enters
directly in the SEMANTIC phase *and* lands in a reused slot. Its `prefix.npy` is
`cmp`-identical to the same job run alone — which is the real proof that
`llama_memory_seq_rm` cleared the slot, since an external-ABC prefix does not
depend on sampling (deviation 4).

### 6 — `--continue-on-error`

```sh
build_batch/yue2 batch --jobs tests/out/batch/coe.json --parallel 2 --max-semantic 300 \
	--continue-on-error --device vulkan --gpu 1 --summary tests/out/batch/coe_summary.json
build_batch/yue2 batch --jobs tests/out/batch/coe.json --parallel 2 --max-semantic 300 \
	--device vulkan --gpu 1        # same list, no flag
```

With the flag: `error: [2/3 b.flac] tests/out/batch/r_bad.json: needs a "lyrics"
string` on stderr, the other two render (`2/3 jobs, 3 986 tokens`), exit **1**,
and the summary is a three-element array with one `"status": "error"`. Without
it: the same message and exit 1 **before the model loads** (no `backend:` line in
the log).

### `tests/regress.sh`

`tests/regress.sh` is **unchanged** and is a device-0 script (`--gpu 0`,
`BIN=build/yue2-vae`), which a sub-agent may not run. It was run verbatim except
for those two settings and its output directory:

```sh
sed -e 's|--gpu 0|--gpu 1|' -e 's|^BIN=build/yue2-vae|BIN=build_batch/yue2-vae|' \
    -e 's|^OUT_DIR=tests/out/regress|OUT_DIR=tests/out/batch/regress_gpu1|' \
    tests/regress.sh > tests/out/batch/regress_gpu1.sh   # + an absolute cd, since it moved
nice -n 10 tests/out/batch/regress_gpu1.sh
```

48 of 48 songs decoded, zero failures, SNR 89.33–113.54 dB against their
`audio.flac`. Nothing in stage 5 touches the VAE, so this is a regression guard,
not a measurement. The coordinator should re-run the real `tests/regress.sh` on
device 0 before committing.

## 6. Deviations

| # | what | why |
|---|---|---|
| 1 | `run_ar_batch` takes `const std::vector<ArJob> &` (§4.1 wrote it non-const) | nothing in it is mutated; `run_batch` owns its own copy and fills in temp artifact paths before the call |
| 2 | one `load_jobs_file(path, need_out, jobs)` instead of two loaders | `yue2 batch` requires `out`, `yue2 ar --requests` requires `artifacts` (§3.1 / §3.2); one flag beats two near-identical parsers. It also rejects unknown keys, so a typo'd `"seeed"` is an error rather than a silently ignored seed |
| 3 | **acceptance 5's cot=off half could not be run** | `protocol.SongRequest.guidance` is 1.01 for cot=off and yue2.cpp rejects any `cfg_scale != 1` (stage 2b+). `build/yue2 ar --cot off` fails identically, so this is pre-existing and unrelated to batching. The external-ABC path — the other way into a direct SEMANTIC entry — is tested |
| 4 | **acceptance 4 needs a job whose ABC is not sampled** | a cot=full job's `prefix.npy` embeds the *sampled* abc ids, so it is not deterministic across batch shapes: comparing the refilled cot=full job to a single run diverges at index 446 (the 445-token request prefix matches exactly; the abc ids do not). Redone with an external-ABC job, where the whole prefix is deterministic → byte-identical |
| 5 | `ar_request.json` is written after the AR, not before | so a job rejected under `--continue-on-error` never creates a directory. Same bytes, same place |
| 6 | per-job `e2e_seconds` is cumulative from the start of the batch | there is no meaningful per-song wall clock when four songs share the AR. For a one-job batch (`yue2 song`) it is exactly today's number |
| 7 | `--seed` on `yue2 batch` / `yue2 ar --requests` is a default for jobs that name none | §3.4 does not list it either way; rejecting it would make `--seed` mean two different things depending on the subcommand |
| 8 | `yue2 ar` also gained `--parallel` and `--continue-on-error`; `--request` and `--requests` are mutually exclusive | §3.2 asks for the list form on `yue2 ar`; these are what make it usable |
| 9 | the §1 hypothesis is not confirmed | measured 1.98× at N=4 (§1). Reported, not worked around |
| 10 | NAR failure is fatal even under `--continue-on-error` | exactly as §4.6 instructs — `run_nar`'s `die()`s were left alone |
| 11 | `tests/regress.sh` was run through a device-1 copy | it hardcodes `--gpu 0`; CLAUDE.md forbids a sub-agent device 0. §5 |
| 12 | the drain-phase measurement is an upper bound | holed steps are also the longest-context, fewest-token steps of the run; separating the two would need a synthetic batch. §1 |

Not taken: the `kv_unified = true` fallback. Non-unified KV initialised and
decoded correctly on ggml-vulkan (Intel BMG G31) at `parallel` 2, 3, 4 and 8.

## 7. Not built, on purpose (SPEC_BATCH §2)

No scheduler, queue format, daemon or server; no NAR/VAE batching; no
`cfg_scale`; no cross-GPU overlap; and nothing in the binary or the README knows
about any particular driver. Bit-identical output between batched and
single-stream decode is not claimed — §3 measures exactly how it differs and the
README says so in three sentences.

## Review

Cold read of the uncommitted diff against SPEC_BATCH.md, build in `build_review/`
(`YUE2_WARN_FLAGS` clean: the only `-Wshadow` hits are upstream
`ggml-backend.h:419`, pre-existing, in every TU). Golden re-run on CPU with the
review build: `abc_tokens.npy` == `tests/golden/ar_greedy_32.npy`, and
`abc_tokens.npy` / `prefix.npy` / `semantic.npy` / `score.abc` sha256-identical
across old `build/yue2 --request`, new `--request`, new `--requests` (one job,
`parallel 1`) — that case is abc-TRUNCATED at 32, so the three-token bridge
(`src/stage_ar.cpp:895`) was checked against the old 1+2 decode.

Verified by reading, no change needed:

- Logits ordering: `sample()` runs over every active slot before any `apply()`
  (`src/stage_ar.cpp:1103`); the only decodes inside `apply()` are
  `feed()`s, which consume their own logits immediately; refill happens at the
  top of the loop after all applies. No stale pointer is reachable.
- Per-slot state: `rng.seed(job seed)` + `history.clear()` + `step = 0` at both
  phase entries; `pos` explicit, reset in `finish_job` and `enter`;
  `llama_memory_seq_rm` + `pos_max == -1` postcondition
  (`src/stage_ar.cpp:1016`); `expect_pos` checked in every `feed()`.
- Batch of one: same `n_ctx_want`, `max_feed`, `n_batch = n_ubatch`, `n_seq_max
  = 1`, `kv_unified = false` (llama's default) as the old `run_ar`; the
  per-token decode sequence is the same (one token, seq 0, logits on last only).
- `llama_batch_init`/`free` paired on both paths; all `Seq` fields initialised;
  temp artifacts are a list, anchored, removed by `atexit`.

Fixed in the working tree (all small):

| where | what |
|---|---|
| `src/stage_ar.cpp:908` | `feed()` now stores the feed's logits index in `q.i_batch` before sampling, so the `llama_get_logits_ith(...) returned NULL` die names the right index (it printed the previous lockstep index); also drops the `last` temporary |
| `src/stage_ar.cpp:747` | removed `JobState::error` — written once, never read (`ArResult::error` is the one that is used) |
| `src/stage_song.cpp:430` | `yue2 batch --max-abc/--max-semantic` went through bare `atoi`, so `0`, negatives and garbage were silently "not set" where `yue2 ar` dies; now dies on `< 1` |
| `README.md:75` | example used `--gpu 1`; every other example in the README is `--gpu 0`, and device 1 does not exist on most boxes |

Not fixed — coordinator decides:

1. **"prefix exceeds capacity" dies after the model loads**
   (`src/stage_ar.cpp:1530`), and is a hard `die()` even under
   `--continue-on-error`. The check needs the tokenizer, so it cannot move
   before `llama_model_load_from_file` without a `vocab_only` pre-load —
   same as the old single path. With default limits it needs a ~11k-token
   request text, so it is theoretical.
2. The printed KV estimate (`src/stage_ar.cpp:1586`) uses the unpadded
   `n_ctx_want`; §4.2 asked for the padded per-stream size. Off by < 256 tokens
   per slot; the §4 table says the estimate is already within measurement.
3. The mid-run `semantic prefix + max_tokens > n_ctx_seq` die
   (`src/stage_ar.cpp:879`) kills a whole batch for one job's long abc.
   Only reachable when `n_ctx_want` was clamped to `CONTEXT`, i.e. never with
   the default limits. Same behaviour as the old single path.
4. `SPEC_BATCH.md` refers to the author's batch script and llama-server launcher; the
   README/ROADMAP diff is clean (no driver, no home paths, no identifiers);
   the `SPEC*.md` files are covered by the pre-publish scrub pass.
5. `tests/regress.sh` (unchanged) is device 0 and was not run here per CLAUDE.md.
