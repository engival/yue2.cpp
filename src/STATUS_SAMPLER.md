# STATUS_SAMPLER — stage 6: the sparse `sample_step`

What was built against `SPEC_SAMPLER.md`, what it measures, and where it
differs. The agent's numbers are the Intel Arc B70 (Vulkan device 1),
`yue2-ar-q8_0.gguf`; the AMD 7900 XTX (device 0) table under §1 was added by
the coordinator afterwards.

```sh
cmake -B build_sampler -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON
nice -n 10 cmake --build build_sampler -j8

# the "before" binary, built from 07a5b89 (stage 5) for every comparison below
cmake -B build_sampler_base -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
nice -n 10 cmake --build build_sampler_base -j8
```

## 0. What it is

`sample_step` no longer materialises the vocabulary. The masks became bounds,
the penalty became a sorted side list, and `nth_element` over every finite entry
became a bounded min-heap:

- The allowed set is at most two ascending, disjoint id ranges (`abc`:
  `[0, EOD)` then `ABC_END`; `semantic`: `MUSIC_END` then the codec block), with
  the `end` range collapsed to empty below `min_tokens`. `scan_allowed()` walks
  them; nothing is written.
- The window penalty builds the same frequency map, then a
  `vector<pair<id, penalised>>` of at most `penalty_window` entries sorted by id.
  `scan_allowed` walks it with one cursor, so the lookup is a compare per id, not
  a hash or a binary search. Ids outside the allowed set used to be penalised
  into a `-inf` that stayed `-inf`; they are simply never visited now.
- Two scans over the allowed range: the first counts the candidates and keeps the
  `top_k` largest in a min-heap (front = the k-th largest); the second collects
  everything `>=` that threshold. When the candidate count is `<= top_k` the
  heap already *is* the candidate set and the second scan is skipped.
- From `std::sort` down — the score-then-id order, the double softmax, the
  `cum - p_i > top_p` cut with its `keep_head`, the single `uniform(rng)` draw —
  the stage-5 code is used verbatim, on the same small candidate vector.
- `SampleScratch` (four vectors, all bounded by `top_k`/`penalty_window`) hangs
  off each `Seq` and is passed in, so a decode step allocates nothing. It is an
  optional trailing parameter; `sample_step`'s signature is otherwise unchanged
  and the `Runner` loop's decode/sample/apply structure is untouched (it gained
  only `--verify-sampler` and the 10 s progress line, neither of which samples).

Temperature is still applied *before* the top-k threshold, on both scans. `x/t`
for `t > 0` is monotone, so the k-th largest could have been picked on untempered
values and tempered afterwards — but distinct `x` can round to the same float
after the division, which would add a tied survivor at the boundary that stage 5
kept. Dividing twice costs one instruction per allowed id in the abc phase and
nothing at all in the semantic phase (`temperature == 1` there), which is cheaper
than bounding that collapse would have been. See the comment above `sample_step`.

## 1. Cost per token, before and after

`tests/out/sampler/sweep.sh` — `tests/out/batch/sweep.sh` copied with two
variables added (`BIN`, `TAG`) and nothing else changed, so this is exactly the
STATUS_BATCH §1 method: identical requests (`tests/out/batch/r1.json`, seed
831001), sampled, full length, AR only, VRAM polled the same way.

```sh
for n in 1 4 8; do BIN=build_sampler_base/yue2 TAG=base tests/out/sampler/sweep.sh $n 1; done
for n in 1 4 8; do BIN=build_sampler/yue2      TAG=new  tests/out/sampler/sweep.sh $n 1; done
```

"AR wall" is the decode loop (model load excluded); at `parallel 1` there is no
`ar batch:` line, so it is the printed abc + semantic seconds. "CPU" is the
whole process's `user + sys`, "ms/step" the `ar steps:` figure — that one times
`llama_decode` **only**, so the difference between it and the wall is the
sampler plus the loop's own bookkeeping. GPU peak is baseline free (29 354 MiB)
minus min free. Both binaries decode the *same* tokens in every row (identical
token counts, and `parallel 1` gives the identical 1 989 + 5 165 split).

| parallel | songs | tokens | AR wall (s) base → **new** | aggregate tok/s base → **new** | ms/step base → **new** | CPU (s) base → **new** | CPU/token (ms) base → **new** | GPU peak (MiB) | AMD (dev 0) |
|---:|---:|---:|---|---|---|---|---|---:|---|
| 1 | 1 | 7 154 | 67.52 → **66.35** | 105.9 → **107.8** (+1.8 %) | — | 28.96 → **27.74** | 4.05 → **3.88** (−4.2 %) | 3 459 | |
| 4 | 4 | 27 808 | 132.00 → **127.90** | 210.7 → **217.4** (+3.2 %) | 17.37 → **17.35** | 53.67 → **49.54** | 1.93 → **1.78** (−7.7 %) | 7 539 | |
| 8 | 8 | 55 616 | 182.21 → **173.58** | 305.2 → **320.4** (+5.0 %) | 23.10 → **23.00** | 79.73 → **71.28** | 1.434 → **1.282** (−10.6 %) | 12 983 | see below |

**AMD 7900 XTX (Vulkan device 0)**, run by the coordinator on an idle card
(`… sweep.sh $n 0`, `TAG=amd_base` / `amd_new`, stage 6 binary = `build/yue2`),
same requests, same seed, base and new back to back per N:

| parallel | tokens | AR wall (s) base → **new** | aggregate tok/s base → **new** | ms/step base → **new** | CPU (s) base → **new** | CPU/token (ms) base → **new** | GPU peak (MiB) |
|---:|---:|---|---|---|---|---|---:|
| 1 | 5 566 | 25.28 → **24.23** | 220.2 → **229.7** (+4.3 %) | — | 13.77 → **13.01** | 2.47 → **2.34** (−5.5 %) | 3 907 |
| 4 | 24 004 | 49.82 → **46.84** | 481.8 → **512.4** (+6.4 %) | 0.97 → **0.96** | 27.43 → **24.21** | 1.143 → **1.009** (−11.7 %) | 8 639 |
| 8 | 42 944 | 66.03 → **61.66** | 650.4 → **696.4** (+7.1 %) | 1.04 → **1.04** | 36.48 → **31.19** | 0.849 → **0.726** (−14.5 %) | 14 835 |

As predicted in §4: the AMD's step is about half the Arc's, so the same
per-token saving is a larger share — +4.3 / +6.4 / +7.1 % aggregate tok/s
against the Arc's +1.8 / +3.2 / +5.0 %. `ms/step` (`llama_decode` only) is
unchanged, and the stage-5 STATUS_BATCH figures reproduce (651 tok/s at N=8).

The stage-5 column reproduces STATUS_BATCH §1 where it should: CPU/token 1.93 at
`parallel 4` is the same number to three digits. The tok/s are higher than
STATUS_BATCH's across the board (305 vs 282 at N=8) because that run shared the
box; this pair was run back to back on an idle one.

**How much of the step is still CPU.** `ms/step` is flat (23.10 → 23.00 at
N=8): `llama_decode` is GPU-bound on the Arc and the sampler was never inside
it. What shrank is the gap between the step's wall and its decode — 3.30 →
3.14 ms per token at N=8, of which 1.28 ms is still process CPU. So after stage 6
the AR loop on this card is **GPU-bound**, and the remaining CPU/token is mostly
llama's own per-step host work, not `sample_step`.

**The sampler in isolation** (`--only`, and `--shipped` to use only the two real
configs in a song's 28/72 abc:semantic ratio; the harness costs 0.02 ms/case,
measured as `both − ref − new`):

```sh
build_sampler/yue2-sampler-diff --cases 20000 --seed 7 --shipped --block 20000 --only ref
build_sampler/yue2-sampler-diff --cases 20000 --seed 7 --shipped --block 20000 --only new
```

| | per call |
|---|---:|
| `sample_step_ref` (stage 5) | **1.046 ms** |
| `sample_step` (stage 6) | **0.240 ms** |
| | **4.4× faster** |

**These two measurements disagree and the coordinator should know it.** The
microbenchmark says the sampler got 0.81 ms/call cheaper; the end-to-end CPU
says 0.15 ms/token. The gap is stable across repeats, survives
`--block 20000` (no per-case logit churn) and `MALLOC_MMAP_THRESHOLD_`/
`MALLOC_TRIM_THRESHOLD_` at 256 MB (so it is not page-fault churn from the
stage-5 code's 740 KB scratch in a small-heap process), and I did not find the
cause. **Trust the sweep**: it is the real workload, both binaries were run back
to back on an idle box, and the wall and the CPU moved by the same 0.15 ms —
which is the honest saving. The corollary is that STATUS_BATCH §1's
"`sample_step` is most of it" was **wrong**: it was ~0.4 ms of a 4.5 ms step,
not 2.4 ms, and removing it does not unlock the AR loop on this card.

## 2. Equivalence (SPEC_SAMPLER §4)

The sampled token sequence is unchanged. Four independent pieces of evidence:

**1. Goldens** (CPU, `tests/out/yue2-ar-f16.gguf`, exact):

```sh
build_sampler/yue2 ar -m tests/out/yue2-ar-f16.gguf \
	--request tests/out/alley_swing_s1_request.json \
	--artifacts tests/out/sampler/golden_greedy32 --greedy --max-abc 32 --max-semantic 8 \
	--device cpu --threads 12
build_sampler/yue2 ar -m tests/out/yue2-ar-f16.gguf \
	--request tests/out/alley_swing_s1_request.json \
	--dump-logits tests/out/sampler/golden_logits.npy --device cpu --threads 12
```

| golden | result |
|---|---|
| `tests/golden/ar_greedy_32.npy` vs `abc_tokens.npy` | **IDENTICAL (32/32)** — the greedy path |
| `tests/golden/ar_last_logits_f32.npy` vs `--dump-logits` | argmax **55 = 55**, max abs Δ **0.001673** — the same figures STATUS_AR §1 recorded; `--dump-logits` does not call `sample_step` at all |

**2. `--verify-sampler`, full songs on the Arc.** The new flag runs
`sample_step_ref` on a *copy* of the sequence's RNG before `sample_step` runs on
the real one, then compares both the token and the resulting RNG state (so a
sampler that drew a different number of times is caught too) and dies on the
first disagreement. Two runs cover every phase/`legacy_off` combination — the
`cot=off` request needs an explicit `"cfg_scale": 1.0`, because the default 1.01
for `cot=off` is rejected as classifier-free guidance:

```sh
build_sampler/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/sampler/r_full.json \
	--artifacts tests/out/sampler/verify_full --device vulkan --gpu 1 --verify-sampler
build_sampler/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/sampler/r_off.json \
	--artifacts tests/out/sampler/verify_off  --device vulkan --gpu 1 --verify-sampler
```

| run | phases covered | steps verified | mismatches |
|---|---|---:|---:|
| `cot=full` | abc (`legacy_off=0`, temp 0.7, top_k 30) + semantic (`legacy_off=0`, temp 1.0, top_k 100) | 1 949 + 3 912 = **5 861** | 0 |
| `cot=off` | semantic only, **`legacy_off=1`** (`keep_head = 3`) | **4 767** | 0 |

**3. Differential test, 340 000 synthetic cases, zero mismatches.**
`tests/sampler_diff.cpp` → `yue2-sampler-diff`. Each case draws a fresh
`(logits, history, step, phase, legacy_off, Sampling, rng seed)` and asserts both
samplers return the same token *and* leave the same RNG state. Five logit shapes
rotate per block — `normal`, `quantised` (rounded to ¼, so exact ties at the
top-k boundary are the rule, not an accident), `coarse` (rounded to 1),
`flat` (every entry equal), `spiky` — each block seeded fresh and then perturbed
per case, including a deliberate tie injection of 1–200 ids sharing one value and
an `end` token that lands on that value a third of the time. ±inf are sprinkled
in; NaN never is (§4.2b). The grid: `temperature` 0 / 0.7 / 1.0 / 1.3,
`top_k` 1 / 2 / 3 / 30 / 100 / 400, `top_p` 0 / 0.5 / 0.9 / 0.95 / 1.0,
`repetition_penalty` 1.0 / 1.005 / 1.2 / 2.0, `penalty_window` 0 / 1 / 50 / 100,
`step` on `min_tokens - 1` / `min_tokens` / `min_tokens + 1`, histories up to 160
ids with repeats, out-of-range ids and the end tokens.

```sh
for seed in 20260912 1 777; do build_sampler/yue2-sampler-diff --cases 100000 --seed $seed; done
build_sampler/yue2-sampler-diff --cases 20000 --seed 4242 --block 1       # a fresh buffer per case
build_sampler/yue2-sampler-diff --cases 20000 --seed 99   --block 20000   # one buffer, 20k perturbations
```

| run | cases | mismatches |
|---|---:|---:|
| seed 20260912, block 100 | 100 000 | 0 |
| seed 1, block 100 | 100 000 | 0 |
| seed 777, block 100 | 100 000 | 0 |
| seed 4242, block 1 | 20 000 | 0 |
| seed 99, block 20000 | 20 000 | 0 |
| seed 20260912, block 100, re-run on the final binary | 100 000 | 0 |
| **total** | **440 000** | **0** |

**4. Byte-identical song.** One seeded `yue2 song` on the Arc, before and after:

```sh
build_sampler_base/yue2 song --request tests/out/sampler/r_full.json \
	--out tests/out/sampler/song_base.flac --artifacts tests/out/sampler/song_base \
	--seed 424242 --device vulkan --gpu 1
build_sampler/yue2      song --request tests/out/sampler/r_full.json \
	--out tests/out/sampler/song_new.flac  --artifacts tests/out/sampler/song_new \
	--seed 424242 --device vulkan --gpu 1
```

`cmp` is clean on `semantic.npy`, `prefix.npy`, `abc_tokens.npy`, `score.abc`,
`nar_noise.npy`, `latent.npy` and the 33.5 MB FLAC — 1 789 abc ids and 4 082
codes on both sides.

Evidence 2, 3 and 4 were then **re-run on the final binary** (the one that also
carries the line-buffered stdout and the progress line) against the same
unchanged `build_sampler_base`: 5 861 + 4 767 verified steps, 100 000 more
differential cases, and `semantic.npy` / `prefix.npy` / `latent.npy` / FLAC
identical again (`tests/out/sampler/song_new2`).

## 3. Deviations

| # | SPEC_SAMPLER said | what was done, and why |
|---|---|---|
| D1 | §2/§3: select top-k on **untempered** scores with a slack of `k + 64`, temper those, then take the threshold among them | Both scans temper. The slack trick is only safe while no more than 64 candidates below the k-th collapse onto the threshold after `x/t` rounds, which is a bound on float rounding that nothing enforces — and the cost it saves is one divide per allowed id in the abc phase and *zero* in the semantic phase, where `temperature == 1`. Exactness was worth more than the instruction. The argument is in the comment above `sample_step`. |
| D2 | §4.2(a): a `--dump-sample-inputs DIR` flag *or* `--verify-sampler` | Only `--verify-sampler`, as §4.2 offers. It also compares the two RNG states, so a sampler that drew a different number of times fails even when the token happens to agree. Nothing writes 184 704 floats per step to disk. |
| D3 | — | `sample_step` now dies on `top_k < 1` instead of running off the front of the candidate vector. The stage-5 code computed `cand.begin() + (top_k - 1)`, which is undefined for `top_k <= 0`; no reachable config produces it (30 abc, 100 semantic), so this only turns latent UB into a message. |
| D4 | — | The `cot=off` path cannot be reached from an ordinary request: `prepare_request` defaults `cfg_scale` to 1.01 for `cot=off` and then rejects it as classifier-free guidance. `legacy_off = true` — the `keep_head = 3` branch — is therefore only reachable with an explicit `"cfg_scale": 1.0`, which is what `tests/out/sampler/r_off.json` sets. Worth the coordinator's eye: that branch is otherwise dead in this build. |
| D5 | — | `yue2-sampler-diff` grew `--only ref\|new` (run one sampler, for the microbenchmark in §1) and `--shipped` (use only the two real `Sampling` configs, in a song's abc:semantic ratio). Neither affects the differential mode. |
| D6 | — | Two additions at the coordinator's request, neither touching sampling: `setvbuf(stdout, nullptr, _IOLBF, 0)` first thing in `yue2` `main()` (the binary was fully buffered under a pipe), and one batch-level `ar: 84 s, 3 active (1 abc, 2 semantic), 12345 tokens, 486 tok/s` line every 10 s from `Runner::progress()` — never per step, and silent for a batch that finishes inside the first 10 s. |
| D7 | — | `yue2-sampler-diff` `#include`s `src/stage_ar.cpp` rather than linking it, because both samplers live in that file's anonymous namespace. It is a tests-only target behind `-DYUE2_BUILD_TESTS=ON` (default **OFF**), so the default build products are unchanged. |

## 4. Coordinator's call

- **The AMD column.** Filled by the coordinator after the agent finished (table
  under §1): +4.3 / +6.4 / +7.1 % aggregate tok/s at N = 1/4/8. The STATUS_BATCH
  remark that the N=1 process CPU is "~54 % of the step" was wrong — that CPU
  figure includes the fence-wait spin inside `llama_get_logits_ith` (see Review).
- **D4**: whether the `legacy_off` branch should stay, given `cot=off` is
  unreachable without hand-setting `cfg_scale`.
- **The microbenchmark/end-to-end gap in §1.** 0.81 ms/call saved in isolation
  vs 0.15 ms/token end to end, cause not found. It does not affect correctness
  or the decision to keep stage 6 (the sweep is a clean win at every N), but if
  the isolated figure is the true one there is another 0.6 ms/token hiding
  somewhere in the loop.
- **Was this worth it?** On the Arc, yes but modestly: +1.8 / +3.2 / +5.0 %
  aggregate tok/s at N = 1/4/8, and the AR loop is now GPU-bound there
  (`ms/step` unchanged). The AMD is ~2× the Arc's decode speed, so the same
  0.15 ms/token is a larger share of its step — the AMD column is where the
  return on stage 6 will actually show.

## Review

Cold review before commit (CPU + `build_review/`, device 1/0 untouched;
`cmake -B build_review -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON`,
`nice -n 10 cmake --build build_review -j8`; `YUE2_WARN_FLAGS` clean, the only
warning is the pre-existing `ggml-backend.h:419` `-Wshadow` from the submodule).

**Equivalence, read against `sampler_ref.hpp`.** Holds for every input:

- Allowed set: `EOD 151643 < ABC_END 151848` and `MUSIC_END 151852 <
  CODEC_OFFSET 151853`, so both segment pairs are ascending and disjoint — each
  allowed id is visited exactly once, in ascending order, which is what the
  single `pen` cursor and the greedy first-index rule both rely on. The `end`
  segment collapses to empty below `min_tokens` (`seg[phase_abc ? 1 : 0]`), the
  reference's `scores[end] = -inf`.
- Penalty: the reference penalises `scores[id]`, which for an allowed id *is*
  `logits[id]` (the re-enable writes `logits[end]` back before the penalty), and
  for a masked id is `-inf` → `-inf` (`alpha > 0`); the new code penalises
  `logits[id]` and never visits masked ids. Same ids, same values. An id in the
  window that is also outside the range is masked on both sides.
- D1 is exact: both scans compute the identical `v / t` for the same id, so the
  k-th largest tempered value and the `!(x < threshold)` filter are evaluated on
  bit-identical floats to the reference's single tempered pass. The min-heap
  holds the k largest *values* as a multiset, so its front equals
  `nth_element`'s k-th regardless of which tied ids it retained.
  `n_cand <= top_k` → the heap is the whole candidate set. The sort by
  `by_score_then_id` is a strict total order on distinct ids, so the survivor
  order — and therefore softmax, top-p and the draw — is identical.
- Greedy: `>` with `best = 0`, ascending visit order — the reference's first
  index on a tie; a fully masked step returns 0 on both sides.

**Fixed in review.** The `top_k < 1` die (D3) sat before the greedy branch,
which never read `top_k` in stage 5 — moved to just after the greedy return, so
`--greedy` with a degenerate `top_k` behaves as before and the heap code is still
guarded.

**Adversarial cases added** to `tests/sampler_diff.cpp`: (a) a one-ulp
`nextafter(tie, -inf)` injection in the per-case perturbation — distinct logits
that collapse to one float after `/ t` (common at `temperature 1.3`), the exact
D1 scenario; (b) 48 hand-built boundary cases — flat floor, exactly `top_k` ids
at one value, `end` at that same value (`k+1` on the threshold), one id an ulp
under, every temperature × phase × `end` masked/open. Results on this build:

| run | cases | mismatches |
|---|---:|---:|
| fixed boundary cases | 48 | 0 |
| seed 20260912, block 100 | 100 000 | 0 |
| seed 555, block 50 | 30 000 | 0 |
| seed 3, block 1 | 5 000 | 0 |
| seed 7, `--shipped --block 20000` | 20 000 | 0 |
| `tests/golden/ar_greedy_32.npy` vs `--greedy --max-abc 32` (CPU, f16) | 32 | `cmp` identical |

**Extras.** `setvbuf(stdout, nullptr, _IOLBF, 0)` is the first statement of
`main()`, before any stream use; the VAE child leaves through `execv` of the same
binary so it runs the same `main()` and gets the same buffering. The progress
line is wall-clock gated (`t_progress` starts at loop entry, so nothing prints
before 10 s), unprefixed, stdout only, counts one `sampled++` per `sample()`
call, and touches no artifact or `result.json` field.

**Docs.** No driver names, home paths or personal identifiers in the changed
text (the two `RADV` mentions in README/ROADMAP are pre-existing lines).
`docs/ROADMAP.md` quotes 340 000 differential cases — the pre-re-run count;
this section's runs bring the total past 600 000. D4 (`legacy_off` only
reachable with an explicit `cfg_scale: 1.0`) is the coordinator's call, as noted.

**The §1 discrepancy — a reading, not a measurement.** `llama_decode` does not
wait for the GPU: `ggml_backend_vk_graph_compute` records ~hundreds of
dispatches through `ggml_vk_build_graph`, submits in chunks and returns with
`submit_pending` set (`ggml-vulkan.cpp:16560-16572`); the wait is in
`ggml_vk_synchronize` → `ggml_vk_wait_for_fence` (`:2807`) — a sleep on the
`almost_ready` fence, then a `YIELD` spin — reached from `llama_get_logits_ith`
→ `ctx->synchronize()` (`llama-context.cpp:3862`). Two consequences. First,
the sweep's "CPU" includes that spin, so process CPU tracks wall, and the fact
that CPU and wall both moved by the same 0.15 ms/token is expected, not
suspicious — the wall delta is the honest sampler saving. Second, the N=8 gap
arithmetic (`wall/step 26.2 → 25.0 ms` vs `ms/step 23.1 → 23.0`) bounds the
in-situ stage-5 sampler at ≤ 0.39 ms/token, so the 1.05 ms microbenchmark
figure is the one that does not transfer — most plausibly the synthetic rows
make `nth_element` over ~151k abc candidates work harder than a real, sharply
peaked row does; a `perf record` of the base binary under `--only ref` on a real
dumped row would settle it. Where the rest of the host time goes per step: graph
build is skipped once the ubatch shape stabilises (`can_reuse`,
`llama-context.cpp:1350`; the KV `n_kv` padding step forces an occasional
rebuild + `ggml_backend_sched_alloc_graph`), `set_inputs` does one synchronous
`vk_buffer_write_2d` fence round-trip per input tensor (`ggml-vulkan.cpp:8806`),
then command recording, then the fence wait, then the N×740 KB logits memcpy out
of the staging buffer. None of that is yue2's. With ≤ 0.25 ms/token now outside
`llama_decode`, further CPU work in `yue2` cannot move batch scaling; the only
yue2-side lever left would be overlapping the sampler with the GPU (a second
context or pipelined slots), and on the Arc the loop is already GPU-bound.

**Verdict: safe to commit.**
