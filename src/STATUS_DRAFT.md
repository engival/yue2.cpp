# STATUS_DRAFT — stage 12: speculative semantic decoding, EAGLE-3 head (SPEC_DRAFT.md)

Branch `feat/eagle3-draft`, **an experiment — not merged**. llama.cpp = fork dev `9bf5c9e47`
(the EAGLE-3 API is in its `src/llama-ext.h`). The draft head was trained with a separate,
unpublished trainer and is not distributed.

## Verdict

It works and it is correct, but the win is small. Exact mode: 1.15× on the semantic phase on
the Arc (Q8_0 head, K=2); lossy λ=3 adds ~1.2× more. The semantic phase is about a third of
a render (abc and NAR are not drafted), so a whole song gets ~5–10 % faster. The ceiling is
the task, not the head: codec tokens at temperature 1.0 put only ~26 % on the target's top
pick, acceptance tops out near 0.55, and more training data stopped helping. On these GPUs the
verify of 3+ rows is not free and every head call costs ~1–1.4 ms, so K > 2 loses. Not worth
the extra AR code and the dependency on libllama's staging header for master; kept here as
a reference implementation of EAGLE-3 on libllama without `common`.

## What was built

| piece | where |
|---|---|
| `sample_step` = `shape_step` (no RNG) + `draw_step` (one uniform, `inverse_cdf`) | stage_ar.cpp |
| accept / residual / overlap math, model-free | `src/draft_accept.hpp` |
| head runtime: ctx (target cparams, `n_ctx_seq`, 1 seq, `ctx_other`), extraction toggle, feature copy, encoder, 1-row draft decode, feed | `src/draft_eagle3.{hpp,cpp}` (llama_batch_ext, `llama.cpp/src` as SYSTEM include) |
| round (§3.3 steps 1–6), priming at S−1, trace, stats | `Runner::draft_prime/draft_round/draft_dist/draft_trace_row` |
| flags `--draft --draft-k --draft-lambda --draft-q --draft-window --draft-trace` on ar/song/batch; hidden `--draft-extract-only` (§6.1) | `parse_draft_arg`, `check_draft_args` |
| guard (§1): requests rejected per job, flags fatal | `draft_request_error`, `draft_params_error` |
| config.json `draft{file,sha256,type,k,lambda,q,window}`; result.json `timing.semantic.draft_*`, `tested_per_depth`, `accepted_per_depth` — only with `--draft` | stage_song.cpp |

Deviations / additions (small): names `shape_step`/`draw_step` (a free `draw` would be hidden
by `Runner::draw`); `tested_per_depth` added beside `accepted_per_depth` so the conditional
per-depth rate is exact; extraction is switched on in `Runner::feed` for any semantic-phase
feed (bridge, template hand-over, cot=off/given-abc entry), not only the apply bridge; K is
capped at `max_tokens − s − 1`; draft_trace residual rows are `depth 0, accepted 2`, bonus
rows `depth K+1, accepted 0` (the reject depth is the run of accepted rows before it + 1).
cot=off is guided by default (`cfg_scale` → 1.01), so it is rejected unless the request says
`"cfg_scale": 1`.

**SPEC §2.1 caveat found:** the eagle3 graph selects no output rows, and llama copies the
*first* n_outputs rows of logits / masked pre-norm (llama-context.cpp 2018–2034). So "same i
as logits" holds only when the output rows lead the batch. Our draft decodes are 1 row, the
encoder has all rows output — correct. A tried "fused feed" (accepted pairs sent with the next
seed row, seed last, non-output pairs first) read the wrong row: depth-1 acceptance 0.38.
With all rows output it was correct (0.544) but no faster (2.99 vs 2.95 ms/round), so it was
removed.

## Tests

| test | result |
|---|---|
| `yue2-draft-accept` (new): 9 (p,q) pairs × λ {1, 1.5, 2, 3}, 10⁶ rounds each, χ² at α=1e-4 vs p (λ=1) and vs §4.3's law (λ>1), supp ⊆ supp p, edges | PASS 41/41; KL(out‖p) at λ=3 up to 0.067 |
| `yue2-guidance` (+17 draft guard cases: 12 request, 5 flag) | PASS 271 |
| `yue2-sampler-diff` (10⁵ cases), `yue2-handover`, `yue2-bars` | PASS |
| `yue2 ar --verify-sampler`, one song, Arc | 5595 steps matched; semantic.npy == master song's |
| no `--draft` byte identity vs `build_master` (this tree at 24ef902), 2 songs, `yue2 song`, Arc (+ one again with the final binary) | FLAC + every artifact identical; plan.json / plan_manifest.json / result.json differ only in wall-clock fields and the hashes of them (key sets equal) |
| exact K 1..4 × window 0/2048, 7 songs each | all complete, min semantic length 3686 (≥ 200), `pos_last == q.pos − 1` asserted every round |
| `yue2 song --draft Q8_0 --draft-k 2`, one song | `len(semantic.npy)+1 == output_tokens` (4282); FLAC sounds right |
| YUE2_WARN_FLAGS | clean (only the pre-existing ggml-backend.h:422 -Wshadow from master) |

## Measurements (Intel Arc, `--gpu 1`, `yue2 ar --requests --parallel 1`, 7 held-out songs)

Songs: the first seed-1 take of 7 held-out songs, never in the
head's training data (request.json as recorded, abc phase re-run;
same scores in every variant). 35.7 k semantic tokens per variant. c = draft ms per drafted
token / baseline ms per token (8.152). Acceptance = conditional per depth (tested → accepted),
pooled over songs.

### Step 1: baseline and extraction price

| variant | semantic tok/s | vs base |
|---|---|---|
| base (no draft) | 122.66 | 1.000× |
| extraction on for the whole job, nothing drafted | 119.06 | 0.971× (abc phase −3 % too: 130 → 126 tok/s) |

The per-job toggle (semantic phase only) is what `--draft` does; the re-reserve per toggle is
lost in the noise.

### Step 2: exact K sweep, F32 head (as trained)

| variant | tok/s | vs base | tokens/round | acc d1 | d2 | d3 | d4 | draft ms/round | verify ms/round | c |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 w0 | 126.60 | 1.032× | 1.55 | 0.547 | | | | 2.86 | 9.13 | 0.350 |
| K2 w0 | 123.78 | 1.009× | 1.79 | 0.546 | 0.446 | | | 4.63 | 9.55 | 0.284 |
| K3 w0 | 110.92 | 0.904× | 1.90 | 0.553 | 0.445 | 0.398 | | 6.44 | 10.38 | 0.263 |
| K4 w0 | 93.24 | 0.760× | 1.92 | 0.549 | 0.446 | 0.365 | 0.348 | 8.15 | 12.10 | 0.250 |
| K1 w2048 | 129.32 | 1.054× | 1.55 | 0.555 | | | | 2.78 | 9.01 | 0.341 |
| K2 w2048 | 123.24 | 1.005× | 1.78 | 0.539 | 0.444 | | | 4.61 | 9.55 | 0.283 |
| K3 w2048 | 114.07 | 0.930× | 1.91 | 0.553 | 0.458 | 0.401 | | 6.23 | 10.22 | 0.255 |
| K4 w2048 | 95.28 | 0.777× | 1.93 | 0.552 | 0.450 | 0.382 | 0.333 | 7.97 | 11.98 | 0.244 |

Verify vs 1 row (8.15 ms incl. sampling): 2 rows 1.12×, 3 rows 1.17×, 4 rows 1.27×, 5 rows
1.48× — the Arc's verify grows fast past 3 rows, which caps K at 2.

Depth-1 acceptance by frame (pooled): flat, no fall past 2048 with the whole-song KV.

| variant | 0–512 | 512–1k | 1k–2k | 2k–3k | 3k–4k | 4k–6k | 6k+ |
|---|---|---|---|---|---|---|---|
| K2 w0 | 0.516 | 0.526 | 0.543 | 0.552 | 0.558 | 0.556 | 0.566 |
| K2 w2048 | 0.516 | 0.526 | 0.543 | 0.546 | 0.544 | 0.542 | 0.520 |
| K1 w0 | 0.541 | 0.542 | 0.549 | 0.564 | 0.559 | 0.520 | 0.556 |
| K1 w2048 | 0.541 | 0.542 | 0.549 | 0.561 | 0.560 | 0.561 | 0.577 |

Window: no difference beyond noise → keep `--draft-window 0`.

### Step 3: offline α match

| depth | trainer held-out (M3b, scheme B) | online, K4 w0 | online, all K2 runs |
|---|---|---|---|
| 1 | 0.486 | 0.549 | 0.539–0.549 |
| 2 | 0.384 | 0.446 | 0.444–0.459 |
| 3 | 0.316 | 0.365 | — |
| 4 | 0.261 | 0.348 | — |

Online is **higher** by ~5–9 points, not lower, at every depth — no sign of a feed/pairing
bug (those lose acceptance). Cross-check inside the K2 w0 traces: mean Σmin(p,q) at depth 1 =
0.5505 vs empirical acceptance 0.5464 (19 989 rounds), as exact rejection sampling requires.
Not explained here: the online takes are fresh Arc renders (different scores from the
recorded AMD takes), with F32 features (the trainer's cache stores F16).

### Step 4: draft weight type (`llama-quantize` copies, window 0, λ 1)

| head | sha256 | K | tok/s | vs base | acc d1 | d2 | draft ms/round | verify ms/round | c |
|---|---|---|---|---|---|---|---|---|---|
| F32 | fbd3ed0c… | 1 | 126.60 | 1.032× | 0.547 | | 2.86 | 9.13 | 0.350 |
| F16 | 3cdaaac3… | 1 | 131.35 | 1.071× | 0.554 | | 2.29 | 9.30 | 0.280 |
| Q8_0 | 54e2219f… | 1 | 133.22 | 1.086× | 0.558 | | 2.04 | 9.40 | 0.250 |
| Q4_K_M | 905e4f1c… | 1 | 141.35 | 1.152× | 0.546 | | 1.72 | 9.00 | 0.210 |
| F32 | fbd3ed0c… | 2 | 123.78 | 1.009× | 0.546 | 0.446 | 4.63 | 9.55 | 0.284 |
| F16 | 3cdaaac3… | 2 | 136.21 | 1.110× | 0.549 | 0.457 | 3.47 | 9.48 | 0.213 |
| **Q8_0** | 54e2219f… | **2** | **140.97** | **1.149×** | 0.544 | 0.459 | 2.95 | 9.49 | 0.181 |
| Q4_K_M | 905e4f1c… | 2 | 135.06 | 1.101× | 0.543 | 0.449 | 3.12 | 9.82 | 0.192 |
| Q8_0 | 54e2219f… | 3 | 127.03 | 1.036× | 0.555 | 0.451 | 4.29 | 10.40 | 0.175 |

Quantizing costs no acceptance (Q4_K_M included). Best measured: Q8_0 K2 1.149× and Q4_K_M
K1 1.152× (the Q4 K1 verify sample ran 0.3 ms/round faster than the others, so treat the two
as a tie). One full song, Q8_0 K2: 147.7 vs 131.2 tok/s = 1.13×.
Measured c is far above the bandwidth guess (0.065 for Q8_0): a 1-row draft
decode is ~1–1.4 ms on the Arc whatever the weight type — launch/sync bound, plus encoder and
feed calls (≈1 ms/round for K=1 on F32). The cost to attack next is llama_process call count
and per-call overhead, not bytes.

### Step 5: lossy curve (AMD 7900 XTX, full `yue2 song`, Q8_0 head, K=2)

Same 7 songs, 2 seeds. Same seed ⇒ same `score.abc` in every variant (the draft touches only
the semantic phase), so variants differ only in how the score is realised.

| variant | seed 1 tok/s | vs no draft | seed 2 tok/s | tokens/round | acc d1 | d2 |
|---|---|---|---|---|---|---|
| no draft | 225.8 | 1.000× | | | | |
| exact (λ 1) | 249.4 | 1.105× | 256.9 | 1.79 | 0.546 | 0.450 |
| λ 1.5 | 271.9 | 1.204× | 282.3 | 2.00 | 0.645 | 0.545 |
| λ 2 | 289.0 | 1.280× | 295.6 | 2.12 | 0.703 | 0.582 |
| λ 3 | 306.5 | 1.357× | 317.5 | 2.26 | 0.763 | 0.647 |

Listening (exact vs λ, side by side, one ear each): λ 3 could not be called worse than exact
— same song, small differences in timbre and accompaniment, lyrics drift and re-sync at
section boundaries. One example song: semantic 32.2 s exact → 26.3 s at λ 3 out of a 98 s
render (abc 16 s and NAR 44 s are not drafted).

## Exact commands

```sh
# build (own dir, tests on)
cmake -B build_draft -DCMAKE_BUILD_TYPE=Release -DYUE2_BUILD_TESTS=ON
nice -n 10 cmake --build build_draft -j8
build_draft/yue2-draft-accept -v; build_draft/yue2-guidance; build_draft/yue2-sampler-diff

# draft head copies (llama-quantize accepts arch eagle3 unpatched)
llama-quantize draft.gguf draft-Q8_0.gguf Q8_0

# measurement sweep: AR only, 7 requests, one variant per run
yue2-ar -m yue2-ar-q8_0.gguf --requests jobs.json --parallel 1 --gpu 1 \
  --draft draft-Q8_0.gguf --draft-k 2 [--draft-window W] [--draft-lambda L] --draft-trace

# a full drafted song
yue2 song --request request.json --out song.flac --artifacts art \
  --ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf --vae yue2-vae-f32.gguf \
  --gpu 1 --draft draft-Q8_0.gguf --draft-k 2 --draft-trace
```

`draft_trace.npy` columns are in SPEC_DRAFT §5; acceptance per depth is in result.json
(`timing.semantic.tested_per_depth` / `accepted_per_depth`).

## Open (not planned)

- Fewer / cheaper head calls per round (encoder + feed + K draft decodes are K+2
  `llama_process` calls); Q4_K_M at K1 hints the per-call floor is ~1.7 ms/round.
