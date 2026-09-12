# SPEC_SAMPLER — stage 6: sparse `sample_step`

Contract for the C++ agent. Read `SPEC_AR.md` §sampling, `src/stage_ar.cpp`
(`sample_step`, `Sampling`, the `Runner` decode loop from stage 5) and
`src/STATUS_BATCH.md` §1 (CPU/token column) first. **The sampled token sequence
must not change**: for every input, the new `sample_step` returns exactly the
token the old one returned. This is a pure CPU cost change.

## 1. Why

Per decode step the CPU receives 184 704 logits and `sample_step` does
O(n_vocab) work several times over: a full copy into a scratch vector, a mask
pass, a temperature pass, a candidate vector of every finite entry
(~151k in the abc phase), `nth_element` on it, then sort/softmax/top-p on the
≤ 100 survivors. Measured (`STATUS_BATCH.md` §1): 2.4 ms of a 4.5 ms step on
the AMD at `parallel 1` (54 %), 1.74 of 3.03 ms per token at `parallel 8` on the
Arc. The GPU idles meanwhile (decode is synchronous). Only `top_k` (30 abc, 100
semantic) candidates can ever be drawn, so the work above the top-k boundary is
waste.

## 2. Exact semantics to preserve (from the current code)

Order today: allowed-range mask → `end` token re-enabled → `min_tokens` masks
`end` → window repetition penalty over the last `penalty_window` generated ids
(`v<0 ? v*alpha : v/alpha`, `alpha = penalty^count`) → greedy: argmax over
penalised scores, **first index on tie** → else temperature (float division)
→ top-k: keep every candidate whose score is **≥ the k-th largest** (ties kept,
so > k may survive) → sort by score desc, id asc → softmax in double over the
survivors → top-p: cumulative in the sorted order, drop where
`cum − p_i > top_p`, always keep `keep_head` (3 for `legacy_off`, else 1) →
renormalise → one `uniform(rng)` draw → first index where `r < cumsum`, else
last.

Every one of those steps must give the identical result. In particular:
- Temperature is applied as `float / float` **before** top-k today. Since it is
  a positive scalar it does not change the order, so top-k may be selected on
  untempered scores **only if** the threshold comparison is done on the same
  values it was done on before (`!(cand < threshold)` on tempered floats).
  Safest: select top-k on penalised untempered scores, then temper the ≤ k+ties
  survivors and re-check nothing crosses the threshold differently — or prove
  that `x/t` is monotone in float for `t > 0` (it is for finite non-NaN x; ties
  in x stay ties, distinct x may collapse to equal only if they round to the
  same float after division — and that would *add* a tied survivor at the
  boundary). Handle that: do the ≥-threshold test on the **tempered** values
  exactly as before. Simplest correct route: heap-select the top k+ on
  untempered values with a *slack* (take k + 64), temper just those, compute
  the threshold as the k-th largest tempered value among them, then apply the
  original `!(v < threshold)` filter. Document the argument in a comment.
- Softmax/top-p run in double over the sorted survivors exactly as today.
- The greedy path stays a single argmax pass with first-index tie rule.

## 3. Design

- **No scratch copy.** Read logits in place (`const float *`); never allocate
  O(n_vocab) per call. Per-`Seq` reusable small buffers (≤ a few hundred
  entries) are fine.
- **Penalty sparsely.** The penalty touches at most `penalty_window` distinct
  ids. Build the frequency map from the history window as today, then keep a
  small `std::vector<std::pair<llama_token, float>>` of *penalised* values for
  those ids; the scan below consults it (sorted + binary search, or a
  `std::unordered_map` of ≤ 100 entries) instead of mutating a copy.
- **Allowed range as bounds, not a mask.** abc: `[0, EOD)` plus `end`;
  semantic: `[CODEC_OFFSET, CODEC_OFFSET+CODEC_SIZE)` plus `end`. Iterate only
  those ranges. `end` is excluded when `step < min_tokens`.
- **One linear scan** over the allowed range with a bounded min-heap of size
  k + slack (§2) holding (score, id); for the greedy path a plain argmax.
- Everything after top-k is the existing code operating on the small
  candidate vector — reuse it verbatim where possible (`by_score_then_id`,
  the softmax and top-p loops, the draw).
- Keep `sample_step`'s signature; the `Runner` loop does not change. If a
  per-`Seq` scratch is wanted, add it as an extra parameter with a default.

## 4. Proof of equivalence (this is the acceptance)

1. **Golden**: `tests/golden/ar_greedy_32` and `--dump-logits` unchanged
   (CPU, exact).
2. **Differential test, exhaustive on real logits**: add
   `tests/sampler_diff.cpp` (built as `yue2-sampler-diff`, tests-only target,
   not installed) that keeps the *old* `sample_step` under a different name and
   feeds both the old and the new implementation the same `(logits, history,
   step, phase, legacy_off, rng state)` and asserts equal tokens. Logit sources:
   (a) real logits dumped from a full sampled song: add a `--dump-sample-inputs
   DIR` debug flag to `yue2 ar` that writes, for every step, the logits row
   (float32 `[n_vocab]`), history and step — or, cheaper, drive the diff test
   *inside* `yue2 ar` behind `--verify-sampler` which calls both and dies on
   the first mismatch; (b) synthetic: random logits, random ties injected
   (duplicate values at the top-k boundary, values equal to the end token,
   NaN never), all four phase/legacy combos, `temperature` 0 / 0.7 / 1.0,
   `min_tokens` boundary steps. ≥ 100k synthetic cases + one full song per
   phase on the Arc with `--verify-sampler`. Zero mismatches.
3. **Byte-identical song**: one seeded `yue2 song --gpu 1` before/after, `cmp`
   on `semantic.npy` and the FLAC (same card, same `parallel`).

## 5. Report (STATUS_SAMPLER.md)

- CPU/token and step ms before/after at `parallel` 1, 4, 8 on the Arc (the
  `sweep.sh` method from STATUS_BATCH); leave an empty AMD column for the
  coordinator. State the new aggregate tok/s and how much of the step is still
  CPU.
- The equivalence evidence of §4 with exact commands and case counts.
- Deviations table.

## 6. README

Rewrite the reproducibility paragraph from stage 5 to this position (the
coordinator's decision, 2026-09-12): a `seed` **selects** a song; it is not a
promise to re-create it. Exact re-creation is the job of the artifacts —
`semantic.npy` / `latent.npy` from `--artifacts DIR` re-render bit-identically
through `yue2 nar` / `yue2 vae` on the same card with no seed involved. Same
seed + same card + same build + same `--parallel` and batch shape *will* repeat
a song, but treat that as a convenience, not a contract. Three or four
sentences; drop the `result.json` "batch" fields from the promise (keep writing
them). No driver names.
