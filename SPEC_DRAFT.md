# SPEC_DRAFT — speculative semantic decoding with an EAGLE-3 draft head (stage 12)

Status: draft spec, 2026-10-06, hardened the same day (review notes at the end). Branch
`feat/eagle3-draft` — an experiment, not merged (results and verdict: src/STATUS_DRAFT.md).
The measuring harness *is* the opt-in feature. The draft head was trained with a separate,
unpublished trainer on our own renders; it is not distributed, so references to the
trainer's documents below are for provenance only.

Line numbers: `stage_ar.cpp` / `stage_song.cpp` as of commit `b15211d`; `llama.cpp` =
fork `dev` at `9bf5c9e47` (`llama-context.cpp`, `common/speculative.cpp`,
`src/models/eagle3.cpp`, `tools/server/server-context.cpp`).

## 1. Goal and scope

Speed up the **semantic phase** of unguided songs. A small draft head (one decoder
layer, trained separately) proposes K codec tokens. Then one target `llama_decode`
over K+1 rows verifies them. Two acceptance modes:

- **exact** (`--draft-lambda 1`, the default): rejection sampling. The output tokens are
  distributed exactly as the plain sampler's, so speed is the only thing that changes.
  The seed→song mapping does change (RNG use differs), but it is deterministic for a
  given seed + head + K + λ + q mode + window. "Exact" is distributional: p_j comes from
  a K+1-row verify batch, whose kernels differ from the 1-row decode by float noise, so
  it is not bit-identical to the sequential sampler on the same seed either.
- **lossy** (`--draft-lambda λ > 1`): drafted tokens the target finds plausible are
  accepted more readily (§4.3). This is faster, and the song depends on the draft's
  taste. It is the listening test of the head.

Out of scope, so the draft is **rejected with a clear error** (at validation, where
`is_guided` is checked, stage_ar.cpp 7686–7697) if any of these is on:

| condition | where it is decided |
|---|---|
| `is_guided(req)`: `guidance`, `cfg_scale ∉ {null, 1}`, `cfg_score ≠ 1`, `sections` that are not a plain swap | `is_guided` 345, `score_guided` 338, `sections_plain_swap` 324, `cfg_scalar` 316 |
| `semantic_keep` (keep_codes) | the keep validation just above 7686 |
| `handover` | request validation (`SPEC_HANDOVER`) |
| `--greedy`, `--verify-sampler`, `--dump-logits`, `--prefix-only` | `ArParams` (stage_ar.hpp 8–63), `parse_ar_args` 7501 |
| `--parallel > 1` (v1) | the clamped `parallel` at 7684 |

A plain-swap `sections` request only changes the score (`plain_swap`, 3950), so it is
allowed. Score templates (`abc_template`) only touch the abc phase (`in_hole` /
`tpl_cont` are abc-phase states) and are allowed. `cot == "off"` is allowed — both p
and q are shaped by the same `legacy_off` rule, so exactness holds — but the head was
not trained on such takes (the training set excluded them), so expect lower acceptance.
The abc phase decodes exactly as today.

**No `--draft` flag ⇒ byte-identical output to master** (hard requirement, tested).

## 2. Inputs

- Draft head GGUF, arch `eagle3`: `draft.gguf`
  (run "M3b", 561 MB on disk, all tensors F32). It carries its **own** `output.weight` (F32,
  [2048, 32769] — the d2t slice of the target's head), `output_norm`, `fc`, `blk.0.*`
  and `d2t` (I64 [32769], `d2t[i] = 151852 + i`: row 0 = `MUSIC_END`, rows 1..32768 =
  `<codec_0..32767>`). It has **no `token_embd`**: that is the only tensor it borrows
  from the target, through `cparams.ctx_other = ctx_tgt` (`eagle3.cpp:165–171`; the
  loader rejects a `ctx_other`-less context for this arch, `llama-context.cpp:158–163`).
  3 target extract layers (`llama_model_target_layer_ids`, expected 2/14/25 — assert
  `_n == 3` and all `< n_layer` 28, as `speculative.cpp:559–565, 603–612` do). Its
  logits row is full-vocab (184704) with −inf outside the d2t set (`eagle3.cpp:310–319`),
  so `sample_step`'s shaping runs on it unchanged.
- llama.cpp `dev` already has the runtime pieces. **Header trap**: everything EAGLE-3
  specific is in the staging header **`llama.cpp/src/llama-ext.h`**, not
  `include/llama.h`: `llama_set_embeddings_layer_inp`, `llama_get_embeddings_layer_inp`,
  `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn(_ith)`,
  `llama_model_target_layer_ids(_n)`, `llama_get_ctx_other`. Add `llama.cpp/src` to the
  include path of the `yue2` targets.
  `llama_process(ENCODE|DECODE)`, `llama_batch_ext_*` and `llama_context_params.ctx_other`
  (llama.h:424) are public.
- **Reference implementation to port, not link:** `common/speculative.cpp`
  `common_speculative_impl_draft_eagle3` (514–980: ctor 545–620, `process` 652–795,
  `draft` 797–918, `accept` 920–935; the pairing/deferred-boundary comment 478–513).
  yue2 links only libllama (no `common`), so the port uses `llama_batch_ext` directly:
  `llama_batch_ext_add_token` + `llama_batch_ext_set_embd_token` (the (token, g) pair)
  + `llama_batch_ext_set_pos` + `llama_batch_ext_set_output_logits`; encoder rows via
  `llama_batch_ext_add_embd` with `n_embd = 3·2048` (what `common_batch::add/set_embd/
  add_embd` wrap, common.h:1057–1103). Port into `src/draft_eagle3.{hpp,cpp}` (~250 lines).
  Do not use upstream's sampler: it takes the top-1 with `p_min`, and we need the
  draft's full distribution.
- How ctx_dft is created upstream: `common_speculative_init_result` ctor,
  `speculative.cpp:2593–2640`: the **target's** `llama_context_params`, then
  `n_ctx = llama_n_ctx(ctx_tgt)`, `n_rs_seq = 0`, `ctx_other = ctx_tgt`,
  `llama_init_from_model(model_dft, cparams)`. The draft model is loaded with the same
  `llama_model_params` as the target (device list from `vulkan_device(gpu)`, `load_model`
  3842–3870) — same Vulkan device, all layers offloaded.

### 2.1 libllama facts the port relies on (verified, `llama-context.cpp`)

| fact | where |
|---|---|
| `llama_set_embeddings_layer_inp(ctx, L, on)` is a context-lifetime flag, settable any time; every call sets `sched_need_reserve`, so the **next decode re-reserves the graph** (tens of ms). Toggle at phase boundaries only, never per decode. | 1239–1247 |
| Layer-input rows are extracted for **all tokens of every ubatch**, independent of `batch.logits`, into a per-layer host buffer of `n_embd × n_batch` floats, dense in **batch order** (reordered on read). Row i of the last `llama_decode` = batch index i. Overwritten by the next decode on that ctx — copy out first. | 2015, 2280–2311, 2153–2156, 2225–2229, 2381–2387 |
| Enabling the layers grows the output buffer and reallocates it (views are re-laid out on every reserve), so turning extraction on after the first decode is safe. | 2160–2232 |
| Cost when on: the 3 tensors are marked graph outputs (no in-place reuse) + 3 async D2H copies of `n_tokens × 2048 × 4 B` per ubatch (24 KB/token). Measure it (§6.1). | llama-graph.cpp 1410–1415, 2307 |
| ctx_dft with `llama_set_embeddings_nextn(ctx_dft, true, masked=true)`: the pre-norm row exists **only for output rows**, read with `llama_get_embeddings_nextn_ith(ctx_dft, i)` using the **same i as `llama_get_logits_ith`**. Draft logits and prenorm come from **one** `llama_process(DECODE)` — upstream `draft()` does exactly this (854–856). §9 Q2 is answered: yes. | 1014–1042, 2018–2034 |
| `llama_process(ENCODE)`: all rows are outputs (`n_outputs = n_tokens`), the result is dense `[n_tokens, 2048]` from `llama_get_embeddings_nextn(ctx_dft)`; requires `n_ubatch ≥ n_tokens`; runs **without the KV** (positions are placeholders, nothing is written). Each encoder row must be flagged output (`add_embd(..., output=true)`) so the masked nextn row exists. | 1475–1560 (1507, 1532, 1540, 1624–1631) |
| Output rows per decode are capped by `n_outputs_max` (0 = `n_batch`, llama.h:371). yue2 leaves it 0, so K+1 ≤ 9 logits rows on the target and K+1 encoder rows on the draft need nothing. The eagle3 model's pooling resolves to NONE (218–223), which the nextn extraction requires (2024). | — |
| `llama_memory_seq_rm(mem, seq, p0, p1)` removes `[p0, p1)`; `p1 = −1` means to the end, `p0 = −1` from 0. | llama.h:756–761 |

## 3. Semantic loop with a draft (parallel 1)

Notation: s = `q.step` = semantic tokens emitted so far = index of the next token to
sample. The target's sampler-shaped distribution for the token at index s+j given
history H = `p` (exactly what `sample_step` builds: allowed set, end mask when
`s+j < min_tokens` 200, penalty 1.2 over the last 50 of H, top-k 100, top-p 0.95,
renormalised; ≤ 100 pairs). **H includes the drafted tokens before that position**
(§3.3). The draft's distribution `q` = the same shaping applied to the draft's logits
row with the same H and step (`--draft-q shaped`, default), or a plain softmax over
the 32769 d2t ids (`--draft-q raw`, ablation). Training targeted shaped labels, so
shaped should win.

### 3.1 Refactor `sample_step` (no behaviour change)

`sample_step` (1808–2002) ends with the draw at 1990–2001: one
`std::uniform_real_distribution<double>(0,1)(rng)` and an inverse-CDF walk over
`sc.cand[i].second` (ids) / `sc.prob[i]` (double, renormalised), falling back to
`sc.cand.back()`. Split it into `shape(logits, n_vocab, s, history, step, phase_abc,
legacy_off, SampleScratch &, mask_end)` — everything above the draw, consumes no RNG,
leaves `sc.cand`/`sc.prob` as the distribution — and `draw(const SampleScratch &, rng)`.
`sample_step` = `shape` + `draw`, token- and RNG-identical. `--verify-sampler` and
`tests/sampler_diff.cpp` (`yue2-sampler-diff`, built with `-DYUE2_BUILD_TESTS=ON`,
CMakeLists 89–95; it `#include`s `src/stage_ar.cpp`) must still pass. The speculative
path calls `shape` for every p and q, and `draw` for the drafted tokens and the bonus.
Semantic-phase arguments are those of `Runner::sample` 4175–4202: `s_sem`,
`phase_abc = false`, `legacy_off = js.req.cot == "off"`, `mask_end = false`.

### 3.2 Draft context lifetime

- Loaded and created once per `ar_decode_jobs` (6378) when `--draft` is given, after the
  target ctx (6686–6713), on the **same Vulkan device**. cparams = a copy of the
  target's (`n_batch = n_ubatch` ≥ 512, F16 KV, FA auto) with `n_ctx = llama_n_ctx_seq(ctx)`
  (the target's per-slot context already covers semantic max + prefix), `n_seq_max 1`,
  `kv_unified` default, `ctx_other = ctx`. llama pads `n_ctx` up to a multiple of 256.
  Draft seq_id is always 0 (the target's slot is `q.slot`, also 0 at parallel 1).
- `llama_set_embeddings_nextn(ctx_dft, true, true)` once at creation.
- The target enables layer-input extraction for the head's 3 layers at the **phase
  switch** (`Runner::apply` bridge, before `feed(q, bridge, …)` 5011) and disables it in
  `finish_job` (5813) — the semantic phase only. Each toggle costs one graph re-reserve
  (§2.1), i.e. two per job. §6.1 also prices leaving it on for the whole job.
- Draft KV positions are **absolute target positions**, starting at the semantic start
  row `S−1` (the `MUSIC_START` row, whose features predict `t_S`). This matches training
  exactly: cache row r ↔ position `S−1+r`, the
  trainer RoPEs at the absolute `S−1+r(+k)`, depth-0 row r
  = `(t_{S+r}, fc·feat[S−1+r])` = exactly the pair §3.3 writes at position S−1+r, and the
  chained row m (`(d_m, prenorm_{m−1})` at `pos_last+m`) attends to the same key set the
  trainer's depth-m mask allows (depth-0 rows ≤ origin + its own earlier chained rows;
  derived from `speculative.cpp:770–772, 824–827, 856–892`). Prefix rows
  are never fed to the draft. The draft's RoPE is NORM mode (its own hparams); nothing
  to set.
- `--draft-window N` (default 0 = whole song). Each round, before drafting, keep only the
  last N depth-0 rows: `llama_memory_seq_rm(mem_dft, 0, 0, pos_last − N + 1)` when
  `pos_last − N + 1 > S−1`. Training saw ≤ 2048-row windows at **random chunk offsets** so a window is closer to training than the whole-song KV;
  this A/B settles whether the whole-song KV hurts.

### 3.3 One round

Invariants at the top of a round (parallel 1, no branches): target KV holds positions
`0..P−1` with `q.pos == P`; `q.next = t_P`, already in `q.history` (apply 4905–4909
pushed it and bumped `q.step` to s); draft KV holds depth-0 rows for `S−1..P−2`; the
deferred boundary is `(g_last, pos_last)` with **`pos_last == P−1 == q.pos − 1`** and
`g_last` = encoder output of the target features at `P−1`. (After the bridge: P = S,
`q.next = t_S`, the draft KV is empty, boundary `(g[S−1], S−1)`.)

1. **Draft** (mirror `draft()` 797–918). `seq_rm(mem_dft, 0, P−1, −1)` (a no-op in
   steady state, as upstream 824). Seed row `(q.next, g_last)` at `P−1`, output. Decode.
   Row 0's logits shaped with `H_0 = q.history`, step s → `q_1`; draw `d_1 ~ q_1`
   (`draw`, 1 RNG draw); `prenorm_0 = nextn_ith(0)`. Then for i = 2..K: row
   `(d_{i−1}, prenorm_{i−2})` at position `P+i−2`, output; decode; shape with
   `H_{i−1} = q.history ++ [d_1..d_{i−1}]`, step `s+i−1` → `q_i`; draw `d_i`. Stop after
   `d_i == MUSIC_END` (it stays the last drafted token and is verified). `K_eff` =
   tokens drafted. Keep every sparse `q_i` (≤ 100 pairs) for the residual. One
   `llama_process` per depth (K_eff calls, 1 row each).
   Note `MUSIC_END` can be drafted only when `s+i−1 ≥ 200` (shaped q masks it like p;
   with `--draft-q raw` it is **not** masked, so mask it there explicitly — the trainer's
   labels never contain it below 200 and the target would reject it anyway).
2. **Verify**: one target `llama_decode` with rows `r = 0..K_eff`: row 0 = `q.next` at
   position P, row r = `d_r` at `P+r`, `logits = true` on every row (`batch_add` 2008).
   Row r's logits shaped with `H_r`, step `s+r` give `p_{r+1}` (the distribution for the
   token at index s+r). Immediately copy the 3 layer-input rows `0..K_eff` out of
   `llama_get_embeddings_layer_inp(ctx, L) + r·2048` into a `[K_eff+1, 3·2048]` scratch
   in `--layers` order (low, mid, high — `speculative.cpp:685–700`).
3. **Accept** walk j = 1..K_eff: draw `u_j ~ U(0,1)` (same `uniform_real_distribution`
   as `draw`, 1 RNG draw); accept iff `u_j < min(1, λ·p_j(d_j)/q_j(d_j))`
   (`p_j(d_j) = 0` when `d_j ∉ supp(p_j)`, so such a token is always rejected; `q_j(d_j)
   > 0` always, d_j was drawn from it). First reject at j: `n_acc = j−1`, then draw the
   replacement `x ~ r_j` (§4.1, 1 RNG draw) and stop. All accepted: `n_acc = K_eff`;
   if `d_{K_eff} == MUSIC_END` there is no bonus (the song is over; row K_eff's logits
   are unused), else draw the bonus `x ~ p_{K_eff+1}` (row K_eff, `draw`, 1 RNG draw).
   The round emits `d_1..d_{n_acc}` then x (n_acc+1 tokens, or n_acc when MUSIC_END
   was accepted last).
   **RNG order per round, fixed**: K_eff draft draws, then the accept uniforms in depth
   order (stopping at the first reject), then the one residual/bonus draw. Nothing else
   touches `q.rng` (`draft_trace` must not).
4. **Apply** each emitted token through `Runner::apply` (4878) in order: history, step,
   `MUSIC_END` → `finish_job`, `step == max_tokens` → truncated finish (4906–4928). Stop
   applying at the first terminal token; later emitted tokens are dropped. An accepted
   `MUSIC_END` with `s+j−1 < min_tokens` is impossible: p masks it → `p(d) = 0` → reject.
   Keep it that way, no special case.
5. **Rollback the target** (model on `finish_chords` 5253 / `rollback_hole` 5408):
   `llama_memory_seq_rm(mem, q.slot, P+1+n_acc, −1)` removes the rows of
   `d_{n_acc+1}..d_{K_eff}` (nothing when all were accepted — call it anyway, it is
   cheap); `q.pos = P+1+n_acc`; `q.next = x`. If the job finished in step 4, skip this:
   `finish_job` clears the slot (5929). `decodes++` once per llama_decode as today.
6. **Feed the draft** (mirror `process()` 652–795 + `accept()` 920–935, minus upstream's
   waste): run the encoder over rows `0..n_acc` of the feature scratch (`n_acc+1` rows,
   one `llama_process(ENCODE)`), giving `g[0..n_acc]`. `seq_rm(mem_dft, 0, P, −1)` to
   drop the chained rows (`P..P+K_eff−2`; the seed row at P−1 stays — it is exactly the
   pair a bridge would write). Then one `llama_process(DECODE)` with rows
   `k = 0..n_acc−1`: `(d_{k+1}, g[k])` at position `P+k`, no output (skip the call when
   `n_acc = 0`). New boundary: `g_last = g[n_acc]`, `pos_last = P+n_acc` = new `q.pos − 1`
   ✓. Rows past n_acc are never encoded or written. (Upstream encodes and writes all
   K rows, lets the next `draft()`'s `seq_rm(pos_last)` drop the rejected ones —
   478–513 call this out as waste — and relies on the server's
   `seq_rm(ctx_dft, ckpt.pos_max+1)` after drafting, server-context.cpp:3276, to drop
   the chained rows. Same KV content, fewer rows here.)

Bridge priming: `apply` 4955–5011 switches the phase (`q.rng.seed(seed)` 4973, so the
semantic RNG stream starts fresh as today) and `feed` 5039 → `decode_feed` 2024–2049
decodes the bridge in `n_batch` chunks; the last chunk's rows are in the layer-input
buffers in batch order, and the `MUSIC_START` row S−1 is batch index `q.i_batch` (the
same index `draw` 5053 reads logits from). Right after `feed_tokens` returns and before
`draw`, copy its 3 rows, encode them (1 row) → `g[S−1]`, set the boundary
`(g[S−1], S−1)`. The first semantic token `t_S` is then drawn exactly as today from the
prefill logits (`draw` → `sample` → `apply`), and the first round starts with P = S.
With `sec_cuts > 0` (plain-swap sections) the bridge is the whole `prefix_sem`, possibly
several chunks — the last chunk still holds S−1, so nothing changes.

Where the round lives: `Runner::run` 5941–6076 builds the lockstep batch at 5994–6020,
decodes at 6025, samples 6042–6064, applies 6065–6071. With `--draft`, a Seq in
`PHASE_SEM` with a live draft takes the round instead of steps 5994–6071 (parallel 1,
so it is the only Seq; keep `progress()` and the step/time counters meaningful:
count one lockstep step per round).

Bonus-row subtlety: if the walk stops at a reject at depth j, rows j+1..K_eff of the
verify are discarded. Their logits were computed with a wrong token and must not be
used, and their features are never encoded.

## 4. Acceptance math

### 4.1 Exact (λ = 1)

Accept d with probability `min(1, p(d)/q(d))`. On reject, sample from
`r(x) ∝ max(0, p(x) − q(x))`. Since `r(x) > 0` needs `p(x) > 0`, `supp(r) ⊆ supp(p)`:
walk p's candidate list (as `shape` leaves it: p descending, id ascending on ties) with
weights `max(0, p(x) − q(x))` (`q(x) = 0` when x is not in q's list), sum in double,
and draw by inverse CDF with one uniform — the `draw` routine on that list. If
`Σr ≤ 1e-12` (numerically p == q), draw from p instead. The output is distributed
exactly as p (Leviathan/Chen: for any proposal q from which d was actually drawn, the
emitted token is p-distributed). Validity of the shaped q as a proposal is not an
issue: all that is required is that `d` was sampled from the q used in the test and
that `p` is the target's distribution given the same prefix — and both p_j and q_j are
conditioned on `H_{j−1}` (§3.3), including the penalty window over drafted tokens and
the min_tokens mask at step `s+j−1`.

### 4.2 Unit test (required, CPU, no model)

Add `yue2-draft-accept` (`tests/draft_accept.cpp`, under `YUE2_BUILD_TESTS`, same
pattern as CMakeLists 89–118; put the accept/residual math in a header the test can
include without a model, e.g. `src/draft_accept.hpp`): random sparse p, q pairs
(including disjoint supports, q ⊃ p, p ⊃ q, a single-id p, near-equal p≈q), 10⁶ draws
each. A χ² test of the emitted token frequencies vs p must pass at λ = 1. For λ > 1 the
test records the measured output distribution's KL(out‖p) and checks that
support ⊆ supp(p).

### 4.3 Lossy (λ > 1)

Accept with `min(1, λ·p/q)` — this already rejects `d ∉ supp(p)` (p(d) = 0), which is
the guard "every emitted token is one the model could have drawn" (inside top-k 100 /
top-p 0.95 after the penalty). On reject, sample the residual of §4.1 as written (the
λ-free `r ∝ max(0, p − q)`). This is coherent (it is a proper distribution; the
emitted-token law is `out(x) = min(q(x), λp(x))·[x ∈ supp p] + (1 − A)·r(x)` with
`A = Σ_x min(q, λp)`, which sums to 1) and it is the intended bias: tokens with
`p < q ≤ λp` are emitted with the draft's weight q instead of p, and the rest of p's
mass is scaled down by `(1−A)/(1−Σmin(p,q))`. It is not p (lossy by design) and no
alternative residual makes it p while keeping the λ-acceptance. `--draft-lambda` is a
float ≥ 1, recorded in config.json. Entropy-scaled λ is a later option, not v1.

## 5. Flags and outputs

`yue2 ar`, `yue2 song`, `yue2 batch` (all three parsers and the param structs:
`parse_ar_args` 7501, `parse_song_args` stage_song.cpp 398, `parse_batch_args` 468;
`ArParams` stage_ar.hpp 8–63; the copy sites song→ar at stage_song.cpp ~631 and
batch→song at ~827, which also pins `parallel = 1`):

| flag | default | meaning |
|---|---|---|
| `--draft FILE` | off | EAGLE-3 head GGUF; enables speculation for eligible jobs |
| `--draft-k N` | 2 | tokens drafted per round (1..8) |
| `--draft-lambda X` | 1.0 | 1 = exact; > 1 = lossy |
| `--draft-q shaped\|raw` | shaped | how q is formed from the draft logits |
| `--draft-window N` | 0 | draft KV rows kept (0 = whole song) |
| `--draft-trace` | off | write `draft_trace.npy` |

`yue2 batch --draft` with `--parallel > 1` is an error in v1.

- **config.json** (written at stage_song.cpp 305–307 next to `backend`/`ar_gguf`):
  `draft` {file basename, sha256 of the head, weight type as the GGUF reports it
  (`general.file_type`), k, lambda, q, window}. Absent without `--draft`.
- **result.json timing** (`GenStats` stage_ar.hpp:66 + `json_timing` stage_ar.cpp 2146,
  semantic phase): `draft_rounds`, `draft_proposed`, `draft_accepted`,
  `accepted_per_depth[K]`, `draft_seconds` (draft decodes + encoder + feed),
  `verify_seconds` (the K+1-row target decodes), and the existing tokens/tps. Zero /
  absent without `--draft` so the no-draft JSON stays byte-identical.
- **draft_trace.npy** (follow the `guidance_trace.npy` pattern: `Artifacts::trace`
  2341, `write_artifacts` 2353/2391, `trace_step` 4806; never touches the RNG), one row
  per emitted token: `[frame, depth (1..K; K+1 = bonus; 0 = residual), accepted
  (0/1/2 = residual), p(tok), q(tok), Σmin(p,q) at that depth, H(p), p_max]`.
  Σmin(p,q) = 1 − TV is the per-frame drift/acceptance meter. In lossy mode, log p(tok)
  vs p_max for "how far off the model's own taste" each accepted token was.
- The end-of-job summary line: acceptance per depth, tokens/round, semantic tok/s.

## 6. Measurement plan (Arc only: `--gpu 1`)

All timing on the Intel Arc B70 (Vulkan device 1), on an otherwise idle GPU.

Prompts: 7 held-out songs (never in the head's training data), the first seed-1 take of
each: `cot = full`, `cfg_scale 1.0`, no plan keys. Each take's `request.json` is
self-contained and goes straight to `yue2 song --request` — note that re-runs the **abc
phase** too, so the score and the semantic prefix differ from the recorded take (same seed
⇒ same score as the master baseline, which is what the byte-identity test compares).
Outputs go to an eval dir outside the tree, one subdir per song and variant.

1. **Baseline**: master-equivalent run without `--draft` (also proves byte-identity vs
   `build/yue2` master on 2 of them: FLAC + every artifact). Then the same with layer
   extraction on for the whole job and no draft (a hidden `--draft-extract-only` or an
   env var; implementer's choice, documented) to price the extraction on both phases.
2. **Exact sweep**: K = 1, 2, 3, 4 with the M3b head, `--draft-window 0` and `2048`.
   Report semantic tok/s vs baseline, tokens/round, acceptance per depth, draft and
   verify ms per round, and the effective draft cost c (draft ms per drafted token over
   the baseline's ms per token). Check acceptance vs frame (does it fall past frame
   ~2048?).
3. **Offline α match**: the measured depth-1 acceptance should land near the trainer's
   held-out 0.486 (scheme B, shaped; M3b). The online number is on a
   fresh sample with F16 KV / Vulkan draft numerics (the trainer measured logits off by
   ≤ 6e-3), so "near" means within a few points. A big gap means a feed/pairing bug —
   investigate before tuning. First suspects: the (token[P+1], g[P]) shift, the
   `S−1` start, the layer order in the feature row, stale layer-input rows.
4. **Draft weight type**: F32 (as trained) vs F16 vs Q8_0 vs Q4_K_M, best K from step 2,
   λ = 1. Q4 is expected to save ≤ 5 %/round over Q8_0 and to lose acceptance; Q8_0 is the expected default.
   Make the copies with `llama.cpp/build_vulkan/bin/llama-quantize draft.gguf
   draft-f16.gguf F16` / `… Q8_0`. `llama-quant.cpp` quantizes only ≥ 2-D `*.weight`
   tensors that are not `*_norm.weight` (292–302) — so `fc`, `blk.0.attn_{q,k,v,output}`,
   `ffn_{gate,up,down}` and `output` (by default, `quantize_output_tensor`) — and copies
   `d2t` (I64, 1-D) and the norms; the loader it uses is arch-agnostic (971). **Check
   first** that it accepts arch `eagle3` and inspect the result's tensor types
   (`gguf-py/gguf/scripts/gguf_dump.py`); if it refuses, write the F16 copy with the
   trainer's GGUF writer instead. Record the head sha and
   file type per run. Measure draft ms/round and acceptance per depth for each.
   Bandwidth guess: 554 MB F32 / 277 MB F16 / ~147 MB Q8_0 against a 2.3 GB Q8_0
   target step ⇒ c ≈ 0.24 / 0.12 / 0.065, but 1-row decodes on the Arc may be
   launch-bound, so c is what the measurement says, not this.
5. **Lossy curve**: best K and weight type, λ ∈ {1.5, 2, 3}, 2 seeds × 7 songs. Report
   speed and the mean per-token KL proxy from the trace. FLACs under
   `<eval>/<name>/lambda_<λ>_s<seed>/` for a listening test.

Write results to `src/STATUS_DRAFT.md`: tables, exact commands, head sha and file type.

## 7. Tests and acceptance bars

- No `--draft`: byte-identical FLAC + artifacts vs master on 2 songs (Arc).
- `sample_step` split: `yue2-sampler-diff` + a `--verify-sampler` song pass.
- `yue2-draft-accept` χ² passes at λ = 1.
- Exact mode with `--draft-k 1..4` on one song: completes, `MUSIC_END` never before
  frame 200, `len(semantic)` consistent with result.json
  (`len(semantic.npy) + 1 == timing.semantic.output_tokens`), the draft
  invariant `pos_last == q.pos − 1` asserted every round, and decoded audio sounds like
  a song.
- A guard test: every out-of-scope combination in §1 is rejected with an error (extend
  `tests/guidance.cpp` / `handover.cpp`'s request-error pattern).
- Clean under `YUE2_WARN_FLAGS`; tabs, Allman braces; builds with
  `nice -n 10 cmake --build build_draft -j8`.

## 8. Where things are (stage_ar.cpp at `b15211d`)

- `Seq` 3982 (`history`, `step`, `rng` = `std::mt19937_64`, `scratch`, `pos`, `next`,
  `i_batch`), `Runner` 4068. `Runner::sample` 4175, `apply` 4878 (history/step 4905–4909,
  phase end 4921–4931), `finish_job` 5813 (slot cleared 5929). Lockstep `run()`:
  5941–6076, the one `llama_decode` at 6025.
- Bridge into the semantic phase: `apply` 4955–5011 (`q.rng.seed` 4973, `feed` call
  5011), `feed_tokens` 5017, `feed` 5039 → `draw` 5053. `batch_add` 2008, `decode_feed`
  2024–2049.
- `sample_step`: 1808–2002 (draw 1990–2001). `SampleScratch`: 1754. `scan_allowed` 1767.
- Context creation: `ar_decode_jobs` 6378–6790 (`n_ctx_want` 6389/6634, `max_feed`
  6390, cparams 6686–6697, `Runner` wiring 6728–6748), `load_model` 3842
  (`vulkan_device(gpu)` 3859).
- Guidance predicates: `cfg_scalar` 316, `sections_plain_swap` 324, `score_guided` 338,
  `is_guided` 345; `plain_swap` 3950. Guided jobs are **rejected** (not forced) when the
  clamped `parallel > 1`: 7684–7697.
- Trace pattern: `trace_step` 4806, `Artifacts::trace` 2341, `write_artifacts` 2353
  (npy at 2391/2402). `json_timing` 2146; `GenStats` stage_ar.hpp:66.
- CLI: `parse_ar_args` 7501 (flags 7507–7548); `stage_song.cpp` `parse_song_args` 398,
  `parse_batch_args` 468, copies ~631 / ~827, config.json 305–307.
- Tests: `tests/sampler_diff.cpp`, `guidance.cpp`, `handover.cpp`; CMake 89–118.

## 9. Open questions for the implementer to report (not to decide silently)

- Resolved by review (§2.1): extraction is a context flag with a re-reserve per toggle;
  draft logits + prenorm come from one `llama_process`. Still to **measure**: the
  extraction cost per token on the Arc (§6.1).
- Target verify cost on the Arc for K+1 = 2..5 rows vs 1 row (the "1.2×" assumption).
- Whether `llama-quantize` accepts arch `eagle3` (§6.4).

## Review notes (2026-10-06)

Checked against the files and commits named at the top, read-only, no builds, no GPU.
Changed:

- **§2**: the head has its own `output.weight`/`output_norm`/`fc`/`blk.0`/`d2t` (all
  F32, 561 MB); only `token_embd` comes via `ctx_other` (confirmed by `eagle3.cpp:165–171, 298–303` and the file size). The EAGLE-3 API lives
  in `src/llama-ext.h`, not `include/llama.h` — include-path note added. ctx_dft
  creation cited (`speculative.cpp:2593–2640`); the `llama_batch_ext` calls the port needs listed.
- **§2.1 (new)**: verified libllama facts with lines — layer-input rows are *all*
  rows of the batch (not output rows), dense in batch order, buffer n_embd×n_batch;
  the flag is togglable but each call forces a graph re-reserve; encode runs without
  the KV and needs every row flagged output; masked nextn rows share the logits index.
- **§3.3**: rewritten with explicit positions and invariants (`pos_last == q.pos − 1`),
  the seed row at P−1, chained rows at P..P+K−2, verify rows at P..P+K, rollback
  `seq_rm(P+1+n_acc)`, draft feed rows `k = 0..n_acc−1` at P+k with boundary at
  P+n_acc, and the chained-row `seq_rm(mem_dft, 0, P, −1)` that upstream leaves to the
  server (server-context.cpp:3276) — the original step 6 would have left duplicate
  cells at P..P+K−2. Upstream `n_accepted` = accepted draft tokens excluding the final
  sample (server 4314, speculative-simple 291) = our `n_acc` ✓. Added the
  all-accepted-and-last-is-MUSIC_END case (no bonus draw), the raw-q MUSIC_END mask,
  the feature copy-out timing, and where the round hooks into `run()`.
- **§3.2 / item 3**: the draft KV start at S−1 with absolute RoPE positions matches
  the trainer (cache row r ↔ S−1+r, absolute RoPE);
  the trainer's depth mask is the inference key set. Random chunk offsets at training mean
  a `--draft-window` is *closer* to training than the whole-song KV — noted.
- **§4**: exactness argument stated (shaped q is a valid proposal because d is drawn
  from the same q used in the test, and p/q share `H_{j−1}` and step); residual
  ordering fixed (p's candidate order, `supp(r) ⊆ supp(p)`); the lossy residual as
  written is coherent — its emitted-token law is spelled out, and the `d ∈ supp(p)`
  guard is shown to be implied by `p(d) = 0`.
- **§3.3 step 3**: RNG draw order per round fixed (K_eff draft draws, accept uniforms
  in depth order, one residual/bonus draw); `draw` is one
  `uniform_real_distribution<double>` on `mt19937_64` (1990–1991).
- **§1**: gating list mapped to the actual predicates (`is_guided` 345 etc.); "guided
  jobs force parallel 1" corrected to "rejected when parallel > 1" (7686–7697);
  templates and `cot = off` explicitly allowed with the caveat.
- **§5**: parsers/copy sites/config.json lines corrected (stage_song.cpp 631/827/305);
  trace depth column made unambiguous; weight type recorded.
- **§6**: the held-out list resolved (44 dirs, 7 names, all `_s1` takes eligible,
  request.json self-contained but re-runs the abc phase); outputs moved to an eval dir
  outside the tree; added the **draft weight-type axis** (F32/F16/Q8_0 via
  `llama-quantize`, with the quantizer's tensor rules cited and an "check it loads
  eagle3" step) with the bandwidth c estimates flagged as unmeasured; α match
  reworded (depth-1 in this spec's numbering, fresh sample, F16-KV numerics).
- **§8**: every line reference re-derived (`write_artifacts` is 2353 not 2387; Seq
  3982 / Runner 4068; bridge 4955–5011; `is_guided` 345; parallel clamp 7684).
- **§9**: Q1/Q2 answered from the code; what remains is measurement.

Risks not resolved here:

1. **`llama-quantize` on arch `eagle3`** is unverified (no build/run allowed in this
   review); the fallback (trainer's GGUF writer, F16) is in §6.4.
2. **Graph re-reserve cost** on each extraction toggle is not measured; if it is
   large, leave extraction on for the whole job and pay the abc-phase D2H copies
   (§6.1 prices both).
3. **Batch-vs-sequential numerics**: p_j from a K+1-row verify differs from the 1-row
   decode by float noise, so "exact" is distributional, not bit-reproducible against
   the sequential sampler (§1). The byte-identity test covers only the no-draft path.
4. **`--draft-q raw` and MUSIC_END**: the raw softmax does not mask it below step 200;
   §3.3 step 1 now masks it explicitly, but the raw mode is an ablation and gets no
   further hardening.
5. **The S−1 feature row after a multi-chunk bridge** (plain-swap sections re-prefill):
   relies on `decode_feed` returning the index in the *last* chunk (2024–2049) and the
   layer buffers holding only that chunk — true today; assert
   `llama_memory_seq_pos_max(mem, slot) == S−1` there.
