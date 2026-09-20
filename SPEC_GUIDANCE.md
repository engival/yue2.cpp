# SPEC_GUIDANCE — stage 7: guided semantic decoding (`cfg_scale` + mid-song style changes)

Contract for the C++ agent. Read `CLAUDE.md`, `SPEC_AR.md` §sampling, `SPEC_BATCH.md` §4.3–§4.7
and §6, and `SPEC_SAMPLER.md` §3 first. Everything here lives in `src/stage_ar.cpp` /
`src/stage_ar.hpp` plus one tests-only target. The NAR and VAE stages are not touched.

## 1. Why

The reference implements classifier-free guidance in the semantic phase only:

```python
logits = unconditional + cfg_scale * (conditional - unconditional)   # then the unchanged sampler
```

with a *negative* prefix that carries the instruction and the exact score but no tags and no
lyrics. `yue2 ar` rejects `cfg_scale != 1` today (`prepare_request`, "not supported (stage 2b+)").

The same machinery, generalised to **up to two negative branches with time-varying weights**
and a **positive prefix that can change at a given frame**, gives mid-song style changes.
Prototype findings that shaped this design (llama-server, two/three slots, one shared history):

- Swapping the tags mid-stream and continuing does nothing audible: after ~3 s of audio the
  codec history dominates the next-token distribution and the tags barely move it.
- Amplifying what is left of the tags' influence does work:
  `B + w·(B − previous_tags)` with `w ≈ 5–6` re-orchestrates the band coherently.
- A strong push (`w ≈ 11` for ~2.4 s) can bring in a voice the recording has never
  contained, but alone it destroys the lyrics. Adding a second push against the reference's
  blank negative, `+ 2·(B − blank)`, keeps the words intelligible through the change.
- So the engine needs: one formula, per-branch weight curves, a frame-addressed change of
  the positive prefix. *All musical judgement (where, how hard, how long) stays outside the
  engine* in the request JSON; wrappers generate it.

## 2. Semantics

### 2.1 The blend

At semantic step `t`, with `B` the logits row of the positive (primary) sequence and `N_i` the
rows of the live negative branches, all conditioned on the **same** generated history:

```
L = B + Σ_i w_i(t) · (B − N_i)          i ∈ { previous, blank }, each optional
```

- Computed in `float` (f32), only over the ids the semantic sampler can visit: `MUSIC_END`
  and `[CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)`. Other entries of the scratch row are
  never read; leave them untouched.
- `L` goes to the **unchanged** `sample_step` / `sample_step_ref` in place of the raw row:
  blend first, then repetition penalty → temperature → top-k → top-p → draw, exactly the
  reference order. One RNG draw per step, as today.
- The one sampled token is appended to **every** live branch. There is one `history`, one
  `step`, one `rng` per song — the branches are not independent songs.
- A branch with `w_i(t) == 0` contributes nothing; see §4.3 for when it is decoded at all.
- The abc phase is never guided.

### 2.2 `cfg_scale` (reference parity)

`cfg_scale = c`, `c != 1`, is exactly: blank branch live from semantic step 0 to the end with
constant weight `c − 1`, no `previous` branch, no prefix change.
(`uncond + c·(cond − uncond) ≡ cond + (c − 1)·(cond − uncond)`.)

Blank (negative) prefix, as `protocol.negative_prefix`:

- `cot != "off"`: `[EOD] + tokenize(instruction(cot)) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]`
- `cot == "off"`: `[EOD] + tokenize(instruction(cot)) + [MUSIC_START]`

The instruction is tokenised **on its own** (no trailing `"\n"`): its last token differs
from the one inside the positive prefix, where `".\n"` merges. Do not slice the positive
prefix. `abc_ids` are the exact ids of the positive branch (generated or external).

The default stays as the reference: `cfg_scale` absent → `1.0`, or `1.01` when `cot == "off"`.
With this stage the `1.01` default stops being an error and costs a second branch; a request
that wants today's behaviour keeps saying `"cfg_scale": 1.0`.

### 2.3 `guidance` (new request key)

```json
"guidance": [
  { "frame": 3400,
    "style": "…new tags…",
    "against": {
      "previous": [[0, 11], [60, 11], [85, 3]],
      "blank":    [[0, 2]]
    } }
]
```

- `frame` — semantic step at which this entry takes effect: tokens `history[0 .. frame)` were
  sampled under the old regime, token index `frame` is the first sampled under the new one.
  25 frames = 1 s. Entries are strictly increasing in `frame`; `frame >= 0`.
- `style` — the tags of the new positive prefix (same `Request::text()` recipe, same lyrics,
  same score). Optional: absent = the positive prefix does not change (curves only).
- `against.previous` — negative branch = the positive prefix that was in force **before this
  entry**. Requires `style` to be present and to produce a different prefix (else request
  error: nothing to push against). Not allowed on an entry with `frame == 0`.
- `against.blank` — negative branch = §2.2's blank prefix.
- A curve is a non-empty list of `[offset, weight]`: `offset` = integer frames after the
  entry's `frame`, non-decreasing, `>= 0`; `weight` finite, `|weight| <= 20`. Linear
  interpolation between keyframes; before the first keyframe the first weight holds, after
  the last the last weight holds. Two keyframes at the same offset = a step (the later wins
  at that offset).
- A new entry **replaces** all curves of the one before it. A branch kind that the new entry
  does not name has weight 0 from that frame on.
- An entry whose `frame` is never reached (song ended first) is reported in `plan.json` as
  unreached, not an error.

### 2.4 Opt-in and exclusions (request errors, via the `strf` path, before the model loads)

- `guidance` together with `cfg_scale` present and `!= 1` → error (say which to use).
- `guidance` or `cfg_scale != 1` with `abc_template` → error ("not supported yet").
- A guided job (either form) when `parallel > 1` → error for that job:
  "guidance needs --parallel 1". Unguided jobs in the same batch are unaffected.
- **A request with no `guidance` and effective `cfg_scale == 1` must run today's code path:
  same context parameters, same batch shapes, bit-identical `semantic.npy`.** This is the
  first acceptance test (§5.1). No shadow slot, no scratch row, no extra `n_seq_max`.

## 3. Interface

- Request: `guidance` parsed in `parse_request`, validated in `validate_request`. Unknown keys
  inside an entry or inside `against` are errors (this block is new; be strict).
- `request.json` must stay loadable by `SongRequest(**request.json)`: do **not** write
  `guidance` into it (same rule as `abc_template`). Write the normalised block to
  `guidance.json` in the artifacts dir, add it to `plan.json` (`"guidance"`: entries with
  `reached: true|false`) and to `plan_manifest.json`.
- `prefix.npy` stays the **frame-0 positive prefix** (the NAR consumes it; it ignores style
  text, and the prototype renders used exactly this). `semantic.npy` unchanged in meaning.
- `json_timing()["cfg_branches"]`: the maximum number of simultaneously live sequences
  for the song (1 unguided, 2 for plain `cfg_scale`, up to 3). Add `"guided_steps"`: number of
  semantic steps decoded with more than one branch, and `"branch_prefill_seconds"`.
- `ArResult::cfg_scale` / `config.json["cfg_scale"]`: the effective scalar as today
  (1.0 when `guidance` is used; the block itself is in `guidance.json`).
- `--dump-logits` on a guided request: dump the blended row of the first semantic step and,
  beside it, each branch's raw row (`FILE.npy`, `FILE.primary.npy`, `FILE.blank.npy`, …).
- No new CLI flags. `yue2 song` and `yue2 batch --parallel 1` get the feature through
  `run_ar_batch` for free; check that they do.

## 4. Program design (decided — do not re-litigate, but report if it breaks)

### 4.1 One song, up to three sequences

`Seq` stays "one in-flight song". A guided `Seq` additionally owns up to two **shadow
branches**:

```cpp
struct Branch
{
	int       slot;      // llama seq_id, >= parallel (i.e. 1 or 2 — guided songs run at parallel 1)
	int       kind;      // BRANCH_PREVIOUS | BRANCH_BLANK
	llama_pos pos;       // tokens this slot holds
	int       i_batch;   // row in the batch last submitted
	bool      live;
};
```

- Context: when (and only when) at least one accepted job is guided, `n_seq_max = 3` and
  `n_ctx = n_ctx_want * 3` (`kv_unified = false` as today). `n_ctx_want` must cover the
  longest branch prefix + `max_semantic` + 8. Print the KV budget line with the ×3.
- `Runner::parallel` keeps meaning "songs decoded side by side" (it is written to
  `result.json`); shadow slots are not songs. Do not overload it.
- `finish_job` clears the shadow slots too (`llama_memory_seq_rm` + the `pos_max == -1`
  assert) so the next job of a `--parallel 1` batch enters clean.

### 4.2 The guided step

In `Runner::run()`'s lockstep batch, a guided song adds `q.next` once per live sequence
(primary + live shadows), each at its own `pos`/`slot`, all with logits. After the decode:
fetch each row with `llama_get_logits_ith(ctx, i_batch)`, blend into a per-`Seq` scratch row
(allocated once when the song is entered, `n_vocab` floats — a decode step still allocates
nothing), call `sample()` on the scratch row, then `apply()` as today. `--verify-sampler`
therefore checks the blended row with no change to either sampler.

All rows must be read **before** any further `llama_decode` (the existing "sample every slot
before applying any" rule, `SPEC_BATCH §4.7`).

### 4.3 Branch lifecycle

- A shadow branch is **needed at step `t`** if its curve in the entry in force has any
  non-zero weight at or after `t`. A branch that is no longer needed is dropped
  (`seq_rm`, `live = false`) and no longer decoded — this is what keeps the cost to the
  transition window. A needed branch whose current weight happens to be 0 is still decoded
  (its KV must stay in step).
- A branch that becomes needed and is not live is **prefilled**: its prefix + `history[0 ..
  n − 1)` where `n = history.size()`, i.e. everything except the token that is pending in
  `q.next`; the normal lockstep batch then feeds `q.next` to every live sequence and yields
  the first blended row. Prefill runs in its own `llama_decode` calls (`decode_feed`), never
  inside the lockstep batch, and asserts `pos_max + 1 == expected` like `feed_tokens`.
- **Entry with a new `style` (the cut).** The old primary sequence already *is* "previous
  prefix + history": relabel it as the `previous` shadow (no recompute) if this entry needs
  `previous`, else drop it. Prefill a free slot with the new prefix + `history[0 .. n − 1)`
  and make it the primary. If the outgoing entry still had a live `previous` shadow, drop it
  first — three slots always suffice. The `blank` branch's prefix never changes; keep it if
  the new entry needs it.
- Codes are fed to branches as token ids (`code + CODEC_OFFSET`), i.e. `history` as is.
- `cfg_scale != 1` = blank branch needed from step 0: prefill it right after the primary's
  semantic entry (after the abc→semantic bridge, or after the direct `prefix_sem` feed).
- Slot contiguity (`SPEC_BATCH §4.7` item 3) is a performance matter only; prefer keeping
  live slots contiguous when choosing a free slot, and report what you measured.

### 4.4 Prefix construction

There is no function that builds a semantic prefix from `(text, abc_ids)` today — the recipe
is inlined three times. Introduce one helper and use it at all existing sites and for branch
prefixes; the existing sites must produce byte-identical `prefix.npy` (checked in §5.1).
The blank prefix gets its own helper (§2.2).

### 4.5 Context guard

Every branch prefix + `max_semantic` must fit `CONTEXT`; check at request preparation and
report the offending entry.

## 5. Acceptance (Arc = device 1, and `--cpu`; never device 0)

Use a short request of your own making (public, a few lines of lyrics, `--max-semantic` a few
hundred) plus one full-length song. Record exact commands.

1. **Unguided is untouched.** Same request, baseline binary (`build/`) vs this build:
   `cmp` on `prefix.npy`, `abc_tokens.npy`, `semantic.npy`, for `cot=full` generated score,
   external `abc`, and `abc_template`. Also `yue2 batch --parallel 2` on two unguided jobs.
2. **Blend arithmetic.** `--dump-logits` on a `cfg_scale: 3` request: the blended row equals
   `primary + 2·(primary − blank)` recomputed in numpy from the dumped branch rows, exactly
   (same f32 ops) over the allowed ids. Separately, the `blank` row equals a plain
   `--dump-logits` of a request whose prefix is the blank prefix, to the tolerance two
   separately prefilled sequences show on that backend (report it).
3. **Reference parity (CPU torch only, per `CLAUDE.md`).** First-step blended logits and a
   32-token `--greedy` continuation for `cfg_scale: 3` against the reference's
   `unconditional + cfg_scale * (conditional − unconditional)` — same style of golden as
   `ar_greedy_32` / `ar_last_logits_f32`, with a `*_meta.json`. If the reference cannot be
   run in your sandbox, say so in STATUS and leave the script under `convert/`.
4. **Prefix of a guided song.** A `guidance` entry at frame `F` (any weights): the first `F`
   codes of `semantic.npy` are identical to the unguided song with the same seed.
5. **Determinism.** A guided song run twice → identical `semantic.npy`.
6. **Lifecycle.** A curve that returns to 0 drops its branch (visible in `guided_steps` and
   in a `-v`/log line per branch event: prefill, relabel, drop, with step and slot); two
   entries in one song; an entry past the end (`reached: false`); `batch --parallel 1` with a
   guided job followed by an unguided one (slots clean, second job bit-identical to running
   it alone).
7. **`--verify-sampler`** over one full guided song: zero disagreements.
8. **Tests-only target** `yue2-guidance` (`tests/guidance.cpp`, the `#include "stage_ar.cpp"`
   pattern, `YUE2_BUILD_TESTS`): table tests for curve evaluation (holds at both ends, step at
   equal offsets, interpolation), "needed at/after t", and every request error of §2.3/§2.4.
   No model, no device.
9. **Cost.** Wall time of the semantic phase: unguided vs `cfg_scale: 3` vs a three-branch
   window of 500 frames, on the Arc. Expectation: roughly ×2 / ×3 inside the window only.

## 6. Determinism note for the README

A guided song decodes batches of 2–3 rows, so its low bits differ from an unguided decode:
`seed` reproduces a guided song only together with the same `guidance` block (and, as
before, the same card and GGUFs). Steps before the first entry are single-row and match
the unguided song exactly (§5.4).

## 7. Report

`src/STATUS_GUIDANCE.md`: `## 0. What it is`, then tables with exact commands (Arc unless
stated), the measured tolerances of §5.2, timings of §5.9, and a numbered deviations table
(`| G1 | … |`). Build in `build_guidance/` (`nice -n 10 cmake --build build_guidance -j8`).
Do not commit; the coordinator reviews and commits.

## 8. README / docs additions

- `README.md` request table: `cfg_scale` now honoured; new `guidance` key with the schema of
  §2.3, one worked example, the 25 frames/s rule, and the three presets below as starting
  points (they come from listening tests on one song — not laws):

  | intent | `previous` | `blank` |
  |---|---|---|
  | the band changes at a section | `[[0, 5]]` | `[[0, 2]]` |
  | a new lead voice enters | `[[0, 11], [60, 11], [85, 3]]` | `[[0, 2]]` |
  | gradual colouring | `[[0, 0], [750, 5]]` | — |

  Placement advice: start the entry about 1.4 s (35 frames) before the bar line of the section
  that should open in the new style; a push that starts inside a sung phrase garbles it.
- `docs/ROADMAP.md`: strike item 5 (`cfg_scale`); note guided songs at `--parallel > 1` and
  guidance with `abc_template` as the open follow-ups.
- `SPEC_AR.md`: replace the "`cfg_scale != 1` → error" sentence with a pointer here.
