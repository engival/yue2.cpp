# SPEC — score templates: the model writes some lines of a given score

Builds on SPEC_AR.md and SPEC_BATCH.md (the `Runner` in `src/stage_ar.cpp`). Read
both first.

## 1. What it is for

Today the request's `"abc"` is all or nothing: absent, the AR samples a whole score;
present, the score phase is skipped and the text is sung as given. A **template** is
the middle: a score where some lines are given and some are left for the model.
The use it was built for: keep a finished song's `V: Vocal` lines and let the model
write a new `V: Ins` line under each, so the accompaniment is composed *for that
vocal* rather than improvised by the semantic stage over bars of rests.

It is forced decoding with holes. The score is one autoregressive text stream; at a
given line the token is known and fed, at a hole it is sampled. The KV cache cannot
tell the two apart, so every written line is conditioned on everything above it —
including, within a system, the vocal line it sits under (voices alternate
`V: Vocal` / body / `V: Ins` / body).

## 2. Request

New optional string field `"abc_template"`. Mutually exclusive with `"abc"` (both
set → request error) and meaningless with `cot = "off"` (→ request error). It is
score text, exactly what `"abc"` takes, plus three directive lines. A directive
occupies a whole line, is never fed to the model and never reaches `score.abc`:

| line | meaning |
|------|---------|
| `%%yue2-gen` | a hole: the model writes this one line. Optional ` bars=N` (whitespace required, 1 ≤ N ≤ 9999); default N = the bar count of the nearest body line above it (§4) |
| `%%yue2-primer-begin` | start of a primer block |
| `%%yue2-primer-end` | end of it |
| `%%yue2-continue` | last line only: the lines above are the score's beginning and the model writes the rest freely, as a plain score phase would (§3). Anything but blank lines after it is a request error |

Leading whitespace before a directive is allowed, as is a trailing `\r`. Any other
`%%yue2-…` line is a request error rather than a comment. A primer block's lines
are context only: they never supply a hole's default N or change the `M:`/`L:` a
hole is measured against, since the score never holds them.

**Primer block.** The lines between the two markers are fed to the model as
ordinary score context at that point and then *dropped from the emitted score*. It
lets the caller show the model a verse before it writes the intro (the model is
causal; the intro otherwise never sees the tune it introduces). The block holds
given lines only — a `%%yue2-gen` inside it is a request error. Unterminated or
nested blocks are request errors. What goes in the block is the caller's business;
the C++ does not know what a verse is. A template with a primer block gets one
`warning:` line on stderr at job setup: in practice the model quotes the primed
melody instead of introducing it, and the chord symbols on the resting vocal line
above a hole already supply the tune's harmony, so the primer is kept as
specified but not recommended.

**Continue.** `%%yue2-continue` as the last line ends the given part of the score
and hands the rest to the model: from there it samples as the plain score phase
does, one token at a time with `ABC_END` allowed, no line checks and no bar
counting, until it ends the score or the phase's `--max-abc` cap, the slot or the
semantic phase's room stops it (then `truncated`, and the stop is printed). What it
wrote is appended to the emitted score and the job finishes exactly as a template
that ran out of segments. The given part need not be well-formed — an opening with
only one voice, say — that is the point: the caller reads what the model makes of
it. Context is sized as if the tail could run to the abc cap.

A template with no directives at all is legal and must behave exactly like `"abc"`
with the same text (§6, test 1).

## 3. Decode

Parse the template into an ordered list of segments: `given` (text, fed), `primer`
(text, fed, not emitted) and `hole`. Adjacent given lines coalesce into one segment.
Line endings stay with their line: a segment's text ends in `\n` wherever the
template's did.

The abc phase for a template job:

1. Prefill `prefix_abc` as now.
2. For each segment in order:
   - **given / primer**: tokenize the segment text and feed it in its own
     `llama_decode` (the same way the phase bridge is fed — SPEC_BATCH §4.4 — not
     one token per lockstep step). Its tokens go into the slot's `history`, so the
     repetition-penalty window sees them as if the model had written them.
   - **hole**: sample, in the lockstep batch like any abc token, until the text
     sampled for this hole contains `\n`. A sampled token is not in the KV cache
     until the next decode, so the closing token is never fed as sampled: the hole's
     text is cut after its first `\n`, and whatever piece of the closing token lies
     before the cut is re-tokenized and fed with the next given segment. Anything
     the model put after the `\n` is discarded.
   - `ABC_END` sampled inside a hole, or the hole exceeding its token budget
     (`max(64, 8 × tokens of the body line above)`), closes it as a failed attempt.
3. After the last segment the phase ends as if `ABC_END` had been sampled. The
   phase is never reported truncated because of forced tokens; `--max-abc` still
   caps the *sampled* tokens of the job, counting the draws a rolled-back attempt
   spent. Reaching the cap rest-fills what is left of the template and is the only
   thing that reports the phase truncated.

`ABC_END` is masked for the whole hole, not only below `min_tokens`: a hole writes
one line and may never end the score. The mask is a template-only flag into the
sampler, so a request without a template draws exactly the tokens it did before.

**Validation and retry (§4 for the counting).** A closed hole is accepted when its
line is a body line (does not start with a field `X:`-style prefix, `%`, or `w:`)
and its bar count equals the hole's N. Otherwise the attempt is rolled back —
`llama_memory_seq_rm(mem, slot, pos_at_hole_start, -1)`, the slot's `pos`, `history`
and `step` restored to the hole's start, the RNG **not** restored so the retry draws
differently — and sampled again, up to 4 attempts. After the 4th failure the hole is
filled with rests: `Z|` for N = 1, else `ZN|`, plus `\n`, fed as a given line. Every
retry and every rest-fill prints one line (`abc: hole 7 (bars=4): attempt 2, got 5
bars`), and the totals land in the result JSON (§5). Rolling back needs the logits
of the token before the hole again: re-decode that one token rather than caching
logits.

**Emitted score.** `abc_text` = the given segments and the accepted hole lines, in
order, primer blocks and directives gone. `abc_ids` = `tokenize(abc_text)` — the
whole text tokenized in one go, exactly as the external-`"abc"` path does, *not* the
concatenation of the per-segment ids (BPE merges across segment seams differ).

**Semantic phase.** A template job never reuses the abc phase's KV cache: clear the
slot (`llama_memory_seq_rm(mem, slot, -1, -1)`), build `prefix_sem` from the
re-tokenized `abc_ids` as the external path does, and prefill it whole. That is what
makes the primer free to discard, and it gives the invariant of §6 test 3. Cost: one
~2k-token prefill.

Context sizing (`n_ctx_seq`, `max_feed`, the `CONTEXT` check) must account for the
abc phase of a template job: prefix + every given and primer token + an *expected*
written length per hole (the budgets summed are an order of magnitude more than any
real template writes). The difference is covered at runtime: before a hole takes
another token, both the room left in the slot and the room the re-tokenized score
would leave the semantic phase are checked against what the rest of the template
still needs, and a hole that would overrun either is rest-filled. Nothing in a
template may reach `llama_decode` with no cells left, and nothing may die after the
abc phase for want of context — that would take a whole batch with it.

## 4. Counting bars

On a body line, after removing `"..."` chord symbols, `[K:..]`-style inline fields,
`!...!` decorations and anything after an unquoted `%`: a bar is a maximal run
between bar lines (`|`, `||`, `|]`, `[|`, `|:`, `:|`, `::` each count as one bar
line) that contains at least one note or rest. `[1` / `[2` repeat endings are not
chords, and an unclosed `"`, `!` or `[` is an ordinary character — none of the
three may swallow the bar lines after it. A multi-bar rest `Z` counts 1 and `Zn` counts n. A trailing run with no
closing bar line still counts if it has a note or rest.

Bar *length* (note units per bar against `M:`/`L:`) is measured over the *written*
lines and reported as a count of off-length bars in the result JSON, but is not a
reason to reject: the
reference model's own scores are not always exact and the semantic stage follows the
score loosely.

Put the counter in its own small function with a table of cases in the test (§6).

## 5. Artifacts and result JSON

- `score.abc`, `abc_tokens.npy`, `plan.json` carry the emitted score (no primer, no
  directives).
- `request.json`, and plan.json's copy of it, must stay loadable by the reference's
  `SongRequest(**request.json)`, which raises on an unknown key — so they carry no
  `"abc_template"`. They are written after the abc phase, and record the emitted
  score as `"abc"`: the artifacts directory is then a plain request that reproduces
  the song. The template as given is a side file, `template.abc`; `ar_request.json`
  is the request echo and carries `"abc_template"` because it is a byte copy.
- `result.json` and the `yue2 batch --summary` entry gain, for template jobs only:
  `template: { holes, retries, rest_filled, primer_tokens, given_tokens,
  sampled_tokens, offlength_bars, continued_tokens }`. `yue2 ar` writes no result JSON and prints the
  same counters instead.

## 6. Acceptance

Agents run on Vulkan device 1 (`--gpu 1`) and CPU only — never device 0. Build in
`build_template/` with `nice -n 10 cmake --build build_template -j8`. Test inputs are
made under `tests/out/` (gitignored); `../songs/` is read-only source material.
Use `yue2 ar` for the score-level tests (fast) and one full `yue2 song` at the end.

1. **No-directive template ≡ `"abc"`.** Same text, same seed: `abc_tokens.npy`,
   `prefix.npy` and `semantic.npy` byte-identical between the two requests.
2. **Holes.** From a real two-voice score, a template with every `V: Ins` body line
   replaced by `%%yue2-gen`: every given line appears byte-identical and in order in
   `score.abc`; every written line passes §4 with the right N; no directive text
   anywhere in the artifacts dir except the request echo. Three seeds. Report
   retries / rest-fills per seed.
3. **Round trip.** The `score.abc` from test 2 re-submitted as plain `"abc"` with
   the same seed gives a byte-identical `semantic.npy`.
4. **Primer.** Test 2's template plus a primer block holding the first verse's
   lines, placed before the intro: the primer text appears nowhere in `score.abc` /
   `plan.json`; the slot was cleared (the §3 position check in `feed()` holds for
   the semantic prefill); the intro's written lines differ from test 2's at the same
   seed (proof the primer was seen).
5. **Forced failure.** A hole with `bars=999` exhausts its retries and is
   rest-filled; the run completes and the JSON counts it.
6. **Batch.** `yue2 batch --parallel 2` with one template job and one plain job:
   the template job's outputs are byte-identical to running *it* alone; the plain
   job's are identical as far as SPEC_BATCH §6 allows (two jobs decoding in the
   same lockstep batch diverge in the low bits — show the control, two plain jobs,
   diverging the same way).
7. **Regression.** `tests/regress.sh` and a `--verify-sampler` run pass unchanged;
   a request without `"abc_template"` takes no new code path that changes a byte.
8. **Request errors**: `abc` + `abc_template`; template with `cot=off`; gen inside
   primer; unterminated primer; `bars=0`.
9. `YUE2_WARN_FLAGS` clean. Bar-counter table test (a small `tests/*.cpp` like
   `sampler_diff.cpp`, or a hidden `yue2 ar --selftest-bars`, whichever fits).

Deliver `src/STATUS_TEMPLATE.md`: what was built, deviations from this SPEC and
why, a table of the test results with exact commands, and three written Ins lines
next to the vocal lines they sit under so a reader can judge them. README gets a
short section next to the existing re-skin-via-`"abc"` text. No local paths, host
names or song titles in anything committed.
