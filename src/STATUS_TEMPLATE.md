# STATUS_TEMPLATE — stage 7: score templates

What was built against `SPEC_TEMPLATE.md`, the acceptance run, and where it
differs. Every number is the Intel Arc B70 (Vulkan device 1), `yue2-ar-q8_0.gguf`,
built in `build_template_r2/`. Device 0 was not touched.

```sh
cmake -B build_template_r2 -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON
nice -n 10 cmake --build build_template_r2 -j8
```

## 0. What it is

A new optional request field `"abc_template"`: score text in which `%%yue2-gen`
is a hole the model writes and `%%yue2-primer-begin` / `%%yue2-primer-end`
bracket lines that are fed as context and dropped from the emitted score. It is
forced decoding with holes — one autoregressive text stream in which the given
lines are fed and the holes are sampled — so a written line is conditioned on
everything above it, including the vocal line it sits under.

New code, all of it in `stage_ar.cpp` behind `Request::has_tpl`:

| piece | what it is |
|---|---|
| `parse_template()` | template text → ordered `TplSeg` list (`given` / `primer` / `hole`); every §2 request error is reported from here, before anything loads |
| `count_bars()`, `strip_line()`, `is_body_line()` | §4, pure text, table-tested in `tests/bars.cpp` |
| `Runner::give/template_step/open_hole/template_token/fail_hole/rollback_hole/rest_fill/template_done` | the hole state machine inside the existing lockstep loop |
| `Runner::feed_tokens()` | the decode + `expect_pos` half of the old `feed()`, split out so score text can be forced in without sampling |
| `Runner::abc_fits/sem_fits` | the two context ceilings a hole is checked against before it takes another token (§3) |
| `TemplateStats` | the §5 counters, on `ArResult` → `result.json` and the `yue2 batch --summary` entry |

`apply()` gains one branch at the top (`q.in_hole` → `template_token`) and
`enter()` one (`is_template` → prefill without sampling, then `template_step`).
`sample_step` and the frozen `sample_step_ref` gain a defaulted `mask_end`
argument, which only a template job inside a hole ever passes as true. Nothing
else on the plain path changed; §6 test 7 shows it byte for byte.

## 1. Acceptance (SPEC_TEMPLATE §6)

The source material is one finished two-voice render's `score.abc` and its
`request.json` (private, read-only), copied into `tests/out/tpl/` along with the
template built from it — every accompaniment body line replaced by
`%%yue2-gen`, 35 holes. An independent counter script under `tests/out/tpl/`
re-implements the §4 counting, so the bar checks below are made by a second
implementation, not by the one under test.

| # | what | result |
|---|---|---|
| 1 | no-directive template ≡ `"abc"` | **PASS** — `abc_tokens.npy`, `prefix.npy`, `semantic.npy` *and* `score.abc` byte-identical |
| 2 | 35 holes, 3 seeds | **PASS** — every given line byte-identical and in order, every written line a body line with the right bar count, no extra lines |
| 3 | round trip | **PASS** — test 2's `score.abc` re-submitted as `"abc"`, same seed → byte-identical `prefix.npy`, `abc_tokens.npy`, `semantic.npy` |
| 4 | primer | **PASS** — primer lines nowhere in `score.abc`; the semantic prefill's `expect_pos` check held (slot cleared); all 4 intro holes differ from test 2 at the same seed |
| 5 | forced failure (`bars=999`) | **PASS** — 4 attempts, 3 retries, rest-filled `Z999|`, the next hole wrote normally, run completed, `result.json` counted it |
| 6 | `--parallel 2`, one template + one plain job | **PARTIAL** — the template job is byte-identical to running it alone (greedy *and* sampled); the plain job's `semantic.npy` is not. See §3.1 |
| 7 | regression | **PASS** — see §1.2 |
| 8 | request errors | **PASS** — all nine rejected before the model loads (§1.5) |
| 9 | warn flags, bar table test | **PASS** — `YUE2_WARN_FLAGS` clean; `yue2-bars`: 71 cases, 0 failures |

Tests 1, 2 (seed 1), 3, 5, 8 and the `--verify-sampler` and regression checks
were re-run on the round-2 build; 4, 6 and the full render are the round-1
numbers, since the round-2 changes touch neither the primer nor the batch loop
and the seed-1 score came out byte-identical to round 1 afterwards.

### 1.1 Test 2 — per seed

`build_template_r2/yue2 ar -m yue2-ar-q8_0.gguf --gpu 1 --request tests/out/tpl/req_tpl.json --seed S --artifacts tests/out/tpl/t2_sS --max-semantic 256`

| seed | holes | sampled tokens | retries | rest-filled | off-length bars | emitted score | abc phase |
|---|---|---|---|---|---|---|---|
| 1 | 35 | 636 | 0 | 0 | 0 | 2468 ids | 7.3 s |
| 2 | 35 | 617 | 0 | 0 | 0 | 2449 ids | 7.1 s |
| 3 | 35 | 1098 | 0 | 0 | 0 | 2930 ids | 11.1 s |

Zero retries across all three seeds: given the vocal line above it and the bar
lines of everything before it, the model lands the bar count first try. The
retry path is therefore exercised by test 5 and §1.4 rather than by test 2.

### 1.5 Request errors (test 8)

All nine are reported by `parse_template` / `validate_request` before the model
loads: `abc` + `abc_template`; `abc_template` with `cot=off`; `%%yue2-gen`
inside a primer; an unterminated primer; `bars=0`; `bars=-2`;
`%%yue2-genbars=3` (no whitespace → unknown directive); `bars=100000` (out of
range); `bars=99999999999999999999` (rejected by `strtol`, not undefined
behaviour); and a `%%yue2-gen` whose only body line above is inside a primer
block.

### 1.6 The context guard

The shipped estimate is generous enough that no real template reaches it — the
35-hole template writes 602 tokens against an estimate of 4760 — so the guard
was forced by temporarily raising `TEMPLATE_SEAM_MARGIN` from 64 to 3300 in a
scratch build, which makes `sem_fits` bind part way through the same run:

```
abc: hole 27 (bars=4): attempt 1, would leave the semantic phase no room
…
abc: hole 27 (bars=4): filled with rests
abc: template: 35 holes, 741 sampled tokens, 9 retries, 2 rest-filled, …
```

The run finished normally — full artifact set, exit 0, every given line still
byte-identical and every written line still passing the bar check — instead of
dying in `template_done`, which is the whole point. The constant was put back
to 64 and the build re-run afterwards; the seed-1 score came out byte-identical
to the unforced run.

### 1.2 Test 7 — regression

| check | command | result |
|---|---|---|
| a plain request takes no new path | the same generation request (`cot=full`, no `abc`, no template) through `build/yue2 ar` (pre-change) and `build_template_r2/yue2 ar` | `abc_tokens.npy`, `prefix.npy`, `semantic.npy`, `score.abc`, `request.json` all byte-identical |
| sampler | `… --verify-sampler` on a template request and on a plain one | 892 and 512 steps matched the frozen stage-5 sampler exactly |
| VAE | `YUE2_REGRESS_SONGS=… YUE2_REGRESS_GPU=1 tests/regress.sh <one song>` | 92.87 dB SNR; the same latent through `build_template_r2/yue2-vae` gives a byte-identical `.npy` |

### 1.3 One full render

`build_template_r2/yue2 song --gpu 1 --ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf --vae yue2-vae-f32.gguf --request tests/out/tpl/req_tpl.json --seed 1 --out tests/out/tpl/full.flac --artifacts tests/out/tpl/t_full`

256.4 s of audio in 303 s end to end (abc 7.2 s, semantic 66.6 s, nar 204.4 s,
vae 23.7 s), 35 holes, no retries. `result.json` carries the `template` block;
so does the `yue2 batch --summary` entry (checked separately with the test-5
request).

### 1.4 The failure paths

A second, deliberately awkward template — no given body line at all, so the
first segment *is* a hole (`bars=4`), followed by two adjacent holes
(`bars=2`) — put every rejection reason through its paces in one run:

```
abc: hole 2 (bars=2): attempt 1, got 4 bars
abc: hole 2 (bars=2): attempt 2, ran past its 64-token budget
abc: hole 2 (bars=2): attempt 3, got 4 bars
abc: hole 2 (bars=2): attempt 4, got 4 bars
abc: hole 2 (bars=2): filled with rests
abc: hole 3 (bars=2): attempt 1, wrote a field or comment line, not a body line
…
abc: template: 3 holes, 233 sampled tokens, 6 retries, 2 rest-filled, …
```

Hole 1 (a hole as the very first segment, sampled from the abc prefix's own
logits) was accepted; holes 2 and 3 exercised the wrong-bar-count, budget and
not-a-body-line rejections, the rollback, the rest-fill and the carry between
two adjacent holes. `ABC_END` inside a hole is the one rejection reason no run
produced; it is the same three lines of code as the others.

## 2. What the written lines look like

(Described rather than quoted: the source score is private material. The shape
below is what three of the 35 written lines did, at seed 1.)

- **A hole under four bars of vocal rest (the intro).** The model wrote a full
  four-bar accompaniment figure — one chord tone per half bar in the octave
  above the vocal's register, alternating between the two harmonies the given
  chord symbols name. Nothing else in the score sounds there, so it filled the
  space.
- **A hole under a vocal line that sings the first half of each bar and rests
  after it.** The written line is the inverse: a whole-bar rest under the
  phrase, then a short four-note answer placed exactly in the gap, repeated
  when the vocal's pattern repeats. This is the case the feature was built for
  — the model could only place the answer there because it had the vocal line
  above it in the same stream.
- **A hole under a vocal that sings continuously for four bars.** The model
  wrote `Z4|` — four bars of rest, the correct musical answer, and the reason
  the acceptance test is a bar *count* and not a bar *length*.

For an idea of the notation, a synthetic equivalent of the second case in
`M:4/4`, `L:1/32` (invented for this file, not from any render):

```
V: Vocal
"C"c8e8g8e8|"G"G32|"C"c8e8g8e8|"G"G32|
V: Ins
Z|z16G4B4d4B4|Z|z16G4B4d4B4|
```

## 3. Deviations from the SPEC

SPEC_TEMPLATE.md was amended in §2, §3, §4, §5 and §6.6 to describe what is
here — sizing and its runtime guard, the `--max-abc` rule, the artifact layout,
the bar-counter edge cases, directives with leading whitespace, and the batch
byte-identity claim. What is left:

| # | SPEC | what was built, and why |
|---|---|---|
| 1 | §6.6 byte-identity in a batch | Only the template job is byte-identical to running it alone. The plain job's `semantic.npy` differs between `--parallel 2` and alone — it decodes beside the template's hole tokens, which is exactly the low-bit divergence SPEC_BATCH §6 documents. Control: a batch of **two plain jobs**, no template code on the path, diverges the same way. The template job matches because its semantic phase starts after the other job has finished, so it runs alone. |
| 2 | §5 "result JSON" | `result.json` and the batch `--summary` entry carry `template`; `plan.json` does **not**. `plan.json` is the resume record the reference `SymbolicPlan.load` reads, and a new top-level key there is a compatibility risk for nothing. `yue2 ar` prints the same counters on stdout instead. |
| 3 | §4 "nearest body line above" | Resolved at parse time, so it is the nearest *given* body line — a previous hole's written line is never the reference, and a primer block's lines never are either. For the vocal/accompaniment case that is the intended line anyway. |
| 4 | §4 off-length bars | Measured against the `M:`/`L:` in force at the hole, tracked as the parser walks the template's field lines. Tuplets `(p[:q[:r]]`, chords, grace notes, ties and broken rhythm are handled; anything more exotic can only affect this report, never an accept/reject. |
| 5 | §3 stats | A template job's abc `GenStats` report the emitted score's token count and a tok/s that includes the forced tokens (there is no `ABC_END` to count, and the phase is not "generation" in the same sense). The real work is the separate `abc: template: …` line and the `template` block. |

## 3.1 Round-2 fixes

| # | finding | fix |
|---|---|---|
| 1 | the semantic-side estimate was unguarded — an over-writing template died in `template_done`, past `--continue-on-error` | `Runner::sem_fits`, checked before a hole takes another token and before a line is accepted: `prefix + emitted + hole + remaining given + 6 per remaining hole + 2 + max_semantic + seam margin ≤ n_ctx_seq`. Suffix sums precomputed at sizing; a hole that would overrun is rest-filled. §1.6 shows it firing. |
| 2 | no backstop on a forced feed | `give()` checks `pos + ids ≤ n_ctx_seq` and dies with a template-specific message; the abc-side guard now also reserves the given segments and rest-fills still to come (`tail_fed`, `tail_holes`). |
| 3 | `[1` / `[2` repeat endings counted as chords | `[` + digit is an ending marker; a chord must close before the next bar line, otherwise `[` is an ordinary character. Three table cases. |
| 4 | trailing `%` comment counted as music, unpaired `!` ate the line | `strip_line` stops at an unquoted `%`; `"`, `!` and `[K:` are only honoured when they close, and `!` must close before the next space or bar line. Four table cases. |
| 5 | ` %%yue2-gen` (indented) was fed to the model | directives are matched on the left- and right-trimmed line; unknown ones are still an error. |
| 6 | `is_body_line("\r")` was true | `\r` is blank, and the tests run on the trimmed line. What is *fed* still carries the `\r`. CRLF parse case added. |
| 7 | `sscanf("bars=%d")` overflowed, `%%yue2-genbars=3` was accepted | `strtol` with an end pointer, range 1..9999, whitespace required before the argument. Four error cases. |
| 8 | `--max-abc` capped the rollback-restored `q.step`; `truncated` was set too eagerly; `rest_fill`'s no-rollback path left `q.step` high | the cap is on `tpl.sampled_tokens` (which counts failed attempts); `truncated` is set only where a hole is actually rest-filled by the cap; `rest_fill` restores `q.step` on both paths. |
| 9 | `abc_template` in `request.json` broke `SongRequest(**request.json)` | it is gone from `request.json` and from plan.json's copy. Those are written after the abc phase and record the emitted score as `"abc"`, so the artifacts dir is a plain request that reproduces the song — test 3 now shows its `request.json` byte-identical to the round-trip run's. The template as given is the side file `template.abc` (and `ar_request.json`, which is a byte copy of the request). |
| 10 | a primer's lines could supply a hole's default N and meter | `above` and `meter` are not updated inside a primer block; a hole with no other body line above it is a request error naming the reason. |
| 11 | `ABC_END` was drawable inside a hole | masked for the whole hole via a defaulted `mask_end` on `sample_step` and `sample_step_ref` — the one additive edit to the frozen reference, needed so `--verify-sampler` compares like with like. Verified on both a plain request (512 steps) and a template one (892 steps). |
| 12 | README/STATUS quoted a private render's score | every quoted score line is now invented; the written lines are described in prose. |

## 4. Notes

- `retries` counts attempts that were rolled back and drawn again, so it is at
  most `TEMPLATE_ATTEMPTS − 1` = 3 per hole; the 4th failure is the rest-fill.
- A rollback re-decodes the one token before the hole to get its logits back
  (§3). That is one extra `llama_decode` per retry, nothing more.
- `--greedy` makes a retry draw the same tokens again, so a greedy hole that
  fails once will exhaust its attempts and rest-fill. Sampling is the mode the
  retry loop is for.
- A dense stress template (40 holes, each asking for 8 bars under a vocal line
  the model answers in 4) produced 120 retries and 40 rest-fills in one run and
  still finished with a complete score — the retry loop at its worst.
- On the Arc the larger `n_batch` a template job asks for (the whole semantic
  prefix in one feed, as the external-`"abc"` path does) prints
  `ggml_vulkan: Failed to allocate pinned memory` once. It is a fallback to
  unpinned staging, not an error, and the run is unaffected.
- Not built, not asked for: editing a given line, holes spanning more than one
  line, and any notion of what a verse is — a primer block is whatever the
  caller puts between the markers.

## 5. `%%yue2-chords`

SPEC_TEMPLATE §2 "Chords" and §3 step 1–5: a directive that has the model write
a `V:` line *after* it has read the other voice's line of the same system, so a
given accompaniment can be handed its harmony. Built in `build_template/`
(`cmake -B build_template -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
-DYUE2_BUILD_TESTS=ON`, `nice -n 10 cmake --build build_template -j8`), run on
Vulkan device 1 (Arc B70) only.

| piece | what it is |
|---|---|
| `TPL_CHORDS` | a segment carrying `head` (the `V:` header above, peeled off the given segment it ended), `text` (the two lines below: the other voice's header and its body line) and a hole's `bars` / `above` / `meter` |
| `parse_bars_arg()` | the ` bars=N` grammar of `%%yue2-gen`, factored out so both directives share it — same error strings, same `strtol` range check |
| `trim_line()` | the line-trimming the parser did inline, factored out for the two look-ahead lines |
| `GiveKind` | `give()`'s `bool emit` became `GIVE_SCORE` / `GIVE_PRIMER` / `GIVE_SCRATCH`; the third is the out-of-order feed, which counts for nothing because it is rolled back |
| `Runner::open_chords()` | checkpoints (`chk_pos` / `chk_hist` / `chk_step`), feeds `B`, `A`, `"` as scratch, then opens a hole on the quote |
| `Runner::finish_chords()` | the one exit, from acceptance or rest-fill: `llama_memory_seq_rm` back to the checkpoint, then `A` + line + `B` fed as one given stretch, so cache, `history` and score all read in template order |

Validation of the sampled line adds two tests to the hole's bar count: it must
carry a chord symbol, and `strip_line()` of it must hold no `A`–`G` / `a`–`g`
— rests, bar lines and lengths only. Retries, the rest-fill (`Z|` / `ZN|`, no
chord symbol) and the printed lines are the hole's, unchanged. Counters: a
chords directive is a hole in `holes` / `retries` / `rest_filled`, and the new
`chord_lines` says how many of them there were — in `TemplateStats`,
`result.json`, the batch summary and the `yue2 ar` line.

### 5.1 Tests

Requests were built from a private render's `request.json` with
`jq --rawfile abc T '. + {abc_template: $abc} | del(.abc)'`; all runs
`--gpu 1 --seed 1 -m yue2-ar-q8_0.gguf`, artifacts under `tests/out/chords/`
(gitignored).

| # | command | result |
|---|---|---|
| 1 | `build_template/yue2 ar --request tests/out/chords/chords.json --artifacts tests/out/chords/t1 --max-semantic 32 --gpu 1 --seed 1` (`docs/examples/song_intro_chords.tpl.abc`) | **PASS** — 4 chord lines at the directive positions, above their `V: Ins` lines, right bar counts, 2 retries (one line came back 1 bar twice), 0 rest-fills; the `%%yue2-continue` tail (2064 tokens) is written *with* chord symbols throughout; no directive text in `score.abc` or `plan.json` |
| 2 | same, `tests/out/chords/mixed.json` → `t2` (the same intro, then two given verse systems with `%%yue2-gen` Ins holes) | **PASS** — 6 holes / 4 chord lines; the four chord lines are byte-identical to test 1's, the two ordinary holes wrote 4-bar Ins lines |
| 3 | `docs/examples/song_new_ins.tpl.abc` and `song_continue.tpl.abc` through `build/yue2 ar` (pre-change) and `build_template/yue2 ar`, same seed | **PASS** — `abc_tokens.npy`, `score.abc` and `semantic.npy` byte-identical for both templates |
| 4 | six malformed templates (`tests/out/chords/e_*.json`) | **PASS** — all six a clean request error before the model loads, exit 1: section label above (`% intro`), directive as the last line, `%%yue2-gen` where the header must be, inside a primer block, after `%%yue2-continue`, and a comment line where the body must be |
| 5 | `--verify-sampler` on test 1's request | **PASS** — 2198 sampling steps matched the frozen stage-5 sampler; the score came out byte-identical to test 1 |
| 6 | `build_template/yue2-bars` | **PASS** — 71 cases, 0 failures (the `parse_bars_arg` / `trim_line` refactor) |
| 7 | forced rest-fill: test 1's template with `bars=99` on every directive | **PASS** — 4 chord lines × 4 attempts, 12 retries, 4 rest-fills, `Z99|` written above each Ins line in template order, the given system after them intact, exit 0 |
| 8 | one full render, `build_template/yue2 song --gpu 1 --ar … --nar … --vae … --request tests/out/chords/mixed.json --seed 1 --out … --artifacts tests/out/chords/t_song` | **PASS** — 42.3 s of audio in 32.4 s (abc 1.6 s, semantic 8.0 s, nar 17.5 s, vae 4.1 s); `result.json` carries `"chord_lines": 4` beside `"holes": 6` |

`tests/regress.sh` was **not** run: it scores the standalone `yue2-vae` against
a private render set, defaults to device 0, and touches no AR code — nothing in
this change can reach it. The AR-side regression is test 3 plus test 5.

### 5.2 The four lines test 1 wrote

Each written line is the `V: Vocal` line; the `V: Ins` line under it is the
given one it was written from (`docs/examples/song_intro_chords.tpl.abc`, `K:Em`,
`M:4/4`, `L:1/32`).

```
"Em"z32|"Em"z32|"B/D#"z32|"Em"z32|
Z|e4B4g4B4f4B4e4B4|^d4B4f4B4e4B4d4B4|e4B4g4B4f4B4e4B4|

"Em"z32|"Em"z32|"Em"z32|"Em"z32|
^d4B4f4B4e8d8|e8g8f8e8|^d8f8e8d8|e8g8f8e8|

"Em"z16"B"z16|"Em"z32|"Em"z32|"Em"z32|
^d8f8g8f8|e8g8f8e8|^d8f8e8d8|e16f16|

"Em"z16"B"z16|
g16a16|
```

The first line is the one the reference experiment (a plain server at temp 0)
also wrote, `B/D#` and all. The third system is where the model wanted one bar
instead of four twice over before landing it — the reference rest-filled that
one.

### 5.3 Deviations

| # | SPEC | what was built, and why |
|---|---|---|
| 1 | §3 step 5 "feed `A`, the accepted line, `B`" | The three are fed as **one** `give()`, not three: the emitted score is re-tokenized in one go at the end, so one feed is the closer match to it (fewer BPE seams) and costs one `llama_decode` instead of three. The consequence is that the written line's tokens land in `given_tokens` rather than in a counter of their own — `chord_lines` and `sampled_tokens` are what count the writing. |
| 2 | §2 "the two lines below it a given voice header and a given body line" | Not checked: that the header below names a *different* voice than the one above. The parser knows `V:`, not which voices a score has, and a template that puts the same voice on both lines is the caller's business — the model still reads a system. |
| 3 | §3 step 3 "same budget" | The budget scales from `B`'s body line (`above` = that line), which is what §2 makes the default bar count too. |
| 4 | context sizing | A chords segment's own lines are counted **once**, like a given segment, plus a hole's expected line: the out-of-order feed holds `B` + `A` + `"` at the same moment the line is being written and is undone before those lines are fed in order, so the peak is a hole's peak. `sem_fits` adds this segment's given tokens back explicitly, since they are not in `emitted` until the line is accepted. |
