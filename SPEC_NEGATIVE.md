# SPEC_NEGATIVE — stage 11: a negative style for `cfg_scale`

Builds on stage 7 (SPEC_GUIDANCE.md §2.2, `src/STATUS_GUIDANCE.md`). Same files
(`src/stage_ar.cpp` / `.hpp`, `tests/guidance.cpp`), same rules.

## 1. Why

`cfg_scale = c` samples from `N + c·(B − N)`, where `N` is the reference's blank
branch: the instruction and the score, no tags, no lyrics (`blank_prefix`). So a
guided song is pushed away from "no description at all".

A negative style works like a diffusion negative prompt: it **replaces** the blank,
so the song is pushed away from a description the user writes
(`"children's song"`) as well as towards its own tags and lyrics. The formula and
the machinery do not change; only the negative branch's prefix does.

## 2. Request

```json
{ "style": "…", "lyrics": "…", "cfg_scale": 3, "negative_style": "children's song" }
```

- `negative_style`: a non-empty string (after trimming whitespace). Absent or
  `null` = today's blank branch.
- It is the text of the negative branch's `[Tags]`. The negative branch carries
  **no lyrics**, deliberately: the blank has none either, so the lyric push that
  `cfg_scale` gives is kept, and the negative style is added on top of it.

Request errors (exact, tested messages, in `validate_request`, before any GPU work):
- `negative_style` with an effective `cfg_scale` of 1 (absent or `1.0`): the
  negative branch has weight `c − 1 = 0`, so it would silently do nothing.
  Message says to set `cfg_scale`. (`cot: "off"`'s default 1.01 counts as ≠ 1 and
  is allowed.)
- `negative_style` empty or whitespace only, or not a string.
- `negative_style` together with `guidance`, `sections` or `handover` (those
  already reject `cfg_scale != 1`; one mechanism per request).
- `abc_template` is already rejected with `cfg_scale != 1`; nothing new.
- `semantic_keep` + `cfg_scale` + `negative_style`: **allowed**. The negative
  branch is born at step N exactly where the blank is today.

## 3. The negative prefix

Built by the same recipe as the **positive** prefix, with the tags replaced and the
lyrics empty, then the positive branch's exact score:

```
[EOD] + tokenize(instruction(cot) + "\n[Tags]\n" + negative_style + "\n[Lyrics]\n" + "" + "\n")
      + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
```

i.e. `Request::text()` (`stage_ar.cpp`, `text()`) on a copy of the request with
`style = negative_style`, `lyrics = ""`, tokenized exactly as the positive prefix's
text is. Reuse the positive prefix's builder with those two fields swapped; do
not hand-assemble a third string recipe. `cot == "off"`: the same text, then
`[MUSIC_START]` with no score, mirroring what `blank_prefix` does for that mode.

Write this as one function beside `blank_prefix` (e.g. `negative_prefix`) and
select between them at the two call sites that build `BRANCH_BLANK`'s prefix
(stage 7 prefill and the `semantic_keep` birth). The branch slot, weight
(`c − 1`), blend, sampler, `--parallel 1` rule and lifecycle are unchanged.

## 4. Artifacts and logs

- `request.json` must stay loadable by the reference's `SongRequest(**request.json)`:
  follow the `guidance` precedent and do **not** write `negative_style` into the
  request the engine writes. Record it in `config.json` (`"negative_style": text`
  or `null`) and `plan.json`.
- `--dump-logits`: the branch file keeps its name (`FILE.blank.npy`); add
  `"negative_style"` to whatever metadata sits next to it, if any.
- Save the negative prefix as `negative_prefix.npy` (int32) in the artifacts dir
  when `negative_style` is set, so it can be checked by eye with the tokenizer.
- One log line when the branch is prefilled:
  `guidance: negative style "<text>" (P tokens) replaces the blank branch`.
- `--guidance-trace` works unchanged (its "blank" column is now the negative branch;
  say so in the trace header when `negative_style` is set).

## 5. Acceptance (device 1 or CPU within the limits below; never device 0)

1. **Untouched paths.** No `negative_style`: `cmp` of `prefix.npy`,
   `abc_tokens.npy`, `semantic.npy` against `build/yue2` (the current committed
   binary; run a copy of it, don't rebuild it) for an unguided request and for
   `cfg_scale 3`, same seed, a few hundred frames via the existing frame cap.
2. **Prefix recipe.** `negative_prefix.npy` decoded with the tokenizer reads
   `instruction\n[Tags]\nchildren's song\n[Lyrics]\n\n` followed by the score
   tokens, and its score segment is identical to the positive `prefix.npy`'s.
   Also check that the text tokens equal tokenizing that string in one call (no
   seam where the pieces were joined).
3. **It does something.** `cfg_scale 3` with and without `negative_style`, same
   seed: the first-step blended rows differ (`--dump-logits`), and `semantic.npy`
   diverges. Report the first differing frame.
4. **Determinism.** Same negative request twice → identical `semantic.npy`.
5. `cot: "off"` + `negative_style` (with the 1.01 default) runs and its negative
   prefix has no score.
6. `semantic_keep` + `cfg_scale 3` + `negative_style`: first N codes equal the
   file; the log shows the negative branch born at step N.
7. `--verify-sampler` on a negative run: all sampled steps match.
8. Every §2 error at the CLI and as table cases in `yue2-guidance`.

## 6. Rules

- Build dir: `build_negative/` (new). Do NOT rebuild or touch `build/` or any other
  `build_*`. `nice -n 10 cmake --build build_negative -j8`.
- Model runs on the Arc (`--gpu 1`). CPU model runs only if unavoidable: `--threads 4`,
  `nice -n 19`, q8_0, a few hundred frames, one at a time. **Never device 0** (the
  AMD is the user's card).
- Keep the renders short (frame cap a few hundred); stop after the AR where the NAR
  isn't needed (`yue2 ar`).
- No commits. Public repo: no home paths, no song titles, no private docs.
- Style: tabs; braces on their own line except `} else {`; `YUE2_WARN_FLAGS` clean.
- Report: `src/STATUS_NEGATIVE.md` (results table, exact commands, deviations).
  README: document `negative_style` next to `cfg_scale`, one example.
  `docs/ROADMAP.md`: one Done line (stage 11).

## 7. Stage 11b — `negative_lyrics`

### 7.1 Why

Measured on a real render (cfg 3, same seed and score, traced on the AMD): the stage-11
negative `"children's song, happy"` with no lyrics disagreed with the song exactly as much
as the blank did (mean TV(song, negative) 0.095 vs 0.093, the same in every 30 s window).
Without lyrics, a short tag line is nearly indistinguishable from no tags: both branches
mostly measure "the song without its words". To push along the *style* difference alone,
the negative must carry the same lyrics.

### 7.2 Request

```json
{ "cfg_scale": 3, "negative_style": "children's song", "negative_lyrics": true }
{ "cfg_scale": 3, "negative_lyrics": true }
```

- `negative_lyrics`: boolean, default `false` (= stage 11 exactly). `null` = absent.
- `true`: the negative prefix is built exactly as in §3 but with the request's own
  lyrics instead of `""`:
  `negative_request` = the positive request with `style = negative_style` (or `""` when
  `negative_style` is absent) and the lyrics **unchanged**.
- `negative_lyrics: true` without `negative_style` is allowed: tags `""`, same lyrics,
  so `cfg_scale` amplifies the style tags only. The text is still the full
  `text()` recipe (`[Tags]\n` followed directly by `\n[Lyrics]`), not the blank.
- Errors (exact, tested, in `validate_request` / `parse_request_json` as in §2):
  non-boolean; `true` at an effective `cfg_scale` of 1; `true` beside `guidance`,
  `sections` or `handover`. `false` is always accepted and changes nothing.
- `cot: "off"`: same text rule, then `[MUSIC_START]`, as §3.
- `semantic_keep`: allowed, same as §2.

### 7.3 Artifacts

- `negative_prefix.npy` is written whenever the negative is not the blank (either key).
- `config.json`: `"negative_lyrics": true|false` beside `"negative_style"`. `plan.json`:
  only when `true`. Not in the engine's `request.json` (reference-loadable).
- The log line of §4 says which: `… negative style "<text>" with the song's lyrics …` /
  `… an empty style with the song's lyrics …`.

### 7.4 Acceptance (Arc or CPU only, never device 0)

1. Untouched: no `negative_lyrics` (or `false`) → `prefix.npy`, `abc_tokens.npy`,
   `semantic.npy`, `negative_prefix.npy` byte-identical to the stage-11 binary
   (commit 4bdac91; build it into `build_neg11/` from a `git worktree` or `git show`
   export, don't touch `build/`) for unguided, `cfg_scale 3`, and `cfg_scale 3` +
   `negative_style`.
2. Recipe: with `negative_lyrics: true` and `negative_style` equal to the request's
   own `style`, `negative_prefix.npy` equals `prefix.npy` exactly, and
   `--dump-logits`' blended first row equals the positive row (`B + 2·(B − B)`).
3. Recipe: with `negative_style` absent, decode `negative_prefix.npy` and show the
   text reads `…[Tags]\n\n[Lyrics]\n<lyrics>…`, and its tokens equal the reference's
   `protocol.token_prefixes(style="", lyrics=<same>)` if the reference accepts an
   empty style (say so if it doesn't).
4. The point of the stage: same request as the 7.1 measurement (cfg 3, a real
   multi-verse lyric, seed fixed, `--guidance-trace`, a full song via the AR only):
   three runs on the Arc — blank, `negative_style` without lyrics, `negative_style`
   with lyrics. Report mean TV(song, negative) (trace col 4) and col 6 for each, whole
   song and per 30 s window. Expected: the lyrics-kept one clearly lower than the other
   two. Report whatever it is.
5. Determinism, `--verify-sampler`, `semantic_keep` + `negative_lyrics` (first N codes
   equal, branch born at N), every §7.2 error at the CLI and in `yue2-guidance`.

Rules: §6, with `build_negative11b/` as the build dir. Report: a "Stage 11b" section in
`src/STATUS_NEGATIVE.md`. README: `negative_lyrics` next to `negative_style`, and one
sentence on why a negative without lyrics behaves like the blank (the 7.1 numbers).

## 8. Stage 11c — `cfg_score`: guiding the score phase

### 8.1 Why

`cfg_scale` (reference and stages 7/11/11b) guides the semantic phase only; the abc phase
(the score) is sampled plain, so every cfg/negative take of a seed shares one `score.abc`.
Listening says the score bakes in much of a style's performance (a eurodance score keeps
its beat under a lullaby prompt), so the semantic phase cannot undo what the score
committed to. 11c lets the same negative act while the score is written. This goes
beyond the reference (it has no guided abc phase); keep it opt-in and separate.

### 8.2 Request

```json
{ "cfg_score": 2, "negative_style": "children's song", "negative_lyrics": true, "cfg_scale": 3 }
```

- `cfg_score`: number, finite, `0 < c ≤ 20`; absent / `null` / `1` = the abc phase is
  unguided exactly as today. Independent of `cfg_scale` (either may be 1).
- The abc phase samples from `B + (c_s − 1)·(B − N)` with `c_s = cfg_score`, over exactly the
  ids the abc sampler visits (`sample_step`'s abc segments: `[0, EOD)` and `ABC_END`), in
  f32, then the unchanged `sample_step` (penalty → temperature → top-k/top-p → draw). One
  draw per step, one rng, one history, as §2.1 of SPEC_GUIDANCE.
- `N` is the negative branch of §7: **`negative_lyrics: true` is required** (a lyric-less
  negative would push on the lyric text that places the words). `negative_style` may be
  absent (empty tags + same lyrics). The negative's abc-phase prefix is
  `prefix_head(negative_request(r).text())` — the positive abc prefix's own recipe.
- **One branch, both phases.** The negative stream is prefilled at abc step 0 and gets
  every sampled abc token. At the abc→semantic transition it must hold exactly
  `negative_prefix(vocab, r, abc_ids)` (§3/§7) once the same closing tokens the positive
  gets are appended — do not re-prefill it. If the effective `cfg_scale` is 1, drop the
  branch at the transition (the semantic phase is plain). If `cfg_scale ≠ 1` it continues
  as the §7 negative with weight `cfg_scale − 1`.
- The §2/§7.2 "does nothing at cfg_scale 1" errors for `negative_style` / `negative_lyrics`
  become "at cfg_scale 1 **and** cfg_score 1".
- Errors (exact, tested, `parse_request_json` / `validate_request`, before any GPU work):
  non-number / out of range; `cfg_score ≠ 1` without `negative_lyrics: true`; with
  `"abc"` given, `abc_template`, `cot: "off"`, or `semantic_keep` (no abc phase is
  sampled / the score is not ours to change); with `guidance`, `sections`, `handover`.
  Guided jobs keep the `--parallel 1` rule (a `cfg_score` job counts as guided).
- `request.json` stays reference-loadable (no `cfg_score`); `config.json`:
  `"cfg_score": number` always (1 when absent); `plan.json`: when ≠ 1.
- Log line at the branch's birth: `guidance: score phase guided at <c> by <negative_label>`.
- Context/KV sizing: the negative stream now exists from the abc phase on; make sure the
  sizing (head_max, n_ctx, streams) covers negative head + max abc tokens + semantic.

### 8.3 Trace and dumps

- `--guidance-trace` also writes `guidance_trace_abc.npy` for a `cfg_score` job: same 8
  columns and meanings as `guidance_trace.npy`, one row per abc step (TV over the abc
  segments). `guidance_trace.npy` is unchanged.
- `--dump-logits`: not required for 11c; if it is cheap to dump the first abc-step rows
  (primary / blank / blended) for a guided request, do it, otherwise say so.

### 8.4 Acceptance (Arc or CPU only, never device 0)

1. **Untouched.** No `cfg_score` (and `cfg_score: 1`): `prefix.npy`, `abc_tokens.npy`,
   `semantic.npy`, `negative_prefix.npy` byte-identical to `build_negative11b/yue2`
   (= commit 2488932; copy the binary aside first, don't rebuild that dir) for unguided,
   `cfg_scale 3`, `cfg_scale 3` + `negative_style`, `cfg_scale 3` + `negative_lyrics`,
   300 frames.
2. **Blend exactness.** Negative = the song (`negative_style == style`, lyrics): the abc
   trace shows TV = 0 and blended == primary at every step. (The score may still differ
   from the unguided one: a 2-row decode is not bit-identical to a 1-row one — known,
   STATUS_GUIDANCE; report the first differing abc token but it is not a failure.)
3. **Branch continuity.** At the transition the negative stream's token sequence equals
   `negative_prefix(vocab, r, abc_ids)` exactly (assert in a debug check or dump both;
   show it), and with `cfg_scale 3` the semantic phase's first negative row equals what a
   fresh §7 prefill of that prefix gives within f32 noise (report max |Δ|).
4. **Determinism** (same request twice → identical score and semantic), `--verify-sampler`
   covering abc steps if it can, every §8.2 error at the CLI and in `yue2-guidance`.
5. **The point.** On the Arc, AR only, a full song, one fixed seed, a real multi-verse
   lyric: style = a gentle lullaby/orchestral wording, negative_style "eurodance, dance
   beat, four on the floor", `negative_lyrics: true`, `cfg_scale 1`, `cfg_score` 1 / 1.5 /
   2 / 3. Report per run: first differing abc line vs cfg_score 1, `Q:`/`M:`/`L:`/`K:`,
   bars, mean notes per bar, chord symbols per bar, rests share, abc tokens, and the abc
   trace means (col 4 / col 6 / surprise). No listening claims.

Rules: §6, with `build_negative11c/` as the build dir. Report: a "Stage 11c" section in
`src/STATUS_NEGATIVE.md`. README: `cfg_score` next to `negative_lyrics`, one example, one
sentence that it has no reference equivalent.

## 9. Stage 11d — `cfg_score` leaves the score header alone

### 9.1 Why

At `cfg_score 3` (§8.4 item 5) the header broke: `M:6/8` came out as `M/8`. The header
(`X: T: M: L: Q: V:… K:`) is where the song and the negative agree most, so amplifying
their small differences there only damages syntax. A missing `M:` then fails silently
downstream (`Meter` defaults to 4/4, so `sections`/`handover` of a remix of that score
would time 6/8 bars as 4/4). The sampler's vocabulary clamp (`[0, EOD)` + `ABC_END`) is
already right; this is about which positions are guided, not which ids.

### 9.2 Rule

- The abc-phase blend weight is **0** (the drawn row is the primary's, bit-exact, as a
  weight-0 branch is today) until the score text has completed its first `K:` line: the
  newline that ends the first line beginning with `K:`. From the next step on, the
  weight is `cfg_score − 1` as in §8. Decide on the **decoded text** of the sampled abc
  tokens (a token may carry the newline, the `K`, or both), not on token ids; keep the
  detection incremental (no re-decode of the whole score per step).
- The negative stream is still fed every token from abc step 0 (§8.2 "one branch, both
  phases" and the §8.4 item-3 continuity check are unchanged). Only the weight changes.
- If the score reaches its first body line without any `K:` line (a line starting with
  `V:` after at least one line not starting with a header field, or a bar line `|`, or
  `%`), start guiding there and log a warning. Use the real headers in the 11c scratch
  scores to pin the exact rule; state it in STATUS.
- The abc trace keeps one row per abc step; header rows show the branch disagreement
  (col 4) with col 6 = 0.
- Log line: `guidance: score phase guided at <c> from abc step <n> (after K:)`.

### 9.3 Header check (all jobs, cheap)

After the abc phase — and when a request's given `"abc"` is parsed — if the header lacks
a well-formed `M:`, `L:` or `K:` field, print one warning naming the missing field(s) and
record `"score_header_ok": false` in `plan.json` (absent when fine, so unguided plans stay
byte-identical). Do not fail the job.

### 9.4 Acceptance (Arc or CPU only, never device 0)

1. Untouched: no `cfg_score` → byte-identical to the 11c binary (`build_negative11c/yue2`,
   copy it aside first) for the §8.4 item-1 set, `plan.json` included.
2. The header of every `cfg_score` run equals the unguided run's header byte-for-byte for
   the same seed (the §8.4 item-5 request) — or, if 2-row numerics flip a header token,
   show it is a near-tie (primary row's top-2 gap) and say so.
3. Item 5 again: `cfg_score` 1 / 1.5 / 2 / 3 / 4, same columns as the §8.4 table, plus
   the step where guidance started and whether the header check passed.
4. Header check: a hand-broken `"abc"` (drop `M:`) warns and records the flag; a good one
   doesn't. Unit cases in `yue2-guidance` for the `K:`-line detector across token splits
   (`"\nK:Eb\n"` split every possible way) and the no-`K:` fallback.
5. Determinism, `--verify-sampler`, all existing test binaries pass.

Rules: §6, build dir `build_negative11d/`. Report: a "Stage 11d" section in
`src/STATUS_NEGATIVE.md`; README one sentence (header is never guided). Timing
bookkeeping (§8 review note) is out of scope.

## 10. Stage 11d, part 2 — `score_tempo`: forcing the score's tempo

### 10.1 Why

A sampled score picks its own tempo (`Q:1/4=78`), and the tempo is one of the things a
score bakes in. The user wants to ask for a speed. The score phase already knows how to
force text (`abc_template`, SPEC_TEMPLATE §3); this forces one header value.

### 10.2 Request

```json
{ "style": "…", "lyrics": "…", "score_tempo": 90 }
```

- `score_tempo`: number, quarter notes per minute, `20 ≤ t ≤ 300` (integer or not; write
  it the way the scores do, e.g. `90`, and say in STATUS how a non-integer is written).
  Absent / `null` = unchanged. Independent of guidance: works with or without
  `cfg_scale` / `cfg_score`.
- The score's `Q:` line reads `Q:1/4=<t>`. Check the real scores (tests/out/neg11c,
  and the unit is `1/4` even for `M:6/8`) and confirm the unit is always `1/4`; if the
  model writes other units, force `1/4=<t>` anyway and say so.
- Mechanism: sampling runs as usual until the current line of the sampled text starts
  with `Q:`. From there the rest of that line is **forced**: the text is cut after
  `Q:`, anything the model sampled past the cut is discarded, and the remainder
  `1/4=<t>\n` (plus whatever piece of the cut token lay before the cut) is re-tokenized
  and fed as a given segment — the same cut-and-feed technique as a template hole's
  closing token (SPEC_TEMPLATE §3). The fed tokens enter `history` (penalty window) and,
  with `cfg_score`, the negative stream too (§8.2 continuity unchanged). They consume no
  rng draws.
- If the header reaches its `K:` line (or the §9.2 body fallback) without a `Q:` line,
  insert `Q:1/4=<t>\n` before that line by the same cut-and-feed, and log it. If that is
  unreasonably hard, warn and leave the score unforced instead, and say which in STATUS.
- Errors (exact, tested, before any GPU work): not a number / out of range; with a given
  `"abc"`, `abc_template` or `cot: "off"` (no header is sampled). `semantic_keep` is
  already abc-given — same error.
- Log line: `score: tempo forced to 1/4=<t> (the model wrote <what it sampled or "no Q:">)`.
- `plan.json`: `"score_tempo": t` when set, and `"score_tempo_sampled"`: the value the
  model had begun to write (string, may be partial) or null. `config.json`:
  `"score_tempo": t | null`. Not in the engine's `request.json`.

### 10.3 Acceptance (Arc only, never device 0; add to the §9.4 report)

1. Untouched: without `score_tempo`, byte-identical to the 11c baseline (§9.4 item 1).
2. The score's `Q:` line reads exactly `Q:1/4=<t>` for `t` = 60, 90, 140 on the §8.4
   item-5 request (unguided), and the rest of the header is unchanged from the unforced
   run up to the `Q:` line.
3. **It changes the speed.** Unguided, same seed and lyric, `score_tempo` 60 / 90 / 140,
   full song AR only: report per run the bar count, the song's semantic seconds, and
   seconds per bar vs the expected `bar beats × 60 / t` (for 6/8 in 1/4 units that is 3
   quarter notes per bar). Say plainly whether the audio followed the forced tempo.
4. With `cfg_score 2` + `negative_lyrics`: forced `Q:` still exact, continuity check
   (§8.4 item 3) passes, guidance starts after `K:` as §9.
5. Unit cases in `yue2-guidance` for the `Q:` detector across token splits and for the
   errors. Determinism with `score_tempo` set.
