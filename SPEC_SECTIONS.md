# SPEC_SECTIONS — stage 8: a style per section

Builds on stage 7 (SPEC_GUIDANCE.md, SPEC_KEEP.md, `src/STATUS_GUIDANCE.md`).

## 1. Why (measured, by ear and by trace)

- In the semantic phase the *score* decides who sings: a vocal line moved an octave
  down from one section on brought in a male singer at that bar on every seed tried,
  even with "female vocal" still in the tags. The tags then colour the voice. A tag
  change the score contradicts (a deep voice over a line written in a high register)
  is a fight: coin-flip takeovers, wobble, garbled words under strong guidance.
- In the score phase the tags do bend what is written (register, note density,
  range), section by section, when the prefix carries other tags while that section
  is written.
- So the mid-song style change is: write each section's score under that section's
  tags, then swap the tags at the matching frame of the semantic phase — a plain
  swap, no guidance branches. The weighted curves of stage 7 stay available on top.

## 2. Request

```json
"sections": [
  { "section": "verse", "nth": 3, "style": "…tags from here on…" },
  { "section": "chorus", "nth": 2, "style": "…", "lead_frames": 35,
    "against": { "previous": [[0, 4], [250, 2]], "blank": [[0, 2]] } }
]
```

- `section` + `nth` (1-based, default 1) name the nth score label of that name: the
  planner's `% verse` / `% chorus` / … lines (closed vocabulary; see §6 survey).
  Entries must resolve to strictly increasing score positions; order in the array is
  the order in the song.
- `style`: required, the full tag string from that section on (replaces, does not
  append). The request's own `style` covers everything before the first entry.
- `lead_frames` (default 35, 0..250): the semantic swap happens this many frames
  before the section's bar line (the render runs ahead of the score clock by about a
  second, and the swap wants to land in the breath before the phrase).
- `against`: optional, exactly the stage-7 object. Absent = plain swap.
- Unknown keys are errors. `sections` with `guidance`: error (one mechanism per
  request; `sections` compiles into guidance entries). With `abc_template`: error
  for now. With `cfg_scale != 1`: error, as for guidance.
- An entry whose label never appears in the written score is reported
  (`"reached": false`), not an error — like an unreached guidance frame.

## 3. Score phase (only when the engine writes the score)

- While sampling the score, when a completed line is the label line an entry names
  (nth occurrence counted over completed lines), the sequence is re-prefilled as
  `prefix(with the entry's style) + score text so far` and sampling continues. The
  label line itself was written under the old tags; everything after it under the
  new ones. Same machinery as the template path's re-prefill. RNG: no reset, no
  extra draws.
- The bar index of each label line is recorded as it is written (bars counted over
  the `V: Vocal` body lines completed before it; the engine already counts bars for
  templates — share that code).
- `prefix.npy`, the NAR prefix and the frame-0 semantic prefix use the request's own
  `style` (NAR ignores style text; keep one definition).
- With `"abc"` given, nothing is written, so this phase only locates the labels and
  their bars in the given score.

## 4. Semantic phase

- Frames per bar = 25 Hz × 60 / Q × beats per bar, from the score's `M:` and `Q:`
  header (the planner writes `Q:1/4=N`; inline `[M:…]` changes are honoured when
  counting: sum the bars' own lengths). Entry frame = round(bar start in seconds ×
  25) − lead_frames, clamped to ≥ 1 and to strictly increasing.
- The resolved entries are handed to the stage-7 machinery as guidance entries
  `{frame, style, against}`. A plain swap is an entry with no curves: at the cut
  the new prefix is prefilled, the old sequence is dropped at once, no extra KV
  stream is ever live. Make sure that path allocates nothing it does not use: a
  sections request without any `against` must run with `n_seq_max` as small as the
  swap allows (2 during the prefill hand-over is fine if that is what the relabel
  code needs; say which in STATUS) and should NOT be forced to `--parallel 1` if it
  can be avoided cheaply — if it cannot, keep the stage-7 rule and say so.
- `semantic_keep` combines: entries whose frame < N are an error as in SPEC_KEEP.

## 5. Artifacts

- `sections.json`: the entries as resolved — label, nth, line number in score.abc,
  bar index, bar-start seconds, frame, reached.
- `guidance.json` is still written (the compiled entries), so stage-7 tooling and
  `--guidance-trace` work unchanged.
- `request.json` stays reference-loadable (no `sections` in it), as for guidance.

## 6. Survey first (no GPU, do this before coding)

Over whatever planner-written scores are available to you read-only
(`tests/out/**/score.abc`, `tests/golden/`, `docs/examples/`; NOT `../songs`):
- confirm the label vocabulary and spelling (`% verse`, `% pre-chorus`?, …);
- how lyric `[Section]` tags map to score labels — does the kth `[Verse]` in the
  lyrics become the kth `% verse`? The planner inserts `% interlude`/`% intro` on
  its own; per-name counting is meant to be robust to that. Report exceptions.
Put the findings in STATUS; if per-name counting is unreliable, say so loudly and
propose the anchor that is.

## 7. Acceptance (Arc `--gpu 1`; no CPU model runs)

1. No `sections`: `cmp` prefix/abc_tokens/semantic vs HEAD's binary for unguided,
   guided, keep requests.
2. Score phase: a request with one entry → `abc_tokens.npy` identical to the
   no-sections run up to and including the label line, different after; the log
   shows one re-prefill; `sections.json` line/bar are right (check by hand against
   score.abc). Determinism: twice → identical.
3. An entry whose style equals the request's style → the score after the label is
   allowed to differ only through prefill-vs-decode numerics; report whether it is
   identical (informative, not pass/fail).
4. Frame arithmetic: table tests in `yue2-guidance` for 4/4 Q=120 (50 frames/bar),
   3/4, 6/8, Q=90, an inline `[M:3/4]` mid-score, lead clamp, non-increasing
   entries, every §2 error.
5. Semantic phase: codes identical to the no-swap run (same written score fed back
   as "abc", same seed) up to the entry frame, different after; plain swap shows no
   branch prefill in the log and `cfg_branches`/max live streams as designed.
6. `"abc"` + sections (no score phase) works; `semantic_keep` + sections works.
7. `--verify-sampler` matches on a sections run.
8. Cost: re-prefill seconds per entry; plain-swap song vs unguided song wall time.

## 8. Rules

Build dir `build_sections/` only (`nice -n 10 cmake --build build_sections -j8`);
other build dirs untouched. Arc only, never device 0, no CPU model runs. No commits.
Public repo hygiene. Report: `src/STATUS_SECTIONS.md`. README: a "style per section"
part next to `guidance`, leading with the finding in §1 (the score decides the
singer; match the voice tag to the written register) — plainly, no song titles.
`docs/ROADMAP.md`: Done line + follow-ups you found.
