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
