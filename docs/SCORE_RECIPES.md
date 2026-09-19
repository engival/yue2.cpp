# Working with a score

A render started with `--artifacts DIR` leaves the model's score in
`DIR/score.abc` and, next to it, a `request.json` that reproduces the song. This
page is a how-to for the edits you can make to that score and feed back through
`"abc"` or `"abc_template"`, so the same song comes back in another key, another
style, or with a different accompaniment. The request fields themselves are in
the [README](../README.md#quick-start). One real score in every form below is in
[examples/](examples/).

## 1. What a score looks like

```
X:1
T:
M:4/4                       meter
L:1/32                      unit length: a bare number after a note counts 32nds
Q:1/4=120                   tempo
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Em                        key
% intro                     section label, one per [Section] tag in the lyrics
V: Vocal
z24z4"Em"z4|"Em"z32|"B"z32|"Em"z32|
V: Ins
Z|e4B4g4B4f4B4e4B4|^d4B4f4B4e4B4d4B4|e4B4g4B4f4B4e4B4|
V: Vocal
"B"z32|"Em"z32|"B"z32|"Cmaj7"z32|
V: Ins
^d4B4f4B4e8d8|e8g8f8e8|^d8f8e8d8|e8g8f8e8|
% verse
V: Vocal
"Em"z4e8B4e4B4e4B4|"B"^d8z24|"Em"z4e8B4e8B4B4-|"B"B4A8z16z4|
V: Ins
Z4|
...
```

- Everything after `K:` is one timeline read top to bottom. A `V: Vocal` body
  line and the `V: Ins` body line under it cover the **same bars**: that pair is
  one system. Time advances system by system.
- `% name` lines are section labels. They follow the `[Section]` tags of the
  lyrics in order.
- Bars are separated by `|`. In a bar: `e4` is the note E for 4 units (an eighth
  at `L:1/32`), `z4` a rest of that length, `Z` a whole-bar rest, `Z4` four of
  them, `e32` a whole bar, `-` a tie, `^d` D sharp, `=f` natural, `_b` flat.
  `E`, `e`, `e'` are successive octaves; `B,` is the octave under `B`.
- `"Em"` before a note or rest is a chord symbol. Chord symbols sit on the
  **Vocal** line only, including on its rests, so a section the singer sits
  out (intro, interlude, outro) still carries the harmony on the Vocal line.
- Header fields (`X: T: M: L: Q: V: K:`) start with a letter and a colon;
  body lines never do. Editing tools can tell them apart by that.

## 2. Same score, another performance

Put the score text in the request's `"abc"` and change `style`, `seed`, or both.
The AR skips writing a score and sings the given one; the semantic tokens are
sampled fresh, so each seed is another take.

```json
{ "style": "…new style…", "lyrics": "…same lyrics…", "abc": "…score.abc…", "seed": 3 }
```

Keep the lyrics as they were. A kept score fixes the phrase lengths, and other
words sung to them come out garbled. Section tags may change (§6).

## 3. Another key, or another register for the singer

`scripts/abc_transpose.lua` moves every note of both voices, the chord symbols
and the `K:` field, respelling in the new key, and re-reads the result to check
each pitch before printing it. A second number moves the Vocal line a further
number of octaves on top, the Ins line staying where it is. Plain Lua 5.3+.

```bash
lua scripts/abc_transpose.lua DIR/score.abc  5     > up5.abc      # Em -> Am
lua scripts/abc_transpose.lua DIR/score.abc  5 -1  > up5_low.abc  # Am, singer an octave down
lua scripts/abc_transpose.lua DIR/score.abc -2     > down2.abc    # Em -> Dm
```

Vocals sound an octave below the written pitch. The written register of the
Vocal line is part of what picks the singer, so a register move and a vocal tag
in the style text work together.

Loaded from Lua instead (`dofile("scripts/abc_transpose.lua")`) the file
returns `transpose(abc, semitones, vocal_octaves) -> abc, "Em -> Am"`.

## 4. Melody only: the harmony is the model's

Remove every chord symbol from the body lines and ask with `"cot": "melody"`:

```bash
sed -E '/^[A-Za-z]:/! s/"[^"]*"//g' DIR/score.abc > melody.abc
```

The `V:` header lines carry quoted names and must keep them; the `/^[A-Za-z]:/!`
guard skips them.

## 5. Another accompaniment

The Ins line can be handed back to the model in two ways.

**Unwritten.** Replace each Ins body line with a whole-bar rest per bar it
held (a four-bar line becomes `Z4|`). The score still goes as `"abc"`; the
accompaniment is then whatever the semantic stage improvises under the kept
vocal, differently per seed.

**Rewritten.** Replace each Ins body line with the directive `%%yue2-gen` and
send the text as `"abc_template"` instead of `"abc"` (needs `cot` `full` or
`melody`). The model writes each Ins line in turn, seeing every line above it,
including the Vocal line it sits under and its chord symbols. Holes take their
bar count from the body line above; `%%yue2-gen bars=N` overrides it. Details
of retries and rest-fills are in [SPEC_TEMPLATE.md](../SPEC_TEMPLATE.md).

```
% intro
V: Vocal
z24z4"Em"z4|"Em"z32|"B"z32|"Em"z32|
V: Ins
%%yue2-gen
V: Vocal
"B"z32|"Em"z32|"B"z32|"Cmaj7"z32|
V: Ins
%%yue2-gen
```

A small script does this for a whole score:

```bash
awk '/^V:/ { voice = $2 }
     /^[A-Za-z]:/ || /^%/ || NF == 0 || voice != "Ins" { print; next }
     { print "%%yue2-gen" }' DIR/score.abc > template.abc
```

The artifacts of a template render hold the finished score with the holes
filled (`score.abc`) and the template as sent (`template.abc`), and their
`request.json` carries the finished score as a plain `"abc"`.

**The Vocal line the same way.** Replace each Vocal body line with
`%%yue2-gen bars=N`, N counted from the line it replaces (a hole's default count
comes from the line above it, which for a Vocal hole is the previous system's
Ins line, and section tails are often shorter). The model then writes melody
and chord symbols over the kept accompaniment. A Vocal hole is written before
the Ins line under it is fed, so the harmony it picks is its own.

**Keep a section, rewrite the rest.** Hole both voices everywhere except the
sections to keep, `% intro` say. The kept opening then steers what follows.

**An opening, not a form.** Give the beginning of a score and end the template
with `%%yue2-continue`: the model writes the rest freely, without the section
labels and bar counts of the source. The opening may be incomplete, one voice
only for instance; the render shows what the model makes of it.

**Harmony for a given accompaniment.** A score written top to bottom gives a
section's Vocal line no view of the Ins line under it, and an opening handed
over without chord symbols is continued without them (the model treats the
score as the chordless dialect from then on). `%%yue2-chords` in place of a
Vocal body line has the model read the Ins line below first, write a line of
rests with a chord symbol per bar, and put it back above the Ins line. An intro
given as Ins lines only becomes a proper opening this way, and what follows,
holes or a free continuation, is written with chords:

```
% intro
V: Vocal
%%yue2-chords
V: Ins
Z|e4B4g4B4f4B4e4B4|^d4B4f4B4e4B4d4B4|e4B4g4B4f4B4e4B4|
V: Vocal
%%yue2-chords
V: Ins
^d4B4f4B4e8d8|e8g8f8e8|^d8f8e8d8|e8g8f8e8|
%%yue2-continue
```

The directive must sit between a `V:` line and the other voice's `V:` line
plus body line, one system at a time. The chords it picks follow the key
field and the figure: the same intro figure moved to another key gets another
progression.

The template format also has a primer block (`%%yue2-primer-begin` …
`%%yue2-primer-end`): lines fed as context at that point and dropped from the
score. `yue2` prints a warning when a template carries one; see the README.

## 6. Other section tags, same words

Lyrics tags may carry a qualifier after a colon, `[Verse: uneasy]`, and the
qualifier can differ from the original render's. Keep the standard word first,
and keep every non-tag line identical to the original (§2).

## 7. Read the score before rendering

Writing the score is the AR's first phase and takes a few seconds; the
semantic phase is the long one. To look at what a seed writes before spending a
render on it, run the AR alone with the semantic phase cut to a stub:

```bash
build/yue2 ar -m yue2-ar-q8_0.gguf --request plan.json --artifacts plan/s3 --max-semantic 32 --gpu 0
```

`plan/s3/score.abc` is the finished score. With a template, the written Ins
lines are the ones that were `%%yue2-gen`; the rest is what was given. Pick the
seed you want, then render it with that score as a plain `"abc"`, or rerun the
template with the same `seed`.

## 8. Combining edits

The edits compose, in this order: transpose (§3), strip chords (§4), rest out
or hole the Ins line (§5), then decide `"abc"` versus `"abc_template"` and
`cot`. Name the output after the edits (`song__style_tr+5_vo-1_newins_s3`) so
an artifacts directory says how it was made; its `request.json` records the
score it sang either way.
