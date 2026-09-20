# Cookbook: change the singer or the genre in the middle of a song

One song, several styles: a lullaby that turns into eurodance at the chorus, a
quartet that takes verse three, and back. This page is the process that came out of
a long series of test renders, judged by ear, on two songs. The request fields are
documented in the [README](../README.md); score editing is in
[SCORE_RECIPES.md](SCORE_RECIPES.md).

## 1. What the model listens to

YuE2 first writes a score (ABC: a vocal line, an instrument line, chords), then
"performs" it as a stream of semantic audio tokens at 25 per second. While
performing, three things steer it — strongest first:

1. **The score.** The written register decides *who* sings. A vocal line an octave
   lower brings in a man, whatever the tags say.
2. **What it has already played.** The long history is the band; the last few
   seconds are the singer and the groove. The model continues what it hears.
3. **The style tags.** They colour the result. They do not win against 1 or 2.

Everything below follows from that order. A new *singer* is a score edit plus a tag
swap. A new *genre* needs a new history, and the way to get one is to borrow it from
another render of the same score.

## 2. Start from a render you like

```sh
yue2 song --request song.json --out base.flac --artifacts base --seed 1
```

`base/score.abc` is now the song. Every recipe here keeps that score, so the tune,
the words and the timing stay and only the performance changes.

## 3. A new singer from a section on

Move the vocal line by whole octaves from the section where the new singer enters
(the band line and the chords stay where they are, so only octaves are safe), and
swap the tags at the same place with `sections`:

```sh
lua scripts/abc_transpose.lua base/score.abc 0 -1 verse:3 > male_v3.abc
```

```json
{
  "lyrics": "…as before…",
  "style": "…the base tags, female vocal…",
  "abc": "…contents of male_v3.abc…",
  "sections": [ { "section": "verse", "nth": 3,
                  "style": "…the same tags with: male vocal, warm tenor…" } ]
}
```

- −1 octave is a tenor, −2 is a deep voice (pair it with tags that say so). Tags
  that ask for a deeper voice than the line is written for give strain and blips.
- Later choruses tend to come back as a duet in octaves: once both voices are in
  the history, both are available. Most people like it.
- Add `"semantic_keep": {"file": "base/semantic.npy", "frames": N}` with `N` just
  below the cut (see `sections.json` of a first run for the frame) to keep the
  first half bit-identical to the render you liked.

## 4. A new genre from a section on: the handover

Tags alone will not turn a string ballad into eurodance at bar 60: the history says
ballad. So render the whole song once per style from the same score, and at the
cut let the new style's renderer take over — with its own take as its long memory
and only the last few seconds of the song so far spliced in, so it knows where the
song is. It picks up the phrase and the beat, and within a bar or two it is playing
its own style.

**Audition the takes first.** A style is only as different as this score lets it
be: some tags barely move a given song (we had a "reggae" that was not, and a
"power metal" that dropped the metal whenever the singing started). The handover
delivers the take as it is, so listen to it before you build on it.

```sh
# take.json = song.json with another "style" and "abc" = base/score.abc
yue2 song --request take_euro.json --out take_euro.flac --artifacts takes/euro --seed 1
```

Then one request names the cuts:

```json
{
  "lyrics": "…as before…",
  "style": "…the base tags…",
  "base_take": "base",
  "handover": [
    { "section": "chorus", "nth": 1, "take": "takes/euro" },
    { "section": "verse",  "nth": 3, "take": "takes/quartet", "seconds": 1 },
    { "section": "verse",  "nth": 4 }
  ]
}
```

```sh
yue2 song --request cycle.json --out cycle.flac --artifacts cycle
```

An entry with `"style": "…"` instead of `"take"` makes the engine render that take
itself (saved as `cycle/take_<k>/`, reusable later); an entry with neither goes back
to the base. `cycle/handover.json` records where each cut landed and how well the
takes were aligned.

### The one knob: `seconds`

| `seconds` | what you hear |
|---|---|
| 5 (default) | A blend. The band changes at the cut while the old singer finishes the thought; the new singer enters at the next natural entry. 5–15 s sound about the same. |
| 1 | A hard cut on the section boundary: new band and new singer at once. |

### Alignment is automatic, and it matters

Two renders of one score do not run at the same pace; one leads the other by up to
a couple of seconds and the lead can drift through the song. The engine measures
the offset at every cut from the tokens themselves (at the right lag two takes of
one score share a few percent of *identical* tokens, at any other lag none). A 5 s
handover forgives half a second of error; a 1 s handover stumbles on it. If
`handover.json` says `"confident": false` (long instrumental stretches give the
measurement little to hold on to), prefer 5 s there or move the cut to a sung
section.

### Choosing the order

- A strong style (four-on-the-floor, a quartet, anything with its own groove)
  takes over at once, from anything.
- Going *back* to a gentle style is the hard direction: the voice returns on time
  but the previous beat can ride along for half a minute, because nothing in the
  gentle style displaces it. Use a 1 s handover for the return, or return where the
  outgoing band rests.
- Sections shorter than ~30 s are fine; the cut does not need time to "settle".

## 5. Both at once (untried)

The two recipes should stack — a handover whose incoming take was rendered from a
score with the lowered vocal line — but the engine insists that every take's
`score.abc` matches the base byte for byte (the alignment depends on it), so today
that means rendering the base from the edited score too. Not tested yet; a 1 s
handover already changes the singer along with the band.

## 6. What did not work (so you need not try)

- **Forcing tags against the score** (`guidance` weights above ~6, or a deep-voice
  tag over a high line): coin-flip takeovers, wobble, garbled words.
- **`guidance` together with `semantic_keep` for a genre change:** the kept history
  wins; the push only damages the singing. A gentle push (≤ 6, plus `blank` 2) on a
  *fresh* take does bend the genre and is the fallback when no full take is wanted.
- **Cross-fading the tags after a handover, staging the handover in two steps, or
  giving the new renderer the old style as its long memory:** all sound the same as
  the plain handover. The last seconds plus the long memory decide; nothing else is
  audible.
- **Long intrusions (30 s):** no better than 5 s.

## 7. Cost

One AR pass per style plus one short pass per cut, then a single NAR and VAE pass
for the final stream. A four-style cycle of a 4-minute song is roughly three times
the AR work of a plain song and the same NAR/VAE work.
