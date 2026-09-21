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

What survives a fixed score is a style that changes *how the notes sound*: a drum
kit, a guitar tone, a vocal group, a synth palette (ska, chiptune, eurodance, death
metal and a barbershop quartet all came through on a gentle lullaby score, some
with a singer of their own). What does not is a style that needs *other notes*:
opera over a line written for the base singer stayed the base singer, polka was
polka "if you squint", dubstep came out as generic electronic, gospel as a
clap-along. The same goes for a mood: "despair" or "acceptance" are compositions —
slower notes, other chords — and tags for them changed nothing we could hear over
five seeds each. And name what you want, never what you don't: the one take tagged
"no drums" had drums.

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

**A cut inside a verse.** Sections are the easy anchors, but a cut can go anywhere:
play the base take, note the time, and write `"at": "1:18.6"` (or `"at": 78.6`)
instead of `section` — `"frame"` is the same thing in 25 Hz frames. The time stays
good through a whole chain, because every leg is spliced back onto the song's own
clock. For a 1 s cut, aim for the gap *between* words: cut while a word is still
held and the incoming take decides afresh how it ends (ours turned a drawn "bow"
into a bowed one). Expect to nudge such a cut by half a second after hearing it.

An entry with `"style": "…"` instead of `"take"` makes the engine render that take
itself (saved as `cycle/take_<k>/`, reusable later); an entry with neither goes back
to the base. `cycle/handover.json` records where each cut landed and how well the
takes were aligned.

### What the engine actually does

The method has a name: **history intrusion**. The incoming style's context is
intruded on by a few seconds of the real song, and that is all — no `guidance`,
no weights, no second decoder steering the first. It is the best method we have;
logit guidance (README, `guidance`) is the older one, kept for colouring a band
around a singer, and the two have not been combined successfully (§6).

Nothing runs in parallel and nothing is blended; it is one request, rendered
without anyone watching.

1. **Every take is rendered in full first**, one after another: the base style,
   then each other style from the same score and seed. Each is an ordinary song
   you could listen to on its own.
2. **There is one master stream: the song so far.** It starts as a copy of the base
   take, and it is the only thing the audio stage ever sees.
3. **At a cut, a leg is a fresh generation under the incoming tags with a forced
   history**: the incoming take's own tokens from the start of the song up to
   `seconds` before the cut, then the last `seconds` of the song so far. Those
   tokens are fed in, not sampled; sampling starts at the cut. The model sees a
   minute of its own style, then a few seconds of what is really playing, and goes
   on from there.
4. **The leg's new tokens replace the song so far from the cut on**, shifted by
   the measured offset so they land on the song's clock. A leg stops just past the
   next cut; the last one runs to the song's own end.
5. The next cut does the same, with the new song so far as the outgoing side — so
   the seconds each leg hears are the song as it really went, not the previous
   style's full take. That is why a chain holds together.

So the timeline never changes hands: the song's clock is the base take's
throughout, and legs are grafted onto it. The *context* does change hands,
completely: after a cut the model's long memory is the new take's, and all that is
left of the old style is what came through those few seconds. The full takes are
donors of "what a minute of this style sounds like"; their audio is never used.

The history is set once, at the cut, and then the model runs free. The gradual
change you hear afterwards is the model's own doing: it continues the singer it
just heard, every token it writes pushes those seconds further back, and the new
style usually wins at the next breath. Nothing in the engine ramps or fades.

The audio stage runs once, over the finished stream, with the base request's
prefix; which take's prefix it uses makes no audible difference.

### The one knob: `seconds`

| `seconds` | what you hear |
|---|---|
| 5 (default) | A blend. The band changes at the cut while the old singer finishes the thought; the new singer enters at the next natural entry. 5–15 s sound about the same. |
| 1 | A hard cut on the section boundary: new band and new singer at once. |

Use 5 unless you have a reason not to. A change is not an instant: the model takes
around ten seconds to let go (more or less, depending on how far apart the styles
are and where the next breath falls). The singer in those last seconds holds on
while the incoming take's long memory pulls the band across, and the voice
follows. With 5 s the model
has enough of the singer to carry a line through the change — a little of the old
style bleeding into the new one is the price of a mix that holds together. With 1 s
inside a verse it has almost nothing to hold on to: the same line, cut at the same
spot, came out as a singer who did not know who she was, and became a clean morph
at 5 s. Keep 1 s for section boundaries between styles that share nothing.

The incoming tags are read *against* what was just played. "doom ballad, distant
funeral drum" handed five seconds of metal drumming kept the hard drums; the same
cut with tags that named no drum at all ("slow lament, sparse, bare low cello")
dropped to the quiet it was meant to be. Name what should arrive, and do not name
a quieter version of what should leave.

### A softer change: prepare the cut

Hand the same take over twice — first with a long intrusion, ten seconds before the
cut, then normally at the cut:

```json
{ "at": "1:09.3", "take": "takes/metal", "seconds": 30 },
{ "at": "1:19.3", "take": "takes/metal" }
```

The first entry gives the model a history that is half the new take and half the
song as it really went, so for those ten seconds it stays in the old style — but it
knows what is coming, and it plays a lead-in: a cymbal riser into the downbeat
where the plain handover had a gap and a breath, a single drum hit announcing the
next section. We saw the riser on three seeds out of three with the preparation
and on none of three without. The change itself then arrives later and cleaner: on
our hardest join (a lullaby singer handed to symphonic metal in the middle of a
verse) the line was finished in her own voice and the metal came in on the next
phrase, with no wobble. The cost is that the new style is ten seconds or so late.

The new take has to be a real part of that first history. With only ten seconds of
it, at the far end, the preparation did nothing at all.

What did *not* help was the opposite, handing the same take over *again* ten
seconds after the cut to "reinforce" it: harmless, never audible as a seam, and
never closer to the take than the song got on its own in the same ten seconds.

### Is it the recipe or the luck?

Every leg is drawn from the request's `seed`. Give the takes as directories
(`"take"`, `"base_take"`) and change only `seed`, and you re-roll the legs while
the takes stay what they were — a second and third opinion on a transition for the
price of the legs alone. Use it before believing any difference between two
recipes, and when one leg came out poorer than its take deserves.

None of this is automatic. `handover.json` says whether two takes line up in
time, not whether the join is musical; listen, then move a cut by a line, change
`seconds`, or reword the incoming tags, and render again.

### Alignment is automatic, and it matters

Two renders of one score do not run at the same pace; one leads the other by up to
a couple of seconds and the lead can drift through the song. The engine measures
the offset at every cut from the tokens themselves (at the right lag two takes of
one score share a few percent of *identical* tokens, at any other lag none). A 5 s
handover forgives half a second of error; a 1 s handover stumbles on it. When the
measurement cannot be believed — a cut in the first half minute has little to
match, and so does a long instrumental stretch — the engine splices at offset 0,
warns, and `handover.json` says `"confident": false` with the value it did not
trust as `"measured"`. Early in a song 0 is right (the takes have not drifted
yet); later, prefer 5 s there, move the cut to a sung section, or give
`"offset"` yourself.

### Choosing the order

- A strong style (four-on-the-floor, a quartet, anything with its own groove)
  takes over at once, from anything.
- Going *back* to a gentle style is the hard direction: the voice returns on time
  but the previous beat can ride along for half a minute, because nothing in the
  gentle style displaces it. Use a 1 s handover for the return, or return where the
  outgoing band rests.
- The cut does not need time to "settle", but the listener needs time to recognise
  the style: legs of 25–60 s read as genres; ten hops with several 15 s legs
  blurred into one, seven hops over the same song were fun.

## 5. Both at once (untried)

The two recipes should stack — a handover whose incoming take was rendered from a
score with the lowered vocal line — but the engine insists that every take's
`score.abc` matches the base byte for byte (the alignment depends on it), so today
that means rendering the base from the edited score too. Not tested yet; a 1 s
handover already changes the singer along with the band.

The same route is the honest answer when a section needs other *notes* (a quieter
ending, a mood): edit the score — we rested out the accompaniment line from one
verse to the end, bar counts unchanged — render every take from the edited score,
then chain as usual. It worked (the tail changed character where tags alone had
not), but it is hand-editing ABC, not a request key, and takes of an edited score
paced less alike than before (one came in 2 s behind; the alignment absorbed it).

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
- **Asking the model to rewrite a section in the new mood** (`abc_template` holes
  over both voices of a verse and chorus, new tags, four seeds): it copied the
  pattern of the verses above it almost note for note. Written score history
  outweighs the tags, just as audio history does.

## 7. Cost

One AR pass per style plus one short pass per cut, then a single NAR and VAE pass
for the final stream. A four-style cycle of a 4-minute song is roughly three times
the AR work of a plain song and the same NAR/VAE work.
