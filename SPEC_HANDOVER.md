# SPEC_HANDOVER — stage 9: hand the song over to another style's take

Builds on stage 7 (`semantic_keep`, SPEC_KEEP.md) and stage 8 (the score clock,
SPEC_SECTIONS.md). Opt-in; a request without `handover` is bit-identical to before.

## 1. Why (measured by ear, two songs, ~25 renders)

- In the semantic phase the model follows, strongest first: the score, the audio
  history (long history = the band, last seconds = the singer), the tags. Swapping
  tags mid-song (`sections`) changes the singer when the score agrees, but a *genre*
  change needs the history to change too, and pushing tags against history
  (`guidance`) is seed-dependent.
- What works every time: render one full take per style from the SAME score. At the
  cut, run the incoming style's renderer with a forced history = its OWN take up to
  X frames before the cut, then the last X frames of the song so far; it samples
  from the cut on. The song = everything before the cut + that continuation. The
  renderer's long memory is already the new style, the short intrusion ties it to
  where the song is (phrase, beat, singer), and it recovers into its own style
  within a bar or two. X = 5 s gives a gradual blend (band first, singer at the next
  natural entry); X = 1 s on a section boundary gives a hard cut. Chains work
  (A→B→C→D→A).
- Two takes of one score do not run at the same pace: a take leads or lags another
  by up to ~2 s, and the lead can drift over the song. Short handovers stumble when
  the offset is wrong by half a second. The offset is measurable from the semantic
  tokens alone: at the right lag two takes of one score share ~2–3 % IDENTICAL
  tokens over a 30 s window, ~0 % at every other lag (z ≈ 8–13).

## 2. Request

```json
"style": "…the song's own tags…",
"handover": [
  { "section": "chorus", "nth": 1, "style": "…tags…" },
  { "section": "verse",  "nth": 3, "take": "takes/reggae", "seconds": 1 },
  { "section": "verse",  "nth": 4 }
]
```

- Where: `section` + `nth` exactly as in `sections` (same resolver, same
  `lead_frames`, default 35), or `"frame": N` instead (the base take's timeline),
  or `"at"` = the same place as a time in the base take's AUDIO, the thing a
  listener reads off a player: a number of seconds (`78.6`) or a string
  `"78.6"`, `"1:18.6"`, `"1:18"` (minutes:seconds, seconds < 60, at most one
  colon, decimals optional, nothing else). frame = round(seconds × 25); no
  `lead_frames` is applied (the listener already heard where it is). Exactly one
  of `section` / `frame` / `at` per entry; `nth` and `lead_frames` only beside
  `section`. From there on an `at` entry IS a `frame` entry — same ordering rule,
  same arithmetic checks, same errors (naming the entry by its `at` text).
  Cuts must be strictly increasing; order in the array = order in the song. The
  entries that name a frame are held to that before anything else happens,
  whatever the labels between them resolve to.
- What takes over, one of:
  - `style`: the engine renders that style's full take itself (§3).
  - `take`: directory of an earlier render's artifacts (`semantic.npy`,
    `score.abc`, `request.json`), relative to the request file like `semantic_keep`.
    Its `score.abc` must be byte-identical to the base score — otherwise a request
    error naming both files (alignment and the whole premise depend on it). What
    is compared is always a written score: when the base is the request's own
    `"abc"`, the comparison is against what the engine *would write* as its
    `score.abc` (the same tokenize → detokenize round trip), so the very request
    that produced a take is never refused over trailing whitespace. Its style is
    read from its `request.json`.
  - neither: back to the request's own style = the base take.
  - Entries with an equal style string share one take.
- `seconds` (default 5, 0.2–30): the intrusion X, in frames = round(seconds × 25).
- `offset`: `"auto"` (default) or an integer = frames the incoming take runs AHEAD
  of the song so far at the cut (§4).
- Top level `"base_take": "dir"`: use an earlier render as the base instead of
  rendering it (same directory contract; supplies the score when the request has no
  `"abc"`; with `"abc"` given the two must be the same score). Its own
  `request.json` style is *not* adopted — a leg that hands back to the base
  renders under the request's `style` over that take's history — so a base take
  rendered under other tags is worth one warning naming both strings.
- What is arithmetic is a request error before anything is rendered: `frame - x <
  1` for an entry that names a frame, and `frame - x - offset < 1` when the
  offset is given too.
- Unknown keys are errors. `handover` beside `guidance`, `sections`,
  `semantic_keep`, `abc_template`, `cot: "off"` or `cfg_scale != 1`: request error.
  `yue2 batch`: request error for now.

## 3. Takes

- Base take = the request as it stands without `handover` (writes the score unless
  `"abc"` is given). Every other take = the same request with that `style` and
  `"abc"` = the base score; same seed. A take is exactly what `yue2 ar` would have
  produced for that request on its own (bit-identical; that is the test).
- The model is loaded once. Takes MAY be rendered concurrently over the existing
  batched AR path if that is cheap to wire; sequential is acceptable. Say which in
  STATUS and what it costs.
- With `--artifacts DIR`, each engine-rendered take is saved as an ordinary
  artifacts directory `DIR/take_<k>/` (k = 0 base, then in order of first use), so
  it can be auditioned (`yue2 nar` + `yue2 vae`) and reused later via `take`.

## 4. Offset (auto)

Integer-only, deterministic. `S` = the song so far, `T` = the incoming take, `c` =
the cut. For k in −100…100: rate(k) = mean(S[t] == T[t − k]) for t in [lo, c), lo =
max(150, c − 750), indices outside `T` skipped. Offset = argmax (lowest |k| on
ties, and the negative k when two are equally far from 0); z = (max − mean) / std
over the 201 rates.

A lag is believed when **z ≥ 6 and the peak holds at least 6 identical tokens** —
an absolute count, because over a short window a single coincidental hit at one
lag and none at the others is a z of 14, and roughly seven in ten unrelated
stream pairs pass a z test alone. Two takes of one score clear both by a wide
margin (2–3 % of a 750-frame window is ~20 tokens). If the fit is not believed,
retry with lo = 150; if it is still not, warn, **use offset 0** and record
`"confident": false` plus `"measured": k` (the best k that was not believed).
Why 0 and not k: the song so far is always re-indexed onto its own clock (§5),
so 0 is the neutral guess, and a wrong k is worse than none — measured: a cut
18 s into a song (300 frames to match) took an unbelieved +18, skipped 0.7 s of
song, and every later cut then measured ~+18 against every take. An explicit
integer `offset` is never second-guessed.

## 5. A leg

For entry i with cut `c`, offset `off`, intrusion `x`:

- keep = T[0 : c − x − off] ++ S[c − x : c]; the leg is exactly the generation that
  a request `{style: T's style, abc: base score, semantic_keep: keep, frames:
  c − off}` with the request's seed produces today (same RNG contract as
  SPEC_KEEP). Do not write temp files for it — hand the codes over in memory.
- The leg stops 25 frames after the next entry's cut (in its own clock: next_c −
  off + 25); the last leg runs to its natural end.
- S ← S[0 : c] ++ leg[c − off : ].
- Errors (request error if knowable before sampling, otherwise the entry is skipped
  and reported): c − x − off < 1, c − off beyond T, c beyond S (the song ended
  before the cut: `"reached": false`, later entries too).
- The final `S` is the semantic stream the NAR gets; `prefix.npy` and the NAR prefix
  are the base request's, as in SPEC_SECTIONS §3.

## 6. Artifacts / logging

- `handover.json`: per entry — label, nth, bar, seconds, `at` (verbatim, when the entry had one), cut frame, take (path or
  `take_<k>`), style, x, offset, z, confident, reached, frames contributed.
- `semantic.npy` = the final S. `request.json` verbatim as always — the
  `handover` block and `base_take` included, so the song reproduces from its own
  artifacts directory; this is the one request.json the reference's
  `SongRequest(**request.json)` cannot load. `plan.json` gains the resolved
  block.
- One log line per take and per leg (cut, offset, z, tok/s).

## 7. Tests

- Table test (no model) for §4 on synthetic streams: planted lag recovered,
  ties, short streams, low-z fallback (offset 0 + `measured`); and for §2
  validation errors, the `at` forms included (number, the three string forms,
  rounding, and every malformed one: negative, "1:75", "1:2:3", "", "abc",
  `at` beside `frame` or `section`, `nth` beside `at`).
- Model test (Arc): a one-entry and a four-entry request reproduce, bit for bit,
  the streams the reference driver script produces from the same takes with the
  same offsets (the coordinator supplies takes + reference streams outside the
  repo); `"offset": "auto"` reproduces the reference's measured offsets.
- A request without `handover`: `semantic.npy` unchanged vs the current build, on
  one guided and one plain request.

## 8. Docs

README: a section after `sections` — what it is, the request block, the two
settings (5 s blend / 1 s cut), "audition the takes first", pointer to
`docs/COOKBOOK_STYLE_CHANGE.md` (written by the coordinator). `docs/examples/`: one
request. No private paths, no song titles.
