# STATUS — stage 9: hand the song over to another style's take (`handover`)

Contract: [SPEC_HANDOVER.md](../SPEC_HANDOVER.md), built on stage 7b
(`semantic_keep`) and stage 8 (the score clock). Build `build_handover/`
(`cmake -B build_handover -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
-DYUE2_BUILD_TESTS=ON`, `nice -n 10 cmake --build build_handover -j8`). Every
number below was measured on **Vulkan device 1 (Intel Arc Pro B70)** with
`yue2-ar-q8_0.gguf`; device 0 was never touched, no CPU model was run, and no
other build directory was rebuilt. The "vs HEAD" baseline is `build/yue2`.

The reference material — one base take and four takes of the same score, plus
the two semantic streams a reference driver script produced from them — is
coordinator-supplied and lives outside the repo. It is called **take A–D** and
**the 4-leg / 1-leg reference stream** below.

## 0. What it is

```json
"base_take": "out/base",
"handover": [
  { "section": "chorus", "nth": 1, "style": "…tags…" },
  { "section": "verse",  "nth": 3, "take": "takes/reggae", "seconds": 1 },
  { "section": "verse",  "nth": 4 }
]
```

One score, one full take per style. At each cut the incoming style's renderer is
run over a forced history — its own take up to `x` frames before the cut, then
the last `x` frames of the song so far — and what it samples from the cut on is
spliced in. The offset the incoming take runs ahead by is measured from the
tokens (`"offset": "auto"`).

- New code is all in `src/stage_ar.cpp`: `HandoverEntry`, `HandoverTake`, a pure
  table-tested block (`OffsetFit` / `offset_hits` / `offset_scan` /
  `handover_offset`, `handover_x`, `handover_keep`, `handover_locate`,
  `handover_cuts_check`, `handover_want`, `handover_preflight`),
  `parse_handover` + the §2 rules in `validate_request`, `json_handover` + the
  `handover.json` artifact and the `plan.json` block, `request_relative` and
  `semantic_cap` (both shared with `semantic_keep`), `load_take` /
  `takes_score_check` / `handover_render` / `handover_cap`, the driver
  `run_handover`, and `JobState::no_files`.
- **One refactor**, which is what lets the model be loaded once: everything
  `run_ar_batch` did after validation is now `ar_decode_jobs(p, jobs, states,
  results, model, vocab, card, parallel, rejected)`, which neither loads nor
  frees the model. `run_ar_batch` validates, loads, calls either the driver or
  `ar_decode_jobs`, and frees. No behaviour change: §1.5 is the `cmp`.
- **Order inside the driver** (what the cold review's item 1 is about): read
  every file the request names → resolve which take takes over where → check
  everything that is arithmetic → render the takes → check again now that every
  take is in memory → play the legs. Only a request whose score the *engine*
  writes needs one render before the labels and the takes' scores can be checked.

## 1. §7 — acceptance

`REQ/` in the commands below is the gitignored `tests/out/handover/`, where a
small generator built the model-test requests from the reference takes.

| # | check | result |
|---|---|---|
| 1 | the four-leg run reproduces the 4-leg reference stream | **PASS** — 6425 frames, bit-identical |
| 2 | the one-leg `"offset": "auto"` run reproduces the 1-leg reference stream | **PASS** — 6391 frames, bit-identical, offset **+34**, z **7.7**, 13 frames agreeing |
| 3 | a `section` + `nth` resolves to the reference cut | **PASS** — frame **3415**, same stream as the frame form |
| 4 | a take the engine renders is what `yue2 ar` renders on its own | **PASS** — bit-identical to **HEAD's** `build/yue2` for the same request |
| 4b | that take vs the reference take of that style | **differs — expected**: that take's `config.json` says `vulkan:0` (§1.4) |
| 5 | no `handover`: `cmp` vs HEAD, one plain and one `sections` request | **PASS** — 2 requests × 3 files identical; also a 2-job `--parallel 2` batch |
| 6 | table tests, no model | **PASS** — `yue2-handover` **79 cases**; `yue2-guidance` 133 and `yue2-bars` 71 still pass |
| 7 | determinism: the same request twice | **PASS** — identical `semantic.npy` |
| 8 | `yue2 song` end to end on a handover request | **PASS** — 6371 frames, 295.3 s, FLAC written |
| 9 | a `take` of a different score, and a `base_take` that is not the `"abc"` | **PASS** — both refused, message names both files |
| 10 | a cut no run could reach (`--max-semantic` under its forced frames) | **PASS** — `rc 1` before any take is rendered, not a short song |
| 11 | a `handover` in a two-song `--requests` batch | **PASS** — rejected before anything loads (**H9**) |
| 12 | **a take too short for its cut** | **PASS** — `rc 1`, refused **before any take is rendered**, nothing written |
| 13 | the artifacts' own `request.json` re-renders the song | **PASS** — identical `semantic.npy` |
| 14 | an entry whose `take` *is* the `base_take` | **PASS** — one take, shared |
| 15 | a `base_take` rendered under other tags | **PASS** — one warning naming both strings |
| 16 | `--prefix-only` on a handover request | **PASS** — writes the base request's `prefix.npy`, identical to the full run's |

### 1 and 2 — the reference streams

```bash
./build_handover/yue2 ar -m yue2-ar-q8_0.gguf --request REQ/cycle4.json \
    --artifacts tests/out/handover/v2_cycle4 --gpu 1 --seed 1
```

`cycle4.json` is the reference base take in `base_take` and four entries giving
the reference legs' cut frames and offsets (`2315:+12, 3415:+13, 4215:+10,
5015:+0`, the last with neither `style` nor `take`, i.e. back to the base take).
The resulting `semantic.npy` is **byte for byte** the reference stream, so the
leg construction, the `semantic_keep` hand-over, the per-leg cap and the splice
all match the script the feature was designed from.

The one-leg run measures its own offset:

```
handover: leg 1: frame 3415 <- take D, offset +34 (z 7.7, 13 frames agree),
          x 25, 2976 new frames in 35.9 s = 82.9 frames/s, song 6391 frames
```

`+34` and `z ≈ 7.7` are the reference's numbers, and the stream is identical to
the reference's. The other takes measured at the same cut against the base take
come out at **+11**, **+13** and **+15**, as measured independently in numpy.

### 3 — a label instead of a frame

The request names the section and `nth` instead of the frame; the score is
`M:4/4 Q:1/4=120` (a 2 s bar, 50 frames), and the label resolves — line, bar and
bar-start seconds all recorded in `handover.json` — to frame **3415**, the
reference cut. Its `semantic.npy` is identical to the frame-form run's.

### 4 — a take the engine renders itself

With the incoming tags given as an entry `style`, the engine renders `take_1` and
then the leg. `take_1/{semantic,prefix,abc_tokens}.npy` and `score.abc` are
**identical to HEAD's binary** rendering the same request (those tags, `abc` =
the base score, same seed) on its own:

```bash
./build/yue2 ar -m yue2-ar-q8_0.gguf --request REQ/take_euro_standalone.json \
    --artifacts tests/out/handover/out_take_euro_head --gpu 1 --seed 1
cmp tests/out/handover/out_take_euro_head/semantic.npy \
    tests/out/handover/out_style_euro/take_1/semantic.npy   # identical
```

That is SPEC §3's actual test. It is **not** identical to the reference take of
that style (6388 vs 6408 frames, first difference at frame 17): that take's
`config.json` reads `"device": "vulkan:0"`, so it was rendered on the AMD. Same
GGUF, same seed, different card — reported, not chased.

### 5 — the requests without a handover are untouched

```bash
for r in plain sections; do for b in build build_handover; do
  ./$b/yue2 ar -m yue2-ar-q8_0.gguf --request REQ/$r.json \
      --artifacts tests/out/handover/reg_${b}_$r --gpu 1 --seed 1 --max-semantic 300
done; done
```

`prefix.npy`, `abc_tokens.npy` and `semantic.npy` identical on both — a plain
given-score request and a `sections` plain swap. The same two as a two-job
`--requests` batch at `--parallel 2` are identical to HEAD's as well (119
full-width steps, 0 holed, both binaries).

### 6 — the table tests

```bash
./build_handover/yue2-handover      # PASS: 79 cases, 0 failures
```

Synthetic streams from an LCG over the 32768 codec indices with a known lag
planted in them: 13 offset cases (six lags recovered including ±100; the 2.5 %
density two real takes share; **30 unrelated stream pairs, none confident**; a
single planted frame — a z of 14 — refused; §4's retry from frame 150; ties by
`|k|` and by sign; a cut before frame 150; a take that ends early; a take of
itself), 8 leg-history cases, 11 cut cases, **7 pre-flight cases** and 35 request
cases covering every rule, error and default of §2.

### 8 — `yue2 song`

```bash
./build_handover/yue2 song --ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf \
    --vae yue2-vae-f32.gguf --request REQ/song_short.json \
    --artifacts tests/out/handover/song_out --out tests/out/handover/song_out.flac --gpu 1 --seed 1
```

`abc 0.0 s, semantic 71.0 s, nar 199.8 s, vae 23.4 s`, 6371 frames, FLAC written.
The artifacts hold `handover.json` beside the usual files, `plan.json` carries
the resolved block, `plan_manifest.json` lists it, and `request.json` is the
request as given — `handover` block and `base_take` included — so the song
reproduces from its own directory (§1.13). `prefix.npy` is byte-identical to the
base take's own.

### 12 — nothing renders before the checks are in

```
handover: base take …: 6425 frames of an earlier render, 2417 abc ids
error: REQ/err_short_take.json: "handover" entry 1: short_take has 3000 frames,
       and the cut at 3415 needs frame 3315 of it even at the largest offset the
       scan can pick
```

`rc 1`, no artifacts directory created, no take rendered. The take is a copy of a
reference take truncated to 3000 frames. The same holds for a cut no run could
reach (§1.10), where the cap rather than the take is what cannot work:

```
error: REQ/late.json: "handover" entry 2: its 5815 forced frames leave nothing
       to sample under the 500-step semantic cap
```

Before the review this entry was *skipped* at runtime and the song ended 25
frames past the previous cut with `rc 0` — the bug item 1 of the review was
about.

## 2. Cost on the Arc

Takes are **sequential**, one generation at a time (**H1**).

| what | forced frames (prefill) | sampled | wall | rate |
|---|---|---|---|---|
| a full take (engine, 6388 frames) | — | 6388 | **66.0 s** | 96.8 frames/s |
| leg at cut 2315, to the next cut | 2303 (1.62 s) | 1125 | 12.8 s | 87.6 |
| leg at cut 3415, to the next cut | 3402 (2.29 s) | 825 | 11.0 s | 74.9 |
| leg at cut 4215, to the next cut | 4205 (2.73 s) | 825 | 11.9 s | 69.4 |
| leg at cut 5015, to the end | 5015 (3.25 s) | 1410 | 19.9 s | 71.0 |
| one leg at cut 3415, to the end | 3381 (2.28 s) | 2976 | 35.9 s | 82.9 |

| run | takes | legs | wall |
|---|---|---|---|
| four legs, every take read from disk | 4 (0 rendered) | 4 | **56.9 s** |
| one leg, take read from disk | 2 (0 rendered) | 1 | 36.0 s |
| one leg, the engine renders the take | 2 (1 rendered) | 1 | 112.6 s |

So the price of the feature is **one full take per new style** (about a song's
sampling time each) plus a leg per cut, and a leg is cheaper than the song it
replaces because its history is forced (a 5015-frame prefill is 3.25 s against
the ~70 s sampling those frames would cost) and it stops 25 frames past the next
cut. Reading takes from disk is free, which is why `base_take` + `take` is the
workflow to use when the takes already exist.

## 3. Deviations

| # | what | why |
|---|---|---|
| H1 | Takes and legs are rendered **sequentially**, one job per context, not batched over the existing `--parallel` path. | §3 requires a take to be *bit-identical* to what `yue2 ar` would produce on its own, and decoding two songs side by side changes the batch width and with it the numerics (SPEC_GUIDANCE §6, STATUS_SECTIONS §"--parallel"). Batching would buy throughput and lose the property the whole stage is tested on. The SPEC allows sequential and asks for the cost: §2. |
| H2 | Every take is rendered **before any leg**, in order of first use, so a take for an entry the song never reaches is rendered anyway. | Keeps the numbering, the log order and the pre-flight checks independent of what the legs do. Rendering lazily would save a take in exactly one case — a cut past the end of the song — which the run reports anyway. |
| H3 | A request whose score the **engine writes** has its base take rendered before the labels resolve and before the takes' scores are compared — those checks cannot come first. Everything else (the files, the arithmetic, every take's score when the request or `base_take` gives the score) is checked before anything is rendered. | There is no score to resolve a label against until one exists. The take *files* are still all read up front, so a missing or malformed directory is refused before the base take is rendered either way. |
| H4 | A tie in §4 goes to the **smallest `\|k\|`**, and at equal `\|k\|` to the negative one. The reference driver's `max(rate, key=rate.get)` takes the most negative `k` of all ties. | Written into SPEC §4 and table-tested both ways round. On real streams the rates are floats over hundreds of frames and no measured case ties. |
| H5 | An entry whose `style` is the request's own style is the **base take**, not a second render of it. An entry whose `style` happens to equal a `take` directory's style is **not** shared with that directory. | §2's "entries with an equal style string share one take" plus "neither: back to the request's own style = the base take". A directory take is not what the engine would render (other card, other seed, possibly another model), so sharing across the two forms would silently substitute one for the other. An entry whose `take` *is* the `base_take` directory does share it (§1.14). |
| H6 | `lead_frames` is accepted on a `section` entry. | §2 says "same resolver, same `lead_frames`, default 35", which reads as the key being there. It is the same 0…250 range as `sections`. |
| H7 | `handover.json` carries three keys §6 does not list: `line` (the label's line in the score, as `sections.json` has it), `hits` (frames agreeing at the chosen lag) and `note` (why an entry was skipped). `z`, `hits` are `null` and `confident` `true` for an offset the request gave. | The line number is what makes a resolved entry checkable by hand; `hits` is half of §4's confidence rule and useless to record only in the log; a skipped entry with no reason is a report nobody can act on. A given offset was not measured, so there is nothing to be unconfident about. |
| H8 | The run's `timing`: the abc block is the base take's (all zeros when `base_take` supplied the score), and the semantic block's `seconds` is **every** take's and leg's decode time with `output_tokens` = the final song's frames. | There is no single generation to report. The sum is what the run cost and the length is what came out; `output_tps` is their ratio, which is the throughput of the whole handover rather than of any one leg. Per-leg numbers are in the log and `handover.json`. |
| H9 | `handover` in a `--requests` / `--jobs` batch of more than one song is a request error. A **one-job** `yue2 batch` is accepted. | §2 says "`yue2 batch`: request error for now", but `yue2 song` reaches `run_ar_batch` as a one-job batch too and cannot be told apart from `yue2 batch --jobs` with one job. Rejecting on `jobs.size() > 1` is the only rule that keeps `yue2 song` working, and a batch of one is a song by another name. README says this rather than the SPEC's shorthand. |
| H10 | An entry whose label the score never writes takes the entries **behind** it with it: all of them report `reached: false`, the ones behind with the note "not resolved: an earlier entry's label was not found". | Stage 8's `sections_locate` matches entries in the order given and stops at the first it cannot find — "same resolver" (§2). Table-tested, and the song is simply the base take from that point on. |
| H11 | `--prefix-only` on a handover request writes the base request's `prefix.npy` and decodes nothing, instead of being an error — `base_take`'s `score.abc` stands in for the `"abc"` it would otherwise need. | That prefix *is* what the handover would have written (§5), so the flag does for a handover exactly what it does for any other request. |
| H12 | `base_take` without `handover` is a request error. | Nothing else in the engine reads it, so accepting it would silently ignore the key. |
| H13 | Legs write no artifacts at all (`JobState::no_files`); takes write `take_<k>/`. A `--max-semantic` also clamps a leg's cap, and is what the pre-flight checks a forced history against when no next cut does. | §5 says the codes are handed over in memory; §3 asks for the takes on disk. A `--max-semantic` the user typed has to mean something on the run they typed it on. |
| H14 | `plan.json` carries a `"handover"` block, which §6 asks only of `handover.json`; `request.json` carries the `handover` and `base_take` keys as given. | The first mirrors `"guidance"` and `"sections"`. The second is §6's "verbatim", and it makes the artifacts directory reproduce its own song (§1.13) — at the price of being the one `request.json` the reference's `SongRequest(**request.json)` cannot load, which is now written into SPEC §6. |
| H15 | `"frame"` must be ≥ 1, not ≥ 0. | A cut at frame 0 has no history to intrude into: §5's `c − x − off ≥ 1` could never hold. |

## 4. Not done / open

- **No batching.** H1. If a future stage wants the takes decoded side by side, the
  bit-identity test of §3 has to be relaxed to "identical at the same batch
  width" first — that is a SPEC decision, not a code one.
- **The offset is measured once per cut**, against the song so far. §1 of the
  SPEC notes the lead *drifts* over a song; a long leg therefore ends further out
  of step than it started. Nothing corrects for that, and the cut is the only
  place it matters.
- The pre-flight of §5 uses the offset that would suit an `"auto"` entry best, so
  it refuses only certain failures. What is left — an entry the measured offset
  turns out not to fit — is a hard error naming that entry and the one whose leg
  was already cut short for it, never a short song. That backstop is not covered
  by a model test: producing it needs a measured offset that contradicts the
  pre-flight, which no real pair of takes has done.
- The one **reported** non-error left is a song that genuinely ends before a cut
  — the model emitted its end token early. A `--max-semantic` too small for an
  entry no longer produces it (§1.10): that is now refused up front, because a
  cap the user typed cannot be allowed to shorten the song in silence. No model
  test forces the genuine case.
- A `take` directory is trusted for its `semantic.npy`, `score.abc` and the
  `style` in its `request.json`; nothing checks that its codes were sung to that
  score (the score comparison is what stands in for it).
- `tests/regress.sh` was not extended, for the reason stages 7 and 8 give: it
  runs the goldens, and this stage's model tests need takes that live outside
  the repo.
