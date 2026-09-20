# STATUS — stage 8: a style per section (`sections`)

Contract: [SPEC_SECTIONS.md](../SPEC_SECTIONS.md), built on stages 7 / 7b / 7c
(`src/STATUS_GUIDANCE.md`). Build `build_sections/` (`cmake -B build_sections
-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON`, `nice -n 10
cmake --build build_sections -j8`). Every number below was measured on **Vulkan
device 1 (Intel Arc Pro B70)** with `yue2-ar-q8_0.gguf`; device 0 was never
touched, no CPU model was run, and no other build directory was rebuilt. The
"vs HEAD" baseline is `build_trace/yue2`, which is HEAD's code.

## 0. What it is

```json
"sections": [
  { "section": "verse", "nth": 2, "style": "…tags from here on…" },
  { "section": "chorus", "nth": 2, "style": "…", "lead_frames": 35,
    "against": { "previous": [[0, 4], [250, 2]], "blank": [[0, 2]] } }
]
```

An entry names one of the score's own section labels instead of a semantic
frame. The engine keeps a **score clock** — bars and elapsed seconds of the
`V: Vocal` voice, from the score's `M:`/`Q:` and any inline `[M:…]` — so the
label's bar becomes `round(bar start x 25) - lead_frames`, and the entries
compile into stage 7's guidance entries. Nothing about the blend, the branches
or the sampler changed.

- Without an `against` an entry is a **plain swap**: the new prefix is prefilled,
  the old sequence is dropped at once, no branch is ever live. Such a song owns
  **one** KV stream (`n_seq_max = parallel`, `n_ctx = n_ctx_want`), so it decodes
  beside other songs at `--parallel > 1` — which stage 7's rule forbids for
  `guidance` and still forbids for a `sections` entry that carries curves.
- When the engine writes the score, a completed label line re-prefills the slot
  as `prefix(entry style) + score so far` and sampling goes on; the semantic
  phase then starts from a cleared slot and the request's own prefix, the way a
  template hands over. **Do not expect much of the score phase**: see §3.
- New code is all in `src/stage_ar.cpp` unless said otherwise: `SectionEntry`,
  `sections_plain_swap` / `sections_plan`, the `ScoreClock` block
  (`bar_seconds`, `read_tempo_field`, `body_line_clock`, `clock_line`,
  `sections_locate`, `sections_frames`, `sections_keep_check` — all pure and
  table-tested), `parse_sections` and the §2 rules in `validate_request`,
  `json_sections` + the `sections.json` artifact, `sections_compile`,
  `Runner::{sections_token, sections_cut}`, `JobState::{sections, plain_swap,
  sec_seconds}`, `Seq::{clock, line_ids, line_pre, sec_next, sec_cuts}`, one
  `js.plain_swap` branch in `take_entry`, and `GenStats::{section_cuts,
  section_prefill_seconds}` (`src/stage_ar.hpp`, both copies of `json_timing`).
  `parse_against` / `json_against` are the stage-7 `against` code factored out so
  both request forms share one definition.

## 1. §6 — the survey, first and without a GPU

Counted before any of this stage's own runs existed: the **278** `score.abc`
files then under `tests/out/**` (24 truncated, 254 complete, each beside its own
`request.json`), and for the form questions all **320** `*.abc` files under
`tests/out/` + `docs/examples/`. `../songs/` was not touched.

| question | answer |
|---|---|
| label form | always a line of its own, `"% "` + a bare name, no trailing space or text, column 0. **1610** model-written ones in the 278 scores. |
| vocabulary | `intro` 258, `verse` 579, `chorus` 419, `interlude` 274, `outro` 77, `bridge` 3. **Closed, six names.** (`% dense` and `% not music` also occur, three times between them, all in hand-written stage-7 test templates — never in model output.) |
| first label | `% intro` in **all** 258 scores that have one. |
| what follows a label | `V: Vocal` 1609 times, `V: Ins` twice. |
| `M:` | over the 320 `*.abc`: `4/4` 170, `2/4` 108, `3/4` 80, `5/8` 2, `6/8` 1 |
| `Q:` | **always** `Q:1/4=N`, N in 70…128 |
| inline `[M:…]` / `[Q:…]` | **never seen** (implemented and table-tested anyway, per §4) |
| voices | always exactly `V: Vocal` and `V: Ins` |

**The lyric → label mapping is not reliable, and that is worth saying loudly.**

- `[Pre-chorus]` is in the lyrics of **150** of these 278 requests and the
  planner **never** writes `% pre-chorus`. It has no score label at all.
- The planner inserts `% intro` and `% interlude` of its own accord, and it
  repeats or drops sections freely. Over the *complete* scores, the number of
  `% verse` labels differs from the number of `[Verse]` tags in 162 of 254.
- The label sequence is not a function of the lyrics. One ten-tag lyric
  (`intro verse pre-chorus verse chorus bridge verse chorus verse outro`)
  produced **13 distinct** label sequences over complete runs, e.g.
  `intro verse chorus interlude verse chorus outro` and
  `intro verse chorus verse chorus verse chorus verse`. A one-tag lyric
  (`[Verse]` alone) produced a nine-label score.

So "the kth `[Verse]` becomes the kth `% verse`" is **false**. It does not put
`section` + `nth` on sand, because SPEC §2 defines `nth` over the **score's**
labels, not the lyrics' — and that anchor is exact and deterministic:

- with `"abc"` given (the restyle workflow, and the path that matters), the
  score is fixed and known, so `% verse 2` is unambiguous before the run starts;
- when the engine writes the score, `nth` is still exact within the run, and
  per-name counting is exactly what makes it robust to the `% interlude` the
  planner slips between two verses — but *which lyric section* it lands on
  cannot be predicted from the request. `sections.json` reports what it resolved
  to (label, line, bar, seconds, frame, `reached`), and an entry whose label was
  never written is reported rather than an error.

No anchor change proposed. The honest usage rule, which is what README now says:
**name the section against a score you have, not against your lyrics.**

## 2. The test requests

Public, made for this stage, in the gitignored `tests/out/sections/`. They are
stage 7's public `short.json` (`cot=full`, the score is written, seed 7) and
`ext.json` (`short.json` + that score as `"abc"`, 609 abc ids) plus one block.
`SKA` below is `"uptempo ska, offbeat guitar, horn section, bright tenor vocal"`.

| file | block added |
|---|---|
| `sec_short.json` | `short.json` + `"sections": [{"section": "verse", "nth": 1, "style": SKA}]` |
| `sec_short_same.json` | the same entry whose `style` **is** the request's own |
| `sec_ext.json` | `ext.json` + the same entry (the plain swap) |
| `sec_ext_lead0.json` | the same at `"lead_frames": 0` |
| `sec_ext_intro.json` | the entry on `% intro` — bar 0, so the frame clamps to 1 |
| `sec_ext_against.json` | the same entry + `"against": {"previous": [[0,11],[60,11],[85,3]], "blank": [[0,2]]}` |
| `sec_ext_two.json` | two entries, the second on `% outro`, which this score never writes |
| `keep_only.json` | `ext.json` + `"semantic_keep": {"file": "ext_R/semantic.npy", "frames": 64}` |
| `sec_keep.json` | `keep_only.json` + the verse entry (resolves to frame 358 > 64) |
| `sec_keep_atN.json` | keep **358** + the verse entry, i.e. the cut exactly at frame N |
| `sec_keep_atN_ag.json` | the same with `previous [[0,5]]`, `blank [[0,2]]` |
| `sec_keep_far.json` | keep 64 + an entry on `% interlude`, frame 1057, never reached |
| `sec_keep_bad.json` | keep **400** + the verse entry at 358 → request error |
| `err/*.json` | one per §2 error |
| `jobs_par*.json`, `jobs_seq.json`, `jobs2.json` | `--requests` batches |

`ext_R` = `ext.json --max-semantic 600` (the no-swap baseline every semantic
comparison below is against).

The score `short.json`/`ext.json` carry is `M:3/4`, `Q:1/4=103` (1.74757 s a
bar), and its first `% verse` is **line 22 after 9 Vocal bars = 15.7282 s**, so
the entry resolves to `round(15.7282 x 25) - 35 = 393 - 35 = `**358**. Checked by
hand against `score.abc`: lines 11, 15 and 19 are the intro's Vocal bodies with
4, 4 and 1 bars.

## 3. §7 — acceptance

| # | check | result |
|---|---|---|
| 1 | no `sections`: `cmp` vs HEAD for unguided / guided / keep | **PASS** — 5 requests x 3 files identical, plus a `--parallel 2` batch |
| 2 | score phase: one entry, one re-prefill, `sections.json` right, deterministic | **PASS** |
| 3 | an entry whose style is the request's own | **informative — not identical**, diverges at abc token 251 |
| 4 | frame arithmetic table tests in `yue2-guidance` | **PASS** — 133 cases (91 before) |
| 5 | semantic phase: identical to the no-swap run up to the frame, different after; no branch prefill | **PASS** — first diff exactly at 358, `cfg_branches` 1 |
| 6 | `"abc"` + sections; `semantic_keep` + sections | **PASS**, including the cut at `frame == N` |
| 7 | `--verify-sampler` on a sections run | **PASS** — 600 steps, zero disagreements |
| 8 | cost | **PASS** — see §3.8 |
| — | plain swap at `--parallel 2`, in either slot | **PASS** — one stream per song, neighbours untouched |
| — | a sections job with curves at `--parallel 2` | **rejected**, as stage 7's rule says |
| — | a guided sections job then an unguided one at `--parallel 1` | **PASS** — job 2 bit-identical to running it alone |
| — | `yue2 song` end to end on a sections request | **PASS** — 1432 frames, 57.28 s of FLAC |
| — | every §2 request error at the CLI | **PASS** — 9 checked, all before the model loads |
| — | `--guidance-trace` on a plain swap | **PASS** — 0 rows, one log line, same song |

### 1 — the unguided, guided and keep paths are untouched

```bash
for r in short ext ext_g64 ext_cfg3; do for b in build_trace build_sections; do
  ./$b/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/$r.json \
      --artifacts tests/out/sections/${b}_$r --gpu 1 --max-semantic 300
done; done
# and tests/out/keep/keep_n64.json the same way
```

`prefix.npy`, `abc_tokens.npy` and `semantic.npy` are byte-identical on all
five: `short` (score written), `ext` (external score), `ext_g64` (a stage-7
cut), `ext_cfg3` (`cfg_scale: 3`), `keep_n64`. A two-job `--parallel 2` batch of
`short`/`short2` is identical too, and both binaries report the same step
counts (`1077 full-width, 0 holed, 1081 decode calls`, 7.53 vs 7.52 ms/step).
`yue2-guidance` PASS (133), `yue2-bars` PASS (71).

### 2 — the score phase

```bash
./build_sections/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/sections/sec_short.json \
    --artifacts tests/out/sections/out_sec_short --gpu 1 --max-semantic 60
```

```
abc: line 22: % verse 1 (bar 9, 15.73 s): the score goes on under its own tags (81 + 241 tokens, 0.08 s)
```

**One** re-prefill. `sections.json`:

```
{"section": "verse", "nth": 1, "style": "uptempo ska, …", "lead_frames": 35,
 "reached": true, "line": 22, "bar": 9, "seconds": 15.728155339805824, "frame": 358}
```

which is the hand check of §2 exactly — line 22 of `score.abc` is `% verse`, and
the nine Vocal bars above it are 15.7282 s at `M:3/4 Q:1/4=103`.

| check | result |
|---|---|
| `abc_tokens.npy` vs the no-sections run | identical for 247 tokens, then different |
| where 247 falls | the two `score.abc` texts share **23 complete lines** — through `% verse` (22) and `V: Vocal` (23) — and part company inside line 24 |
| `prefix.npy` | the request's own 81-token head + the new score + 2, as SPEC §3 asks |
| `timing.section_cuts` / `section_prefill_seconds` | 1 / 0.0775 s |
| `request.json` keys | `abc cfg_scale cot id lyrics seed style` — no `sections` |
| `plan_manifest.json` | + `sections.json` |
| the same command twice | `abc_tokens.npy`, `semantic.npy`, `score.abc`, `sections.json` **identical** |

The SPEC asks for "identical up to and including the label line, different
after". It is in fact identical four tokens *past* the label line (the
`V: Vocal\n` under it survives the cut), which is the stronger half of both
halves.

**What the score phase is worth.** The coordinator measured this independently
while the stage was being built and it agrees with what this run shows: a plain
tag swap late in a score barely bends what is written — 89 golden lines then a
free continuation under very different tags came back with the same register,
rhythm and chords, because the score already written outvotes the tags the same
way the audio history outvotes them in the semantic phase. The same tags on a
*fresh* score give a different key, tempo and meter entirely. So the score phase
here does what SPEC §3 specifies — the rest of the score is conditioned on the
new tags, and the label line itself is not — and the effect shrinks with the
length of score already written. The frame arithmetic is the reason to run it;
the audible change is the semantic swap (§5). SPEC §1's second bullet holds for
short histories only.

### 3 — an entry whose style is the request's own (informative)

`sec_short_same.json`: the cut fires at the same place (`line 22 … 81 + 241
tokens`) and the score is **not** identical to the no-sections run — it diverges
at abc token 251, four tokens past the cut, and then runs 778 ids instead of
609. So prefill-vs-decode numerics flip a near-tie there, exactly as
`SPEC_KEEP` §5.3 measured for the keep prefill (`STATUS_GUIDANCE` §3b: a 0.003 nat
top-2 gap is enough). Reported, not forced to pass — the SPEC calls it
informative.

### 4 — the table tests

```bash
./build_sections/yue2-guidance      # PASS: 133 cases, 0 failures
```

42 new cases: 12 `sections_locate` + `sections_frames` clock cases (4/4 at
`Q:1/4=120` = **50 frames a bar**, 3/4 = 37.5, 6/8 counted in the quarters `Q:`
names, `Q:1/4=90` = 66.67 with the frame rounded once, an inline `[M:3/4]`
mid-line, a `Z4` multi-measure rest, the lead clamp to frame 1, a label the
score never writes, an `nth` the score never reaches, `%%yue2-gen` not being a
label, a score with no `V:` line at all), 4 multi-entry cases (per-name `nth`
stepping over the planner's `% interlude`, the strictly-increasing clamp, an
entry the score orders the other way round going unfound, and the plan a plain
swap compiles to) and 26 request cases covering every rule and error of §2.

### 5 — the semantic phase

```bash
./build_sections/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/sections/sec_ext.json \
    --artifacts tests/out/sections/out_sec_ext --gpu 1 --max-semantic 600
```

```
context: 1300 tokens/slot x 1 slot, batch 692, KV ~142 MiB/slot = 0.14 GiB
guidance: step 358: new positive prefilled into slot 0 (692 + 357 tokens)
```

| check | result |
|---|---|
| `semantic.npy` vs `ext_R` (same seed, no swap) | identical over frames 0…357, **first difference at 358** |
| how much differs | 234 of the 242 frames after the cut |
| `prefix.npy`, `abc_tokens.npy` | identical to `ext_R` |
| branch prefill in the log | **none** — one line, the swap itself |
| `cfg_branches` / `guided_steps` | **1 / 0** |
| KV streams | **one slot**, `n_ctx` not multiplied (SPEC §4's "allocates nothing it does not use") |
| twice | `semantic.npy`, `sections.json`, `guidance.json` identical |

`sec_ext_intro.json` (the entry on `% intro`, bar 0) resolves to frame **1** and
cuts there: `guidance: step 1: new positive prefilled into slot 0 (692 + 0
tokens)` — the smallest cut there is.

With curves (`sec_ext_against.json`) it is stage 7's machinery unchanged:

```
context: 1100 tokens/slot x 3 slots (1 song + up to 2 guidance branches)
guidance: step 358: slot 0 relabelled as the previous branch
guidance: step 358: new positive prefilled into slot 1 (692 + 357 tokens)
guidance: step 358: blank prefilled into slot 2 (633 + 357 tokens)
```

`cfg_branches` 3, `guided_steps` 242 = 600 − 358, `branch_prefill_seconds`
0.442, first differing frame 358.

`sec_ext_two.json` (verse 1 + an `% outro` this score never writes):
`sections.json` is `[verse 1 reached frame 358, outro 1 reached:false frame
null]`, `guidance.json` holds the one compiled entry, and the run is otherwise
the plain swap.

### 6 — `"abc"` and `semantic_keep`

`sec_keep.json` (keep 64, cut at 358) against `keep_only.json`:

```
keep: 64 frames from semantic.npy prefilled (692 + 64 tokens, 0.18 s)
guidance: step 358: new positive prefilled into slot 0 (692 + 357 tokens)
```

first differing frame **358**, kept 64 equal to the file, one slot.

`sec_keep_atN.json` is the workflow that matters — keep the take up to the
section, change the style from there — with the cut at exactly `frame == N`:

```
keep: 358 frames from semantic.npy prefilled (692 + 358 tokens, 0.24 s)
guidance: step 358: new positive prefilled into slot 0 (692 + 357 tokens)
```

`semantic.npy[:358]` is the file bit for bit and frame 358 is the first that
differs. `sec_keep_atN_ag.json` does the same through the three-slot path
(relabel + new positive + blank, all at step 358), with the kept stretch equally
exact. `sec_keep_far.json` (frame 1057 under a 600-step cap) reports
`"reached": false` in `guidance.json`, and `sec_keep_bad.json` is a request
error before the model loads:

```
error: …/sec_keep_bad.json: "sections" entry 1: % verse 1 lands on frame 358, inside the kept 400 frames
```

### 7 — `--verify-sampler`

```
guidance: step 358: new positive prefilled into slot 0 (692 + 357 tokens)
ar verify: 600 sampling steps matched the stage-5 sampler exactly
```

and `semantic.npy` is identical to the same run without the flag.

### `--parallel` — the plain swap decodes beside other songs

```bash
./build_sections/yue2 ar -m yue2-ar-q8_0.gguf --gpu 1 --parallel 2 \
    --requests tests/out/sections/jobs_par.json --max-semantic 600
```

| batch | result |
|---|---|
| `sec_ext` (slot 0) + `ext` (slot 1) | accepted, **2 slots**, `[1/2] guidance: step 358: new positive prefilled into slot 0`, 599 full-width steps, 0 holed |
| the same reversed, sections in slot 1 | `[2/2] … prefilled into slot 1` — its **own** slot, never the neighbour's |
| both jobs' `prefix.npy` / `abc_tokens.npy` vs running alone | identical |
| both jobs' `semantic.npy` vs running alone | differ — the *unguided* job differs too, so it is the known two-row batch-width numerics of `SPEC_GUIDANCE` §6, not the swap |
| `sec_ext_against` + `ext` at `--parallel 2` | `error: …: guidance needs --parallel 1 (this batch decodes 2 songs side by side)` |
| `sec_ext_against` then `ext` at `--parallel 1` | 3 slots; the second job's three files **identical** to running it alone |

That last row is the one the cold review was after: the primary moves between
streams at a cut, and the slots are still clean for the job behind it.

### `yue2 song` end to end

```bash
./build_sections/yue2 song --ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf \
    --vae yue2-vae-f32.gguf --request tests/out/sections/sec_ext.json \
    --artifacts tests/out/sections/song_sec --out tests/out/sections/song_sec.flac --gpu 1
```

`abc 0.0 s, semantic 10.8 s, nar 24.2 s, vae 5.4 s`, 1432 frames, 57.28 s of
audio, `sections.json` and `guidance.json` beside the rest.

### The §2 errors at the CLI

All before the model loads, all through `strf`:

```
"sections" and "guidance" are two ways to ask for the same machinery: … not both
"sections" and "cfg_scale" are two ways to ask for the same machinery: put the push under an entry's "against", not in "cfg_scale"
"sections" with "abc_template" is not supported yet: a template re-prefills the slot per hole, and a cut re-prefills it per section
"sections" names labels in a score, and cot=off has none — use cot=melody or cot=full
"sections" entry 2: % verse 2 cannot come after entry 1's % verse 4 — entries are in the order the song plays them
"sections": entry 1 needs a "style": the tags from this section on
"sections": entry 1: unknown key "frame" (section, nth, style, lead_frames, against)
"sections": entry 1: "lead_frames" must be an integer in [0, 250] (25 frames = 1 s)
"sections" entry 1: "style" is the one already in force, so "against"."previous" would push against itself
```

### 3.8 — what it costs

Arc, `--max-semantic 1000` so all three decode exactly 1000 semantic steps:

| case | wall | semantic | tok/s |
|---|---:|---:|---:|
| unguided (`ext.json`) | 8.38 s | 7.26 s | 137.8 |
| plain swap (`sec_ext.json`) | **8.69 s** | 7.53 s | 132.8 |
| one entry with both curves (`sec_ext_against.json`) | 16.10 s | 14.94 s | 66.9 |

So a plain-swap song costs **+0.31 s over the whole song** — one prefix prefill
and nothing else, because no branch is ever decoded. The curves cost what stage
7 measured (§9 there), confined to the window.

Re-prefill per entry:

| prefill | tokens | seconds |
|---|---|---:|
| score phase, one cut | 81 + 241 | **0.078** |
| semantic phase, plain swap | 692 + 357 | **0.230** |
| semantic phase, relabel + new positive + blank | 692+357, 633+357 | **0.442** total |
| keep of 358 frames beside it | 692 + 358 | 0.24 |

A score-phase cut also adds one whole `prefix_sem` prefill at the abc→semantic
hand-over instead of the two-token bridge (deviation **S3**); on `sec_short.json`
that is inside the 0.0775 s the run reports.

## 4. Deviations

| # | what | why |
|---|---|---|
| S1 | SPEC §4 asked whether a plain swap can avoid `--parallel 1` and what `n_seq_max` it needs. It needs **1**, not 2, and it is **not** forced to `--parallel 1`. | `take_entry`'s no-`previous` branch already clears the outgoing slot before prefilling the replacement, so the replacement goes into the slot just freed. One `js.plain_swap` test in `take_entry` makes it *that* slot rather than `free_slot`'s "lowest stream this song is not using", which at `--parallel > 1` would have handed out a neighbour's (measured: the sections job in slot 1 re-prefills slot 1). `is_guided()` returns false for such a request, so the `--parallel 1` rule does not apply, and the `guided` test that triples `n_ctx` skips it. A `sections` entry with an `against` is unchanged from stage 7: three streams, `--parallel 1`. |
| S2 | `sections` with `cot: "off"` is a request error, which §2 does not list. | `cot=off` has no score phase and no score, so no label can ever resolve: every entry would report `reached: false` and the request would silently do nothing. §2's spirit ("unknown keys are errors") says reject it. |
| S3 | A score phase that made at least one cut clears the slot and prefills the **whole** `prefix_sem` at the abc→semantic hand-over, instead of §4.4's two-token bridge. | After a cut the slot holds *that entry's* prefix plus the score, and SPEC §3 requires the frame-0 semantic prefix to be the request's own `style`. This is the hand-over `template_done` already makes, same code shape. A sections run that made **no** cut keeps the bridge, so it stays on HEAD's path. |
| S4 | `guidance.json` for a sections request holds only the entries whose label was **found**; `sections.json` holds all of them with `reached`. Indices therefore do not line up between the two files. | `guidance_step` walks the plan by frame, so an unresolved entry in the middle would stop every entry behind it. `sections_compile` compacts the plan (and `js.heads` with it) to the found entries, which is the only well-defined plan. `sections.json` is the file that says what happened to each request entry. |
| S5 | `reached` means two different things in the two files: "the label turned up in the written score" in `sections.json`, "the decode got to that frame" in `guidance.json`. | Both are what their own SPEC asks for (§2 here, `SPEC_GUIDANCE` §2.3 there). `sec_keep_far.json` shows a `sections.json` `reached: true` with a `guidance.json` `reached: false` — found in the score, never reached by a 600-step run. |
| S6 | `plan.json` carries a `"sections"` block too, which §5 does not ask for. | Mirrors `"guidance"`, which is already there, and keeps `plan.json` the one file that describes the whole run. `request.json` stays free of it (§5). |
| S7 | `section_cuts` / `section_prefill_seconds` are on **both** phases' timing blocks, the semantic phase included, which never cuts a score. | Same reason as **G5**/**K6**: they are song-level numbers and the only `json_timing()` that reaches an artifact is the abc one. It also means two new keys appear in every job's `plan.json`, sections or not. |
| S8 | `--dump-logits` on a sections request dumps the unguided row and says nothing about the sections. | The flag dumps the *first* step, and the first entry is at frame ≥ 1 by the clamp, so the unguided row **is** the right one. Unlike **G3**/**K2** there is nothing to reject. |
| S9 | The bar clock follows the voice literally named `Vocal`; a score whose voices are named otherwise counts no bars (and every entry clamps to frame 1). A score with no `V:` line at all counts all its body lines. | All 278 surveyed scores name exactly `V: Vocal` and `V: Ins` (§1), and SPEC §3 says "`V: Vocal` body lines". Making it "the first voice seen" would silently double-count a two-voice score whose voices are named `1` and `2`. Listed as a follow-up in ROADMAP item 6. |
| S10 | An inline `[M:…]` that does **not** sit at a bar line takes effect at the next bar line, not where it stands. | Otherwise cutting the line there would close a partial bar and count it twice. No surveyed score has an inline `[M:…]` at all, so this is a choice about malformed input, and the at-bar-line case is the one table-tested. |
| S11 | Line numbers in `sections.json` are counted over the score text the clock walked: `req.abc` for a given score, the detokenized output for a written one. | They agree for ASCII ABC (checked: the given-score run reports line 22, which is line 22 of the written `score.abc`), but `score.abc` for an external score is `detokenize(tokenize(req.abc))` and a pathological round trip could differ. |
| S12 | A label line completed by the **very last** abc token of a truncated score phase is not seen. | `apply()` only walks the clock on the path that keeps sampling; the token that hits `max_tokens` falls through to the hand-over. The entry then reports `reached: false`, which is the right answer for a score that ended on that line anyway. |
| S13 | `--guidance-trace` on a plain-swap sections song writes a `guidance_trace.npy` of **0 rows**. | Correct rather than special-cased: the trace has one row per step with a live branch, and a plain swap never has one. The run logs `guidance trace: 0 rows x 8`. |

## 5. Not done / open

- **No guidance in the score phase.** An entry's `against` curves are carried
  through to the compiled guidance entry and used only in the semantic phase.
  Nothing in the design paints that into a corner — the curves live on the
  `SectionEntry` and survive `sections_plan` — but the abc phase never decodes
  beside a branch today, and the coordinator's measurement says a plain swap late
  in a long score is close to a null anyway. ROADMAP item 6.
- `sections` with `"abc_template"` is rejected, as `guidance` is.
- The `% pre-chorus` finding of §1 is a documentation problem, not a code one:
  there is no score label to name. A wrapper that wants "the second pre-chorus"
  has to find the bar itself and use `guidance`.
- `tests/regress.sh` was not extended, for the reason stages 7 and 7b give.
