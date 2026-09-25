# STATUS — stage 11: `negative_style` (a negative prompt for `cfg_scale`)

Contract: [SPEC_NEGATIVE.md](../SPEC_NEGATIVE.md), built on stage 7
(`STATUS_GUIDANCE.md`). Build `build_negative/`
(`cmake -B build_negative -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON -DYUE2_OPUS=ON`,
`nice -n 10 cmake --build build_negative -j8`; `YUE2_WARN_FLAGS` clean — the only
warnings are the pre-existing `-Wshadow` ones from ggml's own `ggml-backend.h`).
Every model run below is **Vulkan device 1 (Intel Arc Pro B70)** with
`yue2-ar-q8_0.gguf`; device 0 was never touched, and no other build directory was
rebuilt. The baseline for item 1 is a copy of the committed `build/yue2` binary.

## 0. What it is

```json
{ "style": "…", "lyrics": "…", "cfg_scale": 3, "negative_style": "children's song" }
```

The blank branch of a plain `cfg_scale` gets a prefix of its own:

```
[EOD] + tokenize(instruction(cot) + "\n[Tags]\n" + negative_style + "\n[Lyrics]\n" + "" + "\n")
      + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]          (cot=off: text + [MUSIC_START])
```

Branch slot, weight `c - 1`, blend, sampler, `--parallel 1` rule and lifecycle
are stage 7's, unchanged.

New code, all in `src/stage_ar.cpp` unless said otherwise:
`Request::{has_negative, negative_style}` + parsing, the §2 rules in
`validate_request`, `negative_request` (the request with the tags swapped and the
lyrics emptied) and `negative_prefix` beside `blank_prefix` (it calls
`prefix_head(vocab, negative_request(r).text())` — the positive prefix's own
builder — then `semantic_prefix`), the selection in `Runner::guidance_enter` and
in `run_ar_dump`, the context sizing on the negative head,
`JobState::negative_prefix` → `Artifacts::negative` → `negative_prefix.npy`,
`plan.json["negative_style"]`, the log lines, `ArResult::negative_style`
(`src/stage_ar.hpp`) → `config.json["negative_style"]` (`src/stage_song.cpp`).

## 1. The test requests

In the gitignored `tests/out/negative/`. `short.json` and `ext.json` are stage 7's
public requests (STATUS_GUIDANCE §1: a slow waltz, seed 7; `ext.json` carries its
609-id score in `"abc"`, so the run is all semantic phase).

| file | what |
|---|---|
| `ext_cfg3.json` | `ext.json` + `"cfg_scale": 3` |
| `neg.json` | `ext_cfg3.json` + `"negative_style": "children's song"` |
| `off.json` / `off_neg.json` | `short.json` with `"cot": "off"` / plus the negative style (no `cfg_scale`: the 1.01 default) |
| `keep_neg.json` | `neg.json` + `"semantic_keep": {"file": "R/semantic.npy", "frames": 64}` |
| `keep_cfg3.json` | the same without the negative style |
| `err/*.json` | one per §2 error |

`R/semantic.npy` is the 300-code `ext.json` render of item 1 (`new_ext`).

## 2. Acceptance

| # | check | result |
|---|---|---|
| 1 | no `negative_style`: `cmp` vs `build/yue2` | **PASS** — `short`, `ext`, `ext_cfg3`, `off` × `prefix.npy`/`abc_tokens.npy`/`semantic.npy`, all identical |
| 2 | prefix recipe | **PASS** — equals the reference's `token_prefixes` on the swapped request; text tokens = one-call encode; score segment = the positive one's |
| 3 | it does something | **PASS** — first-step blended rows differ (max abs Δ 3.20); `semantic.npy` first differs at **frame 1** (299 of 300 differ) |
| 4 | determinism | **PASS** — same request twice: `semantic.npy`, `prefix.npy`, `abc_tokens.npy`, `negative_prefix.npy` identical |
| 5 | `cot: "off"` + `negative_style` | **PASS** — runs at the 1.01 default; negative prefix is 23 tokens ending `…[Lyrics]\n\n` + `[MUSIC_START]`, no score |
| 6 | `semantic_keep` + `cfg_scale 3` + `negative_style` | **PASS** — first 64 codes equal the file; the negative branch is born at step 64 |
| 7 | `--verify-sampler` | **PASS** — 300 sampling steps matched the stage-5 sampler |
| 8 | every §2 error, CLI + `yue2-guidance` | **PASS** — 8 at the CLI, before the model loads; `yue2-guidance` 133 → **155** cases, 0 failures |

`yue2-bars` PASS (71), `yue2-handover` PASS (109) on the same build.

### 1 — untouched paths

```bash
D=tests/out/negative; cp build/yue2 $D/baseline_yue2
for r in short ext ext_cfg3 off; do
  nice -n 10 $D/baseline_yue2 ar -m yue2-ar-q8_0.gguf --request $D/$r.json \
      --artifacts $D/base_$r --gpu 1 --max-semantic 300
  nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/$r.json \
      --artifacts $D/new_$r --gpu 1 --max-semantic 300
  for f in prefix.npy abc_tokens.npy semantic.npy; do cmp $D/base_$r/$f $D/new_$r/$f; done
done
```

All 12 files identical: `short` (cot=full, score written, 300 frames), `ext`
(external score), `ext_cfg3` (the blank branch), `off` (cot=off's 1.01 blank
branch, no score).

### 2 — the prefix recipe

```bash
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/neg.json \
    --artifacts $D/neg_a --gpu 1 --max-semantic 300
```

Checked against the reference tokenizer (`YuE2TextTokenizer`, tiktoken, CPU; a
scratch script outside the repo):

| check | result |
|---|---|
| `negative_prefix.npy` (644 int32) == `protocol.token_prefixes(SongRequest(style="children's song", lyrics="", cot, abc), tok, abc_ids)` | **identical** — the reference's own positive-prefix builder on the swapped request |
| decoded text | `…from the given conditions.\n[Tags]\nchildren's song\n[Lyrics]\n\n` |
| `negative[1 : 1+n]` == `tok.encode(that whole string)` in one call | **identical** (no seam) |
| score segment `negative[ABC_START:]` == `prefix.npy[ABC_START:]` | **identical**, 612 tokens (`ABC_START` + 609 ids + `ABC_END`, `MUSIC_START`) |

### 3 — it does something

```bash
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/ext_cfg3.json --dump-logits $D/dump_cfg3.npy --gpu 1
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/neg.json      --dump-logits $D/dump_neg.npy  --gpu 1
```

| check (over `MUSIC_END` + the codec range) | result |
|---|---|
| `*.primary.npy` of the two runs | bit-identical (the positive branch is untouched) |
| `*.blank.npy` (the branch rows) | differ, max abs Δ 1.598 |
| blended rows | differ, max abs Δ 3.197 (= 2 × the branch Δ, as `B + 2(B − N)` says) |
| `dump_neg.npy` == `P + 2(P − N)` recomputed in numpy | **exact** |
| argmax blended, cfg3 / negative | 163899 / 163899 |
| `semantic.npy`, `new_ext_cfg3` vs `neg_a` | frame 0 equal, **first difference at frame 1**, 299 of 300 differ |

Frame 0 agreeing is the argmax above: the leader of the first step is the same
under both blends and the seeded draw picks it in both.

### 4 — determinism

`neg_a` and `neg_b` (the item-3 command twice): `semantic.npy`, `prefix.npy`,
`abc_tokens.npy`, `negative_prefix.npy` identical. A third run with
`--verify-sampler --guidance-trace` (`neg_v`) also produced the identical
`semantic.npy`.

### 5 — `cot: "off"`

```bash
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/off_neg.json \
    --artifacts $D/off_neg --gpu 1 --max-semantic 300
```

```
guidance: step 0: blank prefilled into slot 1 (23 + 0 tokens)
guidance: negative style "children's song" (23 tokens) replaces the blank branch
```

`negative_prefix.npy` = `[EOD]` + the one-call encode of the text (checked as in
item 2) + `[MUSIC_START]` — no `ABC_START`/`ABC_END`, no score, which is the
reference's `negative_prefix` form for cot=off (`base + [MUSIC_START]`).

### 6 — with `semantic_keep`

```bash
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/keep_neg.json \
    --artifacts $D/keep_neg --gpu 1 --max-semantic 300
```

```
keep: 64 frames from semantic.npy prefilled (692 + 64 tokens, 0.22 s)
guidance: step 64: blank prefilled into slot 1 (644 + 63 tokens)
guidance: negative style "children's song" (644 tokens) replaces the blank branch
guidance: step 300: blank branch dropped from slot 1 (the song is finished)
```

`semantic.npy[:64]` equals `R/semantic.npy[:64]`. Against the same keep with a
plain `cfg_scale: 3` (`keep_cfg3`), the first difference is frame 64 — the first
sampled one.

### 7 — `--verify-sampler`

```bash
nice -n 10 build_negative/yue2 ar -m yue2-ar-q8_0.gguf --request $D/neg.json \
    --artifacts $D/neg_v --gpu 1 --max-semantic 300 --verify-sampler --guidance-trace
```

```
ar verify: 300 sampling steps matched the stage-5 sampler exactly
guidance trace: 300 rows x 8 in guidance_trace.npy (0.524 ms/row); its blank columns are the negative style's branch
```

### 8 — the request errors

At the CLI (`build_negative/yue2 ar … --request $D/err/X.json`), all exit 1 before
the model loads:

```
nocfg   "negative_style" does nothing at cfg_scale 1: its branch is weighted cfg_scale - 1 = 0; set "cfg_scale", e.g. 3
cfg1    (the same, for an explicit "cfg_scale": 1.0)
empty   "negative_style" must be non-empty text: the tags the song is pushed away from
blank   (the same, for "  \n\t")
notstr  "negative_style" must be a string or null: the tags the song is pushed away from
guid    "negative_style" with "guidance" is not supported: it replaces the blank branch of a plain "cfg_scale", one mechanism per request
sect    "negative_style" with "sections" is not supported: …
hand    "negative_style" with "handover" is not supported: …
```

`tests/guidance.cpp`: 16 new request cases (accepted: the worked example, cot=off's
1.01, `cfg_scale` 0.5, `null`, keep + cfg + negative; rejected: no cfg, cfg 1.0,
cot=off with cfg 1.0, empty, whitespace, a number, a list, beside guidance /
sections / handover, and `abc_template` + cfg, which the existing rule catches),
2 `guidance_plan` cases (a negative style leaves the plan as `cfg_scale`'s), and a
`negative_request` check (the text is `instruction + "\n[Tags]\nchildren's
song\n[Lyrics]\n\n"`, the request itself untouched).

### Also run

`yue2 song` end to end on `neg.json` (`--gpu 1 --ar yue2-ar-q8_0.gguf`, Opus
out): rc 0, semantic 12.4 s, nar 13.6 s, vae 5.5 s; `config.json` has
`"cfg_scale": 3.0, "negative_style": "children's song"`, the artifacts directory
has `negative_prefix.npy`, `request.json` has no `negative_style`.

Cost: none beyond `cfg_scale`'s — `ext_cfg3` 2.52 s vs `neg` 2.54 s for 300
semantic steps; the branch is the same one row, only its prefix differs (644 vs
633 tokens here).

## 3. Deviations

| # | what | why |
|---|---|---|
| N1 | "not a string" is rejected in `parse_request_json`, not `validate_request`. | Every type error of the request form lives in the parser (`cfg_scale`, `abc`, `cot`, …) — `Request` holds a `std::string`, so the value cannot reach the validator. It is still before any GPU work, and `yue2-guidance` runs parse + validate, so the table case covers it. The empty/whitespace, cfg = 1 and combination rules are in `validate_request` as asked. |
| N2 | The "two call sites" of SPEC §3 are one: `Runner::guidance_enter`. | The stage-7 semantic entry, the end of the abc phase and `keep_enter` (the `semantic_keep` birth) all build the branch prefix through it. The only other site building a blank prefix is `--dump-logits` (`run_ar_dump`), which selects the same way. |
| N3 | `config.json` always carries `"negative_style"` (null when absent); `plan.json` only when set. | `config.json`: as SPEC §4 words it. `plan.json`: the `"guidance"` / `"semantic_keep"` precedent, so an unguided run's plan is what it was. |
| N4 | `negative_prefix.npy` is not in `plan_manifest.json`. | The manifest is checked against the reference's `SymbolicPlan.load()` whitelist; like `guidance_trace.npy` (T1) the file is an inspection aid, and `plan.json` (which is in the manifest) carries the text it derives from. |
| N5 | `--dump-logits` writes no metadata beside `FILE.blank.npy`; the run logs `dump: the blank row is the negative style "…"` instead. | There is no metadata file next to the dumps to extend (SPEC §4's "if any"). |
| N6 | The `--guidance-trace` "header" note is on the trace's log line, not in the file. | `guidance_trace.npy` is a bare float32 npy with no header to carry text. |
| N7 | cot=off: the negative prefix is **not** `token_prefixes` of the swapped request (that would end `[ABC_START, ABC_END, MUSIC_START]`). | SPEC §3 asks for "the same text, then `[MUSIC_START]` with no score", mirroring the reference's `negative_prefix` for cot=off (`base + [MUSIC_START]`), which is what `blank_prefix` does. Built as `prefix_head(...)` with its closing `ABC_START` replaced, so the text still comes from the positive builder. |
| N8 | The negative style is used verbatim; trimming only decides whether it is empty. | The positive `style` is not trimmed either, and SPEC §2's "after trimming" is about validity. |
| N9 | `cfg_scale` in `[0, 1)` with a `negative_style` is accepted. | SPEC §2 only rejects an effective 1 (a zero weight). A weight below zero pulls *towards* the negative style — odd, but meaningful, and the stage-7 range `[0, 20]` already allows the same with the blank. |
| N10 | The non-ASCII/NFC note is printed for the positive text only. | Unchanged from stage 7; the negative text is tokenized as given, the same way. |

## 4. Not done / open

- `negative_style` with `guidance` / `sections` (a negative text per entry or per
  branch) is rejected, per SPEC §2.
- No listening test: this stage checks the mechanism, not whether
  "children's song" as a negative is audible.

---

# Stage 11b: `negative_lyrics`

Contract: SPEC_NEGATIVE.md §7. Build `build_negative11b/` (same cmake line as
above; `YUE2_WARN_FLAGS` clean, the same pre-existing `ggml-backend.h` `-Wshadow`
only). Baseline for item 1: commit 4bdac91 exported with `git archive` into the
gitignored `tests/out/neg11_src/` (its `llama.cpp` a symlink to the same
submodule commit) and built into `build_neg11/`. Every model run is **Vulkan
device 1 (Arc Pro B70)**, `yue2-ar-q8_0.gguf`, `yue2 ar`; device 0 was not used.

## 0. What it is

```json
{ "cfg_scale": 3, "negative_style": "children's song", "negative_lyrics": true }
{ "cfg_scale": 3, "negative_lyrics": true }
```

`negative_request` keeps the lyrics when the flag is set (tags = `negative_style`,
or `""` without one); everything else is stage 11's machinery, selected by
`Request::replaces_blank()` (= `has_negative || negative_lyrics`) where stage 11
tested `has_negative`: `Runner::guidance_enter`, `run_ar_dump`, the context
sizing. New: `Request::negative_lyrics` + parsing + the §7.2 rules in
`validate_request`, `negative_label` (the log wording), `plan.json["negative_lyrics"]`
(only when true), `ArResult::negative_lyrics` → `config.json["negative_lyrics"]`
(always, true/false).

## 1. Requests

In the gitignored `tests/out/neg11b/`. `short`, `ext`, `ext_cfg3`, `off`, `neg`
are stage 11's (§1 above). New: `neg_false` / `ext_cfg3_false` (`"negative_lyrics":
false` added), `neglyr` (`neg` + true), `nostyle` (`ext_cfg3` + true, no style),
`self` (`ext_cfg3` + `negative_style` = its own `style` + true), `off_lyr` /
`off_neglyr` (cot=off at the 1.01 default), `keep_lyr` / `keep_neglyr` (+
`semantic_keep` 64 frames of stage 11's `R/semantic.npy`), `err/*.json`. For
item 4, `full_blank` / `full_neg` / `full_neglyr`: an original seven-section
lullaby lyric (three verses, three choruses, a bridge, an outro), an orchestral
storybook-lullaby style, `cot: full`, seed 11, `cfg_scale` 3; identical apart from
`"negative_style": "children's song, happy"` and `"negative_lyrics": true`.

## 2. Acceptance

| # | check | result |
|---|---|---|
| 1 | untouched: absent / `false` vs `build_neg11/yue2` | **PASS** — `short`, `ext`, `ext_cfg3`, `off`, `neg` and `neg_false` / `ext_cfg3_false` (vs `neg` / `ext_cfg3`): `prefix.npy`, `abc_tokens.npy`, `semantic.npy`, `negative_prefix.npy` all identical; `ext_cfg3_false` writes no `negative_prefix.npy`, `neg_false`'s `plan.json` has no `negative_lyrics` |
| 2 | recipe: negative = own style + lyrics | **PASS** — `negative_prefix.npy` == `prefix.npy` (692 tokens); `--dump-logits` blank row == primary row and blended == primary, max abs Δ 0.0 |
| 3 | recipe: no `negative_style` | **PASS** — equals the reference's `protocol.token_prefixes(SongRequest(style="", lyrics=<same>, …))` (the reference accepts an empty style); text reads `…[Tags]\n\n[Lyrics]\n[Verse]\n…`; text tokens = one-call encode; score segment = the positive one's (612 tokens) |
| 4 | full song, three runs | **PASS** (lyrics-kept clearly lower) — table below |
| 5a | determinism | **PASS** — `neglyr` twice (+ a third with `--verify-sampler --guidance-trace`): `semantic.npy` identical; `prefix`/`abc_tokens`/`negative_prefix` identical |
| 5b | `--verify-sampler` | **PASS** — 300 sampling steps matched the stage-5 sampler |
| 5c | `semantic_keep` + `negative_lyrics` | **PASS** — `keep_lyr`, `keep_neglyr`: first 64 codes equal the file; branch born at step 64 |
| 5d | §7.2 errors, CLI + `yue2-guidance` | **PASS** — 8 at the CLI, before the model loads; `yue2-guidance` 155 → **183** checks, 0 failures |

`yue2-bars` PASS (71), `yue2-handover` PASS (109). `yue2 song` end to end on
`nostyle.json` (`--gpu 1`, Opus): rc 0; `config.json` has `"cfg_scale": 3.0,
"negative_style": null, "negative_lyrics": true`; `negative_prefix.npy` present;
`request.json` carries neither key.

### 1 — untouched

```bash
D=tests/out/neg11b
for r in short ext ext_cfg3 off neg; do
  nice -n 10 build_neg11/yue2 ar -m yue2-ar-q8_0.gguf --request $D/$r.json \
      --artifacts $D/base_$r --gpu 1 --max-semantic 300
done
for r in short ext ext_cfg3 off neg neg_false ext_cfg3_false; do
  nice -n 10 build_negative11b/yue2 ar -m yue2-ar-q8_0.gguf --request $D/$r.json \
      --artifacts $D/new_$r --gpu 1 --max-semantic 300
done
# cmp base_X vs new_X (and base_neg vs new_neg_false, base_ext_cfg3 vs new_ext_cfg3_false)
```

### 2, 3 — the recipe

```bash
B="nice -n 10 build_negative11b/yue2 ar -m yue2-ar-q8_0.gguf --gpu 1"
for r in self nostyle neglyr off_lyr off_neglyr; do $B --request $D/$r.json --artifacts $D/$r --max-semantic 300; done
for r in self ext_cfg3 nostyle neglyr; do $B --request $D/$r.json --dump-logits $D/dump_$r.npy; done
```

Checked against `YuE2TextTokenizer` + `protocol` (CPU, a scratch script outside
the repo):

| request | negative tokens | == reference | text = one-call encode | score segment = positive |
|---|---|---|---|---|
| `self` | 692 (== `prefix.npy`) | yes | yes | yes |
| `nostyle` | 676 | yes (`token_prefixes`, `style=""`) | yes | yes |
| `neglyr` | 680 | yes | yes | yes |
| `off_lyr` | 55 | yes (`[EOD]` + text + `[MUSIC_START]`, N7) | yes | — (no score) |
| `off_neglyr` | 59 | yes (same form) | yes | — |

`--dump-logits` over the 32 769 semantic ids: `nostyle` / `neglyr` primary rows
bit-identical to `ext_cfg3`'s; branch rows differ from the blank by max abs 1.27 /
1.71; blended == `P + 2(P − N)` recomputed in float32 exactly. `semantic.npy`
vs the lyric-less `neg`: `neglyr` first differs at frame 3 (296/300 differ);
`nostyle` vs `ext_cfg3` at frame 3.

`self` sampled with `--guidance-trace` has TV(song, negative) = 0 and TV(blend)
= 0 at all 300 steps (the blend moved nothing), yet its `semantic.npy` differs
from the unguided `ext` from frame 1: the positive row of a two-stream guided
context is not bit-identical to the single-stream decode on this backend (the
batch shape changes the kernels). Not a recipe difference; noted, not chased.

### 4 — the point of the stage

```bash
for r in full_blank full_neg full_neglyr; do
  nice -n 10 build_negative11b/yue2 ar -m yue2-ar-q8_0.gguf --gpu 1 \
      --request $D/$r.json --artifacts $D/$r --guidance-trace
done
```

All three wrote the same score: `abc_tokens.npy` (1458 ids), `score.abc` and
`prefix.npy` identical. ~59 s semantic each; 4808 / 4747 / 4738 codes (~190 s).
`semantic.npy` differs from frame 0 (blank vs either negative) and frame 2
(the two negatives). Mean of trace col 4 (TV(song, negative)) / col 6 (TV of the
blend), weight 2 throughout:

| window | blank | negative, no lyrics | negative + lyrics |
|---|---|---|---|
| **whole song** | **0.1216 / 0.1893** | **0.1238 / 0.1927** | **0.0292 / 0.0584** |
| 0–30 s | 0.1360 / 0.2012 | 0.1412 / 0.2229 | 0.0538 / 0.1065 |
| 30–60 s | 0.1724 / 0.2471 | 0.1676 / 0.2430 | 0.0354 / 0.0715 |
| 60–90 s | 0.1188 / 0.1915 | 0.1198 / 0.1848 | 0.0227 / 0.0460 |
| 90–120 s | 0.0976 / 0.1683 | 0.1034 / 0.1759 | 0.0253 / 0.0507 |
| 120–150 s | 0.1637 / 0.2377 | 0.1676 / 0.2380 | 0.0221 / 0.0442 |
| 150–180 s | 0.0637 / 0.1204 | 0.0646 / 0.1221 | 0.0187 / 0.0373 |
| 180–end | 0.0664 / 0.1157 | 0.0579 / 0.1012 | 0.0200 / 0.0402 |

The §7.1 finding reproduces on a second song: the lyric-less negative sits as far
from the song as the blank (0.124 vs 0.122, window by window). With the lyrics
kept, the negative is about 4× closer (0.029), so at the same weight the blend
moves the sampler about a third as much (col 6 0.058 vs 0.19) — and what it
moves is the tag difference alone. Equal *push* to the lyric-less negative would
need a larger `cfg_scale`; not measured, and no listening test was done.

### 5 — errors

```
nocfg     "negative_lyrics" does nothing at cfg_scale 1: its branch is weighted cfg_scale - 1 = 0; set "cfg_scale", e.g. 3
cfg1      (the same, explicit 1.0)
off_cfg1  (the same, cot=off with "cfg_scale": 1.0)
str       "negative_lyrics" must be true, false or null: whether the negative branch sings the song's lyrics
num       (the same, for 1)
guid      "negative_lyrics" with "guidance" is not supported: it changes the blank branch of a plain "cfg_scale", one mechanism per request
sect      "negative_lyrics" with "sections" is not supported: …
hand      "negative_lyrics" with "handover" is not supported: …
```

`tests/guidance.cpp`: 16 request cases (accepted: both §7.2 examples, cot=off's
1.01, keep + cfg + true, `false` with no cfg, `null`, `false` beside a negative
style, `false` beside a guidance block; rejected: the eight above), 2 plan cases
(true leaves `cfg_scale`'s plan; false alone asks for no branch), and
`negative_request` / `negative_label` / `replaces_blank` checks (with lyrics,
without a style, own style = `text()` exactly, false = blank).

## 3. Deviations (11b)

| # | what | why |
|---|---|---|
| L1 | The non-boolean error is in `parse_request_json`. | As N1: `Request` holds a `bool`; still before any GPU work and covered by `yue2-guidance`. |
| L2 | `config.json["negative_lyrics"]` is always written (`false` when absent); `"negative_style"` stays `null` for an empty-tags negative. | §7.3 wording ("true\|false"); `negative_style` records the request's key, not the branch's tags. |
| L3 | The trace and `--dump-logits` log lines changed wording for the stage-11 case too: `…its blank columns are the negative branch: negative style "X"` / `dump: the blank row is the negative branch: negative style "X"`. The `guidance: negative style "X" (P tokens) replaces the blank branch` line is unchanged. | One label function for the three forms of §7.3; logs are not compared. |
| L4 | cot=off + `negative_lyrics`: `[EOD]` + text + `[MUSIC_START]` (stage 11's N7 form), not `token_prefixes` of the swapped request. | §7.2 "same text rule, then `[MUSIC_START]`, as §3". |
| L5 | `"negative_lyrics": false` is accepted beside `guidance` / `sections` / `handover` and with no `cfg_scale`. | §7.2 "false is always accepted and changes nothing". |
| L6 | README cites the §7.1 numbers and this stage's second-song numbers together. | §7.1's render is not in the public tree; both are labelled. |
