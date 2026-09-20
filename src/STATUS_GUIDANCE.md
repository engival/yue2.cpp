# STATUS — stage 7: guided semantic decoding (`cfg_scale` + mid-song style changes)

Contract: [SPEC_GUIDANCE.md](../SPEC_GUIDANCE.md). Build `build_guidance/`
(`cmake -B build_guidance -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DYUE2_BUILD_TESTS=ON`,
`nice -n 10 cmake --build build_guidance -j8`). Everything here was measured on
**Vulkan device 1 (Intel Arc Pro B70)** or on the CPU; device 0 was never touched.

A note on the CPU runs: the §4 reference-parity pass (torch f32 + one `--cpu`
prefill + 32 greedy steps) and the `--dump-logits` golden are memory-bandwidth
heavy enough to make the desktop unusable, `nice` or not. They are one-off
checks; everything repeatable here runs on the Arc, and a bit-identity check is
just as valid there as long as both sides use the same device and flags.

## 0. What it is

`yue2 ar` used to die on `cfg_scale != 1`. It now decodes the semantic phase
beside up to two *negative* sequences that carry the same generated history under
a different prefix, and hands the sampler their blend:

```
L = B + sum_i w_i(t) * (B - N_i)          i in { previous, blank }
```

- `"cfg_scale": c` is the reference's classifier-free guidance: the `blank`
  branch (instruction + exact score, no tags, no lyrics — `protocol.negative_prefix`)
  live from step 0 at the constant weight `c - 1`. `cot: "off"`'s 1.01 default is
  no longer an error.
- `"guidance"` is the same machinery with per-branch weight curves in frames and
  a positive prefix that can change at a given frame — a mid-song style change.
  Entries, curves and every request error are in README.md and SPEC_GUIDANCE §2.
- One song, up to three KV streams (`n_seq_max = 3`, `n_ctx = n_ctx_want * 3`).
  The extra rows ride in the *same* lockstep `llama_decode`, so a guided step
  costs 1.2–2.5×, not 2–3× (§9).
- `sample_step` and `sample_step_ref` are untouched: they are handed a blended
  row instead of a raw one, so `--verify-sampler` verifies guided songs for free.
- **The unguided path is bit-identical to the baseline binary** — same context
  parameters, same batch shapes, no scratch row, no extra stream (§1).

New code all lives in `src/stage_ar.cpp`: `Keyframe`/`GuidanceEntry` +
`curve_at`/`curve_needed_from`, `parse_guidance`/`parse_curve` and the §2.3/§2.4
rules in `validate_request`, `guidance_plan`, the `prefix_head`/`semantic_prefix`/
`blank_prefix` helpers (§4.4 — the three inlined copies of the prefix recipe are
gone), `struct Branch` and `Runner::{free_slot, prefill_branch, branch_drop,
take_entry, guidance_enter, guidance_step, guidance_clear, fetch_row, blend_row}`.

## 1. The test requests

Public, made for this stage, and kept in the gitignored `tests/out/guidance/`;
reproduced here so the commands below can be re-run.

`short.json` — the score-generating case:

```json
{
  "style": "slow waltz, upright bass, brushed drums, warm baritone vocal",
  "lyrics": "[Verse]\nThe kettle sings a flat blue note\nThe window keeps the rain outside\n\n[Chorus]\nCount the lamps along the road\nOne for every year we hide\n",
  "cot": "full",
  "seed": 7
}
```

`ext.json` is `short.json` plus the `score.abc` that the run below wrote as
`"abc"` (856 B, 609 abc ids, sha256 `7a0ca7734d946932…`), so the decode starts in
the semantic phase. `short2.json` is `short.json` with
`"style": "uptempo ska, offbeat guitar, horn section, bright tenor vocal"` and
`"seed": 11`. `tpl.json` is `short.json` with the score's first 12 lines as an
`"abc_template"` ending in `%%yue2-gen`. The guided requests add, to `ext.json`:

| file | block |
|---|---|
| `ext_cfg3.json` | `"cfg_scale": 3` |
| `ext_g64.json` | `"guidance": [{"frame": 64, "style": "uptempo ska, …", "against": {"previous": [[0,11],[60,11],[85,3]], "blank": [[0,2]]}}]` |
| `ext_g900.json` | the same entry at `"frame": 900` |
| `ext_life.json` | three entries: `{40, ska, previous [[0,5],[40,0]], blank [[0,2],[40,0]]}`, `{150, lo-fi bedroom pop, previous [[0,6]], blank [[0,2]]}`, `{9000, marching band, previous [[0,5]]}` |
| `cost_cfg3.json` | `"cfg_scale": 3` |
| `cost_window2.json` | one entry at frame 200, both curves `[[0,w],[500,w],[501,0]]` |

## 2. §5.1 — the unguided path is untouched

`cmp` on `prefix.npy`, `abc_tokens.npy`, `semantic.npy` between `build/yue2`
(baseline) and `build_guidance/yue2`, all on `--gpu 1`:

```bash
for b in build build_guidance; do
  ./$b/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/R.json \
      --artifacts tests/out/guidance/${b}_R --gpu 1 --max-semantic 300
done
```

| request | what it exercises | result |
|---|---|---|
| `short.json` | `cot=full`, score generated (609 abc ids) | all three **identical** |
| `ext.json` | external `"abc"`, straight into the semantic phase | all three **identical** |
| `tpl.json` | `"abc_template"`, one hole | all three **identical** |
| `jobs2.json`, `--parallel 2` | two unguided jobs in one context | both jobs, all three **identical** |

The `--parallel 2` batch also reports the same step counts on both binaries
(1077 full-width steps, 0 holed, 1081 decode calls, 7.53 ms/step). It was run as
`yue2 ar --requests`, which is the same `run_ar_batch` call `yue2 batch` makes —
`yue2 batch`'s own extra is the per-job NAR and VAE, which this stage does not
touch. `yue2 song` was run end to end on a guided request instead (§7).

Stage-2 goldens, on the private reference request:

| check | device / gguf | result |
|---|---|---|
| `--dump-logits` vs `build/yue2` | CPU, f16 | **byte-identical** |
| `--dump-logits` vs `tests/golden/ar_last_logits_f32.npy` | CPU, f16 | argmax **55 = 55**, max abs Δ **0.0017** (matches `STATUS_AR.md` §11's re-measurement, not the 0.2386 in §1's table) |
| `--greedy --max-abc 32` vs `build/yue2` | Arc, q8_0 | **byte-identical** |
| `--greedy --max-abc 32` vs `tests/golden/ar_greedy_32.npy` | Arc, q8_0 | **identical, 32/32** |
| `build_guidance/yue2-sampler-diff` | CPU | **0 mismatches** over 100 000 cases (191 s) |
| `build_guidance/yue2-bars` | CPU | PASS, 71 cases |

## 3. §5.2 — the blend arithmetic

```bash
./build_guidance/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/ext_cfg3.json \
    --dump-logits tests/out/guidance/cfg3.npy --gpu 1
```

writes `cfg3.npy` (blended), `cfg3.primary.npy`, `cfg3.blank.npy`. Recomputed in
numpy, `P + np.float32(2.0)*(P-N)`:

| check | result |
|---|---|
| blended == recomputed over the allowed ids (`[MUSIC_END, CODEC_OFFSET+CODEC_SIZE)`) | **exact**, max abs Δ 0.0 |
| outside the allowed ids the file keeps the positive row | exact |
| argmax blended / primary | 163899 / 163899 |

The second half of §5.2 could not be run as written — see deviation **G1**. Its
substitute measures the same thing: the *positive* row prefilled alone in a
one-stream context vs the same prefix prefilled in the two-stream guided context
*after* the branch's decode (`ext_late.json`, whose only entry is at frame 10, so
step 0 has no branch and the context is one stream):

| check | result |
|---|---|
| `cfg3.primary.npy` vs `late.primary.npy` | **bit-identical**, max abs Δ **0.0** over all 184 704 ids |

So on this backend a separately prefilled sequence costs no tolerance at all:
prefill is its own `llama_decode` with the same shape either way.

## 4. §5.3 — reference parity (torch CPU float32)

`convert/reference_guidance.py` (new; CPU only, asserts the model stays on CPU)
runs `unconditional + cfg_scale * (conditional - unconditional)` on
`ext_cfg3.json` and continues 32 greedy semantic steps. Goldens land in
`tests/out/guidance/` with `ref_guidance_meta.json` (see deviation **G2**).

```bash
nice -n 10 venv_yue2/bin/python convert/reference_guidance.py \
    --request ../tests/out/guidance/ext_cfg3.json          # 8.1 s load, 6.1 s prefill, 13.0 s greedy
nice -n 10 ./build_guidance/yue2 ar -m yue2-ar-f16.gguf --cpu --threads 8 \
    --request tests/out/guidance/ext_cfg3.json --dump-logits tests/out/guidance/cpu_cfg3.npy
nice -n 10 ./build_guidance/yue2 ar -m yue2-ar-f16.gguf --cpu --threads 8 --greedy --max-semantic 32 \
    --request tests/out/guidance/ext_cfg3.json --artifacts tests/out/guidance/cpu_greedy32
```

positive prefix 692 tokens, negative 633, `cfg_scale` 3, logits of magnitude ~20:

| row | max abs Δ over the allowed ids | argmax (torch / ours) |
|---|---|---|
| primary (conditional) | 0.00281 | 163899 / 163899 |
| blank (unconditional) | 0.00301 | 163899 / 163899 |
| **blended** | 0.00853 | 163899 / 163899 |

The blended row's error is 3× the branches' — exactly what `B + 2(B-N)` does to
two independently rounded rows — and is the F16-GGUF-vs-f32-torch gap of
`src/STATUS_AR.md` §1, not a guidance error. The two algebraic forms of the blend
(**G4**) differ by at most **9.5e-7** in f32 on identical rows.

| check | result |
|---|---|
| 32 greedy semantic codes vs `ref_greedy_32.npy` | **identical, 32/32** |

## 5. §5.4, §5.5 — prefix and determinism

```bash
./build_guidance/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/ext_g64.json \
    --artifacts tests/out/guidance/g64 --gpu 1 --max-semantic 300
```

| check | result |
|---|---|
| first 64 codes vs the unguided `ext.json` run, same seed | **identical** |
| first differing code | index **64** — the first step the entry touches |
| the same command twice (`g64` vs `g64b`) | `semantic.npy` **identical** |

`prefix.npy` of a guided song (`yue2 song` on `ext_g900.json`) is byte-identical
to the unguided run's: it stays the frame-0 positive prefix, which is what the
NAR consumes.

## 6. §5.6 — lifecycle

`ext_life.json`, `--max-semantic 300`, one command:

```
guidance: step 40: slot 0 relabelled as the previous branch
guidance: step 40: new positive prefilled into slot 1 (692 + 39 tokens)
guidance: step 40: blank prefilled into slot 2 (633 + 39 tokens)
guidance: step 80: previous branch dropped from slot 0 (its curve is zero from here on)
guidance: step 80: blank branch dropped from slot 2 (its curve is zero from here on)
guidance: step 150: slot 1 relabelled as the previous branch
guidance: step 150: new positive prefilled into slot 0 (689 + 149 tokens)
guidance: step 150: blank prefilled into slot 2 (633 + 149 tokens)
guidance: step 300: previous branch dropped from slot 1 (the song is finished)
guidance: step 300: blank branch dropped from slot 2 (the song is finished)
```

| check | result |
|---|---|
| a curve that returns to 0 drops its branch | yes, at step 80 (offset 40 of entry 1) |
| two entries in one song | yes; the second relabels the *new* primary and re-prefills |
| an entry past the end | `guidance.json`: `[(40, true), (150, true), (9000, false)]` |
| `guided_steps` | **190** = (80−40) + (300−150), i.e. only the windows |
| `cfg_branches` / `branch_prefill_seconds` | 3 / 0.650 s |
| `yue2 batch --parallel 1`, guided job then unguided one | the second job's `prefix.npy`, `abc_tokens.npy`, `semantic.npy` are **identical** to running it alone |

The slots the primary moves through are cleared by `guidance_clear` +
`finish_job`, and the next job re-enters its home slot — which is why the
unguided second job of that batch reproduces bit for bit even though the context
has three streams.

Artifacts of a guided song: `guidance.json` (the normalised block with a
`reached` flag per entry), the same block under `plan.json`'s `"guidance"`, and
`guidance.json` in `plan_manifest.json`. `request.json` carries only `cfg_scale`,
so it stays loadable by the reference's `SongRequest(**request.json)`.

Request errors (all before the model loads, all via `strf`):

```
$ yue2 ar --request bad_both.json …
error: …: "guidance" and "cfg_scale" are two ways to ask for the same machinery: …not both
$ yue2 ar --request bad_tpl.json …
error: …: classifier-free guidance with "abc_template" is not supported yet: …
$ yue2 ar --requests jobs_par.json --parallel 2 …
error: …/ext_g64.json: guidance needs --parallel 1 (this batch decodes 2 songs side by side)
```

with `--continue-on-error` the guided job alone is rejected and the unguided one
decodes.

## 7. §5.7 — `--verify-sampler` over a full guided song

```bash
./build_guidance/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/ext_g900.json \
    --artifacts tests/out/guidance/full_verify --gpu 1 --verify-sampler
```

Full semantic phase to `MUSIC_END`: 1517 codes (60.7 s of audio), the entry at
frame 900 live for the last 618 steps.

```
ar verify: 1518 sampling steps matched the stage-5 sampler exactly
```

**Zero disagreements.** The frozen `sample_step_ref` sees the blended row too, so
this verifies the blend feeds a byte-identical input to both samplers.

`yue2 song` on the same request rendered end to end (`--gpu 1`): abc 0.0 s,
semantic 19.4 s, nar 23.5 s, vae 6.0 s, 60.68 s of audio;
`config.json` `cfg_scale` 1.0 (the block is in `guidance.json`),
`result.json` `timing.abc` `cfg_branches` 3, `guided_steps` 618.

## 8. §5.8 — the tests-only target

```bash
./build_guidance/yue2-guidance          # PASS: 70 cases, 0 failures
```

`tests/guidance.cpp` (`#include "stage_ar.cpp"`, target `yue2-guidance` under
`YUE2_BUILD_TESTS`, no model and no device): 17 `curve_at` cases (holds at both
ends, a step at equal offsets, interpolation, negative and zero-crossing curves),
11 `curve_needed_from` cases (a zero tail, a zero stretch with a push behind it,
the frame before and after a landing), 31 request cases covering every error of
§2.3/§2.4 (including G11) and the forms that must be accepted, and 6
`guidance_plan` cases
(`cfg_scale` → one blank entry at `c - 1`, `cot=off`'s 1.01, `1.0` → unguided).

## 9. §5.9 — what it costs

Arc, `yue2-ar-q8_0.gguf`, `ext*.json` (external score, so the whole run is the
semantic phase), `--max-semantic 1000` so all three decode exactly 1000 steps:

| case | live rows | semantic seconds | tok/s | ms/step |
|---|---|---:|---:|---:|
| unguided (`ext.json`) | 1 | 7.25 | 137.8 | 7.25 |
| `cfg_scale: 3` (`cost_cfg3.json`) | 2, all 1000 steps | 8.86 | 112.8 | 8.86 |
| 3-branch window of 501 frames (`cost_window2.json`) | 1 / 3 / 1 | 13.13 | 76.2 | — |

Derived, with the 0.65 s of branch prefill taken out of the window run:

| region | ms/step | vs unguided |
|---|---:|---:|
| one row | 7.25 | ×1.00 |
| two rows | 8.86 | **×1.22** |
| three rows | ~17.7 | **×2.44** |

The SPEC's expectation was ×2 / ×3 (deviation **G9**): two rows are much cheaper
than that because they ride in one `llama_decode` — the same reason `yue2 batch`
exists. Three rows fall off more because the Arc's step is no longer
launch-bound at that width. The cost is confined to the window either way: the
`ext_life.json` song paid for 190 of 300 steps, and a real transition is 100–200
frames (4–8 s).

Branch prefill is 0.65–0.71 s per entry on the Arc (two or three prefills of
~700 + the codes so far) and is reported as `branch_prefill_seconds`.

Slot contiguity (SPEC_BATCH §4.7 item 3): `free_slot` always hands out the lowest
free stream, and in every guided run measured the live set was `{0}`, `{0,1}`,
`{1,2}` or `{0,1,2}` — never a hole in the middle. The batch counter (which now
counts branch slots too) agrees: the guided + unguided `--parallel 1` batch of §6
reports `ar steps: 1207 full-width, 0 with an idle slot in the middle`, so
`split_equal` never split a guided step into two ubatches.

## 10. Deviations

| # | what | why |
|---|---|---|
| G1 | §5.2's second half — "the `blank` row equals a plain `--dump-logits` of a request whose prefix is the blank prefix" — was **not** run as written. | No request can produce the blank prefix: `Request::text()` always inserts `"\n[Tags]\n…\n[Lyrics]\n…"`, and §2.2 forbids slicing the positive prefix. Substituted the equivalent measurement (§3): the *positive* row, prefilled alone vs prefilled in the guided two-stream context. Result: bit-identical, so the tolerance being asked about is 0 on this backend. |
| G2 | The §5.3 goldens are in `tests/out/guidance/` with `ref_guidance_meta.json`, not in `tests/golden/`. | `tests/golden/` is read-only for this agent (CLAUDE.md). The script writes wherever `--out` says; moving them is one `mv` plus the SHA-256s already in the meta file. Note they derive from a request of mine (public, §1) rather than the private reference request, so they *can* ship as the public goldens ROADMAP item 6 asks for. |
| G3 | `--dump-logits` on a guided request requires the score in `"abc"` (or `cot: "off"`), and errors otherwise. | The row asked for is the first *semantic* step. Reaching it from a `cot=full` request with no score means decoding a whole abc phase inside the goldens' path — that is what `--artifacts` is for. |
| G4 | The blend is computed as `B + Σ w(B−N)` (SPEC §2.1), not as the reference's `N + c(B−N)`. | The SPEC mandates the first form, and it is what the generalisation to per-branch curves needs. Measured difference between the two f32 forms on identical rows: ≤ 9.5e-7 (§4). |
| G5 | `cfg_branches` / `guided_steps` / `branch_prefill_seconds` are written on **both** phases' timing blocks, including the abc phase, which is never guided. | They are song-level numbers, and the only `json_timing()` call that reaches an artifact is the one for the *abc* stats (`plan.json`, `result.json["timing"]["abc"]`). Writing them on one phase only would hide them. |
| G6 | `guidance.json` is added to `plan_manifest.json`, which makes the manifest fail the reference's `SymbolicPlan.load()` whitelist for guided jobs. | SPEC §3 asks for it, and `template.abc` already did exactly this (SPEC_TEMPLATE §5). `request.json` and `plan.json` themselves stay loadable. |
| G7 | `ArResult::slot` (→ `result.json["batch"]["slot"]`) records the slot the job *started* in, not the one it ended in. | A guided song's primary moves between streams at every cut. The home slot is the one that means something for batch composition, and it is the one the next job re-enters. |
| G8 | The "guidance needs `--parallel 1`" rejection is made against the **clamped** parallel, not `--parallel` as given. | Otherwise `yue2 batch --jobs one.json` (default `--parallel 4`, clamped to 1 for a single job) would reject a guided song that in fact decodes alone. |
| G9 | §5.9's "roughly ×2 / ×3 inside the window" was not observed; measured ×1.22 and ×2.44. | The branches ride in the same lockstep `llama_decode` as the primary, so a guided step is one wider batch, not two or three separate decodes. Reported rather than "fixed". |
| G11 | An entry at `frame: 0` carrying a `style` is a **request error** ("a `style` at frame 0 is the request's own `style`; put it there instead"), where SPEC §2.3 only forbids `against.previous` there. | Found in the cold review pass: the cut relabels the sequence the song has been decoding and prefills its replacement, and at frame 0 that sequence does not exist yet — the branches are prefilled *before* the primary's own feed, so a frame-0 cut would have prefilled the new prefix and then fed the old one on top of it. It is also not a loss: the entry's tags from frame 0 on are exactly what the request's `style` says, and `prefix.npy` (the frame-0 positive prefix, which the NAR reads) has to agree with the prefix the song started from. SPEC's "previous is not allowed at frame 0" then needs no rule of its own — `previous` requires a `style`, and a `style` cannot be there. |
| G10 | `branch_prefill_seconds` also counts the re-prefill of the *positive* sequence at a cut, not only the negative branches. | It is the same event and the same cost — a cut prefills a new slot with the new prefix + history — and separating them would report a number that does not add up to what the cut took. |

## 11. Not done / open

- Guided jobs at `--parallel > 1` and guidance with `"abc_template"` are rejected
  with a clear message, as SPEC §2.4 asks; both are now ROADMAP item 5.
- The abc phase is never guided (SPEC §2.1), so `guidance` entries always address
  semantic steps.
- `tests/regress.sh` was not extended: it decodes the author's private renders
  (ROADMAP item 6 is the public replacement, and §5.3's goldens are a start).

# STATUS — stage 7b: `semantic_keep` (keep the start of an earlier render)

Contract: [SPEC_KEEP.md](../SPEC_KEEP.md), built on stage 7 above. Build
`build_keep/` (`cmake -B build_keep -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
-DYUE2_BUILD_TESTS=ON`, `nice -n 10 cmake --build build_keep -j8`). Every number
below was measured on **Vulkan device 1 (Intel Arc Pro B70)**; device 0 was never
touched, and `build/` and `build_guidance/` were not rebuilt.

## 0b. What it is

```json
"abc": "…the score that render sang…",
"semantic_keep": { "file": "earlier/semantic.npy", "frames": 3400 }
```

Frames `0 .. N-1` are not sampled: they are read out of an earlier run's
`semantic.npy` and prefilled as history, and the first draw is step N. The
mechanism is one function, `Runner::keep_enter`, and it needs no new machinery
because of one choice: **the slot is prefilled with `prefix_sem + kept[0 .. N-2]`
and `kept[N-1]` is left pending in `q.next`** — which is exactly the state a
sampled step leaves behind. The lockstep batch then feeds that token to the
primary and to every guidance branch at once, and the first row it yields is step
N's. `prefill_branch`, `guidance_step` and the lockstep loop are untouched.

The one ordering difference from an unkept song: `guidance_enter` runs *after*
the primary's prefill, not before it, because an entry at `frame == N` is a cut
and a cut relabels the sequence the song has been decoding (SPEC_KEEP §3). That
is what "prefill the old-tags sequence and the new one back to back" comes out as.

New code, all in `src/stage_ar.cpp` unless said otherwise: `Request::{has_keep,
keep_file, keep_frames}` + `parse_semantic_keep`, the §2 rules in
`validate_request`, `dir_of` + `load_keep_codes` (which calls the NAR's reader,
`npy::load_i32`, and holds the file to the same 1-D-int32 and codec-range rules),
`JobState::{keep_codes, keep_name, keep_seconds}`, `Runner::keep_enter`, the
`semantic_keep` block of `plan.json`, `GenStats::{kept_frames,
keep_prefill_seconds}` (`src/stage_ar.hpp`, in both copies of `json_timing`) and
`ArJob::base_dir`. The `Sampling` block of `run_ar_batch` moved above the
validation loop so the frame cap is available there (deviation **K4**).

## 1b. The test requests

`tests/out/keep/` (gitignored). They are stage 7's public `ext.json` (§1 above:
external `"abc"`, so the whole run is the semantic phase, seed 7) plus one block:

| file | block added to `ext.json` |
|---|---|
| `keep_n1/64/299.json` | `"semantic_keep": {"file": "R/semantic.npy", "frames": N}` |
| `kg_N.json` | the same against the `--greedy` baseline `Rg/semantic.npy` |
| `keep_g64.json` | keep 64 + the `ext_g64.json` entry, moved to `"frame": 64` |
| `keep_g1.json` | keep 1 + the same entry at `"frame": 1` (the smallest cut) |
| `keep_cfg3.json` | keep 64 + `"cfg_scale": 3` |
| `keep_n1000.json` | keep 1000 of a 1100-frame render |
| `err/*.json` | one per §2 error |

`R` = `ext.json` at `--max-semantic 300` (sampled, 300 codes); `Rg` = the same
with `--greedy`; `R1100` = the same at `--max-semantic 1100`.

## 5b. Acceptance

| # | check | result |
|---|---|---|
| 1 | no `semantic_keep`: `cmp` vs `build_guidance/yue2` | **PASS** — 4 requests x 3 files, all identical |
| 2 | kept codes at N = 1, 64, len-1 | **PASS** — `semantic.npy[:N]` equals the file, `plan.json` sha matches |
| 3 | teacher forcing at N = 100 | **PARTIAL** — exact for 55 frames, then flips at frame 155 (below) |
| 4 | determinism | **PASS** — `semantic.npy` identical, `plan.json` differs only in `keep_prefill_seconds` |
| 5 | guided + keep at `frame == N` | **PASS** — keep, relabel, new positive, blank, `reached: true` |
| 6 | `--verify-sampler` on a keep run | **PASS** — 236 steps = 300 - 64, zero disagreements |
| 7 | every §2 error, CLI + `yue2-guidance` | **PASS** — 10 at the CLI, 13 new table cases (70 -> 83) |
| 8 | cost of keeping 1000 frames | **PASS** — 0.43 s prefill vs 7.25 s to sample them, **x17** |

### 1 — the unguided and guided paths are untouched

```bash
for r in short ext ext_cfg3 ext_g64; do for b in build_guidance build_keep; do
  ./$b/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/guidance/$r.json \
      --artifacts tests/out/keep/${b}_$r --gpu 1 --max-semantic 300
done; done
```

`prefix.npy`, `abc_tokens.npy` and `semantic.npy` are byte-identical on all four:
`short` (cot=full, score generated), `ext` (external score), `ext_cfg3`
(`cfg_scale: 3`), `ext_g64` (a mid-song cut). `yue2-bars` PASS (71 cases) and
`yue2-guidance` PASS.

### 2 — the kept codes

```bash
./build_keep/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/keep/keep_n64.json \
    --artifacts tests/out/keep/out_n64 --gpu 1 --max-semantic 300
```

| N | log line | `semantic.npy[:N]` == file | `plan.json` |
|---:|---|---|---|
| 1 | `keep: 1 frames from semantic.npy prefilled (692 + 1 tokens, 0.16 s)` | yes | `{frames: 1, sha256: 83cf74f7…}` |
| 64 | `… (692 + 64 tokens, 0.20 s)` | yes | `{frames: 64, sha256: 4397bbd4…}` |
| 299 | `… (692 + 299 tokens, 0.25 s)` | yes | `{frames: 299, sha256: fbdf6780…}` |

The sha is of the N kept codes as int32 LE, and it is what the run read, not what
it wrote — recomputed in numpy from `R/semantic.npy[:N]` it matches in all three.
`timing.kept_frames` / `timing.keep_prefill_seconds` carry the same N and the
prefill seconds; `yue2 song` end to end on a keep request rendered 1441 codes to
FLAC and put both into `result.json`'s `timing.abc`.

### 3 — teacher forcing: exact except at measured near-ties

`Rg` = `ext.json --greedy --max-semantic 300`. Re-run with `--greedy` and
`semantic_keep` of `Rg` at several N; "first diff" is against `Rg` itself:

| N | first sampled frame == `Rg[N]` | first differing frame |
|---:|---|---|
| 1 | yes | **none — byte-identical over all 300** |
| 2 | yes | **none — byte-identical over all 300** |
| 3 | no | 3 |
| 100 | yes | 155 |
| 120 | yes | 121 |
| 150, 154 | yes | 155 |
| 155 | no | 155 |
| 200, 250, 299 | yes | **none — byte-identical over all 300** |

So the machinery is exact and the divergence is prefill-vs-decode numerics, in
three independent ways:

- **N = 1 and N = 2 reproduce the baseline bit for bit.** At those two the keep
  prefill decodes exactly the shapes the baseline did (692, then 1), so nothing
  can differ — and nothing does, over all 300 frames. At N = 3 the slot takes a
  2-token ubatch where the baseline took two 1-token decodes, ggml-vulkan swaps
  the vector kernel for the matrix one, and the low bits move.
- **N = 200 / 250 / 299 are exact over every remaining frame.** A repetition
  window that was not primed with the kept codes (penalty 1.2 over the last 50)
  could not survive 100 greedy steps; it does.
- **The frames that flip are measured near-ties.** Probed with `llama-server` on
  the Arc (`--device Vulkan1`, port 8795, `n_probs` at the exact position, the
  prefix + the kept ids as a token array):

  | frame | top-2 logprob gap under a prefilled context | what happened |
  |---:|---:|---|
  | 3 | **0.017** | the keep run picked the probe's top-1 (6443), the baseline the #2 (10157) |
  | 155 | **0.0027** | the two leaders are 0.0027 nats apart |
  | 121 | 0.0050 between the two tokens actually in contention (8713 / 7653, the probe's #3 and #4) | |

  SPEC_KEEP §5.3's escape clause applies: reported, not forced to pass. The
  probe is a different llama.cpp build from the vendored one, so its ranking is
  an estimate of the gap, not of which token wins.

### 4 — determinism

```bash
./build_keep/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/keep/keep_n64.json \
    --artifacts tests/out/keep/det_{a,b} --gpu 1 --max-semantic 300     # twice
```

`prefix.npy`, `abc_tokens.npy`, `semantic.npy` identical. `plan.json` differs in
one field, `keep_prefill_seconds` (0.187 vs 0.202) — it always carried timings.

### 5 — guided + keep, the cut at `frame == N`

```bash
./build_keep/yue2 ar -m yue2-ar-q8_0.gguf --request tests/out/keep/keep_g64.json \
    --artifacts tests/out/keep/out_g64 --gpu 1 --max-semantic 300
```

```
context: 1000 tokens/slot x 3 slots (1 song + up to 2 guidance branches), batch 692
keep: 64 frames from semantic.npy prefilled (692 + 64 tokens, 0.19 s)
guidance: step 64: slot 0 relabelled as the previous branch
guidance: step 64: new positive prefilled into slot 1 (692 + 63 tokens)
guidance: step 64: blank prefilled into slot 2 (633 + 63 tokens)
guidance: step 300: previous branch dropped from slot 0 (the song is finished)
```

`guidance.json`: `[(64, true)]`. The first 64 codes equal the file. The `63` in
the branch lines is the invariant working: every sequence holds its prefix plus
`history` minus the pending token, and `kept[63]` is pending. `keep_g1.json` (N
= 1, cut at frame 1) does the same with `692 + 0` — the smallest cut there is.
`keep_cfg3.json` (keep 64, `cfg_scale: 3`) opens its blank branch at step 64,
which is SPEC §2's "a plain `cfg_scale` is allowed: its blank branch is simply
born at step N".

`--requests` with two jobs at `--parallel 2`, one of them a keep job whose
`"file"` is `"../R/semantic.npy"`: rc 0, 299 full-width steps, 0 holed, and the
path resolved against the **batch file's** directory as §2 asks.

### 6 — `--verify-sampler`

```
keep: 64 frames from semantic.npy prefilled (692 + 64 tokens, 0.18 s)
ar verify: 236 sampling steps matched the stage-5 sampler exactly
```

236 = 300 - 64: only the sampled steps are verified, which is the point, and the
run is identical to the same request without the flag.

### 7 — the request errors

`yue2-guidance`: **PASS, 83 cases** (70 before), the 13 new ones being every §2
rule that needs no file. At the CLI, all before the model loads:

```
$ yue2 ar --request err/noabc.json …
error: …: "semantic_keep" needs the score its codes were sung to, in "abc" — a freshly written score would not match them
$ … err/tpl.json
error: …: "semantic_keep" with "abc_template" is not supported: the kept codes were sung to one exact score, …
$ … err/under.json
error: …: "guidance" entry 1: guidance frame 99 is inside the kept 100 frames
$ … err/nof.json
error: …: "semantic_keep": needs a "frames": how many leading codes of the file to keep
$ … err/unk.json
error: …: "semantic_keep": unknown key "from" (file, frames)
$ … err/missing.json
error: …: "semantic_keep": cannot open tests/out/keep/err/../R/nope.npy
$ … err/toolong.json
error: …: "semantic_keep": "frames" 5000 exceeds the 300 codes in semantic.npy
$ … err/overcap.json --max-semantic 200
error: …: "semantic_keep": "frames" 250 leaves nothing to sample under the 200-step semantic cap
$ … err/notcodec.json          (fed prefix.npy instead of semantic.npy)
error: …: "semantic_keep": prefix.npy[0] = 151643 is not a codec index in [0, 32768)
$ … err/f32.json               (a float32 .npy)
error: …: "semantic_keep": only little-endian '<i4' npy input is supported, got: {'descr': '<f4', …}
```

The last two come out of `npy::load_i32` and the NAR's own range test, which is
the shared reader §2 asks for. The `../R/nope.npy` in the third-from-last is the
relative path resolving against the request file's directory.

### 8 — what it costs

Arc, `yue2-ar-q8_0.gguf`, `ext.json` (external score, so the run is all semantic):

| case | command | semantic seconds |
|---|---|---:|
| sampling 1000 steps | `--max-semantic 1000` | **7.25** |
| keeping 1000 frames | `keep_n1000.json --max-semantic 1001` | **0.44**, of which 0.43 s is the keep prefill |

**x17**, and it is one prefill of 1691 tokens (the 692-token prefix plus 999 kept
codes), chunked by `decode_feed` at `n_batch` 692 into 692 + 692 + 307 — the
case SPEC §3's "chunked by the batch size like any long
prefix" is about. `n_batch` is deliberately *not* grown for the kept codes: it is
shared with the other jobs of a batch, and growing it would move their numerics.

## 10b. Deviations

| # | what | why |
|---|---|---|
| K1 | §2's "when the file's last code is the end marker it cannot be kept" needed no code of its own. | A `semantic.npy` the AR stage writes never holds one — `apply()` drops `MUSIC_END` before `finish_job` converts to codes — and a `MUSIC_END` (151852) in the file fails the codec-index test anyway. The SPEC's own "(if the file stores one)" says as much. |
| K2 | `--dump-logits` on a `semantic_keep` request is a request error, not a dump at frame N. | The flag dumps the *first* semantic step, and reaching frame N means prefilling the whole kept stretch — which is what `--artifacts` is. Same shape as **G3**. |
| K3 | The log line's `(P + N tokens)` is the logical content. The slot receives `P + N - 1`; `kept[N-1]` rides in the first lockstep batch. | That one token pending in `q.next` is what makes the primary and every guidance branch line up with no change to `prefill_branch` or the lockstep loop (§0b). Printing `P + N - 1` would describe the implementation instead of the request, and the SPEC asks for the `P + N` form. |
| K4 | `"frames" >= max_semantic` is a request error, which §2 does not list. | §3 says kept steps count against every per-step limit. Without the check, `frames == max_semantic` would sample one step past the cap and write N+1 codes. Needing it moved `run_ar_batch`'s `Sampling` block above the validation loop (pure reorder; it reads only `p`). |
| K5 | §5.3 (teacher forcing at N = 100) does not pass as written: the continuation flips at frame 155. | The SPEC's own escape clause. Measured: a 0.0027-nat top-2 gap there, and the same mechanism reproduces at N = 1/2 (bit-identical) and N = 200/250/299 (bit-identical). See §3 above. |
| K6 | `kept_frames` / `keep_prefill_seconds` are on **both** phases' timing blocks, the abc phase included, which never keeps anything. | Same reason as **G5**: they are song-level numbers and the only `json_timing()` that reaches an artifact is the abc one. |
| K7 | `semantic_keep` is not written into `request.json`. | Same rule as `guidance` and `abc_template`: `request.json` must stay loadable by `SongRequest(**request.json)`. The consequence is that a keep run's artifacts directory is not a self-contained reproducing request — `plan.json` carries `frames` and the digest, and the path is deliberately nowhere (§4). |
| K8 | The `semantic:` line of a keep run reports the kept frames as output tokens over a wall time that is mostly prefill (`1001 tokens in 0.44 s = 2278 tok/s`). | It is the existing convention — the prefix prefill has always been inside `st.seconds` — and `keep_prefill_seconds` is beside it in the timing block for anyone who wants the decode rate alone. |
| K9 | `ArJob::base_dir` is new, and for `--requests`/`--jobs` it is the **batch** file's directory, not each job's request file's. | §2 says so in as many words. The single-request paths (`yue2 ar --request`, `yue2 song --request`) use the request file's own directory, also per §2. |

## 11b. Not done / open

- `semantic_keep` with `"abc_template"`, and with `cot: "off"` (which cannot
  carry an `"abc"`), are rejected rather than supported.
- Nothing keeps the *end* of a render, or a middle slice: §2 is leading codes
  only, and the pending-token trick is what makes that the cheap case.
- `tests/regress.sh` was not extended, for the reason stage 7 gives above.
