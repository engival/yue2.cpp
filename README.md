# yue2.cpp

ggml / Vulkan runtime for pieces of the [YuE2](https://huggingface.co/m-a-p/YuE2-3B)
song-generation pipeline. Motivation: the reference implementation is PyTorch
on CUDA; on AMD (ROCm) the VAE decode alone takes ~194 s for a 3.4-minute song
and MIOpen occasionally takes the GPU down with it. ggml's Vulkan backend sidesteps
rocBLAS/MIOpen entirely.

**Provenance and warranty.** This code was written entirely by Claude (Anthropic's
Fable 5.1 model) working as a coding agent, directed by a human who set the goals,
ran the renders and listened to the results, but did not closely review the
source. Every stage has a numeric acceptance test against the PyTorch reference
(see the `SPEC*.md` / `src/STATUS*.md` pairs), and the audio is checked by ear,
but treat it as unreviewed software: use at your own risk. Issues and pull
requests are welcome; expect the maintainer to answer them with the same tool.

Status:
- **stage 1 — Oobleck VAE decoder** (`yue2-vae`, latent → 48 kHz stereo): done. [SPEC.md](SPEC.md)
- **stage 2 — ABC plan + semantic tokens** (`yue2-ar`, request → tokens, on
  libllama with a standard `qwen3` GGUF): done. [SPEC_AR.md](SPEC_AR.md)
- **stage 3 — NAR flow matching** (`yue2-nar`, tokens → `[T, 64]` latent, raw
  ggml): done. [SPEC_NAR.md](SPEC_NAR.md)
- **stage 4 — one `yue2` binary end to end** (request → FLAC, no python at
  runtime): done. [SPEC_SINGLE.md](SPEC_SINGLE.md)

Notes and measurements in [docs/](docs/); per-stage status in `src/STATUS*.md`.

Weights are not included. The converters in `convert/` build the three GGUFs
from your own copies of [`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B)
(AR + NAR, one checkpoint) and [`m-a-p/YuE2-Vae`](https://huggingface.co/m-a-p/YuE2-Vae)
(the decoder), both CC BY-NC 4.0 — see the Quick start. Code is MIT; NOTICE.md
has the lineage.

## Quick start

From a fresh clone to a FLAC. Needs libFLAC 1.5 (`pkg-config flac`), a Vulkan
driver + `glslc`, and python3; libopusenc is optional and adds `.opus` output.
All paths below are relative to the repo root.

**Tested on** one box only: Slackware64-current (Linux), an AMD Radeon RX 7900
XTX (RADV) and an Intel Arc Pro B70 (ANV), both through ggml's Vulkan backend.
**Not tested:** NVIDIA, Windows, macOS, or any backend other than Vulkan. The
code has no platform-specific parts beyond `/proc/self/exe` (used to find the
GGUFs next to the binary), so other Vulkan setups may well work; reports either
way are welcome.

**1. Build.** `GGML_VULKAN=ON` is the only option that matters (it is also the
default); llama.cpp's examples/tests/tools/server are forced off here.

```bash
git submodule update --init
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build -j8
```

Binaries land in `build/`: `yue2` (everything) plus `yue2-ar`, `yue2-nar`,
`yue2-vae`, one-line mains over the same `src/stage_*.cpp` as `yue2 ar|nar|vae`.
`llama-quantize` is **not** one of them (`LLAMA_BUILD_TOOLS OFF`) — step 4 needs
it from an ordinary llama.cpp build; see [the `yue2-ar` section](#generate-the-symbolic-plan--semantic-tokens-yue2-ar).

**2. Get the weights** (CC BY-NC 4.0, **non-commercial**, per NOTICE.md; the
GGUFs you make inherit that licence). One checkpoint holds both the AR and NAR
halves:

```bash
hf download m-a-p/YuE2-3B      # AR + NAR, model.safetensors + config.json + qwen.tiktoken
hf download m-a-p/YuE2-Vae     # VAE, model.safetensors + config.json
```

They go to `~/.cache/huggingface/hub`. The converters find them there on their
own — `convert/common.py:resolve_snapshot()` calls `snapshot_download(repo_id,
local_files_only=True)` — so `--src` is only needed for a snapshot dir
elsewhere.

**3. Converter environment** (CPU torch only; no GPU is used):

```bash
python3 -m venv venv && venv/bin/pip install -r convert/requirements.txt
```

**4. Convert, once.** `yue2 song` looks for `yue2-ar-q8_0.gguf`,
`yue2-nar-f16.gguf` and `yue2-vae-f32.gguf` next to the executable and then one
directory up, so write them to the repo root:

```bash
venv/bin/python convert/convert_ar.py  --out yue2-ar-f16.gguf    # 4,338,917,568 B
venv/bin/python convert/convert_nar.py --out yue2-nar-f16.gguf \
    --ar-gguf yue2-ar-f16.gguf                                   # 2,939,691,584 B
venv/bin/python convert/convert_vae.py --out yue2-vae-f32.gguf   #   265,437,856 B
llama-quantize yue2-ar-f16.gguf yue2-ar-q8_0.gguf Q8_0           # 2,308,448,480 B
```

The F16 AR half is only an intermediate for `song`; keep it if you want the
exact-F32 NAR path. Tensor tables and the full invocations are in
`convert/STATUS.md`, `convert/STATUS_AR.md`, `convert/STATUS_NAR.md`.

**5. Write a request.** `style` and `lyrics` are required strings; `cot`
(`off|melody|full`, default `full`), `seed` (integer in `[0, 2**63)`), `id`,
`abc`, `abc_template`, `cfg_scale`, `guidance`, `sections`, `semantic_keep`,
`handover` and `base_take` are optional.
Lyrics carry `[Section]` tags on
their own lines:

```json
{
  "style": "slow dream pop, reverb guitar, brushed drums, breathy female vocal",
  "lyrics": "[Verse]\nThe kettle sings a flat blue note\n\n[Chorus]\nStay a while, the rain is warm\n",
  "cot": "full",
  "seed": 1
}
```

**6. Render.**

```bash
build/yue2 song --request song.json --out song.flac --gpu 0
```

`--gpu N` picks the Vulkan device (and with it the NAR's flags). Measured on an
Intel Arc B70: 155 s and ~8 GB peak VRAM for 156 s of audio; a 7900 XTX is
roughly twice as fast per GPU stage (NAR 52 s vs 92 s, VAE 7.5 s vs 15 s). The
**first** Vulkan run spends ~16 min compiling RADV pipelines; later runs start in
under a second.

**Re-skin a song you like.** With `--artifacts DIR` a render leaves its score in
`DIR/score.abc`. Put that text in a new request's `"abc"` and the AR skips
writing a score and sings the given one (ignored with `cot: "off"`):

```json
{ "style": "orchestral rock, distorted guitars, gravelly male vocal", "lyrics": "…same lyrics…", "abc": "…score.abc…" }
```

The score carries the tune, the timing and much of the vocal delivery; the
style decides the voice and the band. In our tests the re-skinned song came out
within 0.3 % of the original's length with the same melody and phrasing in
another genre, and ~25 % faster (no score phase). Nothing after the AR can do
this: the NAR follows the semantic tokens and ignores both its noise seed and a
changed style text, audibly. `yue2 ar --prefix-only` writes just `prefix.npy`
for a request that carries its `"abc"` (tokenizer only, no GPU).

**Let the model write part of a score.** `"abc_template"` is the middle between
no score and a whole one: score text in which a `%%yue2-gen` line is a hole the
model fills. Everything else is fed to it verbatim, so a written line is
conditioned on every line above it — which is what makes the accompaniment
answer the vocal it sits under instead of being improvised over bars of rests:

```
V: Vocal
"C"c8e8g8e8|"G"d8B8G16|"Am"c8e8a8e8|"G"G32|
V: Ins
%%yue2-gen
```

A hole must be written as a body line with the right number of bars — by
default the bar count of the nearest body line above it, or `%%yue2-gen bars=N`.
A line that comes out wrong is rolled back and drawn again up to four times,
then filled with rests (`ZN|`). Lines between `%%yue2-primer-begin` and
`%%yue2-primer-end` are fed as context and then dropped from the score: the
model is causal, so that is how an intro gets to see the verse it introduces.
In practice it then quotes that verse, and the intro sounds like an interlude;
`yue2` warns when a template carries a primer. The chord symbols on the intro's
resting vocal line already give the hole the tune's harmony bar by bar, which
turns out to be the tether that matters.
A template whose last line is `%%yue2-continue` is an opening rather than a
form: the lines above it are fed as the score's beginning and the model writes
the rest freely, as it would with no score at all. The opening need not be
well-formed; the result shows what the model makes of it.
`%%yue2-chords` in place of a Vocal body line asks for the harmony of a given
accompaniment: the model is shown the Ins line under it first, writes a line of
rests carrying chord symbols, and that line is put back above the Ins line, as
if it had been written in order. An intro handed over as Ins lines only gets
its chords that way, and everything after it is then written with chords too.
`score.abc` and the rest of the artifacts hold the finished score with no
directives and no primer; `request.json` records the score the holes produced as
a plain `"abc"`, so the artifacts directory is a request that reproduces the
song, and the template as given is kept beside it as `template.abc`.
`result.json` gains a `template` block counting the holes, retries and
rest-fills. `"abc_template"` is mutually exclusive with `"abc"` and needs `cot`
`full` or `melody`.

**Push the semantic phase around: `cfg_scale` and `guidance`.** The codec phase
can be decoded beside up to two *negative* sequences carrying the same generated
history under a different prefix, and the blend of the three is what the sampler
sees:

```
L = B + sum_i w_i(t) * (B - N_i)
```

`"cfg_scale": c` is the reference's classifier-free guidance and needs nothing
else: one negative branch holding the instruction and the exact score but no tags
and no lyrics, at the constant weight `c - 1`, for the whole phase. (`cot: "off"`
defaults to `1.01`, as the reference does; say `"cfg_scale": 1.0` for the plain
single-sequence decode.)

`"guidance"` is the same machinery with time-varying weights and a positive
prefix that can change part-way through — which is how a song changes band or
gains a voice mid-stream:

```json
"guidance": [
  { "frame": 3400,
    "style": "brass band, snare rolls, bright trumpet lead",
    "against": {
      "previous": [[0, 11], [60, 11], [85, 3]],
      "blank":    [[0, 2]]
    } }
]
```

- `frame` is the semantic step the entry takes effect at — **25 frames = 1 s of
  audio**. Entries are strictly increasing in `frame`; one that the song never
  reaches is reported as `"reached": false` rather than being an error.
- `style` replaces the tags of the positive prefix (same lyrics, same score). It
  is optional: without it the entry only changes the curves. It cannot sit on an
  entry at frame 0 — from the first frame on, the request's own `style` is that.
- `against.previous` pushes away from the prefix that was in force *before* this
  entry — it needs a `style` to have something to push away from. `against.blank`
  pushes away from `cfg_scale`'s negative prefix.
- A curve is a list of `[offset, weight]` with `offset` in frames after the
  entry's `frame`: linear in between, the first weight before the first keyframe,
  the last weight after the last, and two keyframes at one offset a step. A curve
  that reaches zero and stays there drops its branch, and the song goes back to
  full speed.
- A new entry replaces the previous entry's curves; a branch it does not name has
  weight 0 from there on.

Swapping the tags mid-stream on its own does nothing audible — after ~3 s of
audio the codec history dominates the next token and the tags barely move it.
Amplifying what is left of their influence is what works. Starting points, from
listening tests on one song (not laws):

| intent | `previous` | `blank` |
|---|---|---|
| the band changes at a section | `[[0, 5]]` | `[[0, 2]]` |
| a new lead voice enters | `[[0, 11], [60, 11], [85, 3]]` | `[[0, 2]]` |
| gradual colouring | `[[0, 0], [750, 5]]` | — |

Start the entry about 1.4 s (35 frames) **before** the bar line of the section
that should open in the new style; a push that starts inside a sung phrase
garbles it. A strong `previous` push alone will bring in a voice the recording
never held but destroys the words with it — the small `blank` push beside it is
what keeps them intelligible.

The normalised block, with a `reached` flag per entry, is written to
`guidance.json` in the artifacts directory and into `plan.json`; `request.json`
keeps only `cfg_scale`, so it stays loadable by the reference. Guidance needs
`--parallel 1` (the branches are the other KV streams) and does not combine with
`"abc_template"` yet. Only the semantic phase is guided; the score phase never is.

**What the push actually did: `--guidance-trace`.** A flag on `yue2 ar`,
`yue2 song` and `yue2 batch`. A guided job run with it and `--artifacts DIR` writes
`DIR/guidance_trace.npy` beside the rest: float32, one row per semantic step that
had a live branch, eight columns. It costs about 0.5–0.8 ms a step (two or three
softmaxes over the 32 769 ids the semantic sampler can visit) and changes
nothing else — the same request, seed and card sing the same `semantic.npy`
traced or not. Without `--artifacts`, or on a request with no guidance, it writes
nothing and says so.

```python
import numpy as np
t = np.load("DIR/guidance_trace.npy")      # [steps, 8] float32

step      = t[:, 0]        # the semantic step; 25 = 1 s of audio
w_prev    = t[:, 1]        # the weights that were in force, 0 where the
w_blank   = t[:, 2]        #   branch is not live (its curve had dropped it)
tv_prev   = t[:, 3]        # how far each branch's own distribution stands
tv_blank  = t[:, 4]        #   from the positive one — NaN where not live
same_top  = t[:, 5]        # 1 where `previous` still wants the same token
tv_blend  = t[:, 6]        # how far the blend moved what the sampler saw
logp      = t[:, 7]        # log p of the token drawn, under the *unguided* model

print("pushed hardest at frame", int(step[tv_blend.argmax()]),
      "- the unguided model gave that token", float(np.exp(logp[tv_blend.argmax()])))
print("frames where the branches had parted:", int(np.nansum(same_top == 0)))
```

A `tv_blend` near 1 with a very negative `logp` is a step the guidance carried
on its own — musically the interesting ones, and, if a whole run looks like
that, the sign that the weights are too high for the lyrics to survive.

`scripts/trace_summary.py DIR/guidance_trace.npy` prints the same trace as a
per-5-seconds timeline with the hardest-pushed moments as timestamps.

**Keep the start of an earlier render: `semantic_keep`.** A guided song is still
sampled from frame 0, so feeding the same score back gives a *new* performance.
`"semantic_keep"` does the opposite — it forces the codes of a render you already
like as history and only changes what comes after the cut:

```json
"abc": "X:1\nM:3/4\n…the score that render sang…",
"semantic_keep": { "file": "earlier/semantic.npy", "frames": 3400 },
"guidance": [
  { "frame": 3400,
    "style": "uptempo ska, offbeat guitar, horn section, bright tenor vocal",
    "against": { "previous": [[0, 11], [60, 11], [85, 3]], "blank": [[0, 2]] } }
]
```

- It is not automatically the better way to restyle a render you like. Guidance
  only pushes where the old and the new tags disagree *given the history*: a kept
  history that lacks what the new style expects (no drums under a voice the model
  associates with a beat) gives the transition the most to fight, and the band may
  be rebuilt along with the voice. A fresh guided take, whose own band suits both
  halves, often cuts cleaner. Try fresh seeds first; keep is for A/B against a known
  first half and for skipping its sampling time.
- `file` is a `semantic.npy` as `yue2 ar|song --artifacts` writes it; a relative
  path resolves against the request file (for `--requests`/`--jobs`, against the
  batch file). `frames` is how many of its leading codes to keep — required,
  there is no implicit "all of it".
- It needs the score those codes were sung to, in `"abc"`: a freshly written
  score would not match them. `"abc_template"` is out for the same reason.
- Frames `0 .. N-1` are not sampled and cost one prefill instead (1000 frames in
  0.4 s on the Arc, against 7.3 s to sample them). Everything else counts them as
  steps: the repetition window, the frame cap, the end token.
- The run's `semantic.npy` is the kept codes followed by the sampled ones, so the
  NAR and VAE need nothing new, and `plan.json` records `frames` and a SHA-256 of
  what was kept (not the path — artifacts stay relocatable).
- Combine it with `guidance` at `"frame": N` for "keep this take up to the cut,
  change the band from there". A `guidance` entry *below* `N` is a request error.

**A style per section: `sections`.** `guidance` addresses semantic *frames*, and
counting frames by hand to land on a section is tedious. `sections` names the
score's own section labels instead and the engine works out the frame:

```json
"abc": "X:1\nM:3/4\n…the score…",
"sections": [
  { "section": "verse", "nth": 2,
    "style": "uptempo ska, offbeat guitar, horn section, bright tenor vocal" },
  { "section": "chorus", "nth": 2, "style": "marching band, snare, brass",
    "lead_frames": 35,
    "against": { "previous": [[0, 5]], "blank": [[0, 2]] } }
]
```

The finding behind it: **in the semantic phase the score decides who sings.** A
vocal line moved an octave down from one section on brought in a male singer at
that bar on every seed tried, even with "female vocal" still in the tags — the
tags then colour the voice the score has already chosen. So a tag change the
score contradicts (a deep voice over a line written high) is a fight: coin-flip
takeovers, wobble, garbled words under strong guidance. **Match the voice tag to
the register the score actually writes**, and use
[docs/SCORE_RECIPES.md](docs/SCORE_RECIPES.md) to move that register if you want
a different singer.

- `section` is a label as the score writes it after the `% `. Planner-written
  scores use a closed vocabulary — `intro`, `verse`, `chorus`, `interlude`,
  `bridge`, `outro` — and `nth` (1-based, default 1) picks the occurrence,
  counted over every label of that name, so the `% interlude` the planner
  inserts between two verses does not shift the count.
- `style` is required and is the full tag string from that label on; it replaces
  the request's `style`, which covers everything before the first entry.
- `lead_frames` (default 35, `0..250`) moves the semantic swap that many frames
  *before* the section's bar line: the render runs ahead of the score clock by
  about a second, and the swap wants to land in the breath before the phrase.
- `against` is exactly a `guidance` entry's, with the same presets. Left out, the
  entry is a plain swap: the new prefix is prefilled, the old sequence is dropped,
  and the song still owns one KV stream — so a plain-swap request decodes beside
  others at `--parallel > 1`, where `guidance` and an `against` do not.
- Entries are in the order the song plays them. They resolve against the score
  that was actually written, and **a label the score never writes is reported,
  not an error**: `sections.json` in the artifacts directory records each entry
  with `reached`, its line number in `score.abc`, its bar, that bar's start in
  seconds and the frame it became. The compiled entries are in `guidance.json`
  as always, so `--guidance-trace` and the rest work unchanged; `request.json`
  carries neither, so it stays loadable by the reference.
- It combines with `semantic_keep` — "keep this take up to the second chorus,
  change the band from there" — and an entry that lands inside the kept frames is
  a request error, as a `guidance` frame there is. `abc_template`, `cot: "off"`,
  a `guidance` block and a `cfg_scale` other than 1 are all errors beside it.
- When the engine writes the score (no `"abc"`), the label line is written under
  the old tags and the score is re-prefilled under the new ones from there. Do
  not expect much of that: the score already written outvotes the tags much as
  the audio history does in the semantic phase, so a swap late in a long score
  conditions the rest of it only weakly — the same tags on a *fresh* score give a
  different key, tempo and register entirely. The frame arithmetic is the point
  of the score phase; the audible change is the semantic swap.
- Frames per bar come from the score's `M:` and `Q:` (25 Hz, so 4/4 at
  `Q:1/4=120` is 50 frames a bar); inline `[M:…]` changes are honoured, and the
  bar clock follows the `V: Vocal` voice.

**Hand the song over to another style's take: `handover`.** Swapping the tags
mid-song changes the singer when the score agrees, but a *genre* change needs the
audio history to change too — and pushing tags against the history is
seed-dependent. What works every time is to render **one full take per style from
the same score** and hand the song from one take to the next at a cut:

```json
"base_take": "out/base",
"handover": [
  { "section": "chorus", "nth": 1, "style": "…tags…" },
  { "section": "verse",  "nth": 3, "take": "takes/reggae", "seconds": 1 },
  { "section": "verse",  "nth": 4 }
]
```

At each cut the incoming style's renderer is run over a forced history — its own
take up to a few seconds before the cut, then the last few seconds of the song so
far — and samples on from there. Its long memory is already the new style, the
short intrusion ties it to where the song is (phrase, beat, singer), and it
recovers into its own style within a bar or two. Chains work (A→B→C→D→A).

- **Audition the takes first.** The handover can only be as good as the take it
  hands to: render each style on its own, listen, and hand over to the ones that
  work. `--artifacts DIR` saves every take the engine renders as an ordinary
  artifacts directory `DIR/take_<k>/`, so it can be auditioned with `yue2 nar` +
  `yue2 vae` and reused later through `take`.
- The two settings that matter: `"seconds"` (default 5) is how much of the song
  so far the incoming renderer is shown. **5 s is a gradual blend** — the band
  changes first, the singer at the next natural entry. **1 s on a section
  boundary is a hard cut.** The range is 0.2–30.
- Where: `section` + `nth` exactly as in `sections` (same labels, same
  `lead_frames`), `"frame": N` in the base take's own timeline, or `"at"` — the
  same place as a time in the base take's **audio**, the thing you read off a
  player: `78.6`, `"78.6"`, `"1:18.6"` or `"1:18"` (25 frames = 1 s, and no lead
  is subtracted — you already heard where it is). One of the three per entry;
  `nth` and `lead_frames` belong to a `section`. Cuts must be strictly
  increasing, and the array is the order the song plays them in.
- What takes over: `style` (the engine renders that take itself, once per
  distinct style), `take` (an artifacts directory of an earlier render — its
  `score.abc` must be byte-identical to the base score, and its tags are read
  from its `request.json`), or neither, which is back to the request's own style
  and its base take.
- `"base_take": "dir"` uses an earlier render as the base instead of rendering
  it, and supplies the score when the request has no `"abc"`. Without it the
  engine renders the base take first, as `take_0`.
- **Two takes of one score do not run at the same pace** — one leads the other by
  up to a couple of seconds, and the lead drifts over the song — so the incoming
  take is read at its own clock. `"offset"` is how many frames it runs ahead at
  the cut; the default `"auto"` measures it from the tokens (at the right lag two
  takes of one score share 2–3 % *identical* tokens over a 30 s window and next
  to none at every other lag). When no lag stands out the measurement is not
  used: the leg is handed over at **offset 0** with a warning, and
  `handover.json` records `confident: false` and `measured` — the lag that was
  not believed. A lag measured out of noise skips real song; 0 does not. An
  offset you give as an integer is never second-guessed.
- `handover.json` also records, per entry, the label and bar it resolved to (or
  the `at` as you wrote it), the cut
  frame, which take took over, the intrusion and how many frames of the final
  song that leg contributed. An entry whose label the score never writes, or
  whose cut is past the end of the song, is reported there rather than being an
  error.
- The model is loaded once and every take and leg decodes through it, but each
  one is its own generation: a take is bit-identical to rendering that request
  with `yue2 ar` by hand. Expect a take to cost a full song's sampling time and a
  leg rather less (it starts from a forced history and stops after the next cut).
- `guidance`, `sections`, `semantic_keep`, `abc_template`, `cot: "off"` and a
  `cfg_scale` other than 1 are all errors beside it, and so is a batch of more
  than one song (`yue2 batch --jobs` / `yue2 ar --requests`) — render a handover
  on its own with `yue2 song` or `yue2 ar`.
- Everything the request names is read and everything that can be decided by
  arithmetic is decided **before the first take is rendered**: a take that is not
  there, is not of this score or does not reach a cut, a cut with less than the
  intrusion before it, a `--max-semantic` that would leave a leg nothing to
  sample. A leg that cannot be played is an error, never a silently short song;
  the one thing that is reported instead is a cut past the end of the song.

A worked request is in [docs/examples/handover.json](docs/examples/handover.json);
which styles to hand over to, and in what order, is
[docs/COOKBOOK_STYLE_CHANGE.md](docs/COOKBOOK_STYLE_CHANGE.md).

Transposing a score, moving the singer's register, stripping the chords, resting
out or re-writing the accompaniment, reading a seed's score before rendering it:
[docs/SCORE_RECIPES.md](docs/SCORE_RECIPES.md), with `scripts/abc_transpose.lua`.

Several songs at once: [`yue2 batch`](#render-several-songs-yue2-batch). Running
the stages separately: [`yue2-ar`](#generate-the-symbolic-plan--semantic-tokens-yue2-ar),
[`yue2-nar`](#flow-match-the-semantic-tokens-into-a-latent-yue2-nar),
[`yue2-vae`](#decode).

## Render a whole song (`yue2 song`)

```bash
build/yue2 song --request song.json --out song.flac --artifacts out/song
```

One process: AR on libllama → NAR in raw ggml → VAE. `--seed N` overrides the
request's seed for both the AR sampling and the noise; `--noise FILE` replaces
the generated noise; `--gpu 0|1` picks the Vulkan device and with it the NAR's
flags (AMD `--flash-attn`, Intel one untiled query chunk — by device *name*, and
falling back to the default tiling on songs whose score buffer would not fit).
`--steps N` is the NAR's ODE step count: the default is 16, the reference's
`ode_steps` is 32, and the NAR's time scales with it.
`--nar-f32` trades the fp16-staged matmuls for the exact ones; because
ggml-vulkan reads that switch once per device init, it applies to the AR as
well, whose logits then shift — so `--nar-f32` yields a *different song*, not the
same song rendered more precisely. The three GGUFs
default to `yue2-ar-q8_0.gguf`, `yue2-nar-f16.gguf`, `yue2-vae-f32.gguf` next to
the executable, then one directory up. `--artifacts` writes the same file set the
reference pipeline did (`plan.json`, `prefix.npy`, `semantic.npy`, `nar_noise.npy`,
`latent.npy`, `config.json`, `result.json`, …); without it a temporary directory
is used and removed.

**Every stage runs in one process, at one Vulkan precision.** ggml-vulkan fixes
matmul operand staging at device init (`GGML_VK_DISABLE_F16`/`_COOPMAT`, read
once), so the VAE decodes at whatever precision the NAR is using — fp16-staged
by default. That decode is 58.5 dB from the exact-F32 one (~40 dB at the worst
burst) and a listening test could not separate the two, so the exact path is a
numeric reference, not a render default. To get it, run the stage on its own
against the latent the render left behind:

```bash
build/yue2 vae -m yue2-vae-f32.gguf -i DIR/latent.npy -o exact.flac --device vulkan --gpu 0
```

`song` and `batch` refuse `--vk-f16-matmul` (and the matching `request.json`
key) rather than silently downgrading — the error names that command. `--nar-f32` is the one switch that does move the whole process,
VAE included, onto the exact pipelines.

Measured on the Intel Arc B70, 156 s of audio: 155 s end to end (abc 15 s,
semantic 37 s, NAR 88 s, VAE 14 s), ~8.7 GB peak VRAM — the VAE figure is from
the exact-F32 child. The two precisions cost about the same: on this card exact
is ~6 % faster (16.2 s vs 17.1 s fp16-staged, 191 s song), on the AMD the
fp16-staged path is the quicker one (9.2 s vs 9.5 s). Details, the per-stage
goldens and the Q8_0/F16 prefix measurement are in
[src/STATUS_SINGLE.md](src/STATUS_SINGLE.md).

**Tags in the FLAC.** A `.flac` from `song` or `batch` carries Vorbis comments
(FLAC's native tags; players and `metaflac` read them). What goes in is what the
request names, plus the words that are sung:

```json
"tags": { "TITLE": "My Song", "ARTIST": "Someone", "GENRE": "lullaby" }
```

- `tags` is free-form `NAME: "text"` (names printable ASCII without `=`, text
  UTF-8); a bad block is a request error before anything renders.
- Written without being asked: `LYRICS` (the request's lyrics, unless `tags` has
  its own), `ENCODER=yue2.cpp`, and `YUE2_ID` — the first 16 hex digits of the
  SHA-256 of `semantic.npy`, which finds the artifacts directory a file came from
  without saying anything about it.
- Never written: the style, the seed (it repeats a song only on the same card,
  build and batch shape — `semantic.npy` is the durable record), `id`, paths.
- `--no-tags` writes none: the file as earlier versions made it. The standalone
  `yue2 vae` has no request and writes none.

**Opus output.** `song`, `batch` and `vae` pick the format from the output name,
so `--out X.opus` writes an Ogg Opus file instead of FLAC (and `X.wav` a float
WAV, as before). `--opus-bitrate KBPS` sets the rate, 16 to 510, default 160; the
tags above are written as Vorbis comments there too, and with `--no-tags`
libopusenc still puts its own vendor string in the file. Opus is lossy, so it is
for the copy you listen to or upload, never for the exact and regression paths —
those stay FLAC. Nothing is lost by choosing it late either: keep the render's
`--artifacts` directory and `yue2 vae -m yue2-vae-f32.gguf -i DIR/latent.npy -o
X.flac` decodes the lossless file again whenever you want it.

Opus support is optional at build time: it needs libopusenc (pkg-config
`libopusenc`), CMake reports which way it went (`yue2: Opus output ...`), and a
build without it — or one configured with `-DYUE2_OPUS=OFF` — refuses an `.opus`
output name while it parses its arguments, before any model loads.

`yue2 noise --seed N --frames T -o noise.npy` exposes the NAR's noise generator
(`std::mt19937_64` + Box–Muller, no `<random>` distribution, so one seed gives
the same bytes everywhere); torch seed compatibility is explicitly not a goal.

## Render several songs (`yue2 batch`)

```bash
build/yue2 batch --jobs jobs.json --parallel 4 --gpu 0 --summary results.json
```

```json
[
  { "request": "a/request.json", "out": "a.flac", "artifacts": "out/a", "seed": 831001 },
  { "request": "b/request.json", "out": "b.flac" },
  { "request": "c/request.json", "out": "c.flac", "noise": "c/noise.npy" }
]
```

`request` and `out` are required; `artifacts`, `seed` and `noise` are optional
and mean what the `yue2 song` flags of those names mean. **Paths are relative to
the working directory, not to the JSON file.** Two jobs may not share an `out` or
an `artifacts` path. Everything else — the GGUFs, `--device`/`--gpu`, `--steps`,
`--nar-f32`, the sampling limits — is per batch.

Only the AR stage is batched, and that is where the time is: it decodes one token
per `llama_decode` per song, which leaves the card mostly idle, so up to
`--parallel` songs share each step. The NAR and the VAE are already
compute-bound, so they run per song, in file order, after the whole AR is done
and its context is freed — peak VRAM is `max(AR batch, one NAR)`, not the sum.

`--parallel` costs KV cache: **112 KiB per token per sequence** (28 layers × 8 kv
heads × 128 × K+V × 2 B), about 1.5 GiB per slot at a full-length song's ~13.8k
context, printed before anything is allocated. 4 is the default because that is
roughly where the AR batch stops being cheaper than the NAR peak. Jobs are taken
in file order and a slot is refilled as soon as its song finishes, so
`--parallel` is a ceiling on concurrent songs, not a wave size.

`--summary FILE` writes one JSON array of `{out, status, error, audio_seconds,
e2e_seconds}` so a driver can read the outcome without parsing the log; per-job
log lines are prefixed `[k/N name]`. While the AR loop runs it prints one
unprefixed `ar: 84 s, 3 active (1 abc, 2 semantic), 12345 tokens, 486 tok/s`
line every 10 s — batch-level, never per step, and absent entirely from a batch
that finishes inside the first 10 s. stdout is line-buffered, so a driver reading
a pipe sees all of this as it happens. `--continue-on-error` turns a bad request or
a failed VAE into a recorded failure and a non-zero exit instead of stopping the
batch; anything else (including a NAR failure) is still fatal.

`yue2 ar -m AR.gguf --requests jobs.json` is the same list for the AR stage
alone, for a caller that runs the NAR and VAE elsewhere; there `out` is ignored
and `artifacts` is required.

**Reproducibility.** A `seed` *selects* a song; it is not a promise to re-create
one. A batched step runs the matmuls with N rows instead of one, and ggml-vulkan
picks different kernels and reduction orders for different shapes, so
per-sequence logits differ from single-stream logits in the low bits — and
sampled decoding turns that into a different, equally valid song as soon as the
top-p cut or the uniform draw lands inside the noise. Exact re-creation is the
job of the artifacts instead: `semantic.npy` and `latent.npy` from
`--artifacts DIR` re-render bit-identically through `yue2 nar` and `yue2 vae` on
the same card, with no seed involved. The same seed with the same card, build,
`--parallel` and batch shape will in practice repeat a song — treat that as a
convenience, not a contract. A guided song (`cfg_scale`, `guidance`) decodes
batches of two or three rows for as long as a branch is live, so the same rule
applies to it: a seed repeats it only together with the same guidance block. The
steps before the first entry are single-row and match the unguided song exactly.

## Decode

```bash
build/yue2-vae -m yue2-vae-f32.gguf -i latent.npy -o song.wav --device vulkan --gpu 0
```

(or `build/yue2 vae …` — same flags.) Every stage runs on Vulkan device `--gpu N`
(default 0) and never falls back to the CPU: `--cpu` (= `--device cpu`) is the
only way onto it, and a missing Vulkan device is an error. An `-o` ending in `.flac` is written as
24-bit FLAC through libFLAC instead of float WAV; its integer samples are
identical to what `soundfile`'s `PCM_24` wrote for the reference pipeline. One
ending in `.opus` goes through libopusenc at `--opus-bitrate KBPS` (default 160),
when the build has it.

`latent.npy` is the `[T, 64]` float32 array the reference pipeline writes with
`--artifacts` (also accepts `[1, 64, T]`). Output is 48 kHz stereo float WAV,
`1920·T − 64` samples. `--npy out.npy` dumps the unclamped `[2, samples]` array
for numeric comparison; `--device cpu` is bit-exact with the torch reference
(tiling included), Vulkan is within ~65–70 dB (see `src/STATUS.md`).

Measured, 164 s song: Vulkan 7900 XTX 7.5 s, Intel Arc B70 15 s, CPU (i9-11900K)
157 s. torch + MIOpen on the same AMD card: ~194 s.

## Generate the symbolic plan + semantic tokens (`yue2-ar`)

```bash
convert/convert_ar.py --src ~/.cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/<hash> --out yue2-ar-f16.gguf
llama-quantize yue2-ar-f16.gguf yue2-ar-q8_0.gguf Q8_0
build/yue2-ar -m yue2-ar-q8_0.gguf --request song.json --artifacts out/song --device vulkan --gpu 0
```

`--src` can be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache. `llama-quantize` (and `llama-tokenize`, used by
`convert/check_tokenizer.py`) are **not** built by this repo's own CMake —
`CMakeLists.txt` sets `LLAMA_BUILD_TOOLS OFF` on purpose, to keep `yue2-ar`
and `yue2-vae` small single-binary builds instead of pulling in `common` and
`cpp-httplib`. Get them from a separate, ordinary llama.cpp build instead:

```bash
git clone https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
cmake -B /path/to/llama.cpp/build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /path/to/llama.cpp/build -j --target llama-quantize llama-tokenize
```

(any llama.cpp build with `LLAMA_BUILD_TOOLS=ON` — including this project's
own `llama.cpp/` submodule built standalone that way, or a prebuilt release
tarball — works; the quantize/tokenize formats are stable across versions.)

`song.json` is the reference pipeline's request format (`style`, `lyrics`,
optional `cot`, `seed`). The artifacts dir gets `score.abc`, `abc_tokens.npy`,
`prefix.npy`, `plan.json`, `plan_manifest.json`, `semantic.npy` in the exact
layout the Python pipeline's `SymbolicPlan.load()` verifies, so the reference
NAR can resume from them. Q8_0 on the 7900 XTX: ~200 tokens/s in both phases
(torch ROCm: 42 / 33). The first Vulkan run of the 3B spends ~16 min compiling
RADV pipelines; later runs start in under a second.

## Flow-match the semantic tokens into a latent (`yue2-nar`)

```bash
convert/convert_nar.py --src ~/.cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/<hash> --out yue2-nar-f16.gguf
build/yue2-nar --ar yue2-ar-f16.gguf -m yue2-nar-f16.gguf \
    --artifacts out/song --noise noise.npy --steps 32 --device vulkan --gpu 0
```

Reads `prefix.npy` + `semantic.npy` from the `yue2-ar` artifacts dir and writes
`latent.npy`, exactly what `yue2-vae -i` takes. `--noise FILE` or `--seed N`
supplies the noise (with `--artifacts` and neither flag, the seed comes from
`request.json`).

On the **exact-F32** path use the F16 AR half, not Q8_0: its K/V feeds all 64
velocity evaluations unchanged and Q8_0 costs ~55 dB there. On the **fp16-staged**
fast path the staging error (~33 dB) already dominates and Q8_0 is free —
measured 33.44 dB vs F16's 32.71 on the full song — which is why `yue2 song`
loads one Q8_0 file for both stages.

Speed/accuracy flags (SNR is against the torch f32 CPU reference; the torch
ROCm bf16 artifact is itself 35 dB from it, so ~30 dB is inaudible here):

| flag | what it does |
|---|---|
| `--kv-f16` | F16 K/V cache. Halves it, and puts the attention matmuls on ggml-vulkan's f16 pipelines — the difference between 8 and 27 TFLOP/s on an Arc B70. |
| `--vk-f16-matmul` | Skips the `GGML_VK_DISABLE_F16`/`COOPMAT` guard, so Vulkan stages matmul operands as f16. Big speedup, ~33 dB instead of ~66. |
| `--query-chunk N` | Query tile for the materialized-score path. `8192` (one tile) is fastest and needs 2.8 GB of arena at `N = 4111`; `512`–`1024` trade ~10 % for a quarter of that. |
| `--flash-attn` | `ggml_flash_attn_ext` instead of materialized scores. Wins on the 7900 XTX, loses badly on the Arc (Intel's scalar FA kernel). |
| `--weights f32` | Widen F16 weights at load. Needed only on `--device cpu`, where ggml's F16 `mul_mat` rounds the activations too. |

Measured on the 164 s `alley_swing_s1`, 32 steps: Arc B70 92 s with
`--kv-f16 --vk-f16-matmul --query-chunk 8192` (32.7 dB) or 252 s on the exact
F32 path (66.0 dB); 7900 XTX 52 s with `--flash-attn --vk-f16-matmul` (29.3 dB)
or 124 s exact. torch ROCm bf16 on the same AMD card: 46 s. Details in
`src/STATUS_NAR.md` and `src/STATUS_NAR_PERF.md`.

### Trying a NAR LoRA (`--nar-lora`)

`yue2 song`, `yue2 batch` and `yue2-nar` take `--nar-lora LORA.safetensors[:S]`,
repeatable, and fold the adapter into the NAR weights while they are read — no
new GGUF, no graph change, no per-step cost:

```bash
build/yue2 song --request song.json --out song.opus --nar-lora LORA.safetensors:0.8
```

`S` is the strength (default 1.0). Per weight the merge is `W += S·(B @ A)`, and
for the `vae2llm` / `llm2vae` full replacements some adapters carry,
`W += S·(W_new − W_base)`. Both are deltas against the base, so several
`--nar-lora` flags stack in any order and `:0` is exactly the stock model. Each
tensor is widened to F32 once, takes every adapter's delta and is narrowed back
once; tensors no adapter names are not touched at all, so a run without the flag
is bit-identical to one from a build that has never heard of LoRA. It is a
property of the model, not of a request: one setting covers a whole `batch`, and
`config.json` records the file, the strength and its SHA-256.

Only the **plain** adapter layout is read: `layers.{i}.nar_self_attn.{q,k,v,o}_proj`
and `layers.{i}.nar_mlp.{gate,up,down}_proj` as `lora_A` `[r, in]` + `lora_B`
`[out, r]`, plus optional `vae2llm.*` / `llm2vae.*` replacements; `F32`, `F16`
and `BF16` all load. A fused ComfyUI file (`diffusion_model.…`, block-diagonal
`qkv_proj` / `gate_up_proj`) is refused with a message saying so — use the plain
file. The AR half goes through libllama and is a separate job; this flag does not
touch it.

Once an adapter is a keeper, bake it in instead of merging it every load:

```bash
convert/convert_nar.py --out yue2-nar-lora.gguf --lora LORA.safetensors:0.8
```

Same merge rule, so the baked GGUF and the runtime merge agree to ~90 dB on the
latent (they differ only in float summation order and in whether the base was
rounded to F16 before the delta or after). `src/STATUS_LORA.md` has the numbers.

The layout above is the one published by
[Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4),
whose `nar_lora_joint_v*.safetensors` files this was built and tested against
(the plain ones, not `_comfyui`; its README documents the merge rule used here).

Adapter weights carry their own licence, which is not this repo's: the NAR LoRAs
published so far are CC BY-NC, and none are distributed here.
