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
driver + `glslc`, and python3. All paths below are relative to the repo root.

**Tested on** one box only: Slackware64-current (Linux), an AMD Radeon RX 7900
XTX (RADV) and an Intel Arc Pro B70 (ANV), both through ggml's Vulkan backend.
**Not tested:** NVIDIA, Windows, macOS, or any backend other than Vulkan. The
code has no platform-specific parts beyond `/proc/self/exe` for the VAE child
process, so other Vulkan setups may well work; reports either way are welcome.

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
`abc` and `cfg_scale` are optional. Lyrics carry `[Section]` tags on their own
lines:

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
`--nar-f32` trades the fp16-staged matmuls for the exact ones; because
ggml-vulkan reads that switch once per device init, it applies to the AR as
well, whose logits then shift — so `--nar-f32` yields a *different song*, not the
same song rendered more precisely. The three GGUFs
default to `yue2-ar-q8_0.gguf`, `yue2-nar-f16.gguf`, `yue2-vae-f32.gguf` next to
the executable, then one directory up. `--artifacts` writes the same file set the
reference pipeline did (`plan.json`, `prefix.npy`, `semantic.npy`, `nar_noise.npy`,
`latent.npy`, `config.json`, `result.json`, …); without it a temporary directory
is used and removed.

**The VAE runs as a child process of the same binary**, re-executed through
`/proc/self/exe`. It needs `GGML_VK_DISABLE_F16`/`_COOPMAT` set for true-F32
matmuls while the NAR's fast path needs them unset, and ggml-vulkan reads both
once per device init — so one Vulkan context cannot serve both. The VAE loads in
~0.2 s, so residency would buy nothing. The latent crosses as `DIR/latent.npy`;
everything else is handed over in memory.

Measured on the Intel Arc B70, 156 s of audio: 155 s end to end (abc 15 s,
semantic 37 s, NAR 88 s, VAE 14 s), ~8.7 GB peak VRAM. Details, the per-stage
goldens and the Q8_0/F16 prefix measurement are in
[src/STATUS_SINGLE.md](src/STATUS_SINGLE.md).

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
convenience, not a contract.

## Decode

```bash
build/yue2-vae -m yue2-vae-f32.gguf -i latent.npy -o song.wav --device vulkan --gpu 0
```

(or `build/yue2 vae …` — same flags.) An `-o` ending in `.flac` is written as
24-bit FLAC through libFLAC instead of float WAV; its integer samples are
identical to what `soundfile`'s `PCM_24` wrote for the reference pipeline.

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
