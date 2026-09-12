# SPEC — stage 1: YuE2 Oobleck VAE decoder on ggml

This file is the contract between the converter (`convert/`), the C++ decoder
(`src/`), and the verification harness (`tests/`). Change it only by agreement;
note any deviation in your STATUS.md.

Reference implementation (read-only, for the math):
`../venv_yue2/lib64/python3.12/site-packages/yue2/modeling_vae.py`
Reference weights + config (read-only):
`~/.cache/huggingface/hub/models--m-a-p--YuE2-Vae/snapshots/95535e72a97bc0f09b8ada125d26b4009428c0e8/{model.safetensors,config.json}`

## 1. Architecture (decoder only; the encoder is NOT ported)

Config values (all fixed for the released checkpoint):

| key | value |
|---|---|
| channels | 64 |
| c_mults | [1, 2, 4, 8, 16, 32] → effective list `[1] + c_mults` = [1,1,2,4,8,16,32], depth = 7 |
| strides | [2, 2, 4, 4, 5, 6] |
| latent_dim (decoder input channels) | 64 |
| out_channels | 2 |
| use_snake | true (SnakeBeta with alpha_logscale=True) |
| final_tanh | false |
| sample_rate | 48000 |
| downsampling_ratio | 1920 (= 2·2·4·4·5·6) |
| decode_core_frames / decode_halo_frames | 1024 / 16 (torch defaults) |

Layer list, torch names, in execution order (`decoder.layers.N`):

```
layers.0   Conv1d   64 → 2048, k=7, pad=3, bias
layers.1   DecoderBlock 2048 → 1024, stride 6
layers.2   DecoderBlock 1024 →  512, stride 5
layers.3   DecoderBlock  512 →  256, stride 4
layers.4   DecoderBlock  256 →  128, stride 4
layers.5   DecoderBlock  128 →   64, stride 2
layers.6   DecoderBlock   64 →   64, stride 2
layers.7   SnakeBeta(64)
layers.8   Conv1d   64 → 2, k=7, pad=3, NO bias
layers.9   Identity (final_tanh=false)  — nothing to do
```

DecoderBlock(in, out, s) = `layers.0..4`:

```
.layers.0  SnakeBeta(in)
.layers.1  ConvTranspose1d in → out, k=2s, stride=s, pad=ceil(s/2), bias
.layers.2  ResidualUnit(out, dilation 1)
.layers.3  ResidualUnit(out, dilation 3)
.layers.4  ResidualUnit(out, dilation 9)
```

ResidualUnit(c, d) = `x + layers(x)` with `.layers.0..3`:

```
.layers.0  SnakeBeta(c)
.layers.1  Conv1d c → c, k=7, dilation=d, pad=3d, bias
.layers.2  SnakeBeta(c)
.layers.3  Conv1d c → c, k=1, bias
```

SnakeBeta forward (alpha_logscale=True in the checkpoint):

```
a = exp(alpha)  ; b = exp(beta)          # per-channel, shape [C]
y = x + (1 / (b + 1e-9)) * sin(x * a)^2
```

Output length for T latent frames: **1920·T − 64** samples per channel
(each conv keeps length; ConvT lengths: s=6 → 6L, s=5 → 5L−1, s=4 → 4L, s=2 → 2L).

## 2. GGUF file (converter output)

`general.architecture = "yue2-vae"`, `general.name = "YuE2-Vae decoder"`,
plus metadata (all under prefix `yue2vae.`): `channels`, `c_mults` (array
i32, the 6 raw values), `strides` (array i32), `latent_dim`, `out_channels`,
`sample_rate`, `downsampling_ratio`, `decode_core_frames`, `decode_halo_frames`,
`snake_folded` (bool, true), `source_sha256` (the safetensors sha256).

Tensor names: **keep the torch names**, drop the weight-norm split, keep dtype
F32 (a `--type f16` converter option may store Conv1d weights as F16; ConvT
weights and all biases/alpha/beta stay F32 — the Vulkan conv_transpose_1d
kernel is F32-only).

Weight-norm folding (torch `weight_norm` default `dim=0` for BOTH conv kinds):

```
w = g * v / ||v||      where ||v|| is the L2 norm over every dim except dim 0
```
- Conv1d: `v` is [Cout, Cin, K], `g` is [Cout, 1, 1] → norm over (Cin, K).
- ConvTranspose1d: `v` is [Cin, Cout, K], `g` is [Cin, 1, 1] → norm over (Cout, K).
Store the folded result as `<prefix>.weight`; `<prefix>.bias` unchanged.
Sanity-check against `torch.nn.utils.remove_weight_norm` / the parametrized
module's `.weight` in the converter and assert max-abs-diff < 1e-6.

SnakeBeta: store `alpha` and `beta` **already exponentiated** (effective
positive values), same names `<prefix>.alpha`, `<prefix>.beta`, shape [C].
Graph then computes `x + sin(x*alpha)^2 / (beta + 1e-9)` with no exp.

Layout note: gguf-py writes numpy arrays so that torch `[Cout, Cin, K]` shows up
in ggml as `ne = [K, Cin, Cout]` (ne[0] fastest) — exactly what `ggml_conv_1d`
wants for its kernel. ConvT torch `[Cin, Cout, K]` → ggml `ne = [K, Cout, Cin]`,
which is what `ggml_conv_transpose_1d` wants. **No transposes in the converter.**

Suggested output path: `tests/out/yue2-vae-f32.gguf`
(gitignored). The converter runs in `../venv_yue2/bin/python`
(has torch CPU, safetensors, numpy); install `gguf` from PyPI into that venv
(`venv_yue2/bin/pip install gguf`) or `pip install -e ../llama.cpp/gguf-py`.

## 3. Decoder graph (C++)

- Activations F32. Input latent tensor ggml `ne = [T, 64, 1]` (i.e. torch [1, 64, T]).
- Conv1d: do NOT call `ggml_conv_1d` (it hard-codes an F16 im2col). Call
  `ggml_im2col(ctx, w, x, s0=1, s1=0, p0=pad, p1=0, d0=dil, d1=0, is_2D=false, dst_type)`
  with `dst_type` = F32 by default (`--im2col f16` flag switches it) and then
  `ggml_mul_mat` exactly as `ggml_conv_1d` does in `ggml/src/ggml.c`; add bias
  (reshape bias to `[1, Cout, 1]` and `ggml_add`).
- ConvTranspose1d: `ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1)` — ggml
  requires p0 == 0, so the torch `padding=ceil(s/2)` is implemented by
  **cropping `p` samples off each end** of ggml's full output with `ggml_view`
  (torch output len = (L−1)·s + 2s − 2p). Then add bias.
- SnakeBeta: `ggml_sin(ggml_mul(x, alpha))`, `ggml_sqr`, `ggml_div` by
  `(beta + 1e-9)` (precompute `inv_beta = 1/(beta+1e-9)` once at load and
  `ggml_mul`), `ggml_add(x, ...)`. alpha/beta broadcast as `ne = [1, C, 1]`.
- Residual: `ggml_add(x, branch)`.
- Backend: `ggml_backend_dev_by_type` / `ggml_backend_init_by_name`; CPU
  always available, Vulkan via `GGML_VULKAN=ON`. Use `ggml_gallocr` for the
  graph and `ggml_backend_alloc_ctx_tensors` for weights (same pattern as
  `ggml/examples/simple/simple-backend.cpp`). One graph per tile length; cache
  by tile length so all full-size tiles share one allocation.
- Tiling (must match torch `decode_tiled` exactly — no crossfade, exact crops):

```
for start in range(0, T, core):
    end   = min(T, start + core)
    left  = max(0, start - halo)
    right = min(T, end + halo)
    tile  = decode(latent[left:right])                 # full graph on (right-left) frames
    out_start, out_end = start*1920, min(end*1920, total)   # total = 1920*T - 64
    crop_start = (start - left) * 1920
    audio[out_start:out_end] = tile[crop_start : crop_start + (out_end - out_start)]
```
  Default `--core-frames 256 --halo-frames 16` (smaller core than torch's 1024
  keeps im2col buffers sane: top layers are 64 ch × k7 × 1920·core samples).
  **Requirement:** output must be independent of `core`: bit-identical on the
  CPU backend (measured 0.0 diff for core 64 vs 256 vs untiled). On Vulkan the
  backend's own matmul/split-k choices depend on tensor shape, so runs at
  different core sizes differ by ~3e-3 peak / ~7e-5 RMS (68 dB) — accepted;
  see src/STATUS.md "Deviation". halo must be ≥ 16; reject smaller.

## 4. CLI contract (`yue2-vae`)

```
yue2-vae -m vae.gguf -i latent.npy -o out.wav
         [--device cpu|vulkan] [--gpu N] [--threads N]
         [--core-frames 256] [--halo-frames 16] [--im2col f32|f16]
         [--frames N]        # decode only the first N latent frames (for golden tests)
         [--npy out.npy]     # also dump raw float32 [2, samples] for numeric tests
         [--full]            # untiled single-graph decode (for golden tests only)
```
- `.npy` input: implement a minimal reader (npy v1/v2 header, `<f4`, C-order).
  Accept shape `[T, 64]` (what `yue2_gen.py --artifacts` writes) or `[1, 64, T]`
  or `[64, T]`; document which it assumed. Reject anything else.
- `.wav` output: RIFF WAVE, 32-bit float, 2 ch, 48000 Hz, samples clamped to [−1, 1]
  only in the WAV writer — the `--npy` dump is the raw unclamped decoder output.
- Print timing: load, per-tile, total; and the backend/device name.
- Exit non-zero with a clear message on any shape/metadata mismatch.

## 5. Acceptance tests

Test latent: `../songs/ref_song/out/alley_swing_s1/latent.npy`
(shape [4109, 64] float32) and its torch render `audio.flac` in the same dir.
That directory is **READ-ONLY** — never write into `songs/`.

Golden (torch CPU, produced by the converter's sibling `convert/reference_decode.py`):
- `tests/golden/alley_swing_s1_f48.npy`: untiled decode of frames [0, 48) → float32
  [2, 1920·48 − 64 = 92096]. Torch on **CPU only** (`device="cpu"`); never run torch on the GPU in this project.
- Same for `_f8.npy` (tiny, for fast iteration).

Pass criteria (C++ `--full --frames 48 --npy` vs golden):
- F32 im2col: SNR ≥ 60 dB and max |err| < 1e-3 (signal is roughly ±1).
- Report the F16 im2col numbers too; no pass threshold, just record them.

Full song (tiled, Vulkan): decode the whole latent, compare to `audio.flac`
(decode the flac with `soundfile`, it is the torch tiled output after
`song.save`, so it may be PCM-quantized — check the subtype and expect SNR to be
bounded by that). Also compare core=64 vs core=256 outputs to each other (bit-identical on CPU; ≥ 60 dB on Vulkan).
Record wall time per tile and total on Vulkan and on CPU.

## 6. Working rules for agents

- Work only inside this repo. Scratch output → `tests/out/`.
- `songs/` and the HF cache are read-only inputs.
- No torch on GPU. No ROCm. Ever. (It has taken the box down today.)
- Keep a `STATUS.md` in your area (`convert/STATUS.md`, `src/STATUS.md`): what
  is done, what deviates from this spec and why, exact
  commands to reproduce, and open questions. Final report to the coordinator:
  ≤ 200 words, point at STATUS.md for details.
- Coding style: tabs for indentation; braces on their own line except `} else {`.
- Don't `git commit`; the coordinator commits.
