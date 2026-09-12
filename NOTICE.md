# Third-party notices

yue2.cpp is MIT licensed (see LICENSE). It ships **no model weights**. The
converter reads weights the user has obtained from Hugging Face themselves.

## Code lineage

- **llama.cpp** (`llama.cpp/` submodule, which also carries **ggml**) —
  Copyright (c) 2023-2026 The ggml authors and the llama.cpp contributors,
  MIT. `yue2-ar` links `libllama`/`libggml` for the whole AR inference path
  (context, batching, sampling primitives it doesn't reimplement, KV cache);
  `yue2-vae` links `libggml` only, for the tensor/graph runtime.
- **nlohmann/json** (`llama.cpp/vendor/nlohmann`, vendored inside the
  llama.cpp submodule) — Copyright (c) 2013-2025 Niels Lohmann, MIT.
  `yue2-ar` includes it directly for `plan.json`/`plan_manifest.json`.
- **llama.cpp `conversion/qwen.py`** (`QwenModel.token_bytes_to_string`,
  `QwenModel.bpe`) — Copyright (c) 2023-2026 The ggml authors and the
  llama.cpp contributors, MIT. Copied nearly verbatim into
  `convert/convert_ar.py` (see the comment above those functions there) to
  turn the raw `qwen.tiktoken` ranks into the gpt2-style vocab + merges GGUF
  expects, since there is no HF `tokenizer.json` to convert from.
- **stable-diffusion.cpp** — Copyright (c) 2023 leejet, MIT. The Conv1D /
  ConvTranspose1D / SnakeBeta block structure in `src/` follows the pattern of
  its `ltx_audio_vae.hpp`.
- **stable-audio-tools** (Oobleck VAE architecture) — Copyright (c) 2023
  Stability AI, MIT. Commit a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45.
- **BigVGAN SnakeBeta** — Copyright (c) 2022 NVIDIA CORPORATION, MIT.
- **YuE2 reference inference** (`yue2_infer`, m-a-p) — the exact-boundary
  tiled decoding scheme is reimplemented from its documented math, not copied.

All of the above are MIT; their full license texts are included with the
respective upstream projects (`llama.cpp/LICENSE`, and the same permissive
text for the smaller snippets — see each project's repository for the
canonical copy). `convert/requirements.txt` additionally installs
**transformers** (Apache-2.0, HuggingFace) at runtime for
`bytes_to_unicode()` (used by the copied BPE code above) and for loading
`modeling_vae.py`'s `PretrainedConfig` base class; it is a dependency, not
copied code.

## Model weights (NOT included, NOT covered by this license)

- **m-a-p/YuE2-Vae** and **m-a-p/YuE2-3B** — CC BY-NC 4.0. Non-commercial.
  Converting them to GGUF with this tool does not change their license.
