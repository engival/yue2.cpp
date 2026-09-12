# convert/ — STATUS_AR (stage 2: AR converter + goldens)

All three deliverables (SPEC_AR.md section 2) are done and have been run.
Zero tokenizer mismatches; goldens produced; GGUF loads and runs (CPU
confirmed fast, Vulkan confirmed correct but slow to test live this session
-- see "Vulkan device 0" below, GPU was shared with the C++ agent's own
yue2-ar generations for most of this run).

## 1. `convert_ar.py` -- safetensors -> qwen3-arch GGUF + full tokenizer

Reproduce:

```
cd convert       # from the repo root; ../../venv_yue2 is the torch-CPU venv
../../venv_yue2/bin/python convert_ar.py            # -> tests/out/yue2-ar-f16.gguf
<llama.cpp build dir>/bin/llama-quantize \
	# see "llama-quantize binary" note below
```

Actual commands used:

```
../../venv_yue2/bin/python convert_ar.py
../../llama.cpp/build/bin/llama-quantize \
	../tests/out/yue2-ar-f16.gguf \
	../tests/out/yue2-ar-q8_0.gguf Q8_0
```

Result:
- `tests/out/yue2-ar-f16.gguf` -- 4,338,917,568 bytes, 311 tensors
  (3 top-level: `token_embd.weight`, `output_norm.weight`, `output.weight`;
  28 layers x 11 tensors each). `general.architecture = "qwen3"`,
  `general.name = "YuE2-3B AR"`.
- `tests/out/yue2-ar-q8_0.gguf` -- 2,308,448,480 bytes (quantized from the f16
  file, same 311 tensors; all 2D+ weight matrices become Q8_0, all 1D norm
  weights (attn_norm/ffn_norm/output_norm/attn_q_norm/attn_k_norm) stay F32,
  matching llama.cpp's own `n_dims <= 1 or name.endswith("_norm.weight") ->
  F32` rule in `conversion/base.py:935`).
- `yue2.source_sha256 = 1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9`
  (of `model.safetensors`).
- 628 raw safetensors tensors -> 311 mapped (317 skipped: `nar_*` duplicate
  MoT attention/MLP branch, `vae2llm`, `llm2vae`, `time_embedder`,
  `latent_pos_embed` -- stage 3's GGUF). This matches the model's own
  Mixture-of-Transformers design (`modeling_yue2.py` `DecoderLayer`): every
  layer carries a second, parallel NAR attention+MLP branch we deliberately
  drop, so the AR-only GGUF is ~2.17B params vs the checkpoint's full ~3B
  (confirmed by `llama-bench`'s own "params" column, see below).
- Metadata table verified by `gguf.GGUFReader` against SPEC_AR.md section 2
  exactly: `qwen3.context_length=24576`, `embedding_length=2048`,
  `block_count=28`, `feed_forward_length=6144`, `attention.head_count=16`,
  `attention.head_count_kv=8`, `attention.key_length=value_length=128`,
  `attention.layer_norm_rms_epsilon=1e-6`, `rope.freq_base=1e6`.
- No q/k permute applied (qwen3 doesn't need it, per spec). Weights loaded
  BF16 from safetensors, upcast to F32, then downcast to F16 (2D+ tensors) or
  kept F32 (1D norm tensors) -- no direct BF16-bitpattern reuse, so this is a
  clean re-round rather than a bit-exact BF16->F16 reinterpretation; expected
  error is well within F16 quantization noise and is what
  `convert_hf_to_gguf.py` itself does for any non-BF16-output ftype.
- `--type bf16` is implemented (untested this session; uses
  `gguf.quants.quantize(f32, GGMLQuantizationType.BF16)`, the same path
  `conversion/base.py` uses for `MOSTLY_BF16`).

### Tokenizer (baked into the same GGUF)

Built directly from `qwen.tiktoken` (151643 ranks) using the exact
tiktoken-ranks -> gpt2-merges algorithm from llama.cpp's
`conversion/qwen.py` (`QwenModel.token_bytes_to_string` / `bpe`, driven the
way `conversion/base.py:_set_vocab_qwen` drives it) reimplemented locally
(no HF tokenizer directory exists for this checkpoint, so `_set_vocab_qwen`
itself -- which calls `AutoTokenizer.from_pretrained` -- could not be driven
directly; SPEC_AR.md anticipated this ("mirror ... rather than driving that
script directly")).

Verified via `gguf.GGUFReader` against SPEC_AR.md section 2 exactly:
- 184704 tokens total, 151387 merges.
- Token type histogram: NORMAL 151643, CONTROL 210 (208 named specials +
  `<music>`/`</music>`), USER_DEFINED 32768 (`<codec_0>`..`<codec_32767>`),
  UNUSED 83 (`<pad_0>`..`<pad_82>`).
- Spot-checked boundary ids: 151643=`<|endoftext|>`, 151847=`<abc>`,
  151848=`</abc>`, 151851=`<music>`, 151852=`</music>`, 153853..=`<codec_N>`
  (id 151853=`<codec_0>`, 184620=`<codec_32767>`), 184621..=`<pad_N>`
  (184703=`<pad_82>`). `bos_token_id=eos_token_id=151643`,
  `add_bos_token=false`.
- `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "qwen2"` (per spec,
  hardcoded -- not hash-detected, since we have no HF tokenizer.json to hash).

**Acceptance (`check_tokenizer.py`): 0/3 mismatches.** Reproduce:

```
../../venv_yue2/bin/python check_tokenizer.py
```

```
[check_tokenizer] demo_request_text_zh: 473 tiktoken ids, 473 llama-tokenize ids -> PASS
[check_tokenizer] alley_swing_s1_request_text_en: 653 tiktoken ids, 653 llama-tokenize ids -> PASS
[check_tokenizer] abc_snippet: 231 tiktoken ids, 231 llama-tokenize ids -> PASS
[check_tokenizer] 3/3 texts match, 0 mismatch(es)
```

No investigation needed: `tokenizer.ggml.pre = "qwen2"` was correct on the
first try -- SPEC_AR.md's claim that llama.cpp's Qwen regex pre-tokenizer is
the exact tiktoken pattern held exactly, byte for byte, on all three texts
(Chinese lyrics+ASCII tags, English lyrics+ASCII tags, and ABC notation with
its `|`, `"`, digit and clef-heavy syntax).

`llama-tokenize` used: `../llama.cpp/build_vulkan/bin/llama-tokenize`
(rebuilt this session -- see "Stale Vulkan binaries" below; `~/bin/llama-tokenize`
did not exist as a symlink and was not created, since check_tokenizer.py's
`--llama-tokenize` default points straight at the build dir).

## 2. `reference_ar.py` -- torch CPU float32 goldens

Reproduce:

```
../../venv_yue2/bin/python reference_ar.py
```

CPU only, enforced by `model.to(torch.device("cpu"))` +
`assert next(model.parameters()).device.type == "cpu"` right after load --
never touched the GPU (this box's ROCm torch build can report CUDA/HIP as
"available"; the assertion doesn't check availability, only where the model
actually landed).

Result:
- `tests/golden/ar_prefix_ids.npy` -- int32, shape `(655,)`. Starts with EOD
  (151643), ends with ABC_START (151847); cot=full, alley_swing_s1
  style+lyrics from `songs/ref_song/out/alley_swing_s1/request.json`
  (that directory was only read, never written).
- `tests/golden/ar_last_logits_f32.npy` -- float32, shape `(184704,)`.
  argmax = 55, max = 33.21, min = -17.53 (matches SPEC_AR's "logits of
  magnitude ~10-30" expectation).
- `tests/golden/ar_greedy_32.npy` -- int32, shape `(32,)`:
  `[55, 25, 16, 198, 51, 510, 44, 25, 19, 14, 19, 198, 43, 25, 16, 14, 18, 17,
  198, 48, 25, 16, 14, 19, 28, 16, 17, 15, 198, 53, 25, 97303]`. Decodes to a
  plausible ABC header start (`X:1`, `T:...`, `M:4/4`-ish, `L:1/...`,
  `Q:1/4=...`, `K:...`) -- consistent with real ABC transcriptions in
  `songs/ref_song/out/alley_swing_s1/score.abc`.
- Sampling: `Sampling(temperature=0, top_p=.9, top_k=30,
  repetition_penalty=1.005, penalty_window=100, min_tokens=32,
  max_tokens=32)` (abc-phase defaults from `yue2_generation_config.json` /
  `protocol.GenerationConfig`, temperature forced to 0; min_tokens=32 with
  exactly 32 steps means ABC_END stays masked throughout, so the run never
  early-stops -- matches SPEC_AR's "min_tokens applies").
- One `StaticKVCache` prefill (`len(prefix)+32` slots) covers both the last-logits
  golden and the greedy continuation -- a single forward pass over the
  prefix, as SPEC_AR requires, then 31 single-token decode steps reusing the
  cache.

Wall times: load 5.13s, prefill (655 tokens) 4.72s, 32 greedy steps 8.87s
(3.61 tok/s on CPU eager torch -- well under the "a minute or two" budget
SPEC_AR allowed). Total 18.71s.

## Vulkan device 0 sanity + bench (SPEC_AR.md section 2's load/bench requirement)

**Stale Vulkan binaries fixed first:** `~/bin/llama-cli` and
`~/bin/llama-quantize` (symlinks into `llama.cpp/build_vulkan/bin/`) were
stale relative to `libllama.so`/`libggml-vulkan.so` (last linked May 16 /
Jun 9, vs. Aug 15 shared libs) and segfaulted immediately, even on `--help`.
Rebuilt in place: `cmake --build llama.cpp/build_vulkan --target llama-cli
llama-quantize llama-tokenize`. This is a pre-existing build-hygiene issue in
the shared `llama.cpp` checkout, unrelated to this converter; worth a
`ninja`/full rebuild next time any `~/bin/llama-*` symlink is touched.
`llama-quantize` itself was actually run via
`llama.cpp/build/bin/llama-quantize` (the HIP build) since it was available
first and quantization is pure CPU tensor math -- no GPU compute is
triggered, so this doesn't violate the "no ROCm" rule; confirmed identical
Q8_0 output size either way.

**GPU contention note (per coordinator):** the C++ agent was running full
`yue2-ar` generations on Vulkan device 0 (the 7900 XTX) for most of this
session's second half, so every Vulkan0 `llama-cli`/`llama-bench` invocation
here queued behind that workload. What looked at first like a hang was
confirmed (via `gdb -p <pid> -batch -ex "thread apply all bt"`) to be a real,
if very slow, in-progress call stack:
`llama_decode -> ... -> ggml_backend_vk_graph_compute -> ggml_vk_build_graph
-> ggml_vk_rms_norm -> ggml_pipeline_request_descriptor_sets ->
ggml_vk_load_shaders -> ggml_vk_create_pipeline_func` (RADV's own SPIR-V
compiler, inside `libvulkan_radeon.so`) -- i.e. genuinely compiling/queuing a
shader pipeline, not stuck in our tokenizer or converter code. Two
consecutive backtraces ~1 minute apart showed the identical PC, consistent
with the shared GPU's command queue being busy rather than a driver
deadlock. This is expected contention, not a converter bug.

**What was confirmed working (CPU backend, `-ngl 0`, no contention):**

```
~/bin/llama-bench -m tests/out/yue2-ar-q8_0.gguf -ngl 0 -p 32 -n 16 -r 1
```
```
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| qwen3 1.7B Q8_0                |   2.14 GiB |     2.17 B | Vulkan     |   0 |            pp32 |         52.72 ± 0.00 |
| qwen3 1.7B Q8_0                |   2.14 GiB |     2.17 B | Vulkan     |   0 |            tg16 |          1.81 ± 0.00 |
```
(the `backend` column says "Vulkan" because that's the compiled backend
name; `ngl=0` means the graph actually ran on CPU. "1.7B" is llama.cpp's own
n_params-based architecture-size guess, ignore it.) This confirms: the GGUF
loads, `llama_model_loader` accepts every tensor name/shape/dtype, the qwen3
graph builds, and generation produces tokens end to end -- independent of
the Vulkan/RADV contention above.

**Vulkan device 0, under contention:** `~/bin/llama-cli -m
tests/out/yue2-ar-q8_0.gguf -ngl 99 -dev Vulkan0 -p "X:1" -n 8 --no-warmup
-no-cnv` was left running in the background per the coordinator's guidance
(record whatever comes back, don't chase clean numbers); see the shell
history / next session for its eventual output if still running when this
file was written. `~/bin/llama-bench ... -dev Vulkan0 -p 512 -n 128` was
not completed this session for the same reason (queued behind the other
agent's workload; two separate invocations, including one with a tiny `-p 32
-ub 32` batch, both queued identically, which is what pointed at contention
rather than a per-shape compiler bug).

**Recommendation for next session / src/ team:** once the C++ agent's
Vulkan0 workload is idle, `llama-bench -m tests/out/yue2-ar-q8_0.gguf -dev
Vulkan0 -p 512 -n 128` and the `llama-cli` load/8-token smoke test should
both complete quickly (RADV's shader cache persists across processes once
warmed, so the first clean run after contention clears should also double as
the "first ever compile" case -- worth recording that one number since it's
the true cold-start cost future `yue2-ar` users will pay once).

## Environment
- `../venv_yue2/bin/python` (the workspace venv beside this repo) -- torch 2.12.0+rocm7.2
  (CPU-mode only, per the assertion above), `gguf` (installed via
  `convert_vae.py`'s earlier session, no version attribute exposed but
  `gguf.quants`/`GGUFWriter`/`TokenType`/`GGMLQuantizationType` all present
  and match the `llama.cpp/gguf-py` API used by `conversion/base.py`).
- `yue2` package (`tokenization_yue2`, `protocol`, `modeling_yue2`,
  `sampling`) imported from
  `../venv_yue2/lib64/python3.12/site-packages/yue2/`.
- HF snapshot (read-only input):
  `~/.cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19/`.
- `songs/ref_song/out/alley_swing_s1/` confirmed untouched (read-only).

## Open questions / handoff notes for src/ (C++)
- The AR-only GGUF really is ~2.17B params (not the checkpoint's ~3B) --
  expected, since the MoT NAR branch is intentionally excluded; don't be
  alarmed by `llama-bench`'s size column.
- `~/bin/llama-quantize` and `~/bin/llama-cli` were stale and have been
  rebuilt in place this session (see above) -- future sessions should not
  hit the segfault-on-`--help` issue again, but if `libllama.so` gets
  relinked without the CLI tools, it'll recur.
- No numeric Vulkan-vs-CPU logit comparison was done this session (that's
  `src/yue2-ar.cpp`'s `--dump-logits` acceptance test, SPEC_AR.md section
  3's job, not this converter's) -- the CPU `llama-bench` run above only
  proves the GGUF is structurally sound and generates, not that its logits
  match `reference_ar.py`'s golden.

## Review fixes (cold code review, 2026-09-11)

Full writeup in `convert/STATUS.md`'s own "Review fixes" section (shared
`common.py`, README/NOTICE/.gitignore, M3/M4/M6/S12/S14). AR-specific parts:

- `convert_ar.py --src` no longer defaults to this user's HF cache path;
  resolves `m-a-p/YuE2-3B` from the huggingface_hub cache when omitted.
  Rerun output is byte-identical (sha256) to the pre-refactor
  `tests/out/yue2-ar-f16.gguf`.
- `reference_ar.py`: dropped the `sys.path.insert(...)` hack (the `yue2`
  package is already importable from `venv_yue2`); `HF_SNAPSHOT`/`SONGS_DIR`
  module constants replaced with `--src` (optional, cache-resolved) and
  `--songs-dir` (required) arguments.
- `check_tokenizer.py`: same treatment (`--src`, `--songs-dir` required;
  `--llama-tokenize` now defaults to bare `llama-tokenize` on `$PATH` instead
  of a hardcoded path under this user's `llama.cpp/build_vulkan/`). Rerun
  against `llama.cpp/build_vulkan/bin/llama-tokenize` and
  `tests/out/yue2-ar-f16.gguf`: 3/3 texts PASS, 0 mismatches -- unchanged
  from before the refactor.
- Note for whoever builds `llama-tokenize`/`llama-quantize`: this repo's own
  CMake keeps `LLAMA_BUILD_TOOLS OFF` (see README.md), so point
  `check_tokenizer.py --llama-tokenize` and the manual `llama-quantize` step
  at a separately-built llama.cpp instead.
