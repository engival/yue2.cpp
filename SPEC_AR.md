# SPEC — stage 2: YuE2-3B autoregressive stages (ABC plan + semantic tokens)

Contract between `convert/convert_ar.py` (converter + goldens) and `src/yue2-ar.cpp`
(C++ generator linking libllama). Read SPEC.md §6 (working rules) first — they
all apply, plus: **the 3B never runs on the GPU under torch. Torch CPU only.**

Reference (read-only):
- code: `../venv_yue2/lib64/python3.12/site-packages/yue2/{modeling_yue2,protocol,sampling,pipeline,tokenization_yue2}.py`
- weights: `~/.cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19/{model.safetensors,config.json,qwen.tiktoken,yue2_generation_config.json,examples/}`
- a finished run's artifacts (the target file formats): `../songs/ref_song/out/alley_swing_s1/` (READ-ONLY)
- llama.cpp checkout with a working Vulkan build: `../llama.cpp` (binaries in `~/bin/llama-*`, e.g. `llama-cli`, `llama-bench`, `llama-tokenize`, `llama-server`; device 0 = 7900 XTX)

## 1. Why libllama

The AR path of YuE2-3B is **exactly llama.cpp's `qwen3` architecture**:
hidden 2048, 28 layers, 16 heads / 8 kv heads, head_dim 128, intermediate 6144,
SwiGLU, RMSNorm eps 1e-6 (`x * rsqrt(mean(x²)+eps) * w`), per-head q_norm/k_norm
(RMSNorm over head_dim, applied BEFORE RoPE), NO qkv bias, RoPE theta 1e6
**NEOX style** (rotate half: `[x1*cos - x2*sin, x2*cos + x1*sin]` on the two
halves), untied lm_head, context 24576, vocab 184704. So: convert the AR subset
to a standard qwen3 GGUF, and the C++ side gets tokenizer, KV cache, Vulkan
attention and quantization from libllama. Only the prompt protocol and the
sampling rules are ours.

## 2. GGUF (converter output) — `convert/convert_ar.py`

Architecture `qwen3`. Tensor subset and llama.cpp names:

| safetensors | gguf | shape |
|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | [184704, 2048] |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | [2048] |
| `model.layers.N.self_attn.{q,k,v,o}_proj.weight` | `blk.N.attn_{q,k,v,output}.weight` | q,o [2048,2048]; k,v [1024,2048] |
| `model.layers.N.self_attn.{q,k}_norm.weight` | `blk.N.attn_{q,k}_norm.weight` | [128] |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` | [2048] |
| `model.layers.N.mlp.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}.weight` | gate,up [6144,2048]; down [2048,6144] |
| `model.norm.weight` | `output_norm.weight` | [2048] |
| `lm_head.weight` | `output.weight` | [184704, 2048] |

**Skip** everything else (`nar_*`, `vae2llm`, `llm2vae`, `time_embedder`,
`latent_pos_embed`) — that is stage 3's GGUF. Do NOT permute q/k (that is a
llama-1/2 thing; qwen3 conversion in `convert_hf_to_gguf.py` doesn't).
Metadata: the standard `qwen3.*` keys (`context_length` 24576, `embedding_length`
2048, `block_count` 28, `feed_forward_length` 6144, `attention.head_count` 16,
`attention.head_count_kv` 8, `attention.key_length`/`value_length` 128,
`attention.layer_norm_rms_epsilon` 1e-6, `rope.freq_base` 1e6) plus
`general.name = "YuE2-3B AR"` and `yue2.source_sha256`. Use `gguf-py`
(`GGUFWriter`, `gguf.MODEL_ARCH.QWEN3`, `TensorNameMap`) — mirror what
`../llama.cpp/convert_hf_to_gguf.py` does for `Qwen3ForCausalLM`
rather than driving that script directly (it expects an HF tokenizer dir we don't have).
Output dtype: `--type f16` (default) and `--type bf16`; Q8_0 is produced afterwards
with `~/bin/llama-quantize file-f16.gguf file-q8_0.gguf Q8_0`.
Outputs → `tests/out/yue2-ar-{f16,q8_0}.gguf` (gitignored).

### Tokenizer (must be in the same GGUF)

Source: `qwen.tiktoken` = 151643 base64 `token rank` lines. The GGUF vocab is
`tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "qwen2"` (llama.cpp's
Qwen regex pre-tokenizer is this exact tiktoken pattern), with:

- ids 0..151642: the tiktoken ranks, converted with the GPT-2 byte→unicode map
  (`convert_hf_to_gguf.py` has `QwenModel.token_bytes_to_string` and
  `QwenModel.bpe` for deriving the **merges** from tiktoken ranks — reuse them).
  type NORMAL.
- ids 151643..151850: the 208 specials in this exact order (from
  `tokenization_yue2.py`): `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<R>`,
  `<S>`, `<X>`, `<mask>`, `<sep>`, then `<extra_0>`..`<extra_199>` EXCEPT that
  `<extra_196>`/`<extra_197>` (ids 151847/151848) are `<abc>`/`</abc>`. type CONTROL.
- ids 151851, 151852: MUSIC_START / MUSIC_END (`protocol.py`) — the two ids
  right after the 208-entry specials list. Name them `<music>` / `</music>`
  (CONTROL). Then CODEC_OFFSET = 151853: ids 151853..184620 are
  `<codec_0>`..`<codec_32767>` (USER_DEFINED), and
  184621..184703 are `<pad_0>`..`<pad_82>` (UNUSED) so the vocab is exactly
  184704 and logit indices match torch bit-for-bit.
- `bos_token_id`/`eos_token_id` = 151643 (`<|endoftext|>` = EOD), `add_bos = false`.

**Acceptance:** `~/bin/llama-tokenize -m tests/out/yue2-ar-f16.gguf -p "<text>"
--ids` must equal `YuE2TextTokenizer.encode(text)` (tiktoken, after NFC) for:
the demo request text (`examples/tonight-awake.json` → `SongRequest.text()`,
Chinese lyrics), the alley_swing_s1 `request.json` text (English), and a
snippet of ABC notation (`score.abc` from that dir). Write the check as
`convert/check_tokenizer.py`; zero mismatches required.

### Goldens (torch CPU only — never GPU)

`convert/reference_ar.py`: load `YuE2ForCausalLM` on CPU in **float32**
(`torch_dtype=torch.float32`, from the snapshot dir, `trust_remote_code` via the
snapshot's `modeling_yue2.py` — see how `pipeline.py` loads it), build the
alley_swing_s1 prefix with `protocol.token_prefixes(request, tokenizer)` for
`cot="full"` with `abc_ids=None` (that is the ABC-phase prefix: `[EOD] + text + [ABC_START]`),
run ONE forward pass, save:
- `tests/golden/ar_prefix_ids.npy` (int32) — the prefix,
- `tests/golden/ar_last_logits_f32.npy` — logits at the last position, float32 [184704],
- `tests/golden/ar_greedy_32.npy` — 32 greedy tokens continued from that prefix
  (temperature 0 path of `sampling.distribution` with the ABC-phase allowed mask, i.e.
  ids < EOD or ABC_END; min_tokens applies).
Record the wall time. 3B F32 on the i9 for ~700 tokens is fine (a minute or two);
32 greedy steps with a KV cache too. If the CPU path needs the yue2 `StaticKVCache`,
use it; if simpler, just re-run the full prefix each step (32 × ~1 min is too slow —
use the cache).

## 3. C++ generator — `src/yue2-ar.cpp`, binary `yue2-ar`

Build: add **llama.cpp as a submodule** (`git submodule add https://github.com/ggml-org/llama.cpp.git llama.cpp`)
and build ggml from `llama.cpp/ggml` instead of the separate `ggml/` submodule
(two ggml copies in one process would clash). Remove the `ggml/` submodule
(`git rm ggml`, keep `.gitmodules` clean). `yue2-vae` must still build and pass its
golden test afterwards (`build/yue2-vae … --full --frames 48 --npy` vs
`tests/golden/alley_swing_s1_f48.npy` ≥ 60 dB — same command as in src/STATUS.md).
CMake: `LLAMA_BUILD_EXAMPLES/TESTS/TOOLS/SERVER OFF`, `GGML_VULKAN ON`.

Protocol (from `protocol.py`, reimplement):
```
text = INSTRUCTIONS[cot] + "\n[Tags]\n" + style + "\n[Lyrics]\n" + lyrics + "\n"
INSTRUCTIONS:
  off:    "Generate music with codec tokens from the given conditions."
  melody: "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
  full:   "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
prefix_abc      = [EOD] + tokenize(NFC(text)) + [ABC_START]          (cot != off)
prefix_semantic = prefix_abc + abc_ids + [ABC_END, MUSIC_START]
cot == off:       [EOD] + tokenize(text) + [ABC_START, ABC_END, MUSIC_START]  (no abc phase)
```
(`tokenize` = `llama_tokenize` with `add_special=false, parse_special=false`;
NFC-normalize the text first — pull a small NFC routine or document that input
is assumed NFC; the artifacts' `request.json` text is plain ASCII/UTF-8.)
Constants: EOD 151643, ABC_START 151847, ABC_END 151848, MUSIC_START 151851,
MUSIC_END 151852, CODEC_OFFSET 151853, CODEC_SIZE 32768, CONTEXT 24576.

Sampling per step (from `sampling.py`, on the raw float logits from
`llama_get_logits_ith`), per phase with the `yue2_generation_config.json` values:

| | abc | semantic |
|---|---|---|
| allowed ids | `< EOD` (0..151642) ∪ {ABC_END} | `[CODEC_OFFSET, CODEC_OFFSET+CODEC_SIZE)` ∪ {MUSIC_END} |
| end token | ABC_END | MUSIC_END |
| temperature / top_p / top_k | 0.7 / 0.9 / 30 | 1.0 / 0.95 / 100 |
| repetition_penalty / penalty_window | 1.005 / 100 | 1.2 / 50 |
| min_tokens / max_tokens | 32 / 4096 | 200 / 9000 |

Order: mask disallowed to −inf → if step < min_tokens mask `end` → window
penalty over the last `penalty_window` **generated** ids: `freq[id]` = count in
window, `alpha = penalty^freq`, `logit<0 ? logit*alpha : logit/alpha` → if
temperature == 0: argmax → else divide by temperature → top-k (keep ≥ k-th value)
→ top-p (sort desc, softmax, drop where `cumsum − p > top_p`, always keep the top 1)
→ sample from softmax with a seeded RNG (`std::mt19937_64(seed)`; exact RNG
parity with torch is not expected — greedy is the parity path). Stop on `end`
or `max_tokens`. Refuse prefix + max_tokens > CONTEXT. `cfg_scale != 1` →
error "not supported" (stage 2b+).

CLI:
```
yue2-ar -m yue2-ar-q8_0.gguf --request song.json --artifacts DIR
        [--seed N] [--cot full|melody|off] [--device cpu|vulkan] [--gpu 0]
        [--dump-logits FILE.npy] [--greedy]  [--max-abc N] [--max-semantic N]
```
`--request` is the same JSON as `yue2_gen.py` (`style`, `lyrics`, optional `cot`, `seed`).
Artifacts written to DIR (match the torch formats so the Python pipeline can
resume from them — read `pipeline.py` `SymbolicPlan.save`/`SongResult.save_artifacts`
and look at the alley_swing_s1 dir): `prefix.npy` (int32 semantic-phase prefix),
`abc_tokens.npy` (int32 abc ids, no ABC_START/END), `score.abc` (detokenized
abc), `semantic.npy` (int32 **codes** = id − CODEC_OFFSET, without MUSIC_END),
`request.json`, and a `plan.json`/`plan_manifest.json` if the resume path needs
them (document what you could/couldn't reproduce). Print per-phase tokens,
seconds, tokens/s.

`--dump-logits`: run only the prefix (ABC phase) and write the last-position
float32 logits [184704] as .npy; `--greedy` forces temperature 0 in both phases.

### Acceptance
1. `yue2-ar --dump-logits` on the alley_swing_s1 request vs
   `tests/golden/ar_last_logits_f32.npy`: report max|Δ| and the argmax; with
   the F16 GGUF on CPU expect argmax equal and max|Δ| well under 0.1 on logits
   of magnitude ~10–30; record Q8_0 and Vulkan numbers too.
2. `--greedy --max-abc 32` first 32 abc tokens vs `tests/golden/ar_greedy_32.npy`:
   expect identical on CPU F16 (a divergence after a near-tie is acceptable —
   report the first differing index and the logit gap there).
3. Full run, default sampling, seed from the request, Q8_0 on Vulkan device 0:
   artifacts written; abc phase ends with ABC_END within max_tokens; `score.abc`
   is plausible ABC text (starts with `X:`/`L:`/`M:`/`K:` headers); semantic phase
   produces ≥ 200 codes all in [0, 32768). Record tokens/s for each phase — the
   torch numbers to beat are 42 t/s (abc) and 33 t/s (semantic, decaying).
   Also `llama-bench` numbers for the Q8_0 GGUF on device 0 (`-p 512 -n 128`).

STATUS files: `convert/STATUS_AR.md`, `src/STATUS_AR.md`.
