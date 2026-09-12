# SPEC — stage 3: YuE2-3B NAR acoustic flow matching (semantic tokens → VAE latent)

Contract between `convert/convert_nar.py` + `convert/reference_nar.py` (converter
and goldens agent) and `src/yue2-nar.cpp` (C++ agent). The two agents do **not**
talk to each other; everything they must agree on is in this file.

Read **SPEC.md §6 (working rules) first — all of it applies**, plus the additions
in §6 below. In particular: **the 3B never runs on a GPU under torch. Torch CPU
only.**

Reference (read-only):

- code: `../venv_yue2/lib/python3.12/site-packages/yue2/{nar,modeling_yue2,pipeline,protocol,storage}.py`
- weights + config: `~/.cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/1a96eca688d6ae5d7f0feb88573fec89920fcd19/{model.safetensors,config.json,yue2_generation_config.json,weights_manifest.json}`
- a finished run's artifacts (real shapes, the target file formats):
  `../songs/ref_song/out/alley_swing_s1/` (**READ-ONLY**)
- the fp16 lesson: `docs/vulkan_burst_investigation.md` — mandatory reading for
  the C++ agent before writing any backend-init code.

The snapshot's `modeling_yue2.py` and the installed package's
`yue2/modeling_yue2.py` are **byte-identical** (verified by `diff`), so every
`modeling_yue2.py:NNN` line number below is valid in either copy. `nar.py:NNN`
refers to the installed package copy.

---

## 1. What the NAR computes

### 1.0 Inputs and the top-level call

The pipeline entry point is `pipeline.py:285 YuE2Pipeline.synthesize()`, which
calls `nar.synthesize(model, prefix, codec, seed, steps=32, context=24576)`
(`nar.py:227`). Its three data inputs, exactly as they appear on disk in an
artifacts dir:

| input | artifact | dtype / shape | meaning |
|---|---|---|---|
| `prefix` | `prefix.npy` | int32 `[P]` | the semantic-phase AR prefix, i.e. `[EOD] + text_ids + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]` (`protocol.py:115`). For `alley_swing_s1`, `P = 2549`, last id `151851 = MUSIC_START`. |
| `codec` | `semantic.npy` | int32 `[T]`, values in `[0, 32768)` | the semantic **codes**, already offset-stripped (`pipeline.py:283`: `int(t) - CODEC_OFFSET`). For `alley_swing_s1`, `T = 4109`. |
| `seed` | `request.json` `"seed"` | int, `0 ≤ seed < 2**63` | the *song* seed, shared with the AR sampler. `alley_swing_s1` used `seed = 1`. |

Output: `[T, 64]` **float32, C-order** — `latent.npy`. Verified on the artifact:
`latent.npy` is `(4109, 64) float32`, `T` equals `len(semantic.npy)` exactly, and
the VAE decodes `1920·T − 64 = 7 889 216` samples ≈ 164.36 s, matching
`result.json`'s `audio_seconds`. Confirmed.

Constants (`protocol.py:7-12`, `config.json`): `EOD 151643`, `ABC_START 151847`,
`ABC_END 151848`, `MUSIC_START 151851`, `MUSIC_END 151852`, `CODEC_OFFSET 151853`,
`CODEC_SIZE 32768`, `CONTEXT 24576`, `VOCAB_SIZE 184704`, `hidden 2048`,
`layers 28`, `heads 16`, `kv_heads 8`, `head_dim 128`, `intermediate 6144`,
`rms_norm_eps 1e-6`, `rope_theta 1e6`, `latent_dim 64`, `max_latent_frames 24576`,
`max_position_embeddings 24576`, **`timestep_shift 1.0`**.
Generation defaults (`yue2_generation_config.json`): `ode_steps 32`,
`ode_method "midpoint"`, `context 24576`.

### 1.1 Chunking — `song_chunks()` / `chunk_ranges()`

`nar.py:34` `song_chunks(prefix, codec, seed, context=CONTEXT)`:

```
size   = min((context - len(prefix) - 3) // 2, CONTEXT)          # protocol.py:142
ranges = [(a, min(a + size, T)) for a in range(0, T, size)]      # protocol.py:145
noise  = randn([T, 64], f32, generator=Generator("cpu").manual_seed(seed))   # nar.py:45-46
chunks = [Chunk(ar_tokens = prefix + [c + CODEC_OFFSET for c in codec[a:b]] + [MUSIC_END],
                noise     = noise[a:b],
                nar_cond_end = 0)                                # nar.py:47-48
          for (a, b) in ranges]
```

Reading, stated flatly because the C++ must get it right:

- **No overlap. No crossfade. No carry-over of latents between chunks.** The
  ranges tile `[0, T)` with stride `size`; the last one is short.
- Every chunk gets **the same text prefix** and **only its own codec slice**,
  followed by a single `MUSIC_END`. Chunk `k > 0` does not see chunk `k−1`'s
  codec tokens or latents at all.
- The noise is drawn **once for the whole song** and then sliced. That is why a
  song that happens to fit in one chunk and the same song re-chunked produce
  different results only through the AR context, never through the noise.
- Stitching (`nar.py:261`) is `torch.cat(outputs, dim=0)` — plain concatenation
  along the frame axis.
- `nar_cond_end` is `0` for every chunk produced by `song_chunks()` (it is the
  dataclass default, `nar.py:24`, and nothing in the inference path sets it).
  It is a training-time codec-dropout knob. **Decision: `yue2-nar` does not
  implement it and must reject a nonzero value with a clear error.**

`alley_swing_s1` is therefore **a single chunk**: `size = (24576 − 2549 − 3)//2
= 11012 > 4109`, so `ranges == [(0, 4109)]`, `ar_tokens` has
`2549 + 4109 + 1 = 6659` entries and `nar_length = 4111`; `S = 10770 ≤ 24576`.
The multi-chunk path is therefore **untested by the primary song** — the C++
agent must construct a synthetic small-`context` run (`--context 1200`, see §5)
to exercise it.

`CachedNAR.__init__` (`nar.py:114-121`) additionally enforces
`ar_length + nar_length ≤ 24576` and `min(ar_tokens) ≥ 0`,
`max(ar_tokens) < 184704`.

### 1.2 Noise initialisation — bit-reproducibility

`nar.py:45-46`:

```
generator = torch.Generator(device="cpu").manual_seed(int(seed))
noise     = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
```

This is ATen's CPU normal kernel, i.e.:

1. `torch.Generator(device="cpu")` is a **Mersenne Twister MT19937**
   (`at::mt19937`), seeded by `init_with_uint32(uint32_t(seed))` — the classic
   `init_genrand` recurrence, not `init_by_array`.
2. `randn` on a contiguous float tensor of `n = T·64 ≥ 16` elements dispatches to
   ATen's `normal_fill<float>`: first the whole buffer is filled with uniforms
   `u ∈ [0,1)` produced as `(next_uint32() & ((1<<24) − 1)) · 2^-24`, in
   C-order; then each aligned block of 16 elements is converted in place by
   `normal_fill_16`:

	```
	for j in 0..7:
		u1 = 1 - data[j]
		u2 = data[j + 8]
		r  = sqrt(-2 * log(u1))
		th = 2 * pi * u2
		data[j]     = r * cos(th)
		data[j + 8] = r * sin(th)
	```

   If `n % 16 != 0` the **last 16 elements are re-drawn and re-converted** (the
   tail fixup) — here `n = 64·T` is always a multiple of 16, so the tail path
   never runs for this model.

**Decision.** Replicating that bit-exactly in C++ is possible but it depends on
an ATen implementation detail (the `normal_fill` blocking and the exact uniform
quantisation), not on a documented API, and a silent one-bit drift would be
diagnosed as an ODE bug. So:

- **The contract path is a file.** `yue2-nar` takes `--noise noise.npy`
  (float32 `[T, 64]`) and every golden ships the noise it used
  (`tests/golden/nar_noise*.npy`, and for the full song
  `tests/out/nar_noise_alley_swing_s1.npy`, produced by `convert/reference_nar.py
  --dump-noise`). The converter agent owns noise generation.
- `--seed N` is **optional** for the C++ agent. If implemented, it must
  reproduce `tests/golden/nar_noise_short.npy` **bit-exactly** for that seed and
  length, verified by a test; if it does not, the flag must be removed rather
  than shipped approximately. A binary built without it must error on `--seed`
  with "seed-driven noise not implemented; pass --noise".
- Consequence for §3/§5: **`songs/.../latent.npy` is not bit-reproducible** by
  anything here regardless of RNG — see §5.3.

### 1.3 The ODE solver

`CachedNAR.solve` (`nar.py:169-198`), `steps = 32` from the generation config:

```
state = noise                         # [Tc, 64], cast to the model dtype
dt    = 1.0 / steps                   # python double
for k in 0 .. steps-1:
	t      = 1.0 - k * dt                                  # nar.py:185
	raw    = clamp(logit_f64(t), -20, 20)                  # nar.py:186
	v1     = velocity(state, raw)                          # nar.py:187
	mid    = state - v1 * (dt / 2)                         # nar.py:188
	raw2   = clamp(logit_f64(t - dt/2), -20, 20)           # nar.py:191
	state  = state - velocity(mid, raw2) * dt              # nar.py:192
result = state.float()                                         # nar.py:195
```

Exactly: `logit_f64(x) = log(x / (1 − x))` evaluated in **float64**
(`torch.logit(torch.tensor(t, dtype=torch.float64))`), then `.clamp(-20, 20)`,
then `.item()` to a python double. This is a **midpoint (RK2) scheme integrating
backwards from t = 1 (noise) to t = 0 (data)**, with `2·steps = 64` velocity
evaluations per chunk. There is no adaptive stepping, no solver state beyond
`state`. Non-finite output raises (`nar.py:196`).

The timestep the network actually sees is `_shift_t_value`
(`modeling_yue2.py:603-606`) with `timestep_shift = 1.0`:

```
t_sig    = sigmoid(tensor(raw, dtype=<model dtype>))          # NOTE: cast to model dtype first
shifted  = shift * t_sig / (1 + (shift - 1) * t_sig)  ==  t_sig   when shift == 1.0
```

So `t_shifted = sigmoid(fp32(raw))`, i.e. the logit/sigmoid pair is an identity
**except at k = 0**, where `t = 1` → `logit = +inf` → clamped to `20.0` →
`sigmoid(20) = 0.999999998…`, which **rounds to exactly `1.0f` in float32**.
Practical consequence: with `steps = 32` the 64 shifted timesteps are exactly

```
k = 0 .. 31:   t_shifted[2k]   = 1 - k/32   (with 1.0 at k = 0)
               t_shifted[2k+1] = 1 - k/32 - 1/64
```

**Implement the general expression anyway** (`raw` in double, clamp, then
`sigmoid` evaluated in **float32 on the float32-rounded raw**, to match torch's
`torch.tensor(raw, dtype=torch.float32)` then `sigmoid`), so a `--steps` other
than 32 still matches.

### 1.4 `velocity()` — one network evaluation

`CachedNAR.velocity` (`nar.py:151-167`). With `Tc` = frames in this chunk,
`N = Tc + 2` = `nar_length`, `H = 2048`, `D = 64`:

```
x_nar = pad(state, (0,0,1,1))                 # nar.py:156 → [N, 64]; row 0 and row N-1 are ZERO
sh    = _shift_t_value(raw_t)                 # nar.py:157, scalar
x     = vae2llm(x_nar)                        # nar.py:158 → [N, 2048], Linear WITH bias
x     = x + time_embedder(sh.expand(N))       # nar.py:159 → same 2048-vector added to every row
x     = x + pos_emb                           # nar.py:160 → latent_pos_embed[0 .. N-1]
for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):     # nar.py:161
	q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
	k, v    = cat(ar_k, k), cat(ar_v, v)      # nar.py:163 → keys/values [ar_length + N, 8, 128]
	h       = attention(q, k, v, causal=False)# nar.py:164 → NO MASK AT ALL
	x       = x + layer.nar_self_attn.o_proj(h.flatten(1))           # nar.py:165
	x       = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))      # nar.py:166
return llm2vae(model.model.norm(x))[1:-1]     # nar.py:167 → [Tc, 64]
```

Component formulas (all from `modeling_yue2.py`):

| piece | line | formula |
|---|---|---|
| `RMSNorm` | 132 | `y = x * rsqrt(mean(x².float(), dim=-1) + 1e-6) * w` — the reduction is in **float32** even in a bf16 run |
| `project_qkv` | 174-189 | `q = q_norm(Wq x).view(N,16,128)`, `k = k_norm(Wk x).view(N,8,128)`, `v = (Wv x).view(N,8,128)`; **q_norm/k_norm are RMSNorm over head_dim, applied BEFORE RoPE**; then RoPE on q and k, not on v. No biases. |
| RoPE | 142-156 | `inv_freq[i] = 1 / 1e6^(2i/128)`, `i = 0..63`; `angles = pos * inv_freq`; **NEOX / rotate-half**: `x1 = x[..., :64]`, `x2 = x[..., 64:]`, `out = cat(x1·cos − x2·sin, x2·cos + x1·sin)`. Identical to the AR half (SPEC_AR §1). |
| attention | `nar.py:95` | `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=True)` → softmax(`q·kᵀ / sqrt(128)`)·v, GQA with 16 q-heads over 8 kv-heads (kv head `j` serves q heads `2j, 2j+1`). |
| MLP | 224 | `down(silu(gate(x)) * up(x))`, no biases |
| `TimestepEmbedder` | 312-331 | `half = 128`; `freqs[i] = exp(−log(10000)·i/128)`, `i = 0..127`; `args = t·freqs`; `emb = cat(cos(args), sin(args))` → `[256]`; then `Linear(256→2048) → SiLU → Linear(2048→2048)`, **both with bias**. |
| `AudioPositionEmbedding` | 334-349 | fixed sinusoid table `pe[p, 2i] = sin(p·d_i)`, `pe[p, 2i+1] = cos(p·d_i)` with `d_i = exp(2i·(−log(10000)/2048))` — **interleaved**, unlike the timestep embedder's concatenated layout. Indexed by `local = arange(N).clamp(max=24575)` (`nar.py:124`) — the clamp can never fire because `N ≤ 24576`, but implement it. |
| `vae2llm` | 472 | `Linear(64 → 2048, bias=True)` |
| `llm2vae` | 471 | `Linear(2048 → 64, bias=True)` |
| final norm | `nar.py:167` | `model.model.norm` — the **shared** backbone `RMSNorm(2048)`, the same tensor the AR half uses before `lm_head`. |

Two points the C++ agent will otherwise get wrong:

- **Rows 0 and N−1 are not skipped.** `vae2llm` has a bias, so the zero latent
  rows still produce a nonzero hidden state; they carry the time and position
  embeddings and participate in attention as ordinary tokens. Only the *output*
  slice drops them (`[1:-1]`).
- **The time embedding is a single 2048-vector broadcast to all N rows.**
  `sh.expand(N)` makes N identical scalars; compute it once per velocity call.

### 1.5 RoPE positions

Two independent position encodings, do not conflate them:

- **RoPE** for the NAR tokens: `positions = arange(ar_length, ar_length + N)`
  (`nar.py:122`) — the NAR block continues the AR position axis. `cos`/`sin` are
  computed once in `__init__` (`nar.py:123`) and reused for all 64 velocity
  evaluations. The AR keys in the cache already carry their own RoPE at
  positions `0 .. ar_length−1` (`nar.py:136-140`).
- **Latent position embedding**: `arange(N)` — restarts at 0 at the first NAR
  token (the zero-padded START row), independent of `ar_length` (`nar.py:124`).

### 1.6 The "hybrid mask" — and why there isn't one

`modeling_yue2.py:668-685` documents the intended mask for the fused
training-shaped path:

```
AR → AR : causal        AR → NAR : blocked
NAR → AR: full          NAR → NAR: bidirectional (full)
```

`CachedNAR` realises exactly that geometry **without ever materialising a
mask**:

- AR→AR causal and AR→NAR blocked are realised by `_prefill()` running the AR
  tokens alone through a causal attention (`nar.py:147`, `causal=True`) and
  never revisiting them.
- NAR→AR full and NAR→NAR bidirectional are realised by concatenating the cached
  AR K/V in front of the NAR K/V and calling attention with
  `attn_mask=None, is_causal=False` (`nar.py:163-164`).

**Decision: the C++ implements the `CachedNAR` reading. There is no mask tensor
in the NAR graph — `ggml_soft_max_ext(kq, NULL, 1/sqrt(128), 0.0f)`.** A causal
mask is needed only inside the AR prefill.

(For the record, the fused `YuE2ForCausalLM.nar_velocity` at
`modeling_yue2.py:608-700` and the `torch.where` MoT routing in
`DecoderLayer.forward` at `modeling_yue2.py:256-295` are **dead code for
inference** — nothing in `pipeline.py` calls them. They are mathematically
equivalent to `CachedNAR`: AR positions never attend to NAR positions, so their
hidden states — and hence their K/V — are the same as in the AR-only prefill;
and the per-position `torch.where` merge of the AR/NAR O-projections only
affects AR rows, which the NAR output slice discards. Where the two disagree in
spirit, **`nar.py` is normative**, because that is what produced `latent.npy`.)

### 1.7 MoT — which weights are AR, which are NAR, which are shared

Per `DecoderLayer.__init__` (`modeling_yue2.py:230-243`) each of the 28 layers
holds **two complete sets**:

| role | tensors (per layer) | used by |
|---|---|---|
| AR half | `input_layernorm`, `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj` | `_prefill()` only |
| NAR half | `nar_input_layernorm`, `nar_self_attn.{q,k,v,o}_proj`, `nar_self_attn.{q,k}_norm`, `nar_pre_mlp_layernorm`, `nar_mlp.{gate,up,down}_proj` | `velocity()` only |

Shared between the halves: `model.embed_tokens` (prefill only), `model.norm`
(NAR output only, `nar.py:167`), and the RoPE formula/`inv_freq` (not a stored
tensor). NAR-only auxiliaries: `vae2llm`, `llm2vae`, `time_embedder.mlp.{0,2}`,
`latent_pos_embed.pe`. **Never used by stage 3: `lm_head`.**

Token routing is therefore *positional and static*: AR token → AR weights,
NAR token → NAR weights, with the only coupling being the K/V concatenation in
`nar.py:163`. No per-token `where`, no gating, no router.

### 1.8 `_prefill()` — the AR context

`nar.py:132-149`:

```
ids       = chunk.ar_tokens                       # [ar_length]
positions = arange(ar_length)
cos, sin  = rotary_emb(positions)
x         = embed_tokens(ids)                     # no scaling
for layer in layers:                              # AR weights only
	q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
	cache.append((k, v))                          # POST-RoPE k, raw v; [ar_length, 8, 128] each
	h = attention(q, k, v, causal=True)
	x = x + layer.self_attn.o_proj(h.flatten(1))
	x = x + layer.mlp(layer.post_attention_layernorm(x))
```

- This is **exactly the qwen3 / SPEC_AR §1 block**, run once per chunk over the
  full `ar_tokens`.
- What is kept is the **post-RoPE K and the raw V of every layer**, 28 pairs of
  `[ar_length, 8, 128]`. The prefill's final hidden state, `model.norm`, and
  `lm_head` are **not** used — no logits are produced.
- `self.visible_length` (`nar.py:121`) equals `ar_length` because
  `nar_cond_end == 0`, so the `clone()` in `nar.py:143-145` is a no-op.
- **The AR KV is built once and reused unchanged for all 64 velocity
  evaluations of the chunk** ("AR prefix KV is invariant during the ODE",
  `nar.py:103`). It is freed at `close()` between chunks (`nar.py:200`).

### 1.9 dtype — what the reference actually uses

- The production run that produced `songs/.../latent.npy` loaded the model as
  **bfloat16** (`pipeline.py:216`: `torch_dtype=torch.bfloat16`) on the ROCm
  GPU, with `allow_tf32=False`, `float32_matmul_precision("highest")`,
  `cudnn.deterministic=True` (`pipeline.py:138-144`).
- `CachedNAR` takes its dtype from `model.vae2llm`'s weight (`nar.py:108-109`),
  so **every activation, the ODE state, the KV cache and `t_shifted` are bf16**
  in that run; only the final `state.float()` (`nar.py:195`) returns f32.
- **Our goldens are torch CPU `float32`** (§3), because SPEC.md §6 forbids GPU
  torch and CPU bf16 matmul is both slow and not the accuracy we want to certify
  against. Loading the bf16 checkpoint as f32 is lossless (every bf16 value is
  exactly representable in f32).
- **The C++ runs f32 activations throughout.** The weights are stored F16, which
  for this checkpoint is also lossless — see §2.

---

## 2. GGUF

### 2.1 Decision: two GGUFs, no duplication

`yue2-nar` loads **both**:

1. **`yue2-ar-{f16,q8_0}.gguf`** — the existing stage-2 file, unchanged. It
   already contains every AR-half tensor the prefill needs plus
   `token_embd.weight` and `output_norm.weight` (= `model.norm`, which the NAR
   output path also uses). Nothing to re-convert, and stage 2's acceptance tests
   keep guarding it.
2. **`yue2-nar-f16.gguf`** — new, **NAR-specific tensors only** (§2.2).

Rejected alternative: a single self-contained NAR GGUF that duplicates the AR
half. That would duplicate ~1.79 B parameters (≈ 1.9 GB at Q8_0, 3.6 GB at F16)
on disk for no benefit, and in the **stage-4 single binary** — which runs AR
generation on libllama from `yue2-ar-*.gguf`, then NAR, then VAE, in one process
— it would mean shipping and possibly holding two copies of the same weights.
The two-file split keeps exactly one copy of every parameter across the whole
pipeline (`ar 1.79 B + nar 1.46 B + vae`), and the price is ~30 lines of a second
`gguf_init_from_file` call.

Guard against a mismatched pair: both files carry `yue2.source_sha256` (SPEC_AR
§2 already writes it into the AR GGUF); `yue2-nar` **must** compare them and
abort if they differ.

### 2.2 Tensors — `convert/convert_nar.py`

`general.architecture = "yue2-nar"`, `general.name = "YuE2-3B NAR"`.
`N = 0 .. 27`. Shapes are given as **torch** shapes; gguf-py writes numpy arrays
row-major, so torch `[out, in]` appears in ggml as `ne = [in, out]`, which is
what `ggml_mul_mat(w, x)` wants. **No transposes in the converter** (same rule
as SPEC.md §2).

| safetensors | gguf | torch shape | dtype |
|---|---|---|---|
| `model.layers.N.nar_input_layernorm.weight` | `blk.N.nar_attn_norm.weight` | [2048] | F32 |
| `model.layers.N.nar_self_attn.q_proj.weight` | `blk.N.nar_attn_q.weight` | [2048, 2048] | F16 |
| `model.layers.N.nar_self_attn.k_proj.weight` | `blk.N.nar_attn_k.weight` | [1024, 2048] | F16 |
| `model.layers.N.nar_self_attn.v_proj.weight` | `blk.N.nar_attn_v.weight` | [1024, 2048] | F16 |
| `model.layers.N.nar_self_attn.o_proj.weight` | `blk.N.nar_attn_output.weight` | [2048, 2048] | F16 |
| `model.layers.N.nar_self_attn.q_norm.weight` | `blk.N.nar_attn_q_norm.weight` | [128] | F32 |
| `model.layers.N.nar_self_attn.k_norm.weight` | `blk.N.nar_attn_k_norm.weight` | [128] | F32 |
| `model.layers.N.nar_pre_mlp_layernorm.weight` | `blk.N.nar_ffn_norm.weight` | [2048] | F32 |
| `model.layers.N.nar_mlp.gate_proj.weight` | `blk.N.nar_ffn_gate.weight` | [6144, 2048] | F16 |
| `model.layers.N.nar_mlp.up_proj.weight` | `blk.N.nar_ffn_up.weight` | [6144, 2048] | F16 |
| `model.layers.N.nar_mlp.down_proj.weight` | `blk.N.nar_ffn_down.weight` | [2048, 6144] | F16 |
| `vae2llm.weight` | `nar.vae2llm.weight` | [2048, 64] | F32 |
| `vae2llm.bias` | `nar.vae2llm.bias` | [2048] | F32 |
| `llm2vae.weight` | `nar.llm2vae.weight` | [64, 2048] | F32 |
| `llm2vae.bias` | `nar.llm2vae.bias` | [64] | F32 |
| `time_embedder.mlp.0.weight` | `nar.time_embd.0.weight` | [2048, 256] | F32 |
| `time_embedder.mlp.0.bias` | `nar.time_embd.0.bias` | [2048] | F32 |
| `time_embedder.mlp.2.weight` | `nar.time_embd.1.weight` | [2048, 2048] | F32 |
| `time_embedder.mlp.2.bias` | `nar.time_embd.1.bias` | [2048] | F32 |
| `latent_pos_embed.pe` | `nar.latent_pos_embd.weight` | [24576, 2048] | F16 |

That is `11 × 28 + 9 = 317` tensors. Total ≈ 1.46 B parameters ≈ 2.9 GB at F16.
Every source tensor in `model.safetensors` is **BF16** (verified from the
safetensors header; all 628 entries are BF16).

**Why F16 and not Q8_0 for the default.** BF16 has an 8-bit significand and F16
has an 11-bit significand over a narrower exponent range. Every weight in this
checkpoint is well inside `[6.1e-5, 65504]` in magnitude, so **BF16 → F16 is
exact**, and an F16 GGUF *is* the checkpoint, bit for bit, not an approximation
of it. Q8_0 is not: it is a real 8-bit quantisation on top. A `--type q8_0`
option may be produced for memory comparisons, but **acceptance in §5 is against
F16**, and a Q8_0 run is report-only. (`--type f32` may also be offered; it
doubles the file for zero accuracy gain.)

**`latent_pos_embed.pe` must be shipped, not recomputed.** It is a
non-learnable sinusoid, but the checkpoint stores it in BF16 and `from_pretrained`
loads those stored values over the freshly computed ones, so the torch reference
— including our f32 CPU golden — uses BF16-rounded sinusoids. Recomputing them in
F32 would introduce a ~4e-3 relative difference on an additive term. Store the
tensor as F16 (exact, as above). The converter **must** assert that the stored
`pe` agrees with the formula in `modeling_yue2.py:339-346` to within BF16
rounding (`max|Δ| < 8e-3`), as a check that the layout reading (interleaved
sin/cos) is right.

### 2.3 Metadata

Under prefix `yue2nar.`:
`block_count 28`, `embedding_length 2048`, `feed_forward_length 6144`,
`attention.head_count 16`, `attention.head_count_kv 8`,
`attention.key_length 128`, `attention.value_length 128`,
`attention.layer_norm_rms_epsilon 1e-6`, `rope.freq_base 1e6`,
`context_length 24576`, `latent_dim 64`, `max_latent_frames 24576`,
`timestep_shift 1.0`, `time_embd_frequency_size 256`,
`ode_steps 32`, `ode_method "midpoint"`.
Plus `yue2.source_sha256` = the sha256 of `model.safetensors`
(`1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9`, and
`weights_manifest.json` carries the same) — **must** match the AR GGUF's.

Converter environment and conventions: reuse `convert/common.py`
(`resolve_snapshot`, `sha256_of`, the closing tensor table) and run under
`../venv_yue2/bin/python`, exactly as `convert_ar.py` does.
Output → `tests/out/yue2-nar-f16.gguf` (gitignored).

---

## 3. Goldens — torch CPU only

`convert/reference_nar.py`, in the style of `convert/reference_ar.py`: load
`YuE2ForCausalLM` from the snapshot on **CPU in float32**
(`torch_dtype=torch.float32, low_cpu_mem_usage=True`), `.eval()`, and **assert
`next(model.parameters()).device.type == "cpu"` after load** regardless of what
`torch.cuda.is_available()` claims. Export `CUDA_VISIBLE_DEVICES=""` and
`HIP_VISIBLE_DEVICES=""` in the script itself. Deleting `model.lm_head` right
after load saves 1.5 GB of the ~14.5 GB f32 footprint (the box has 62 GB total,
~39 GB available — fine, but do it anyway).

### 3.1 The short test

Built from the `alley_swing_s1` artifacts (read-only) so it exercises real token
distributions:

```
prefix_short = concat(prefix.npy[:255], [151851])      # 256 ids, still ends in MUSIC_START
codec_short  = semantic.npy[:128]                      # 128 codes, all in [0, 32768)
seed         = 1                                       # from request.json
steps        = 32 (primary) and 2 (fast-iteration variant)
context      = 24576   → one chunk: size = (24576-256-3)//2 = 12158 > 128
```

so `ar_length = 256 + 128 + 1 = 385`, `N = 130`, `S = 515`.

Drive it through the real code path — `nar.song_chunks(...)` then
`nar.CachedNAR(model, chunk)` then `.solve(steps)` — with monkeypatched hooks
(or a copy of `velocity` in the script) to capture the intermediates. Do **not**
reimplement the math in the golden script; a golden that shares a bug with the
C++ is worthless.

### 3.2 Files (all float32 C-order unless stated, under `tests/golden/`)

| file | shape | what |
|---|---|---|
| `nar_prefix_ids.npy` | int32 [256] | `prefix_short` |
| `nar_codec_ids.npy` | int32 [128] | `codec_short` |
| `nar_noise.npy` | [128, 64] | `chunk.noise`, i.e. the exact torch draw for `seed = 1`, `T = 128` — **the C++'s `--noise` input** |
| `nar_ar_kv_l0.npy` | [2, 385, 8, 128] | layer-0 AR cache after `_prefill`: `stack(cache[0][0], cache[0][1])` — post-RoPE K and raw V |
| `nar_ar_kv_l27.npy` | [2, 385, 8, 128] | same for the last layer (catches error accumulation through the prefill) |
| `nar_x_in_step0.npy` | [130, 2048] | `velocity()`'s `x` after `vae2llm + time_emb + pos_emb`, first call (`raw = 20.0`) — i.e. the state entering layer 0 |
| `nar_x_l0_step0.npy` | [130, 2048] | `x` after NAR layer 0 of that same call |
| `nar_v_step0.npy` | [128, 64] | the returned velocity of that same call |
| `nar_latent_s2.npy` | [128, 64] | `solve(steps=2)` result |
| `nar_latent_s32.npy` | [128, 64] | `solve(steps=32)` result — the primary acceptance target |
| `nar_tshift.npy` | float64 [2·steps] | the `raw` values fed to `velocity`, in order, for `steps = 32` — a 1-line check that the schedule matches |

Also write `tests/golden/nar_short_meta.json`: `{prefix_len, codec_len, ar_length,
nar_length, seed, steps, source_sha256, torch_version, wall_seconds}`.

### 3.3 CPU cost estimate

Per NAR layer: `q/k/v/o = 12.6 M` + `mlp = 37.7 M` ≈ **50.3 M** parameters, so
`2 × 50.3 M × 28 ≈ 2.82 GFLOP` per NAR token per velocity evaluation; the AR half
is the same per AR token. For the short test:

- prefill: `385 × 2.82 GFLOP ≈ 1.1 TFLOP`, once.
- each velocity: `130 × 2.82 GFLOP ≈ 0.37 TFLOP`; attention is negligible at
  `S = 515`.
- `steps = 32` → 64 evaluations → `≈ 23 TFLOP`; `steps = 2` → `≈ 1.5 TFLOP`.

On the i9-11900K (16 threads, f32 MKL sgemm, ~200–400 GFLOP/s) that is roughly
**1–2 minutes for the 32-step golden**, seconds for the 2-step one, plus ~1–2
minutes to load and upcast 7.26 GB of BF16. **Budget 5 minutes total.** If it
comes out much worse, report the number rather than shrinking the test.

### 3.4 The optional full-song CPU golden

`tests/out/nar_full_cpu_latent.npy` — the whole `alley_swing_s1` chunk
(`P = 2549`, `T = 4109`, `steps = 32`) on torch CPU f32, plus
`tests/out/nar_noise_alley_swing_s1.npy` (the `[4109, 64]` draw for `seed = 1`).
Cost: `64 × 4111 × 2.82 GFLOP ≈ 742 TFLOP` of GEMM plus ~23 TFLOP of attention at
`S = 10770` → **roughly 40–120 minutes**. Run it once, in the background, `nice`d;
it is not on the critical path but §5.3 wants it. It is a `tests/out/` artifact,
not a checked-in golden (goldens are gitignored anyway — `.gitignore` has
`*.npy`; they are regenerated locally).

### 3.5 The multi-chunk golden (mandatory, cheap)

`alley_swing_s1` fits in one chunk, so nothing above exercises `chunk_ranges`.
Produce `tests/out/nar_multichunk_cpu.npy` and
`tests/out/nar_multichunk_noise.npy` by calling the real
`nar.synthesize(model, prefix_short, semantic.npy[:512], seed=1, steps=2,
context=1200)`. That gives `size = (1200 − 256 − 3)//2 = 470` and two chunks,
`(0,470)` and `(470,512)`; output `[512, 64]`. Cost: 2 prefills of ~727 and
~299 tokens plus `2 × 4` velocity evaluations over `N = 472` and `N = 44` →
well under a minute. Record the chunk ranges and both `ar_length` values in
`nar_short_meta.json` so the C++ agent can check its chunking without rerunning
torch.

### 3.6 Is `songs/.../latent.npy` reproducible?

**No — not bit-exactly, and not even close to f32 precision.** Reasons, in
descending order of magnitude:

1. That run was **bf16 end to end** on a GPU (`pipeline.py:216`): the ODE state,
   the AR KV cache and every activation carried 8 significand bits.
2. It ran ROCm SDPA with a fused kernel and its own reduction order, and
   `attention()` used `query_chunk_size = len(q)` on CUDA versus 256 on CPU
   (`nar.py:70`).
3. Only then does the RNG question arise — and the RNG *is* reproducible here,
   because `song_chunks` draws on the **CPU** generator regardless of the model
   device (`nar.py:45`), so the same `seed = 1` gives the same noise in our CPU
   golden as it did in that run.

So the seed and the noise are shared with the artifact; nothing downstream of
them is. Expect the f32 latent to differ from `latent.npy` by roughly bf16
precision compounded over 64 evaluations — **treat `latent.npy` as a plausibility
reference (§5.3), never as a golden**.

---

## 4. The C++ program — `src/yue2-nar.cpp`, binary `yue2-nar`

### 4.1 Should it link libllama? — No.

libllama could run the AR prefill (`llama_decode` over `ar_tokens` on the qwen3
graph), but stage 3 needs **the per-layer K and V tensors themselves**, and
libllama offers no supported way to get them: `llama_get_embeddings*` returns
only the final hidden state, and `llama_state_seq_get_data` serialises an
internal, version-unstable cell layout (cell map, per-sequence rearrangement,
the cache's own type which may be F16 or quantised) that we would have to parse
and would silently break on a submodule bump. Meanwhile the NAR half needs a
hand-written qwen3-shaped block in raw ggml **anyway** — attention over a
concatenated `[AR KV ; NAR KV]` with no mask, GQA, and a different weight set
per position class — and the AR prefill is *the same block code with the AR
tensors and a causal mask*. Reuse cost of doing the prefill ourselves: near zero.

**Decision: `yue2-nar` is raw ggml only. It does not link libllama.** It reads
both GGUFs with `gguf_init_from_file` + `ggml_backend_alloc_ctx_tensors`, the
same pattern `src/yue2-vae.cpp` already uses. CMake: `add_executable(yue2-nar
src/yue2-nar.cpp)`, `target_link_libraries(yue2-nar PRIVATE ggml)` — note it
links `ggml` only, like `yue2-vae`, not `llama`.

Stage-4 consequence, stated so nobody is surprised: the single binary will run
the AR *generation* through libllama and then re-run the prefix through the raw
ggml prefill to build the NAR's AR KV. That is one extra ~13 k-token prefill,
about a second on Vulkan, in exchange for not depending on libllama internals.

### 4.2 CLI

```
yue2-nar --ar yue2-ar-f16.gguf -m yue2-nar-f16.gguf
         (--artifacts DIR | --prefix prefix.npy --codec semantic.npy)
         (--noise noise.npy | --seed N)
         -o latent.npy
         [--steps 32] [--context 24576] [--query-chunk 1024]
         [--device cpu|vulkan] [--gpu N] [--threads N]
         [--frames N]            # truncate the codec to the first N frames (golden tests)
         [--dump-dir DIR]        # write the §3.2 intermediates as .npy for comparison
         [--vk-f16-matmul]       # restore the lossy fast matmul path, for comparison only
```

- `--artifacts DIR` reads `DIR/prefix.npy`, `DIR/semantic.npy` and
  `DIR/request.json` (for `seed`), and writes `DIR/latent.npy` unless `-o` says
  otherwise. This is the form `yue2_gen.py --nar vulkan` will call.
- `.npy` I/O through the existing `src/npy.hpp` (`load_i32` / `load` / `save`).
  `prefix.npy` and `semantic.npy` are `'<i4'` 1-D; `--noise` is `'<f4' [T, 64]`.
- **Output layout is fixed: `latent.npy`, float32, C-order, shape `[T, 64]`**,
  `T` = number of input codec frames (after `--frames`). That is exactly what
  `yue2-vae -i` accepts (SPEC.md §4) and what `pipeline.decode()` expects
  (`pipeline.py:338`: "Expected latents [T,64] or [1,64,T]").
- Validate on entry: `0 ≤ codec[i] < 32768`, `0 ≤ prefix[i] < 184704`,
  `T ≥ 1`, `1 ≤ context ≤ 24576`, `steps ≥ 1`, noise shape `[T, 64]` and all
  finite, `yue2.source_sha256` equal in both GGUFs. Non-zero exit with a clear
  message on any of them. Refuse `nar_cond_end`-style options entirely (§1.1).
- Print, per chunk: `ar_length`, `nar_length`, prefill seconds, seconds per ODE
  step, total; and the backend/device name — same reporting style as `yue2-vae`.

### 4.3 Backend init — the fp16 trap

`docs/vulkan_burst_investigation.md` applies verbatim: ggml-vulkan's `mul_mm`
stages **both** operands as `float16_t` whenever the device advertises fp16, and
the KHR_coopmat path is fp16 by construction, even for an `F32 × F32`
`GGML_OP_MUL_MAT`. This graph is ~1700 matmuls per velocity evaluation and the
ODE integrates 64 of them, so the loss compounds rather than cancels.
`yue2-nar` **must** contain this block, copied from `src/yue2-vae.cpp:614-629`,
before `ggml_backend_load_all()`:

```
	// ggml-vulkan's mul_mm shader stages BOTH operands as float16_t whenever the
	// device advertises fp16 (and the KHR_coopmat path is fp16 by construction),
	// even for an F32 x F32 matmul. Every Conv1d here is im2col + mul_mat, so that
	// silently costs ~1.7e-4 relative error per conv, which the decoder's ~40
	// SnakeBeta non-linearities amplify into a ~65 dB noise floor with occasional
	// ~85 ms bursts. Disabling both selects the genuinely-F32 matmul pipelines.
	// Must happen before the backend is initialised; a value already in the
	// environment wins (overwrite = 0). See docs/vulkan_burst_investigation.md.
	if (p.device == "vulkan" && !p.vk_f16_matmul)
	{
		setenv("GGML_VK_DISABLE_F16", "1", 0);
		setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
	}

	ggml_backend_load_all();
```

(Adapt the comment's second and third sentences to this graph — "every attention
and MLP projection here is a mul_mat, and the ODE integrates 64 evaluations of a
28-layer transformer, so operand truncation compounds" — but keep the rule, the
`overwrite = 0` semantics, the pointer to the doc, and `--vk-f16-matmul` as the
escape hatch for measuring what it costs.)

**Weight dtype: F16 stored, F32 activations, F32 KV cache.** As §2.2 argues, F16
*is* the BF16 checkpoint exactly, so there is no accuracy argument for F32
weights and a 2× memory argument against. Q8_0 (via `--ar yue2-ar-q8_0.gguf` for
the AR half, or a Q8_0 NAR file) is permitted but is report-only in §5; note in
STATUS what it costs in dB.

### 4.4 Graph structure

Tensors are described in ggml `ne` order (`ne[0]` fastest).

**Persistent backend tensors, allocated once per chunk** (not in the compute
graph, so `ggml_gallocr` never touches them):

- `ar_k[l]`, `ar_v[l]` for `l = 0..27`: `ne = [128, ar_length, 8]` F32 each.
  `ar_k` holds post-RoPE keys, `ar_v` raw values.

**Phase A — AR prefill, once per chunk.** Because a single graph over up to
~13 k AR tokens would materialise a `16 × ar_len × ar_len` score matrix
(10 GB+), run it **chunked**: blocks of `--prefill-block` (default 512) tokens,
each block's graph reading the already-written `ar_k/ar_v[0 : block_end]` through
`ggml_view_3d` and using a causal mask of shape `[ar_length_so_far, block]`
(`ggml_soft_max_ext(kq, mask, 1/sqrt(128), 0.0f)`, mask F32 with `-INFINITY`
above the diagonal, `ggml_new_tensor` fed from the host each block). Each block's
graph:

```
	x = get_rows(token_embd, ids_block)                        # [2048, B]
	for l in 0..27:
		h  = rms_norm(x, 1e-6) * blk.l.attn_norm
		q  = mul_mat(blk.l.attn_q, h)      -> reshape [128, 16, B]
		k  = mul_mat(blk.l.attn_k, h)      -> reshape [128,  8, B]
		v  = mul_mat(blk.l.attn_v, h)      -> reshape [128,  8, B]
		q  = rms_norm(q, 1e-6) * blk.l.attn_q_norm             # per head, over 128
		k  = rms_norm(k, 1e-6) * blk.l.attn_k_norm
		q  = rope_neox(q, pos_block, 128, 1e6)
		k  = rope_neox(k, pos_block, 128, 1e6)
		cpy(k -> view of ar_k[l][.., off..off+B, ..])
		cpy(v -> view of ar_v[l][.., off..off+B, ..])
		kq = mul_mat(view(ar_k[l], 0..off+B), permute(q))       # GQA broadcast 16 over 8
		kq = soft_max_ext(kq, causal_mask, 1/sqrt(128), 0)
		o  = mul_mat(permute(view(ar_v[l], 0..off+B)), kq)
		x  = x + mul_mat(blk.l.attn_output, reshape(o, [2048, B]))
		h2 = rms_norm(x, 1e-6) * blk.l.ffn_norm
		x  = x + mul_mat(blk.l.ffn_down, silu(mul_mat(blk.l.ffn_gate, h2)) * mul_mat(blk.l.ffn_up, h2))
```

`ggml_rope_ext` with `mode = GGML_ROPE_TYPE_NEOX`, `n_dims = 128`,
`freq_base = 1e6`, `freq_scale = 1`, everything else neutral — the same settings
llama.cpp's qwen3 uses, and the same convention `_apply_rotary`
(`modeling_yue2.py:152-156`) implements. Verify the convention against
`nar_ar_kv_l0.npy` **before** building anything else; a rotate-half vs
interleaved mix-up is the single most likely silent bug here.

`ggml_mul_mat` broadcasts `ne2` when the divisor is exact, which is how GQA
(16 q-heads over 8 kv-heads) is expressed — the same trick llama.cpp's
`build_attn_mha` uses. Do not `repeat_interleave` the KV.

**Phase B — one graph per velocity evaluation.** Built once per chunk and
re-evaluated `2 × steps` times with only three inputs changing: the ODE state,
the timestep embedding, and nothing else. Inputs set from the host each call:

- `x_nar`: `ne = [64, N]` F32, rows `0` and `N−1` zero, rows `1..N−2` = the ODE
  state. (Cheaper: keep the state on-device and `ggml_set` into a zeroed buffer.)
- `t_emb256`: `ne = [256]` F32 — `cat(cos(t·freqs), sin(t·freqs))` computed on
  the **host in f32** exactly per `modeling_yue2.py:326-330`, with
  `t = sigmoid(f32(clamp(logit_f64(...), −20, 20)))` per §1.3.

Graph:

```
	x = mul_mat(nar.vae2llm.weight, x_nar) + nar.vae2llm.bias        # [2048, N]
	te = mul_mat(nar.time_embd.1.weight, silu(mul_mat(nar.time_embd.0.weight, t_emb256)
	                                          + nar.time_embd.0.bias)) + nar.time_embd.1.bias
	x = x + te                                                       # broadcast over N
	x = x + get_rows(nar.latent_pos_embd.weight, 0..N-1)             # cast F16 -> F32
	for l in 0..27:
		h = rms_norm(x, 1e-6) * blk.l.nar_attn_norm
		q,k,v  <- nar_attn_{q,k,v}, then nar_attn_{q,k}_norm, then rope_neox
		          with positions arange(ar_length, ar_length + N)
		K = concat(ar_k[l], k, dim=1)      # [128, ar_length + N, 8]
		V = concat(ar_v[l], v, dim=1)
		for each query tile qt of --query-chunk rows:
			kq = soft_max_ext(mul_mat(K, qt), NULL, 1/sqrt(128), 0)   # NO MASK
			o  = mul_mat(permute(V), kq)
		x = x + mul_mat(blk.l.nar_attn_output, reshape(concat(o tiles), [2048, N]))
		h2 = rms_norm(x, 1e-6) * blk.l.nar_ffn_norm
		x  = x + mul_mat(blk.l.nar_ffn_down, silu(mul_mat(blk.l.nar_ffn_gate, h2)) * mul_mat(blk.l.nar_ffn_up, h2))
	v_out = mul_mat(nar.llm2vae.weight, rms_norm(x, 1e-6) * output_norm) + nar.llm2vae.bias
	v_out = view(v_out, rows 1 .. N-2)                               # [64, Tc]
```

The `concat` of AR and NAR K/V can be avoided by allocating the per-layer KV
buffer as `[128, ar_length + N, 8]` up front and writing the NAR K/V into its
tail each evaluation; do that — it saves `28 × 2 × (ar_len+N) × 4 KiB` of copies
per evaluation.

**Query tiling** is a memory measure only: softmax is per-query-row, so tiling is
mathematically identical, and on the CPU backend bit-identical. torch used
`block = 256` on CPU and `len(q)` on CUDA (`nar.py:70`); neither is normative.
Default `--query-chunk 1024`.

**Host-side ODE loop** (outside the graph), per §1.3: two graph evaluations per
step, `mid = state − v1·dt/2`, `state = state − v2·dt`. Keep `state` in a
backend tensor and do the axpy either with a tiny 3-node graph or on the host —
either is fine, it is `T × 64` floats; do it in **f32 on the host** if that is
simpler, it costs nothing and removes a backend numerics variable.

**Chunk loop**: for each `(a, b)` from `chunk_ranges`, build `ar_tokens`, run
Phase A, run `2·steps` Phase-B evaluations, append `[b−a, 64]` to the output,
free the chunk's KV. Weights stay loaded across chunks.

### 4.5 Memory at CONTEXT on a 24 GB card

Worst realistic case, `P = 2549` (the artifact's prefix) and a full context
chunk: `size = 11012`, `ar_length = 13562`, `N = 11014`, `S = 24576`.

| item | bytes |
|---|---|
| AR-half weights, F16 (28 layers + `token_embd` + `output_norm`) | 3.6 GB |
| NAR weights, F16 (incl. `latent_pos_embd` 100 MB) | 2.9 GB |
| AR+NAR KV, F32: `28 × 2 × S × 8 × 128 × 4` = 224 KiB/token × 24576 | 5.5 GB |
| Phase-B activations, `--query-chunk 1024`: scores `16 × 1024 × 24576 × 4` = 1.6 GB, FFN `11014 × 6144 × 4` = 271 MB ×2 live | ~2.3 GB |
| Phase-A activations, block 512: scores `16 × 512 × 13562 × 4` | ~0.9 GB (not live at the same time as Phase B) |
| **total resident during Phase B** | **≈ 14.3 GB** |

Fits, with room. `alley_swing_s1` itself (`ar_length = 6659`, `N = 4111`,
`S = 10770`) needs ≈ 8.8 GB. Levers if a card is tighter: `--query-chunk 256`,
`--ar yue2-ar-q8_0.gguf`, and a smaller `--context` (which changes the *result*,
so it is a last resort and must be recorded). Use `ggml_gallocr` for the graph
and a separate `ggml_backend_buffer` for the KV so the allocator does not try to
reuse it.

---

## 5. Acceptance

Comparisons use `convert/compare.py` (already in the repo) reporting max|Δ| and
SNR in dB. Run each on **CPU first, then Vulkan device 1 (Intel Arc), then
Vulkan device 0**; record all three.

### 5.1 Short golden (mandatory)

```
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--prefix tests/golden/nar_prefix_ids.npy --codec tests/golden/nar_codec_ids.npy \
	--noise tests/golden/nar_noise.npy --steps 32 \
	-o tests/out/nar_latent_s32_<dev>.npy --dump-dir tests/out/nar_dump_<dev> --device <dev>
```

| check | vs golden | required |
|---|---|---|
| `nar_ar_kv_l0.npy` (prefill, layer 0) | CPU | SNR ≥ 100 dB |
| `nar_ar_kv_l27.npy` (prefill, layer 27) | CPU | SNR ≥ 85 dB |
| `nar_x_in_step0.npy` | CPU | SNR ≥ 95 dB |
| `nar_v_step0.npy` | CPU | SNR ≥ 85 dB |
| `nar_latent_s2.npy` | CPU | SNR ≥ 80 dB |
| **`nar_latent_s32.npy`** | **CPU** | **SNR ≥ 70 dB and max\|Δ\| < 5e-3** |
| **`nar_latent_s32.npy`** | **Vulkan dev 0 and dev 1** | **SNR ≥ 50 dB and max\|Δ\| < 5e-2** |

Rationale for the CPU bar: F16 weights are the exact BF16 checkpoint (§2.2) and
the activations are F32 on both sides, so the only difference is GEMM reduction
order, compounded through 28 layers × 64 evaluations — an order of magnitude
more accumulation than the VAE's 121 dB, hence 70 dB not 110. The Vulkan bar
allows for shape-dependent kernel choices with the fp16 staging **disabled**;
if a run lands below 50 dB the C++ agent must, **before proposing any threshold
change**, produce (a) the per-ODE-step SNR curve of `state` versus a CPU run
(does it grow smoothly or jump at one step?), (b) the same run with
`--vk-f16-matmul` to confirm the env block is actually taking effect, and (c)
the layer-0 prefill KV SNR, and put all three in `src/STATUS_NAR.md`.

Also required, independent of any golden: the output is finite everywhere, its
range is within roughly `[-8, 8]` (the artifact's is `[-5.69, 4.59]`), and
running twice on the same device gives bit-identical output.

### 5.2 End-to-end: same latent → VAE → audio

```
build_vae/yue2-vae -m yue2-vae-f32.gguf -i tests/out/nar_latent_s32_<dev>.npy \
	-o tests/out/nar_s32_<dev>.wav --npy tests/out/nar_s32_<dev>_audio.npy --device cpu --full
```

128 frames → `1920 × 128 − 64 = 245 696` samples ≈ 5.1 s of stereo. Decode both
the golden latent and the C++ latent through the **same** VAE settings and
compare the audio: **SNR ≥ 45 dB** required on Vulkan (the VAE's SnakeBeta chain
amplifies input noise — see `docs/vulkan_burst_investigation.md` — so the audio
bar sits below the latent bar by design). Also listen to it once: it must be
music, not noise.

### 5.3 Full song

```
build_nar/yue2-nar --ar tests/out/yue2-ar-f16.gguf -m tests/out/yue2-nar-f16.gguf \
	--artifacts <a copy of alley_swing_s1, NOT songs/ itself> \
	--noise tests/out/nar_noise_alley_swing_s1.npy --device vulkan --gpu 0 \
	-o tests/out/nar_full_vk0.npy
```

- Shape must be exactly `[4109, 64]` float32.
- Against `tests/out/nar_full_cpu_latent.npy` (§3.4, if produced):
  **SNR ≥ 45 dB** required on Vulkan, ≥ 65 dB on CPU.
- Against `songs/ref_song/out/alley_swing_s1/latent.npy`: **report only, no
  pass/fail** — that run was bf16 on ROCm (§3.6). A figure in the 20–35 dB range
  is expected and is not a bug. The meaningful check is the audio: decode both
  with `yue2-vae` and confirm they are the same performance (same structure,
  same words, same arrangement) on a listen; a large structural divergence *is*
  a bug and must be chased.
- **Multi-chunk path.** The real song is one chunk (§1.1), so chunking needs its
  own test. Use the short prefix (`nar_prefix_ids.npy`, 256 ids), the **first
  512** codec frames, `--context 1200` and `--steps 2`. Then
  `size = (1200 − 256 − 3)//2 = 470`, giving ranges
  `(0,470) (470,940) (940,1410) ... (1410,512→)` — concretely 2 chunks over 512
  frames: `(0,470)` and `(470,512)`. Bump to 1600 frames if a 3+-chunk case is
  wanted (`ceil(1600/470) = 4`). Required: the output is exactly
  `[frames, 64]`, the chunk boundaries land at multiples of 470, and the result
  matches the converter agent's torch-CPU golden for the *same* arguments
  (`tests/out/nar_multichunk_cpu.npy`, produced by driving `nar.synthesize(...,
  steps=2, context=1200)`) at **SNR ≥ 60 dB on CPU, ≥ 45 dB on Vulkan**. This is
  the **only** test that covers chunking; it is not optional. Note that the
  latent for frames `[470, 512)` in this run will *not* match the single-chunk
  run — different AR context — and that is correct behaviour, not a bug.

### 5.4 Timing

Baselines, from `songs/ref_song/out/alley_swing_s1/result.json`: torch on ROCm
took **`nar_seconds = 46.38 s`** for 4109 frames = 164.36 s of audio
(≈ 0.28 s of compute per second of audio; the ~64 s figure quoted for a 205 s
song is the same rate). Report:

| metric | target |
|---|---|
| prefill (6659 tokens), Vulkan dev 0 | report |
| seconds per ODE step (2 velocity evals, N = 4111) | report |
| **total, `alley_swing_s1`, F16, Vulkan dev 0** | **must beat 46.4 s; target ≤ 30 s** |
| same, Q8_0 AR half | report, with the dB cost |
| same, `--vk-f16-matmul` | report, with the dB cost (this is the "what the fp16 staging buys" number) |
| same, Vulkan dev 1 (Arc) | report |
| same, CPU | report (expect tens of minutes; may be skipped if §3.4 already measured torch CPU) |

A rough FLOP budget for the target: `64 evals × 4111 tokens × 2.82 GFLOP ≈
742 TFLOP` plus ~23 TFLOP of attention; 30 s implies ~26 TFLOP/s sustained, which
is in range for F16 weights on a 7900 XTX but not free — if the first working
version is at 60 s, that is a fine place to stop and report rather than to start
optimising.

---

## 6. Working rules

**SPEC.md §6 applies in full** (work only inside `yue2.cpp/`, scratch to
`tests/out/`, `songs/` and the HF cache are read-only inputs, no torch on GPU
ever, keep a STATUS file, tabs for indentation, braces on their own line except
`} else {`, don't `git commit` — the coordinator commits). Additions:

- **Torch CPU only for the 3B**, with `CUDA_VISIBLE_DEVICES=""` and
  `HIP_VISIBLE_DEVICES=""` set inside `reference_nar.py`, and an assert that the
  model is on `cpu` after load. A wedged HIP run takes the whole box down.
- `songs/` and `~/.cache/huggingface/` are **read-only**. Do not write into
  them, not even a temp file. Copy `alley_swing_s1` into `tests/out/` if a test
  needs a writable artifacts dir.
- **Converter/goldens agent uses no GPU at all.** Reading the safetensors header
  (`safetensors.safe_open`, metadata only) and streaming tensors one at a time
  is the expected access pattern for `convert_nar.py`; never load the whole
  7.26 GB more than necessary. Python only — no build dir needed.
- **C++ agent uses Vulkan device 1 (Intel Arc) for all correctness work** and
  device 0 (7900 XTX) **only for the final timing numbers in §5.4**. Device
  indices are stable across boots (Vulkan enumerates by PCIe slot). Two GPUs
  agreeing is also the strongest evidence that a numeric difference is real and
  not per-device reduction noise — `docs/vulkan_burst_investigation.md` used
  exactly that.
- **Build directories are per-agent.** The C++ agent builds in `build_nar/` and
  touches only `src/yue2-nar.cpp`, `CMakeLists.txt` (the one `add_executable`
  block) and `src/STATUS_NAR.md`. The converter agent needs no build dir; it
  runs `../venv_yue2/bin/python`. Neither agent edits the
  other's files, `src/yue2-vae.cpp`, `src/yue2-ar.cpp`, or this spec.
- **`yue2-vae` and `yue2-ar` must still build and pass their existing golden
  tests** after the CMake change (`src/STATUS.md` and `src/STATUS_AR.md` have
  the exact commands). Run them; record the numbers.
- STATUS files: `convert/STATUS_NAR.md` and `src/STATUS_NAR.md` — what is done,
  every deviation from this spec and why, exact reproduction commands, open
  questions. Final report to the coordinator ≤ 200 words, pointing at STATUS.
- Where this spec is wrong or ambiguous, **say so in STATUS and pick a reading**;
  do not silently diverge, and do not wait for the other agent.
