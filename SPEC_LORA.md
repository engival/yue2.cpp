# SPEC_LORA — stage 10: trial a NAR LoRA without rebuilding the GGUF

Builds on SPEC_NAR.md / SPEC_SINGLE.md (`src/STATUS_NAR.md`). Same files, same rules.

## 1. Why

Community LoRAs for the NAR half exist (rank-32 adapters over the attention and
MLP projections). Trying one today means folding it into the
HF checkpoint and re-running `convert_nar.py` for every file and every strength.
The NAR has its own tensor loader, so the delta can be folded in **at load**:
no graph change, no per-step cost, no new GGUF. `--nar-lora FILE[:S]`.

The AR half goes through libllama and is a different job (llama adapters); it is
out of scope here. So is the fused ComfyUI layout (§3.4).

## 2. CLI

```
--nar-lora FILE[:STRENGTH]      repeatable; STRENGTH is a finite float, default 1.0
```

- Accepted by `yue2 nar`, `yue2 song` and `yue2 batch` (one setting for the whole
  batch: it is a property of the loaded model, like `--nar`, not of a request).
  Not a request key.
- `FILE[:S]`: split on the **last** `:` and only when what follows parses fully
  as a float; otherwise the whole argument is the path.
- Repeating the flag stacks adapters, applied in command-line order (§3.3).
- `song` / `batch` record it in `config.json` as
  `"nar_lora": [ { "file": "...", "strength": S, "sha256": "..." } ]`
  (key absent when the flag is not given, so existing `config.json` output is
  unchanged).
- One log line per adapter at load: file, rank, strength, how many tensors were
  patched / replaced, and the seconds the merge took.

Without the flag the build MUST stay bit-identical to the current one (same
latent bytes for the same inputs).

## 3. Adapter file

### 3.1 Container

Plain `safetensors`: `u64` little-endian header length, that many bytes of JSON,
then raw tensor data; each entry has `dtype`, `shape`, `data_offsets` relative to
the end of the header. Parse the header with the JSON library the tree already
uses. Accept dtypes `F32`, `F16`, `BF16`; anything else is an error. `__metadata__`
is ignored except for the checks in §3.4. Read tensors on demand (seek + read);
do not slurp the file.

### 3.2 Keys (the plain, unfused layout)

| adapter key | GGUF tensor | kind |
|---|---|---|
| `layers.{i}.nar_self_attn.{q,k,v,o}_proj.lora_A` `[r, in]` + `.lora_B` `[out, r]` | `blk.{i}.nar_attn_{q,k,v,output}.weight` | LoRA |
| `layers.{i}.nar_mlp.{gate,up,down}_proj.lora_A` + `.lora_B` | `blk.{i}.nar_ffn_{gate,up,down}.weight` | LoRA |
| `vae2llm.weight`, `vae2llm.bias`, `llm2vae.weight`, `llm2vae.bias` | `nar.vae2llm.*`, `nar.llm2vae.*` | full replacement |

The authoritative name map is the one in `convert/convert_nar.py`; mirror it, do
not invent a second convention. The `blk.{i}.` prefix above is illustrative: use
whatever the GGUF really calls them. A torch `[out, in]` matrix and a ggml tensor
with `ne = [in, out]` have the same row-major bytes, so no transpose anywhere.

Rank is read from the shapes (`lora_A.shape[0]`), per tensor. There is no alpha:
the file's scale is 1.0.

### 3.3 Merge rule

For strength `s`:

- LoRA pair: `W += s · (B @ A)`
- replacement: `W += s · (W_new − W_base)` where `W_base` is the tensor **as read
  from the GGUF**, before any adapter touched it.

Both are deltas against the base, so stacked adapters commute and `s = 0` is the
identity. Per tensor: widen the GGUF bytes to F32, add every adapter's delta,
then narrow to the tensor's backend type (F16 via `ggml_fp32_to_fp16_row`, or
leave F32 under `--weights f32`) and upload. One narrowing per tensor, after all
adapters. A NAR GGUF tensor that an adapter targets and that is neither F16 nor
F32 is an error (no quantised NAR exists; do not write a requantiser).

Tensors no adapter names take the existing path untouched (no widen/narrow
round trip), which is what keeps the no-flag run bit-identical.

### 3.4 Errors (exact, tested messages; all before any GPU work beyond the load)

- file unreadable / not safetensors / header not JSON / data shorter than the
  offsets claim.
- a key that maps to nothing. If any key starts with `diffusion_model.`, say so
  specifically: this is the fused ComfyUI layout; use the plain file.
- `lora_A` without its `lora_B` or the reverse.
- shape mismatch: `A = [r, in]`, `B = [out, r]` against the GGUF's `ne = [in, out]`;
  replacements must match the base shape exactly.
- unsupported dtype; non-finite strength; non-finite value in any adapter tensor.
- `--nar-lora` with a target tensor of quantised type.

An adapter that patches only part of the model (some layers, no replacements) is
valid: every key must be *consumed*, not every tensor *covered*.

## 4. Implementation notes

- The merge belongs in the NAR tensor-load loop in `src/stage_nar.cpp` (the one
  that already widens F16→F32 for `--weights f32`). Index the adapter(s) by GGUF
  tensor name up front; in the loop, a hit takes the merge path.
- **The merge is single-threaded and must stay that way.** No thread pool, no
  OpenMP, no ggml CPU graph. Write `B @ A` as the axpy form (for each output
  row, for each rank `k`: `row += (s·B[o,k]) · A[k,:]`) over F32 so the compiler
  vectorises it; this is ~45 G multiply-adds for a rank-32 adapter over 28
  layers. Budget: ≤ 10 s added to the load on one core. Report the measured
  number. If it cannot be met single-threaded, report that instead of adding
  threads.
- Memory: one F32 scratch row-block per tensor (the largest is 6144×2048 F32 =
  50 MB). Do not hold the whole adapter in RAM.
- `yue2 batch` loads the NAR once; the merge happens once.
- The sha256 for `config.json` reuses whatever helper the tree already has for
  file hashes.

## 5. Baking (the reference the runtime is checked against)

`convert/convert_nar.py` gains `--lora FILE[:S]` (repeatable, same parsing, same
merge rule in the same order of operations: F32 delta sum, one narrowing). This
is both the way to make a permanent merged GGUF once a LoRA is a keeper, and the
independent implementation §6.3 compares against. Torch on **CPU** only, as for
every converter here.

## 6. Acceptance

All GPU runs on Vulkan device 1; short runs (`--steps 2`, a `--frames` cut of an
existing golden) unless stated. Put everything under `tests/out/lora/`.

1. **No flag**: latent byte-identical to the `build/` binary on the same inputs.
2. **`:0`**: `--nar-lora FILE:0` latent byte-identical to no flag (proves the
   widen → add-zero → narrow round trip and the replacement interpolation are
   exact at 0).
3. **Runtime merge == baked GGUF**: bake `FILE` at 1.0 and at 0.5 with §5, run
   the baked GGUF without the flag and the stock GGUF with the flag. Report, per
   strength: count of F16 weights that differ between the two merges (expected:
   a handful at most, each by one F16 step — float summation order), and the
   latent PSNR between the two runs (bar: ≥ 80 dB).
4. **It does something**: stock vs `:1.0` at `--steps 32` on one golden: both
   finite, PSNR between them reported (expected: clearly different, tens of dB,
   not ≥ 80).
5. **Stacking**: `--nar-lora F:0.5 --nar-lora F:0.5` vs `--nar-lora F:1.0`:
   latent PSNR ≥ 80 dB.
6. **Errors**: every message in §3.4, each triggered by a small hand-made
   safetensors (a Python helper under `tests/` that writes them with numpy; no
   torch needed).
7. **Load cost**: seconds added by the merge, single core, for a rank-32
   all-layers adapter.
8. `YUE2_WARN_FLAGS` clean; README gets one short subsection (flag, merge rule,
   plain layout only, baking with `--lora`, and that adapter weights carry their
   own licence — the ones known today are CC BY-NC and are not distributed
   here). `src/STATUS_LORA.md` with the tables and exact commands.

Listening tests on full songs are the coordinator's, not part of acceptance.
