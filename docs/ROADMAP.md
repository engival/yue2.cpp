# Roadmap

Where things stand (2026-09-11) and what is worth doing next, in rough order of
value. Numbers are for the reference song unless stated (alley_swing_s1, 164 s
of audio; `yue2 song` on the AMD 7900 XTX renders a 196 s song in 120.8 s).

## Done

- Stage 1–4: VAE, AR (libllama), NAR (raw ggml), single `yue2` binary. No PyTorch
  at runtime, no ROCm anywhere. Per-card NAR flags chosen automatically.
- Stage 5: `yue2 batch` / `yue2 ar --requests` decode several songs' AR phases in
  one context, one `llama_decode` per step per slot (`kv_unified = false`, one KV
  stream per slot). NAR and VAE stay per song. Numbers in
  [src/STATUS_BATCH.md](../src/STATUS_BATCH.md).
- Stage 6: sparse `sample_step` — allowed-set bounds instead of a 184 704-float
  mask and copy, the window penalty as a sorted side list, a bounded min-heap
  instead of `nth_element` over every finite entry, per-`Seq` scratch so a decode
  step allocates nothing. Token-for-token identical to stage 5 (over 600 000
  differential cases including one-ulp tie injection + full songs under
  `--verify-sampler` + a byte-identical seeded song). Numbers in [src/STATUS_SAMPLER.md](../src/STATUS_SAMPLER.md).
- Stage 7: guided semantic decoding. `cfg_scale` is honoured (the reference's
  classifier-free guidance), and the `guidance` request key generalises it to
  time-varying weight curves against up to two negative branches plus a positive
  prefix that can change at a given frame — mid-song style changes. One song,
  up to three KV streams, one blended row into the unchanged sampler. Numbers in
  [src/STATUS_GUIDANCE.md](../src/STATUS_GUIDANCE.md).
- Stage 7b: `semantic_keep` — the leading codes of an earlier render are forced
  as history instead of being sampled, so "keep this take up to the cut, change
  it from there" is a prefill (1000 frames in 0.4 s on the Arc, against 7.3 s to
  sample them) and combines with a `guidance` entry at that frame. Numbers in
  [src/STATUS_GUIDANCE.md](../src/STATUS_GUIDANCE.md) §"Stage 7b".
- Stage 8: `sections` — a style per section. An entry names a score label
  (`% verse`, `nth`) instead of a semantic frame; the engine walks the score's
  bar clock (`M:`/`Q:`, inline `[M:…]`, the `V: Vocal` voice), turns the label's
  bar into a frame `lead_frames` ahead of it, and compiles the entries into
  guidance entries. Without an `against` it is a plain swap that owns one KV
  stream, so it decodes at `--parallel > 1`. Numbers in
  [src/STATUS_SECTIONS.md](../src/STATUS_SECTIONS.md).
- Stage 9: `handover` — one take per style from one score, and at each cut the
  incoming style's renderer is forced through its own take up to a few seconds
  before the cut and the song's last few seconds after it, then samples on. The
  model is loaded once; every take and leg is its own generation, bit-identical
  to rendering it by hand. The lag one take runs ahead of another is measured
  from the tokens (`"offset": "auto"`). Numbers in
  [src/STATUS_HANDOVER.md](../src/STATUS_HANDOVER.md).
- Stage 10: `--nar-lora FILE[:S]` — a plain-layout safetensors LoRA is folded
  into the NAR weights while they are read (single-threaded, a few seconds, no
  graph change), so an adapter or a strength can be tried without a new GGUF;
  `convert_nar.py --lora` bakes a keeper. A run without the flag is bit-identical
  to stage 9. The AR half (libllama adapters) is not done. Numbers in
  [src/STATUS_LORA.md](../src/STATUS_LORA.md).
- Stage 11: `negative_style` — a negative prompt for `cfg_scale`. The text
  replaces the blank branch's prefix (the positive recipe with those tags and no
  lyrics, then the same score), so the song is pushed away from it as well as
  towards its own tags and lyrics; weight, blend and sampler are unchanged, and a
  request without it is bit-identical to stage 10. Numbers in
  [src/STATUS_NEGATIVE.md](../src/STATUS_NEGATIVE.md).
- Stage 11b: `negative_lyrics` — the negative branch keeps the song's lyrics, so
  `cfg_scale` pushes along the tag difference alone (a lyric-less negative traced
  as far from the song as the blank did); `false` is bit-identical to stage 11.
  Numbers in [src/STATUS_NEGATIVE.md](../src/STATUS_NEGATIVE.md).
- Stage 11c: `cfg_score` — the lyric-carrying negative also guides the score
  phase (no reference equivalent): one branch from abc step 0 that becomes
  `cfg_scale`'s negative at the transition (or is dropped at `cfg_scale` 1);
  without it every path is bit-identical to stage 11b. Numbers in
  [src/STATUS_NEGATIVE.md](../src/STATUS_NEGATIVE.md).
- Stage 11d: `cfg_score` leaves the score header unguided (weight 0 until the
  first `K:` line), a header check (`score_header_ok`) for every score, and
  `score_tempo` forcing the score's `Q:1/4=t` line; without them every path is
  bit-identical to stage 11c. Numbers in
  [src/STATUS_NEGATIVE.md](../src/STATUS_NEGATIVE.md).
- The VAE fork (2026-09-13): `song`/`batch` decode in-process at the NAR's
  Vulkan precision. A listening test found the fp16-staged decode (58.5 dB from
  exact) indistinguishable, and on the AMD the exact path is not faster — so
  exact-F32 is a numeric reference reached through the standalone `yue2 vae`,
  and asking `song`/`batch` for it is an error, not a downgrade. SPEC_SINGLE §2.2.
- VAE ConvTranspose1d as a kernel-2 Conv1d (2026-10-05): each decoder block's
  transposed conv (kernel 2·stride) runs as im2col + mul_mat with `stride·Cout`
  outputs and a permute, its weight rearranged at load. ggml-vulkan's
  `conv_transpose_1d` kernel had been 82 % of the decode. 191 s song: Arc
  16.1 → 3.3 s, AMD 7.6 → 1.9 s; CPU golden 122.09 dB (was 121.95), fp16-staged
  65 dB from exact (was 67).
- Opus output: an output named `X.opus` is encoded with libopusenc at
  `--opus-bitrate` (default 160 kbit/s) in `song`, `batch` and `vae`, tags and
  all. Optional at build time (`-DYUE2_OPUS`, default on when pkg-config finds
  `libopusenc`); a build without it refuses the name while parsing arguments.
  The exact and regression paths stay FLAC, and a kept `latent.npy` decodes the
  lossless file again at any time.
- Accuracy: VAE 117 dB vs torch CPU; AR prefix + greedy bit-identical; NAR fast
  path 29 dB (AMD) / 33 dB (Arc) from the f32 reference — same class as torch's
  own bf16 (35 dB), inaudible in A/B.

## Next

1. **ggml-vulkan: per-op F32 precision for `mul_mat`.** Honour `GGML_PREC_F32`
   in the Vulkan mul_mat pipeline selection the way flash-attention already does.
   Would let one process hold both precisions: the exact-F32 VAE could then be a
   `song`/`batch` option instead of a standalone-only route, and `--nar-f32`
   would stop shifting AR sampling. A legitimate upstream PR. Needs both shader variants compiled per
   device (`ggml-vulkan.cpp` ~L6584/L7513 read the env once).
2. ~~**ggml-vulkan: Intel Battlemage flash-attention tuning.**~~ Done in our
   llama.cpp fork (`dev`): coopmat1 FA on the Arc's 8x16x16 coopmat shape runs at
   ~29 TFLOP/s at the NAR shape (was 2.6 scalar), plus upstream PR #29882 for the
   per-head norm and a fused SwiGLU. The Arc NAR now uses `--flash-attn` like the
   AMD: ~0.5 s/eval at 16 steps on a full song (was 1.1 with unfused tiles).
   Upstreaming the FA work is still open.
3. **Beat torch's 46 s on the AMD.** Attention is 66 % of a velocity eval and
   RADV's FA kernel runs at 19 TFLOP/s; the GEMMs are already at 56. Upstream
   is tuning this kernel for NVIDIA and Intel, not RDNA3 — so it is ours after
   all. The shape to tune: `hsk = hsv = 128`, 16 heads, 4–7k queries against a
   10–17k F16 K/V with an F16 mask, F32 accumulation, non-causal, ~1 800 calls
   per song; `test-backend-ops perf -o FLASH_ATTN_EXT` with that case added is
   the harness, `get_fa_tuning_params_coopmat1` in `ggml-vulkan.cpp` the knob.
   Re-time after each submodule bump regardless.
4. **Overlap a batch's NAR with the next batch's AR.** `yue2 batch` runs the AR
   for every job, frees it, then NAR+VAE per job: two compute-bound stages that
   never share the box. With two cards (AR on one, NAR on the other) a batch
   would cost `max(AR, NAR)` instead of their sum. Needs a thread and two device
   contexts; deliberately out of stage 5.
5. **Guidance at `--parallel > 1`, and with `abc_template`.** A guided song owns
   all three KV streams today, so it rejects a batch that decodes several songs
   side by side; lifting that means slot bookkeeping for `3 x parallel` streams
   (and the VRAM for it). A template job re-prefills its slot when the score is
   done, which the branches would have to follow — neither is hard, both were
   out of stage 7. Stage 8's plain-swap `sections` job is the one case that does
   run side by side — it opens no branch at all — so the remaining work is
   genuinely the branch bookkeeping, not the cut.
6. **Guidance in the *score* phase.** A `sections` entry's `against` curves are
   carried through to the compiled guidance entry and only ever used in the
   semantic phase; nothing in the design stops them driving the score phase too,
   which is the only way to make a mid-score tag change bite (measured: a plain
   swap after ~90 written lines barely moves the register, rhythm or chords — the
   score history outvotes the tags). It would need the abc phase to decode beside
   a branch, which today it never does. Two smaller follow-ups from stage 8:
   `sections` with `abc_template`, and a label clock that does not depend on the
   planner naming its voices `Vocal` / `Ins`.
7. **Regression harness in-tree.** `tests/regress.sh` decodes a directory of
   the author's own renders (45 styles of one song, 63–117 dB against the
   reference audio, all passing); a public repo needs one that runs from the
   committed goldens alone (`convert/reference_*.py` regenerate them from the HF
   checkout on CPU). That means shipping a public reference `request.json` —
   today's goldens derive from a private one, so a stranger cannot reproduce the
   committed SHA-256s.
8. **Converter ergonomics.** One `convert/convert.py` that writes all three GGUFs
   and cross-checks `yue2.source_sha256`; document the HF snapshot layout it expects.

## Deliberately not planned

- Server/residency mode: model load is ~2.7 s of a 120 s render; not worth it.
- Torch seed compatibility: lost at AR sampling (Philox vs mt19937_64); noise is
  our own MT19937 and stable across machines.
- Drain-phase slot compaction (`llama_memory_seq_cp` slot→hole + `seq_rm` so
  `split_equal` sees consecutive seq ids): measured on the 7900 XTX at 777 of
  9 361 steps in a 4-song batch paying ~9 ms each — ~7 s of a 50 s AR phase,
  ~1.5 % of the batch's wall time once NAR+VAE are counted, and only while the
  queue is draining. A full KV-stream copy per hole and a second slot-table
  path are not worth that.
- Q8_0 NAR weights: untested; NAR F16 is 2.8 GB and the stage is attention-bound,
  so quantizing the weights would not move the time.

## Parked — discuss later, probably not worth it

- Output gain / peak flags (`--gain dB`, `--peak-normalize dBTP`). Measured
  2026-09-12 (`tests/out/loudness/REPORT.md`, gitignored): the VAE output is
  bit-exact with torch (gain ratio 0.99999997, 108–118 dB SNR), YuE2 masters
  to −13…−16 LUFS, unclamped peaks reach ~1.2 but only 0.0001–0.0007 % of
  samples exceed ±1.0 in runs of ≤ 6 samples; 2 of 5 songs never clip. The
  writers (`src/wav.hpp`, `src/common/flac.hpp`) hard-clamp like soundfile
  does. A flat −2 dB trim would clear every over-range sample seen; peak
  normalisation is the wrong tool (it would *raise* the non-clipping songs by
  ~1 dB). Default must stay bit-exact (goldens). Revisit only if the clamp is
  actually audible on a quiet box.
