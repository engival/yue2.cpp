# Roadmap

Where things stand (2026-09-11) and what is worth doing next, in rough order of
value. Numbers are for the reference song unless stated (alley_swing_s1, 164 s
of audio; `yue2 song` on the AMD 7900 XTX renders a 196 s song in 120.8 s).

## Done

- Stage 1–4: VAE, AR (libllama), NAR (raw ggml), single `yue2` binary. No Python
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
- Accuracy: VAE 117 dB vs torch CPU; AR prefix + greedy bit-identical; NAR fast
  path 29 dB (AMD) / 33 dB (Arc) from the f32 reference — same class as torch's
  own bf16 (35 dB), inaudible in A/B.

## Next

1. **ggml-vulkan: per-op F32 precision for `mul_mat`.** Honour `GGML_PREC_F32`
   in the Vulkan mul_mat pipeline selection the way flash-attention already does.
   Removes the VAE child process, lets `--nar-f32` stop shifting AR sampling,
   and is a legitimate upstream PR. Needs both shader variants compiled per
   device (`ggml-vulkan.cpp` ~L6584/L7513 read the env once).
2. **ggml-vulkan: Intel Battlemage flash-attention tuning.** ggml's
   `get_fa_tuning_params_scalar` disables subgroups and halves `block_rows` for
   Intel (tuned on Xe1). On the B70 FA runs at 2.6 TFLOP/s vs 19 on RADV; the Arc
   NAR uses unfused tiles at 92 s because of it. A tuned FA could bring the Arc
   under 60 s. Also a submodule/upstream change.
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
5. **`cfg_scale` (classifier-free guidance).** yue2-ar dies on it today. The
   reference implements it in the AR stage; the NAR is unaffected.
6. **Regression harness in-tree.** `tests/regress.sh` decodes a directory of
   the author's own renders (45 styles of one song, 63–117 dB against the
   reference audio, all passing); a public repo needs one that runs from the
   committed goldens alone (`convert/reference_*.py` regenerate them from the HF
   checkout on CPU). That means shipping a public reference `request.json` —
   today's goldens derive from a private one, so a stranger cannot reproduce the
   committed SHA-256s.
7. **Converter ergonomics.** One `convert/convert.py` that writes all three GGUFs
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
