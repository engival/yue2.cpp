# yue2.cpp — orientation for Claude

YuE2-3B song generation (lyrics + style → FLAC) on ggml/Vulkan, no Python at
runtime. One binary: `build/yue2 song|ar|nar|vae|noise`. README.md is the user
manual; this file is where things are and the rules that were learned the hard way.

## Map

- `SPEC.md` (VAE), `SPEC_AR.md`, `SPEC_NAR.md`, `SPEC_SINGLE.md` — the contracts
  each stage was built to. Math with line refs into the reference Python, GGUF
  layout, goldens, acceptance bars. Read the SPEC before touching a stage.
- `src/STATUS*.md` — what was actually built, measured numbers, deviations from
  the SPEC, exact commands. `STATUS_NAR_PERF.md` has the per-op profiles.
- `docs/vulkan_burst_investigation.md` — why F32 on ggml-vulkan needs
  `GGML_VK_DISABLE_F16` + `GGML_VK_DISABLE_COOPMAT` (fp16 staging of F32 matmul).
- `docs/SCORE_RECIPES.md` — how-to for editing a `score.abc` and feeding it back
  (`"abc"` / `"abc_template"`); `scripts/abc_transpose.lua` (plain Lua) is the
  one script it needs. Patterns only: no opinions, findings or private paths.
- `docs/ROADMAP.md` — what's next and why. The public tree carries no local
  paths, private scripts or addresses (the checklist lives outside this repo);
  keep it that way.
- `convert/` — safetensors → GGUF (`convert_vae.py`, `convert_ar.py`,
  `convert_nar.py`) and the torch-CPU reference scripts that make goldens.
- `tests/golden/` — golden inputs/outputs (`*.npy` gitignored, `*_meta.json`
  committed with SHA-256s). `tests/out/` — scratch, gitignored.

## Rules

- **Goldens come from torch on CPU only.** Never torch on a GPU here (ROCm
  crashed the card; that's why this repo exists).
- **Two Vulkan devices, stable by PCIe slot:** 0 = AMD 7900 XTX (the user's
  render card), 1 = Intel Arc B70. Sub-agents use device 1 and CPU only; device 0
  is timed by the coordinator after `pgrep -af yue2` shows it idle.
- Per-agent build dirs (`build_*/`, gitignored); `build/` is the one the drivers use.
  Build with `nice -n 10 cmake --build DIR -j8` — never `-j` at the full core count
  (it stalls the user's desktop).
- `../songs/`, `~/.cache/huggingface`, `tests/golden/` are read-only.
- Style: tabs; braces on their own line except `} else {`; every variable earns
  its existence; `YUE2_WARN_FLAGS` clean.
- Agents report in a `STATUS*.md` (short, tables, exact commands); the
  coordinator commits. A cold review pass before commit has caught real bugs
  every time — keep doing it.

## ggml/Vulkan lessons (each cost hours)

- ggml-vulkan stages **both** F32 operands as fp16 when the device has fp16;
  `GGML_PREC_F32` is honoured only by flash-attention. Env vars are read once per
  device init → per-process. So a run has exactly one precision: `yue2 song` /
  `yue2 batch` decode the VAE at the NAR's, and the exact-F32 decode is the
  standalone `yue2 vae` (SPEC_SINGLE §2.2 — it used to be a forked child).
- F32 K/V makes attention an F32×F32 matmul (slowest path); F16 K/V (`--kv-f16`)
  puts it on coopmat. Flash-attention wins on RADV, loses on the Arc (ggml's FA
  tuning is for Xe1).
- `ggml_gallocr` recycles graph-owned input tensors between evaluations; inputs
  that must survive live in their own backend buffer.
- ggml-vulkan returned zeros for kv heads ≥ 1 on a strided-ne2 F32 `mul_mat`
  src0; attend over the contiguous cache with a wider mask instead.
- libllama exposes no per-layer K/V → the NAR re-prefills the AR prefix in raw
  ggml (1–3 s per chunk); Q8_0 AR weights are fine for that (33 dB).
