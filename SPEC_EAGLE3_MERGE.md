# SPEC_EAGLE3_MERGE — bring `feat/eagle3-draft` onto master behind a build switch

`feat/eagle3-draft` (one squashed commit, 10dd57d) adds EAGLE-3 speculative
decoding for the semantic phase (`--draft …`; SPEC_DRAFT.md, src/STATUS_DRAFT.md
on that branch). It goes onto master with a CMake switch so a llama.cpp bump
that breaks the experimental API can never block a master build.

## 1. The switch

`option(YUE2_EAGLE3 "EAGLE-3 speculative decoding (--draft); uses llama.cpp's experimental llama-ext.h" ON)`

- **ON:** exactly the branch's behaviour.
- **OFF:** `src/draft_eagle3.cpp` is not compiled and no target gets the
  `llama.cpp/src` SYSTEM include. `draft_eagle3.hpp` provides inline stubs with
  the same signatures under `#ifndef YUE2_HAVE_EAGLE3`; `--draft*` flags still
  parse (usage text unchanged, so scripts don't break) and `--draft` fails at
  argument check with `yue2: built without EAGLE-3 support (-DYUE2_EAGLE3=ON)`.
- `stage_ar.cpp` / `stage_song.cpp` should need **no** `#ifdef`s — if one is
  unavoidable, keep it to a single site and say why in STATUS.
- `llama-ext.h` / `llama.cpp/src` may be included by `draft_eagle3.cpp` only.
  `draft_accept.hpp` (pure C++) stays unconditional.
- `tests/draft_accept.cpp` builds in both configurations (it has no llama
  dependency); anything needing the head builds only when ON.

## 2. Bringing it over

Rebase/cherry-pick the branch commit onto current master (which by then has
`yue2 convert` and the Windows path fixes — expect conflicts in
`stage_song.cpp`, resolve keeping both). Squash to one commit's worth of
change; the coordinator commits.

## 3. Acceptance

1. Build ON and OFF (`build_e3on/`, `build_e3off/`), both `YUE2_WARN_FLAGS` clean.
2. OFF: `yue2 song … --draft H.gguf` exits non-zero with the message above;
   without `--draft`, output byte-identical to master.
3. ON, no `--draft`: one `yue2 song`, fixed seed, `--gpu 1` → FLAC + artifacts
   identical to master (wall-clock fields excepted), as STATUS_DRAFT did.
4. ON, `--draft` exact (K=2) with the published head
   `YuE2-3B-EAGLE3-draft-Q8_0.gguf` (sha256 `54e2219f7d021373d05bcd4c932b7bf7d362e705b7f10e59195ea6af9d374d4b`,
   bit-identical to the branch STATUS's Q8_0 head — fetch it with
   `hf download engival/YuE2-3B-EAGLE3-draft-GGUF` or use that local head after
   checking the hash): runs, speedup line reported. Exact mode samples the same
   distribution but spends the random draws differently, so the song differs
   from the no-draft run at the same seed; check instead: score/prefix artifacts
   identical to master, two `--draft` runs at one seed bit-identical, and
   `--draft-extract-only` identical to master.
5. `tests/draft_accept` passes in both builds.
6. README: a short "Speculative decoding (optional)" section — flag, the
   exact/lossy distinction, measured speedups, `-DYUE2_EAGLE3=OFF`, and the
   head: <https://huggingface.co/engival/YuE2-3B-EAGLE3-draft-GGUF>
   (`YuE2-3B-EAGLE3-draft-Q8_0.gguf`, 154 MB, CC BY-NC 4.0 as a derivative of
   the YuE2 weights; trained against yue2.cpp's own Q8_0 AR GGUF — i.e. what
   `yue2 convert` makes by default — other AR quants untested). Do **not** link
   a trainer repo; none is public yet.

## 4. Constraints

Vulkan device 1 (Arc) and CPU only — never device 0. `nice -n 10 … -j8`.
Never modify `llama.cpp/`. Repo style per CLAUDE.md. Report in
`src/STATUS_EAGLE3_MERGE.md`. No commit, no push.
