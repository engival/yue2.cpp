# STATUS_EAGLE3_MERGE — `feat/eagle3-draft` onto master behind `YUE2_EAGLE3` (SPEC_EAGLE3_MERGE.md)

Base: master `c12f0cd` (has `yue2 convert` + the Windows path fixes). Branch commit `10dd57d`
applied with `git cherry-pick -n`; llama.cpp = fork dev `9bf5c9e47` (unchanged). Not committed.

## What changed on top of the branch

| piece | change |
|---|---|
| cherry-pick | one conflict, `CMakeLists.txt` (tests block: `yue2-numpy-exp` vs `yue2-draft-accept`) — kept both. `stage_song.cpp/.hpp` auto-merged cleanly with convert / portability changes. |
| `CMakeLists.txt` | `option(YUE2_EAGLE3 … ON)` + a `yue2: EAGLE-3 speculative decoding (--draft) ON/OFF` status line. `yue2_add_draft()` adds `src/draft_eagle3.cpp`, the `llama.cpp/src` SYSTEM include and `YUE2_HAVE_EAGLE3` only when ON; OFF it adds nothing. `yue2-draft-accept` builds in both. |
| `src/draft_eagle3.hpp` | `#ifdef YUE2_HAVE_EAGLE3`: the declarations + `constexpr bool eagle3_built = true`; else inline stubs, same signatures, `eagle3_built = false`. The two `llama_batch_ext *` members are ON-only, so OFF needs nothing beyond stable `llama.h`. |
| `src/stage_ar.cpp` | **no `#ifdef`**: `check_draft_args` starts with `if (!d.file.empty() && !eagle3_built) die("yue2: built without EAGLE-3 support (-DYUE2_EAGLE3=ON)")` — before the file-exists check, so the message doesn't depend on the path. `stage_song.cpp`: untouched beyond the branch. |
| `llama-ext.h` | included by `src/draft_eagle3.cpp` only (grep-checked). `draft_accept.hpp` unconditional, no llama dependency. |
| README | new "Speculative decoding (optional)" section before "Decode": flag, exact vs lossy, speedups, scope, `-DYUE2_EAGLE3=OFF`, HF head link. No trainer repo linked. |
| SPEC_DRAFT / STATUS_DRAFT | status lines: merged behind the switch, Q8_0 head published (were "not merged / not distributed"). |
| CLAUDE.md | one Map entry: SPEC_DRAFT + the switch + the include rule. |

## Acceptance

| # | item | result |
|---|---|---|
| 1 | build ON (`build_e3on`) and OFF (`build_e3off`), `YUE2_WARN_FLAGS` clean | **PASS** — no warning from `src/` or `tests/`; only master's pre-existing `ggml-backend.h:422 -Wshadow` (+ ggml-vulkan's own) in both. OFF flags.make: no `llama.cpp/src`, no `YUE2_HAVE_EAGLE3`, no draft object; `nm build_e3off/yue2` has no eagle3 / `llama_batch_ext` symbols. |
| 2a | OFF: `--draft H.gguf` exits non-zero with the message | **PASS** — `yue2 song` and `yue2 batch`: `error: yue2: built without EAGLE-3 support (-DYUE2_EAGLE3=ON)`, exit 1. `--draft*` usage text identical to ON (md5 of `song --help` draft lines equal). |
| 2b | OFF, no `--draft`: byte-identical to master | **PASS** — FLAC sha256 `cd5575be…` = master; every artifact `cmp`-equal except plan.json / plan_manifest.json / result.json, whose differing keys are only `timing.*` wall-clock fields and the hashes/byte counts of those files. |
| 3 | ON, no `--draft`, seed 1, `--gpu 1`: identical to master | **PASS** — same as 2b (FLAC `cd5575be…`, same key-level diff). |
| 4 | ON, `--draft` Q8_0 K=2, exact | **PASS with a spec correction** — runs; head sha256 `54e2219f7d021373d05bcd4c932b7bf7d362e705b7f10e59195ea6af9d374d4b` verified. The semantic tokens are **not** identical to the no-draft run, and cannot be: exact mode is distributionally exact, but it spends the RNG differently, so the seed→song mapping changes (SPEC_DRAFT §1, STATUS_DRAFT). What does hold: score.abc / abc_tokens.npy / prefix.npy identical to master (draft touches only the semantic phase); same seed + head + K repeats bit-identically (2 runs, FLAC `bd226b17…` both); `len(semantic.npy)+1 == output_tokens` (6123+1); `--draft-extract-only` (head loaded, extraction on, nothing drafted) gives semantic.npy and FLAC identical to master. Distributional exactness is item 5's χ² test. |
| 5 | `yue2-draft-accept` in both builds | **PASS** 41/41 both. Also both builds: `yue2-guidance` 271, `yue2-sampler-diff` 0 mismatches, `yue2-handover` 109, `yue2-bars` 71 — all PASS. |
| 6 | README section | **done** (see above). |

## Measured on the Arc (`--gpu 1`, one song, seed 1, Q8_0 AR, Q8_0 head, K 2, λ 1)

| run | abc s | semantic tokens | semantic s | tok/s | nar s | vae s | e2e s |
|---|---|---|---|---|---|---|---|
| master (`c12f0cd`) | 19.24 | 6141 | 49.88 | 123.11 | 45.09 | 5.01 | 120.42 |
| ON, no draft | 19.33 | 6141 | 50.02 | 122.77 | 45.34 | 5.00 | 120.91 |
| ON, `--draft` | 19.28 | 6124 | 42.70 | **143.44** | 46.17 | 4.98 | 115.37 |

Semantic phase **1.17×** (143.44 / 122.77), whole song 4.6 % faster. Draft line:
`3357 rounds, 1.82 tokens/round, acceptance per depth 0.551 / 0.495, draft 2.93 + verify 9.52 ms/round`
— in line with STATUS_DRAFT (Q8_0 K2: 1.149× over 7 songs, d1 0.544, 2.95 + 9.49 ms/round).

## Exact commands

```sh
git cherry-pick -n 10dd57d        # resolve CMakeLists.txt, git add CMakeLists.txt
cmake -B build_e3on  -DCMAKE_BUILD_TYPE=Release -DYUE2_BUILD_TESTS=ON -DYUE2_EAGLE3=ON
cmake -B build_e3off -DCMAKE_BUILD_TYPE=Release -DYUE2_BUILD_TESTS=ON -DYUE2_EAGLE3=OFF
nice -n 10 cmake --build build_e3on -j8; nice -n 10 cmake --build build_e3off -j8
for t in yue2-draft-accept yue2-guidance yue2-sampler-diff yue2-handover yue2-bars; do build_e3on/$t; build_e3off/$t; done

# master baseline: git archive c12f0cd into scratch, llama.cpp symlinked, built the same way
M="--ar yue2-ar-q8_0.gguf --nar yue2-nar-f16.gguf --vae yue2-vae-f32.gguf --gpu 1 --request request.json"
build_e3off/yue2 song $M --out off.flac --artifacts off
build_e3on/yue2  song $M --out on.flac  --artifacts on
build_e3on/yue2  song $M --out draft.flac --artifacts draft \
	--draft YuE2-3B-EAGLE3-draft-Q8_0.gguf --draft-k 2 --draft-trace
build_e3off/yue2 song $M --out x.flac --draft YuE2-3B-EAGLE3-draft-Q8_0.gguf   # -> exit 1
```

Head: `hf download engival/YuE2-3B-EAGLE3-draft-GGUF YuE2-3B-EAGLE3-draft-Q8_0.gguf` (was not
in the cache). Request: an unguided seed-1 take (`cot: full`, `cfg_scale: null`) from the
author's render set. Scratch renders deleted; logs kept in `tests/out/eagle3_merge/`.

## Open

- Spec item 4's "semantic tokens identical to the no-draft run" is not achievable by design;
  the SPEC wording should be "same distribution; same score; repeatable per seed".
- Git state: the branch's files and `CMakeLists.txt` (resolution + switch) are staged
  (cherry-pick `-n`); the header stubs, the `check_draft_args` line, README, CLAUDE.md,
  SPEC_DRAFT/STATUS_DRAFT status lines and this file are unstaged. `docs/SCORE_RECIPES.md`, `scripts/abc_transpose.lua` untouched (user's edits).
- Not run: the AMD (coordinator's), lossy λ > 1, `yue2 batch --draft` render.
