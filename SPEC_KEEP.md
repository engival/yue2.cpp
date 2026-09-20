# SPEC_KEEP — stage 7b: keep the start of an earlier render

Builds on stage 7 (SPEC_GUIDANCE.md, `src/STATUS_GUIDANCE.md`). Same files, same rules.

## 1. Why

A guided song (SPEC_GUIDANCE) is sampled from step 0, so it is a new performance
even when the score is fed back as `"abc"`: the seed's draws land differently.
The use case is the opposite: "this render is the one I like; keep it up to the
cut, change it from there". The semantic codes of the earlier render are forced
as history instead of being sampled.

## 2. Request

```json
"semantic_keep": { "file": "earlier/semantic.npy", "frames": 3400 }
```

- `file`: a `semantic.npy` as `yue2 ar|song --artifacts` writes it. A relative
  path is resolved against the directory of the request file (for `--requests`
  batches: of the batch file). Same dtype/shape rules as the NAR's `--codec` reader;
  share that reader, do not write a second one.
- `frames`: N, the number of leading codes to keep. `1 <= N <= len(file)`; when the
  file's last code is the end marker (if the file stores one) it cannot be kept.
  Omitted = request error (no implicit "all").
- Every kept value must be a valid codec index; otherwise request error.

Request errors (exact, tested messages, in `validate_request` where the file is
not needed, otherwise when the file is read, before any GPU work):
- `semantic_keep` without `"abc"` (the codes belong to one score; a freshly
  written score would not match them). Also with `abc_template`.
- a `guidance` entry with `frame < N` ("guidance frame F is inside the kept
  N frames"). `frame == N` is the normal case. A plain `cfg_scale` is allowed:
  its blank branch is simply born at step N.
- unknown keys inside `semantic_keep`.

## 3. Semantics

- Steps `0 .. N-1` are not sampled. The primary sequence is prefilled with
  `prefix + kept codes` (chunked by the batch size like any long prefix), `step = N`,
  and the first sampled token is step N.
- The rng is seeded as always and makes no draws for kept steps: the first draw
  is step N's.
- The repetition-penalty window is primed with the kept codes exactly as if they
  had been sampled (last `window` of them).
- Every per-step limit (max frames, context budget, end detection) counts kept
  steps as steps.
- Guidance: `guidance_enter` runs with `step = N`; branches prefilled there use
  `prefix + history` as they already do at a cut. An entry at `frame == N` with a
  `style` must work: at that moment the primary has just been prefilled and is
  relabelled like any other cut. Do it the simple way even if that means prefilling
  the old-tags sequence and the new one back to back.
- `semantic.npy` written by the run = kept codes + sampled codes, so the artifact
  is complete and the NAR needs nothing new.
- Works unguided and with `--parallel N` (it is only a longer prefill); the
  guided + `--parallel 1` rule is unchanged.

A request without `semantic_keep` MUST stay bit-identical to the stage-7 build.

## 4. Artifacts

- `plan.json`: `"semantic_keep": { "frames": N, "sha256": "<of the N kept codes as int32 LE>" }`.
  The path is not recorded (artifacts stay relocatable and carry no local paths).
- timing json: `kept_frames`, `keep_prefill_seconds`.
- one log line: `keep: N frames from <basename> prefilled (P + N tokens, T s)`.

## 5. Acceptance (device 1 or CPU within the limits below; never device 0)

1. No `semantic_keep`: `cmp` of `prefix.npy` / `abc_tokens.npy` / `semantic.npy`
   vs the stage-7 binary (`build_guidance/yue2`) for unguided, `cfg_scale 3`, and a
   guided request.
2. Kept codes: output `semantic.npy[:N]` equals the file's first N, for N = 1, 64,
   and len-1.
3. Teacher-forcing is faithful: take a `--greedy` run R (few hundred frames via the
   existing frame cap). Re-run `--greedy` with `semantic_keep` of R at N = 100. The
   continuation should equal R. Prefill-vs-decode numerics may flip a near-tie; if
   it diverges, report the first differing frame and the top-2 logit gap there
   rather than forcing a pass.
4. Determinism: same keep request twice → identical.
5. Guided + keep with `frame == N`: lifecycle log shows keep prefill, relabel,
   new positive, blank; first N codes equal the file; `guidance.json` `reached`.
6. `--verify-sampler` on a keep run: all sampled steps match.
7. Every §2 error at the CLI and in `yue2-guidance` (table cases; file-free ones).
8. Cost: keep N=1000 vs sampling 1000 steps; report prefill seconds.

## 6. Rules

- Build dir: `build_keep/` (new). Do NOT rebuild or touch `build/` or
  `build_guidance/` — the coordinator is rendering from `build_guidance/` right now.
  `nice -n 10 cmake --build build_keep -j8`.
- Model runs on the Arc (`--gpu 1`). CPU model runs only if unavoidable: `--threads 4`,
  `nice -n 19`, q8_0, a few hundred frames, one at a time. Never device 0.
- The tree has uncommitted stage-7 work: build on it, don't revert or reformat it.
  No commits. Public repo: no home paths, no song titles, no private docs.
- Report: append a "Stage 7b — semantic_keep" section to `src/STATUS_GUIDANCE.md`
  (results table, exact commands, deviations). README: document the key next to
  `guidance` with one example combining both. `docs/ROADMAP.md`: one Done line.
