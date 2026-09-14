# SPEC_BATCH — stage 5: multi-song AR decode

Contract for the C++ agent. Read `SPEC_AR.md`, `src/STATUS_AR.md`,
`SPEC_SINGLE.md`, `src/STATUS_SINGLE.md` and `src/stage_ar.cpp` first. Nothing
in this spec changes any math, any GGUF, or the NAR/VAE stages; it changes how
many songs the AR stage decodes per `llama_decode`. Every existing golden must
keep passing.

## 1. Why

`yue2 song` on a 156 s song (Arc, `src/STATUS_SINGLE.md` §6):

| stage | seconds | GPU state |
|---|---:|---|
| AR load | 1.1 | |
| abc + semantic decode | 51.8 | **one token per `llama_decode`, ~106 tok/s: launch/latency-bound, GPU mostly idle** |
| NAR load + prefill + ODE | 87.7 | compute-bound (attention) |
| VAE | 13.4 | compute-bound (measured when it still ran as a child on the exact-F32 path; in-process fp16-staged since 2026-09-13, SPEC_SINGLE §2.2) |

Model loading is 4.5 s of 155 (residency mode: rejected, `docs/ROADMAP.md`).
The AR decode is a third of the run and does not saturate the card: weights are
2.2 GB Q8_0, so bandwidth alone would allow several hundred tok/s per stream,
and decoding N sequences in one batch costs close to what one costs. Decoding N
songs' abc + semantic phases together should turn N × 52 s into not much more
than 52 s. The NAR and VAE are already compute-bound and gain nothing from
batching; they stay strictly per song.

Hypothesis to measure, not assume: aggregate AR throughput scales near-linearly
to N = 4 on both cards and is still rising at N = 8. §7 says what to report.

## 2. Scope and non-goals

In scope:
- One `llama_context` with `n_seq_max = N` decoding up to N independent song
  requests in lockstep, each sequence in its own phase (abc or semantic), each
  with its own RNG, history, sampling constants, stop condition and artifacts.
- A `yue2 batch` subcommand that takes a list of jobs and runs AR batched, then
  NAR and VAE per job. `yue2 ar` gains the same list form.
- `yue2 song` and `yue2 ar` single-request behaviour unchanged, bit for bit.

Not in scope (say so in STATUS, do not build):
- Any scheduler, queue file format, daemon or server. The batch is a list of
  requests given on the command line or in one JSON file; how a caller builds
  that list is the caller's business. **No dependency on, or knowledge of, any
  particular driver** (the author's own batch script is one caller; it gets no
  special treatment and is not mentioned in the binary or the README beyond
  "a driver can…").
- Batching the NAR or VAE. Overlapping a job's NAR with the next batch's AR on
  a second GPU (nice, later, needs a thread and two device contexts).
- Classifier-free guidance (`cfg_scale`), still dies as today.
- Bit-identical output between batched and single decode (§6 — it is not
  achievable on the GPU and we do not pretend).

## 3. Interface

### 3.1 `yue2 batch`

```
yue2 batch --jobs jobs.json [--parallel N] [--gpu G] [--device cpu|vulkan]
           [--ar A.gguf] [--nar B.gguf] [--vae C.gguf] [--nar-f32] [--steps 32]
           [--continue-on-error]
```

`jobs.json` is a JSON array; each element is exactly the per-song options
`yue2 song` takes, minus the device/model options that are per batch:

```json
[
  { "request": "songs/a/request.json", "out": "songs/a/a.flac", "artifacts": "songs/a/art", "seed": 831001 },
  { "request": "songs/b/request.json", "out": "songs/b/b.flac" },
  { "request": "songs/c/request.json", "out": "songs/c/c.flac", "noise": "songs/c/noise.npy" }
]
```

`request` and `out` are required; `artifacts`, `seed`, `noise` are optional with
the same meaning as the `yue2 song` flags. Relative paths are relative to the
process's cwd, not to the JSON file (simplest rule; state it in the README).
Duplicate `out` or `artifacts` paths in one file are a usage error before
anything loads.

Execution: jobs are taken in file order. Up to `--parallel` (default 4) are
active in the AR context at once; as one finishes its semantic phase its slot is
refilled with the next unstarted job (§4.4), so `--parallel` is a ceiling on
concurrent sequences, not a wave size. When every job has finished the AR,
NAR + VAE run per job in file order (§4.5). Per-job stdout lines are prefixed
`[k/N name]` where `name` is the `out` basename, so a log stays readable.

Errors: a job whose request fails validation is reported and, without
`--continue-on-error`, aborts the batch **before the model loads** — validation
of every request happens first. A job that fails at runtime (a `die()` today)
aborts the whole process exactly as today unless `--continue-on-error`, which
turns the per-job failure into a recorded status and continues; the exit code is
then non-zero if any job failed. The summary is written to stdout as one final
JSON line per job when `--summary FILE` is given (a JSON array of
`{ "out", "status": "ok"|"error", "error", "audio_seconds", "e2e_seconds", … }`)
so a driver can read results without parsing the log. Keep it small.

### 3.2 `yue2 ar` multi-request

```
yue2 ar -m AR.gguf --requests jobs.json [--parallel N] …
```

Same `jobs.json` shape, but `out` is ignored and `artifacts` is required per
job (the AR stage's output *is* the artifacts directory). This is the form for a
caller who runs the NAR/VAE elsewhere or later. `--request R --artifacts D`
(single) keeps working and is literally a batch of one.

### 3.3 `yue2 song`

Unchanged. Internally `run_song` builds a one-job batch and calls the same code.

### 3.4 Flags that stay per batch

`--gpu`, `--device`, `--ar/--nar/--vae`, `--nar-f32`, `--steps`, `--threads`,
`--greedy`, `--max-abc`, `--max-semantic`. A per-job override for these is not
offered: `--nar-f32` is per process (Vulkan env), the rest would only add
surface. Per-job `seed`/`noise`/`artifacts` cover what actually differs between
songs.

## 4. Program design (decided — report if it breaks)

### 4.1 Layout

- `src/stage_ar.cpp`: `generate()` (one sequence, one phase, lockstep loop)
  becomes a per-sequence state machine driven by one shared decode loop. The
  existing `sample_step()`, `Sampling`, prefix construction, `write_artifacts()`
  are used as they are; they are already per-request pure functions. Move,
  do not rewrite.
- `struct ArJob` (request path, artifacts, seed override, noise path, out path
  for the summary) and `struct ArBatchParams` (model/device/parallel + the
  sampling overrides). `run_ar(const ArParams &, ArResult *)` stays as a
  wrapper: one `ArJob`, `parallel = 1`.
- New `int run_ar_batch(const ArBatchParams &, std::vector<ArJob> &, std::vector<ArResult> &)`.
- `src/stage_song.cpp`: `run_song` → a one-job call into a new
  `run_batch(const BatchParams &, jobs)`; `run_batch` does §4.5.
- `src/yue2.cpp`: dispatch adds `batch`. `yue2-ar` one-line main gains nothing
  new besides parsing `--requests`.
- Build dir for this work: `build_batch/`. Device 1 (Arc) and CPU only for
  development; device 0 is timed by the coordinator (CLAUDE.md rule).

### 4.2 The llama context

One context for the whole batch. Line references are into the submodule at
`llama.cpp/` (v0.4.0-108-g8ea290247) and into `src/stage_ar.cpp`.

- `n_seq_max = parallel` (`llama_context_params::n_seq_max`, llama.h:363;
  libllama caps it at `LLAMA_MAX_SEQ` = 256, llama-cparams.h:8).
- `kv_unified = false`. This is already what `llama_context_default_params()`
  returns (llama-context.cpp:3653), so today's `n_seq_max = 1` context is the
  same kind of cache with one stream. Each sequence gets its own KV *stream*
  of `n_ctx_seq` cells (llama-kv-cache.cpp:84 `n_stream = unified ? 1 :
  n_seq_max`; llama-model.cpp:2233/2283 pass `cparams.n_ctx_seq` as the
  per-stream size), so a sequence's attention scans only its own cells. Unified
  would make every sequence attend (masked) across the whole `N × n_ctx_seq`
  pool — N× the KV reads per step, the "bad performance" llama.h:407 warns
  about. **Not yet exercised on this box's Vulkan**: the author's llama-server
  launcher runs it with `-kvu`. It is the default path
  of every multi-sequence program in the tree and the graph it builds is
  backend-agnostic (4-D K/V views with `ne[3] = n_stream`; ggml-vulkan's
  `supports_op` only asks `a->ne[3] == b->ne[3]`, ggml-vulkan.cpp:19254), so it
  is expected to work — run acceptance test §5.2 first. Fallback if it does
  not: `kv_unified = true`, same code, only the ubatch splitting changes (§4.7
  item 3 stops applying); report it as a deviation.
- Per-sequence capacity: the same `n_ctx_want` formula as today
  (stage_ar.cpp:932-942: `prefix_abc + abc_ids + (do_abc ? max_abc : 0) + 2 +
  max_semantic + 8`, clamped to `CONTEXT` = 24576), computed as the **max over
  all jobs in the batch** (prefix lengths differ per job; the abc and semantic
  `max_tokens` do not). Request `cparams.n_ctx = n_ctx_want * parallel`. What
  libllama then does (llama-context.cpp:288-303): pads `n_ctx` up to a
  multiple of 256, sets `n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)` and
  rewrites `n_ctx = n_ctx_seq * n_seq_max` (its "rounding down" warning text
  is wrong — both steps round up). So `llama_n_ctx_seq(ctx) >= n_ctx_want`
  always holds; assert it anyway. The accessor exists: `llama_n_ctx_seq`
  (llama.h:568), alongside `llama_n_seq_max` (llama.h:571). Use the padded
  `llama_n_ctx_seq(ctx)` in the printed KV estimate — that is what is
  allocated per stream. `n_ctx_seq > n_ctx_train` (24576) is only a warning
  (llama-context.cpp:325); the `CONTEXT` clamp keeps us at or under it.
- `n_batch = n_ubatch = max(largest single prefill in the batch, 512)`, where
  "largest single prefill" is what stage_ar.cpp:926-931 calls `max_feed`: the
  abc prefix, or the **whole semantic prefix** for a cot=off / external-abc
  job. Prefill and bridge feeds of one job happen in their own `llama_decode`
  calls (§4.4), never mixed with the lockstep step, so the lockstep batch is at
  most `parallel` tokens and `n_batch` needs no more than that. (`decode_feed`
  already chunks by `llama_n_batch`, so a smaller `n_batch` would also work;
  keep today's sizing so `parallel 1` behaves exactly as today.)
- Budget: KV is 28 layers × 8 kv heads × 128 × 2 (K,V) × 2 B = **112 KiB per
  token per sequence**, F16 (`type_k`/`type_v` default F16,
  llama-context.cpp:3646-3647; hparams read back from the GGUF: `block_count`
  28, `head_count_kv` 8, `key_length` 128). At the usual ~13.7k-token
  (padded: 13824) per-sequence context that is ~1.5 GiB per slot: `parallel 8`
  = ~12 GiB KV + 2.2 GiB weights + compute. Fits the 24 GB AMD and the 32 GB
  Arc; `parallel 12` does not fit the AMD next to nothing else. Print the
  per-slot and total KV estimate before allocating and die with a clear message
  if `llama_init_from_model` returns NULL (libllama's reason reaches stderr —
  `quiet_log`, stage_ar.cpp:110, passes WARN and above).
- The `--dump-logits` path (goldens) is `parallel 1` only and unchanged.

### 4.3 Per-sequence state

```
struct Seq
{
	int                      slot;        // llama seq_id, 0..parallel-1
	int                      job;         // index into jobs, -1 = idle
	Phase                    phase;       // ABC, SEMANTIC, DONE
	std::vector<llama_token> history;     // ids generated in the current phase (penalty window + output)
	int                      step;        // step within the phase (min_tokens mask)
	std::mt19937_64          rng;         // re-seeded at each phase start with the job's seed, exactly as generate() does today
	llama_pos                pos;         // next position to write == tokens this slot holds in the cache
	llama_token              next;        // token to feed in the next lockstep batch (sampled from the previous logits)
	int                      i_batch;     // index of this slot's token in the batch last submitted, -1 = none
	GenStats                 st_abc, st_sem;
	double                   t_phase0;
	// request-derived, built once when the job enters the slot:
	Request req; std::vector<llama_token> prefix_abc, abc_ids, prefix_sem; bool do_abc, legacy_off;
};
```

The RNG rule matters: today `generate()` constructs `std::mt19937_64 rng(seed)`
at its top (stage_ar.cpp:342), i.e. once per *phase*, so abc and semantic both
start from the same seed state; `rng.seed(seed)` on the member is that same
state. Keep it exactly, per sequence. `history` is `generate()`'s `out`
(cleared at stage_ar.cpp:343): it is both the repetition-penalty window
`sample_step` reads and the phase's output. The noise generator
(`common/noise.hpp`) is untouched and stays per job.

### 4.4 The decode loop

```
validate every job's request                                  (§3.1: before the model loads)
load model, create ctx (§4.2); batch = llama_batch_init(max(n_batch, parallel), 0, 1)
fill slots 0..min(parallel, jobs)-1: enter(job, slot)
while any slot active:
	batch.n_tokens = 0
	for each active slot (in slot order):
		seq.i_batch = batch.n_tokens
		add(batch, seq.next, pos = seq.pos++, seq_id = slot, logits = 1)  (one token per slot, always)
	ret = llama_decode(ctx, batch); if ret != 0: die with ret in the message (§4.7 item 6)
	for each active slot (in slot order):
		advance(seq, llama_get_logits_ith(ctx, seq.i_batch))       (NULL → die; §4.7 item 2)

advance(seq, logits):
	token = sample_step(logits, n_vocab, sampling[phase], history, step, phase_abc, legacy_off, rng)
	first sample of the job → ttft
	if token == end_token(phase):            finish phase, truncated = false   (token is NOT pushed, as today)
	else:
		history.push_back(token); step++
		if step < max_tokens:                seq.next = token; return          (decoded by the next lockstep batch)
		finish phase, truncated = true       (the kept token has not reached the cache yet — the feed below carries it)
	if phase was ABC:
		phase = SEMANTIC; history.clear(); step = 0; rng.seed(job seed)
		feed(seq, (truncated ? [token] : []) + [ABC_END, MUSIC_START], expect_pos = prefix_sem.size())
	else:
		write artifacts, store ArResult, llama_memory_seq_rm(mem, slot, -1, -1), seq.pos = 0
		refill: enter(next queued job, slot), or mark idle

enter(job, slot):
	build prefixes as run_ar does (stage_ar.cpp:876-920, 1002-1010); phase = ABC if do_abc else SEMANTIC
	assert llama_memory_seq_pos_max(mem, slot) == -1                                  (slot really is empty)
	feed(seq, phase prefix, expect_pos = prefix.size())

feed(seq, tokens, expect_pos):
	own llama_decode call(s), chunked by n_batch as decode_feed does, seq_id = slot, pos = seq.pos++,
	    logits = 1 on the last token only
	assert llama_memory_seq_pos_max(mem, slot) + 1 == expect_pos                       (the expect_pos check, per slot)
	advance(seq, llama_get_logits_ith(ctx, index of the last token in that final chunk))
	    — IMMEDIATELY: the next llama_decode (any slot's) overwrites the logits buffer
```

(changed from: the two bridge tokens `ABC_END, MUSIC_START` ride in the next
lockstep batch, because (a) with `kv_unified=false` llama's `split_equal` puts
a slot's second token into a ubatch of its own anyway — same GPU cost as a
separate call, §4.7 item 3; (b) an abc phase that ends by truncation must also
carry its last kept token, so the bridge is two *or three* tokens; (c) a
separate feed keeps the `expect_pos` check and the first-token sampling exactly
where `generate()` has them (stage_ar.cpp:347-369), and the lockstep batch stays
exactly one token per active slot, so `i_batch` is trivially right.)

Rules carried over from `generate()`, each of which was a bug once:

- **Every kept token is decoded**, including the last one before `max_tokens`
  (stage_ar.cpp:380-390), because the semantic phase continues from the abc
  cache. In the batched loop that token travels in the bridge feed. The one
  permitted difference: a *semantic* phase that ends by truncation no longer
  decodes its last token — the sequence is removed next, and `generate()`'s
  decode of it was wasted work. Artifacts are unaffected.
- **The `expect_pos` check** stays, per slot: after prefill and after the
  abc→semantic bridge, `llama_memory_seq_pos_max(mem, slot) + 1` (llama.h:800,
  per-sequence, works per stream) must equal the prefix length being recorded.
  It is the proof that `prefix.npy` and the KV cache agree.
- **Prefill** of a joining job runs in separate `llama_decode` calls, before
  the next lockstep step. Mixing a 2000-token prefill into the 1-token-per-slot
  decode batch works in llama (the server and `examples/parallel` do it, with
  `n_batch`-sized views) but makes `n_ubatch` and the logits indexing harder to
  reason about for no measurable gain (prefill is 1–3 s per job, once). If the
  cot=off / external-abc path applies to a job, its "prefill" is the whole
  semantic prefix and it starts in the SEMANTIC phase; same code.
- The first sampled token's timestamp gives `ttft_seconds`; per-phase
  `seconds` is measured from that sequence's prefill start to its phase end,
  so per-job stats stay comparable with single runs (they will be longer in
  wall-clock, since the slot shares the GPU — say so in STATUS, and also
  report the batch-level aggregate tok/s, which is the number that matters).
- `sample_step` allocates a `std::vector<float>` of `n_vocab` (184704) per
  call; at `parallel 8` that is 8 copies per step on the CPU, ~6 MB. Measure
  before optimising; if the CPU side shows in the profile, reuse one scratch
  buffer per `Seq`. Do not parallelise sampling across threads in this stage.

### 4.5 `run_batch` (song-level)

```
run_ar_batch(all jobs)           → vector<ArResult>, artifacts written per job
free the llama context and model  (exactly as run_ar does; the NAR reloads the AR weights itself, §STATUS_SINGLE dev. 12)
for each job in order:
	noise → run_nar → run_vae → config.json / result.json    (today's run_song body, per job)
```

The AR frees before the first NAR so peak VRAM is `max(AR batch, one NAR)`, not
the sum. With `parallel 8` the AR batch (~14 GiB) is the new peak on the AMD;
with `parallel 4` (~8.5 GiB) it is about even with the NAR's 7.9 GiB. Default 4
for that reason. Print peak estimates; measure real ones (§7).

Artifacts: each job's directory gets exactly the file set `yue2 song` writes
today. Without `artifacts` a per-job temp directory is used and removed on
success, as today; `g_temp_artifacts` becomes a list. `result.json` gains
`"batch": { "parallel": N, "slot": k, "jobs": total }` so a rendered song
records how it was made (§6 explains why that matters).

### 4.6 Failure semantics

Default: identical to today. Any `die()` ends the process, exit 1, the
finished jobs' outputs are on disk, unfinished ones are not (temp artifacts
removed by the `atexit` handler). `--continue-on-error` is implemented at job
boundaries only: request validation errors. The NAR and the VAE report failure
by `die()`, so both are fatal even under `--continue-on-error` — the VAE was the
exception while it ran as a child process and its exit code could be read; since
the fork was removed (2026-09-13, SPEC_SINGLE §2.2) it behaves like the NAR.
Turning either stage's `die()`s into returns is a separate piece of work.
Inside the shared AR decode loop, a failure in one sequence is a process-level
bug (malformed sampled token, decode error) and stays fatal.

### 4.7 Implementation notes for the C++ agent

Concrete traps, each verified against the submodule (`llama.cpp/`,
v0.4.0-108-g8ea290247) or `src/stage_ar.cpp`:

1. **No `common_batch_add`.** `LLAMA_BUILD_COMMON` is OFF (CMakeLists.txt:17),
   so write the five-line equivalent of common/common.cpp:1838-1852.
   `llama_batch_init(n, 0, 1)` leaves every member **uninitialised**
   (llama.h:954-960): set `token`, `pos`, `n_seq_id[i] = 1`, `seq_id[i][0]`
   and `logits[i]` for every token, and `n_tokens` yourself. `logits` is a
   non-NULL array here, so a token you do not want output from needs an
   explicit 0 (a NULL `logits` pointer means "last token only", llama.h:257-260
   — that is what `llama_batch_get_one` relies on). `llama_batch_get_one`
   (used by `decode_feed`, stage_ar.cpp:322) pins `seq_id` to 0 and `pos` to
   NULL: it cannot be used for any slot but 0, so `decode_feed` gains
   (slot, pos) parameters or a sibling.
2. **Logits index = the token's index in the batch passed to *this*
   `llama_decode`** (llama.h:1037-1041; `output_ids` translates batch index →
   row, llama-context.cpp:872-874, and llama-context.cpp:1989-2032 undoes the
   reordering `split_equal` did). Never index `llama_get_logits()` by slot
   number or by row. The pointer is into the context's output buffer and is
   invalid after the next `llama_decode`. It returns NULL for a token that had
   `logits = 0` or an out-of-range index (llama-context.cpp:889-903; a debug
   build aborts instead). Reference pattern: examples/parallel/parallel.cpp:297-299
   (`client.i_batch = batch.n_tokens` before `common_batch_add`) and :439
   (sample with `i_batch - view offset`); tools/server/server-context.cpp:3601
   and :3851 do the same.
3. **`kv_unified=false` splits the batch with `split_equal(n_ubatch,
   sequential=true, 0)`** (llama-kv-cache.cpp:713). A ubatch takes the *same*
   number of tokens from each participating sequence (llama-batch.cpp:566-600)
   and only from sequences with **consecutive seq ids** (llama-batch.cpp:537-539),
   because the streams a ubatch touches must be one contiguous range
   (`find_slot` s0..s1, llama-kv-cache.cpp:993-994). Consequences: (a) an idle
   slot *between* active slots splits a step into two ubatches = two graph
   evaluations ({0,2,3} → {0},{2,3}; {1,2,3} is one ubatch, leading/trailing
   idle slots are fine). Refilling from the queue closes holes while jobs
   remain; in the drain phase (queue empty, songs finishing at different
   lengths) it is a real tax — **measure and report it (§7.1), do not fix it in
   this stage** (the fix would be `llama_memory_seq_cp` slot→hole plus
   `seq_rm`, a full stream copy, llama-kv-cache.cpp:827-846). (b) Two tokens
   from one slot in a lockstep batch become a second ubatch — why the bridge is
   a separate feed (§4.4). Under the `kv_unified=true` fallback the splitter is
   `split_simple` and none of this applies.
4. **Positions are validated.** A sequence's tokens in a batch must start at
   `seq_pos_max + 1` and be contiguous (llama-batch.cpp:290-318), and every
   `seq_id` must be `< n_seq_max` (llama-batch.cpp:61); either violation makes
   `llama_decode` return -1 after an ERROR log line, which `quiet_log`
   (stage_ar.cpp:110) does print. Keep `Seq.pos` explicit, reset it to 0 after
   `llama_memory_seq_rm`, and assert `llama_memory_seq_pos_max(mem, slot) == -1`
   before a slot is refilled (-1 is the documented empty-sequence value,
   llama.h:797-802). `n_tokens > n_batch` is still a `GGML_ASSERT` abort
   (llama-context.cpp:1724) — keep the chunking `decode_feed` does.
5. **The first token of each phase is sampled from the feed's logits**
   (stage_ar.cpp:363-369), before anything else is decoded; in the batched
   loop that means inside `feed()`, not at the top of the next lockstep step.
   `ttft` follows the same rule.
6. **`llama_decode` return codes** (llama.h:985-990): 0 ok; 1 = no KV slot
   (memory restored — with §4.2 sizing this means a bookkeeping bug, so die
   with the code, do **not** copy the server's halve-and-retry,
   server-context.cpp:3692-3735 / parallel.cpp:405-420); 2 = aborted; -1 =
   invalid batch (item 4); < -1 = compute error, memory partially updated.
   Always print `ret` in the `die()` text.
7. **`llama_memory_seq_rm(mem, slot, -1, -1)` never fails for a whole
   sequence** (llama.h:745-747) — no need to check its return; do check the
   `pos_max == -1` postcondition (item 4) because §5.4 depends on it.
8. **RNG and history are per phase** (stage_ar.cpp:342-343); `end` token is
   `ABC_END` for abc and `MUSIC_END` for semantic (stage_ar.cpp:340) and is
   never pushed to `history` (stage_ar.cpp:371-376); truncation is `!eos`
   (stage_ar.cpp:399). `sample_step` is already pure in `(logits, history,
   step, rng)` — reuse it untouched.
9. **`n_outputs_max_per_seq` defaults to 1** (llama-context.cpp:3626) but is
   only enforced when backend sampler chains are attached
   (llama-context.cpp:1677-1704). We sample on the CPU: leave
   `cparams.samplers` NULL and the limit is moot.
10. **Attention cost per step follows the longest active sequence**, not the
    average: `n_kv` for a ubatch is the max over its streams, padded to 256
    (llama-kv-cache.cpp:1250-1262). The identical-request benchmark (§7.1)
    hides this; the mixed-length end-to-end run (§7.2) shows it. Flash
    attention stays `LLAMA_FLASH_ATTN_TYPE_AUTO` as today.

## 5. Goldens and acceptance

All existing goldens and `tests/regress.sh` pass unchanged. Then:

1. **Batch of one is today's binary.** `yue2 ar --request R --artifacts D` and
   `yue2 ar --requests one_job.json` (one element, `parallel 1`) produce
   byte-identical `prefix.npy`, `semantic.npy`, `plan.json` to the current
   `build/yue2` for `tests/golden/ar_prefix_ids` / `ar_greedy_32` and for one
   full sampled song (compare `semantic.npy` sha256). `yue2 song` likewise
   byte-identical FLAC for one seeded run (Arc; note the card).
2. **Greedy batch agrees with greedy single.** The same request ×4 with
   `--greedy --max-semantic 512`, `parallel 4`: all four `semantic.npy` are
   identical to each other. Compare to the single run: expected identical;
   if they diverge, report the first differing position and the two logit
   margins there (this tells us whether it is a near-tie, §6). Diverging on a
   genuine near-tie is not a failure; diverging elsewhere is a bug.
3. **Mixed phases.** Four *different* requests of different prefix lengths,
   sampled, `parallel 2` (so refill happens): every job writes a complete
   artifact set, every `expect_pos` check passes, `result.json` per job is
   well-formed, and each job's `semantic.npy` decodes through NAR+VAE to a
   FLAC of the expected length (`1920·T − 64` samples).
4. **Refill correctness.** After a slot is reused, its new job's `prefix.npy`
   matches an independent single run of that job (prefix is deterministic,
   so this must be exact) — proves `llama_memory_seq_rm` really cleared it.
5. **cot=off and external-abc jobs** in the same batch as normal jobs
   (they enter directly in the SEMANTIC phase); artifacts as today.
6. **`--continue-on-error`**: one job with an invalid request among three
   valid ones → the two others render, exit non-zero, summary lists all three.

## 6. Determinism — what to promise in the README

Batched decode runs the matmuls with N rows instead of one; ggml-vulkan picks
different kernels / reduction orders for different shapes, so per-sequence
logits differ from single-stream logits in the low bits. Greedy argmax is
almost always unaffected; sampled decode diverges at the first step where the
top-p cut or the uniform draw lands within that noise, and from then on it is a
different (equally valid) song. Therefore:

- **`seed` reproduces a song only together with the same `parallel` value,
  the same batch composition (which slots were live at each step), the same
  GGUFs and the same card.** `result.json` records `parallel`, `slot`, and
  the card for this reason.
- `parallel 1` (or plain `yue2 song`) is the reproducible path and stays the
  default for `song`.
- Measure how often it actually bites: same seed, same request, `parallel 1`
  vs one slot of `parallel 4`; report whether `semantic.npy` matched and if
  not at which token it diverged. One data point per card is enough.

This is the same property llama-server has with slots; we are just writing it
down instead of letting users discover it.

## 7. Report (STATUS_BATCH.md)

Tables, exact commands, Arc unless stated; the coordinator adds device 0.

1. Aggregate AR throughput vs `parallel` ∈ {1, 2, 4, 8}: identical requests
   (so lengths match), sampled, full length. Columns: wall seconds for the AR
   stage, aggregate tok/s, per-slot tok/s, GPU peak (free-memory poller as in
   STATUS_SINGLE §6), CPU time of the process (`sample_step` share). Add
   one drain-phase data point: per-step wall time with all slots active vs
   with a hole in the slot set (§4.7 item 3), `parallel 4`.
2. End to end for a 4-job batch vs 4 × `yue2 song`: wall seconds, and the
   breakdown AR / NAR / VAE. This is the headline number.
3. The determinism data point of §6.
4. Peak VRAM at `parallel 4` and `8`, against the §4.2 estimate.
5. Deviations from this spec, numbered, with why — the usual table.

## 8. README additions

A "Render several songs" section: the `jobs.json` shape, `--parallel` and its
VRAM cost (112 KiB/token/sequence), the §6 determinism paragraph in three
sentences, and one line saying the NAR/VAE remain per song because they are
compute-bound. No mention of any particular driver.
