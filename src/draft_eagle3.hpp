// The EAGLE-3 draft head beside the AR target (SPEC_DRAFT.md §2, §3.2): its
// context, the target features it reads and the deferred boundary. A port of
// llama.cpp common/speculative.cpp's draft-eagle3 onto llama_batch_ext, since
// yue2 links libllama only; the sampling is the caller's.
#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct Eagle3
{
	llama_context *    tgt  = nullptr;
	llama_context *    ctx  = nullptr;   // the head's own, ctx_other = tgt
	llama_memory_t     mem  = nullptr;
	llama_batch_ext *  dec  = nullptr;   // (token, embedding) rows
	llama_batch_ext *  enc  = nullptr;   // feature rows
	int32_t            layer[3]   = {};  // target layer inputs, in the head's order
	int                n_embd_tgt = 0;
	int                n_embd     = 0;
	std::vector<float> feat;      // [rows][3 * n_embd_tgt], copied out of the target's last decode
	std::vector<float> g;         // [rows][n_embd], the encoder's output for them
	std::vector<float> g_last;    // the deferred boundary: g at pos_last
	llama_pos          pos_last = -1;
	std::vector<float> prenorm;   // the last eagle3_draft row's pre-norm state
};

// Checks `draft` is an eagle3 head for this target and creates its context from
// the target's parameters (§3.2). Returns "" or the error.
std::string eagle3_init(Eagle3 & e, llama_model * draft, llama_context * tgt,
	llama_context_params cparams);
void eagle3_free(Eagle3 & e);

// The target's layer-input extraction for the head's layers. Every call costs
// the target a graph re-reserve at its next decode (§2.1): phase boundaries only.
void eagle3_extract(Eagle3 & e, bool on);

// Copies batch rows first .. first + n − 1 of the target's last decode into
// `feat`. Before the target decodes again: the buffers are overwritten.
void eagle3_features(Eagle3 & e, int first, int n);

// g[0 .. n) = the encoder over feat[0 .. n). No KV is read or written.
void eagle3_encode(Eagle3 & e, int n);

// One draft row (tok, embd) at pos with output: returns its logits row and
// leaves its pre-norm state in `prenorm`. `embd` may point into `prenorm`: it
// is copied into the batch before the decode.
const float * eagle3_draft(Eagle3 & e, llama_token tok, const float * embd, llama_pos pos);

// Writes rows k = 0 .. n − 1: (toks[k], g[k]) at pos0 + k, no output.
void eagle3_feed(Eagle3 & e, const llama_token * toks, int n, llama_pos pos0);
