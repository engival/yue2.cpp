// The EAGLE-3 draft head's runtime. SPEC_DRAFT.md §2–§3; the reference is
// llama.cpp common/speculative.cpp common_speculative_impl_draft_eagle3.

#include "draft_eagle3.hpp"

#include "llama-ext.h"

#include "common/util.hpp"

#include <cstring>

std::string eagle3_init(Eagle3 & e, llama_model * draft, llama_context * tgt,
	llama_context_params cparams)
{
	const llama_model * target = llama_get_model(tgt);

	char arch[64] = "";
	llama_model_meta_val_str(draft, "general.architecture", arch, sizeof(arch));
	if (std::string(arch) != "eagle3")
	{
		return std::string("the draft head's architecture is \"") + arch + "\", not eagle3";
	}
	// speculative.cpp's ctor makes the same checks (559–565, 603–612).
	if (llama_model_target_layer_ids_n(draft) != 3)
	{
		return "the draft head reads " + std::to_string(llama_model_target_layer_ids_n(draft)) +
		       " target layers, eagle3 reads 3";
	}
	const int32_t * ids = llama_model_target_layer_ids(draft);
	for (int k = 0; k < 3; k++)
	{
		if (ids[k] < 0 || ids[k] >= llama_model_n_layer(target))
		{
			return "the draft head reads target layer " + std::to_string(ids[k]) +
			       ", the target has " + std::to_string(llama_model_n_layer(target));
		}
		e.layer[k] = ids[k];
	}
	if (llama_vocab_n_tokens(llama_model_get_vocab(draft)) !=
	    llama_vocab_n_tokens(llama_model_get_vocab(target)))
	{
		return "the draft head's vocabulary is not the target's";
	}

	e.tgt        = tgt;
	e.n_embd_tgt = llama_model_n_embd(target);
	e.n_embd     = llama_model_n_embd(draft);

	// The target's parameters (common_speculative_init_result, speculative.cpp
	// 2593–2640): one sequence over the target's per-slot context, which already
	// covers the prefix plus the semantic cap. The head has no token_embd of its
	// own and reads the target's through ctx_other.
	cparams.n_ctx     = llama_n_ctx_seq(tgt);
	cparams.n_seq_max = 1;
	cparams.ctx_other = tgt;
	e.ctx = llama_init_from_model(draft, cparams);
	if (e.ctx == nullptr)
	{
		return "failed to create the draft head's " + std::to_string(cparams.n_ctx) + "-token context";
	}
	e.mem = llama_get_memory(e.ctx);
	llama_set_embeddings_nextn(e.ctx, true, true);
	e.dec = llama_batch_ext_init(e.ctx);
	e.enc = llama_batch_ext_init(e.ctx);
	e.g_last.assign((size_t) e.n_embd, 0.0f);
	e.prenorm.assign((size_t) e.n_embd, 0.0f);
	return "";
}

void eagle3_free(Eagle3 & e)
{
	if (e.dec != nullptr)
	{
		llama_batch_ext_free(e.dec);
		llama_batch_ext_free(e.enc);
	}
	if (e.ctx != nullptr)
	{
		llama_free(e.ctx);
	}
	e = Eagle3();
}

void eagle3_extract(Eagle3 & e, bool on)
{
	for (int k = 0; k < 3; k++)
	{
		llama_set_embeddings_layer_inp(e.tgt, (uint32_t) e.layer[k], on);
	}
}

void eagle3_features(Eagle3 & e, int first, int n)
{
	const size_t row = (size_t) e.n_embd_tgt;
	e.feat.resize((size_t) n * 3 * row);
	for (int k = 0; k < 3; k++)
	{
		const float * src = llama_get_embeddings_layer_inp(e.tgt, (uint32_t) e.layer[k]);
		if (src == nullptr)
		{
			die("draft: target layer %d input was not extracted", e.layer[k]);
		}
		for (int i = 0; i < n; i++)
		{
			memcpy(e.feat.data() + ((size_t) i * 3 + (size_t) k) * row,
			       src + (size_t) (first + i) * row, row * sizeof(float));
		}
	}
}

void eagle3_encode(Eagle3 & e, int n)
{
	// The encoder reads no KV and checks no position continuity (llama-context
	// encode()); placeholders past the cache, as upstream makes them.
	llama_batch_ext_clear(e.enc);
	llama_pos pos = llama_memory_seq_pos_max(e.mem, 0) + 1;
	for (int i = 0; i < n; i++)
	{
		const llama_embd row = { e.feat.data() + (size_t) i * 3 * (size_t) e.n_embd_tgt, 1,
		                         (size_t) (3 * e.n_embd_tgt) };
		const int32_t idx = llama_batch_ext_add_embd(e.enc, 0, row);
		if (idx < 0 || !llama_batch_ext_set_pos(e.enc, idx, &pos) ||
		    !llama_batch_ext_set_output_embd(e.enc, idx, true))
		{
			die("draft: cannot add feature row %d to the encoder batch (%d)", i, idx);
		}
		pos++;
	}
	const int32_t rc = llama_process(e.ctx, LLAMA_PROCESS_TYPE_ENCODE, e.enc);
	if (rc != 0)
	{
		die("draft: the encoder returned %d on %d rows", rc, n);
	}
	const float * out = llama_get_embeddings_nextn(e.ctx);
	if (out == nullptr)
	{
		die("draft: the encoder produced no output");
	}
	e.g.assign(out, out + (size_t) n * (size_t) e.n_embd);
}

// One (token, embedding) row of a decoder batch.
static void add_pair(Eagle3 & e, llama_token tok, const float * embd, llama_pos pos, bool output)
{
	const int32_t idx = llama_batch_ext_add_token(e.dec, 0, tok);
	if (idx < 0 || !llama_batch_ext_set_embd_token(e.dec, idx, { embd, 1, (size_t) e.n_embd }) ||
	    !llama_batch_ext_set_pos(e.dec, idx, &pos) ||
	    !llama_batch_ext_set_output_logits(e.dec, idx, output))
	{
		die("draft: cannot add token %d at position %d to the draft batch (%d)",
		    (int) tok, (int) pos, idx);
	}
}

const float * eagle3_draft(Eagle3 & e, llama_token tok, const float * embd, llama_pos pos)
{
	// One row and it is the output: the eagle3 graph selects no output rows, and
	// llama copies the first n_outputs rows of its logits and pre-norm state, so
	// an output row behind a non-output one would be handed the wrong row
	// (llama-context.cpp 2018–2034).
	llama_batch_ext_clear(e.dec);
	add_pair(e, tok, embd, pos, true);
	const int32_t rc = llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, e.dec);
	if (rc != 0)
	{
		die("draft: decode returned %d at position %d", rc, (int) pos);
	}
	// Logits and pre-norm share the output index (llama-context 2018–2034).
	const float * prenorm = llama_get_embeddings_nextn_ith(e.ctx, 0);
	const float * logits  = llama_get_logits_ith(e.ctx, 0);
	if (prenorm == nullptr || logits == nullptr)
	{
		die("draft: no output row at position %d", (int) pos);
	}
	memcpy(e.prenorm.data(), prenorm, (size_t) e.n_embd * sizeof(float));
	return logits;
}

void eagle3_feed(Eagle3 & e, const llama_token * toks, int n, llama_pos pos0)
{
	llama_batch_ext_clear(e.dec);
	for (int k = 0; k < n; k++)
	{
		add_pair(e, toks[k], e.g.data() + (size_t) k * (size_t) e.n_embd, pos0 + k, false);
	}
	const int32_t rc = llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, e.dec);
	if (rc != 0)
	{
		die("draft: feeding %d rows at position %d returned %d", n, (int) pos0, rc);
	}
}
