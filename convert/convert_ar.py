#!/usr/bin/env python3
"""safetensors -> GGUF converter for the YuE2-3B autoregressive (AR) subset.

See ../SPEC_AR.md sections 1 and 2 for the contract this implements. The AR
subset is exactly llama.cpp's qwen3 architecture (hidden 2048, 28 layers, 16/8
heads, head_dim 128, ffn 6144, RMSNorm eps 1e-6, per-head q/k norm, RoPE theta
1e6 NEOX style, untied lm_head). Only the AR tensors are ported -- nar_*,
vae2llm, llm2vae, time_embedder, latent_pos_embed belong to stage 3.

The 184704-token tokenizer (151643 tiktoken BPE ranks + 208 named specials +
music start/end + 32768 codec tokens + 83 pad tokens) is baked into the same
GGUF, using the same tiktoken-ranks -> gpt2-style merges algorithm as
llama.cpp's convert_hf_to_gguf.py (conversion/qwen.py QwenModel.token_bytes_to_string
/ bpe, driven from conversion/base.py TextModel._set_vocab_qwen) -- reimplemented
here directly against qwen.tiktoken since we have no HF tokenizer directory.

Run with the venv_yue2 interpreter (torch CPU + safetensors + numpy + gguf):
	venv_yue2/bin/python convert_ar.py \
		--src /path/to/hf/snapshot/models--m-a-p--YuE2-3B/snapshots/<rev> \
		--out tests/out/yue2-ar-f16.gguf \
		[--type f16|bf16]

`--src` may be omitted if `m-a-p/YuE2-3B` is already in your huggingface_hub
cache -- see common.py:resolve_snapshot().
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from common import print_tensor_table, resolve_snapshot, sha256_of

REPO_ID = "m-a-p/YuE2-3B"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "tests" / "out" / "yue2-ar-f16.gguf")

EOD = 151643
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
VOCAB_SIZE = 184704

# from yue2.tokenization_yue2.YuE2TextTokenizer: the 208-entry specials list,
# ids 151643..151850, in order, with extra_196/197 renamed to <abc>/</abc>.
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
SPECIALS += [f"<extra_{i}>" for i in range(200)]
SPECIALS[204:206] = ["<abc>", "</abc>"]
assert len(SPECIALS) == 208
ABC_START = EOD + SPECIALS.index("<abc>")   # 151847
ABC_END = EOD + SPECIALS.index("</abc>")    # 151848
MUSIC_START = EOD + len(SPECIALS)           # 151851
MUSIC_END = MUSIC_START + 1                 # 151852
assert ABC_START == 151847 and ABC_END == 151848 and MUSIC_START == 151851 and MUSIC_END == 151852
assert CODEC_OFFSET == MUSIC_END + 1


# ── tiktoken ranks -> gpt2-style vocab + merges, copied from llama.cpp's own
# conversion/qwen.py (QwenModel.token_bytes_to_string / bpe), MIT -- see
# ../NOTICE.md "Code lineage". ──

def token_bytes_to_string(b: bytes) -> str:
	from transformers.convert_slow_tokenizer import bytes_to_unicode
	byte_encoder = bytes_to_unicode()
	return "".join(byte_encoder[ord(char)] for char in b.decode("latin-1"))


def bpe(mergeable_ranks: dict, token: bytes, max_rank: int | None = None) -> list:
	parts = [bytes([b]) for b in token]
	while True:
		min_idx = None
		min_rank = None
		for i, pair in enumerate(zip(parts[:-1], parts[1:])):
			rank = mergeable_ranks.get(pair[0] + pair[1])
			if rank is not None and (min_rank is None or rank < min_rank):
				min_idx = i
				min_rank = rank
		if min_rank is None or (max_rank is not None and min_rank >= max_rank):
			break
		assert min_idx is not None
		parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
	return parts


def load_tiktoken_ranks(path: Path) -> dict:
	ranks = {base64.b64decode(t): int(r) for t, r in
	         (line.split() for line in path.read_bytes().splitlines() if line)}
	if len(ranks) != 151643:
		raise ValueError(f"Expected 151643 tiktoken ranks, got {len(ranks)}")
	return ranks


def build_vocab(tiktoken_path: Path):
	"""Returns (tokens, toktypes, merges) sized VOCAB_SIZE, gguf.TokenType values."""
	import gguf

	mergeable_ranks = load_tiktoken_ranks(tiktoken_path)

	merges = []
	base_vocab = {}
	for token, rank in mergeable_ranks.items():
		base_vocab[token_bytes_to_string(token)] = rank
		if len(token) == 1:
			continue
		merged = bpe(mergeable_ranks, token, max_rank=rank)
		assert len(merged) == 2
		merges.append(" ".join(map(token_bytes_to_string, merged)))

	tokens = [None] * VOCAB_SIZE
	toktypes = [None] * VOCAB_SIZE

	reverse_base = {rank: tok for tok, rank in base_vocab.items()}
	for i in range(len(mergeable_ranks)):
		tokens[i] = reverse_base[i]
		toktypes[i] = gguf.TokenType.NORMAL

	for i, name in enumerate(SPECIALS):
		tokens[EOD + i] = name
		toktypes[EOD + i] = gguf.TokenType.CONTROL

	tokens[MUSIC_START] = "<music>"
	toktypes[MUSIC_START] = gguf.TokenType.CONTROL
	tokens[MUSIC_END] = "</music>"
	toktypes[MUSIC_END] = gguf.TokenType.CONTROL

	for c in range(CODEC_SIZE):
		i = CODEC_OFFSET + c
		tokens[i] = f"<codec_{c}>"
		toktypes[i] = gguf.TokenType.USER_DEFINED

	pad_start = CODEC_OFFSET + CODEC_SIZE  # 184621
	n_pad = VOCAB_SIZE - pad_start          # 83
	for p in range(n_pad):
		i = pad_start + p
		tokens[i] = f"<pad_{p}>"
		toktypes[i] = gguf.TokenType.UNUSED

	assert all(t is not None for t in tokens)
	return tokens, toktypes, merges


# ── AR tensor subset: safetensors name -> gguf name (SPEC_AR.md section 2) ──

def layer_tensor_map(n_layers: int) -> dict:
	m = {
		"model.embed_tokens.weight": "token_embd.weight",
		"model.norm.weight": "output_norm.weight",
		"lm_head.weight": "output.weight",
	}
	for n in range(n_layers):
		p = f"model.layers.{n}."
		g = f"blk.{n}."
		m[p + "input_layernorm.weight"] = g + "attn_norm.weight"
		m[p + "self_attn.q_proj.weight"] = g + "attn_q.weight"
		m[p + "self_attn.k_proj.weight"] = g + "attn_k.weight"
		m[p + "self_attn.v_proj.weight"] = g + "attn_v.weight"
		m[p + "self_attn.o_proj.weight"] = g + "attn_output.weight"
		m[p + "self_attn.q_norm.weight"] = g + "attn_q_norm.weight"
		m[p + "self_attn.k_norm.weight"] = g + "attn_k_norm.weight"
		m[p + "post_attention_layernorm.weight"] = g + "ffn_norm.weight"
		m[p + "mlp.gate_proj.weight"] = g + "ffn_gate.weight"
		m[p + "mlp.up_proj.weight"] = g + "ffn_up.weight"
		m[p + "mlp.down_proj.weight"] = g + "ffn_down.weight"
	return m


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=None,
	                 help="HF snapshot dir with model.safetensors + config.json + qwen.tiktoken "
	                      f"(default: resolve {REPO_ID!r} from the huggingface_hub cache)")
	ap.add_argument("--out", default=DEFAULT_OUT, help="output .gguf path")
	ap.add_argument("--type", choices=["f16", "bf16"], default="f16",
	                 help="dtype for 2D+ weight matrices; 1D norm weights always stay F32")
	args = ap.parse_args()

	import gguf

	src = resolve_snapshot(args.src, REPO_ID)
	st_path = src / "model.safetensors"
	cfg_path = src / "config.json"
	tok_path = src / "qwen.tiktoken"
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	config = json.loads(cfg_path.read_text())
	n_layers = config["num_hidden_layers"]
	assert n_layers == 28
	assert config["hidden_size"] == 2048
	assert config["num_attention_heads"] == 16
	assert config["num_key_value_heads"] == 8
	assert config["head_dim"] == 128
	assert config["intermediate_size"] == 6144
	assert config["vocab_size"] == VOCAB_SIZE
	assert config["rms_norm_eps"] == 1e-6
	assert config["rope_theta"] == 1000000
	assert config["max_position_embeddings"] == 24576

	source_sha256 = sha256_of(st_path)
	print(f"[convert_ar] source: {st_path}")
	print(f"[convert_ar] source sha256: {source_sha256}")

	print(f"[convert_ar] building vocab from {tok_path}")
	tokens, toktypes, merges = build_vocab(tok_path)
	print(f"[convert_ar] vocab: {len(tokens)} tokens, {len(merges)} merges")

	writer = gguf.GGUFWriter(str(out_path), "qwen3")  # sets general.architecture
	writer.add_name("YuE2-3B AR")

	writer.add_context_length(24576)
	writer.add_embedding_length(2048)
	writer.add_block_count(n_layers)
	writer.add_feed_forward_length(6144)
	writer.add_head_count(16)
	writer.add_head_count_kv(8)
	writer.add_key_length(128)
	writer.add_value_length(128)
	writer.add_layer_norm_rms_eps(1e-6)
	writer.add_rope_freq_base(1000000.0)
	writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.type == "f16" else gguf.LlamaFileType.MOSTLY_BF16)
	writer.add_key_value("yue2.source_sha256", source_sha256, gguf.GGUFValueType.STRING)

	writer.add_tokenizer_model("gpt2")
	writer.add_tokenizer_pre("qwen2")
	writer.add_token_list(tokens)
	writer.add_token_types(toktypes)
	writer.add_token_merges(merges)
	writer.add_bos_token_id(EOD)
	writer.add_eos_token_id(EOD)
	writer.add_add_bos_token(False)

	tmap = layer_tensor_map(n_layers)
	table = []
	qtype = gguf.GGMLQuantizationType.F16 if args.type == "f16" else gguf.GGMLQuantizationType.BF16

	with safe_open(st_path, framework="pt", device="cpu") as f:
		keys = set(f.keys())
		missing = [k for k in tmap if k not in keys]
		if missing:
			raise ValueError(f"Missing expected tensors in safetensors: {missing[:5]}")
		for src_name, gguf_name in tmap.items():
			t = f.get_tensor(src_name).to(torch.float32).numpy()
			t = np.ascontiguousarray(t)
			if t.ndim <= 1 or gguf_name.endswith("_norm.weight"):
				out = t.astype(np.float32)
				writer.add_tensor(gguf_name, out)
				table.append((gguf_name, list(out.shape), "F32"))
			else:
				out = gguf.quants.quantize(t, qtype)
				writer.add_tensor(gguf_name, out, raw_dtype=qtype)
				table.append((gguf_name, list(t.shape), qtype.name))

	print(f"[convert_ar] {len(tmap)} tensors mapped (of {len(keys)} in safetensors; "
	      f"{len(keys) - len(tmap)} skipped: nar_*/vae2llm/llm2vae/time_embedder/latent_pos_embed)")

	writer.write_header_to_file()
	writer.write_kv_data_to_file()
	writer.write_tensors_to_file(progress=False)
	writer.close()

	print(f"\n[convert_ar] wrote {out_path} ({out_path.stat().st_size} bytes)\n")
	print_tensor_table(table)

	return 0


if __name__ == "__main__":
	sys.exit(main())
