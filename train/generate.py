"""Text generation with temperature, top-p, and optional KV cache.

Usage:
    python train/generate.py --checkpoint checkpoints/final.pt \
        --vocab-size 10000 --d-model 512 --n-layers 4 --n-heads 16 \
        --d-ff 2048 --device cuda --max-new-tokens 100 \
        --temperature 0.8 --top-p 0.9 --prompt "Once upon a time"
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from model.lm import LanguageModel
from train.checkpoint import load
from tokenizer.bpe_codec import decode, encode


def load_model(checkpoint_path, vocab_size, d_model, n_layers, n_heads, d_ff, device):
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        device=device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # dummy for load
    from train.schedule import CosineWithWarmup
    scheduler = CosineWithWarmup(optimizer, 1e-3, 1, 1)
    load(model, optimizer, scheduler, checkpoint_path)
    model.eval()
    return model


def sample_next(logits, temperature=1.0, top_p=None, rng=None):
    """Sample the next token from logits with temperature and top-p."""
    if rng is None:
        rng = torch.Generator(device=logits.device)
        rng.seed()

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above top_p.
        sorted_indices_to_remove = cumulative > top_p
        # Keep the first token above the threshold (shift by 1).
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # Unmap sorted removal mask back to original positions.
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove[sorted_indices] = sorted_indices_to_remove
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1, generator=rng)
    return next_token


def generate(model, tokenizer_vocab, tokenizer_merges, special_tokens,
             prompt, max_new_tokens=50, temperature=1.0, top_p=None,
             device="cuda", use_kv_cache=False):
    """Generate text autoregressively."""
    ids = encode(prompt, tokenizer_vocab, tokenizer_merges, special_tokens)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = list(ids)

    if use_kv_cache:
        return _generate_with_kv_cache(
            model, input_ids, generated, max_new_tokens, temperature, top_p,
            device
        )
    else:
        return _generate_vanilla(
            model, input_ids, generated, max_new_tokens, temperature, top_p,
            device
        )


def _generate_vanilla(model, input_ids, generated, max_new_tokens, temperature,
                      top_p, device):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[0, -1, :]
        next_token = sample_next(next_logits, temperature, top_p)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return generated


def _generate_with_kv_cache(model, input_ids, generated, max_new_tokens,
                            temperature, top_p, device):
    """Generation with a per-layer KV cache to avoid recomputing past keys/values."""
    batch, seq_len = input_ids.shape
    n_layers = len(model.layers)
    n_heads = model.layers[0].attn.n_heads
    head_dim = model.layers[0].attn.head_dim
    d_model = model.d_model

    # Allocate cache: (batch, n_heads, max_seq, head_dim)
    max_seq = seq_len + max_new_tokens
    k_cache = torch.zeros(batch, n_heads, max_seq, head_dim, device=device)
    v_cache = torch.zeros(batch, n_heads, max_seq, head_dim, device=device)
    cache_pos = 0  # number of tokens currently cached

    # Precompute RoPE freqs for the full max length.
    from model.rope import precompute_freqs_cis
    freqs_cis = precompute_freqs_cis(head_dim, max_seq, device=device)

    # Process the prompt through the cache.
    with torch.no_grad():
        h = model.tok_emb(input_ids)
        for layer in model.layers:
            # Attention with KV cache.
            attn = layer.attn
            norm1_out = layer.norm1(h)
            q = attn.w_q(norm1_out).view(batch, seq_len, n_heads, head_dim)
            k = attn.w_k(norm1_out).view(batch, seq_len, n_heads, head_dim)
            v = attn.w_v(norm1_out).view(batch, seq_len, n_heads, head_dim)
            q = apply_rope(q, freqs_cis[:seq_len])
            k = apply_rope(k, freqs_cis[:seq_len])
            k_cache[:, :, :seq_len, :] = k.transpose(1, 2)
            v_cache[:, :, :seq_len, :] = v.transpose(1, 2)
            cache_pos = seq_len

            # Attention over cached K,V.
            q_t = q.transpose(1, 2)
            k_t = k_cache[:, :, :cache_pos, :]
            v_t = v_cache[:, :, :cache_pos, :]
            scores = q_t @ k_t.transpose(-2, -1) / (head_dim ** 0.5)
            # Causal mask.
            mask = torch.tril(torch.ones(cache_pos, cache_pos, device=device, dtype=torch.bool))
            # q is (batch, n_heads, seq_len, head_dim) but we only attend to cache_pos keys.
            # For generation, we only need the last query position against all cached keys.
            # Simplify: full attention over cached prefix.
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = attn_weights @ v_t
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
            h = h + attn.w_o(attn_out)
            h = h + layer.ffn(layer.norm2(h))

        # Now generate token by token.
        for step_idx in range(max_new_tokens):
            last_input = input_ids[:, -1:]
            with torch.no_grad():
                h = model.tok_emb(last_input)
                pos = seq_len + step_idx  # RoPE position for this token
                max_pos = freqs_cis.shape[0] - 1
                safe_pos = min(pos, max_pos)
                for layer in model.layers:
                    attn = layer.attn
                    norm1_out = layer.norm1(h)
                    q = attn.w_q(norm1_out).view(batch, 1, n_heads, head_dim)
                    k = attn.w_k(norm1_out).view(batch, 1, n_heads, head_dim)
                    v = attn.w_v(norm1_out).view(batch, 1, n_heads, head_dim)
                    q = apply_rope(q, freqs_cis[safe_pos:safe_pos + 1])
                    k = apply_rope(k, freqs_cis[safe_pos:safe_pos + 1])
                    k_cache[:, :, safe_pos, :] = k.transpose(1, 2).squeeze(2)
                    v_cache[:, :, safe_pos, :] = v.transpose(1, 2).squeeze(2)

                    q_t = q.transpose(1, 2)  # (batch, n_heads, 1, head_dim)
                    k_t = k_cache[:, :, :safe_pos + 1, :]  # all cached keys up to current
                    v_t = v_cache[:, :, :safe_pos + 1, :]
                    scores = q_t @ k_t.transpose(-2, -1) / (head_dim ** 0.5)
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_out = attn_weights @ v_t
                    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, 1, d_model)
                    h = h + attn.w_o(attn_out)
                    h = h + layer.ffn(layer.norm2(h))

                logits = model.lm_head(model.norm(h))
                next_logits = logits[0, -1, :]
            next_token = sample_next(next_logits, temperature, top_p)
            next_token_t = next_token.unsqueeze(0)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token_t], dim=1)

    return generated


def apply_rope(x, freqs_cis):
    from model.rope import apply_rope as _apply_rope
    return _apply_rope(x, freqs_cis)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-kv-cache", action="store_true")
    args = p.parse_args()

    # Load tokenizer vocab/merges.
    tok_dir = os.path.join(_REPO_ROOT, "data", "tokenizer")
    vocab_path = os.path.join(tok_dir, "vocab.txt")
    merges_path = os.path.join(tok_dir, "merges.txt")

    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            i, h = line.strip().split("\t")
            vocab[int(i)] = bytes.fromhex(h)
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split("\t")
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    special_tokens = ["<|endoftext|>"]

    model = load_model(
        args.checkpoint, args.vocab_size, args.d_model, args.n_layers,
        args.n_heads, args.d_ff, args.device
    )

    t0 = time.perf_counter()
    generated = generate(
        model, vocab, merges, special_tokens,
        args.prompt, args.max_new_tokens, args.temperature, args.top_p,
        args.device, args.use_kv_cache
    )
    dt = time.perf_counter() - t0

    text = decode(generated, vocab)
    print(f"\nGenerated ({'KV cache' if args.use_kv_cache else 'vanilla'}):")
    print(text)
    print(f"Time: {dt:.2f}s, {args.max_new_tokens / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
