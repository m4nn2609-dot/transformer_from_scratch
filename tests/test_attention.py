"""Test custom scaled_dot_product_attention against a manual reference."""

import math
import torch

from model.attention import scaled_dot_product_attention


def test_sdpa_matches_manual_causal():
    torch.manual_seed(7)
    batch, n_heads, seq, head_dim = 2, 3, 6, 4
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)

    out_custom = scaled_dot_product_attention(q, k, v, causal=True)

    d_k = head_dim
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out_ref = attn @ v

    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )


def test_sdpa_matches_manual_noncausal():
    torch.manual_seed(8)
    batch, n_heads, seq, head_dim = 2, 3, 6, 4
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    v = torch.randn(batch, n_heads, seq, head_dim)

    out_custom = scaled_dot_product_attention(q, k, v, causal=False)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    out_ref = attn @ v

    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5)
