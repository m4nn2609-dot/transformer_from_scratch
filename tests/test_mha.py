"""Test custom MultiHeadAttention against a manual per-head reference."""

import math
import torch

from model.attention import MultiHeadAttention


def test_mha_matches_manual():
    torch.manual_seed(9)
    batch, seq, d_model, n_heads = 2, 5, 8, 2
    head_dim = d_model // n_heads
    x = torch.randn(batch, seq, d_model)

    custom = MultiHeadAttention(d_model, n_heads)

    # Manual reference using the same weights
    q = custom.w_q(x).view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = custom.w_k(x).view(batch, seq, n_heads, head_dim).transpose(1, 2)
    v = custom.w_v(x).view(batch, seq, n_heads, head_dim).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
    out_ref = custom.w_o(out)

    out_custom = custom(x)

    assert out_custom.shape == (batch, seq, d_model)
    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
