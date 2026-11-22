"""Test custom RoPE against a manual rotation reference."""

import torch

from model.rope import apply_rope, precompute_freqs_cis


def test_rope_matches_manual():
    torch.manual_seed(4)
    batch, seq, n_heads, head_dim = 2, 7, 3, 8
    x = torch.randn(batch, seq, n_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq)

    out_custom = apply_rope(x, freqs_cis)

    # Manual: x1 = even dims, x2 = odd dims; rotate by pos * theta_i
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2) / head_dim)
    )
    t = torch.arange(seq)
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim//2)
    cos = freqs.cos().view(1, seq, 1, head_dim // 2)
    sin = freqs.sin().view(1, seq, 1, head_dim // 2)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out_ref = torch.empty_like(x)
    out_ref[..., 0::2] = x1 * cos - x2 * sin
    out_ref[..., 1::2] = x1 * sin + x2 * cos

    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
