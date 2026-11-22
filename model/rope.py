"""Rotary Positional Embeddings (RoPE).

Complex-number implementation: pairs of dimensions are treated as real/imag
parts and rotated by position-dependent angles.
"""

import torch


def precompute_freqs_cis(head_dim, seq_len, base=10000, device=None):
    """Return complex cis tensor of shape (seq_len, head_dim // 2).

    freqs_cis[pos, i] = exp(i * pos * theta_i),  theta_i = base^(-2i / head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    """Apply RoPE to x.

    Args:
        x: (batch, seq_len, n_heads, head_dim)  float
        freqs_cis: (seq_len, head_dim // 2)  complex64

    Returns:
        same shape as x, same dtype as x
    """
    batch, seq_len, n_heads, head_dim = x.shape
    # View pairs of dims as complex: (batch, seq_len, n_heads, head_dim // 2)
    x_complex = torch.view_as_complex(
        x.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2)
    )
    # freqs_cis: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    # broadcast over batch/heads. Use None-indexing (not .view) because
    # .view on a complex tensor counts real storage elements.
    freqs_cis = freqs_cis[None, :, None, :]
    x_rotated = x_complex * freqs_cis
    out = torch.view_as_real(x_rotated).reshape(batch, seq_len, n_heads, head_dim)
    return out.type_as(x)
