"""Test custom TransformerBlock against a manual pre-norm block reference."""

import math
import torch

from model.attention import MultiHeadAttention, scaled_dot_product_attention
from model.block import TransformerBlock
from model.layers import RMSNorm, SwiGLU
from model.rope import precompute_freqs_cis


def test_block_matches_manual():
    torch.manual_seed(10)
    batch, seq, d_model, n_heads, d_ff = 1, 4, 8, 2, 12
    x = torch.randn(batch, seq, d_model)

    custom = TransformerBlock(d_model, n_heads, d_ff)
    freqs_cis = precompute_freqs_cis(custom.attn.head_dim, seq)

    # Manual reference using same weights / submodules
    h1 = custom.norm1(x)
    attn_out = custom.attn(h1, freqs_cis=freqs_cis)
    h2 = x + attn_out
    h3 = custom.norm2(h2)
    ffn_out = custom.ffn(h3)
    out_ref = h2 + ffn_out

    out_custom = custom(x, freqs_cis=freqs_cis)

    assert out_custom.shape == (batch, seq, d_model)
    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
