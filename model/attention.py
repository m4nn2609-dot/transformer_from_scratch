"""Scaled dot-product attention with causal mask and multi-head attention."""

import math
import os
import sys

import torch
import torch.nn as nn

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from layers import Linear
from rope import apply_rope


def softmax(x, dim=-1):
    """Numerically stable softmax along dim."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, causal=False):
    """Scaled dot-product attention.

    Args:
        q, k, v: (batch, n_heads, seq_len, head_dim)
        causal: apply causal (lower-triangular) mask

    Returns:
        (batch, n_heads, seq_len, head_dim)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if causal:
        seq_len = q.shape[2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, device=None, dtype=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w_q = Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.w_k = Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.w_v = Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.w_o = Linear(d_model, d_model, bias=False, **factory_kwargs)

    def forward(self, x, freqs_cis=None):
        batch, seq_len, _ = x.shape
        # (batch, seq_len, n_heads, head_dim) — RoPE expects this layout
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.head_dim)
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
        # -> (batch, n_heads, seq_len, head_dim) for scaled dot-product
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v, causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.w_o(out)
