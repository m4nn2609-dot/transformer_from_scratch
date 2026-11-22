"""Pre-norm Transformer block.

x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
"""

import os
import sys

import torch
import torch.nn as nn

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from attention import MultiHeadAttention
from layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, norm_cls=None, eps=1e-5,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if norm_cls is None:
            norm_cls = RMSNorm
        self.attn = MultiHeadAttention(d_model, n_heads, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
        self.norm1 = norm_cls(d_model, eps=eps, **factory_kwargs)
        self.norm2 = norm_cls(d_model, eps=eps, **factory_kwargs)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x
