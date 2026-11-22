"""Full decoder-only language model.

Embedding -> N x TransformerBlock -> RMSNorm -> Linear (tied to embedding).
"""

import os
import sys

import torch
import torch.nn as nn

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from block import TransformerBlock
from layers import Embedding, Linear, RMSNorm
from rope import precompute_freqs_cis


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tok_emb = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, eps=eps, **factory_kwargs)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=eps, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        # Tie output projection weights to the input embedding.
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x, freqs_cis=None):
        h = self.tok_emb(x)
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis(
                self.layers[0].attn.head_dim, x.shape[1], device=x.device
            )
        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
