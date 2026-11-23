"""Ablations: RoPE vs none, RMSNorm vs LayerNorm, SwiGLU vs GeLU-MLP.
Runs a short training loop per variant and reports final loss.
"""

import os, sys, time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (_REPO_ROOT, os.path.join(_REPO_ROOT, "model"), os.path.join(_REPO_ROOT, "tokenizer"), os.path.join(_REPO_ROOT, "train")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import RMSNorm, SwiGLU, Linear
from cross_entropy import cross_entropy
from adamw import AdamW
from loader import MemmapDataset


class GeLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class LN(nn.Module):
    def __init__(self, d, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d, eps=eps, device=device, dtype=dtype)
    def forward(self, x):
        return self.norm(x)


class NoRopeBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, norm_cls):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = SwiGLU(d_model, d_ff)
        self.n1 = norm_cls(d_model)
        self.n2 = norm_cls(d_model)
    def forward(self, x, freqs_cis=None):
        h = self.n1(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + out
        x = x + self.ffn(self.n2(x))
        return x


class AblLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, ff, rope=True, norm="rms", ffn="swiglu"):
        super().__init__()
        self.rope = rope
        self.emb = nn.Embedding(vocab, d)
        NC = RMSNorm if norm == "rms" else LN
        FC = SwiGLU if ffn == "swiglu" else GeLUMLP
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if rope:
                from block import TransformerBlock
                self.layers.append(TransformerBlock(d, heads, ff, norm_cls=NC))
            else:
                self.layers.append(NoRopeBlock(d, heads, ff, NC))
        self.norm = NC(d)
        self.head = Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
    def forward(self, x):
        h = self.emb(x)
        from rope import precompute_freqs_cis
        fc = precompute_freqs_cis(self.layers[0].attn.head_dim if self.rope else 8, x.shape[1], device=x.device)
        for layer in self.layers:
            h = layer(h, freqs_cis=fc if self.rope else None)
        h = self.norm(h)
        return self.head(h)


def run(name, memmap, steps=300, batch=8, seq=128, device="cpu"):
    # Resolve device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    device = torch.device(device)

    ds = MemmapDataset(memmap, seq)
    cfg = _cfg(name)
    m = AblLM(**cfg).to(device)
    opt = AdamW(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = ds.get_batch(batch, device, rng=rng)
        opt.zero_grad()
        loss = cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    print("%-26s final_loss=%.4f  %.1fs" % (name, loss.item(), dt))


def _cfg(name):
    b = dict(vocab=10000, d=256, layers=2, heads=8, ff=512)
    if name == "base (rope+rms+swiglu)": return dict(**b, rope=True, norm="rms", ffn="swiglu")
    if name == "no-rope": return dict(**b, rope=False, norm="rms", ffn="swiglu")
    if name == "layer-norm": return dict(**b, rope=True, norm="ln", ffn="swiglu")
    if name == "gelu-mlp": return dict(**b, rope=True, norm="rms", ffn="gelu")
    if name == "no-rope+ln+gelu": return dict(**b, rope=False, norm="ln", ffn="gelu")
    raise ValueError(name)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    print("Ablations (steps=%d, device=%s)" % (a.steps, a.device))
    print("-" * 50)
    for n in ["base (rope+rms+swiglu)", "no-rope", "layer-norm", "gelu-mlp", "no-rope+ln+gelu"]:
        run(n, a.memmap, steps=a.steps, device=a.device)
    print("-" * 50)
    print("Done. Compare final_loss across variants.")


if __name__ == "__main__":
    main()
