"""Systems benchmarks: bf16 autocast, torch.compile, Triton FlashAttention-2.
Reports tokens/sec and GPU memory per configuration.
"""

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.lm import LanguageModel
from train.cross_entropy import cross_entropy
from train.adamw import AdamW


def make_model(vocab_size, d_model, n_layers, n_heads, d_ff, device, dtype):
    m = LanguageModel(vocab_size, d_model, n_layers, n_heads, d_ff, device=device, dtype=dtype)
    m.train()
    return m


def make_batch(batch, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    return x, y


def time_fn(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def memory_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size, d_model, n_layers, n_heads, d_ff = 10000, 512, 4, 16, 2048
    batch, seq_len = 8, 256
    tokens_per_batch = batch * seq_len

    print("Device:", device)
    print("Model: d_model=%d layers=%d heads=%d d_ff=%d vocab=%d" % (vocab_size, n_layers, n_heads, d_ff, vocab_size))
    print("Batch=%d seq_len=%d" % (batch, seq_len))
    print()

    x, y = make_batch(batch, seq_len, vocab_size, device)

    configs = [
        ("fp32 eager", False, False, torch.float32, False),
        ("bf16 autocast", True, False, torch.bfloat16, False),
    ]
    if torch.cuda.is_available():
        try:
            import triton
            configs.append(("fp32 + Triton FlashAttn-2", False, False, torch.float32, True))
        except ImportError:
            print("(Triton not installed; skipping FlashAttention-2)")
        try:
            import torch._dynamo
            configs.append(("fp32 + torch.compile", False, True, torch.float32, False))
        except Exception:
            print("(torch.compile not available; skipping)")

    for name, use_autocast, use_compile, dtype, flash_attn in configs:
        model = make_model(vocab_size, d_model, n_layers, n_heads, d_ff, device, dtype)
        opt = AdamW(model.parameters(), lr=1e-3)

        f = None
        if use_compile:
            try:
                cf = torch.compile(model.forward, mode="reduce-overhead")
            except Exception as e:
                print(f"  ({name} failed to compile: {e})")
                continue

            def f():
                with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
                    logits = cf(x)
                    loss = cross_entropy(logits, y)
                loss.backward()
                opt.zero_grad(set_to_none=True)
        elif flash_attn:
            from train.flash_attn_triton import flash_attn_forward, HAS_TRITON
            if not HAS_TRITON:
                continue
            import train.attention as attn_mod
            attn_mod.scaled_dot_product_attention = lambda q, k, v, causal=False: flash_attn_forward(q, k, v, causal)

            def f():
                with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
                    logits = model(x)
                    loss = cross_entropy(logits, y)
                loss.backward()
                opt.zero_grad(set_to_none=True)
        else:
            def f():
                with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
                    logits = model(x)
                    loss = cross_entropy(logits, y)
                loss.backward()
                opt.zero_grad(set_to_none=True)

        try:
            sec = time_fn(f)
        except Exception as e:
            print(f"  ({name} failed at runtime: {e})")
            continue
        mem = memory_mb()
        print("%-32s %7.2f ms/iter  %8.0f tok/s  mem %7.1f MB" % (name, sec * 1000, tokens_per_batch / sec, mem))
    print("\nBaseline: initial loss ~ ln(10000) = 9.21")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
