"""Main training loop for the language model.

Usage:
    python train/training_loop.py --memmap data/tokenizer/tinystories_uint16.bin \
        --vocab-size 10000 --d-model 512 --n-layers 4 --n-heads 16 \
        --d-ff 2048 --batch-size 16 --seq-len 256 --max-steps 5000 \
        --device cuda --save-dir checkpoints/

Device-agnostic: --device cuda (if available) or cpu.
"""

import argparse
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (_REPO_ROOT, os.path.join(_REPO_ROOT, "model"), os.path.join(_REPO_ROOT, "tokenizer")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
if os.path.join(_REPO_ROOT, "train") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))

import torch
import numpy as np

from lm import LanguageModel
from cross_entropy import cross_entropy
from adamw import AdamW
from schedule import CosineWithWarmup
from loader import MemmapDataset
from checkpoint import save, load


def gradient_clipping(parameters, max_norm):
    """Clip gradient norm of a list of parameters in place."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm


def train(args):
    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Model -----------------------------------------------------------
    model = LanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=torch.float32,
    )
    model.train()

    # ---- Optimizer + schedule -------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineWithWarmup(optimizer, args.lr_max, args.warmup_steps, args.max_steps)

    # ---- Data ------------------------------------------------------------
    dataset = MemmapDataset(args.memmap, args.seq_len)
    rng = np.random.default_rng(args.seed)

    # ---- Resume ----------------------------------------------------------
    start_step = 0
    best_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_step, best_loss = load(model, optimizer, scheduler, args.resume)
        print(f"Resumed from step {start_step}, best_loss={best_loss:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Training loop ---------------------------------------------------
    step = start_step
    t0 = time.perf_counter()
    tokens_seen = 0
    while step < args.max_steps:
        scheduler.step()
        lr = scheduler.get_lr()

        x, y = dataset.get_batch(args.batch_size, device, rng=rng)
        tokens_seen += x.numel()

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        grad_norm = gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1

        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            toks_per_sec = tokens_seen / dt if dt > 0 else 0
            print(
                f"step {step:6d}/{args.max_steps}  "
                f"loss {loss.item():.4f}  "
                f"lr {lr:.2e}  "
                f"grad_norm {grad_norm:.2f}  "
                f"{toks_per_sec:.0f} tok/s"
            )
            tokens_seen = 0
            t0 = time.perf_counter()

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{step}.pt")
            save(model, optimizer, scheduler, step, best_loss, ckpt_path)
            print(f"  checkpoint saved -> {ckpt_path}")

    # Final checkpoint.
    final_path = os.path.join(args.save_dir, "final.pt")
    save(model, optimizer, scheduler, step, best_loss, final_path)
    print(f"Training complete. Final checkpoint -> {final_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", default="checkpoints/")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    train(args)
