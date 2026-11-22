"""Checkpoint save / resume.

Saves model state, optimizer state, scheduler step count, and training meta
(step, best loss) to a single file.
"""

import os
import torch


def save(model, optimizer, scheduler, step, best_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler.step_count,
        "step": step,
        "best_loss": best_loss,
    }
    torch.save(ckpt, path)


def load(model, optimizer, scheduler, path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without weights_only support.
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.step_count = ckpt["scheduler_step"]
    # Re-sync optimizer lr to current schedule value without advancing step.
    for group in optimizer.param_groups:
        group["lr"] = scheduler.get_lr()
    step = ckpt["step"]
    best_loss = ckpt["best_loss"]
    return step, best_loss
