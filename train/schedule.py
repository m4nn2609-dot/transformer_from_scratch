"""Cosine learning-rate schedule with linear warmup.

lr(t) = lr_max * 0.5 * (1 + cos(pi * (t - warmup) / (max_steps - warmup)))
for t >= warmup, linearly ramped from 0 to lr_max during warmup.
"""

import math


class CosineWithWarmup:
    def __init__(self, optimizer, lr_max, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_count = 0

    def get_lr(self):
        t = self.step_count
        if t < self.warmup_steps:
            return self.lr_max * (t / max(1, self.warmup_steps))
        progress = (t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
