"""Distributed training: DDP with gradient bucketing + sharded optimizer.

Correctness on CPU with gloo; benchmarks on a rented 2-GPU machine.
Communication is explicit so the engineering is transparent.
"""

import os, sys, math

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist


def setup(backend="gloo"):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    # Set defaults for single-process / local runs
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, world, local


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_bucketed(grads, bucket_mb=25):
    world = dist.get_world_size()
    if world == 1:
        return
    bucket = []
    bbytes = 0
    target = bucket_mb * 1024 * 1024

    def flush():
        nonlocal bucket, bbytes
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        off = 0
        for g in bucket:
            n = g.numel()
            g.copy_(flat[off:off + n].view_as(g))
            off += n
        bucket = []
        bbytes = 0

    for g in grads:
        if g is None:
            continue
        gb = g.numel() * g.element_size()
        if gb > target:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(world)
        elif bbytes + gb > target and bucket:
            flush()
            bucket = [g]
            bbytes = gb
        else:
            bucket.append(g)
            bbytes += gb
    flush()


class ShardedAdamW(torch.optim.Optimizer):
    """ZeRO-1 style sharded AdamW: each rank optimizes 1/world_size of params."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        plist = list(params)
        self.param_numels = [p.numel() for p in plist]
        total = sum(self.param_numels)
        shard = (total + self.world - 1) // self.world
        self.start = self.rank * shard
        self.end = min(total, self.start + shard)
        self.snumel = self.end - self.start

        flat = torch.zeros(total, dtype=plist[0].dtype, device=plist[0].device)
        off = 0
        for p in plist:
            flat[off:off + p.numel()] = p.data.reshape(-1)
            off += p.numel()
        if dist.is_initialized():
            dist.broadcast(flat, src=0)
        super().__init__([flat], defaults)
        self.flat = flat
        self.state[flat]["step"] = 0
        self.state[flat]["m"] = torch.zeros(self.snumel, dtype=flat.dtype, device=flat.device)
        self.state[flat]["v"] = torch.zeros(self.snumel, dtype=flat.dtype, device=flat.device)
        self.original_params = plist

    def set_original_params(self, params):
        self.original_params = list(params)

    def step(self, closure=None):
        loss = None
        if closure:
            loss = closure()
        world = self.world
        flat = self.param_groups[0]["params"][0]
        st = self.state[flat]
        st["step"] += 1
        t = st["step"]
        b1, b2 = self.param_groups[0]["betas"]
        lr = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]
        wd = self.param_groups[0]["weight_decay"]

        g = torch.cat([p.grad.data.reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
                       for p in self.original_params])
        if dist.is_initialized():
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(world)

        gs = g[self.start:self.end]
        m = st["m"]; v = st["v"]
        m.mul_(b1).add_(gs, alpha=1 - b1)
        v.mul_(b2).addcmul_(gs, gs, value=1 - b2)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        upd = mh / (vh.sqrt() + eps)
        if wd > 0:
            upd = upd + wd * flat.data[self.start:self.end]
        flat.data[self.start:self.end].add_(upd, alpha=-lr)

        if dist.is_initialized():
            dist.broadcast(flat.data, src=0)
        off = 0
        for p in self.original_params:
            n = p.numel()
            p.data.copy_(flat.data[off:off + n].view_as(p))
            off += n
        return loss


def ddp_step(model, optimizer, x, y, criterion, grad_clip=1.0, bucketed=True):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if bucketed:
        all_reduce_bucketed(grads)
    else:
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(dist.get_world_size())
    tn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tn += p.grad.data.norm(2).item() ** 2
    tn = math.sqrt(tn)
    if tn > grad_clip:
        cf = grad_clip / (tn + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(cf)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    """Quick DDP correctness demo: single-process, CPU, gloo backend."""
    import torch
    import torch.nn as nn

    rank, world, local = setup(backend="gloo")
    print(f"Distributed setup: rank={rank}, world_size={world}, local_rank={local}")

    # Tiny model for the demo
    model = nn.Linear(16, 16)
    criterion = nn.MSELoss()
    optimizer = ShardedAdamW(model.parameters(), lr=1e-3)

    # Single training step
    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    loss = ddp_step(model, optimizer, x, y, criterion, grad_clip=1.0, bucketed=True)
    print(f"DDP step loss: {loss:.4f}")
    print(f"Optimizer step complete (sharded AdamW, gradient bucketing)")

    cleanup()
    print("Distributed demo complete.")
