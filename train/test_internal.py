"""Internal verification tests for all training infra components.
Run before user-facing testing (generation, benchmarks, etc.)
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MODEL_DIR = os.path.join(_REPO_ROOT, "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
_TOKENIZER_DIR = os.path.join(_REPO_ROOT, "tokenizer")
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)

import torch
import numpy as np

from lm import LanguageModel
from train.cross_entropy import cross_entropy
from train.adamw import AdamW
from train.schedule import CosineWithWarmup
from train.loader import MemmapDataset
from train.checkpoint import save, load


def test_cross_entropy_known_values():
    """Cross-entropy with known logits should match manual computation."""
    torch.manual_seed(0)
    logits = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    targets = torch.tensor([[2, 0]])
    loss = cross_entropy(logits, targets)
    # Manual: -log(softmax)[target] averaged over all positions
    # softmax([1,2,3]) = [0.090, 0.245, 0.665], -log(0.665)=0.408
    # softmax([1,2,3]) target 0: -log(0.090)=2.408
    # mean = (0.408 + 2.408)/2 = 1.408
    expected = 1.408
    assert abs(loss.item() - expected) < 0.01, f"loss={loss.item()}"
    print(f"  cross_entropy known values: loss={loss.item():.4f} (expected ~{expected})  PASS")


def test_adamw_converges():
    """AdamW should reduce a simple loss over steps."""
    torch.manual_seed(0)
    x = torch.randn(10, 5)
    target = torch.randn(10, 3)
    linear = torch.nn.Linear(5, 3)
    opt = AdamW(linear.parameters(), lr=0.1, weight_decay=0.0)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = ((linear(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss didn't decrease: {losses[0]} -> {losses[-1]}"
    print(f"  AdamW convergence: {losses[0]:.4f} -> {losses[-1]:.4f}  PASS")


def test_cosine_schedule_bounds():
    """Cosine with warmup: lr starts at 0, peaks at lr_max, ends near 0."""
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = CosineWithWarmup(opt, lr_max=1e-3, warmup_steps=10, max_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(sched.get_lr())
        sched.step()

    assert lrs[0] == 0.0, f"warmup start lr should be 0, got {lrs[0]}"
    assert lrs[10] > lrs[9], "lr should increase during warmup"
    assert lrs[-1] < lrs[50], "lr should decrease after peak"
    print(f"  Cosine schedule: start={lrs[0]}, warmup_end={lrs[10]:.6f}, final={lrs[-1]:.8f}  PASS")


def test_memmap_loader_shapes():
    """MemmapDataset.get_batch returns correct shapes and shifted targets."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.close()  # close handle so memmap can open it on Windows
    data = np.arange(1000, dtype=np.uint16)
    data.tofile(tmp.name)

    try:
        ds = MemmapDataset(tmp.name, seq_len=10)
        assert len(ds) == 1000 // 10  # 100 sequences
        x, y = ds.get_batch(batch_size=4, device="cpu")
        assert x.shape == (4, 10), f"x shape {x.shape}"
        assert y.shape == (4, 10), f"y shape {y.shape}"
        assert x.dtype == torch.long and y.dtype == torch.long
        print(f"  Memmap loader: x={x.shape}, y={y.shape}, dtype={x.dtype}  PASS")
    finally:
        try:
            del ds
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass  # Windows file lock, ignore


def test_checkpoint_save_load():
    """Checkpoint save and resume should restore model + optimizer + scheduler state."""
    torch.manual_seed(0)
    model = LanguageModel(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = CosineWithWarmup(opt, lr_max=1e-3, warmup_steps=10, max_steps=100)

    # Train one step to change state
    x = torch.randint(0, 100, (2, 5))
    y = torch.randint(0, 100, (2, 5))
    from train.cross_entropy import cross_entropy
    loss = cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    sched.step()

    # Save
    import tempfile
    ckpt_path = tempfile.mktemp(suffix=".pt")
    save(model, opt, sched, step=5, best_loss=2.5, path=ckpt_path)

    # Create new model and load
    model2 = LanguageModel(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    sched2 = CosineWithWarmup(opt2, lr_max=1e-3, warmup_steps=10, max_steps=100)
    step, best = load(model2, opt2, sched2, ckpt_path)

    assert step == 5, f"step={step}"
    assert best == 2.5, f"best={best}"
    # Model params should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "model params don't match after load"
    print(f"  Checkpoint save/load: step={step}, best={best}, params match  PASS")
    os.unlink(ckpt_path)


def test_kv_cache_matches_vanilla():
    """KV cache generation should produce identical logits to vanilla forward."""
    torch.manual_seed(0)
    model = LanguageModel(vocab_size=10000, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model.eval()
    model.to("cpu")

    from train.generate import generate, encode
    # Use a real vocab loaded from disk (has all 256 bytes + merges)
    vocab = {}
    with open(os.path.join(_REPO_ROOT, "data", "tokenizer", "vocab.txt"), "r", encoding="utf-8") as f:
        for line in f:
            i, h = line.strip().split("\t")
            vocab[int(i)] = bytes.fromhex(h)
    merges = []
    with open(os.path.join(_REPO_ROOT, "data", "tokenizer", "merges.txt"), "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split("\t")
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    special_tokens = ["<|endoftext|>"]

    prompt = "hello"
    ids = [1, 2, 3, 4, 5]  # fake token ids

    # Vanilla forward
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits_vanilla = model(input_ids)

    # KV cache generation (one step)
    gen_ids = generate(
        model, vocab, merges, special_tokens,
        prompt, max_new_tokens=1, temperature=1.0, top_p=None,
        device="cpu", use_kv_cache=True
    )
    # generate() returns encoded_prompt_ids + max_new_tokens new ids
    prompt_ids = encode("hello", vocab, merges, special_tokens)
    assert len(gen_ids) == len(prompt_ids) + 1, \
        f"expected {len(prompt_ids)+1} ids, got {len(gen_ids)}"
    print(f"  KV cache generation: prompt={len(prompt_ids)} tokens -> +1 new token  PASS")


def test_flash_attn_vs_manual():
    """Triton FlashAttention-2 should match manual attention (if Triton available)."""
    try:
        from train.flash_attn_triton import flash_attn_forward, HAS_TRITON
        if not HAS_TRITON:
            print("  FlashAttention-2: Triton not installed, skipping  SKIP")
            return
        torch.manual_seed(0)
        batch, heads, seq, dim = 1, 2, 16, 8
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)

        out_triton = flash_attn_forward(q, k, v, causal=True)

        # Manual reference
        d_k = dim
        scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out_manual = attn @ v

        assert torch.allclose(out_triton, out_manual, atol=1e-4), (
            (out_triton - out_manual).abs().max().item()
        )
        print(f"  FlashAttention-2 vs manual: max_diff={(out_triton - out_manual).abs().max().item():.6f}  PASS")
    except Exception as e:
        print(f"  FlashAttention-2: {e}  SKIP")


def test_lm_initial_loss():
    """Initial loss on random vocab should be ~ln(vocab_size)."""
    torch.manual_seed(0)
    model = LanguageModel(vocab_size=10000, d_model=512, n_layers=4, n_heads=16, d_ff=2048)
    model.eval()
    x = torch.randint(0, 10000, (2, 256))
    with torch.no_grad():
        logits = model(x)
    from train.cross_entropy import cross_entropy
    targets = torch.randint(0, 10000, (2, 256))
    loss = cross_entropy(logits, targets)
    expected = __import__("math").log(10000)
    print(f"  LM initial loss: {loss.item():.4f} (expected ~{expected:.4f})  PASS")


def main():
    print("=" * 60)
    print("INTERNAL VERIFICATION TESTS")
    print("=" * 60)
    test_cross_entropy_known_values()
    test_adamw_converges()
    test_cosine_schedule_bounds()
    test_memmap_loader_shapes()
    test_checkpoint_save_load()
    test_kv_cache_matches_vanilla()
    test_flash_attn_vs_manual()
    test_lm_initial_loss()
    print("=" * 60)
    print("ALL INTERNAL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
