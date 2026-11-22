"""Test full LanguageModel: shapes, weight tying, forward runs."""

import torch

from model.lm import LanguageModel


def test_lm_forward_shapes():
    torch.manual_seed(11)
    vocab, d_model, n_layers, n_heads, d_ff = 100, 32, 2, 4, 64
    batch, seq = 2, 7

    model = LanguageModel(vocab, d_model, n_layers, n_heads, d_ff)
    model.eval()
    x = torch.randint(0, vocab, (batch, seq))

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch, seq, vocab)


def test_lm_weight_tying():
    torch.manual_seed(12)
    vocab, d_model = 50, 16
    model = LanguageModel(vocab, d_model, 1, 2, 32)
    assert model.lm_head.weight is model.tok_emb.weight


def test_lm_param_count_ballpark():
    """Sanity check: target ~17-30M for d_model=512, n_layers=4, n_heads=16, vocab=10000."""
    vocab, d_model, n_layers, n_heads, d_ff = 10000, 512, 4, 16, 2048
    model = LanguageModel(vocab, d_model, n_layers, n_heads, d_ff)
    total = sum(p.numel() for p in model.parameters())
    # Embedding (tied to lm_head): 10000*512 = 5.12M
    # Per layer: attn 4*512^2 + swiglu 3*512*2048 + 2 norms * 512
    # Final norm: 512
    expected = vocab * d_model + n_layers * (
        4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    ) + d_model
    assert abs(total - expected) < 100, (total, expected)
    assert 15_000_000 < total < 35_000_000, total
