"""Test custom RMSNorm against a manual reference."""

import torch

from model.layers import RMSNorm


def test_rmsnorm_matches_manual():
    torch.manual_seed(2)
    batch, seq, d = 2, 5, 8
    x = torch.randn(batch, seq, d)
    weight = torch.randn(d)
    eps = 1e-5

    custom = RMSNorm(d, eps=eps)
    custom.weight.data = weight.clone()

    out_custom = custom(x)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out_ref = (x / torch.sqrt(variance + eps)) * weight.view(1, 1, d)

    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
