"""Test custom Linear against nn.Linear (same weights)."""

import torch

from model.layers import Linear


def test_linear_matches_nn_linear():
    torch.manual_seed(0)
    batch, seq, in_f, out_f = 2, 5, 8, 6
    x = torch.randn(batch, seq, in_f)

    custom = Linear(in_f, out_f)
    ref = torch.nn.Linear(in_f, out_f)
    ref.weight.data = custom.weight.data.clone()
    ref.bias.data = custom.bias.data.clone()

    out_custom = custom(x)
    out_ref = ref(x)

    assert out_custom.shape == (batch, seq, out_f)
    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
