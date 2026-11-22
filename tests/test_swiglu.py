"""Test custom SwiGLU against a manual reference."""

import torch
import torch.nn.functional as F

from model.layers import SwiGLU


def test_swiglu_matches_manual():
    torch.manual_seed(3)
    batch, seq, d_model, d_ff = 2, 5, 8, 12
    x = torch.randn(batch, seq, d_model)

    custom = SwiGLU(d_model, d_ff)

    gate = F.silu(custom.w_gate(x))
    up = custom.w_up(x)
    out_ref = custom.w_down(gate * up)
    out_custom = custom(x)

    assert out_custom.shape == (batch, seq, d_model)
    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5), (
        (out_custom - out_ref).abs().max().item()
    )
