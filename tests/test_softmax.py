"""Test custom softmax against F.softmax."""

import torch
import torch.nn.functional as F

from model.attention import softmax


def test_softmax_matches_F_softmax():
    torch.manual_seed(5)
    x = torch.randn(3, 4, 6)
    out_custom = softmax(x, dim=-1)
    out_ref = F.softmax(x, dim=-1)
    assert torch.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5)


def test_softmax_rows_sum_to_one():
    torch.manual_seed(6)
    x = torch.randn(2, 3, 4)
    out = softmax(x, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones_like(out[..., 0]), atol=1e-5)
