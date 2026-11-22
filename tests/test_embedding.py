"""Test custom Embedding against nn.Embedding (same weights)."""

import torch

from model.layers import Embedding


def test_embedding_matches_nn_embedding():
    torch.manual_seed(1)
    vocab, dim = 17, 8
    idx = torch.randint(0, vocab, (3, 4))

    custom = Embedding(vocab, dim)
    ref = torch.nn.Embedding(vocab, dim)
    ref.weight.data = custom.weight.data.clone()

    out_custom = custom(idx)
    out_ref = ref(idx)

    assert out_custom.shape == (3, 4, dim)
    assert torch.allclose(out_custom, out_ref, atol=1e-6, rtol=1e-6)
