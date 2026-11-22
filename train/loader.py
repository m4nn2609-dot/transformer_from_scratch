"""Memmap data loader for tokenized uint16 data.

Yields (inputs, targets) batches of shape (batch, seq_len) where targets are
inputs shifted by one. Uses numpy memmap so the full dataset never needs to
fit in RAM.
"""

import numpy as np
import torch


class MemmapDataset:
    def __init__(self, memmap_path, seq_len, dtype=np.uint16):
        self.data = np.memmap(memmap_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.num_tokens = len(self.data)
        # Number of complete sequences we can form.
        self.num_sequences = self.num_tokens // seq_len

    def __len__(self):
        return self.num_sequences

    def get_batch(self, batch_size, device, rng=None):
        """Sample `batch_size` random sequences.

        Returns (inputs, targets) as LongTensors on `device`, each
        (batch_size, seq_len). targets[i] = inputs[i] shifted by 1.
        """
        if rng is None:
            rng = np.random.default_rng()
        max_start = self.num_sequences - 1
        starts = rng.integers(0, max_start, size=batch_size)
        offsets = starts * self.seq_len
        # Build input and target arrays.
        inputs = np.stack([self.data[o : o + self.seq_len] for o in offsets])
        targets = np.stack(
            [self.data[o + 1 : o + self.seq_len + 1] for o in offsets]
        )
        inputs_t = torch.from_numpy(inputs.copy()).long().to(device)
        targets_t = torch.from_numpy(targets.copy()).long().to(device)
        return inputs_t, targets_t
