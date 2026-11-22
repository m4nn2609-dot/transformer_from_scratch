"""Hand-written cross-entropy loss.

Implements log-softmax + NLL from scratch (not nn.CrossEntropyLoss) so every
gradient is transparent. Returns the mean loss over all tokens and the logits
for downstream use.
"""

import torch


def cross_entropy(logits, targets):
    """Cross-entropy loss.

    Args:
        logits: (batch, seq_len, vocab_size) raw model outputs
        targets: (batch, seq_len) integer token ids

    Returns:
        loss: scalar mean cross-entropy over all (batch, seq) positions
    """
    batch, seq_len, vocab_size = logits.shape
    # Flatten for the computation.
    logits = logits.reshape(batch * seq_len, vocab_size)
    targets = targets.reshape(batch * seq_len)

    # Numerically stable log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(
        torch.exp(shifted).sum(dim=-1, keepdim=True)
    )  # (N, 1)
    log_softmax = shifted - log_sum_exp  # (N, vocab)

    # NLL: pick the log-prob of the target token.
    target_log_prob = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)

    loss = -target_log_prob.mean()
    return loss
