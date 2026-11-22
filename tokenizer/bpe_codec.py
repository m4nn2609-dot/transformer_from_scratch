"""BPE encode/decode.

Given a trained (vocab, merges) from bpe_train_opt:

  encode(text) -> list[int]     token ids
  decode(ids) -> str            original text (lossy for invalid utf-8)

Encoding:
  1. Pre-tokenize the text the same way training did (split on special tokens,
     GPT-2 regex -> tuples of one-byte bytes).
  2. For each pre-token word, apply every merge in order (left-to-right scan,
     merging adjacent pairs) -> final tokens.
  3. Map each token (bytes) -> id via the reverse vocab.

Decoding:
  1. Map id -> bytes via vocab, concatenate.
  2. Decode bytes as utf-8 (errors='replace').
"""

import os
import sys

import regex

_TOKENIZER_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)

from bpe_train_opt import PAT, merge_word, pretokenize_counts


def build_merge_apply_order(merges):
    """Return merges in the order they should be applied during encoding."""
    return list(merges)


def encode(text, vocab, merges, special_tokens):
    """Encode a string into a list of token ids.

    Special tokens in the text are encoded as their own token id (they are in
    the vocab) and act as boundaries — merges never cross them.
    """
    # Reverse vocab: bytes -> id
    bytes_to_id = {b: i for i, b in vocab.items()}

    # Split on special tokens so we can preserve them as single tokens.
    if special_tokens:
        import re as _re

        split_pattern = "|".join(_re.escape(t) for t in special_tokens)
        # Keep the delimiters so we can re-insert them as tokens.
        parts = _re.split(f"({split_pattern})", text)
    else:
        parts = [text]

    token_ids = []
    for part in parts:
        if part in special_tokens:
            tok = part.encode("utf-8")
            token_ids.append(bytes_to_id[tok])
            continue

        # Pre-tokenize this chunk.
        counts = pretokenize_counts(part, [])  # no special tokens inside
        for word, freq in counts.items():
            # Apply every merge in order.
            for pair in merges:
                if pair[0] in word:
                    word = merge_word(word, pair)
            # Map final tokens to ids, repeating the whole word's token
            # sequence for the word's frequency.
            token_sequence = [bytes_to_id[tok] for tok in word]
            token_ids.extend(token_sequence * freq)
    return token_ids


def decode(ids, vocab):
    """Decode a list of token ids back into a string."""
    byte_chunks = [vocab[i] for i in ids]
    raw = b"".join(byte_chunks)
    return raw.decode("utf-8", errors="replace")
