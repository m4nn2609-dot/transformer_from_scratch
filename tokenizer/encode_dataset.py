"""Train BPE on the TinyStories slice and encode it to a uint16 memmap.

Pipeline:
  1. train_bpe_opt(slice_path, vocab_size, special_tokens) -> (vocab, merges)
  2. Save vocab + merges to data/tokenizer/ for Phase 3 to load.
  3. Encode the slice text to token ids and write a uint16 memmap.

All ids fit in uint16 since vocab_size=10000 < 65536.
"""

import os
import sys
import time

_TOKENIZER_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)

from bpe_train_opt import train_bpe_opt
from bpe_codec import encode

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICE = os.path.join(REPO_ROOT, "data", "tinystories_slice.txt")
TOK_DIR = os.path.join(REPO_ROOT, "data", "tokenizer")
VOCAB_PATH = os.path.join(TOK_DIR, "vocab.txt")
MERGES_PATH = os.path.join(TOK_DIR, "merges.txt")
MEMMAP_PATH = os.path.join(TOK_DIR, "tinystories_uint16.bin")
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 10000


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        for i in sorted(vocab):
            f.write(f"{i}\t{vocab[i].hex()}\n")


def save_merges(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()}\t{b.hex()}\n")


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    os.makedirs(TOK_DIR, exist_ok=True)

    print(f"Loading slice: {SLICE}")
    text = load_text(SLICE)
    print(f"Slice chars: {len(text):,}")

    # --- Stage 1: train BPE ------------------------------------------------
    print(f"Training BPE vocab_size={VOCAB_SIZE} ...")
    t0 = time.perf_counter()
    vocab, merges = train_bpe_opt(SLICE, VOCAB_SIZE, SPECIAL_TOKENS)
    train_t = time.perf_counter() - t0
    print(f"Training done: {train_t:.1f} s, vocab={len(vocab)}, merges={len(merges)}")

    # --- Stage 2: save vocab + merges -------------------------------------
    save_vocab(vocab, VOCAB_PATH)
    save_merges(merges, MERGES_PATH)
    print(f"Saved vocab -> {VOCAB_PATH}")
    print(f"Saved merges -> {MERGES_PATH}")

    # --- Stage 3: encode to uint16 memmap ---------------------------------
    print("Encoding slice to token ids ...")
    t0 = time.perf_counter()
    ids = encode(text, vocab, merges, SPECIAL_TOKENS)
    encode_t = time.perf_counter() - t0
    print(f"Encoding done: {encode_t:.1f} s, {len(ids):,} tokens")

    # Write memmap as raw uint16 little-endian bytes.
    import numpy as np

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(MEMMAP_PATH)
    print(f"Wrote memmap -> {MEMMAP_PATH} ({arr.nbytes / 1024 / 1024:.2f} MB)")
    print(f"First 20 ids: {arr[:20].tolist()}")
    print(f"Max id: {arr.max()} (vocab size {len(vocab)})")


if __name__ == "__main__":
    main()
