"""Fast memmap writer: load existing vocab.txt + merges.txt, encode slice, write uint16 bin.
Use this after bpe_train_opt has already produced vocab.txt and merges.txt.
"""

import os, sys, time
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.join(_REPO_ROOT, "tokenizer"))
from bpe_codec import encode

SLICE = os.path.join(_REPO_ROOT, "data", "tinystories_slice.txt")
TOK_DIR = os.path.join(_REPO_ROOT, "data", "tokenizer")
VOCAB_PATH = os.path.join(TOK_DIR, "vocab.txt")
MERGES_PATH = os.path.join(TOK_DIR, "merges.txt")
MEMMAP_PATH = os.path.join(TOK_DIR, "tinystories_uint16.bin")
SPECIAL_TOKENS = ["<|endoftext|>"]


def load_vocab(path):
    v = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            i, h = line.strip().split("\t")
            v[int(i)] = bytes.fromhex(h)
    return v


def load_merges(path):
    m = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split("\t")
            m.append((bytes.fromhex(a), bytes.fromhex(b)))
    return m


def main():
    vocab = load_vocab(VOCAB_PATH)
    merges = load_merges(MERGES_PATH)
    print("vocab=%d merges=%d" % (len(vocab), len(merges)))

    with open(SLICE, "r", encoding="utf-8") as f:
        text = f.read()
    print("slice chars=%d" % len(text))

    t0 = time.perf_counter()
    ids = encode(text, vocab, merges, SPECIAL_TOKENS)
    enc_t = time.perf_counter() - t0
    print("encoded %d tokens in %.1fs" % (len(ids), enc_t))

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(MEMMAP_PATH)
    print("wrote %s (%.2f MB), max_id=%d" % (MEMMAP_PATH, arr.nbytes / 1024 / 1024, arr.max()))


if __name__ == "__main__":
    main()
