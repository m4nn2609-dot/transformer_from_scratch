"""Fast multiprocessing memmap generator for the TinyStories slice.

Correct BPE encoding: applies merges in order, using a bytes_in_word set for
O(1) first-byte checks instead of O(len(word)) tuple scans. Splits text into
chunks, batches them, encodes in parallel, writes uint16 memmap.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

_REPO_ROOT = os.path.dirname(_TRAIN_DIR)

import re as _re

import regex

SLICE = os.path.join(_REPO_ROOT, "data", "tinystories_slice.txt")
TOK_DIR = os.path.join(_REPO_ROOT, "data", "tokenizer")
VOCAB_PATH = os.path.join(TOK_DIR, "vocab.txt")
MERGES_PATH = os.path.join(TOK_DIR, "merges.txt")
MEMMAP_PATH = os.path.join(TOK_DIR, "tinystories_uint16.bin")
SPECIAL_TOKENS = ["<|endoftext|>"]

BATCH_SIZE = 100


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


def _split_chunks(text, special_tokens):
    if special_tokens:
        pattern = "|".join(_re.escape(t) for t in special_tokens)
        return _re.split(pattern, text)
    return [text]


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk(chunk):
    counts = {}
    for match in regex.finditer(PAT, chunk):
        token_bytes = match.group(0).encode("utf-8")
        word = tuple(bytes([b]) for b in token_bytes)
        counts[word] = counts.get(word, 0) + 1
    return list(counts.items())


def _encode_word(word, merges, bytes_to_id):
    """Apply all merges in order to one word, return token id list."""
    word_list = list(word)
    bytes_in_word = set(word_list)
    for first, second in merges:
        if first not in bytes_in_word:
            continue
        merged = first + second
        result = []
        i = 0
        merged_any = False
        while i < len(word_list):
            if i < len(word_list) - 1 and word_list[i] == first and word_list[i + 1] == second:
                result.append(merged)
                i += 2
                merged_any = True
            else:
                result.append(word_list[i])
                i += 1
        if merged_any:
            word_list = result
            bytes_in_word = set(word_list)
    return [bytes_to_id[tok] for tok in word_list]


def _encode_batch(args):
    parts, merges, bytes_to_id, special_tokens_set = args

    token_ids = []
    for part in parts:
        if part in special_tokens_set:
            tok = part.encode("utf-8")
            token_ids.append(bytes_to_id[tok])
            continue

        words = _pretokenize_chunk(part)
        for word, freq in words:
            token_sequence = _encode_word(word, merges, bytes_to_id)
            token_ids.extend(token_sequence * freq)
    return token_ids


def main():
    print("Loading vocab and merges...")
    vocab = load_vocab(VOCAB_PATH)
    merges = load_merges(MERGES_PATH)
    bytes_to_id = {b: i for i, b in vocab.items()}
    print(f"vocab={len(vocab)} merges={len(merges)}")

    print("Loading slice...")
    with open(SLICE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"slice chars={len(text):,}")

    print("Splitting into chunks...")
    parts = _split_chunks(text, SPECIAL_TOKENS)
    special_tokens_set = set(SPECIAL_TOKENS)
    print(f"num parts={len(parts):,}")

    batches = []
    for i in range(0, len(parts), BATCH_SIZE):
        batches.append(parts[i : i + BATCH_SIZE])
    print(f"num batches={len(batches):,} (batch_size={BATCH_SIZE})")

    print("Encoding batches in parallel...")
    t0 = time.perf_counter()

    num_workers = os.cpu_count() or 4
    args_list = [
        (batch, merges, bytes_to_id, special_tokens_set) for batch in batches
    ]

    all_ids = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i, ids in enumerate(ex.map(_encode_batch, args_list, chunksize=1)):
            all_ids.extend(ids)
            if (i + 1) % 10 == 0:
                print(f"  encoded {i + 1}/{len(batches)} batches...")

    enc_t = time.perf_counter() - t0
    print(f"Encoding done: {enc_t:.1f} s, {len(all_ids):,} tokens")

    print("Writing uint16 memmap...")
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(MEMMAP_PATH)
    print(f"Wrote {MEMMAP_PATH} ({arr.nbytes / 1024 / 1024:.2f} MB), max_id={arr.max()} (vocab size {len(vocab)})")
    print(f"First 20 ids: {arr[:20].tolist()}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
