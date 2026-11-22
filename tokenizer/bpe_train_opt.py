"""Optimized byte-level BPE trainer.

Optimizations over the naive trainer (tokenizer/train_bpe.py):

  1. Multiprocessing for pre-tokenization only (no asyncio):
     the file is split at <|endoftext|> boundaries, each chunk is
     pre-tokenized in a separate worker, and the Counters are merged
     in the parent process.

  2. Persistent pair counts + pair->words index in the merge loop:
     after each merge, pair counts are updated only for words that
     contained the merged pair, instead of re-scanning every word.

  3. Heap with lazy deletion to fetch the max (count, pair) pair while
     preserving the tie-break rule: on equal count, the lexicographically
     greater byte pair wins.

Produces byte-level BPE merges. Everything stays as bytes.
"""

import heapq
import os
import re as _re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import regex

# GPT-2 pre-tokenization pattern.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_TOKENIZER_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)


def init_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab


def merge_word(word, pair):
    first, second = pair
    merged = first + second
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            result.append(merged)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def pretokenize_counts(text, special_tokens):
    import regex
    if special_tokens:
        pattern = "|".join(_re.escape(t) for t in special_tokens)
        chunks = _re.split(pattern, text)
    else:
        chunks = [text]
    counts = Counter()
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):
            token_bytes = match.group(0).encode("utf-8")
            word = tuple(bytes([b]) for b in token_bytes)
            counts[word] += 1
    return counts


class _MaxPair:
    """Wrapper so a min-heap pops the lexicographically greatest pair on ties.

    Heap entry: (-count, _MaxPair(pair)).
    heapq compares entries lexicographically: first -count (smaller = larger
    real count), then _MaxPair. Our reversed __lt__ makes a *larger* pair
    compare as *smaller*, so ties resolve to the greatest pair.
    """

    __slots__ = ("p",)

    def __init__(self, p: tuple[bytes, bytes]) -> None:
        self.p = p

    def __lt__(self, other: "_MaxPair") -> bool:
        return self.p > other.p

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MaxPair) and self.p == other.p


def _split_chunks(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on escaped special tokens; delimiter text is discarded."""
    if special_tokens:
        pattern = "|".join(_re.escape(t) for t in special_tokens)
        return _re.split(pattern, text)
    return [text]


def _worker_count(chunk_text: str, special_tokens: list[str]) -> Counter:
    """Top-level worker: pre-tokenize one chunk and return its Counter.
    Must be module-level so it is picklable for multiprocessing.
    """
    return pretokenize_counts(chunk_text, special_tokens)


def pretokenize_counts_parallel(
    text: str, special_tokens: list[str], num_workers: int | None = None
) -> Counter:
    """Pre-tokenize `text` in parallel across chunks split on special tokens.

    Falls back to single-process pretokenize_counts when num_workers <= 1 or
    there is only one chunk.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    chunks = _split_chunks(text, special_tokens)
    chunks = [c for c in chunks if c]  # drop empty chunks

    if num_workers <= 1 or len(chunks) <= 1:
        return pretokenize_counts(text, special_tokens)

    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for counter in ex.map(_worker_count, chunks, [special_tokens] * len(chunks)):
            total.update(counter)
    return total


def train_bpe_opt(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
    progress_cb=None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer (optimized).

    Returns (vocab, merges) trained from the input text.
    If `progress_cb` is given, called each iteration as
    `progress_cb(iteration, elapsed_seconds, num_pairs)`.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Stage 1: parallel pre-tokenization -------------------------------
    t_pre = time.perf_counter()
    word_counts: Counter = pretokenize_counts_parallel(
        text, special_tokens, num_workers=num_workers
    )
    t_pre_elapsed = time.perf_counter() - t_pre

    # --- Stage 2: build pair counts + pair->words index + heap ------------
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    pair_counts: Counter = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set] = {}
    heap: list[tuple[int, _MaxPair]] = []

    def _add_word(word: tuple[bytes, ...], freq: int) -> None:
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] += freq
            pair_to_words.setdefault(p, set()).add(word)
            heapq.heappush(heap, (-pair_counts[p], _MaxPair(p)))

    def _remove_word(word: tuple[bytes, ...], freq: int) -> None:
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] -= freq
            if pair_counts[p] <= 0:
                del pair_counts[p]
            s = pair_to_words.get(p)
            if s is not None:
                s.discard(word)
                if not s:
                    del pair_to_words[p]
            if p in pair_counts:
                heapq.heappush(heap, (-pair_counts[p], _MaxPair(p)))

    for word, freq in word_counts.items():
        _add_word(word, freq)

    # --- Stage 3: merge loop ----------------------------------------------
    start = time.perf_counter()
    iteration = 0

    while len(vocab) < vocab_size:
        # Best pair with lazy deletion; ties -> lexicographically greater pair.
        best_pair: tuple[bytes, bytes] | None = None
        while heap:
            neg_c, wp = heap[0]
            c = -neg_c
            p = wp.p
            if pair_counts.get(p, 0) == c:
                best_pair = p
                break
            heapq.heappop(heap)  # stale entry
        if best_pair is None:
            break

        first, second = best_pair
        new_id = len(vocab)
        vocab[new_id] = first + second
        merges.append(best_pair)

        # Only words containing the pair are affected.
        affected = list(pair_to_words.get(best_pair, ()))
        merged_freqs: dict[tuple[bytes, ...], int] = {}
        for W in affected:
            freq = word_counts.pop(W)
            _remove_word(W, freq)
            Wp = merge_word(W, best_pair)
            merged_freqs[Wp] = merged_freqs.get(Wp, 0) + freq

        for Wp, freq in merged_freqs.items():
            word_counts[Wp] = word_counts.get(Wp, 0) + freq
            _add_word(Wp, freq)

        iteration += 1
        if progress_cb is not None:
            progress_cb(iteration, time.perf_counter() - start, len(pair_counts))

    # (Optional) report pre-tokenization time via a sentinel on the first call.
    # Stored as an attribute for the caller to read if desired.
    train_bpe_opt.pretokenize_seconds = t_pre_elapsed  # type: ignore[attr-defined]

    return vocab, merges


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    import tempfile
    import os

    CORPUS = (
        "low low low low low lower lower widest widest widest "
        "newest newest newest newest newest newest"
    )
    SPECIAL = ["<|endoftext|>"]

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(CORPUS)
        tmp_path = f.name

    try:
        ok = compare_naive_vs_optimized(tmp_path, 263, SPECIAL)
        print(f"Tiny corpus: naive == optimized merges? {ok}")

        _, merges_opt = train_bpe_opt(tmp_path, 263, SPECIAL)
        print("Optimized merges:")
        for m in merges_opt:
            print(f"  {m}")
    finally:
        os.unlink(tmp_path)
