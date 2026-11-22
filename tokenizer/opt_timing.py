"""Optimized BPE trainer timing on the ~50 MB TinyStories slice.

Prints:
  - pre-tokenization time (multiprocessing stage)
  - total training time
  - per-iteration time at start vs. end
  - the 20 longest tokens in the final vocab

Naive baseline was 333.17 s for comparison.
"""

import os
import statistics
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICE = os.path.join(REPO_ROOT, "data", "tinystories_slice.txt")
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_SIZE = 10000


def main() -> None:
    from bpe_train_opt import train_bpe_opt

    samples: list[tuple[int, float]] = []  # (iteration, elapsed_seconds)

    def progress_cb(iteration: int, elapsed: float, num_pairs: int) -> None:
        samples.append((iteration, elapsed))

    print(f"Slice: {SLICE}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print("Training optimized BPE...")

    t0 = time.perf_counter()
    vocab, merges = train_bpe_opt(
        SLICE, VOCAB_SIZE, SPECIAL_TOKENS, progress_cb=progress_cb
    )
    total = time.perf_counter() - t0

    pre = getattr(train_bpe_opt, "pretokenize_seconds", None)

    if len(samples) >= 2:
        diffs = [
            (samples[i + 1][0], samples[i + 1][1] - samples[i][1])
            for i in range(len(samples) - 1)
        ]
        n_start = max(1, len(diffs) // 10)
        start_avg = statistics.mean(d for _, d in diffs[:n_start])
        end_avg = statistics.mean(d for _, d in diffs[len(diffs) - n_start:])
    else:
        start_avg = end_avg = 0.0

    print(f"\nPre-tokenization time: {pre:.2f} s" if pre is not None else "")
    print(f"Total training time:   {total:.2f} s")
    print(f"Total merges: {len(merges)}")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Per-iteration time (start avg): {start_avg*1000:.2f} ms")
    print(f"Per-iteration time (end avg):   {end_avg*1000:.2f} ms")

    longest = sorted(vocab.values(), key=len, reverse=True)[:20]
    print("\n20 longest tokens:")
    for tok in longest:
        print(f"  {tok!r}  ({len(tok)} bytes)")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
