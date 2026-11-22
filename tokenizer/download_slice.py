"""Download a ~50 MB TinyStories slice cut at the last <|endoftext|>."""

import os
from datasets import load_dataset

TARGET_BYTES = 50 * 1024 * 1024  # ~50 MB
SPECIAL = "<|endoftext|>"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data", "tinystories_slice.txt")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("Loading TinyStories (this downloads the dataset the first time)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Join stories with the special token between them.
    parts = []
    total = 0
    for example in ds:
        text = example["text"]
        if parts:
            parts.append(SPECIAL)
            total += len(SPECIAL.encode("utf-8"))
        encoded = text.encode("utf-8")
        parts.append(text)
        total += len(encoded)
        if total >= TARGET_BYTES:
            break

    joined = SPECIAL.join(parts) if parts else ""
    joined_bytes = joined.encode("utf-8")

    # Cut at the last <|endoftext|> so we never split a story.
    last = joined_bytes.rfind(SPECIAL.encode("utf-8"))
    if last == -1:
        slice_bytes = joined_bytes
    else:
        slice_bytes = joined_bytes[: last + len(SPECIAL.encode("utf-8"))]

    with open(OUT, "wb") as f:
        f.write(slice_bytes)

    print(f"Wrote {OUT}: {len(slice_bytes)} bytes "
          f"({len(slice_bytes) / 1024 / 1024:.2f} MB), "
          f"{slice_bytes.count(SPECIAL.encode('utf-8'))} special tokens")


if __name__ == "__main__":
    main()
