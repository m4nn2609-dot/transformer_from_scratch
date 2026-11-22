"""Gradio web UI for the hand-written transformer language model.

Run:
    .venv\Scripts\python.exe web_ui.py

Then open the URL it prints (usually http://127.0.0.1:7860) in your browser.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_MODEL_DIR = os.path.join(_REPO_ROOT, "model")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
_TOKENIZER_DIR = os.path.join(_REPO_ROOT, "tokenizer")
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)

import gradio as gr
import torch

from lm import LanguageModel
from bpe_codec import encode, decode


# ---- Load model + tokenizer once at startup --------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = LanguageModel(
    vocab_size=10000,
    d_model=512,
    n_layers=4,
    n_heads=16,
    d_ff=2048,
    device=DEVICE,
)
checkpoint = torch.load("checkpoints/final.pt", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()

vocab = {}
with open(os.path.join(_REPO_ROOT, "data", "tokenizer", "vocab.txt"), "r", encoding="utf-8") as f:
    for line in f:
        i, h = line.strip().split("\t")
        vocab[int(i)] = bytes.fromhex(h)

merges = []
with open(os.path.join(_REPO_ROOT, "data", "tokenizer", "merges.txt"), "r", encoding="utf-8") as f:
    for line in f:
        a, b = line.strip().split("\t")
        merges.append((bytes.fromhex(a), bytes.fromhex(b)))

SPECIAL_TOKENS = ["<|endoftext|>"]


# ---- Generation ------------------------------------------------------------

def generate_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Generate text from the model given a prompt."""
    ids = encode(prompt, vocab, merges, SPECIAL_TOKENS)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                indices_to_remove[sorted_indices] = sorted_indices_to_remove
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return decode(input_ids[0].tolist(), vocab)


# ---- Gradio interface ------------------------------------------------------

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            value="Once upon a time",
            placeholder="Type your story starter here...",
            lines=3,
        ),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=50,
            step=1,
            label="Max new tokens",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="Temperature (higher = more creative/random)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p (nucleus sampling, lower = more focused)",
        ),
    ],
    outputs=gr.Textbox(
        label="Generated text",
        lines=10,
    ),
    title="My Transformer Language Model",
    description=(
        "Hand-written decoder-only transformer trained on TinyStories. "
        f"Device: {DEVICE} | Vocab: 10,000 | Layers: 4 | Heads: 16 | d_model: 512"
    ),
    examples=[
        ["Once upon a time", 50, 0.8, 0.9],
        ["The little girl", 50, 0.8, 0.9],
        ["Once there was a", 50, 0.8, 0.9],
        ["Hello", 50, 0.8, 0.9],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
