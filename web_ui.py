"""Gradio web UI for the hand-written transformer language model.

Device-agnostic: runs on CUDA if available, otherwise CPU.
Deployment-ready: absolute paths, graceful missing-checkpoint handling.

Run:
    python web_ui.py

Then open the URL it prints (usually http://127.0.0.1:7860) in your browser.
"""

import os
import sys

# ---- Make repo root, model/, and tokenizer/ importable from any CWD ----
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in (_REPO_ROOT, os.path.join(_REPO_ROOT, "model"), os.path.join(_REPO_ROOT, "tokenizer")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import gradio as gr
import torch

from lm import LanguageModel
from bpe_codec import encode, decode


# ---- Configuration ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 10000
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 16
D_FF = 2048
SPECIAL_TOKENS = ["<|endoftext|>"]


# ---- Load model + tokenizer once at startup --------------------------------

def _load_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, h = line.split("\t")
            vocab[int(i)] = bytes.fromhex(h)
    return vocab


def _load_merges(path):
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split("\t")
            merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return merges


MODEL = None
VOCAB = {}
MERGES = []
LOAD_ERROR = None

try:
    MODEL = LanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        device=DEVICE,
    )
    ckpt_path = os.path.join(_REPO_ROOT, "checkpoints", "final.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        MODEL.load_state_dict(checkpoint["model"])
        MODEL.eval()
    else:
        LOAD_ERROR = f"Checkpoint not found at {ckpt_path}. Train first, then place final.pt in checkpoints/."

    VOCAB = _load_vocab(os.path.join(_REPO_ROOT, "data", "tokenizer", "vocab.txt"))
    MERGES = _load_merges(os.path.join(_REPO_ROOT, "data", "tokenizer", "merges.txt"))
except Exception as exc:  # noqa: BLE001
    LOAD_ERROR = f"Failed to load model/tokenizer: {exc}"


# ---- Generation -------------------------------------------------------------

def generate_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Generate text from the model given a prompt."""
    if LOAD_ERROR:
        return f"Model not loaded: {LOAD_ERROR}"

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    ids = encode(prompt, VOCAB, MERGES, SPECIAL_TOKENS)
    if not ids:
        return "Could not encode prompt. Try different text."

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            logits = MODEL(input_ids)
            next_logits = logits[0, -1, :] / max(temperature, 1e-6)
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
                probs_sum = probs.sum(dim=-1, keepdim=True)
                if probs_sum.item() > 0:
                    probs = probs / probs_sum
                else:
                    probs = torch.full_like(probs, 1.0 / probs.shape[-1])

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return decode(input_ids[0].tolist(), VOCAB)


# ---- Gradio interface -------------------------------------------------------

DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

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
        f"Device: {DEVICE_NAME} ({DEVICE}) | Vocab: 10,000 | Layers: 4 | "
        f"Heads: 16 | d_model: 512 | Params: ~22M"
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
