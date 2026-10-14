"""Temperature-based character generation (reusable, no retraining needed)."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from model import load_checkpoint


def resolve_device(pref="auto"):
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_artifacts(checkpoint_path=None, vocab_path=None, device="cpu"):
    checkpoint_path = checkpoint_path or cfg.CHECKPOINT_PATH
    model, char2idx, idx2char, ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    seq_len = ckpt.get("config", {}).get("seq_len", cfg.SEQ_LEN)
    return model, char2idx, idx2char, seq_len


def _filter_seed(seed, char2idx):
    """Keep known chars; raise if nothing usable remains."""
    if seed is None or not isinstance(seed, str) or seed == "":
        raise ValueError("Seed must be a non-empty string.")
    unknown = sorted(set(seed) - set(char2idx))
    filtered = "".join(ch for ch in seed if ch in char2idx)
    if not filtered:
        raise ValueError(
            "Seed contains no characters from the model vocabulary. "
            f"Unknown chars: {unknown!r}. Vocab size: {len(char2idx)}."
        )
    return filtered, unknown


def generate_text(model, seed, n_chars, temperature, char2idx, idx2char,
                  seq_len=None, device="cpu", rng=None):
    """Generate n_chars after seed, one char at a time.

    Lower temperature (~0.5) -> predictable; higher (~1.0) -> varied.
    Returns seed + generated chars; every char is in-vocabulary.
    """
    if temperature is None or not isinstance(temperature, (int, float)) \
            or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be a positive number, got {temperature!r}.")
    if n_chars is None or int(n_chars) < 0:
        raise ValueError(f"n_chars must be >= 0, got {n_chars!r}.")
    n_chars = int(n_chars)
    seq_len = seq_len or cfg.SEQ_LEN

    seed_known, _unknown = _filter_seed(seed, char2idx)
    rng = rng or np.random.default_rng()
    model.eval()

    vocab_size = len(char2idx)
    context = [char2idx[ch] for ch in seed_known]
    out_chars = list(seed_known)
    with torch.no_grad():
        for _ in range(n_chars):
            window = context[-seq_len:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits, _ = model(x)              # (1, T, V)
            logits = logits[0, -1] / float(temperature)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            nxt = int(rng.choice(vocab_size, p=probs))
            context.append(nxt)
            out_chars.append(idx2char[nxt])
    return "".join(out_chars)


def main():
    ap = argparse.ArgumentParser(description="Generate Shakespeare-style text with the char-LSTM.")
    ap.add_argument("--seed", default="To be, or not to be",
                    help="Starting text (must contain in-vocabulary chars).")
    ap.add_argument("--length", type=int, default=cfg.DEFAULT_GEN_LENGTH)
    ap.add_argument("--temperature", type=float, default=cfg.DEFAULT_TEMPERATURE)
    ap.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, char2idx, idx2char, seq_len = load_artifacts(args.checkpoint, device=device)
    text = generate_text(model, args.seed, args.length, args.temperature,
                         char2idx, idx2char, seq_len=seq_len, device=device)
    print(text)


if __name__ == "__main__":
    main()
