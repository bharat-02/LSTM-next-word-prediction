"""Corpus cleaning, vocabulary, and sequence generation for Hamlet char-LM.

Keeps hamlet.txt unchanged; writes artifacts/hamlet_clean.txt + vocab.json.
Cleaning is deliberately minimal: no lowercasing, no spelling modernisation,
no removal of dialogue / names / stage directions / punctuation.
"""
import json
import os
import re

import numpy as np


def load_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def clean_text(text):
    """Minimal cleanup preserving content.

    - Unify line endings to \\n
    - Strip trailing whitespace on each line (keep leading indentation:
      dialogue indentation is structural)
    - Collapse 3+ consecutive newlines to exactly 2 (keep scene breaks)
    - Ensure file ends with a single \\n
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip("\n") + "\n"
    return text


def clean_corpus(raw_path, clean_path):
    raw = load_text(raw_path)
    cleaned = clean_text(raw)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    with open(clean_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(cleaned)
    return {
        "raw_chars": len(raw),
        "clean_chars": len(cleaned),
        "raw_lines": raw.count("\n"),
        "clean_lines": cleaned.count("\n"),
    }


def build_vocab(text):
    """Sorted-char vocabulary -> (chars, char2idx, idx2char)."""
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return chars, char2idx, idx2char


def save_vocab(chars, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return chars, char2idx, idx2char


def encode(text, char2idx):
    return np.array([char2idx[ch] for ch in text], dtype=np.int64)


def decode(indices, idx2char):
    return "".join(idx2char[int(i)] for i in indices)


def split_corpus(text, train_frac=0.80, val_frac=0.10):
    """Contiguous split in original text order (no shuffling)."""
    n = len(text)
    i_train = int(n * train_frac)
    i_val = i_train + int(n * val_frac)
    return text[:i_train], text[i_train:i_val], text[i_val:]


def make_sequences(indices, seq_len, stride=1):
    """Fixed-length inputs + next-char targets with verified alignment.

    X[k] = indices[k*stride : k*stride+seq_len]
    y[k] = indices[k*stride+seq_len]
    so decode(X[k]) + decode(y[k]) == the underlying contiguous slice.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_seq = (len(indices) - seq_len - 1) // stride + 1
    if n_seq <= 0:
        raise ValueError(
            f"Text too short ({len(indices)} chars) for seq_len={seq_len}"
        )
    X = np.empty((n_seq, seq_len), dtype=np.int64)
    y = np.empty((n_seq,), dtype=np.int64)
    for k in range(n_seq):
        s = k * stride
        X[k] = indices[s: s + seq_len]
        y[k] = indices[s + seq_len]
    return X, y


def prepare_corpus(raw_path="hamlet.txt", clean_path=None,
                   vocab_path=None, seq_len=100, stride=1,
                   train_frac=0.80, val_frac=0.10):
    """End-to-end: clean -> vocab -> splits -> integer sequences."""
    import config as cfg

    clean_path = clean_path or cfg.CLEAN_PATH
    vocab_path = vocab_path or cfg.VOCAB_PATH
    stats = clean_corpus(raw_path, clean_path)
    text = load_text(clean_path)
    chars, char2idx, idx2char = build_vocab(text)
    save_vocab(chars, vocab_path)
    train_txt, val_txt, test_txt = split_corpus(text, train_frac, val_frac)
    data = encode(text, char2idx)
    n_train = int(len(data) * train_frac)
    n_val = n_train + int(len(data) * val_frac)
    splits = {}
    for name, part in (("train", data[:n_train]),
                       ("val", data[n_train:n_val]),
                       ("test", data[n_val:])):
        X, y = make_sequences(part, seq_len, stride)
        splits[name] = (X, y)
    stats.update({
        "vocab_size": len(chars),
        "train_chars": len(train_txt),
        "val_chars": len(val_txt),
        "test_chars": len(test_txt),
        "train_seqs": splits["train"][0].shape[0],
        "val_seqs": splits["val"][0].shape[0],
        "test_seqs": splits["test"][0].shape[0],
    })
    return {
        "text": text,
        "chars": chars,
        "char2idx": char2idx,
        "idx2char": idx2char,
        "splits": splits,
        "stats": stats,
    }
