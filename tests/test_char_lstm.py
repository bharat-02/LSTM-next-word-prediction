"""Automated tests for the Hamlet char-LSTM pipeline."""
import json
import os

import numpy as np
import pytest
import torch

import config as cfg
from data_preprocessing import (
    build_vocab,
    clean_text,
    decode,
    encode,
    load_text,
    load_vocab,
    make_sequences,
    save_vocab,
    split_corpus,
)
from generate import _filter_seed, generate_text
from model import CharLSTM, count_parameters, load_checkpoint, save_checkpoint

SEED = "To be, or not to be"


def test_corpus_loads_and_raw_untouched():
    assert os.path.exists(cfg.RAW_PATH)
    text = load_text(cfg.RAW_PATH)
    assert len(text) > 100_000
    assert "Barnardo" in text  # dialogue preserved


def test_cleaning_preserves_content():
    raw = load_text(cfg.RAW_PATH)
    cleaned = clean_text(raw)
    for keep in ("Barnardo", "Enter", "Exit", "vnfold", "thou", ",", ".", "?", "'"):
        assert keep in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned  # collapsed
    assert cleaned.endswith("\n") and not cleaned.endswith("\n\n")
    # case preserved (no lowercasing)
    assert any(c.isupper() for c in cleaned)


def test_vocab_roundtrip(tmp_path):
    chars, c2i, i2c = build_vocab("Hamlet: To be!")
    assert len(chars) == len(set("Hamlet: To be!"))
    enc = encode("To be!", c2i)
    assert decode(enc, i2c) == "To be!"
    p = tmp_path / "vocab.json"
    save_vocab(chars, str(p))
    chars2, c2i2, i2c2 = load_vocab(str(p))
    assert chars2 == chars and decode(encode("Hamlet", c2i2), i2c2) == "Hamlet"


def test_split_is_contiguous_no_leakage():
    text = "abcdefghijklmnopqrstuvwxyz" * 100
    tr, va, te = split_corpus(text, 0.8, 0.1)
    assert tr + va + te == text  # order preserved, no overlap
    assert abs(len(tr) / len(text) - 0.8) < 0.01


def test_sequences_shapes_and_alignment():
    idx = np.arange(200, dtype=np.int64)
    X, y = make_sequences(idx, seq_len=50)
    assert X.shape == (150, 50) and y.shape == (150,)
    for k in (0, 7, 149):
        assert y[k] == idx[k + 50]
        np.testing.assert_array_equal(X[k], idx[k:k + 50])
    with pytest.raises(ValueError):
        make_sequences(np.arange(10), seq_len=50)


def test_model_forward_shapes():
    m = CharLSTM(vocab_size=67, embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.1)
    x = torch.randint(0, 67, (4, 20))
    logits, _ = m(x)
    assert logits.shape == (4, 20, 67)
    assert count_parameters(m) > 0


def test_short_train_runs_and_reloads(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    V, T, N = 20, 16, 128
    m = CharLSTM(V, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()
    X = torch.randint(0, V, (N, T))
    y = torch.randint(0, V, (N, T))
    m.train()
    opt.zero_grad()
    logits, _ = m(X)
    loss = crit(logits.reshape(-1, V), y.reshape(-1))
    loss.backward()
    opt.step()
    assert float(loss) > 0

    c2i = {chr(65 + i): i for i in range(V)}
    i2c = {i: chr(65 + i) for i in range(V)}
    p = str(tmp_path / "mini.pt")
    save_checkpoint(p, m, c2i, i2c, {"seq_len": T})
    m2, c2i2, i2c2, _ = load_checkpoint(p, map_location="cpu")
    m.eval()
    m2.eval()
    with torch.no_grad():
        a, _ = m(X[:2])
        b, _ = m2(X[:2])
    assert torch.allclose(a, b, atol=1e-6)


def _tiny_model():
    chars = sorted(set(SEED + " Hamlet Horatio Ophelia Enter Exit,.'?!\n"))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    m = CharLSTM(len(chars), embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    m.eval()
    return m, c2i, i2c


def test_generation_length_and_vocab():
    m, c2i, i2c = _tiny_model()
    for temp in (0.5, 0.8, 1.0):
        out = generate_text(m, SEED, 50, temp, c2i, i2c,
                            seq_len=10, device="cpu",
                            rng=np.random.default_rng(0))
        assert len(out) == len(SEED) + 50
        assert set(out) <= set(c2i)


def test_temperature_validation():
    m, c2i, i2c = _tiny_model()
    for bad in (0, -1, 0.0, float("nan"), float("inf"), "hot", None):
        with pytest.raises(ValueError):
            generate_text(m, SEED, 10, bad, c2i, i2c, seq_len=10, device="cpu")


def test_empty_and_unknown_seed():
    m, c2i, i2c = _tiny_model()
    with pytest.raises(ValueError):
        generate_text(m, "", 10, 0.8, c2i, i2c, seq_len=10, device="cpu")
    with pytest.raises(ValueError):
        generate_text(m, None, 10, 0.8, c2i, i2c, seq_len=10, device="cpu")
    with pytest.raises(ValueError):
        generate_text(m, "\x00\x01\x02", 10, 0.8, c2i, i2c,
                      seq_len=10, device="cpu")


def test_reproducibility_with_fixed_rng():
    m, c2i, i2c = _tiny_model()
    kw = dict(model=m, seed=SEED, n_chars=30, temperature=0.8,
              char2idx=c2i, idx2char=i2c, seq_len=10, device="cpu")
    a = generate_text(rng=np.random.default_rng(123), **kw)
    b = generate_text(rng=np.random.default_rng(123), **kw)
    assert a == b
