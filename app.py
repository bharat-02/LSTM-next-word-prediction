"""Streamlit app: Shakespeare-style text generation with the char-level LSTM."""
import numpy as np
import streamlit as st
import torch

import config as cfg
from generate import generate_text, load_artifacts, resolve_device


@st.cache_resource
def get_model():
    device = resolve_device("auto")
    model, char2idx, idx2char, seq_len = load_artifacts(
        cfg.CHECKPOINT_PATH, device=device)
    return model, char2idx, idx2char, seq_len, device


st.title("Hamlet LSTM — Shakespeare-style Text Generation")
st.caption("Character-level LSTM trained on Shakespeare's *Hamlet*. "
           "Samples are model pastiches, not authentic quotes.")

try:
    model, char2idx, idx2char, seq_len, device = get_model()
    st.success(f"Model loaded (device={device}, vocab={len(char2idx)}, seq_len={seq_len}).")
except Exception as e:
    st.error(f"Could not load model from {cfg.CHECKPOINT_PATH}: {e}\n\n"
             "Train first: `python train.py`.")
    st.stop()

seed = st.text_input("Seed text", "To be, or not to be")
length = st.slider("Characters to generate", 50, 1000, 300, 50)
temperature = st.slider("Temperature (lower = predictable, higher = varied)",
                        0.1, 2.0, 0.8, 0.1)

if st.button("Generate"):
    try:
        rng = np.random.default_rng()
        text = generate_text(model, seed, length, float(temperature),
                             char2idx, idx2char, seq_len=seq_len, device=device, rng=rng)
        st.text_area("Generated text", text, height=300)
    except ValueError as e:
        st.error(str(e))
