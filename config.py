"""Shared configuration for the Hamlet character-level LSTM project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = os.path.join(BASE_DIR, "hamlet.txt")
CLEAN_PATH = os.path.join(BASE_DIR, "artifacts", "hamlet_clean.txt")
VOCAB_PATH = os.path.join(BASE_DIR, "artifacts", "vocab.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts", "best.pt")
HISTORY_PATH = os.path.join(BASE_DIR, "artifacts", "history.json")
CONFIG_PATH = os.path.join(BASE_DIR, "artifacts", "config.json")

# Corpus splits (contiguous, in text order — no shuffling across splits)
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # derived: 1 - train - val

# Sequence generation
SEQ_LEN = 100
STRIDE = 1

# Model architecture (selected on validation; see README for comparison)
VOCAB_SIZE = None  # filled at runtime from vocab.json
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

# Training
BATCH_SIZE = 128
LR = 3e-3
MAX_EPOCHS = 30
PATIENCE = 5
GRAD_CLIP = 1.0
SEED = 42

# Generation defaults
DEFAULT_TEMPERATURE = 0.8
DEFAULT_GEN_LENGTH = 300

DEVICE = "auto"  # "auto" | "cpu" | "cuda"


def to_dict():
    return {
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
    }
