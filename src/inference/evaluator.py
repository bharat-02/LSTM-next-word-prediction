"""Final evaluation: test loss/perplexity + multi-temperature samples."""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config as cfg
from src.data.preprocessing import prepare_corpus
from src.inference.generator import generate_text, load_artifacts, resolve_device
from src.training.trainer import evaluate_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", default="To be, or not to be")
    ap.add_argument("--length", type=int, default=300)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, char2idx, idx2char, seq_len = load_artifacts(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} | device={device} | seq_len={seq_len}")

    prep = prepare_corpus(cfg.RAW_PATH, cfg.CLEAN_PATH, cfg.VOCAB_PATH,
                          seq_len=seq_len, train_frac=cfg.TRAIN_FRAC,
                          val_frac=cfg.VAL_FRAC)
    Xte, yte = prep["splits"]["test"]
    loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                        batch_size=cfg.BATCH_SIZE)
    test_loss = evaluate_loss(model, loader, nn.CrossEntropyLoss(), device)
    print(f"TEST loss={test_loss:.4f} perplexity={math.exp(test_loss):.2f} "
          f"({len(yte)} seqs)")

    rng = np.random.default_rng(cfg.SEED)
    for temp in (0.5, 0.8, 1.0):
        print(f"\n===== temperature {temp} =====")
        print(generate_text(model, args.seed, args.length, temp,
                            char2idx, idx2char, seq_len=seq_len,
                            device=device, rng=rng))
    print("\nNote: samples are model pastiches, not authentic Shakespeare quotes.")


if __name__ == "__main__":
    main()
