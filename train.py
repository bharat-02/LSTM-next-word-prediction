"""Training + checkpointing for the Hamlet char-LSTM.

- Contiguous 80/10/10 splits (no leakage)
- CrossEntropy next-char loss, Adam, grad clip, early stopping on val
- Saves best-val checkpoint, history.json, config.json
- Reports test loss + perplexity at the end
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from data_preprocessing import prepare_corpus
from model import CharLSTM, count_parameters, save_checkpoint


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref="auto"):
    if pref == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but torch.cuda.is_available()==False; using CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if pref == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        logits, _ = model(Xb)                      # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
        bs = Xb.size(0)
        total_loss += loss.item() * bs
        total_tokens += bs
    return total_loss / total_tokens


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits, _ = model(Xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total += loss.item() * Xb.size(0)
        n += Xb.size(0)
    return total / n


def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                vocab_size, char2idx, idx2char,
                embed_dim=cfg.EMBED_DIM, hidden_dim=cfg.HIDDEN_DIM,
                num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT,
                batch_size=cfg.BATCH_SIZE, lr=cfg.LR,
                max_epochs=cfg.MAX_EPOCHS, patience=cfg.PATIENCE,
                device=None, checkpoint_path=None, history_path=None):
    device = device or resolve_device(cfg.DEVICE)
    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    n_params = count_parameters(model)
    print(f"Device: {device} | params: {n_params:,} | "
          f"arch: embed={embed_dim} hidden={hidden_dim}x{num_layers} drop={dropout}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = evaluate_loss(model, val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"epoch {epoch:02d}/{max_epochs}  train={tr:.4f} (ppl {math.exp(tr):.2f})  "
              f"val={va:.4f} (ppl {math.exp(va):.2f})", flush=True)
        if va < best_val - 1e-4:
            best_val = va
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.4f}).")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    test_loss = evaluate_loss(model, test_loader, criterion, device)
    test_ppl = math.exp(test_loss)
    print(f"BEST val={best_val:.4f} (ppl {math.exp(best_val):.2f}) | "
          f"TEST={test_loss:.4f} (ppl {test_ppl:.2f}) | time={train_time:.1f}s on {device}")

    history.update({"test_loss": test_loss, "test_ppl": test_ppl,
                    "train_seconds": train_time, "device": str(device),
                    "n_params": n_params, "best_val": best_val})
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_checkpoint(checkpoint_path, model, char2idx, idx2char,
                        {"seq_len": cfg.SEQ_LEN, "embed_dim": embed_dim,
                         "hidden_dim": hidden_dim, "num_layers": num_layers,
                         "dropout": dropout, "lr": lr, "batch_size": batch_size},
                        history=history)
        print(f"Saved checkpoint -> {checkpoint_path}")
    if history_path:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=cfg.SEQ_LEN)
    ap.add_argument("--embed", type=int, default=cfg.EMBED_DIM)
    ap.add_argument("--hidden", type=int, default=cfg.HIDDEN_DIM)
    ap.add_argument("--layers", type=int, default=cfg.NUM_LAYERS)
    ap.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    ap.add_argument("--batch", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=cfg.PATIENCE)
    ap.add_argument("--device", default=cfg.DEVICE)
    ap.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH)
    ap.add_argument("--history", default=cfg.HISTORY_PATH)
    args = ap.parse_args()

    set_seed()
    device = resolve_device(args.device)
    print(f"Resolved device: {device} (cuda_available={torch.cuda.is_available()})")
    prep = prepare_corpus(cfg.RAW_PATH, cfg.CLEAN_PATH, cfg.VOCAB_PATH,
                          seq_len=args.seq_len,
                          train_frac=cfg.TRAIN_FRAC, val_frac=cfg.VAL_FRAC)
    print(f"Corpus stats: {prep['stats']}")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (prep["splits"]["train"],
                                          prep["splits"]["val"],
                                          prep["splits"]["test"])
    train_model(Xtr, ytr, Xva, yva, Xte, yte,
                len(prep["chars"]), prep["char2idx"], prep["idx2char"],
                embed_dim=args.embed, hidden_dim=args.hidden,
                num_layers=args.layers, dropout=args.dropout,
                batch_size=args.batch, lr=args.lr, max_epochs=args.epochs,
                patience=args.patience, device=device,
                checkpoint_path=args.checkpoint, history_path=args.history)
    with open(cfg.CONFIG_PATH, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
