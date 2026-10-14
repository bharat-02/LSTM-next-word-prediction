"""Character-level LSTM language model (PyTorch)."""
import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """Embedding -> stacked LSTM -> Dropout -> per-char logits."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)              # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        out = self.dropout(out)
        logits = self.fc(out)                # (B, T, V)
        return logits, hidden


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, char2idx, idx2char, cfg_dict, history=None):
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout.p,
        "char2idx": char2idx,
        "idx2char": {int(k): v for k, v in idx2char.items()},
        "config": cfg_dict,
        "history": history or {},
    }, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = CharLSTM(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    idx2char = {int(k): v for k, v in ckpt["idx2char"].items()}
    return model, ckpt["char2idx"], idx2char, ckpt
