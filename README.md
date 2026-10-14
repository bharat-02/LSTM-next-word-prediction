# Hamlet LSTM — Character-Level Text Generation

Train a small character-level LSTM on Shakespeare's *Hamlet* and generate
Shakespeare-style text, one character at a time, with temperature sampling.

> Generated passages are model pastiches for study — **not** authentic
> Shakespeare quotations.

## 1. Objective

Replace the original word-level next-word prototype (which suffered train/test
leakage from random splits of overlapping n-grams, per-line contexts, and
lowercasing) with a clean character-level language model:

- 67-character vocabulary (letters with original case, punctuation, stage
  directions, Early Modern spelling kept as-is: *vnfold*, *selfe*, *haue*).
- Contiguous 80/10/10 train/val/test splits in text order — no leakage.
- LSTM next-character prediction with early stopping on validation loss.
- Reusable temperature-based generation from any seed.

## 2. Dataset

- Source: NLTK Gutenberg `shakespeare-hamlet.txt` (First Folio text), stored
  unchanged as `hamlet.txt`.
- Measured: **162,881 chars, 4,922 lines, 67 unique chars**, 956 blank lines
  (structural scene/paragraph breaks, preserved).
- Format: dialogue, speaker names (`Barnardo.`), stage directions (`Enter`,
  `Exit`, `Exeunt`), scene headings, original spelling/punctuation. No line
  numbers, no Gutenberg boilerplate to strip.
- Cleaning (minimal, into `artifacts/hamlet_clean.txt`, original untouched):
  unify line endings → `\n`, strip trailing whitespace, collapse 3+ blank
  lines → 2, single trailing newline. Result: **162,874 chars** (7 chars of
  trailing whitespace/collapsed blanks removed). No lowercasing, no spelling
  changes, no section removal.

## 3. Method: character-level language modeling

1. Build vocab of unique chars → `char2idx` / `idx2char` (`artifacts/vocab.json`).
2. Encode full text to indices; split contiguously 80/10/10.
3. Slice each split into fixed windows: `X[k] = ids[i:i+SEQ_LEN]`,
   `y[k] = ids[i+SEQ_LEN]` (stride 1, alignment-tested).
4. Train LSTM with cross-entropy; validate each epoch; keep best-val checkpoint.
5. Report test loss and perplexity `exp(loss)`; generate with temperature.

## 4. Architecture

`Embedding(67 → 128) → LSTM(256 × 2, dropout 0.2) → Dropout(0.2) → Linear(→ 67)`

- Configs compared (selected on **validation loss**):

| Config | Layers × Hidden | Embed | Dropout | Params | Val loss |
|---|---|---|---|---|---|
| A (partial: 4 epochs, CPU, stopped) | 1 × 256 | 128 | 0.2 | 421,059 | 1.7445 |
| B (selected, full run on CUDA) | 2 × 256 | 128 | 0.2 | 947,395 | **1.6861** |
| C (not run — stopped per user) | 2 × 128 | 64 | 0.3 | 244,355 | — |

- Optimizer Adam (lr 3e-3), gradient clip 1.0, batch 128, early stopping
  (patience 5, max 30 epochs). Loss: cross-entropy over next-char logits.

## 5. Installation

Python 3.11, Windows 64-bit. Install `torch` **first** from the PyTorch index,
then the rest:

```bat
"C:\Program Files\Python311\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"C:\Program Files\Python311\python.exe" -m pip install -r requirements.txt
```

GPU (RTX 3050 6 GB, CUDA 12.6 build) — same code, no changes needed:

```bat
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

## 6. Training

```bat
python -m src.training.trainer                          :: defaults (2x256, seq 100, batch 128)
python -m src.training.trainer --layers 1 --hidden 256  :: config A
python -m src.training.trainer --device cuda            :: force GPU (warns + falls back if absent)
```

Artifacts: `artifacts/hamlet_clean.txt`, `artifacts/vocab.json`,
`artifacts/best.pt`, `artifacts/history.json`, `artifacts/config.json`.
Training prints device, params, per-epoch train/val loss + perplexity, and
final test loss/perplexity. Curves: `artifacts/history.json`.

## 7. Evaluation

```bat
python -m src.inference.evaluator --seed "To be, or not to be" --length 300
```

Reports held-out test loss/perplexity (test Act-V region, never trained on)
plus samples at temperatures 0.5 / 0.8 / 1.0.

### Results (actual, this machine)

- Device: **cuda — NVIDIA GeForce RTX 3050 6GB Laptop GPU** (`torch 2.14.1+cu126`, `cuda_available=True`); CPU-only run used for config A partial.
- Config B, 12 epochs (early stopped, patience 5; best val at epoch 7):
  train loss 1.3996 (final) / best val loss **1.6861** (ppl 5.40) / test loss **1.5721** (ppl **4.82**).
- Training time: **421.9 s (~7 min)** on CUDA, batch 128, seq 100.
- Overfitting check: mild and controlled — train loss kept falling (2.04 → 1.40)
  while val plateaued ≈1.69 after epoch 6; early stopping halted at epoch 12.
  Test loss (1.57, final-act region) is *below* val, so no generalization collapse.

## 8. Text generation

```bat
python -m src.inference.generator --seed "To be, or not to be" --length 300 --temperature 0.8
```

- Lower temperature (≈0.5): more predictable, repetitive.
- Higher temperature (≈1.0): more varied, more spelling/punctuation errors.
- Seed must be a non-empty string with at least one in-vocabulary character;
  unknown chars are reported; temperature must be > 0.
- Streamlit app: `streamlit run app.py` (seed box + length/temperature sliders).

### Samples (actual, seed `To be, or not to be`, 300 chars)

temperature 0.5 (most coherent; learns speaker names, verse breaks, Early Modern spelling):
> To be, or not to be our with proue of the speciall:
> But the speech shall speake of the Armes
>
>    Ham. Hamlet come so but my Lord, ...

temperature 0.8 (varied; some novel word-forms, punctuation mostly intact):
> To be, or not to before,
> But not in this disposition;
> Seick you standed not them
>
>    Polon. I do my Mothers to that? ...

temperature 1.0 (most varied; spelling degrades, e.g. `Vnotiue`, `Moniuic`):
> To be, or not to be suites
> Vnotiue? Lackes you must
> Paistoring ou; naue, this giue it you my all agaid, ...

## 9. Project structure

| File | Role |
|---|---|
| `src/config.py` | Hyperparameters + paths (single source of truth) |
| `src/data/preprocessing.py` | Cleaning, vocab, contiguous splits, sequences |
| `src/models/lstm.py` | `CharLSTM`, param count, checkpoint save/load |
| `src/training/trainer.py` | Training loop, early stopping, test eval |
| `src/inference/generator.py` | Temperature sampling + CLI |
| `src/inference/evaluator.py` | Test metrics + multi-temperature samples |
| `app.py` | Streamlit generation UI |
| `tests/test_char_lstm.py` | 11 automated tests |
| `brain.md` / `plan.md` | Design notes / execution plan |

## 10. Tests

```bat
python -m pytest tests/ -v
```

Covers: corpus load/clean, vocab roundtrip, contiguous splits, sequence
alignment, model shapes, mini training run, checkpoint reload equality,
generation length/vocab at 3 temperatures, temperature validation,
empty/unknown-seed handling, fixed-seed reproducibility.

## 11. Known limitations

- Single play (~130k train chars): learns surface style (speaker names,
  `Enter/Exit`, verse punctuation, *thou/thee*), not plot or semantics.
- Overfits if trained too long — watch the train/val gap; early stopping is on.
- Small context (100 chars ≈ 1–2 verse lines); no long-range plot coherence.
- Outputs are pastiches; never cite them as Shakespeare.
