# brain.md — LSTM Hamlet Text Generation: Project Brain

## 1. Current state (inspected 2026-10-14)

### Files present
- `hamlet.txt` (167,803 bytes on disk, 162,881 chars, 4,922 lines) — NLTK Gutenberg `shakespeare-hamlet.txt`, First Folio spelling (`vnfold`, `selfe`, `haue`, `Leige-men`). Preserved as-is; never edit destructively.
- `experiemnts.ipynb` (note typo in filename) — word-level next-word prediction tutorial: lowercases text, Keras `Tokenizer` (word-level), n-gram sequences per line, `pad_sequences`, random `train_test_split(test_size=0.2)`, `Embedding(total_words,100) + LSTM(150, return_sequences=True) + Dropout(0.2) + LSTM(100) + Dense(total_words, softmax)`, `categorical_crossentropy` + Adam, `EarlyStopping(patience=3)`, 50 epochs max. Also defines an unused GRU variant. No saved outputs, no metrics recorded.
- `app.py` — Streamlit next-word predictor loading `next_word_lstm.h5` + `tokenizer.pickle`, neither of which exists in the repo. Broken as-is (model file missing, `np.argmax` index-compare bug: compares array to int).
- `requirements.txt` — `tensorflow==2.15.0, numpy, scikit-learn, matplotlib, streamlit`. No torch. No versions pinned except TF.
- `README.md` — single line `# LSTM-next-word-prediction`.
- Git: remote `origin https://github.com/bharat-02/LSTM-next-word-prediction.git`, branch `main`, working tree has `M experiemnts.ipynb` (line-ending normalization only).

### Dataset profile (measured, not guessed)
- 162,881 chars, 4,922 lines, 956 blank lines (~19% — structural, keep as paragraph/scene breaks).
- 67 unique chars: `\n space ! & ' ( ) , - . 1 5 9 : ; ? A-Z (no J,U,X? actually has... check: A-Z minus J? has j lower) [ ] a-z`. Full set measured via Counter. No line numbers, no Gutenberg license header/footer inside file (starts directly `[The Tragedie of Hamlet...]`, ends `FINIS...`). No action needed to strip headers.
- Issues: trailing whitespace, `\r\n` vs `\n` risk, 3+ consecutive blank lines in places, wrapped verse lines (e.g. `Stand & vnfold\nyour selfe` — keep, it is real line structure). No destructive normalization: keep case, punctuation, Early Modern spelling, character names, stage directions (`Enter`, `Exit`, `Exeunt`).

### Baseline problems (why rebuild)
1. **Wrong granularity for corpus size**: word-level vocab on ~30k tokens Hamlet → large softmax (~4-5k words), sparse, memorizes. Char-level vocab (~67) is the right choice per task.
2. **Train/test leakage**: random `train_test_split` on overlapping n-gram windows from same lines → near-duplicate sequences across splits. Must use contiguous splits in text order.
3. **Line-fragmented contexts**: n-grams built per-line, so cross-line/scene context never learned. Must build one continuous stream.
4. **Lowercasing**: destroys `Hamlet` vs `hamlet`, speaker names (`Barnardo.` vs `barnardo.`). Keep case for char-level.
5. **No validation split, no perplexity, no checkpointing to disk besides final `.h5`, no seed/temperature sampling** — only argmax next-word.
6. **TF 2.15 on Windows = CPU-only**; no GPU path. PyTorch gives a clean CUDA path for RTX 3050 6GB + CPU fallback.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Framework | **PyTorch** (device-agnostic `cpu`/`cuda`) | TF≥2.11 has no Windows GPU; torch CUDA wheels bundle CUDA, code runs on both. Small model fits 6GB VRAM easily. |
| Level | **Character-level** | 67-char vocab, 163k chars → dense signal; task requirement; avoids sparse word softmax. |
| Cleaning | Minimal: unify `\n`, strip trailing spaces, collapse 3+ newlines→2, keep everything else byte-identical in spirit | Preserve dialogue/names/punct/spelling; produce `hamlet_clean.txt`, never touch `hamlet.txt`. |
| Splits | Contiguous 80/10/10 in text order (train/val/test), sequences built *within* each split | Zero cross-split leakage; val/test measure generalization to later acts. |
| Seq len | 100 (tune 60 vs 100) | ~1-2 verse lines of context; longer → fewer sequences + slower; shorter → no speaker/verse structure. |
| Architecture | `Embedding(67, 128) → LSTM(256×2, dropout 0.2) → Dropout → Linear(67)`; compare 1-layer/256 vs 2-layer/256 vs 2-layer/128 | Small enough to not memorize 163k chars (~0.8-1.5M params), large enough to learn spelling/punct patterns. |
| Loss/opt | CrossEntropy (logits), Adam lr=3e-3, grad clip 1.0, StepLR/ReduceLROnPlateau, early stop patience 5, best-val checkpoint | Standard for char-LM; clipping stabilizes LSTM; early stop on val not train. |
| Batch | 128 (64 if OOM) | Fits 6GB VRAM with 2×256 LSTM at seq 100; CPU-friendly too. |
| Generation | Temperature sampling, seed validation, OOV handling | Task requirement; temps 0.5/0.8/1.0. |
| Python env | Use Python 3.11 (`C:\Program Files\Python311`) + `torch --index-url cpu` (or CUDA if user wants) | 3.11 has project deps (numpy/sklearn/matplotlib); 3.10 install broken; 3.13 lacks torch compat breadth. CPU training here; CUDA path verified by code + `torch.cuda.is_available()` probe. |

## 3. Risks / limitations (honest)
- Single play (~130k train chars) → model learns surface style (names, `Enter/Exit`, punctuation, `thou/thee`), not plot/semantics. Will overfit if trained too long; val gap is the signal.
- No human-level quality; samples are style pastiches, never authentic quotes.
- Actual GPU training in *this* session runs on CPU unless user installs CUDA torch build (documented command provided). Device actually used is logged and reported.
- `experiemnts.ipynb` typo kept (don't rename destructively); new code lives in modules, notebook preserved as experiment record.
- `app.py` word-level artifacts (`next_word_lstm.h5`, `tokenizer.pickle`) don't exist → old app can't run; replaced by char-level `generate.py` + updated Streamlit app.

## 4. File map (target)
- `src/config.py` — single source of truth (paths, splits, seq len, model/training hyperparams, device).
- `src/data/preprocessing.py` — clean → vocab → encode → contiguous splits → sequences.
- `src/models/lstm.py` — `CharLSTM` (Embedding-LSTM-Dropout-Linear), param count, checkpoint save/load.
- `src/training/trainer.py` — train/val loop, early stopping, perplexity, history.json, best.pt.
- `src/inference/generator.py` — `load_artifacts()` + `generate_text(seed, n, temperature)` + CLI.
- `src/inference/evaluator.py` — test loss/ppl + multi-temperature samples.
- `tests/test_char_lstm.py` — preprocessing, vocab roundtrip, shapes, short train, reload, generation, temperature/empty-input validation.
- Artifacts: `artifacts/hamlet_clean.txt`, `artifacts/vocab.json`, `artifacts/config.json`, `artifacts/best.pt`, `artifacts/history.json`. Note: `hamlet_clean.txt` also saved at repo root? No — keep in `artifacts/` to avoid clutter; original `hamlet.txt` stays at root.
- `app.py` — rewritten for char-level generation (keeps Streamlit entry point, same filename to avoid restructuring).
- `requirements.txt`, `.gitignore`, `README.md`, `brain.md`, `plan.md` — updated.
