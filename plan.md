# plan.md — Execution Plan: Character-Level LSTM for Hamlet

## Objective
Replace the leaky word-level next-word prototype with a clean character-level LSTM pipeline (preprocess → train → evaluate → generate), verified on CPU with a CUDA-ready code path, tested, documented, and committed.

## Phase 0 — Inspect & record (DONE)
- [x] Read `app.py`, `experiemnts.ipynb`, `requirements.txt`, `README.md`, git remote/branch/status.
- [x] Measure `hamlet.txt`: 162,881 chars, 4,922 lines, 67 unique chars, First Folio spelling, no Gutenberg boilerplate to strip.
- [x] Probe envs: Py3.11 (project deps, no torch/TF), Py3.13 (no torch), Py3.10 (broken), GPU `RTX 3050 6GB Laptop, driver 581.86`, `nvidia-smi` present, `nvcc` absent (fine — torch bundles CUDA).
- [x] Write `brain.md` (this plan's context) + `plan.md` (this file).

## Phase 1 — Environment
- [ ] Install `torch` (CPU wheel) + `pytest` into Python 3.11: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu pytest`.
- [ ] If user later wants GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu126` (verify `torch.cuda.is_available()`).
- [ ] Update `requirements.txt` with pinned working set.
- [ ] Add `.gitignore` (venv, `__pycache__`, artifacts `*.pt`, IDE, etc. — but keep a small committed checkpoint? Decision: gitignore `artifacts/*.pt` + `runs/` if large; commit `vocab.json`/`config.json`/`history.json`).

## Phase 2 — Preprocessing (`data_preprocessing.py`, `config.py`)
- [ ] `clean_text()`: `\r\n`→`\n`, strip trailing spaces per line, collapse `\n{3,}`→`\n\n`, ensure single trailing newline. No lowercasing, no spelling changes.
- [ ] Save `artifacts/hamlet_clean.txt`; assert original `hamlet.txt` untouched (hash/size check in tests).
- [ ] `build_vocab()`: sorted unique chars → `char2idx`/`idx2char` → `artifacts/vocab.json`.
- [ ] `split_corpus()`: contiguous 80/10/10 (train/val/test) on the char stream.
- [ ] `make_sequences(split, seq_len, stride=1)`: inputs `[i:i+seq]`, target `i+seq`. Verify alignment: `decode(x)+decode(y) == contiguous slice`.
- [ ] Unit-test all of the above.

## Phase 3 — Model (`model.py`)
- [ ] `CharLSTM(vocab, embed_dim, hidden, layers, dropout)`: Embedding → LSTM(batch_first) → Dropout → Linear(vocab). `count_parameters()` helper.
- [ ] Baseline comparison (small grid, ≤3 configs, short runs e.g. 5 epochs, then full run for winner):
  - A: 1 layer × 256, embed 128, drop 0.2
  - B: 2 layers × 256, embed 128, drop 0.2 (expected winner)
  - C: 2 layers × 128, embed 64, drop 0.3 (regularized small)
- [ ] Select on **val loss**, not train loss.

## Phase 4 — Training (`train.py`)
- [ ] Device: `cuda` if available else `cpu`; log + print actual device. Batch 128, Adam 3e-3, clip 1.0, early stop patience 5 (max ~30 epochs), save best-val `artifacts/best.pt` + `artifacts/history.json` + `artifacts/config.json`.
- [ ] Metrics: train/val loss curves, test loss, perplexity `exp(loss)`. Training time + param count recorded.
- [ ] Overfit check: train≪val gap → more dropout/smaller model; both high → bigger model/longer.

## Phase 5 — Generation (`generate.py`, `app.py`)
- [ ] `generate_text(model, seed, n_chars, temperature, char2idx, idx2char, device)`: validate seed non-empty, temperature > 0, encode (map unknown chars → skip with warning or closest? Decision: raise informative error listing supported set? Better UX: keep known chars, error if seed has zero known chars), sliding window of `seq_len`, temp-scaled sampling, decode, return exactly `len(seed)+n_chars` chars all in vocab.
- [ ] CLI: `python generate.py --seed "To be, or not to be" --length 300 --temperature 0.8`.
- [ ] Rewrite `app.py` (Streamlit): seed box + length slider + temperature slider + generate button, loads `artifacts/best.pt` + `vocab.json`.
- [ ] Sample at 0.5/0.8/1.0 for the report; comment on readability/repetition/punct/names; never present as authentic quotes.

## Phase 6 — Evaluation (`evaluate.py`)
- [ ] `evaluate.py`: loads best checkpoint, reports test loss/ppl, writes sample generations to stdout/md.
- [ ] Save real numbers into README + final report (no fabrication).

## Phase 7 — Tests (`tests/test_char_lstm.py`)
- [ ] `pytest`: corpus load+clean, vocab encode/decode roundtrip, sequence/target shapes + alignment, model forward shapes, 1-epoch mini-train (tiny seq/model), checkpoint save/reload equality, generation length+vocab membership at 3 temps, temperature validation (`<=0` raises), empty/invalid seed handling, determinism under fixed seed.
- [ ] All tests must pass before commit.

## Phase 8 — Cleanup & docs
- [ ] Keep: `hamlet.txt`, `artifacts/*` (clean corpus, vocab, best model, history), modules, tests, updated README/requirements.
- [ ] Delete only confirmed-unused: `__pycache__`, `.ipynb_checkpoints`, stale `*.h5`/`*.pickle` if any appear (none currently). Keep `experiemnts.ipynb` as record (note typo, don't rename).
- [ ] `.gitignore`: venvs, caches, `runs/`, large `*.pt`? (commit small best.pt ~5-15MB? GitHub warns >50MB; ours ~4-10MB so commit is fine — document decision).
- [ ] `README.md`: objective, data, char-LM explanation, arch, preprocessing/splits, install (CPU+GPU), train/eval/generate commands, artifacts, real metrics, limitations.

## Phase 9 — GitHub
- [ ] `git status`, stage only intended files, review `git diff --cached`, commit with descriptive message, push to `main` only if authorized + network works; else print exact remaining commands + any auth error verbatim.

## Phase 10 — Final report
- [ ] 12-point report per task spec with actual numbers, device, samples, tests, git status, limitations.

## Checkpoint protocol
- After each phase: run relevant tests, record real outputs, stop-and-report on failure rather than fabricating.
- Never claim GPU use without `torch.cuda.is_available()==True` + device string in training log.
