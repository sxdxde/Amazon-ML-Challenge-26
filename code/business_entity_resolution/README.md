# Business Entity Resolution — Amazon ML Challenge 2026

Blocking (multi-retriever union) → pairwise features → stage-1 GBM (also the last blocking filter: cascade)
→ cross-encoder on the uncertainty band → stacked GBM with self-training for unseen countries → calibrated
expected-F0.5 set selection with one-S1-per-record assignment. GBM backend: LightGBM when installable, otherwise
XGBoost (the final submission was produced with XGBoost 2.1.4 on a glibc-2.17 A100 host).
Everything is driven by `src/run_all.py`; every stage caches its artifacts and is resumable.

## 0. Final submission configuration (what produced `output/`)

The defaults in `src/config.py` (profile `full`) are exactly the configuration of the final submission:

| Stage | Setting |
|---|---|
| `embed` | `BAAI/bge-m3` (MIT), CLS pooling, fp16, max_len 96, text `"{name} \| {address}"` |
| `block` | 11 retrievers per country string (dense + 6 TF-IDF + 4 key blocks), greedy-tuned k, RRF union, cap 80 → **38.7 blocked candidates / S1** (train pair-recall 0.9803) |
| `features` | 96 pairwise + context features |
| `gbm` | stage-1 XGBoost, 5 S1-grouped folds |
| `xenc` | `BAAI/bge-reranker-v2-m3` (Apache-2.0) fine-tuned per fold on the stage-1 uncertainty band (`xenc_band_lo=0.01`, `xenc_band_hi=0.998`, `xenc_max_per_s1=8`) |
| **cascade** (`prune.py`) | the stage-1 GBM is the last blocking filter: only candidates with stage-1 rank < `prune_k=15` and p1 ≥ `prune_min_p1=0.003` reach the final matcher → **5.16 candidates / S1 on test** (8.9 M pairs instead of 65.9 M; `candidate_pairs.tsv`); train OOF macro F0.5 unchanged (0.98773) |
| `gbm2` | stacked XGBoost (108 features) + **self-training for countries absent from train** (`pseudo_hi=0.97`, `pseudo_lo=0.03`; the unseen countries are derived from the data, nothing is hard-coded) |
| `decide` | cross-fitted isotonic calibration → one-S1-per-record assignment → expected-F0.5 set selection (empty set allowed = singleton) |
| `predict` | writes `output/matching_results.tsv` + `output/candidate_pairs.tsv`, internal checks + official validator |

Reproduce end to end (from the challenge root, on a CUDA machine):

```bash
python code/business_entity_resolution/src/run_all.py --profile full --device auto
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Wall-clock on a shared A100-40GB + 48 CPU threads: prepare 6 min, embed ~5 h, block ~15 h, features 1.5 h,
gbm 1.6 h, xenc ~7.6 h, gbm2 ~1.5 h (two passes because of self-training), decide 15 min, predict 8 min.
Every stage caches its artifacts under `artifacts/full/<stage>/` and is resumable
(`--from-stage <stage> --force` recomputes a stage and everything after it).

Train OOF (5-fold, grouped by Source-1 entity, 2.2 M S1): macro F0.5 **0.9877**. Public leaderboard 0.981.

**Reproducibility note.** `output/matching_results.tsv` is byte-identical to the file scored on the leaderboard.
Re-running `decide` + `predict` from the saved stage artifacts with this code reproduces it exactly except for
2 of 1,732,544 rows (the expected-F0.5 choice for two entities shifts once near-zero-probability candidates are
pruned); every leaderboard match lies inside `output/candidate_pairs.tsv` (validator: PASS with `--check-ids`).
GPU training (cross-encoder) and multi-threaded XGBoost are not bit-deterministic across machines, so a full
retrain reproduces the scores up to such small differences.


## Layout

```
code/business_entity_resolution/
  src/
    config.py      paths, model names, k grids, seeds, profile overrides (full / dev / smoke)
    io_utils.py    TSV I/O, logging, stage markers, submission writers + internal validator
    eda.py         Phase-1 EDA -> reports/eda.md
    normalize.py   transliteration, accent folding, legal/street dictionaries, address parsing
    prepare.py     stage `prepare`: load, (sub)sample, normalise -> parquet
    embed.py       stage `embed`: dense embeddings of "{name} | {address}"
    blocking.py    stage `block`: candidate generation, k tuning, recall diagnostics
    features.py    stage `features`: pairwise + context features
    split.py       S1-grouped hash folds, LOCO helpers
    metric.py      macro F0.5 exactly as defined
    train_gbm.py   stages `gbm` (stage 1 + LOCO) and `gbm2` (stacked)
    train_xenc.py  stage `xenc`: cross-encoder fine-tuning on the stage-1 uncertainty band
    prune.py       cascade: stage-1 GBM as the last blocking filter -> final candidate set
    decide.py      stage `decide`: calibration, assignment, expected-F0.5 selection
    predict.py     stage `predict`: output TSVs + validation
    run_all.py     orchestrator (--smoke, --profile, --stage, --from-stage, --force, --set k=v)
  tests/           metric + decision-rule unit tests
  requirements.txt pinned (torch installed per platform by the scripts)
```

All commands below run from the challenge root (the folder containing `dataset/` and `utils/`).
Artifacts go to `artifacts/<profile>/`, reports to `reports/`, logs to `logs/`.

## 1. Local smoke test (CPU, ~2–5 min + venv install)

```bash
bash scripts/smoke_test.sh
```

Creates `.venv_smoke`, installs the pinned requirements, runs the unit tests, runs
`run_all.py --smoke` on CPU (300 train S1 + their matches + 6k distractors; 150 test S1 stratified by
country, so France is included), validates `output_smoke/` with the official validator, writes a
full-size all-empty submission through the real writer path and validates it against the real
test set, and checks the smoke OOF score beats the all-empty baseline.

## 2. Upload to the remote machine (run on your Mac)

```bash
bash scripts/upload_to_remote.sh      # REMOTE=user@host REMOTE_DIR=amazon_ml_2026 to override
```

Creates `~/amazon_ml_2026` on the remote and rsyncs `dataset/ utils/ code/ scripts/ docs/ reports/`
and the documentation template (no venvs, artifacts or models — those are created on the remote).
Re-running only sends changed files (use it after every local code fix).

## 3. Remote setup + start (one command, on the remote)

```bash
cd ~/amazon_ml_2026 && bash scripts/start_remote.sh
```

First run: picks an existing Python ≥ 3.10 (or `PYTHON=/path`), finds a system CA bundle that works
through HTTPS-intercepting proxies (or `CA_BUNDLE=/path`; last resort `ALLOW_INSECURE_DOWNLOADS=1`),
runs `scripts/remote_setup.sh` (`.venv`, torch 2.7.1 with the `cu118` wheel auto-selected for NVIDIA
drivers < 525 such as 470/CUDA 11.4, pinned requirements, CUDA/bf16 check), downloads the model
weights to `hf_cache/` (`scripts/download_models.py`), checks the GPU has ≥ 16 GB free (it may be
shared), then launches the full run in the background (offline w.r.t. the model hub).
Re-running resumes. `GPU=0` selects the nvidia-smi GPU index.

## 4. Full run

```bash
bash scripts/run_full.sh                  # background (nohup), log -> logs/full_<timestamp>.log
tail -f logs/full_*.log
# or attached, e.g. inside tmux:
FOREGROUND=1 bash scripts/run_full.sh
```

Resumable: re-running skips every finished stage. Useful variants:

```bash
bash scripts/run_full.sh --stage decide --force          # recompute one stage
bash scripts/run_full.sh --from-stage gbm --force        # recompute gbm and everything after
bash scripts/run_full.sh --set use_xenc=False            # skip the cross-encoder (gbm2 still runs)
bash scripts/run_full.sh --set xenc_test_models="'first'" # 3× faster cross-encoder test scoring
```

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`; the predict stage runs
the official validator and fails loudly unless it prints PASS.

### Expected runtimes on the A100 (fill in after the first run)

| stage | what | expected | measured |
|---|---|---|---|
| prepare | load + normalise 12.5M train + 11.7M test records | 5–10 min | |
| embed | bge-m3 fp16 on ~24M texts | 60–120 min | |
| block | tuning pass + train/test retrieval (GPU kNN, CPU sparse top-n) | 60–120 min | |
| features | ~40M train + ~60M test pairs | 40–90 min | |
| gbm | 5 folds + LOCO + test | 45–90 min | |
| xenc | 3 folds fine-tune + OOF + test scoring | 2–4 h | |
| gbm2 | 5 folds + test | 45–90 min | |
| decide / predict | calibration, selection, writing, validation | 10–20 min | |

## 5. Reports

`reports/eda.md` (EDA), `reports/blocking.md` (recall per retriever/union/country, reduction ratio,
k-tuning trace, test candidate stats), `reports/gbm.md` / `reports/gbm2.md` (AUC, PR-AUC, per-country,
LOCO, feature importance, cross-encoder standalone vs stacked), `reports/decision.md` (macro F0.5 per
decision variant, per country, singleton accuracy, non-singleton F0.5), `reports/predict.md`.

## 6. Build the submission zip

```bash
bash scripts/package_submission.sh <team_name>   # -> <team_name>_submission.zip
```

Contains `output/` (both TSVs), `code/business_entity_resolution/` and `Documentation_template.md`
(fill it in first; running notes are in `docs/methodology_notes.md`, model licenses in `docs/models.md`).

## Rules compliance

* No external data: only the provided train/test files; hand-written dictionaries in `normalize.py`.
* Models: MIT / Apache-2.0, ≤ 8B parameters (see `docs/models.md`).
* Country is an open set: blocking groups by the country string; country is never a model feature.
* No GPL dependencies (no `unidecode`; NFKD accent stripping + a Brahmic transliteration table).
