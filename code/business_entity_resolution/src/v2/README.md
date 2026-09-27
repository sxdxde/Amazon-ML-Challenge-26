# v2 — "cache once, iterate fast" pipeline

v2 reuses v1's expensive, finished work (normalised records, bge-m3 embeddings) and moves all
retrieval to the GPU. Every stage caches its output under `artifacts_v2/<profile>/` with a
`manifest.json` (config hash, counts, wall time, peak RSS/VRAM) and is skipped when the manifest matches.

## Reproduction chain (raw data -> submission)

All commands from `code/business_entity_resolution/`, with the project venv (`../../.venv`).

| # | Stage | Command | Produces | Full-run time (A100 40GB shared, 16 CPU jobs) |
|---|---|---|---|---|
| v1-a | prepare (v1) | `python src/run_all.py --profile full --stage prepare` | `artifacts/full/prepare/{train,test}_records.parquet`, `train_truth.parquet` | ~6 min |
| v1-b | embed (v1) | `python src/run_all.py --profile full --stage embed` | `artifacts/full/embed/{train,test}.npy` | ~4.9 h |
| 1 | records | `python -m src.v2.run --profile full --stage records` | `artifacts_v2/full/records/…`, `truth_train.parquet`, `folds.parquet` | ~10 min (est.) |
| 2 | lexvec | `python -m src.v2.run --profile full --stage lexvec` | `artifacts_v2/full/lexvec/{split}_{name_lex,addr_lex}.npy` | ~30 min (est.) |
| 3 | search | `python -m src.v2.run --profile full --stage search` | `artifacts_v2/full/cand/{split}/country=*/part.parquet`, `reports/v2_recall.md` | ~1–2.5 h (est.) |

Stages 4–9 (adapter, select, features, model, xenc, decide) are added next.

### Embeddings reused from v1 (never recomputed by v2)
`src/embed.py` (v1): model `BAAI/bge-m3` (MIT), **CLS pooling + L2 normalisation**, fp16 on CUDA,
max_len 96, text = raw `"{name} | {address}"` (just `name` when the address is empty), length-sorted
batches of 512, written in 200k-row shards to `artifacts/full/embed/{split}.npy` (float16, 1024-d).
Row *i* of the memmap = row *i* (`uid`) of `artifacts/full/prepare/{split}_records.parquet`.
Audit (2026-09-26): train (12,527,040 × 1024), test (11,702,133 × 1024); sampled norms 0.9999–1.0001.

## Stages 1–3

* **records** — reads v1's normalised tables (transliteration of Brahmic scripts, NFKD accent folding,
  casefold, `&`→and, EN/IN/FR legal-suffix and address-abbreviation dictionaries; no unidecode) and writes
  per split/country parquet ordered by (country, src, uid) so each country's S1 and pool are contiguous
  `pos` ranges; adds name/address token lists and digit sets; maps true pairs to positions; assigns
  S1-grouped folds (`folds.parquet`, 5 folds, stratified by country) reused by every later stage;
  flags v1's exact 150k tuning subsample for an apples-to-apples recall reference.
* **lexvec** — char_wb 3–4-gram hashing (2^20), sublinear tf × IDF from the same split+country pool
  (n-grams in > 0.5% of pool docs zeroed), fixed-seed Gaussian projection to 256-d, L2, float16.
* **search** — GPU exact top-50 per retriever (`sem` = bge-m3, `name_lex`, `addr_lex`), reverse top-10
  (pool → S1) for `sem` and `name_lex`, CPU key joins (postal+street number, street number+street word,
  rare name token), union, and the exact cosine of every dense retriever for every union pair.

## Profiles and resources
* `--profile dev`: 20k S1 per split (all countries) + their true matches + 150k random pool per country.
* Defaults: `n_jobs=16`, `vram_gb=10` (tiles and chunks are sized from it; the stage refuses to start
  if less than 90% of it is free). Override with `--set vram_gb=8 n_jobs=16`. Run under `nice -n 10`.

## Licenses
| Component | License |
|---|---|
| BAAI/bge-m3 (embeddings, via v1) | MIT |
| torch, numpy, scipy, scikit-learn, pandas, pyarrow, joblib | BSD / Apache-2.0 |
| sparse_dot_topn (dev validation only) | Apache-2.0 |
| XGBoost (stage 7, CPU build on this host) | Apache-2.0 |
