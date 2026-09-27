# Amazon ML Challenge 2026: Business Entity Resolution (Team Convexity)

Match every Source-1 business record to its Source-2 / Source-3 duplicates across noisy, multilingual data
(India, US, and an unseen country, France, in test). Metric: per-entity macro F0.5 (singletons included).

**Public leaderboard 0.981 · train out-of-fold macro F0.5 0.9877 · 5.16 candidates per Source-1 entity.**

## Approach

```
prepare → embed → block → features → gbm (stage 1) → xenc (cross-encoder) → gbm2 (stacked, self-trained) → decide → predict
```

1. **Normalisation:** transliteration of Indic scripts, accent folding, legal-suffix / street-type dictionaries,
   address parsing (postal code, street number, landmarks), name skeletons.
2. **Blocking (per country string, open set):** 11 retrievers: `bge-m3` dense embeddings, character/word
   TF-IDF (name, address, combined), and exact key blocks (postal, street number, name core/skeleton), greedy-tuned
   depths, reciprocal-rank fusion. That gives 38.7 candidates per entity at 0.980 pair recall.
3. **Stage-1 matcher:** XGBoost on 96 string / retrieval / context features, 5 entity-grouped folds.
4. **Cascade:** the stage-1 matcher is the last blocking filter (top-15 per entity, p ≥ 0.003). Only
   **5.16 candidates per entity** reach the final matcher (`candidate_pairs.tsv`), 7.4× fewer, with OOF F0.5 unchanged.
5. **Cross-encoder:** `BAAI/bge-reranker-v2-m3` fine-tuned on the stage-1 uncertainty band. This was the largest single
   gain (leaderboard 0.953 → 0.981) and transfers best to the unseen country.
6. **Stacked matcher + self-training:** XGBoost on 108 features. Test pairs of countries absent from train are
   pseudo-labelled (confident only) and added to every fold, validated by leave-one-country-out (+0.0016 / +0.0105).
7. **Decision:** isotonic calibration, then one Source-1 entity per record, then per-entity *expected*-F0.5-optimal match set
   (Poisson-binomial), which also decides "no match" for singletons.

Full write-up: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Repository layout

| Path | What |
|---|---|
| `code/business_entity_resolution/` | **Final pipeline** (`src/run_all.py` orchestrates all stages; README with exact reproduction steps; pinned requirements; unit tests). `src/v2/` is an experimental GPU-retrieval rewrite (stages records / lexvec / search). |
| `docs/` | Methodology, model licences, design notes |
| `reports/` | Stage reports (EDA, blocking recall, GBM metrics, decision variants) |
| `scripts/` | Remote setup / upload / run helpers, smoke test, offline bundle tools |
| `experiments/` | One-off analyses and experiments run during the challenge (see its README) |
| `entity_resolution.ipynb`, `er_*.py`, `setup_mac.sh`, `RUN.md`, `requirements.txt` | Apple-Silicon (MLX) notebook version of the pipeline for local runs |

## Reproduce

Place the challenge `dataset/` and `utils/` next to `code/` (they are not in this repo), then:

```bash
python code/business_entity_resolution/src/run_all.py --profile full --device auto
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Details, timings and the smoke/dev profiles are in [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Models and data

`BAAI/bge-m3` (MIT), `BAAI/bge-reranker-v2-m3` (Apache-2.0), XGBoost (Apache-2.0). Only the provided challenge data
is used: no external data, APIs or lookups. The dataset, model weights and generated outputs are not part of this repository.
