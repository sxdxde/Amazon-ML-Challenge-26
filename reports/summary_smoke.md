# Entity resolution run summary

Mode: SMOKE. Embeddings: mlx-community/bge-m3-mlx-fp16 (cls+L2, fp16, PCA 256), fields ['name', 'addr', 'combo']. MLP backend: mlx.
Decisions: ONE_TO_ONE=True, SAME_COUNTRY=True. Final rule: threshold (w_lgb=0.3, thr=0.75), OOF macro-F0.5 0.99678.

## CV macro-F0.5 per model (OOF, best global threshold)
| model | oof_macroF05_best_threshold |
|---|---|
| lgb | 0.99551 |
| mlp | 0.99635 |
| blend | 0.99697 |

## Decision variants (OOF, cross-fitted isotonic)
| variant | group | n_s1 | macro_F0.5 | singleton_acc | nonsingleton_F0.5 | threshold |
|---|---|---|---|---|---|---|
| threshold | all | 300 | 0.99678 | 1.0 | 0.99661 | 0.75 |
| threshold | India | 120 | 0.99548 | 1.0 | 0.99524 | 0.75 |
| threshold | US | 180 | 0.99765 | 1.0 | 0.99753 | 0.75 |
| threshold+assign | all | 300 | 0.99678 | 1.0 | 0.99661 | 0.75 |
| threshold+assign | India | 120 | 0.99548 | 1.0 | 0.99524 | 0.75 |
| threshold+assign | US | 180 | 0.99765 | 1.0 | 0.99753 | 0.75 |
| expectedF | all | 300 | 0.99678 | 1.0 | 0.99661 | nan |
| expectedF | India | 120 | 0.99548 | 1.0 | 0.99524 | nan |
| expectedF | US | 180 | 0.99765 | 1.0 | 0.99753 | nan |
| expectedF+assign | all | 300 | 0.99678 | 1.0 | 0.99661 | nan |
| expectedF+assign | India | 120 | 0.99548 | 1.0 | 0.99524 | nan |
| expectedF+assign | US | 180 | 0.99765 | 1.0 | 0.99753 | nan |

## Leave-one-country-out (France proxy; raw probabilities)
| model | train_on | eval_on | auc | pr_auc | macroF05_thr0.5 | macroF05_expectedF |
|---|---|---|---|---|---|---|
| lgb | US | India | 0.99942 | 0.99768 | 0.99139 | 0.98976 |
| lgb | India | US | 0.99994 | 0.99978 | 0.99637 | 0.99637 |
| mlp | US | India | 0.99974 | 0.99887 | 0.97791 | 0.97966 |
| mlp | India | US | 0.99995 | 0.99978 | 0.98495 | 0.98495 |

## Blocking (train sample)
| union_pair_recall | s1_full_recall | cands_per_S1_train | retrievers |
|---|---|---|---|
| 1.0 | 1.0 | 19.4 | char-tfidf + word-tfidf + dense(MLX) + rare/postal |

## Test candidates
| group | S1 | S2uS3 | cands_per_S1 | S1_without_cands |
|---|---|---|---|---|
| France | 30 | 7645 | 26.1 | 0.0 |
| India | 94 | 24990 | 27.8 | 0.0 |
| US | 77 | 16327 | 26.9 | 0.0 |

## Test predictions
| country | n_S1 | cands_per_S1 | matches_per_S1 | empty_rate |
|---|---|---|---|---|
| India | 94 | 27.79 | 0.436 | 0.766 |
| US | 77 | 26.86 | 0.182 | 0.8571 |
| France | 30 | 26.07 | 2.967 | 0.2333 |

## Wall time and memory per section
| section | wall_s | rss_gb | peak_rss_gb |
|---|---|---|---|
| 1 environment | 1.2 | 0.2 | 0.21 |
| 1a parity | 5.4 | 0.2 | 0.21 |
| 1 environment | 0.9 | 0.22 | 0.22 |
| 1a parity | 18.2 | 0.16 | 0.22 |
| 1 environment | 1.1 | 0.18 | 0.22 |
| 1a parity | 22.5 | 0.19 | 0.22 |
| 1b benchmark | 95.0 | 0.28 | 1.6 |
| 2 load | 42.6 | 5.49 | 6.02 |
| 3 EDA | 12.9 | 6.01 | 6.91 |
| 4 normalise | 2.5 | 6.04 | 6.91 |
| 5 metric | 0.0 | 6.04 | 6.91 |
| 6 embeddings | 638.4 | 0.42 | 6.91 |
| 7 blocking | 2.5 | 0.57 | 6.91 |
| 8 features | 0.9 | 0.66 | 6.91 |
| 9 CV splits | 0.1 | 0.67 | 6.91 |
| 10 LightGBM | 3.0 | 0.67 | 6.91 |
| 11 MLP | 3.3 | 0.65 | 6.91 |
| 12 blend/calibrate/decide | 0.1 | 0.61 | 6.91 |
| 13 test inference + write | 105.8 | 2.15 | 6.91 |

No external data was used: no geocoding, APIs, registries, web lookups or external datasets; only pretrained bge-m3 weights and hand-written dictionaries.