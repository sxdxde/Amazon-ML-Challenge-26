# Experiments

One-off scripts run on the A100 host during the challenge, kept for the record. They import the pipeline from
`code/business_entity_resolution/src` and use absolute paths: replace `/path/to/amazon_ml_2026` with the project
root before running. Outcomes are summarised below; the ones that improved the submission were folded into
`src/` (`prune.py`, self-training in `train_gbm.run_stage2`).

| Script | Question | Outcome |
|---|---|---|
| `error_analysis.py` | Where does the remaining OOF loss come from? | 37.5 % blocking misses, 30.7 % low-confidence in-candidate matches, 20.7 % "predicted nothing", ~8 % false merges |
| `prune_tradeoff.py` | How small can the final candidate set be? | Top-10/15 by stage-1 score with p ≥ 0.003–0.01: 38.7 → 4–5 per entity, F0.5 unchanged → `src/prune.py` |
| `loco_pseudo.py` | Does self-training help an unseen country? (leave-one-country-out) | India→US +0.0016, US→India +0.0105 → self-training in `train_gbm.run_stage2` |
| `gbm2_pseudo.py`, `chain_pseudo.sh` | Stage 2 refit with pseudo-labelled unseen-country pairs | Used for the final leaderboard submission (0.981) |
| `romanization_validation.py` | Embed normalised/romanised text instead of raw? | Better true-vs-random separation (India 0.393 → 0.477, US 0.474 → 0.534) |
| `embed_norm.py`, `fast_reblock.py`, `chain_rom.sh` | Re-embed normalised text; dense-only re-blocking that reuses lexical hits | Bit-exact reproduction of production blocking verified; full rebuild did not fit in the time left |
| `expansion_probe.py` | Recover blocking misses via neighbours of found matches? | Only 7–18 % reachable, not adopted |
| `stage3.py` | Third-stage "collective" reranker | +0.0002 OOF, −0.001 leaderboard, not adopted |
| `stage3b_gate.py` | Dedicated "has any match" gate | +0.00001 OOF, not adopted |
| `confusability.py`, `quick_checks.py`, `check_recall_headroom.py` | France vs India/US candidate quality, truth multiplicity, recall headroom | No France-specific data-quality gap; recall curve saturates at ≈0.98–0.99 |
| `reblock_france.py`, `refeaturize_france.py`, `repredict_france.py` | Apply a normalisation fix (`cours` → `crs_` bug) to France only | Fixed; no measurable leaderboard change |
| `fix_test_pred_order.py`, `complete_gbm2.py` | Recovery tools after a row-order bug / crash | Operational |
| `loco_variants.py` | Decision-rule variants under leave-one-country-out | Differences ≤ 0.003 between variants |
