# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Convexity  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A cascaded blocking-and-matching pipeline: 11 complementary retrievers (a multilingual dense encoder, character/word TF-IDF and exact key blocks) generate a recall-oriented candidate list per Source-1 entity; a stage-1 gradient-boosted matcher on 96 string/context features then acts as the last blocking filter, so only **5.2 candidates per Source-1 entity** (out of ~38 retrieved) reach the final matcher. The final matcher stacks the stage-1 model with a fine-tuned multilingual cross-encoder (`bge-reranker-v2-m3`) and is adapted to the unseen country by self-training; per-entity match sets are chosen by maximising the *expected* per-entity F0.5 of calibrated probabilities, which handles singletons natively. Train OOF macro F0.5 **0.9877**, public leaderboard **0.981**.

---

## 2. Methodology

### 2.1 Problem Analysis

* **Scale.** Train: 12.5 M records (2.2 M Source-1), test: 11.7 M records (1.73 M Source-1; India 810 k, US 663 k, France 259 k). All-pairs comparison is impossible; every stage is per country string (an open set: countries are never hard-coded, one-hot encoded or used as a feature; *all* matched pairs in train share the country string, so blocking within a country loses nothing).
* **Matches per entity.** Mean ≈ 3.5 true matches per Source-1 entity (mostly 2–5, up to 10+), 5.6 % singletons. Because the metric is a per-entity macro F0.5 with singletons scored 1/0, a single false merge on a singleton or a missed "has matches" entity costs a full point for that entity.
* **Noise.** Names: legal-suffix variants (Pvt/Private, Ltd/Limited, SAS/SARL/EURL…), abbreviations, DBA names, transpositions, typos, and *replaced* names; addresses: abbreviations, reordering, missing PIN/state, landmarks ("Near SBI ATM"), and Indic scripts (Devanagari, Bengali, Gurmukhi, … mixed with Latin).
* **Unseen country.** France never appears in training. Leave-one-country-out (LOCO) experiments (train on one training country, evaluate on the other) showed that the largest degradation under country shift is on *singletons* (singleton accuracy 0.99 in-distribution → 0.79–0.88 under LOCO), which drove the design of the unseen-country adaptation (§4.4).

### 2.2 Solution Strategy

**Approach Type:** Cascaded blocking + gradient-boosted matcher + fine-tuned cross-encoder (stacked) + expected-F0.5 set decision  
**Core Innovation:** (1) a *cascade* in which the stage-1 matcher is the final blocking filter (7.4× smaller candidate set at no loss of F0.5); (2) a cross-encoder applied only to the uncertainty band of the stage-1 model and stacked back; (3) self-training on the unseen country, validated by leave-one-country-out; (4) per-entity *expected*-F0.5 set selection over calibrated probabilities (Poisson-binomial), which decides both the matched set size and "no match".

Pipeline (`src/run_all.py`, every stage cached and resumable):
`prepare → embed → block → features → gbm (stage 1) → xenc (cross-encoder) → gbm2 (stacked, self-trained) → decide → predict`

---

## 3. Candidate Generation (Blocking)

**Normalisation (`normalize.py`).** Unicode NFKC, accent folding, transliteration of Brahmic scripts to Latin, punctuation/whitespace canonicalisation, legal-suffix extraction (Indian, US and French forms), street-type/abbreviation dictionaries, and address parsing into core tokens, postal code, street number, street word and landmark phrases; name *skeletons* (consonant skeletons robust to vowel/transliteration variation).

**Retrievers (`blocking.py`), all computed per country string:**

- **Dense:** `BAAI/bge-m3` (MIT) CLS embeddings of `"{name} | {address}"`, exact inner-product top-k on GPU.
- **TF-IDF (sparse top-n, `sparse_dot_topn`):** `combo_word` (name skeleton + address words, 1–2-grams), `combo_char` (name core + address, char 3–4-grams), `name_char` (for empty/garbled addresses), `addr_char` (for replaced names), plus word retrievers for names and addresses.
- **Key blocks:** street number + street word, postal code, exact normalised name core, name skeleton (blocks larger than a cap are skipped).

The per-retriever depths *k* were tuned greedily on train to maximise pair recall per candidate; hits are unioned, scored by reciprocal-rank fusion and capped at 80 per entity. This gives **38.7 candidates / Source-1 entity with a train pair-recall of 0.9803**. Going to 4× more candidates (k at its maximum, ~150/entity) only raises recall to ≈0.988, i.e. the remaining ~2 % of true pairs are not reachable by any of these signals.

**Cascade — final candidate set (`prune.py`).** The stage-1 matcher (§4) scores every retrieved candidate; only candidates with stage-1 rank < 15 within their entity **and** stage-1 probability ≥ 0.003 are passed to the final matcher. This is the set written to `candidate_pairs.tsv`:

| | candidates / Source-1 entity | train pair-recall ceiling | train OOF macro F0.5 |
|---|---|---|---|
| retrieved (blocking union) | 38.7 (test 38.0) | 0.9803 | 0.98773 |
| **final candidate set (cascade)** | **4.55 (test 5.16)** | ≈0.980 | **0.98773** |

- **Candidate pairs generated:** 8,935,004 test pairs in `output/candidate_pairs.tsv` (5.16 per Source-1 entity; 65,903,597 before the cascade, i.e. 7.4× smaller).
- **How true matches were not lost:** complementary retrievers (a dense multilingual model for transliteration/paraphrase, char n-grams for typos, separate name-only and address-only retrievers for replaced names / garbled addresses, exact keys for PIN and street number); *k* tuned against measured recall; the cascade thresholds are set so that, on train OOF, no pair the final model would have selected is removed (every selected pair already has stage-1 probability well above the floor).

---

## 4. Matching Model

**Features used (96 pairwise, `features.py`)**
- *Name:* rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on normalised and full names, skeleton and no-space variants, token Jaccard, IDF-weighted Jaccard, max shared-token IDF, number of shared tokens, first/last token equality, acronym match, legal-suffix both/equal/conflict, name frequency in the country (generic names), length ratios, non-Latin / domain-name flags.
- *Address:* ratio / partial / token features, Jaccard and IDF-weighted Jaccard, digit-set Jaccard and subset, postal code both/equal, street number both/equal/prefix, street word equality, landmark flags, empty-address flags, length ratios.
- *Retrieval & context:* dense cosine and per-retriever TF-IDF cosines, per-retriever ranks, RRF score, number of candidates of the entity and of the record, rank / gap of the pair's score within its entity and within its record (both directions), mutual-best flag, corroboration across the entity's other top candidates.

**4.1 Stage-1 matcher.** XGBoost (Apache-2.0; hist, loss-guided, 255 leaves), 5 folds grouped by Source-1 entity, early stopping on held-out entities. OOF pair PR-AUC ≈0.999.

**4.2 Cross-encoder.** `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568 M parameters) fine-tuned (1 epoch per fold, 3 entity-grouped folds, ~800 k pairs per fold) on raw `"{name} | {address}"` pairs from the stage-1 *uncertainty band* (stage-1 probability in [0.01, 0.998], top-8 per entity; 2.8 M train pairs, 46 % positive). Out-of-fold scores for train, fold-averaged scores for test. On the band: cross-encoder alone AUC 0.984, stage-1 alone 0.972, stacked 0.992.

**4.3 Stage-2 matcher (stacked).** XGBoost on the 96 features + stage-1 probability and its context (rank/gap/sum/share within entity and record) + cross-encoder score, rank and gap within the entity (108 features), same folds. OOF pair PR-AUC 0.99948, log-loss 0.0041.

**4.4 Self-training for unseen countries.** Countries present in test but absent from train are derived from the data (here: France). A first stage-2 model scores their candidate pairs; pairs with calibrated probability ≥ 0.97 (positive) or ≤ 0.03 (negative) are added as pseudo-labelled rows to *every* fold's training data, and stage 2 is refit (train OOF stays out-of-fold). Validated by LOCO on the stage-1 features before use:

| held-out country | baseline macro F0.5 | + self-training | Δ | singleton accuracy |
|---|---|---|---|---|
| India → US | 0.96403 | 0.96563 | +0.0016 | 0.877 → 0.897 |
| US → India | 0.90890 | 0.91935 | **+0.0105** | 0.785 → 0.804 |

(pseudo-label accuracy 99.7–99.9 %).

**Model type:** XGBoost (stage 1) + fine-tuned cross-encoder + stacked XGBoost (stage 2).

**Threshold selection method (`decide.py`).** No global threshold. Stage-2 scores are calibrated with isotonic regression (cross-fitted by fold on train). Each Source-2/3 record is assigned to at most one Source-1 entity (its best). For each entity, candidates are sorted by probability and the prefix size *k* (including *k = 0*, i.e. "no match") that maximises the **expected per-entity F0.5** is chosen, computing the exact distribution of the number of true matches (Poisson-binomial dynamic programme). This optimises the leaderboard metric directly, including the singleton rule. Four decision variants (global threshold, threshold + assignment, expected-F0.5, expected-F0.5 + assignment) were compared on OOF; expected-F0.5 + assignment was best.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98773** on train out-of-fold (5-fold, grouped by Source-1 entity, all 2.2 M Source-1 entities, singletons included). Public leaderboard **0.981**.

Progression (train OOF → public leaderboard): stage-1 + stacking 0.98284 → 0.953; + cross-encoder (MiniLM) 0.98635 → 0.971; + `bge-reranker-v2-m3` on a wider band 0.98773 → 0.981. The leaderboard gained far more than OOF from the cross-encoder (+0.028 vs +0.005): the semantic model transfers to the unseen country much better than string features.

Where the remaining OOF loss (1 − 0.98773) comes from:

| source | share of loss |
|---|---|
| true matches never retrieved by blocking | 37.5 % |
| true matches retrieved but not selected (low confidence, median calibrated p 0.28) | 30.7 % |
| entities with matches predicted as "no match" | 20.7 % |
| false merges (all types) | ≈ 8 % (removing *all* would add only +0.0009) |

- **Common false positives (wrong merges):** branches/outlets of the same chain or group with the same name and a nearby/similar address; generic names ("… Enterprises", "… Services", "Lycée …") in the same town; singletons whose name and address closely resemble another business.
- **Common false negatives (missed matches):** records with a replaced/trade (DBA) name *and* a heavily rewritten address, landmark-only addresses, and cross-script records where both name and address differ in script; many of these are never retrieved (blocking ceiling ≈ 98 %).

---

## 6. Conclusion

Recall-oriented multi-retriever blocking plus a cascade (stage-1 matcher as the last filter) gives a compact candidate set (5.2 per entity) without losing accuracy; a fine-tuned multilingual cross-encoder on the uncertainty band was the single largest improvement and, notably, the component that transfers best to the unseen country; per-entity expected-F0.5 decisions handle singletons and variable match counts directly. Main lesson: under country shift, OOF gains on the training countries are a poor predictor of test gains — leave-one-country-out validation was the reliable guide (it rejected several OOF-positive ideas and validated self-training).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — all source in `src/`, `README.md` (exact reproduction steps and timings), `requirements.txt` (pinned), `tests/` (metric and decision-rule unit tests).

Entry point (from the challenge root):

```bash
python code/business_entity_resolution/src/run_all.py --profile full --device auto
```

writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator. Modules: `normalize.py`, `prepare.py`, `embed.py`, `blocking.py`, `features.py`, `train_gbm.py` (stage 1, stage 2 + self-training, LOCO), `train_xenc.py`, `prune.py` (cascade), `decide.py`, `predict.py`, `metric.py`, `split.py`, `config.py`, `io_utils.py`, `gbm_backend.py`, `run_all.py`.

Models and licences: `BAAI/bge-m3` (MIT, 568 M), `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568 M), XGBoost (Apache-2.0). No external data, APIs or lookups; only the provided training/test files.

### B. Additional Results

- **Decision variants (OOF macro F0.5):** threshold 0.98767, threshold + assignment 0.98767, expected-F0.5 0.98772, expected-F0.5 + assignment 0.98773.
- **Candidate-set size vs. accuracy (train OOF):** stage-1 top-10 & p ≥ 0.01 → 4.1 candidates/entity, F0.5 0.98773; top-6 → F0.5 0.98635; top-4 → 0.97050 (many entities have 5–7 true matches, so the per-entity cap must stay ≥ 10).
- **Ideas tested and not adopted:** a third-stage "collective" reranker (candidate-to-candidate similarity; +0.0002 OOF but −0.001 on the leaderboard); a dedicated "has any match" gate (+0.00001); expanding candidates through dense neighbours of already-found matches (only 7–18 % of missed matches reachable).
- **Finding for future work:** embedding the *normalised/romanised* text instead of the raw text improves true-vs-random cosine separation substantially (India 0.393 → 0.477, US 0.474 → 0.534) and ranks missed matches ~5× closer to found ones; rebuilding the pipeline on those embeddings did not fit in the challenge time.
