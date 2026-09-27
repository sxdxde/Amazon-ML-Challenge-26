# Methodology notes (running; feeds Documentation_template.md)

_Last updated: 2026-09-25. Numbers marked **TBD** come from the full A100 run (`reports/*.md`)._

## 0. Statement on data use
No external data of any kind is used: no geocoding, address APIs, registries, web lookups or
external datasets. The only inputs are the provided train/test TSVs. Pretrained model weights
(MIT / Apache-2.0, ≤ 8B params, see `docs/models.md`) only encode the provided text. All
normalisation dictionaries are hand-written in `src/normalize.py` (listed in §3).

## 1. EDA findings (reports/eda.md)
* Scale: train 2.21M S1 / 5.03M S2 / 5.29M S3; test 1.73M S1 (India 810k, US 663k, **France 259k**) / 4.89M S2 / 5.08M S3.
* Singletons only **5.58%**; non-singletons have 3.67 matches on average (max 11). 85% of matched S1 have
  both S2 and S3 matches; S1s commonly have >1 match from the same source (source-internal duplicates).
* **One-to-many holds**: 0 of 7.64M matched S2/S3 ids appear under more than one S1 → exclusive
  assignment is valid.
* **Matches never cross the country string** (100.0000%) → blocking is restricted to the same country string.
* All S1 ids appear in the ground truth; no id / row-order leakage (corr ≈ 0).
* ~26% of S2/S3 records match no S1 (distractors).
* Multi-script names: ~22% of India S2 names and ~23% of India addresses (mostly state names) are in
  Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Gurmukhi; S1 is always Latin.
  ~6–7% of US S2/S3 names and ~24% of France S2/S3 names carry injected accents.
* ~1.8% of true pairs have a random replacement name ("Nexnovi") with an exact address → address-only
  evidence must be able to carry a match.
* Identical normalised names match only 8% of the time (generic names like "Physical Therapy" repeat
  hundreds of times) → address is the key discriminator.
* Street numbers are perturbed in true pairs (4838→483, 2737→273): digits dropped/changed → fuzzy
  number features, not hard vetoes. Postal codes are essentially absent (<1.2% of rows).
* State representation differs by source (OH vs Ohio vs native script); France uses R./RTE/N°.
* Label quality: 4 / 400k sampled true pairs look suspicious (all fake-name + partial-address cases,
  plausibly correct) → labels treated as clean.

## 2. Validation protocol
* Main CV: 5 folds grouped by S1 (fold = md5(seed, S1 id) mod 5); every candidate of an S1 is in one fold.
* Stage-1 fold models are *fit* on a subsample of in-fold S1 groups (`gbm_fit_s1_frac`) with early
  stopping on a disjoint 10% of in-fold groups, but *predict* every held-out pair → OOF for every S1.
* Leave-one-country-out (train US → eval India and reverse) as the proxy for unseen France.
* All thresholds / decision choices come from OOF; calibration is cross-fitted for OOF evaluation.

## 3. Normalisation (src/normalize.py) — hand-written dictionaries
* Brahmic → Latin transliteration: one Devanagari-offset table applied to every ISCII-layout block
  (U+0900–U+0D7F), inherent-vowel handling, virama, nukta, word-final schwa deletion, a few script
  digraphs (Malayalam റ്റ, Tamil ஃப). Zero-width characters removed.
* NFKD + combining-mark stripping + casefold (no unidecode); & / " + " → "and"; punctuation → space.
* Legal forms (EN/IN/FR + romanised Indic spellings) → canonical codes, extracted from anywhere in
  the name (sources move "LLC"/"Private"/"SARL" to the front); name core = name minus legal tokens,
  minus the record's own country word ("(India)", "(France)") and stopwords.
* Sorted-token name, consonant-skeleton name (phonetic unification: ph→f, ck→k, sh→s, …).
* Street vocabulary (EN/IN/FR) → canonical short forms (road/rd, saint/st, rue/r, allée, impasse/imp,
  route/rte, nagar/ngr, marg, …); US and Indian state names (incl. romanised native spellings) → codes.
* Address components: digit set, postal-like token (5–6 digits or "ddd ddd", not the leading number),
  street number, street word, landmark flag (near/opp/behind/beside/adj/next/off).

## 4. Candidate generation (src/blocking.py)
Per country string, union of retrievers from each S1 into S2∪S3:
dense bge-m3 kNN on "{name} | {address}" (exact, torch GPU matmul — replaces FAISS, same result),
char-wb 2–4-gram TF-IDF on names, char-wb 3–4-gram TF-IDF on addresses, word TF-IDF on name
skeletons, word 1–2-gram TF-IDF on addresses (sparse top-n via sparse_dot_topn, Apache-2.0), and
exact keys (street number + street word, postal, core name, skeleton) with block-size cap.
Per-retriever k tuned greedily on a train S1 subsample (best marginal recall per added candidate)
until union pair recall ≥ 98.5% or the mean-candidate budget; union capped per S1 by reciprocal-rank
fusion. The final union is written unchanged to candidate_pairs.tsv and every pair is scored by the
stage-1 GBM.
* Tuned k, recall per retriever / union / country, reduction ratio: **TBD** (reports/blocking.md).

## 5. Features (src/features.py) — country is never a feature
Name: rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on core and full names,
skeleton ratios, no-space ratios (domain-style names), token Jaccard, IDF-weighted Jaccard, max IDF of
shared tokens, first/last token equality, legal both/equal/conflict, acronym, length ratio, name
genericity (log frequency within country), script flags, domain flag, blocking cosines (dense + 4 TF-IDF).
Address: ratio / partial / token-sort / token-set / Jaro-Winkler, (weighted) Jaccard, postal both/equal,
street number equal / fuzzy ratio / prefix-suffix, street word equal/ratio, digit-set Jaccard /
shared / subset, landmark flags, empty-address flag, lengths.
Context: source, per-retriever ranks + RRF, #candidates, rank of the pair within its S1 and within its
record (reverse), gaps to the best other candidate both ways, mutual-best flag, corroboration with the
S1's other top candidates (name/address similarity).

## 6. Models
* Stage 1: LightGBM binary (OOF AUC / PR-AUC per country: **TBD**).
* Cross-encoder: `BAAI/bge-reranker-v2-m3` fine-tuned (BCE, bf16, 3 S1-grouped folds, early stopping)
  on the stage-1 uncertainty band [0.02, 0.995], top-6 per S1 — a cascade inside the matcher (scoring
  ~60M test pairs with a 568M model is infeasible). Standalone vs stacked on the band: **TBD**.
* Stage 2: stacked LightGBM = stage-1 features + stage-1 OOF prob + probability-context (rank / gap /
  sum / share within S1 and within record) + cross-encoder logit (NaN outside the band).

## 7. Decision rule (src/decide.py)
Isotonic calibration → exclusive assignment (each S2/S3 kept only for its best S1) → per S1, choose
k maximising exact expected F0.5 of the top-k (Poisson-binomial over independent Bernoulli
candidates; E[F_0] = Π(1-p)). Compared against a global threshold with and without assignment;
the best OOF variant is applied to test. Results per country / singleton accuracy / non-singleton
F0.5: **TBD** (reports/decision.md).

## 8. Deviations from the original plan (and why)
* FAISS → exact torch matmul kNN on GPU (exact, no extra dependency, ~minutes per country).
* Postal-code blocking demoted (codes <1.2% present); street-number + street-word key instead.
* Sparse top-n runs single-threaded inside forked workers: several OpenMP runtimes (torch, sklearn,
  sparse_dot_topn) in one process deadlocked on macOS.
* Cross-encoder is a cascade on the uncertainty band; `candidate_pairs.tsv` is the set scored by the
  stage-1 GBM (all blocking candidates), of which the cascade band is an internal subset.
* Full profile featurises 50% of train S1 (`feat_s1_frac`, RAM guard); blocking and context features
  use all train S1, so rank/competition features see realistic density.

## 9a. Local dev-profile sanity numbers (NOT the submission; 20k train S1, pool subsampled to 10%,
## e5-small embedder, no cross-encoder — optimistic because confusers are 10× sparser)
* Blocking (tuned k: dense 20, combo_word 15, combo_char 10, name_char 5, …): pair recall India 98.96%,
  US 99.67% at ~40 candidates/S1. combo_word (name skeleton + address tokens) was the key addition:
  R@10 0.98 US / 0.95 India vs 0.73 / 0.62 for name-only char TF-IDF.
* Stage-1 GBM OOF pair AUC 0.9998, PR-AUC 0.9987.
* OOF macro F0.5 = **0.9828** (expectedF chosen; India 0.9797, US 0.9848); threshold 0.9823.
* LOCO: India→US 0.9747, US→India 0.9547 (India is the harder direction: scripts, long addresses).
* Loss breakdown: missed matches ~0.9%, false positives ~0.5%, blocking misses 0.4% (India) /
  0.1% (US), singleton false positives ≤ 0.13%.

## 9. Results (fill from reports after the full run)
| metric | value |
|---|---|
| union pair recall (train, per country) | TBD |
| mean candidates / S1 | TBD |
| stage-1 OOF AUC / PR-AUC | TBD |
| final OOF macro F0.5 (all / India / US) | TBD |
| LOCO F0.5 (US→India, India→US) | TBD |
| singleton accuracy / non-singleton F0.5 | TBD |
