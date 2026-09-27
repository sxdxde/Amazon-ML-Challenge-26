# v2 methodology notes (feeds Documentation_template.md)

## Data facts verified (stage 0 audit, 2026-09-26)
- Train: India 883,188 S1 / 4,133,346 pool; US 1,323,633 / 6,186,873. Test: India 809,986 / 4,717,565;
  US 663,106 / 3,817,031; France 259,452 / 1,434,993.
- 7,638,365 true pairs; no S2/S3 record is matched to more than one S1 (one-to-one assignment is valid);
  every matched pair shares the country string (retrieval is grouped by the country string; the set of
  countries is read from the data, so unseen countries such as France flow through unchanged).

## Blocking (candidate generation)
- Dense semantic retriever: bge-m3 (MIT) CLS embeddings of the raw "name | address" text, exact inner
  product top-k on the GPU per country (no approximate index).
- Dense lexical retrievers replace CPU TF-IDF: hashed char 3–4-gram TF-IDF (IDF from the same country's
  pool, very common n-grams dropped) randomly projected to 256-d. On the dev sample this matched exact
  TF-IDF retrieval within ±1.6 recall points per field (Spearman 0.92–0.96 with the exact cosine) and gave
  the same union recall (India 0.9717 / US 0.9900 at k=10 with bge-m3).
- Key retrievers: exact (postal, street number), exact (street number, street word), rare shared name token.
- Reverse search (candidate → S1) gives mutual-rank signals.
- Dev (20k S1, 150–190k pool per country): union recall 0.9843 at 25.6 candidates/S1 (k=10),
  0.9712 at 12.1 (k=5). Full-scale numbers: reports/v2_recall.md (pending).

## Leakage controls
- One S1-grouped fold assignment (artifacts_v2/<profile>/folds.parquet) reused by the embedding adapter,
  every model and calibration; test predictions never influence any choice.

## External data
None. Only the provided training data, pretrained open-licensed model weights, and hand-written
normalisation dictionaries.
