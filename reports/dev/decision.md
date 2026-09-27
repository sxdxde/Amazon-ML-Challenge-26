# Decision rule comparison (train OOF, cross-fitted isotonic calibration)

Scores: `p2`. Evaluated over every featurised train S1 (S1s with no candidates predict empty; true links missed by blocking count as misses).

**Chosen variant: `expectedF`** (macro F0.5 = 0.98275)

| variant          | group   |   n_s1 |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   threshold |
|:-----------------|:--------|-------:|-------------:|----------------:|--------------------:|------------:|
| threshold        | all     |  20000 |      0.98228 |         0.97878 |             0.98249 |        0.75 |
| threshold        | India   |   7912 |      0.9787  |         0.9697  |             0.97922 |        0.75 |
| threshold        | US      |  12088 |      0.98463 |         0.98433 |             0.98465 |        0.75 |
| threshold+assign | all     |  20000 |      0.98228 |         0.97878 |             0.98249 |        0.75 |
| threshold+assign | India   |   7912 |      0.9787  |         0.9697  |             0.97922 |        0.75 |
| threshold+assign | US      |  12088 |      0.98463 |         0.98433 |             0.98465 |        0.75 |
| expectedF        | all     |  20000 |      0.98275 |         0.96552 |             0.98378 |      nan    |
| expectedF        | India   |   7912 |      0.97966 |         0.95804 |             0.9809  |      nan    |
| expectedF        | US      |  12088 |      0.98477 |         0.97009 |             0.98568 |      nan    |
| expectedF+assign | all     |  20000 |      0.98274 |         0.96552 |             0.98377 |      nan    |
| expectedF+assign | India   |   7912 |      0.97964 |         0.95804 |             0.98087 |      nan    |
| expectedF+assign | US      |  12088 |      0.98477 |         0.97009 |             0.98568 |      nan    |

## Leave-one-country-out (stage-1 GBM + expectedF+assign)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.97475 |         0.90171 |             0.97925 |    0.99972 |       0.9975  |
| US      | India  |      0.95474 |         0.89277 |             0.95829 |    0.99906 |       0.99167 |
