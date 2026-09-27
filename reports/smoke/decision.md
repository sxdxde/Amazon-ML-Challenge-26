# Decision rule comparison (train OOF, cross-fitted isotonic calibration)

Scores: `p2`. Evaluated over every featurised train S1 (S1s with no candidates predict empty; true links missed by blocking count as misses).

**Chosen variant: `threshold`** (macro F0.5 = 0.99107)

| variant          | group   |   n_s1 |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   threshold |
|:-----------------|:--------|-------:|-------------:|----------------:|--------------------:|------------:|
| threshold        | all     |    300 |      0.99107 |         0.96154 |             0.99387 |        0.95 |
| threshold        | India   |    130 |      0.99183 |         1       |             0.99108 |        0.95 |
| threshold        | US      |    170 |      0.99049 |         0.93333 |             0.99602 |        0.95 |
| threshold+assign | all     |    300 |      0.99107 |         0.96154 |             0.99387 |        0.95 |
| threshold+assign | India   |    130 |      0.99183 |         1       |             0.99108 |        0.95 |
| threshold+assign | US      |    170 |      0.99049 |         0.93333 |             0.99602 |        0.95 |
| expectedF        | all     |    300 |      0.98975 |         0.96154 |             0.99242 |      nan    |
| expectedF        | India   |    130 |      0.99183 |         1       |             0.99108 |      nan    |
| expectedF        | US      |    170 |      0.98815 |         0.93333 |             0.99346 |      nan    |
| expectedF+assign | all     |    300 |      0.98975 |         0.96154 |             0.99242 |      nan    |
| expectedF+assign | India   |    130 |      0.99183 |         1       |             0.99108 |      nan    |
| expectedF+assign | US      |    170 |      0.98815 |         0.93333 |             0.99346 |      nan    |

## Leave-one-country-out (stage-1 GBM + expectedF+assign)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.98345 |         0.93333 |             0.9883  |    0.99978 |       0.99945 |
| US      | India  |      0.95978 |         1       |             0.95607 |    0.99923 |       0.99803 |
