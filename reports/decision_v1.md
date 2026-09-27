# Decision rule comparison (train OOF, cross-fitted isotonic calibration)

Scores: `p2`. Evaluated over every featurised train S1 (S1s with no candidates predict empty; true links missed by blocking count as misses).

**Chosen variant: `expectedF+assign`** (macro F0.5 = 0.98284)

| variant          | group   |    n_s1 |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   threshold |
|:-----------------|:--------|--------:|-------------:|----------------:|--------------------:|------------:|
| threshold        | all     | 2206821 |      0.98263 |         0.98734 |             0.98235 |       0.675 |
| threshold        | India   |  883188 |      0.97772 |         0.98492 |             0.97729 |       0.675 |
| threshold        | US      | 1323633 |      0.98591 |         0.98896 |             0.98573 |       0.675 |
| threshold+assign | all     | 2206821 |      0.98263 |         0.98736 |             0.98235 |       0.675 |
| threshold+assign | India   |  883188 |      0.97773 |         0.98494 |             0.9773  |       0.675 |
| threshold+assign | US      | 1323633 |      0.98591 |         0.98897 |             0.98573 |       0.675 |
| expectedF        | all     | 2206821 |      0.98282 |         0.97926 |             0.98304 |     nan     |
| expectedF        | India   |  883188 |      0.97788 |         0.97572 |             0.97801 |     nan     |
| expectedF        | US      | 1323633 |      0.98612 |         0.98162 |             0.98639 |     nan     |
| expectedF+assign | all     | 2206821 |      0.98284 |         0.98025 |             0.983   |     nan     |
| expectedF+assign | India   |  883188 |      0.9779  |         0.97649 |             0.97799 |     nan     |
| expectedF+assign | US      | 1323633 |      0.98614 |         0.98276 |             0.98634 |     nan     |

## Leave-one-country-out (stage-1 GBM + expectedF+assign)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.96403 |         0.87703 |             0.96917 |    0.99945 |       0.99511 |
| US      | India  |      0.9089  |         0.78539 |             0.91621 |    0.99839 |       0.98172 |
