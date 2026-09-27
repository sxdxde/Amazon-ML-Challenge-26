# gbm2: LightGBM pair classifier

OOF overall: `{"n": 85416205, "pos_rate": 0.08767, "auc": 0.9999, "pr_auc": 0.99904, "logloss": 0.00622}`

## OOF per country

|       |           n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|------------:|-----------:|--------:|---------:|----------:|
| India | 3.53923e+07 |    0.08348 | 0.99993 |  0.99927 |   0.00533 |
| US    | 5.00239e+07 |    0.09062 | 0.99987 |  0.99886 |   0.00684 |

## Folds

|   fold |   best_iter |           n |   pos_rate |    auc |   pr_auc |   logloss |
|-------:|------------:|------------:|-----------:|-------:|---------:|----------:|
|      0 |         145 | 1.7069e+07  |    0.08769 | 0.9999 |  0.99904 |   0.00624 |
|      1 |         198 | 1.70697e+07 |    0.08769 | 0.9999 |  0.99903 |   0.00625 |
|      2 |         161 | 1.71068e+07 |    0.08772 | 0.9999 |  0.99904 |   0.0062  |
|      3 |         184 | 1.70494e+07 |    0.08767 | 0.9999 |  0.99905 |   0.00619 |
|      4 |         150 | 1.71213e+07 |    0.08757 | 0.9999 |  0.99904 |   0.0062  |

## Top-40 features (mean gain over folds)

|                 |             gain |
|:----------------|-----------------:|
| p1              |      1.86519e+07 |
| p1_gap_p        |      1.69608e+07 |
| p1_share_p      |      1.56664e+06 |
| rev_gap_qs      | 372954           |
| p1_sum_p        | 269614           |
| p1_gap_q        | 107363           |
| p1_rank_p       |  59078.1         |
| p1_sum_q        |  49668.2         |
| n_q_above_08    |  27592           |
| p1_rank_q       |  16647.8         |
| corr_both_mean  |  13813.9         |
| rank_qs         |  13193           |
| n_ratio_full    |  13081.7         |
| n_maxidf        |  12464.9         |
| qs_minus_mean_q |  11982.5         |
| name_freq_q     |  11729.4         |
| n_cands_q       |  11559           |
| d_jacc          |  11308.8         |
| cos_combo_char  |  10862.3         |
| cos_dense       |  10550           |
| gap_cd          |  10457.1         |
| cos_combo_word  |  10355.2         |
| a_len_q         |   9850.9         |
| rrf             |   9485.7         |
| n_q_for_p       |   8980.3         |
| gap_qs          |   8865.5         |
| name_freq_p     |   8541.1         |
| qs              |   8037.5         |
| rank_cd         |   7515.6         |
| p1_n05_q        |   7407.8         |
| rev_rank_cd     |   7121.5         |
| n_len_p         |   6933.4         |
| n_len_q         |   6635.1         |
| a_maxidf        |   6538.1         |
| rank_combo_word |   6435.3         |
| rev_rank_qs     |   6214.4         |
| rank_qs_src     |   5492           |
| corr_a_max      |   5477.6         |
| n_len_ratio     |   5259.8         |
| n_tset_full     |   5016.6         |
