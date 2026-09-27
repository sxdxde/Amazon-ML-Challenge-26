# gbm2: LightGBM pair classifier

OOF overall: `{"n": 783172, "pos_rate": 0.08786, "auc": 0.99984, "pr_auc": 0.9986, "logloss": 0.00801}`

## OOF per country

|       |      n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|-------:|-----------:|--------:|---------:|----------:|
| India | 321872 |    0.08431 | 0.99979 |  0.99819 |   0.00848 |
| US    | 461300 |    0.09034 | 0.99986 |  0.99881 |   0.00769 |

## Folds

|   fold |   best_iter |      n |   pos_rate |     auc |   pr_auc |   logloss |
|-------:|------------:|-------:|-----------:|--------:|---------:|----------:|
|      0 |          97 | 159096 |    0.08778 | 0.99988 |  0.99871 |   0.00745 |
|      1 |          74 | 155750 |    0.0883  | 0.99982 |  0.99849 |   0.00885 |
|      2 |         104 | 155368 |    0.08899 | 0.99987 |  0.99884 |   0.00755 |
|      3 |          82 | 157831 |    0.08797 | 0.99987 |  0.99881 |   0.00749 |
|      4 |          88 | 155127 |    0.08628 | 0.99984 |  0.99838 |   0.00875 |

## Top-40 features (mean gain over folds)

|                 |     gain |
|:----------------|---------:|
| p1              | 809387   |
| p1_gap_p        | 286731   |
| cos_dense       |  63842.2 |
| rrf             |  22624.5 |
| p1_sum_p        |  20473.9 |
| p1_gap_q        |   3444.6 |
| p1_share_p      |   3156.6 |
| qs              |   1295.2 |
| rev_gap_qs      |   1149.3 |
| p1_sum_q        |    917.2 |
| cos_combo_char  |    838.3 |
| n_ratio_full    |    787.3 |
| n_maxidf        |    763.2 |
| gap_cd          |    723.9 |
| p1_rank_q       |    715.7 |
| name_freq_q     |    655.9 |
| corr_both_mean  |    643.5 |
| qs_minus_mean_q |    642.8 |
| cos_combo_word  |    618.3 |
| p1_rank_p       |    599.3 |
| a_maxidf        |    547.1 |
| a_len_q         |    525.3 |
| n_len_ratio     |    493   |
| gap_qs          |    470.7 |
| n_len_p         |    448.9 |
| n_jw            |    436.1 |
| rank_qs         |    430.6 |
| cos_name_char   |    429.2 |
| n_tset_full     |    413.9 |
| rank_qs_src     |    411.8 |
| n_cands_q       |    411.6 |
| n_len_q         |    399.9 |
| n_wjacc         |    354.8 |
| n_skel_ratio    |    350.1 |
| a_wjacc         |    331.4 |
| name_freq_p     |    327.9 |
| a_jacc          |    313.5 |
| n_nospace_ratio |    282.3 |
| rank_dense      |    271.1 |
| p1_n05_q        |    270.8 |
