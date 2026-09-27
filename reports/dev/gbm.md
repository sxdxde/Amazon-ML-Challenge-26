# gbm: LightGBM pair classifier

OOF overall: `{"n": 783172, "pos_rate": 0.08786, "auc": 0.99987, "pr_auc": 0.99874, "logloss": 0.00789}`

## OOF per country

|       |      n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|-------:|-----------:|--------:|---------:|----------:|
| India | 321872 |    0.08431 | 0.99984 |  0.99838 |   0.00842 |
| US    | 461300 |    0.09034 | 0.99988 |  0.99893 |   0.00752 |

## Folds

|   fold |   best_iter |      n |   pos_rate |     auc |   pr_auc |   logloss |
|-------:|------------:|-------:|-----------:|--------:|---------:|----------:|
|      0 |         228 | 159096 |    0.08778 | 0.99989 |  0.99893 |   0.00731 |
|      1 |         186 | 155750 |    0.0883  | 0.99983 |  0.9985  |   0.00885 |
|      2 |         204 | 155368 |    0.08899 | 0.99989 |  0.99891 |   0.00742 |
|      3 |         168 | 157831 |    0.08797 | 0.99988 |  0.99887 |   0.0073  |
|      4 |         146 | 155127 |    0.08628 | 0.99985 |  0.99844 |   0.00857 |

## Leave-one-country-out (train on one country, evaluate on the other)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.97475 |         0.90171 |             0.97925 |    0.99972 |       0.9975  |
| US      | India  |      0.95474 |         0.89277 |             0.95829 |    0.99906 |       0.99167 |

## Top-40 features (mean gain over folds)

|                   |     gain |
|:------------------|---------:|
| cos_dense         | 407634   |
| rrf               | 361199   |
| qs                | 264407   |
| qs_minus_mean_q   |  27168.2 |
| d_jacc            |  17098   |
| rev_gap_qs        |  15809.8 |
| corr_a_max        |  12618   |
| a_wjacc           |   9346.2 |
| cos_combo_word    |   7527.6 |
| n_ratio_full      |   6321.1 |
| a_jacc            |   6130.1 |
| d_p_subset        |   5457.6 |
| n_len_p           |   5361   |
| sn_prefix         |   4395.3 |
| n_jw              |   4367.2 |
| n_maxidf          |   4355.6 |
| rank_dense        |   3884.5 |
| n_tset_full       |   3764.8 |
| n_partial         |   3491   |
| n_skel_ratio      |   3451.8 |
| sn_ratio          |   3264.9 |
| n_nospace_ratio   |   3153.7 |
| a_maxidf          |   3036.7 |
| n_len_ratio       |   2963.1 |
| legal_conflict    |   2845.4 |
| legal_eq          |   2669.3 |
| rank_combo_word   |   2661   |
| corr_both_max     |   2503   |
| a_tset            |   2388.2 |
| rank_name_char    |   2348.2 |
| cos_name_char     |   2254   |
| corr_both_mean    |   2245.3 |
| a_len_ratio       |   2039.5 |
| rank_cd           |   1948   |
| name_freq_p       |   1826.5 |
| n_wjacc           |   1822.9 |
| rank_qs_src       |   1745.5 |
| name_freq_q       |   1602   |
| n_nospace_partial |   1601.9 |
| n_ratio           |   1560.6 |
