# gbm: LightGBM pair classifier

OOF overall: `{"n": 3644, "pos_rate": 0.27552, "auc": 0.99986, "pr_auc": 0.99964, "logloss": 0.01421}`

## OOF per country

|       |    n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|-----:|-----------:|--------:|---------:|----------:|
| India | 1603 |    0.27885 | 0.9999  |  0.99974 |   0.01329 |
| US    | 2041 |    0.27291 | 0.99981 |  0.99952 |   0.01494 |

## Folds

|   fold |   best_iter |    n |   pos_rate |     auc |   pr_auc |   logloss |
|-------:|------------:|-----:|-----------:|--------:|---------:|----------:|
|      0 |          60 | 1241 |    0.26269 | 0.99985 |  0.9996  |   0.01381 |
|      1 |          60 | 1284 |    0.28816 | 0.99983 |  0.9996  |   0.01642 |
|      2 |          59 | 1119 |    0.27525 | 0.9999  |  0.99976 |   0.01213 |

## Leave-one-country-out (train on one country, evaluate on the other)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.98345 |         0.93333 |             0.9883  |    0.99978 |       0.99945 |
| US      | India  |      0.95978 |         1       |             0.95607 |    0.99923 |       0.99803 |

## Top-40 features (mean gain over folds)

|                 |   gain |
|:----------------|-------:|
| cos_dense       | 5119.1 |
| cos_combo_word  | 2547.9 |
| cos_combo_char  | 1036   |
| a_wjacc         |  169.1 |
| a_jacc          |  116.1 |
| corr_a_max      |   96.7 |
| n_ratio_full    |   84.7 |
| n_nospace_ratio |   43.6 |
| qs              |   41.9 |
| gap_cd          |   32.6 |
| n_jw            |   26.9 |
| n_ratio         |   25.9 |
| sn_ratio        |   25.7 |
| d_jacc          |   23.2 |
| rev_gap_qs      |   22.5 |
| n_tsort         |   19.4 |
| d_p_subset      |   14.7 |
| rank_qs         |   14.5 |
| a_tset          |   14.5 |
| corr_both_max   |   14.2 |
| n_tset_full     |   14   |
| rrf             |   13.3 |
| a_maxidf        |   11.6 |
| n_skel_ratio    |   10.2 |
| sn_prefix       |    7.5 |
| n_maxidf        |    5.8 |
| rank_dense      |    5.8 |
| qs_minus_mean_q |    4.3 |
| rank_cd         |    4.2 |
| cos_addr_word   |    4   |
| n_len_q         |    3.3 |
| a_len_ratio     |    3   |
| a_tsort         |    2.8 |
| n_len_p         |    2.6 |
| a_jw            |    2.6 |
| a_ratio         |    2.4 |
| cos_name_word   |    2.3 |
| legal_eq        |    2.3 |
| rank_qs_src     |    1.8 |
| n_partial       |    1.7 |
