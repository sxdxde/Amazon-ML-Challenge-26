# gbm2: LightGBM pair classifier

OOF overall: `{"n": 3644, "pos_rate": 0.27552, "auc": 0.9997, "pr_auc": 0.99938, "logloss": 0.01353}`

## OOF per country

|       |    n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|-----:|-----------:|--------:|---------:|----------:|
| India | 1603 |    0.27885 | 0.9998  |  0.99952 |   0.01301 |
| US    | 2041 |    0.27291 | 0.99964 |  0.99927 |   0.01394 |

## Folds

|   fold |   best_iter |    n |   pos_rate |     auc |   pr_auc |   logloss |
|-------:|------------:|-----:|-----------:|--------:|---------:|----------:|
|      0 |          60 | 1241 |    0.26269 | 0.99957 |  0.99912 |   0.01104 |
|      1 |          60 | 1284 |    0.28816 | 0.99972 |  0.99941 |   0.01691 |
|      2 |          60 | 1119 |    0.27525 | 0.99983 |  0.99956 |   0.01243 |

## Cross-encoder on the cascade band (pair level)

|                         |   n |   pos_rate |     auc |   pr_auc |   logloss |
|:------------------------|----:|-----------:|--------:|---------:|----------:|
| xenc_standalone_on_band | 900 |    0.81111 | 0.99025 |  0.99792 |   0.50137 |
| p1_on_band              | 900 |    0.81111 | 0.99939 |  0.99986 |   0.02257 |
| p2_on_band              | 900 |    0.81111 | 0.9984  |  0.99967 |   0.02807 |

## Top-40 features (mean gain over folds)

|                 |   gain |
|:----------------|-------:|
| p1              | 4922.2 |
| p1_gap_p        | 3888.3 |
| cos_dense       |  182.1 |
| cos_combo_word  |  151.7 |
| corr_a_max      |   46.3 |
| a_jacc          |   41.9 |
| a_wjacc         |   37.3 |
| sn_ratio        |   22.9 |
| d_jacc          |   14.4 |
| n_ratio_full    |   11.2 |
| n_jw            |    9   |
| a_tset          |    8.2 |
| rank_dense      |    8.1 |
| rrf             |    7.6 |
| d_p_subset      |    7.4 |
| p1_sum_p        |    7.2 |
| xenc_gap_q      |    7   |
| n_tset_full     |    6.4 |
| a_maxidf        |    6.4 |
| n_len_p         |    6.4 |
| cos_addr_word   |    5.4 |
| n_cands_q       |    4.7 |
| n_nospace_ratio |    4.5 |
| p1_gap_q        |    4.5 |
| a_tsort         |    4.5 |
| xenc            |    4.1 |
| rank_qs         |    3.5 |
| a_nshared       |    3.5 |
| gap_cd          |    3.3 |
| stword_ratio    |    2.8 |
| corr_both_max   |    2.6 |
| rank_cd         |    2.6 |
| cos_name_word   |    2.4 |
| qs              |    2.4 |
| corr_both_mean  |    2.3 |
| n_len_q         |    2   |
| legal_eq        |    1.9 |
| rev_gap_qs      |    1.6 |
| n_maxidf        |    1.5 |
| qs_minus_mean_q |    1.5 |
