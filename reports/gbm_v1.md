# gbm: LightGBM pair classifier

OOF overall: `{"n": 85416205, "pos_rate": 0.08767, "auc": 0.99986, "pr_auc": 0.99867, "logloss": 0.00754}`

## OOF per country

|       |           n |   pos_rate |     auc |   pr_auc |   logloss |
|:------|------------:|-----------:|--------:|---------:|----------:|
| India | 3.53923e+07 |    0.08348 | 0.9999  |  0.99899 |   0.00639 |
| US    | 5.00239e+07 |    0.09062 | 0.99982 |  0.99843 |   0.00835 |

## Folds

|   fold |   best_iter |           n |   pos_rate |     auc |   pr_auc |   logloss |
|-------:|------------:|------------:|-----------:|--------:|---------:|----------:|
|      0 |         541 | 1.7069e+07  |    0.08769 | 0.99986 |  0.99867 |   0.00756 |
|      1 |         549 | 1.70697e+07 |    0.08769 | 0.99986 |  0.99867 |   0.00754 |
|      2 |         581 | 1.71068e+07 |    0.08772 | 0.99986 |  0.99867 |   0.00752 |
|      3 |         725 | 1.70494e+07 |    0.08767 | 0.99986 |  0.99868 |   0.00753 |
|      4 |         714 | 1.71213e+07 |    0.08757 | 0.99985 |  0.99867 |   0.00752 |

## Leave-one-country-out (train on one country, evaluate on the other)

| train   | eval   |   macro_F0.5 |   singleton_acc |   nonsingleton_F0.5 |   pair_auc |   pair_pr_auc |
|:--------|:-------|-------------:|----------------:|--------------------:|-----------:|--------------:|
| India   | US     |      0.96403 |         0.87703 |             0.96917 |    0.99945 |       0.99511 |
| US      | India  |      0.9089  |         0.78539 |             0.91621 |    0.99839 |       0.98172 |

## Top-40 features (mean gain over folds)

|                   |             gain |
|:------------------|-----------------:|
| rev_gap_qs        |      1.71339e+07 |
| rev_rank_cd       |      6.33634e+06 |
| rev_rank_qs       |      5.89087e+06 |
| d_jacc            |      1.78419e+06 |
| sn_prefix         |      1.09499e+06 |
| qs                | 909881           |
| d_p_subset        | 518313           |
| corr_a_max        | 400837           |
| n_ratio_full      | 357064           |
| rrf               | 307646           |
| d_nshared         | 287365           |
| a_tset            | 246121           |
| n_len_ratio       | 193425           |
| n_skel_ratio      | 179007           |
| legal_eq          | 152513           |
| n_tset_full       | 142745           |
| legal_conflict    | 128233           |
| n_len_p           | 121639           |
| sn_ratio          | 117320           |
| d_p_n             | 108340           |
| cos_name_char     |  97649.8         |
| d_q_n             |  91896           |
| name_freq_p       |  77971.4         |
| a_jacc            |  76551.6         |
| name_freq_q       |  75802           |
| n_first_eq        |  73587.9         |
| n_wjacc           |  71796.2         |
| corr_both_mean    |  65320.2         |
| n_q_above_08      |  64272.8         |
| n_nospace_ratio   |  63765.9         |
| a_len_q           |  63303.9         |
| n_maxidf          |  59851.7         |
| n_ratio           |  57742.4         |
| cos_dense         |  57365.9         |
| corr_both_max     |  55297.5         |
| n_nospace_partial |  55034.5         |
| a_wjacc           |  54442.7         |
| n_jw              |  54396.9         |
| a_len_ratio       |  54143.4         |
| n_jacc            |  46816.9         |
