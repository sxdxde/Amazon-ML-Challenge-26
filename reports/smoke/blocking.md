# Blocking diagnostics

Profile `smoke`, embedder `intfloat/multilingual-e5-small`. Candidates restricted to the same country string.
Target union pair recall 0.995, budget 30 mean cands/S1, hard cap 30/S1 (reciprocal-rank fusion).

**Tuned k:** `{"dense": 10, "combo_word": 0, "combo_char": 0, "name_char": 0, "addr_char": 5, "name_word": 2, "addr_word": 2, "key_stnum": 1, "key_postal": 0, "key_name": 1, "key_skel": 1}`

## Train, all S1, at tuned k (this is the final candidate set)

| country   | retriever       |   n_s1 |   pair_recall |   s1_full_recall |   mean_cands |   reduction_ratio |
|:----------|:----------------|-------:|--------------:|-----------------:|-------------:|------------------:|
| India     | dense (k=10)    |    130 |        0.9845 |           0.9412 |      10      |            0.9966 |
| India     | addr_char (k=5) |    130 |        0.8825 |           0.6807 |       5      |            0.9983 |
| India     | name_word (k=2) |    130 |        0.4302 |           0.1765 |       1.9923 |            0.9993 |
| India     | addr_word (k=2) |    130 |        0.4967 |           0.2101 |       2      |            0.9993 |
| India     | key_stnum (k=1) |    130 |        0.4922 |           0.3109 |       1.8923 |            0.9994 |
| India     | key_name (k=1)  |    130 |        0.3836 |           0.0672 |       1.3538 |            0.9995 |
| India     | key_skel (k=1)  |    130 |        0.5211 |           0.1597 |       1.9385 |            0.9993 |
| India     | UNION           |    130 |        0.9911 |           0.9664 |      12.3308 |            0.9958 |
| US        | dense (k=10)    |    170 |        0.9982 |           0.9935 |      10      |            0.9976 |
| US        | addr_char (k=5) |    170 |        0.921  |           0.7677 |       5      |            0.9988 |
| US        | name_word (k=2) |    170 |        0.4991 |           0.2194 |       1.9941 |            0.9995 |
| US        | addr_word (k=2) |    170 |        0.5206 |           0.2129 |       2      |            0.9995 |
| US        | key_stnum (k=1) |    170 |        0.6697 |           0.3419 |       2.1941 |            0.9995 |
| US        | key_name (k=1)  |    170 |        0.5727 |           0.2258 |       1.9235 |            0.9995 |
| US        | key_skel (k=1)  |    170 |        0.614  |           0.2581 |       2.0824 |            0.9995 |
| US        | UNION           |    170 |        1      |           1      |      12.0059 |            0.9971 |

## Tuning subsample: each retriever at k_max (upper bounds)

| country   | retriever         |   n_s1 |   pair_recall |   s1_full_recall |   mean_cands |   reduction_ratio |
|:----------|:------------------|-------:|--------------:|-----------------:|-------------:|------------------:|
| India     | dense (k=15)      |    130 |        0.9911 |           0.9664 |      15      |            0.9949 |
| India     | combo_word (k=15) |    130 |        0.9978 |           0.9916 |      15      |            0.9949 |
| India     | combo_char (k=10) |    130 |        0.9911 |           0.9664 |      10      |            0.9966 |
| India     | name_char (k=10)  |    130 |        0.7184 |           0.4958 |      10      |            0.9966 |
| India     | addr_char (k=10)  |    130 |        0.949  |           0.8151 |      10      |            0.9966 |
| India     | name_word (k=10)  |    130 |        0.7738 |           0.4958 |       9.5077 |            0.9967 |
| India     | addr_word (k=10)  |    130 |        0.9534 |           0.8319 |      10      |            0.9966 |
| India     | key_stnum (k=1)   |    130 |        0.4922 |           0.3109 |       1.8923 |            0.9994 |
| India     | key_postal (k=1)  |    130 |        0      |           0      |       0      |            1      |
| India     | key_name (k=1)    |    130 |        0.3836 |           0.0672 |       1.3538 |            0.9995 |
| India     | key_skel (k=1)    |    130 |        0.5211 |           0.1597 |       1.9385 |            0.9993 |
| India     | UNION             |    130 |        1      |           1      |      40.1538 |            0.9862 |
| US        | dense (k=15)      |    170 |        1      |           1      |      15      |            0.9963 |
| US        | combo_word (k=15) |    170 |        0.9964 |           0.9871 |      15      |            0.9963 |
| US        | combo_char (k=10) |    170 |        0.9964 |           0.9871 |      10      |            0.9976 |
| US        | name_char (k=10)  |    170 |        0.9623 |           0.8774 |      10      |            0.9976 |
| US        | addr_char (k=10)  |    170 |        0.9533 |           0.8323 |      10      |            0.9976 |
| US        | name_word (k=10)  |    170 |        0.8743 |           0.6516 |       9.6941 |            0.9976 |
| US        | addr_word (k=10)  |    170 |        0.9533 |           0.8323 |       9.9882 |            0.9976 |
| US        | key_stnum (k=1)   |    170 |        0.6697 |           0.3419 |       2.1941 |            0.9995 |
| US        | key_postal (k=1)  |    170 |        0      |           0      |       0      |            1      |
| US        | key_name (k=1)    |    170 |        0.5727 |           0.2258 |       1.9235 |            0.9995 |
| US        | key_skel (k=1)    |    170 |        0.614  |           0.2581 |       2.0824 |            0.9995 |
| US        | UNION             |    170 |        1      |           1      |      37.7412 |            0.9908 |

## Greedy k trace (tuning subsample)

|   dense |   combo_word |   combo_char |   name_char |   addr_char |   name_word |   addr_word |   key_stnum |   key_postal |   key_name |   key_skel |   recall |   mean_cands |
|--------:|-------------:|-------------:|------------:|------------:|------------:|------------:|------------:|-------------:|-----------:|-----------:|---------:|-------------:|
|       0 |            0 |            0 |           0 |           0 |           0 |           0 |           0 |            0 |          0 |          0 |   0      |       0      |
|       0 |            0 |            0 |           0 |           0 |           0 |           0 |           0 |            0 |          1 |          0 |   0.4881 |       1.6767 |
|       0 |            0 |            0 |           0 |           0 |           0 |           0 |           1 |            0 |          1 |          0 |   0.7778 |       2.73   |
|       0 |            0 |            0 |           0 |           0 |           0 |           0 |           1 |            0 |          1 |          1 |   0.8165 |       2.92   |
|       2 |            0 |            0 |           0 |           0 |           0 |           0 |           1 |            0 |          1 |          1 |   0.874  |       3.3533 |
|       2 |            0 |            0 |           0 |           0 |           0 |           2 |           1 |            0 |          1 |          1 |   0.8988 |       3.71   |
|       5 |            0 |            0 |           0 |           0 |           0 |           2 |           1 |            0 |          1 |          1 |   0.9732 |       5.51   |
|       5 |            0 |            0 |           0 |           2 |           0 |           2 |           1 |            0 |          1 |          1 |   0.9742 |       5.6667 |
|       5 |            0 |            0 |           0 |           2 |           2 |           2 |           1 |            0 |          1 |          1 |   0.9762 |       5.99   |
|       5 |            0 |            0 |           0 |           5 |           2 |           2 |           1 |            0 |          1 |          1 |   0.9831 |       7.49   |
|      10 |            0 |            0 |           0 |           5 |           2 |           2 |           1 |            0 |          1 |          1 |   0.996  |      12.15   |

## Test candidate statistics (same tuned k)

| country   |   s1_total |   pairs |   s1_with_cands |   mean_cands |
|:----------|-----------:|--------:|----------------:|-------------:|
| France    |         50 |     779 |              50 |        15.58 |
| India     |         50 |     829 |              50 |        16.58 |
| US        |         50 |     819 |              50 |        16.38 |
