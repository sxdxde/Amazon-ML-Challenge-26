import json
k = json.load(open("/path/to/amazon_ml_2026/artifacts/full/block/k_choice.json"))
print("tuned k (cap 80):", k["k"])
print()
tr = k["trace"]
print(f"greedy k-search trace: {len(tr)} steps total. Last 10 (recall, mean_cands):")
for row in tr[-10:]:
    print(f"  recall={row['recall']:.5f}  mean_cands={row['mean_cands']:.2f}")
print()
print("k_max diag (upper bound, no cap-80 truncation):")
for d in k["diag_max"]:
    if d["retriever"] == "UNION":
        print(f"  {d['country']}: recall={d['pair_recall']:.5f}  s1_full={d['s1_full_recall']:.5f}  mean_cands={d['mean_cands']:.2f}")
