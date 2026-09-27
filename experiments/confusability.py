import sys
sys.path.insert(0, ".")
import glob
import numpy as np
import pandas as pd

cols = ["q", "cos_dense", "n_ratio", "a_ratio", "n_cands_q"]
for c in ("France", "India", "US"):
    files = sorted(glob.glob(f"/path/to/amazon_ml_2026/artifacts/full/features/test/{c}_*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    q = df["q"].to_numpy()
    cd = df["cos_dense"].to_numpy(np.float32)
    order = np.lexsort((-cd, q))
    qs, cds = q[order], cd[order]
    start = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    rank = np.arange(len(qs)) - np.repeat(start, np.diff(np.r_[start, len(qs)]))
    top1 = cds[rank == 0]
    q1 = qs[rank == 0]
    top2_vals = np.full(len(top1), np.nan, np.float32)
    m2 = rank == 1
    # q1 is unique q values in ascending order (lexsort primary key) -> searchsorted is a valid O(log n) lookup
    idx2 = np.searchsorted(q1, qs[m2])
    top2_vals[idx2] = cds[m2]
    gap = top1 - top2_vals
    gap_valid = gap[~np.isnan(top2_vals)]

    best = df.groupby("q").agg(best_ratio_n=("n_ratio", "max"), best_ratio_a=("a_ratio", "max"),
                               n_cands=("n_cands_q", "first"))
    print(f"== {c} (n_s1={len(top1):,}) ==", flush=True)
    print(f"  cos_dense rank1-rank2 gap (n_s1_with>=2_cands={len(gap_valid):,}): "
         f"mean={np.nanmean(gap_valid):.4f}  median={np.nanmedian(gap_valid):.4f}  "
         f"frac<0.02={np.mean(gap_valid<0.02):.4f}  frac<0.05={np.mean(gap_valid<0.05):.4f}", flush=True)
    print(f"  best n_ratio (order-sensitive): mean={best.best_ratio_n.mean():.2f}  p10={best.best_ratio_n.quantile(.1):.2f}", flush=True)
    print(f"  best a_ratio (order-sensitive): mean={best.best_ratio_a.mean():.2f}  p10={best.best_ratio_a.quantile(.1):.2f}", flush=True)
    del df, q, cd, order, qs, cds, start, rank, top1, q1, top2_vals, gap, gap_valid, best
