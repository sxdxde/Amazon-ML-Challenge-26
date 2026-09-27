import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from config import load_config
from decide import assign_exclusive, crossfit_calibrate, select_expected_f, truth_uids
from prepare import load_records

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
rec = load_records(cfg, "train", ["entity_id", "src", "country"])
tu = truth_uids(cfg, rec).astype(np.int64)

# 1) is ground truth one-to-one on the S2/S3 side?
mult = tu.groupby("p")["q"].nunique()
print(f"[1] S2/S3 records in truth: {len(mult):,} | linked to >1 S1: {(mult > 1).sum():,} ({(mult > 1).mean() * 100:.3f}%)")

s1 = np.load(cfg.art / "gbm" / "eval_s1.npy")
tu = tu[tu["q"].isin(set(s1))]
oof = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet")
oof["pc"] = crossfit_calibrate(oof["p2"].values, oof["y"].values, oof["fold"].values)
ex = assign_exclusive(oof, "pc")
# why were 55.7% of missed pairs zeroed by assign? who got the record instead?
N = len(rec)
oof["key"] = oof["q"].values.astype(np.int64) * N + oof["p"].values
tkey = set((tu["q"].values * N + tu["p"].values).tolist())
oof["true"] = oof["key"].isin(tkey)
zeroed_true = oof[(oof["true"]) & (ex["pc"].values == 0) & (oof["pc"].values > 0)]
winner = ex[ex["pc"] > 0].set_index("p")["q"]
wq = winner.reindex(zeroed_true["p"].values).values
print(f"[2] true pairs zeroed by assign: {len(zeroed_true):,}; the winning S1 is also a true S1 for that record: "
      f"{np.mean(np.isin(wq.astype(np.int64) * N + zeroed_true['p'].values, np.fromiter(tkey, np.int64))) * 100:.1f}%")

# 3) collective signal: similarity of a candidate to the S1's most confident candidate
emb = np.load(cfg.art / "embed" / "train.npy", mmap_mode="r")
rng = np.random.default_rng(0)
qs = rng.choice(oof["q"].unique(), 60_000, replace=False)
d = oof[oof["q"].isin(set(qs))].sort_values(["q", "pc"], ascending=[True, False])
top = d.groupby("q").head(1).set_index("q")["p"]
d = d[d.groupby("q").cumcount().values > 0].copy()
a = np.asarray(emb[d["p"].values], np.float32)
b = np.asarray(emb[top.reindex(d["q"].values).values], np.float32)
d["cos_top"] = (a * b).sum(1)
lo = d[(d["pc"] < 0.5)]
print("[3] among non-top candidates with pc<0.5 (the zone where recall is lost):")
print(f"    true matches : n={int(lo.true.sum()):,} cos_to_top median {lo[lo.true].cos_top.median():.3f} p10 {lo[lo.true].cos_top.quantile(.1):.3f}")
print(f"    non-matches  : n={int((~lo.true).sum()):,} cos_to_top median {lo[~lo.true].cos_top.median():.3f} p90 {lo[~lo.true].cos_top.quantile(.9):.3f}")
from sklearn.metrics import roc_auc_score
print(f"    AUC of cos_to_top alone within this zone: {roc_auc_score(lo.true, lo.cos_top):.4f} | AUC of pc alone: {roc_auc_score(lo.true, lo.pc):.4f}")
