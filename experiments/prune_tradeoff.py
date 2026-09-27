"""Cascade pruning: keep only candidates the stage-1 GBM rates plausible (rank < K and p1 >= tau) before the
final model (xenc + gbm2) runs. Trade-off on honest train OOF: candidates/S1, recall ceiling, macro F0.5."""
import sys
sys.path.insert(0, "/path/to/amazon_ml_2026/code/business_entity_resolution/src")
import numpy as np
import pandas as pd

from config import load_config
from decide import assign_exclusive, crossfit_calibrate, evaluate, select_expected_f, truth_uids
from prepare import load_records

cfg = load_config("full", root="/path/to/amazon_ml_2026", data_dir="/path/to/amazon_ml_2026/dataset")
rec = load_records(cfg, "train", ["entity_id", "src", "country"])
N = len(rec)
tu = truth_uids(cfg, rec)
s1 = np.load(cfg.art / "gbm" / "eval_s1.npy")
tu = tu[tu["q"].isin(set(s1))].astype(np.int64)
o1 = pd.read_parquet(cfg.art / "gbm" / "oof.parquet", columns=["q", "p", "p1"])
o2 = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet")
assert (o1["q"].values == o2["q"].values).all() and (o1["p"].values == o2["p"].values).all()
d = o2.assign(p1=o1["p1"].values)
d["pc"] = crossfit_calibrate(d["p2"].values, d["y"].values, d["fold"].values)
d["r1"] = d.sort_values(["q", "p1"], ascending=[True, False]).groupby("q").cumcount().reindex(d.index).values
pred = select_expected_f(assign_exclusive(d, "pc"), "pc", cfg.mc_max_cands, cfg.decide_chunk)
kt = tu["q"].values * N + tu["p"].values
kp = pred["q"].values.astype(np.int64) * N + pred["p"].values
base = evaluate(pred, tu, s1, None)["all"]["f05"]
print(f"current: {len(d) / len(s1):.2f} cands/S1 | recall ceiling {np.isin(kt, d['q'].values.astype(np.int64) * N + d['p'].values).mean():.5f} | F0.5 {base:.5f}", flush=True)

te = pd.read_parquet(cfg.art / "gbm" / "test_pred.parquet", columns=["q", "p", "p1"])
te["r1"] = te.sort_values(["q", "p1"], ascending=[True, False]).groupby("q").cumcount().reindex(te.index).values
n_te_s1 = 1_732_544
print(f"{'K':>4s} {'tau':>7s} | {'train c/S1':>10s} {'test c/S1':>9s} | {'recall ceil':>11s} | {'F0.5 (pred∩kept)':>16s} {'ΔF':>9s} | {'F0.5 re-decided':>15s}", flush=True)
for K in (20, 15, 10, 8, 6, 5, 4, 3):
    for tau in (0.0, 1e-4, 1e-3, 1e-2):
        keep = (d["r1"].values < K) & (d["p1"].values >= tau)
        kk = d["q"].values[keep].astype(np.int64) * N + d["p"].values[keep]
        rc = np.isin(kt, kk).mean()
        pk = pred[np.isin(kp, kk)]
        f = evaluate(pk, tu, s1, None)["all"]["f05"]
        fr = np.nan
        if K in (10, 6, 4) and tau in (0.0, 1e-3):
            dk = d[keep]
            fr = evaluate(select_expected_f(assign_exclusive(dk, "pc"), "pc", cfg.mc_max_cands, cfg.decide_chunk),
                          tu, s1, None)["all"]["f05"]
        tk = ((te["r1"].values < K) & (te["p1"].values >= tau)).sum() / n_te_s1
        print(f"{K:4d} {tau:7.0e} | {keep.sum() / len(s1):10.2f} {tk:9.2f} | {rc:11.5f} | {f:16.5f} {f - base:+9.5f} | {fr:15.5f}", flush=True)
