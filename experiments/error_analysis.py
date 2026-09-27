"""Where does the remaining OOF macro-F0.5 loss of the big-xenc run come from?"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from config import load_config
from decide import assign_exclusive, crossfit_calibrate, evaluate, select_expected_f, truth_uids
from prepare import load_records

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
oof = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet")
print("oof cols:", list(oof.columns), len(oof), flush=True)
rec = load_records(cfg, "train", ["entity_id", "src", "country"])
tu = truth_uids(cfg, rec)
s1 = np.load(cfg.art / "gbm" / "eval_s1.npy")
tu = tu[tu["q"].isin(set(s1))].astype(np.int64)
oof["pc"] = crossfit_calibrate(oof["p2"].values, oof["y"].values, oof["fold"].values)
pred = select_expected_f(assign_exclusive(oof, "pc"), "pc", cfg.mc_max_cands, cfg.decide_chunk)
print("official evaluate:", round(evaluate(pred, tu, s1, rec["country"])["all"]["f05"], 5), flush=True)

N = len(rec)
key = lambda d: d["q"].values.astype(np.int64) * N + d["p"].values.astype(np.int64)
kt, kp, kc = key(tu), key(pred), key(oof)
t_in_pred, t_in_cand, p_in_t = np.isin(kt, kp), np.isin(kt, kc), np.isin(kp, kt)

S = pd.DataFrame(index=pd.Index(s1, name="q"))
S["n_true"] = tu.groupby("q").size().reindex(S.index, fill_value=0)
S["n_pred"] = pred.groupby("q").size().reindex(S.index, fill_value=0)
S["tp"] = pd.Series(t_in_pred).groupby(tu["q"].values).sum().reindex(S.index, fill_value=0)
S["fn_block"] = pd.Series(~t_in_cand).groupby(tu["q"].values).sum().reindex(S.index, fill_value=0)
S["fn_dec"] = S["n_true"] - S["tp"] - S["fn_block"]
S["fp"] = S["n_pred"] - S["tp"]


def f05(tp, fp, fn):
    tp, fp, fn = map(np.asarray, (tp, fp, fn))
    both_empty = (tp + fp == 0) & (tp + fn == 0)
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    R = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    F = np.where(P + R > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    return np.where(both_empty, 1.0, F)


S["f"] = f05(S.tp, S.fp, S.fn_block + S.fn_dec)
n = len(S)
print(f"\nS1 evaluated: {n:,} | macro F0.5 (recomputed): {S.f.mean():.5f} | total loss {1 - S.f.mean():.5f}")

cats = {
    "singleton, predicted something (FP only)": (S.n_true == 0) & (S.n_pred > 0),
    "has matches, predicted nothing": (S.n_true > 0) & (S.n_pred == 0),
    "partial: FP only": (S.n_true > 0) & (S.n_pred > 0) & (S.fp > 0) & (S.fn_block + S.fn_dec == 0),
    "partial: FN only (decision)": (S.n_true > 0) & (S.n_pred > 0) & (S.fp == 0) & (S.fn_dec > 0) & (S.fn_block == 0),
    "partial: FN only (blocking miss)": (S.n_true > 0) & (S.n_pred > 0) & (S.fp == 0) & (S.fn_block > 0) & (S.fn_dec == 0),
    "partial: FN blocking+decision": (S.n_true > 0) & (S.n_pred > 0) & (S.fp == 0) & (S.fn_block > 0) & (S.fn_dec > 0),
    "partial: FP and FN": (S.n_true > 0) & (S.n_pred > 0) & (S.fp > 0) & (S.fn_block + S.fn_dec > 0),
}
print(f"\n{'category':44s} {'#S1':>9s} {'share S1':>9s} {'loss pts':>9s} {'share of loss':>13s}")
tot = (1 - S.f).sum()
for name, m in cats.items():
    l = (1 - S.f[m]).sum()
    print(f"{name:44s} {int(m.sum()):9,d} {m.mean() * 100:8.3f}% {l / n:9.5f} {l / tot * 100:12.1f}%")
print(f"{'(perfect)':44s} {int((S.f == 1).sum()):9,d}")

# oracle ceilings
f_block_oracle = f05(S.n_true - S.fn_block, 0, S.fn_block).mean()
print(f"\nceiling if every decision were perfect (blocking misses remain): {f_block_oracle:.5f}")
S2 = S.copy(); S2.loc[S.n_true == 0, ["fp"]] = 0
print(f"if all singleton false positives were removed:               {f05(S2.tp, S2.fp, S2.fn_block + S2.fn_dec).mean():.5f}")
S3 = S.copy(); S3["fp"] = 0
print(f"if ALL false positives were removed:                         {f05(S3.tp, S3.fp, S3.fn_block + S3.fn_dec).mean():.5f}")
S4 = S.copy(); S4["tp"] = S.tp + S.fn_dec; S4["fn_dec"] = 0
print(f"if all decision-level misses (in candidates) were recovered: {f05(S4.tp, S4.fp, S4.fn_block).mean():.5f}")

# how confident are the errors?
pd_ = pred.assign(pc=oof.set_index(["q", "p"])["pc"].reindex(pd.MultiIndex.from_frame(pred[["q", "p"]])).values)
fp_pc = pd_["pc"].values[~p_in_t]
tp_pc = pd_["pc"].values[p_in_t]
print(f"\npc of predicted TRUE pairs : median {np.median(tp_pc):.3f} p10 {np.quantile(tp_pc, .1):.3f}")
print(f"pc of predicted FALSE pairs: median {np.median(fp_pc):.3f} p10 {np.quantile(fp_pc, .1):.3f} p90 {np.quantile(fp_pc, .9):.3f} | n={len(fp_pc):,}")
missed = tu[t_in_cand & ~t_in_pred]
mpc = oof.set_index(["q", "p"])["pc"].reindex(pd.MultiIndex.from_frame(missed[["q", "p"]])).values
ex = assign_exclusive(oof, "pc").set_index(["q", "p"])["pc"].reindex(pd.MultiIndex.from_frame(missed[["q", "p"]])).values
print(f"missed-in-candidate true pairs: n={len(missed):,} | pc median {np.nanmedian(mpc):.3f} "
      f"| zeroed by assign (p given to another S1): {np.mean(ex == 0) * 100:.1f}%")
print(f"\nn_true distribution of S1: {S.n_true.value_counts().sort_index().head(8).to_dict()}")
print(f"singleton rate: {(S.n_true == 0).mean():.4f} | predicted-empty rate: {(S.n_pred == 0).mean():.4f}")
