"""Stage 3b: S1-level 'does this S1 have ANY match?' gate on top of stage-3 scores.

expectedF decides 'predict nothing' from P(no positives) assuming candidates are independent, but the
true matches of one business are copies of each other (strongly correlated). A dedicated S1-level
model estimates P(singleton) directly; the rule, tuned on honest OOF:
    predict empty          if P_single > t_hi
    force a non-empty set  if P_single < t_lo and expectedF chose empty
    otherwise              expectedF as before            (t_lo=0, t_hi=1 == baseline)
Writes to runs/stage3b/; production and runs/stage3 outputs are read-only.
"""
import json
import os
import sys
from pathlib import Path

SRC = "/path/to/amazon_ml_2026/code/business_entity_resolution/src"
sys.path.insert(0, SRC)
import numpy as np
import pandas as pd

from config import load_config
from decide import _pb_step, assign_exclusive, crossfit_calibrate, evaluate, fit_isotonic, truth_uids
from io_utils import LOG, save_parquet, setup_logging
from prepare import load_records
from train_gbm import cv_train, predict_models

PROD = Path("/path/to/amazon_ml_2026")
S3 = PROD / "runs" / "stage3"
OUT = PROD / "runs" / "stage3b"
DATA = PROD / "dataset"
cfg = load_config("full", root=PROD, data_dir=DATA)
setup_logging(OUT / "stage3b.log")
GF = ["t1", "t2", "t3", "t4", "t5", "sum", "n05", "n09", "n_c", "r1"]


def both_k(P):
    """Per row of P (sorted desc): best k over k>=0 (free) and best k over k>=1 (forced)."""
    n, L = P.shape
    m = L + 1
    S = np.zeros((L + 1, n, m)); S[L][:, 0] = 1.0
    for k in range(L - 1, -1, -1):
        S[k] = _pb_step(S[k + 1], P[:, k])
    a = np.arange(m)[:, None].astype(float); b = np.arange(m)[None, :].astype(float)
    free_e, free_k = S[0][:, 0].copy(), np.zeros(n, np.int64)
    forc_e, forc_k = np.full(n, -1.0), np.ones(n, np.int64)
    pre = np.zeros((n, m)); pre[:, 0] = 1.0
    for k in range(1, L + 1):
        pre = _pb_step(pre, P[:, k - 1])
        e = ((pre @ (1.25 * a / (0.25 * (a + b) + k))) * S[k]).sum(1)
        bf = e > free_e + 1e-12; free_e = np.where(bf, e, free_e); free_k = np.where(bf, k, free_k)
        bo = e > forc_e + 1e-12; forc_e = np.where(bo, e, forc_e); forc_k = np.where(bo, k, forc_k)
    return free_k, forc_k


def sorted_table(df, col, max_c):
    d = df.loc[df[col] > 1e-4, ["q", "p", col]].sort_values(["q", col], ascending=[True, False])
    pos = d.groupby("q").cumcount().values
    keep = pos < max_c
    d, pos = d[keep], pos[keep]
    qs, qidx = np.unique(d["q"].values, return_inverse=True)
    P = np.zeros((len(qs), max_c)); P[qidx, pos] = d[col].values
    fk, ok = np.zeros(len(qs), np.int64), np.zeros(len(qs), np.int64)
    for s in range(0, len(qs), cfg.decide_chunk):
        fk[s:s + cfg.decide_chunk], ok[s:s + cfg.decide_chunk] = both_k(P[s:s + cfg.decide_chunk])
    return d.reset_index(drop=True), pos, qs, qidx, fk, ok


def s1_features(df, col):
    d = df[["q", col]].sort_values(["q", col], ascending=[True, False])
    r = d.groupby("q").cumcount().values
    g = d.assign(r=r)
    out = pd.DataFrame(index=pd.Index(np.unique(d["q"].values), name="q"))
    for k in range(5):
        out[f"t{k + 1}"] = g[g.r == k].set_index("q")[col].reindex(out.index).fillna(0).values
    grp = d.groupby("q")[col]
    out["sum"] = grp.sum().values
    out["n05"] = d.assign(x=(d[col] > .5)).groupby("q")["x"].sum().values
    out["n09"] = d.assign(x=(d[col] > .9)).groupby("q")["x"].sum().values
    out["n_c"] = grp.size().values
    out["r1"] = np.where(out["t1"] > 0, out["t2"] / np.maximum(out["t1"], 1e-12), 0)
    return out


def select(d, pos, qs, qidx, fk, ok, p_single, t_lo, t_hi):
    ps = pd.Series(p_single).reindex(qs).fillna(1.0).values      # S1 unknown to gate -> treat as single
    k = fk.copy()
    k = np.where((k == 0) & (ps < t_lo), ok, k)
    k = np.where(ps > t_hi, 0, k)
    return d.loc[pos < k[qidx], ["q", "p"]]


def main():
    oof = pd.read_parquet(S3 / "work" / "oof3.parquet")
    oof["pc"] = crossfit_calibrate(oof["p3"].values, oof["y"].values, oof["fold"].values)
    test = pd.read_parquet(S3 / "work" / "test3.parquet")
    test["pc"] = fit_isotonic(oof["p3"].values, oof["y"].values).predict(test["p3"].values)
    choice = json.loads((S3 / "artifacts/full/decide/choice.json").read_text())
    use_assign = choice["variant"].endswith("+assign")
    LOG.info("stage3 chose %s -> assign=%s", choice["variant"], use_assign)

    rec = load_records(cfg, "train", ["entity_id", "src", "country"])
    tu = truth_uids(cfg, rec)
    s1 = np.load(cfg.art / "gbm" / "eval_s1.npy")
    tu = tu[tu["q"].isin(set(s1))]
    n_true = tu.groupby("q").size()

    # ---- gate model (S1 level), honest folds = the S1's fold
    G = s1_features(oof, "pc")
    fold_q = oof.groupby("q")["fold"].first().reindex(G.index).values
    y_single = (n_true.reindex(G.index).fillna(0).values == 0).astype(np.float32)
    meta = pd.DataFrame({"q": G.index.values, "y": y_single, "fold": fold_q})
    goof, gmodels, imp, stats = cv_train(cfg, G[GF].to_numpy(np.float32), meta, GF, "gate")
    g_cal = crossfit_calibrate(goof, y_single, fold_q)
    p_single_oof = pd.Series(g_cal, index=G.index)
    LOG.info("gate importance:\n%s", imp.round(1).to_string())

    # ---- tune (t_lo, t_hi) on OOF
    base_df = assign_exclusive(oof, "pc") if use_assign else oof
    d, pos, qs, qidx, fk, ok = sorted_table(base_df, "pc", cfg.mc_max_cands)
    f0 = evaluate(select(d, pos, qs, qidx, fk, ok, p_single_oof, 0.0, 1.01), tu, s1, None)["all"]["f05"]
    LOG.info("baseline (stage3 %s, no gate) OOF F0.5 = %.5f", choice["variant"], f0)
    best = (0.0, 1.01, f0)
    for t_hi in (1.01, 0.9, 0.8, 0.7, 0.6, 0.5):
        for t_lo in (0.0, 0.1, 0.2, 0.3, 0.4):
            if t_lo >= t_hi:
                continue
            f = evaluate(select(d, pos, qs, qidx, fk, ok, p_single_oof, t_lo, t_hi), tu, s1, None)["all"]["f05"]
            if f > best[2]:
                best = (t_lo, t_hi, f)
                LOG.info("  new best t_lo=%.2f t_hi=%.2f -> %.5f", t_lo, t_hi, f)
    LOG.info("GATE RESULT: baseline %.5f -> best %.5f (t_lo=%.2f, t_hi=%.2f)", f0, best[2], best[0], best[1])

    # ---- apply to test
    Gt = s1_features(test, "pc")
    iso_g = fit_isotonic(goof, y_single)
    pst = pd.Series(iso_g.predict(predict_models(gmodels, Gt[GF].to_numpy(np.float32), cfg.n_jobs)), index=Gt.index)
    tdf = assign_exclusive(test, "pc") if use_assign else test
    d, pos, qs, qidx, fk, ok = sorted_table(tdf, "pc", cfg.mc_max_cands)
    pred = select(d, pos, qs, qidx, fk, ok, pst, best[0], best[1])
    dec = OUT / "artifacts/full/decide"
    dec.mkdir(parents=True, exist_ok=True)
    save_parquet(pred, dec / "test_matches.parquet")
    (OUT / "gate_choice.json").write_text(json.dumps({"t_lo": best[0], "t_hi": best[1], "oof_f05": best[2],
                                                      "baseline_oof": f0, "variant": choice["variant"]}, indent=1))
    LOG.info("stage3b test matches: %s pairs over %s S1", f"{len(pred):,}", f"{pred['q'].nunique():,}")
    if best[2] > f0:
        art = OUT / "artifacts/full"
        for sub in ("prepare", "block", "gbm"):
            if not (art / sub).exists():
                os.symlink(cfg.art / sub, art / sub)
        if not (OUT / "utils").exists():
            os.symlink(PROD / "utils", OUT / "utils")
        import predict as P
        P.run(load_config("full", root=OUT, data_dir=DATA), force=True)
    else:
        LOG.info("gate does not beat stage3 on OOF -> no submission written")


if __name__ == "__main__":
    main()
