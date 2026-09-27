"""Stage 3: collective reranker on top of gbm2 (big-xenc run, LB 0.981). Post-processing only.

Features per candidate (q, p), all derived from gbm2 OOF/test scores + embeddings:
  * S1 context     : rank within S1, #cands, top1/2/3 calibrated score, sum, #>0.5, gap/ratio to top
  * record context : strongest claim on the same record p by OTHER S1s (competition)
  * collective     : cosine of p with the S1's two most confident OTHER candidates (a business's true
                     matches are copies of each other), raw + normalized bge-m3 embeddings
  * cos(q, p)      : raw and normalized embeddings
Trained with the SAME S1-grouped folds as gbm2 (column `fold`) -> honest OOF; decided with the same
calibration + decision variants as decide.py. Writes only to NEW locations; production is read-only.

  python stage3.py features | train | decide | predict | all
"""
import json
import os
import sys
import time
from pathlib import Path

SRC = "/path/to/amazon_ml_2026/code/business_entity_resolution/src"
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import torch

from config import load_config
from decide import (apply_variant, crossfit_calibrate, decide_variants, fit_isotonic, truth_uids)
from io_utils import LOG, mark_done, save_parquet, setup_logging
from prepare import load_records
from train_gbm import cv_train, pair_metrics, predict_models

PROD = Path("/path/to/amazon_ml_2026")
ROM = PROD / "runs" / "rom"
S3ROOT = PROD / "runs" / "stage3"
DATA = PROD / "dataset"
cfg = load_config("full", root=PROD, data_dir=DATA)
WORK = S3ROOT / "work"
WORK.mkdir(parents=True, exist_ok=True)
setup_logging(S3ROOT / "stage3.log")
FEATS = ["p2", "pc", "rank_q", "n_q", "top1", "top2", "top3", "sum_q", "n_q05", "gap_top", "pc_rel",
         "pc_other1", "pc_other2", "pmax_other", "n_q_for_p", "is_best_p", "pc_minus_bestother",
         "cos_raw_qp", "cos_norm_qp", "c1_raw", "c2_raw", "c1_norm", "c2_norm"]
BASELINE_OOF = 0.98773


def base(split: str) -> pd.DataFrame:
    if split == "train":
        d = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet")
        d["pc"] = crossfit_calibrate(d["p2"].values, d["y"].values, d["fold"].values)
    else:
        d = pd.read_parquet(cfg.art / "gbm2" / "test_pred.parquet")
        o = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet", columns=["y", "p2"])
        d["pc"] = fit_isotonic(o["p2"].values, o["y"].values).predict(d["p2"].values)
    return d


def build_features(split: str) -> None:
    t0 = time.time()
    d = base(split)
    n = len(d)
    LOG.info("[%s] %s rows", split, f"{n:,}")
    q, p, pc = d["q"].values.astype(np.int64), d["p"].values.astype(np.int64), d["pc"].values.astype(np.float64)

    # ---- S1 context (sort by q, pc desc)
    o = np.lexsort((-pc, q))
    qs, ps, pcs = q[o], p[o], pc[o]
    start = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    size = np.diff(np.r_[start, n])
    grp = np.repeat(np.arange(len(start)), size)
    rank = np.arange(n) - start[grp]

    def nth(k):
        v = np.zeros(len(start))
        ok = size > k
        v[ok] = pcs[start[ok] + k]
        return v[grp]

    top1, top2, top3 = nth(0), nth(1), nth(2)
    sum_q = np.bincount(grp, weights=pcs)[grp]
    n05 = np.bincount(grp, weights=(pcs > 0.5).astype(float))[grp]
    # the two most confident OTHER candidates of the same S1
    # rank 0 -> others are ranks 1,2 ; rank 1 -> ranks 0,2 ; rank >= 2 -> ranks 0,1
    sg = start[grp]
    o1 = np.where(rank == 0, sg + 1, sg)
    o2 = np.where(rank <= 1, sg + 2, sg + 1)
    has1, has2 = size[grp] > 1, size[grp] > 2
    o1 = np.where(has1, o1, 0); o2 = np.where(has2, o2, 0)
    pc_o1 = np.where(has1, pcs[o1], np.nan); pc_o2 = np.where(has2, pcs[o2], np.nan)
    p_o1 = np.where(has1, ps[o1], -1); p_o2 = np.where(has2, ps[o2], -1)

    F = pd.DataFrame({"rank_q": rank, "n_q": size[grp], "top1": top1, "top2": top2, "top3": top3,
                      "sum_q": sum_q, "n_q05": n05, "gap_top": top1 - pcs,
                      "pc_rel": np.where(top1 > 0, pcs / np.maximum(top1, 1e-12), 0.0),
                      "pc_other1": pc_o1, "pc_other2": pc_o2})

    # ---- record context (competition for p across S1s)
    op = np.lexsort((-pcs, ps))                     # (p, pc desc) order over the q-sorted arrays
    pp, pv = ps[op], pcs[op]
    st = np.r_[0, np.flatnonzero(pp[1:] != pp[:-1]) + 1]
    sz = np.diff(np.r_[st, n])
    g2 = np.repeat(np.arange(len(st)), sz)
    pmax, sec, nqp, rp = (np.empty(n) for _ in range(4))
    pmax[op] = pv[st][g2]
    sec[op] = np.where(sz > 1, pv[np.minimum(st + 1, n - 1)], 0.0)[g2]
    nqp[op] = sz[g2]
    rp[op] = np.arange(n) - st[g2]
    is_best = rp == 0                               # this S1 holds the record's highest score
    best_other = np.where(is_best, sec, pmax)
    F["pmax_other"] = best_other
    F["n_q_for_p"] = nqp
    F["is_best_p"] = is_best.astype(np.float32)
    F["pc_minus_bestother"] = pcs - best_other

    # ---- embedding features (raw production + normalized)
    torch.set_num_threads(32)
    for tag, path in (("raw", cfg.art / "embed" / f"{split}.npy"), ("norm", ROM / "artifacts/full/embed" / f"{split}.npy")):
        t1 = time.time()
        if tag == "norm":                             # never read a split that is still being written
            prog = json.loads((path.parent / f"{split}.progress.json").read_text())
            n_rec = np.load(path, mmap_mode="r").shape[0]
            assert len(prog) == -(-n_rec // 200_000), f"normalized {split} embeddings incomplete"
        E = torch.from_numpy(np.load(path))           # fully in RAM (fp16)
        LOG.info("[%s] loaded %s embeddings %s in %.0fs", split, tag, tuple(E.shape), time.time() - t1)
        cqp, c1, c2 = (np.full(n, np.nan, np.float32) for _ in range(3))
        for s in range(0, n, 1_000_000):
            sl = slice(s, min(s + 1_000_000, n))
            Ep = E.index_select(0, torch.from_numpy(ps[sl])).float()
            Eq = E.index_select(0, torch.from_numpy(qs[sl])).float()
            cqp[sl] = (Ep * Eq).sum(1).numpy()
            del Eq
            for arr, po in ((c1, p_o1), (c2, p_o2)):
                idx = po[sl]
                ok = idx >= 0
                Eo = E.index_select(0, torch.from_numpy(np.where(ok, idx, 0))).float()
                v = (Ep * Eo).sum(1).numpy()
                arr[sl] = np.where(ok, v, np.nan)
                del Eo
            del Ep
        F[f"cos_{tag}_qp"], F[f"c1_{tag}"], F[f"c2_{tag}"] = cqp, c1, c2
        del E
        LOG.info("[%s] %s embedding features done in %.0fs", split, tag, time.time() - t1)

    # ---- back to original row order
    inv = np.empty(n, np.int64); inv[o] = np.arange(n)
    F = F.iloc[inv].reset_index(drop=True)
    F.insert(0, "p2", d["p2"].values); F.insert(1, "pc", d["pc"].values)
    F.insert(0, "p", d["p"].values); F.insert(0, "q", d["q"].values)
    if split == "train":
        F["y"], F["fold"] = d["y"].values, d["fold"].values
    for c in F.columns:
        if F[c].dtype == np.float64:
            F[c] = F[c].astype(np.float32)
    assert (F["q"].values == d["q"].values).all() and (F["p"].values == d["p"].values).all()
    save_parquet(F, WORK / f"{split}_feats.parquet")
    LOG.info("[%s] features saved (%d cols) in %.0fs", split, F.shape[1], time.time() - t0)


def train() -> None:
    tr = pd.read_parquet(WORK / "train_feats.parquet")
    X = tr[FEATS].to_numpy(np.float32)
    meta = tr[["q", "p", "y", "fold"]]
    oof, models, imp, stats = cv_train(cfg, X, meta, FEATS, "s3")
    LOG.info("stage3 OOF pair metrics: %s | gbm2 OOF: %s", pair_metrics(meta["y"].values, oof),
             pair_metrics(meta["y"].values, tr["p2"].values))
    LOG.info("stage3 feature importance (gain):\n%s", imp.round(1).to_string())
    save_parquet(meta.assign(p3=oof), WORK / "oof3.parquet")
    del X, tr
    te = pd.read_parquet(WORK / "test_feats.parquet")
    pt = predict_models(models, te[FEATS].to_numpy(np.float32), cfg.n_jobs)
    save_parquet(te[["q", "p"]].assign(p3=pt), WORK / "test3.parquet")
    LOG.info("stage3 test predictions saved: %s rows, mean %.5f", f"{len(pt):,}", float(pt.mean()))


def decide() -> None:
    oof = pd.read_parquet(WORK / "oof3.parquet")
    rec = load_records(cfg, "train", ["entity_id", "src", "country"])
    tu = truth_uids(cfg, rec)
    s1 = np.load(cfg.art / "gbm" / "eval_s1.npy")
    tu = tu[tu["q"].isin(set(s1))]
    oof["pc"] = crossfit_calibrate(oof["p3"].values, oof["y"].values, oof["fold"].values)
    res, _ = decide_variants(oof, tu, s1, rec["country"], cfg)
    summary = {k: round(v["all"]["f05"], 5) for k, v in res.items()}
    best = max(res, key=lambda k: res[k]["all"]["f05"])
    LOG.info("STAGE3 decision variants (OOF macro F0.5): %s -> best %s | baseline (gbm2 big-xenc) %.5f",
             summary, best, BASELINE_OOF)
    test = pd.read_parquet(WORK / "test3.parquet")
    test["pc"] = fit_isotonic(oof["p3"].values, oof["y"].values).predict(test["p3"].values)
    pred = apply_variant(test, best, res[best].get("threshold"), cfg)
    dec = S3ROOT / "artifacts/full/decide"
    dec.mkdir(parents=True, exist_ok=True)
    save_parquet(pred, dec / "test_matches.parquet")
    (dec / "choice.json").write_text(json.dumps({"score_col": "p3", "variant": best,
                                                 "threshold": res[best].get("threshold"), "results": res},
                                                indent=1, default=str))
    LOG.info("stage3 test matches: %s pairs over %s S1 (variant %s)", f"{len(pred):,}", f"{pred['q'].nunique():,}", best)


def predict() -> None:
    art = S3ROOT / "artifacts/full"
    for sub in ("prepare", "block", "gbm"):          # read-only views of production inputs to predict.run
        link = art / sub
        if not link.exists():
            os.symlink(cfg.art / sub, link)
    if not (S3ROOT / "utils").exists():
        os.symlink(PROD / "utils", S3ROOT / "utils")   # official validator
    import predict as P
    c3 = load_config("full", root=S3ROOT, data_dir=DATA)
    P.run(c3, force=True)


if __name__ == "__main__":
    phases = sys.argv[1:] or ["all"]
    if "all" in phases:
        phases = ["features", "train", "decide", "predict"]
    for ph in phases:
        LOG.info("=========== stage3 phase %s ===========", ph)
        if ph == "features":
            build_features("train"); build_features("test")
        elif ph == "train":
            train()
        elif ph == "decide":
            decide()
        elif ph == "predict":
            predict()
    LOG.info("stage3 phases %s finished", phases)
