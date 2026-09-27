"""Does pseudo-labeling the unseen country help? Validate with leave-one-country-out on stage-1 features.

For a -> b: (1) fit on a, calibrate on held-out a S1s, score b  -> macro F0.5 on b (true labels)
            (2) pseudo-label b's confident pairs (pc >= HI -> 1, pc <= LO -> 0), refit on a + pseudo-b,
                SAME calibration set (real labels of a), score b -> macro F0.5 on b
Read-only w.r.t. all artifacts.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/amazon_ml_2026/code/business_entity_resolution/src")
import numpy as np
import pandas as pd

from config import load_config
from decide import assign_exclusive, evaluate, fit_isotonic, select_expected_f
from io_utils import LOG, setup_logging
from prepare import load_records
from train_gbm import _seed, fit_one, load_train_matrix, pair_metrics, truth_keys

OUT = Path("/path/to/amazon_ml_2026/runs/stage3b")
setup_logging(OUT / "loco_pseudo.log")
cfg = load_config("full", root="/path/to/amazon_ml_2026", data_dir="/path/to/amazon_ml_2026/dataset")
SETTINGS = [(0.97, 0.03)]

rec = load_records(cfg, "train", ["entity_id", "src", "country"])
tk = truth_keys(cfg, rec)
X, meta, feats = load_train_matrix(cfg, rec, tk)
s1_all = np.load(cfg.art / "gbm" / "eval_s1.npy")
country = rec["country"].values
mc = country[meta["q"].values]
truth = pd.DataFrame({"q": tk // (1 << 32), "p": tk % (1 << 32)})
y = meta["y"].values
res = []


def score(pb, iso, ib, b):
    d = meta.iloc[ib][["q", "p"]].assign(pc=iso.predict(pb))
    pred = select_expected_f(assign_exclusive(d, "pc"), "pc", cfg.mc_max_cands, cfg.decide_chunk)
    sb = s1_all[country[s1_all] == b]
    ev = evaluate(pred, truth[truth["q"].isin(set(sb))], sb, None)["all"]
    return ev, d["pc"].values


for a, b in (("US", "India"),):
    ia, ib = np.flatnonzero(mc == a), np.flatnonzero(mc == b)
    qa = meta["q"].values[ia]
    ua = np.unique(qa)
    rng = np.random.default_rng(_seed(cfg, f"loco{a}"))
    cal = np.isin(qa, rng.choice(ua, max(1, int(0.2 * len(ua))), replace=False))
    fa = ia[~cal]
    m0 = fit_one(cfg, X[fa], y[fa], qa[~cal], feats, f"loco{a}")
    iso0 = fit_isotonic(m0.predict(X[ia[cal]], num_threads=cfg.n_jobs), y[ia[cal]])
    pb0 = m0.predict(X[ib], num_threads=cfg.n_jobs)
    ev0, pc0 = score(pb0, iso0, ib, b)
    LOG.info("LOCO %s->%s BASELINE macroF0.5=%.5f singleton_acc=%.4f nonsingleton=%.5f pairPR=%.5f",
             a, b, ev0["f05"], ev0["singleton_acc"], ev0["nonsingleton_f05"], pair_metrics(y[ib], pb0)["pr_auc"])
    for hi, lo in SETTINGS:
        conf = (pc0 >= hi) | (pc0 <= lo)
        ib_ps = ib[conf]
        yps = (pc0[conf] >= hi).astype(np.float32)
        acc = float((yps == y[ib_ps]).mean())
        Xf = np.concatenate([X[fa], X[ib_ps]])
        yf = np.concatenate([y[fa], yps])
        gf = np.concatenate([qa[~cal], meta["q"].values[ib_ps]])
        m1 = fit_one(cfg, Xf, yf, gf, feats, f"loco{a}ps")
        del Xf
        iso1 = fit_isotonic(m1.predict(X[ia[cal]], num_threads=cfg.n_jobs), y[ia[cal]])
        pb1 = m1.predict(X[ib], num_threads=cfg.n_jobs)
        ev1, _ = score(pb1, iso1, ib, b)
        LOG.info("LOCO %s->%s PSEUDO(hi=%.2f,lo=%.3f; %.1f%% of b rows, pseudo-label acc %.4f) macroF0.5=%.5f "
                 "(delta %+.5f) singleton_acc=%.4f nonsingleton=%.5f", a, b, hi, lo, conf.mean() * 100, acc,
                 ev1["f05"], ev1["f05"] - ev0["f05"], ev1["singleton_acc"], ev1["nonsingleton_f05"])
        res.append({"a": a, "b": b, "hi": hi, "lo": lo, "base": ev0["f05"], "pseudo": ev1["f05"],
                    "delta": ev1["f05"] - ev0["f05"], "pseudo_label_acc": acc})
(OUT / "loco_pseudo.json").write_text(json.dumps(res, indent=1))
LOG.info("DONE %s", res)
