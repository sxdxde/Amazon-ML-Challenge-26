"""gbm2 with France pseudo-labels (unsupervised domain adaptation to the unseen country).

Same as train_gbm.run_stage2 (same 108 features, same folds, same cv_train/fit_one), except that every
fold's training data also contains France test pairs the current best model is confident about
(calibrated stage-3 score >= HI -> 1, <= LO -> 0). Train OOF stays out-of-fold w.r.t. train labels, so
decide.run calibrates and picks the variant exactly as usual. Everything is written to runs/pseudo;
production artifacts are read through read-only symlinks.
  python gbm2_pseudo.py HI LO
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/amazon_ml_2026/code/business_entity_resolution/src")
import numpy as np
import pandas as pd

from config import load_config
from decide import fit_isotonic
from features import iter_feature_parts
from io_utils import LOG, Timer, lib_versions, mark_done, save_parquet, setup_logging
from prepare import load_records
from train_gbm import cv_train, fit_one, load_train_matrix, pair_metrics, predict_models, stage2_extra, truth_keys

HI, LO = float(sys.argv[1]), float(sys.argv[2])
PROD = Path("/path/to/amazon_ml_2026")
ROOT = PROD / "runs" / "pseudo"
DATA = PROD / "dataset"
prod = load_config("full", root=PROD, data_dir=DATA)
cfg = load_config("full", root=ROOT, data_dir=DATA)
art = cfg.art
art.mkdir(parents=True, exist_ok=True)
for sub in ("prepare", "embed", "block", "features", "gbm", "xenc"):
    if not (art / sub).exists():
        os.symlink(prod.art / sub, art / sub)
if not (ROOT / "utils").exists():
    os.symlink(PROD / "utils", ROOT / "utils")
setup_logging(ROOT / "gbm2_pseudo.log")
out = art / "gbm2"
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- train matrix (identical to run_stage2)
rec = load_records(prod, "train", ["entity_id", "src", "country"])
tk = truth_keys(prod, rec)
oof1 = pd.read_parquet(prod.art / "gbm" / "oof.parquet")
X1, meta, feats1 = load_train_matrix(prod, rec, tk)
assert (meta["q"].values == oof1["q"].values).all() and (meta["p"].values == oof1["p"].values).all()
ex = stage2_extra(oof1[["q", "p", "p1"]], pd.read_parquet(prod.art / "xenc" / "oof.parquet"))
feats = feats1 + list(ex.columns)
X = np.hstack([X1, ex.to_numpy(np.float32)])
del X1, ex, oof1
LOG.info("train: %s rows, %d features", f"{len(X):,}", len(feats))

# ---------------------------------------------------------------- test matrix + France pseudo-labels
te1 = pd.read_parquet(prod.art / "gbm" / "test_pred.parquet")
ex_t = stage2_extra(te1[["q", "p", "p1"]], pd.read_parquet(prod.art / "xenc" / "test_pred.parquet"))
parts = []
for off, df in iter_feature_parts(prod, "test"):
    n = len(df)
    assert (df["q"].values == te1["q"].values[off:off + n]).all()
    parts.append(np.hstack([df[feats1].to_numpy(np.float32), ex_t.iloc[off:off + n].to_numpy(np.float32)]))
Xt = np.vstack(parts); del parts, ex_t
te_country = load_records(prod, "test", ["country"])["country"].values[te1["q"].values]

# pseudo-label source: the production gbm2 (big-xenc) scores that produced LB 0.981, calibrated exactly as
# decide.run does (isotonic fit on the full train OOF)
o2 = pd.read_parquet(prod.art / "gbm2" / "oof.parquet", columns=["y", "p2"])
t2 = pd.read_parquet(prod.art / "gbm2" / "test_pred.parquet")
assert (t2["q"].values == te1["q"].values).all() and (t2["p"].values == te1["p"].values).all()
pc = fit_isotonic(o2["p2"].values, o2["y"].values).predict(t2["p2"].values)
del o2, t2
fr = te_country == "France"
conf = fr & ((pc >= HI) | (pc <= LO))
ips = np.flatnonzero(conf)
yps = (pc[ips] >= HI).astype(np.float32)
LOG.info("France pseudo-labels: %s pairs (%.1f%% of France), %s positive, HI=%.3f LO=%.4f",
         f"{len(ips):,}", conf.sum() / fr.sum() * 100, f"{int(yps.sum()):,}", HI, LO)

# ---------------------------------------------------------------- cv with pseudo rows added to every fold
y, fold, q = meta["y"].values, meta["fold"].values, meta["q"].values
qps = te1["q"].values[ips].astype(np.int64) + 10**9          # disjoint group ids for test S1s
oof = np.zeros(len(X), np.float32)
models, stats = [], []
for f in range(cfg.n_folds):
    tr, va = fold != f, fold == f
    with Timer(f"gbm2ps fold {f}"):
        Xf = np.concatenate([X[tr], Xt[ips]])
        m = fit_one(cfg, Xf, np.concatenate([y[tr], yps]), np.concatenate([q[tr], qps]), feats, f"gbm2{f}")
        del Xf
        oof[va] = m.predict(X[va], num_threads=cfg.n_jobs)
    st = {"fold": f, "best_iter": m.best_iteration, **pair_metrics(y[va], oof[va])}
    LOG.info("gbm2ps fold %d: %s", f, {k: round(v, 5) if isinstance(v, float) else v for k, v in st.items()})
    stats.append(st); models.append(m)
save_parquet(meta.assign(p2=oof), out / "oof.parquet")
LOG.info("OOF pair metrics: %s", pair_metrics(y, oof))
with Timer("predict test"):
    pt = predict_models(models, Xt, cfg.n_jobs)
save_parquet(te1[["q", "p"]].assign(p2=pt), out / "test_pred.parquet")
for c in ("France", "India", "US"):
    m_ = te_country == c
    LOG.info("test mean p2 %s: %.5f", c, float(pt[m_].mean()))
mark_done(out, {"pseudo_hi": HI, "pseudo_lo": LO, "n_pseudo": int(len(ips)), "versions": lib_versions()})
LOG.info("gbm2 pseudo DONE")
