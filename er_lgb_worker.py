"""LightGBM CV / test / leave-country-out in a clean subprocess.

Why a subprocess: on macOS the notebook kernel can hold other OpenMP runtimes (torch's bundled
libomp, scikit-learn's libomp). A second runtime next to LightGBM's Homebrew libomp segfaults
inside __kmp_suspend_initialize_thread as soon as LightGBM starts its thread pool. This process
imports only numpy, pandas/pyarrow and lightgbm, so exactly one OpenMP runtime is loaded.

Usage (called by notebook cell 10):  python er_lgb_worker.py <job_dir>
job_dir holds job.json, X.npy, y.npy, folds.npy, pair_ctry.npy; outputs are written back to it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(msg, flush=True)


def main(job_dir):
    import lightgbm as lgb  # first (and only) OpenMP runtime in this process
    import pandas as pd

    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    params, feats = job["params"], job["feats"]
    X = np.load(job_dir / "X.npy")
    y = np.load(job_dir / "y.npy")
    folds = np.load(job_dir / "folds.npy")
    pair_ctry = np.load(job_dir / "pair_ctry.npy")
    log(f"worker: X {X.shape}, lightgbm {lgb.__version__}, threads {params.get('num_threads')}")

    full = lgb.Dataset(X, y, free_raw_data=False, params={"max_bin": 255, "verbose": -1}).construct()
    oof = np.zeros(len(y), np.float32)
    best_iters, imps = [], []
    for f in np.unique(folds):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        t0 = time.time()
        m = lgb.train(params, full.subset(tr), job["rounds"], valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(job["es_rounds"], verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        m.save_model(str(job_dir / f"fold{int(f)}.txt"), num_iteration=m.best_iteration)
        best_iters.append(int(m.best_iteration))
        imps.append(m.feature_importance("gain"))
        log(f"    fold {int(f)}: best_iter={m.best_iteration} ({time.time() - t0:,.0f}s)")
    np.save(job_dir / "oof.npy", oof)
    np.save(job_dir / "importance.npy", np.mean(imps, 0))
    del full

    models = [lgb.Booster(model_file=str(job_dir / f"fold{int(f)}.txt")) for f in np.unique(folds)]
    preds = []
    for p in job["test_files"]:
        Xt = pd.read_parquet(p, columns=feats).to_numpy(np.float32)
        preds.append(np.mean([m.predict(Xt) for m in models], axis=0).astype(np.float32))
        del Xt
    np.save(job_dir / "test.npy", np.concatenate(preds) if preds else np.zeros(0, np.float32))
    log(f"    test: {sum(len(p) for p in preds):,} pairs scored")

    n_it = max(10, int(np.median(best_iters)))
    for i, c in enumerate(job["loco_countries"]):
        trm, vam = pair_ctry != c, pair_ctry == c
        t0 = time.time()
        m = lgb.train(params, lgb.Dataset(X[trm], y[trm], params={"verbose": -1}), n_it)
        np.save(job_dir / f"loco_{i}.npy", m.predict(X[vam]).astype(np.float32))
        log(f"    LOCO eval on {c}: {n_it} rounds ({time.time() - t0:,.0f}s)")
    (job_dir / "info.json").write_text(json.dumps({"best_iters": best_iters, "loco_rounds": n_it}))
    log("worker: done")


if __name__ == "__main__":
    main(sys.argv[1])
