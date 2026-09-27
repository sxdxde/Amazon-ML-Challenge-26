"""Complete gbm2 after the crash: the 5 fold models (with xenc) and oof.parquet are already
correct and saved. Only 'predict test (stage 2)' + mark_done need to be redone, now that
gbm/test_pred.parquet's row order is fixed. Reuses the official code path exactly
(iter_feature_parts, stage2_extra, predict_models) so behavior matches run_stage2 precisely.
"""
import json
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import xgboost as xgb

import gbm_backend
from config import load_config
from features import iter_feature_parts, feature_columns, load_features
from io_utils import lib_versions, mark_done, save_parquet
from train_gbm import stage2_extra, predict_models, pair_metrics

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
out = cfg.art / "gbm2"

feats1 = json.loads((cfg.art / "gbm" / "features.json").read_text())
xdir = cfg.art / "xenc"
use_x = cfg.use_xenc and (xdir / "_DONE.json").exists()
print("use_xenc:", use_x, flush=True)

ex_cols = list(stage2_extra(pd.DataFrame({"q": [0], "p": [0], "p1": [0.5]}),
                           pd.DataFrame({"q": [0], "p": [0], "xenc": [0.5]}) if use_x else None).columns)
feats = feats1 + ex_cols
print(f"stage 2: {len(feats)} features (xenc={use_x})", flush=True)

models = []
for i in range(5):
    b = xgb.Booster()
    b.load_model(str(out / f"fold{i}.json"))
    models.append(gbm_backend.Model("xgboost", b, feats))
print(f"loaded {len(models)} fold models", flush=True)

te1 = pd.read_parquet(cfg.art / "gbm" / "test_pred.parquet")
xenc_test = pd.read_parquet(xdir / "test_pred.parquet") if use_x else None
ex_t = stage2_extra(te1[["q", "p", "p1"]], xenc_test)
preds = np.empty(len(te1), np.float32)
for off, df in iter_feature_parts(cfg, "test"):
    n = len(df)
    assert (df["q"].values == te1["q"].values[off:off + n]).all(), f"misaligned at offset {off}"
    Xp = np.hstack([df[feats1].to_numpy(np.float32), ex_t.iloc[off:off + n].to_numpy(np.float32)])
    preds[off:off + n] = predict_models(models, Xp, cfg.n_jobs)
    del df, Xp
save_parquet(te1[["q", "p"]].assign(p2=preds), out / "test_pred.parquet")
print(f"saved gbm2/test_pred.parquet: {len(preds):,} rows, mean p2={preds.mean():.5f}", flush=True)

oof = pd.read_parquet(out / "oof.parquet")
pm = pair_metrics(oof["y"].values, oof["p2"].values)
auc, pr = pm["auc"], pm["pr_auc"]
mark_done(out, {"n_features": len(feats), "use_xenc": use_x, "versions": lib_versions(),
                "auc": auc, "pr_auc": pr, "completed_by": "complete_gbm2.py (post-crash recovery)"})
print("DONE, gbm2 marked complete", flush=True)
