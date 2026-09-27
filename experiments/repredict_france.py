"""Re-predict France test pairs with the ALREADY-TRAINED gbm (stage1) and gbm2 (stage2) fold
models, using the freshly recomputed France features (crs_ fix + reblocked candidates). No
retraining -- France was never in train data, so this is pure inference. Merges the new France
rows into gbm/test_pred.parquet and gbm2/test_pred.parquet, replacing the old France rows
entirely (candidate SET changed slightly after reblocking, so row count differs).
"""
import json
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import xgboost as xgb

import gbm_backend
from config import load_config
from features import feat_dir
from prepare import load_records
from train_gbm import predict_models, stage2_extra

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
N_FOLDS = 5


def load_fold_models(art_dir, feats):
    models = []
    for i in range(N_FOLDS):
        p = art_dir / f"fold{i}.json"
        if not p.exists():
            break
        b = xgb.Booster()
        b.load_model(str(p))
        models.append(gbm_backend.Model("xgboost", b, feats))
    print(f"  loaded {len(models)} fold models from {art_dir}", flush=True)
    return models


info = load_records(cfg, "test", ["entity_id", "src", "country"])
fr_files = sorted(feat_dir(cfg, "test").glob("France_*.parquet"))
print(f"France feature files: {len(fr_files)}", flush=True)

# ---------------------------------------------------------------- stage 1
print("== stage 1: predicting France with existing gbm fold models ==", flush=True)
feats1 = json.loads((cfg.art / "gbm" / "features.json").read_text())
models1 = load_fold_models(cfg.art / "gbm", feats1)

qp_parts, p1_parts = [], []
for f in fr_files:
    df = pd.read_parquet(f, columns=["q", "p"] + feats1)
    p1_parts.append(predict_models(models1, df[feats1].to_numpy(np.float32), cfg.n_jobs))
    qp_parts.append(df[["q", "p"]])
    del df
qp_fr = pd.concat(qp_parts, ignore_index=True)
p1_fr = np.concatenate(p1_parts)
print(f"  new France stage-1 predictions: {len(qp_fr):,} rows, mean p1={p1_fr.mean():.5f}", flush=True)

te1_path = cfg.art / "gbm" / "test_pred.parquet"
te1_old = pd.read_parquet(te1_path)
ctry_of_q_1 = info["country"].values[te1_old["q"].values]
te1_new = pd.concat([te1_old[ctry_of_q_1 != "France"], qp_fr.assign(p1=p1_fr)], ignore_index=True)
te1_new.to_parquet(te1_path, index=False)
print(f"  saved {te1_path}: {len(te1_new):,} rows (was {len(te1_old):,})", flush=True)
del models1, te1_old, te1_new

# ---------------------------------------------------------------- stage 2
print("== stage 2: predicting France with existing gbm2 fold models ==", flush=True)
ex_cols = list(stage2_extra(pd.DataFrame({"q": [0], "p": [0], "p1": [0.5]}), None).columns)
feats2 = feats1 + ex_cols
models2 = load_fold_models(cfg.art / "gbm2", feats2)

fr_p1 = qp_fr.assign(p1=p1_fr)          # France's fresh stage-1 predictions, file order preserved
ex_fr = stage2_extra(fr_p1, None)        # context features computed over France's full candidate set

off = 0
p2_parts = []
for f in fr_files:
    df = pd.read_parquet(f, columns=["q", "p"] + feats1)
    n = len(df)
    assert (df["q"].values == qp_fr["q"].values[off:off + n]).all()
    Xp = np.hstack([df[feats1].to_numpy(np.float32), ex_fr.iloc[off:off + n].to_numpy(np.float32)])
    p2_parts.append(predict_models(models2, Xp, cfg.n_jobs))
    off += n
    del df, Xp
p2_fr = np.concatenate(p2_parts)
print(f"  new France stage-2 predictions: {len(p2_fr):,} rows, mean p2={p2_fr.mean():.5f}", flush=True)

te2_path = cfg.art / "gbm2" / "test_pred.parquet"
te2_old = pd.read_parquet(te2_path)
ctry_of_q_2 = info["country"].values[te2_old["q"].values]
te2_new = pd.concat([te2_old[ctry_of_q_2 != "France"], qp_fr.assign(p2=p2_fr)], ignore_index=True)
te2_new.to_parquet(te2_path, index=False)
print(f"  saved {te2_path}: {len(te2_new):,} rows (was {len(te2_old):,})", flush=True)
print("DONE", flush=True)
