"""Fix the row-order bug: gbm/test_pred.parquet and gbm2/test_pred.parquet must be ordered
France, India, US (matching iter_feature_parts' alphabetical file iteration), but my earlier
France-only repredict script appended France's rows at the end instead. Stable-sort each file
back into the correct country block order, preserving each country's internal row order.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from config import load_config
from prepare import load_records

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
info = load_records(cfg, "test", ["entity_id", "src", "country"])
order_of = {"France": 0, "India": 1, "US": 2}

for stage in ("gbm", "gbm2"):
    p = cfg.art / stage / "test_pred.parquet"
    df = pd.read_parquet(p)
    ctry = info["country"].values[df["q"].values]
    key = np.array([order_of[c] for c in ctry])
    idx = np.argsort(key, kind="stable")
    df2 = df.iloc[idx].reset_index(drop=True)
    ctry2 = info["country"].values[df2["q"].values]
    change = [0] + [i for i in range(1, len(ctry2)) if ctry2[i] != ctry2[i - 1]]
    print(f"{stage}: fixed order blocks:", [(ctry2[i], i) for i in change[:10]], flush=True)
    df2.to_parquet(p, index=False)
    print(f"{stage}: saved {len(df2):,} rows to {p}", flush=True)
