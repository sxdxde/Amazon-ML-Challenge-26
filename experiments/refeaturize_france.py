"""Recompute pairwise features for France test candidates ONLY, using the corrected addr_core
text and the freshly reblocked France candidates. Mirrors features.py run()'s per-country loop
exactly, restricted to France. Deletes only France's existing feature parts, leaves India/US.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

import features as F
from config import load_config
from io_utils import load_candidates, country_slug, save_parquet
from prepare import load_records

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
split = "test"
info = load_records(cfg, split, ["entity_id", "src", "country"])
n_total = len(info)
cand = load_candidates(cfg, split)
ctry_of_q = info["country"].values[cand["q"].values]
del info

ctry = "France"
d = F.feat_dir(cfg, split)
old_france_files = sorted(d.glob(f"{country_slug(ctry)}_*.parquet"))
print(f"deleting {len(old_france_files)} old France feature files", flush=True)
for f in old_france_files:
    f.unlink()

c = cand[ctry_of_q == ctry].reset_index(drop=True)
print(f"features {split} [{ctry}] {len(c):,} pairs", flush=True)
sub = load_records(cfg, split, F.REC_COLS, country=ctry)
A = F.build_arrays(sub)
lut = np.full(n_total, -1, np.int64)
lut[sub["uid"].values] = np.arange(len(sub))
F._IDF["name"] = F.idf_table(sub["name_core"])
F._IDF["addr"] = F.idf_table(sub["addr_core"])
F._A.clear()
F._A.update(A)
local = pd.DataFrame({"q": lut[c["q"].values], "p": lut[c["p"].values], "cos_dense": c["cos_dense"].values})
assert (local[["q", "p"]].values >= 0).all()
ctx = F.context_features(local, A, cfg.n_jobs)
c = pd.concat([c, ctx], axis=1)
n_rows = 0
for i, s in enumerate(range(0, len(c), F.CHUNK)):
    ch = c.iloc[s:s + F.CHUNK].reset_index(drop=True)
    ql, pl = lut[ch["q"].values], lut[ch["p"].values]
    feats = pd.concat([ch, F.string_features(A, ql, pl, cfg.n_jobs), F.py_features(ql, pl, cfg.n_jobs)], axis=1)
    for col in feats.columns:
        if feats[col].dtype == np.float64:
            feats[col] = feats[col].astype(np.float32)
    save_parquet(feats, d / f"{country_slug(ctry)}_{i:04d}.parquet")
    n_rows += len(feats)
print(f"wrote {n_rows:,} France feature rows to {d}", flush=True)
