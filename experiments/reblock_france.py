"""Recompute blocking candidates for France ONLY (test split), using the already-tuned k
(k_choice.json, unchanged) and the corrected addr_core text (crs_ fix). Leaves India/US test
candidates untouched -- avoids redoing their multi-hour blocking runs.
"""
import sys
sys.path.insert(0, ".")
import json

import pandas as pd

from blocking import pass_build
from config import load_config, resolve_device
from io_utils import cand_path, save_parquet
from prepare import load_records
from embed import load_emb

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
device = resolve_device(cfg)
kvec = json.loads((cfg.art / "block" / "k_choice.json").read_text())["k"]
print("device:", device, "| tuned k:", kvec, flush=True)

meta_all = load_records(cfg, "test", ["entity_id", "src", "country"])
meta_fr = meta_all[meta_all["country"] == "France"]
emb = load_emb(cfg, "test")

cand_fr_new = pass_build(cfg, "test", meta_fr, emb, kvec, device)
print("new France candidate pairs:", len(cand_fr_new), flush=True)

old_path = cand_path(cfg.art, "test")
old = pd.read_parquet(old_path)
ctry_of_q = meta_all["country"].values[old["q"].values]
old_non_france = old[ctry_of_q != "France"]
print("old total:", len(old), "| old non-France kept:", len(old_non_france),
     "| old France dropped:", len(old) - len(old_non_france), flush=True)

merged = pd.concat([old_non_france, cand_fr_new], ignore_index=True)
merged["q"] = merged["q"].astype("int32")
merged["p"] = merged["p"].astype("int32")
save_parquet(merged, old_path)
print("saved merged candidates:", len(merged), "rows to", old_path, flush=True)
