"""Re-run leave-one-country-out, but evaluate ALL FOUR decision variants per direction
(the original loco() in train_gbm.py only ever tested expectedF+assign).

Trains once per direction (same as the original LOCO), reusing the exact same fit/calibrate
logic, then applies decide_variants() to get threshold / threshold+assign / expectedF /
expectedF+assign side by side on the SAME held-out cross-country predictions. This directly
answers: which decision rule is most ROBUST under genuine country shift, rather than which
scores best on in-distribution pooled OOF (which is what picked expectedF+assign originally,
and which we now know does not necessarily transfer -- the France resubmission with
threshold+assign scored 0.950, *worse* than expectedF+assign's 0.953).
"""
import json
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from config import load_config
from decide import decide_variants, fit_isotonic
from prepare import load_records, load_truth
from train_gbm import fit_one, load_train_matrix, truth_keys, eval_s1, _seed

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")

rec = load_records(cfg, "train", ["entity_id", "src", "country"])
tk = truth_keys(cfg, rec)
X, meta, feats = load_train_matrix(cfg, rec, tk)
s1_all = eval_s1(cfg, rec)
country_arr = rec["country"].values
mc = country_arr[meta["q"].values]
truth = pd.DataFrame({"q": tk // (1 << 32), "p": tk % (1 << 32)})
country_series = rec["country"]

out_rows = []
countries = sorted(pd.unique(mc))
for a in countries:
    for b in countries:
        if a == b:
            continue
        ia, ib = np.flatnonzero(mc == a), np.flatnonzero(mc == b)
        qa = meta["q"].values[ia]
        ua = np.unique(qa)
        rng = np.random.default_rng(_seed(cfg, f"loco{a}"))
        cal = np.isin(qa, rng.choice(ua, max(1, int(0.2 * len(ua))), replace=False))
        print(f"== training on {a} (excl. calib holdout), evaluating on {b} ==", flush=True)
        m = fit_one(cfg, X[ia[~cal]], meta["y"].values[ia[~cal]], qa[~cal], feats, f"locov{a}")
        iso = fit_isotonic(m.predict(X[ia[cal]], num_threads=cfg.n_jobs), meta["y"].values[ia[cal]])
        pb = m.predict(X[ib], num_threads=cfg.n_jobs)
        d = meta.iloc[ib][["q", "p"]].assign(pc=iso.predict(pb))
        sb = s1_all[country_arr[s1_all] == b]
        tb = truth[truth["q"].isin(set(sb))]
        res, _ = decide_variants(d, tb, sb, country_series, cfg)
        for variant, r in res.items():
            row = {"train": a, "eval": b, "variant": variant, **r["all"]}
            print("  ", row, flush=True)
            out_rows.append(row)

with open("/path/to/amazon_ml_2026/artifacts/full/gbm/loco_variants.json", "w") as f:
    json.dump(out_rows, f, indent=1, default=float)
print("\nSUMMARY (macro F0.5 per variant per direction):")
tab = pd.DataFrame(out_rows)
print(tab.pivot(index="variant", columns=["train", "eval"], values="f05").round(5).to_string())
print("\nmean across both LOCO directions per variant:")
print(tab.groupby("variant")["f05"].mean().sort_values(ascending=False).round(5).to_string())
