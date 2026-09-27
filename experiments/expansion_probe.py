"""Probe: can blocking-missed true matches be reached as dense neighbours of an already-found true match?"""
import sys
sys.path.insert(0, "/path/to/amazon_ml_2026/code/business_entity_resolution/src")
import numpy as np
import pandas as pd
import torch

from config import load_config
from decide import truth_uids
from prepare import load_records

cfg = load_config("full", root="/path/to/amazon_ml_2026", data_dir="/path/to/amazon_ml_2026/dataset")
rec = load_records(cfg, "train", ["entity_id", "src", "country"])
N = len(rec)
tu = truth_uids(cfg, rec).astype(np.int64)
oof = pd.read_parquet(cfg.art / "gbm2" / "oof.parquet", columns=["q", "p", "y", "p2"])
kc = oof["q"].values.astype(np.int64) * N + oof["p"].values
tu["in_cand"] = np.isin(tu["q"].values * N + tu["p"].values, kc)
miss = tu[~tu["in_cand"]]
# anchor: the highest-scored TRUE candidate of the same S1 (a copy we would already have found)
anch = oof[oof["y"] == 1].sort_values(["q", "p2"], ascending=[True, False]).groupby("q").head(1).set_index("q")["p"]
miss = miss[miss["q"].isin(anch.index)].copy()
miss["a"] = anch.reindex(miss["q"].values).values
print(f"blocking-missed true pairs: {int((~tu.in_cand).sum()):,}; with an anchor match found: {len(miss):,}", flush=True)
rng = np.random.default_rng(0)
dev = "cuda"
for tag, path in (("raw", cfg.art / "embed" / "train.npy"),
                  ("norm", cfg.root / "runs/rom/artifacts/full/embed/train.npy")):
    E = np.load(path, mmap_mode="r")
    for ctry in ("India", "US"):
        pool = np.flatnonzero((rec["country"].values == ctry) & (rec["src"].values != 1))
        m = miss[rec["country"].values[miss["q"].values] == ctry]
        m = m.iloc[rng.choice(len(m), min(20000, len(m)), replace=False)]
        P = torch.from_numpy(np.asarray(E[pool], np.float16)).to(dev)
        A = torch.from_numpy(np.asarray(E[m["a"].values], np.float16)).to(dev)
        pos = np.searchsorted(pool, m["p"].values)
        ranks = np.empty(len(m), np.int64)
        for s in range(0, len(m), 64):
            sc = A[s:s + 64] @ P.T
            tgt = sc[torch.arange(sc.shape[0], device=dev), torch.from_numpy(pos[s:s + 64]).to(dev)]
            ranks[s:s + 64] = (torch.count_nonzero(sc > tgt[:, None], dim=1) - 1).clamp(min=0).cpu().numpy()  # -1: anchor itself
        del P, A
        torch.cuda.empty_cache()
        print(f"[{tag:4s}] {ctry:5s} n={len(m):,} | missed match within anchor's top-5: {np.mean(ranks < 5)*100:5.1f}% "
              f"top-10: {np.mean(ranks < 10)*100:5.1f}% top-20: {np.mean(ranks < 20)*100:5.1f}% | median rank {int(np.median(ranks))}",
              flush=True)
