"""Cascade pruning: the stage-1 GBM is the last blocking / candidate-generation filter.

Blocking (blocking.py) retrieves a broad, recall-oriented candidate list (~38 per Source-1 entity). The
cheap stage-1 GBM scores all of them; only candidates with stage-1 rank < cfg.prune_k within their S1
AND stage-1 probability >= cfg.prune_min_p1 reach the final matcher (gbm2 inference, the expected-F0.5
decision). This final set is written to candidate_pairs.tsv (5.16 per S1 on test, 8.9 M pairs vs 65.9 M).

Final setting top-15 / p1 >= 0.003: train OOF candidates/S1 38.7 -> 4.55 with macro F0.5 unchanged (0.98773);
the floor sits below the lowest stage-1 score of any pair the final matcher selects on test (0.0042) and the
cap above the deepest selected stage-1 rank (12), so the cascade removes nothing the matcher would pick.
(top-10 / p1 >= 0.01 gives 4.1 per S1 on train, also loss-free on OOF; top-6 costs -0.0014, top-4 -0.017.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def keep_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Boolean mask over rows of df[q, p1] (any row order): True = candidate reaches the final matcher."""
    m = np.ones(len(df), bool)
    if cfg.prune_min_p1 > 0:
        m &= df["p1"].values >= cfg.prune_min_p1
    if cfg.prune_k > 0:
        order = np.lexsort((-df["p1"].values, df["q"].values))       # by S1, stage-1 score desc (stable)
        qs = df["q"].values[order]
        start = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
        rank = np.empty(len(df), np.int64)
        rank[order] = np.arange(len(df)) - np.repeat(start, np.diff(np.r_[start, len(df)]))
        m &= rank < cfg.prune_k
    return m


def kept_pairs(cfg: Config, split: str) -> pd.DataFrame:
    """(q, p) pairs of `split` that reach the final matcher, from the stage-1 predictions."""
    name = "oof.parquet" if split == "train" else "test_pred.parquet"
    s1 = pd.read_parquet(cfg.art / "gbm" / name, columns=["q", "p", "p1"])
    return s1.loc[keep_mask(s1, cfg), ["q", "p"]].reset_index(drop=True)
