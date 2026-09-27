"""CV splits.

* Main scheme: 5-fold grouped by S1 entity. Fold = hash(entity_id, seed) mod k, so every
  candidate pair of an S1 lands in the same fold, deterministic and independent of row order.
* Leave-one-country-out (LOCO): train on one country string, evaluate on another — the proxy for
  the unseen test country. Countries are discovered from the data (open set).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _hash01(ids: pd.Series, salt: str) -> np.ndarray:
    return np.array([int(hashlib.md5(f"{salt}|{x}".encode()).hexdigest()[:12], 16) for x in ids],
                    dtype=np.int64)


def s1_folds(s1_ids: pd.Series, n_folds: int, seed: int, salt: str = "fold") -> pd.Series:
    """Map each S1 id -> fold in [0, n_folds)."""
    uniq = pd.Series(pd.unique(s1_ids))
    folds = _hash01(uniq, f"{salt}{seed}") % n_folds
    return pd.Series(folds, index=uniq.values)


def s1_subsample(s1_ids: pd.Series, frac: float, seed: int, salt: str = "sub") -> set:
    """Deterministic subset of S1 ids of size ~frac."""
    if frac >= 1.0:
        return set(pd.unique(s1_ids))
    uniq = pd.Series(pd.unique(s1_ids))
    u = (_hash01(uniq, f"{salt}{seed}") % 1_000_000) / 1_000_000
    return set(uniq[u < frac])


def loco_pairs(countries: list[str]) -> list[tuple[str, str]]:
    """All ordered (train_country, eval_country) pairs among countries present in train."""
    return [(a, b) for a in countries for b in countries if a != b]
