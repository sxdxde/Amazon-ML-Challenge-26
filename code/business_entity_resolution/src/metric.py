"""Macro-averaged F0.5 exactly as defined by the challenge.

Per S1 entity:
  * truth empty & pred empty      -> 1.0
  * truth empty & pred non-empty  -> 0.0
  * truth non-empty & pred empty  -> 0.0
  * otherwise F_beta = (1+b^2) P R / (b^2 P + R), b = 0.5
Averaged over ALL S1 entities in the evaluation set.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

BETA2 = 0.25


def f_beta_single(pred: set, truth: set, beta2: float = BETA2) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + beta2) * p * r / (beta2 * p + r)


def macro_f05(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> float:
    """Average over every S1 key in `truth` (missing preds count as empty)."""
    scores = [f_beta_single(set(pred.get(k, ())), set(v)) for k, v in truth.items()]
    return float(np.mean(scores)) if scores else float("nan")


def f05_from_counts(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Vectorised per-entity F0.5 from counts: F = 1.25 TP / (0.25 T + K), with the singleton rule."""
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    denom = BETA2 * n_true + n_pred
    f = np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-12), 0.0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, f)


def per_entity_scores(pred_pairs: pd.DataFrame, truth_pairs: pd.DataFrame, s1_all: pd.Series) -> pd.DataFrame:
    """Vectorised per-S1 F0.5.

    pred_pairs / truth_pairs: DataFrames with columns [s1, id] (one row per predicted / true link).
    s1_all: every S1 in the evaluation set.
    Returns a DataFrame indexed by s1 with tp, n_pred, n_true, f.
    """
    idx = pd.Index(pd.unique(s1_all), name="s1")
    n_pred = pred_pairs.groupby("s1").size().reindex(idx, fill_value=0)
    n_true = truth_pairs.groupby("s1").size().reindex(idx, fill_value=0)
    hit = pred_pairs.merge(truth_pairs, on=["s1", "id"], how="inner")
    tp = hit.groupby("s1").size().reindex(idx, fill_value=0)
    out = pd.DataFrame({"tp": tp.values, "n_pred": n_pred.values, "n_true": n_true.values}, index=idx)
    out["f"] = f05_from_counts(out.tp, out.n_pred, out.n_true)
    return out


def summarize(scores: pd.DataFrame, country: pd.Series | None = None) -> dict:
    """Macro F0.5 overall / per country + singleton accuracy + non-singleton F0.5."""
    def block(s: pd.DataFrame) -> dict:
        single = s.n_true == 0
        return {"n": int(len(s)), "f05": float(s.f.mean()),
                "singleton_acc": float(s.f[single].mean()) if single.any() else float("nan"),
                "nonsingleton_f05": float(s.f[~single].mean()) if (~single).any() else float("nan"),
                "singleton_rate": float(single.mean())}
    out = {"all": block(scores)}
    if country is not None:
        c = country.reindex(scores.index)
        for k in sorted(c.dropna().unique()):
            out[k] = block(scores[c == k])
    return out
