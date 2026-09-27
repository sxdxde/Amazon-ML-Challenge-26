"""Stage `decide`: probabilities -> per-S1 match sets.

1. Isotonic calibration of the final model scores (cross-fitted over folds for honest OOF numbers,
   refit on all OOF for test).
2. One S1 per record (EDA: 0 of 7.6M matched S2/S3 ids belong to >1 S1): each S2/S3 record is kept
   only for its highest-probability S1.
3. Per S1, choose k = 0..K maximising EXPECTED F0.5 of outputting the top-k, treating candidates as
   independent Bernoulli(p). Exact: Poisson-binomial distributions of TP (top-k) and of missed
   positives (rest), E[F_k] = Σ_a Σ_b P(TP=a) P(M=b) · 1.25a / (0.25(a+b) + k); E[F_0] = Π(1-p).
Baselines: global threshold (± assignment). The variant with the best OOF macro F0.5 is used on test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config import Config
from io_utils import LOG, Timer, is_done, mark_done, save_parquet
from metric import per_entity_scores, summarize
from prepare import load_records, load_truth
from prune import keep_mask, kept_pairs


# ------------------------------------------------------------------ calibration
def fit_isotonic(score: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(score.astype(np.float64), y.astype(np.float64))
    return iso


def crossfit_calibrate(score: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(score), np.float64)
    for f in np.unique(folds):
        m = folds == f
        out[m] = fit_isotonic(score[~m], y[~m]).predict(score[m])
    return out


# ------------------------------------------------------------------ assignment
def assign_exclusive(df: pd.DataFrame, col: str = "pc") -> pd.DataFrame:
    """Zero the prob of every (q, p) except the best q for each p (ties -> first)."""
    df = df.copy()
    best = df.groupby("p")[col].transform("max").values
    is_best = df[col].values >= best
    # break exact ties deterministically
    dup = df.assign(_b=is_best).loc[is_best].duplicated("p")
    is_best[np.flatnonzero(is_best)[dup.values]] = False
    df[col] = np.where(is_best, df[col].values, 0.0)
    return df


# ------------------------------------------------------------------ expected F0.5
def _pb_step(D: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Convolve Poisson-binomial distributions D (n × m) with Bernoulli(p) (n,)."""
    out = D * (1 - p)[:, None]
    out[:, 1:] += D[:, :-1] * p[:, None]
    return out


def expected_f_best_k(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P: n × L probabilities sorted descending (0-padded). Returns (best_k, E[F] at best k)."""
    n, L = P.shape
    m = L + 1
    # suffix distributions: S[k] = distribution of #positives among items k..L-1
    S = np.zeros((L + 1, n, m))
    S[L][:, 0] = 1.0
    for k in range(L - 1, -1, -1):
        S[k] = _pb_step(S[k + 1], P[:, k])
    a = np.arange(m)[:, None].astype(np.float64)
    b = np.arange(m)[None, :].astype(np.float64)
    best_e = S[0][:, 0].copy()  # k = 0: probability that there are no positives at all
    best_k = np.zeros(n, np.int64)
    pre = np.zeros((n, m))
    pre[:, 0] = 1.0
    for k in range(1, L + 1):
        pre = _pb_step(pre, P[:, k - 1])
        W = 1.25 * a / (0.25 * (a + b) + k)
        e = ((pre @ W) * S[k]).sum(1)
        better = e > best_e + 1e-12
        best_e = np.where(better, e, best_e)
        best_k = np.where(better, k, best_k)
    return best_k, best_e


def select_expected_f(df: pd.DataFrame, col: str, max_cands: int, chunk: int) -> pd.DataFrame:
    """Return the selected (q, p) pairs under the expected-F0.5 rule."""
    d = df.loc[df[col] > 1e-4, ["q", "p", col]].sort_values(["q", col], ascending=[True, False])
    pos = d.groupby("q").cumcount().values
    d = d[pos < max_cands]
    pos = pos[pos < max_cands]
    qs, qidx = np.unique(d["q"].values, return_inverse=True)
    P = np.zeros((len(qs), max_cands))
    P[qidx, pos] = d[col].values
    ks = np.zeros(len(qs), np.int64)
    for s in range(0, len(qs), chunk):
        ks[s:s + chunk], _ = expected_f_best_k(P[s:s + chunk])
    keep = pos < ks[qidx]
    return d.loc[keep, ["q", "p"]]


def select_threshold(df: pd.DataFrame, col: str, t: float) -> pd.DataFrame:
    return df.loc[df[col] >= t, ["q", "p"]]


# ------------------------------------------------------------------ evaluation
def evaluate(pred: pd.DataFrame, truth_uid: pd.DataFrame, s1_uids: np.ndarray, country: pd.Series) -> dict:
    sc = per_entity_scores(pred.rename(columns={"q": "s1", "p": "id"}),
                           truth_uid.rename(columns={"q": "s1", "p": "id"}), pd.Series(s1_uids))
    out = summarize(sc, country)
    out["all"]["mean_pred_size"] = float(len(pred) / max(len(s1_uids), 1))
    return out


def best_threshold(df: pd.DataFrame, col: str, truth_uid, s1_uids, country) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        f = evaluate(select_threshold(df, col, t), truth_uid, s1_uids, None)["all"]["f05"]
        if f > best[1]:
            best = (float(t), f)
    return best


def decide_variants(df: pd.DataFrame, truth_uid: pd.DataFrame, s1_uids: np.ndarray, country: pd.Series,
                    cfg: Config) -> tuple[dict, dict]:
    """Evaluate all decision variants on calibrated OOF probs (column 'pc'). Returns (results, preds)."""
    res, preds = {}, {}
    ex = assign_exclusive(df, "pc")
    for name, d in [("threshold", df), ("threshold+assign", ex)]:
        t, _ = best_threshold(d, "pc", truth_uid, s1_uids, country)
        preds[name] = select_threshold(d, "pc", t)
        res[name] = {"threshold": t, **evaluate(preds[name], truth_uid, s1_uids, country)}
    for name, d in [("expectedF", df), ("expectedF+assign", ex)]:
        preds[name] = select_expected_f(d, "pc", cfg.mc_max_cands, cfg.decide_chunk)
        res[name] = evaluate(preds[name], truth_uid, s1_uids, country)
    return res, preds


def apply_variant(df: pd.DataFrame, variant: str, threshold: float | None, cfg: Config) -> pd.DataFrame:
    d = assign_exclusive(df, "pc") if variant.endswith("+assign") else df
    if variant.startswith("threshold"):
        return select_threshold(d, "pc", threshold)
    return select_expected_f(d, "pc", cfg.mc_max_cands, cfg.decide_chunk)


# ------------------------------------------------------------------ stage
def final_scores(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """OOF (train) and test scores from the last available model stage."""
    for stage, col in [("gbm2", "p2"), ("gbm", "p1")]:
        d = cfg.art / stage
        if (d / "_DONE.json").exists():
            return pd.read_parquet(d / "oof.parquet"), pd.read_parquet(d / "test_pred.parquet"), col
    raise FileNotFoundError("no model predictions found; run gbm first")


def truth_uids(cfg: Config, rec: pd.DataFrame) -> pd.DataFrame:
    truth = load_truth(cfg)
    uid = pd.Series(rec.index.values, index=rec["entity_id"].values)
    return pd.DataFrame({"q": uid.reindex(truth["s1"]).values, "p": uid.reindex(truth["id"]).values})


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "decide"
    if is_done(out) and not force:
        LOG.info("decide: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    oof, test, col = final_scores(cfg)
    # cascade: calibrate / choose / decide only on the pruned candidate set the final matcher sees
    o1 = pd.read_parquet(cfg.art / "gbm" / "oof.parquet", columns=["q", "p", "p1"])
    assert (o1["q"].values == oof["q"].values).all() and (o1["p"].values == oof["p"].values).all()
    oof = oof[keep_mask(o1, cfg)].reset_index(drop=True)
    test = test.merge(kept_pairs(cfg, "test"), on=["q", "p"], how="inner")
    LOG.info("pruned candidate set: train %.2f / test %.2f pairs per S1 with candidates",
             len(oof) / o1["q"].nunique(), len(test) / max(test["q"].nunique(), 1))
    del o1
    rec = load_records(cfg, "train", ["entity_id", "src", "country"])
    tu = truth_uids(cfg, rec)
    s1_uids = np.load(cfg.art / "gbm" / "eval_s1.npy")
    tu = tu[tu["q"].isin(set(s1_uids))]
    country = rec["country"]
    with Timer("calibrate + decision variants"):
        oof = oof.copy()
        oof["pc"] = crossfit_calibrate(oof[col].values, oof["y"].values, oof["fold"].values)
        res, _ = decide_variants(oof, tu, s1_uids, country, cfg)
    best = max(res, key=lambda k: res[k]["all"]["f05"])
    LOG.info("decision variants (OOF macro F0.5): %s -> best %s",
             {k: round(v["all"]["f05"], 5) for k, v in res.items()}, best)
    iso = fit_isotonic(oof[col].values, oof["y"].values)
    test = test.copy()
    test["pc"] = iso.predict(test[col].values)
    pred = apply_variant(test, best, res[best].get("threshold"), cfg)
    save_parquet(pred, out / "test_matches.parquet")
    save_parquet(test[["q", "p", "pc"]], out / "test_calibrated.parquet")
    choice = {"score_col": col, "variant": best, "threshold": res[best].get("threshold"), "results": res}
    (out / "choice.json").write_text(json.dumps(choice, indent=1))
    write_report(cfg, res, best, col)
    mark_done(out, {"variant": best, "oof_f05": res[best]["all"]["f05"], "score_col": col})


def write_report(cfg: Config, res: dict, best: str, col: str) -> None:
    rows = []
    for name, r in res.items():
        for grp, v in r.items():
            if isinstance(v, dict):
                rows.append({"variant": name, "group": grp, "n_s1": v["n"], "macro_F0.5": v["f05"],
                             "singleton_acc": v["singleton_acc"], "nonsingleton_F0.5": v["nonsingleton_f05"],
                             "threshold": r.get("threshold")})
    df = pd.DataFrame(rows)
    lines = ["# Decision rule comparison (train OOF, cross-fitted isotonic calibration)", "",
             f"Scores: `{col}`. Evaluated over every featurised train S1 (S1s with no candidates predict empty; "
             "true links missed by blocking count as misses).", "",
             f"**Chosen variant: `{best}`** (macro F0.5 = {res[best]['all']['f05']:.5f})", "",
             df.round(5).to_markdown(index=False), ""]
    loco = cfg.art / "gbm" / "loco.json"
    if loco.exists():
        lines += ["## Leave-one-country-out (stage-1 GBM + expectedF+assign)", "",
                  pd.DataFrame(json.loads(loco.read_text())).round(5).to_markdown(index=False), ""]
    cfg.reports.mkdir(parents=True, exist_ok=True)
    (cfg.reports / "decision.md").write_text("\n".join(lines))
