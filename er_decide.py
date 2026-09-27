"""Metric, blending, calibration, decision rules and the submission writer."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ================================================================ metric
def f05(pred, truth):
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    P, R = tp / len(pred), tp / len(truth)
    return 1.25 * P * R / (0.25 * P + R)


def macro_f05(pred, truth):
    """pred, truth: dict s1 -> iterable of ids. Averages over ALL S1 in truth."""
    return float(np.mean([f05(pred.get(s, ()), t) for s, t in truth.items()])) if truth else float("nan")


def per_country_f05(pred, truth, country):
    """country: dict s1 -> country string."""
    out = {}
    for c in sorted(set(country.values())):
        sub = {s: t for s, t in truth.items() if country[s] == c}
        out[c] = macro_f05(pred, sub)
    return out


def f05_vec(q, sel, y, n_true, s1_rows):
    """Vectorised per-S1 F0.5.  q: S1 global id per candidate pair; sel: predicted; y: label;
    n_true: [n_rec] true count per S1 (all true matches, incl. ones blocking missed);
    s1_rows: global ids of every S1 being scored.  -> F0.5 per s1_rows."""
    n = len(n_true)
    tp = np.bincount(q[sel & (y == 1)], minlength=n)[s1_rows].astype(np.float64)
    npred = np.bincount(q[sel], minlength=n)[s1_rows].astype(np.float64)
    nt = n_true[s1_rows].astype(np.float64)
    return np.where(nt == 0, (npred == 0).astype(np.float64),
                    np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * nt + npred, 1e-12), 0.0))


# ================================================================ blending / calibration
def best_threshold(q, p, y, n_true, s1_rows, grid=None):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    scores = [f05_vec(q, p >= th, y, n_true, s1_rows).mean() for th in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def choose_blend(q, p_lgb, p_mlp, y, n_true, s1_rows, grid=np.round(np.arange(0, 1.01, 0.1), 1)):
    rows = []
    for w in grid:
        th, sc = best_threshold(q, w * p_lgb + (1 - w) * p_mlp, y, n_true, s1_rows)
        rows.append({"w_lgb": float(w), "thr": th, "macro_f05": sc})
    df = pd.DataFrame(rows)
    return float(df.loc[df["macro_f05"].idxmax(), "w_lgb"]), df


def isotonic_crossfit(p, y, folds):
    out = np.empty_like(p)
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[tr], y[tr])
        out[va] = iso.predict(p[va])
    return out.astype(np.float32)


def isotonic_fit(p, y):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)


# ================================================================ decision rules
def assign_one_to_one(q, t, p):
    """Each S2/S3 record keeps only its highest-probability S1 (others -> 0)."""
    order = np.lexsort((-p, t))
    ts = t[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    keep = np.zeros(len(p), bool)
    keep[order[first]] = True
    return np.where(keep, p, 0.0).astype(np.float32)


def expected_f_select(q, p, kmax=10, m=20, draws=1000, seed=42, chunk=2000):
    """Per S1: choose k in 0..kmax maximising Monte-Carlo expected F0.5 of outputting its top-k
    candidates (independent Bernoulli(p)). k=0 scores 1 only in draws with no true match."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order]
    start = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    lens = np.diff(np.r_[start, len(qs)])
    rank = np.arange(len(qs)) - np.repeat(start, lens)
    G = len(start)
    grp = np.repeat(np.arange(G), lens)
    P = np.zeros((G, m), np.float32)
    inm = rank < m
    P[grp[inm], rank[inm]] = ps[inm]
    kbest = np.zeros(G, np.int64)
    ks = np.arange(1, kmax + 1, dtype=np.float32)
    for a in range(0, G, chunk):
        Pc = P[a:a + chunk]
        X = rng.random((len(Pc), draws, m), dtype=np.float32) < Pc[:, None, :]
        nt = X.sum(-1).astype(np.float32)
        cum = np.cumsum(X[:, :, :kmax], axis=-1, dtype=np.float32)
        Fk = np.where(nt[..., None] > 0, 1.25 * cum / (0.25 * nt[..., None] + ks), 0.0).mean(1)
        F0 = (nt == 0).mean(1, keepdims=True)
        kbest[a:a + chunk] = np.argmax(np.concatenate([F0, Fk], 1), 1)
    sel_sorted = rank < kbest[grp]
    sel = np.zeros(len(p), bool)
    sel[order] = sel_sorted
    return sel


def apply_rule(rule, q, t, p, thr=None, seed=42, draws=1000):
    pp = assign_one_to_one(q, t, p) if rule.endswith("+assign") else p
    if rule.startswith("threshold"):
        return pp >= thr
    return expected_f_select(q, pp, seed=seed, draws=draws) & (pp > 0)


def evaluate_variants(q, t, p_cal, y, n_true, s1_rows, s1_country, seed, draws, log=print):
    """p_cal: cross-fitted calibrated OOF probabilities. Returns (table, best variant, thresholds)."""
    rows, thr = [], {}
    base_thr, _ = best_threshold(q, p_cal, y, n_true, s1_rows)
    p_as = assign_one_to_one(q, t, p_cal)
    as_thr, _ = best_threshold(q, p_as, y, n_true, s1_rows)
    thr["threshold"], thr["threshold+assign"] = base_thr, as_thr
    sels = {"threshold": p_cal >= base_thr, "threshold+assign": p_as >= as_thr,
            "expectedF": expected_f_select(q, p_cal, seed=seed, draws=draws),
            "expectedF+assign": expected_f_select(q, p_as, seed=seed, draws=draws) & (p_as > 0)}
    for v, sel in sels.items():
        f = f05_vec(q, sel, y, n_true, s1_rows)
        sing = n_true[s1_rows] == 0
        for g in ["all"] + sorted(set(s1_country)):
            m = np.ones(len(s1_rows), bool) if g == "all" else (s1_country == g)
            rows.append({"variant": v, "group": g, "n_s1": int(m.sum()), "macro_F0.5": f[m].mean(),
                         "singleton_acc": f[m & sing].mean() if (m & sing).any() else np.nan,
                         "nonsingleton_F0.5": f[m & ~sing].mean() if (m & ~sing).any() else np.nan,
                         "threshold": thr.get(v, np.nan)})
    tab = pd.DataFrame(rows)
    allrows = tab[tab["group"] == "all"]
    best = allrows.loc[allrows["macro_F0.5"].idxmax(), "variant"]
    return tab, best, thr


# ================================================================ writer
def write_lists(path, header, s1_ids, row, ids_for_pairs, chunk_rows=100_000):
    """s1_ids: every test S1 id in file order. row: S1 row (0..N-1) per pair, sorted ascending.
    ids_for_pairs(a, b) -> list[str] of S2/S3 ids for pairs a..b (unique per S1)."""
    N = len(s1_ids)
    starts = np.searchsorted(row, np.arange(N + 1), side="left")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r0 in range(0, N, chunk_rows):
            r1 = min(N, r0 + chunk_rows)
            a, b = int(starts[r0]), int(starts[r1])
            ids = ids_for_pairs(a, b)
            buf = []
            for r in range(r0, r1):
                s, e = int(starts[r]) - a, int(starts[r + 1]) - a
                buf.append(s1_ids[r] + "\t" + ",".join(ids[s:e]))
            f.write("\n".join(buf) + "\n")
