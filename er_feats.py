"""Pair features (CPU; dense cosines gathered in MLX). Country is never a feature.

Per (split, group): context features on the whole candidate table, then string/dense/jaccard
features in chunks of CHUNK pairs, written as parquet parts:  q, t, y, <features...>.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from er_block import gathered_cos
from er_text import acronym, str_take

SCORERS = {"ratio": fuzz.ratio, "partial": fuzz.partial_ratio, "tsort": fuzz.token_sort_ratio,
           "tset": fuzz.token_set_ratio, "jw": JaroWinkler.normalized_similarity}
TOKEN_FIELDS = ("name_n", "name_core", "addr_n", "digits", "legal")


def record_side_features(rec):
    """Per-record helpers computed once per split."""
    core = rec["name_core"].astype(str)
    out = pd.DataFrame(index=rec.index)
    out["core_len"] = core.str.len().to_numpy(np.float32)
    out["acr"] = [acronym(c) for c in core.tolist()]
    out["compact"] = core.str.replace(" ", "", regex=False).tolist()
    # how common this core name is among S2/S3 of the same country string (open set)
    key = rec["country"].astype(str) + "\x1f" + core
    tgt = rec["src"].to_numpy() != 1
    freq = key[tgt].value_counts()
    out["name_freq"] = np.log1p(key.map(freq).fillna(0).to_numpy(np.float32))
    return out


def token_matrices(rec, idx):
    """Binary token matrices (rows aligned with idx) for jaccard / overlap features."""
    mats = {}
    for f in TOKEN_FIELDS:
        cv = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
        try:
            M = cv.fit_transform(rec[f].iloc[idx].astype(str).tolist()).tocsr()
        except ValueError:  # empty vocabulary (e.g. no legal suffix anywhere in a small group)
            M = sparse.csr_matrix((len(idx), 1), dtype=np.float32)
        mats[f] = (M, np.asarray(M.sum(1)).ravel().astype(np.float32))
    return mats


def _within_rank(key, score):
    """Rank (1 = best) of score within each key group."""
    order = np.lexsort((-score, key))
    k = key[order]
    start = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    lens = np.diff(np.r_[start, len(k)])
    rank = np.empty(len(key), np.int32)
    rank[order] = np.arange(len(k)) - np.repeat(start, lens) + 1
    return rank


def _group_top2(key, score, n):
    top1 = np.full(n, -np.inf, np.float32)
    np.maximum.at(top1, key, score)
    is_top = score >= top1[key]
    s2 = np.where(is_top, -np.inf, score).astype(np.float32)
    top2 = np.full(n, -np.inf, np.float32)
    np.maximum.at(top2, key, s2)
    # ties at the top: second best equals best
    cnt_top = np.bincount(key[is_top], minlength=n)
    top2 = np.where(cnt_top > 1, top1, top2)
    return top1, top2


def context_features(pairs, rec, E_ret, n_top=3):
    q = pairs["q"].to_numpy(np.int64)
    t = pairs["t"].to_numpy(np.int64)
    NR = len(rec)
    src = rec["src"].to_numpy()
    F = {"is_s3": (src[t] == 3).astype(np.float32)}
    for s in ("s_dense", "s_tfidf", "s_combo", "s_rare"):
        F[s] = pairs[s].to_numpy(np.float32)
    for s in ("s_dense", "s_tfidf", "s_combo"):
        sc = pairs[s].to_numpy(np.float32)
        F[f"rank_{s}"] = _within_rank(q, sc).astype(np.float32)
        F[f"revrank_{s}"] = _within_rank(t, sc).astype(np.float32)
        top1, top2 = _group_top2(q, sc, NR)
        best_other = np.where(sc >= top1[q], top2[q], top1[q])
        F[f"gap_{s}"] = np.where(np.isfinite(best_other), sc - best_other, 1.0).astype(np.float32)
        r1, r2 = _group_top2(t, sc, NR)
        rbest_other = np.where(sc >= r1[t], r2[t], r1[t])
        F[f"revgap_{s}"] = np.where(np.isfinite(rbest_other), sc - rbest_other, 1.0).astype(np.float32)
    F["mutual_best"] = ((F["rank_s_dense"] == 1) & (F["revrank_s_dense"] == 1)).astype(np.float32)
    F["mutual_best_tfidf"] = ((F["rank_s_tfidf"] == 1) & (F["revrank_s_tfidf"] == 1)).astype(np.float32)
    F["n_cands"] = np.bincount(q, minlength=NR)[q].astype(np.float32)
    F["n_rev"] = np.bincount(t, minlength=NR)[t].astype(np.float32)
    for r in ("r_tfidf", "r_combo", "r_dense", "r_rare"):
        F[f"hit_{r[2:]}"] = (pairs[r].to_numpy() > 0).astype(np.float32)
    F["n_hits"] = sum(F[f"hit_{r}"] for r in ("tfidf", "combo", "dense", "rare"))

    # corroboration: similarity of this candidate to the S1's top-n candidates from the OTHER source
    rank = F["rank_s_dense"]
    corr = np.full(len(q), -1.0, np.float32)
    for other in (2, 3):
        m_other = src[t] == other
        # top-n of source `other` per S1, by s_dense
        sub = np.flatnonzero(m_other)
        rk = _within_rank(q[sub], F["s_dense"][sub])
        keep = sub[rk <= n_top]
        slots = np.full((NR, n_top), -1, np.int64)
        slots[q[keep], rk[rk <= n_top] - 1] = t[keep]
        tgt = np.flatnonzero(~m_other)  # candidates from the other source get corroborated by these
        for j in range(n_top):
            o = slots[q[tgt], j]
            ok = o >= 0
            if ok.any():
                c = gathered_cos(E_ret, t[tgt][ok], o[ok])
                cur = corr[tgt[ok]]
                corr[tgt[ok]] = np.maximum(cur, c)
    F["corrob_max"] = corr
    del rank
    return pd.DataFrame(F)


def _jacc(mats, f, qa, ta):
    M, n = mats[f]
    inter = np.asarray(M[qa].multiply(M[ta]).sum(1)).ravel().astype(np.float32)
    union = n[qa] + n[ta] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan).astype(np.float32), inter


def _eq3(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    ea, eb = (a == ""), (b == "")
    both = ~ea & ~eb
    return ((both & (a == b)).astype(np.float32), (both & (a != b)).astype(np.float32),
            (ea.astype(np.int8) + eb.astype(np.int8)).astype(np.float32))


def string_features(rec, side, mats, loc, q, t, emb, n_jobs):
    """q, t: global record ids; loc: global id -> row in `mats` (group-local)."""
    F = {}
    for col, pre in (("name_n", "n"), ("name_core", "c"), ("addr_n", "a")):
        A, B = str_take(rec, col, q), str_take(rec, col, t)
        for nm, sc in SCORERS.items():
            F[f"{pre}_{nm}"] = process.cpdist(A, B, scorer=sc, workers=n_jobs, dtype=np.float32).astype(np.float32)
        empty = (np.fromiter((len(x) == 0 for x in A), bool, len(A)) |
                 np.fromiter((len(x) == 0 for x in B), bool, len(B)))
        for nm in SCORERS:
            F[f"{pre}_{nm}"][empty] = np.nan
        del A, B
    qa, ta = loc[q], loc[t]
    F["n_jacc"], _ = _jacc(mats, "name_n", qa, ta)
    F["c_jacc"], _ = _jacc(mats, "name_core", qa, ta)
    F["a_jacc"], F["a_inter"] = _jacc(mats, "addr_n", qa, ta)
    F["d_jacc"], _ = _jacc(mats, "digits", qa, ta)
    _, legal_inter = _jacc(mats, "legal", qa, ta)
    ln = mats["legal"][1]
    F["legal_eq"] = ((ln[qa] > 0) & (ln[qa] == ln[ta]) & (legal_inter == ln[qa])).astype(np.float32)
    F["legal_conflict"] = ((ln[qa] > 0) & (ln[ta] > 0) & (legal_inter == 0)).astype(np.float32)
    F["legal_missing"] = ((ln[qa] == 0).astype(np.float32) + (ln[ta] == 0))
    for f in ("postal", "stnum"):
        F[f"{f}_eq"], F[f"{f}_conflict"], F[f"{f}_missing"] = _eq3(str_take(rec, f, q), str_take(rec, f, t))
    acr, comp = side["acr"].to_numpy(dtype=object), side["compact"].to_numpy(dtype=object)
    cl = side["core_len"].to_numpy()
    F["acronym"] = (((acr[q] == comp[t]) & (cl[t] >= 2)) | ((acr[t] == comp[q]) & (cl[q] >= 2))).astype(np.float32)
    F["len_ratio"] = (np.minimum(cl[q], cl[t]) / np.maximum(np.maximum(cl[q], cl[t]), 1)).astype(np.float32)
    lm = rec["landmark"].to_numpy()
    F["landmark_q"], F["landmark_t"] = lm[q].astype(np.float32), lm[t].astype(np.float32)
    nf = side["name_freq"].to_numpy()
    F["name_freq_q"], F["name_freq_t"] = nf[q], nf[t]
    for name, E in emb.items():
        F[f"cos_{name}"] = gathered_cos(E, q, t)
    return pd.DataFrame(F)


def featurize_group(rec, side, pairs, E_ret, emb, pos_keys, out_dir, tag, n_jobs, chunk=1_000_000, log=print):
    """Writes out_dir/{tag}_{i:04d}.parquet. pos_keys=None for test (y=-1)."""
    pairs = pairs.sort_values(["q", "t"], kind="stable").reset_index(drop=True)
    ctx = context_features(pairs, rec, E_ret)
    gid = np.unique(np.r_[pairs["q"].to_numpy(), pairs["t"].to_numpy()])
    mats = token_matrices(rec, gid)
    loc = np.full(len(rec), -1, np.int64)
    loc[gid] = np.arange(len(gid))
    NR = len(rec)
    files = []
    for i, s in enumerate(range(0, len(pairs), chunk)):
        p = pairs.iloc[s:s + chunk]
        q, t = p["q"].to_numpy(np.int64), p["t"].to_numpy(np.int64)
        sf = string_features(rec, side, mats, loc, q, t, emb, n_jobs)
        df = pd.concat([pd.DataFrame({"q": q.astype(np.int32), "t": t.astype(np.int32)}),
                        ctx.iloc[s:s + chunk].reset_index(drop=True), sf], axis=1)
        df["y"] = (np.isin(q * NR + t, pos_keys).astype(np.int8) if pos_keys is not None
                   else np.full(len(q), -1, np.int8))
        fn = out_dir / f"{tag}_{i:04d}.parquet"
        df.to_parquet(fn, index=False, compression="zstd")
        files.append(fn)
        log(f"      {tag} part {i}: {len(df):,} pairs")
    return files


def feature_columns(df):
    return [c for c in df.columns if c not in ("q", "t", "y")]
