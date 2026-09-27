"""Stage `features`: pairwise features for every candidate pair.

Country is NEVER a feature (the model must transfer to an unseen country). Per country string:
  1. quick similarities for ALL candidates -> context features (ranks both directions, gaps,
     mutual-best, corroboration across the S1's other top candidates);
  2. full feature set in row chunks (rapidfuzz cpdist, multi-threaded C++; token-set features in
     forked workers), written to features/<split>/<country>_<i>.parquet.
For train, context uses every blocked S1 but full features are materialised only for a deterministic
`feat_s1_frac` subset (memory); test always gets everything.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from multiprocessing import get_context

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from config import Config
from io_utils import load_candidates, LOG, Timer, country_slug, is_done, lib_versions, mark_done, save_parquet
from normalize import NAME_STOP
from prepare import load_records
from split import s1_subsample

CHUNK = 2_000_000
REC_COLS = ["uid", "src", "country", "name_norm", "name_core", "name_legal", "name_skel",
            "name_nonlatin", "name_domain", "addr_norm", "addr_core", "addr_digits", "addr_postal",
            "addr_stnum", "addr_stword", "addr_landmark", "addr_nonlatin"]
_IDF: dict[str, dict[str, float]] = {}
_A: dict[str, np.ndarray] = {}  # record-level arrays shared with forked workers


# ------------------------------------------------------------------ helpers
def cp(a: np.ndarray, b: np.ndarray, scorer, n_jobs: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=n_jobs, dtype=np.float32)


def idf_table(docs: pd.Series) -> dict[str, float]:
    df = Counter()
    for d in docs:
        df.update(set(d.split()))
    n = max(len(docs), 1)
    return {t: math.log(n / c) for t, c in df.items()}


def _set_feats(ta: list[str], tb: list[str], idf: dict[str, float], default_idf: float):
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return np.nan, np.nan, np.nan, 0.0
    inter, union = sa & sb, sa | sb
    w = lambda s: sum(idf.get(t, default_idf) for t in s)  # noqa: E731
    wu = w(union)
    return (len(inter) / len(union), w(inter) / wu if wu > 0 else 0.0,
            max((idf.get(t, default_idf) for t in inter), default=0.0), float(len(inter)))


def _initials(s: str) -> str:
    return "".join(t[0] for t in s.split() if t not in NAME_STOP and not t.isdigit())


def _py_chunk(args: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Token-set features for a slice of pairs; record arrays and IDF tables come from fork globals."""
    qa, pa = args
    A, idf_n, idf_a = _A, _IDF["name"], _IDF["addr"]
    dn, da = max(idf_n.values(), default=10.0), max(idf_a.values(), default=10.0)
    n = len(qa)
    out = {k: np.full(n, np.nan, np.float32) for k in [
        "n_jacc", "n_wjacc", "n_maxidf", "n_nshared", "a_jacc", "a_wjacc", "a_maxidf", "a_nshared",
        "legal_both", "legal_eq", "legal_conflict", "acronym", "d_jacc", "d_nshared", "d_p_subset",
        "d_q_n", "d_p_n", "pc_both", "pc_eq", "sn_both", "sn_eq", "sn_prefix", "stword_eq",
        "n_first_eq", "n_last_eq"]}
    for i, (q, p) in enumerate(zip(qa, pa)):
        cq, cpp = A["name_core"][q], A["name_core"][p]
        tq, tp = cq.split(), cpp.split()
        out["n_jacc"][i], out["n_wjacc"][i], out["n_maxidf"][i], out["n_nshared"][i] = _set_feats(tq, tp, idf_n, dn)
        if tq and tp:
            out["n_first_eq"][i] = float(tq[0] == tp[0])
            out["n_last_eq"][i] = float(tq[-1] == tp[-1])
        aq, ap = A["addr_core"][q].split(), A["addr_core"][p].split()
        out["a_jacc"][i], out["a_wjacc"][i], out["a_maxidf"][i], out["a_nshared"][i] = _set_feats(aq, ap, idf_a, da)
        lq, lp = set(A["name_legal"][q].split()), set(A["name_legal"][p].split())
        both = bool(lq) and bool(lp)
        out["legal_both"][i] = float(both)
        if both:
            out["legal_eq"][i] = float(lq == lp)
            out["legal_conflict"][i] = float(not (lq & lp))
        iq, ip = _initials(A["name_norm"][q]), _initials(A["name_norm"][p])
        nq, np_ = cq.replace(" ", ""), cpp.replace(" ", "")
        out["acronym"][i] = float((len(iq) >= 2 and iq == np_) or (len(ip) >= 2 and ip == nq))
        dq, dp = set(A["addr_digits"][q].split()), set(A["addr_digits"][p].split())
        out["d_q_n"][i], out["d_p_n"][i] = len(dq), len(dp)
        if dq and dp:
            inter = dq & dp
            out["d_jacc"][i] = len(inter) / len(dq | dp)
            out["d_nshared"][i] = len(inter)
            out["d_p_subset"][i] = float(dp <= dq)
        pq, pp = A["addr_postal"][q], A["addr_postal"][p]
        out["pc_both"][i] = float(bool(pq) and bool(pp))
        if pq and pp:
            out["pc_eq"][i] = float(pq == pp)
        sq, sp = A["addr_stnum"][q], A["addr_stnum"][p]
        out["sn_both"][i] = float(bool(sq) and bool(sp))
        if sq and sp:
            out["sn_eq"][i] = float(sq == sp)
            out["sn_prefix"][i] = float(sq.startswith(sp) or sp.startswith(sq) or sq.endswith(sp) or sp.endswith(sq))
        wq, wp = A["addr_stword"][q], A["addr_stword"][p]
        if wq and wp:
            out["stword_eq"][i] = float(wq == wp)
    return out


def py_features(q: np.ndarray, p: np.ndarray, n_jobs: int, piece: int = 50_000) -> pd.DataFrame:
    jobs = [(q[s:s + piece], p[s:s + piece]) for s in range(0, len(q), piece)]
    if n_jobs > 1 and len(jobs) > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_py_chunk, jobs, chunksize=1)
    else:
        res = [_py_chunk(j) for j in jobs]
    return pd.DataFrame({k: np.concatenate([r[k] for r in res]) for k in res[0]})


def string_features(A: dict, q: np.ndarray, p: np.ndarray, n_jobs: int) -> pd.DataFrame:
    g = lambda col, idx: A[col][idx]  # noqa: E731
    nq, np_ = g("name_core", q), g("name_core", p)
    fq, fp = g("name_norm", q), g("name_norm", p)
    kq, kp = g("name_skel", q), g("name_skel", p)
    aq, ap = g("addr_core", q), g("addr_core", p)
    anq, anp = g("addr_norm", q), g("addr_norm", p)
    sq, sp = g("addr_stnum", q), g("addr_stnum", p)
    wq, wp = g("addr_stword", q), g("addr_stword", p)
    nsq, nsp = g("name_nospace", q), g("name_nospace", p)
    f = {
        "n_ratio": cp(nq, np_, fuzz.ratio, n_jobs), "n_partial": cp(nq, np_, fuzz.partial_ratio, n_jobs),
        "n_tsort": cp(nq, np_, fuzz.token_sort_ratio, n_jobs), "n_tset": cp(nq, np_, fuzz.token_set_ratio, n_jobs),
        "n_jw": cp(nq, np_, JaroWinkler.normalized_similarity, n_jobs),
        "n_ratio_full": cp(fq, fp, fuzz.ratio, n_jobs), "n_tset_full": cp(fq, fp, fuzz.token_set_ratio, n_jobs),
        "n_skel_ratio": cp(kq, kp, fuzz.ratio, n_jobs), "n_skel_tset": cp(kq, kp, fuzz.token_set_ratio, n_jobs),
        "n_nospace_ratio": cp(nsq, nsp, fuzz.ratio, n_jobs),
        "n_nospace_partial": cp(nsq, nsp, fuzz.partial_ratio, n_jobs),
        "a_ratio": cp(aq, ap, fuzz.ratio, n_jobs), "a_partial": cp(aq, ap, fuzz.partial_ratio, n_jobs),
        "a_tsort": cp(aq, ap, fuzz.token_sort_ratio, n_jobs), "a_tset": cp(aq, ap, fuzz.token_set_ratio, n_jobs),
        "a_jw": cp(anq, anp, JaroWinkler.normalized_similarity, n_jobs),
        "sn_ratio": cp(sq, sp, fuzz.ratio, n_jobs), "stword_ratio": cp(wq, wp, fuzz.ratio, n_jobs),
    }
    lq = np.array([len(x) for x in nq], np.float32)
    lp = np.array([len(x) for x in np_], np.float32)
    alq = np.array([len(x) for x in aq], np.float32)
    alp = np.array([len(x) for x in ap], np.float32)
    f["n_len_q"], f["n_len_p"] = lq, lp
    f["n_len_ratio"] = np.minimum(lq, lp) / np.maximum(np.maximum(lq, lp), 1)
    f["a_len_q"], f["a_len_p"] = alq, alp
    f["a_len_ratio"] = np.minimum(alq, alp) / np.maximum(np.maximum(alq, alp), 1)
    f["a_empty_p"] = (alp == 0).astype(np.float32)
    f["n_exact"] = (nq == np_).astype(np.float32)
    f["p_nonlatin"] = A["name_nonlatin"][p].astype(np.float32)
    f["p_addr_nonlatin"] = A["addr_nonlatin"][p].astype(np.float32)
    f["p_is_domain"] = (A["name_domain"][p] != "").astype(np.float32)
    f["lm_q"] = A["addr_landmark"][q].astype(np.float32)
    f["lm_p"] = A["addr_landmark"][p].astype(np.float32)
    f["src_p"] = (A["src"][p] == 3).astype(np.float32)
    f["name_freq_q"] = A["name_freq"][q]
    f["name_freq_p"] = A["name_freq"][p]
    for k in ("sn_ratio", "stword_ratio"):  # empty strings -> undefined, not 0
        miss = (sq == "") | (sp == "") if k == "sn_ratio" else (wq == "") | (wp == "")
        f[k][miss] = np.nan
    return pd.DataFrame(f)


# ------------------------------------------------------------------ context
def _rank_desc(df: pd.DataFrame, by: str, col: str) -> np.ndarray:
    return df.groupby(by)[col].rank(method="first", ascending=False).values.astype(np.float32)


def _gap(df: pd.DataFrame, by: str, col: str) -> np.ndarray:
    """col minus the best OTHER value in its group (leader: margin over runner-up; others: <= 0)."""
    g, v = df[by].values, df[col].values.astype(np.float64)
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    start = np.r_[True, gs[1:] != gs[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, len(gs)])
    top1 = vs[first]
    top2 = np.where(size > 1, vs[np.minimum(first + 1, len(vs) - 1)], 0.0)
    gap_sorted = np.where(start, vs - top2[gid], vs - top1[gid])
    out = np.empty(len(v), np.float32)
    out[order] = gap_sorted
    return out


def context_features(c: pd.DataFrame, A: dict, n_jobs: int) -> pd.DataFrame:
    """Group-level features over ALL candidates of one country. `c` has q, p, cos_dense."""
    q, p = c["q"].values, c["p"].values
    n_tset = cp(A["name_core"][q], A["name_core"][p], fuzz.token_set_ratio, n_jobs) / 100
    a_tset = cp(A["addr_core"][q], A["addr_core"][p], fuzz.token_set_ratio, n_jobs) / 100
    d = pd.DataFrame({"q": q, "p": p, "qs": 0.4 * n_tset + 0.4 * a_tset + 0.2 * c["cos_dense"].values,
                      "cd": c["cos_dense"].values, "src": A["src"][p]})
    out = pd.DataFrame(index=c.index)
    out["qs"] = d["qs"].values.astype(np.float32)
    out["n_cands_q"] = d.groupby("q")["p"].transform("size").values.astype(np.float32)
    out["n_q_for_p"] = d.groupby("p")["q"].transform("size").values.astype(np.float32)
    out["rank_qs"] = _rank_desc(d, "q", "qs")
    out["rank_cd"] = _rank_desc(d, "q", "cd")
    d["qsrc"] = d["q"].astype(np.int64) * 4 + d["src"].astype(np.int64)
    out["rank_qs_src"] = _rank_desc(d, "qsrc", "qs")
    out["rev_rank_qs"] = _rank_desc(d, "p", "qs")
    out["rev_rank_cd"] = _rank_desc(d, "p", "cd")
    out["gap_qs"] = _gap(d, "q", "qs")
    out["gap_cd"] = _gap(d, "q", "cd")
    out["rev_gap_qs"] = _gap(d, "p", "qs")
    out["mutual_best"] = ((out["rank_qs"] == 1) & (out["rev_rank_qs"] == 1)).astype(np.float32)
    out["qs_minus_mean_q"] = (d["qs"] - d.groupby("q")["qs"].transform("mean")).values.astype(np.float32)
    out["n_q_above_08"] = d.assign(h=d["qs"] > 0.8).groupby("q")["h"].transform("sum").values.astype(np.float32)
    out = out.join(corroboration(d, A, n_jobs))
    return out


def corroboration(d: pd.DataFrame, A: dict, n_jobs: int, top: int = 3) -> pd.DataFrame:
    """Similarity of each candidate to the S1's other top-`top` candidates (by quick score)."""
    order = d.sort_values(["q", "qs"], ascending=[True, False])
    pos = order.groupby("q").cumcount().values
    tops = order[pos < top + 1][["q", "p"]]
    tops = tops.assign(j=tops.groupby("q").cumcount().values)
    wide = tops.pivot(index="q", columns="j", values="p")
    res = {}
    n_sim, a_sim = [], []
    for j in wide.columns:
        other = d["q"].map(wide[j]).values
        ok = ~np.isnan(other)
        other = np.where(ok, other, 0).astype(np.int64)
        valid = ok & (other != d["p"].values)
        ns = cp(A["name_core"][d["p"].values], A["name_core"][other], fuzz.token_set_ratio, n_jobs)
        as_ = cp(A["addr_core"][d["p"].values], A["addr_core"][other], fuzz.token_set_ratio, n_jobs)
        ns[~valid], as_[~valid] = np.nan, np.nan
        n_sim.append(ns)
        a_sim.append(as_)
    N, S = np.vstack(n_sim), np.vstack(a_sim)
    with warnings.catch_warnings():  # all-NaN columns (S1 with a single candidate) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        res["corr_n_max"] = np.nanmax(N, 0)
        res["corr_a_max"] = np.nanmax(S, 0)
        res["corr_both_max"] = np.nanmax((N + S) / 2, 0)
        res["corr_both_mean"] = np.nanmean((N + S) / 2, 0)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in res.items()}, index=d.index)


# ------------------------------------------------------------------ stage
def feat_dir(cfg: Config, split: str):
    return cfg.art / "features" / split


def build_arrays(sub: pd.DataFrame) -> dict:
    """Positional arrays for one country's records (local index = row position in `sub`)."""
    A = {c: sub[c].values for c in REC_COLS}
    A["name_nospace"] = np.array([s.replace(" ", "") for s in A["name_core"]], dtype=object)
    # how generic is the name within its country (log count of identical core names)
    A["name_freq"] = np.log1p(sub.groupby("name_core")["src"].transform("size").values).astype(np.float32)
    return A


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "features"
    if is_done(out) and not force:
        LOG.info("features: cached, skipping")
        return
    meta = {"versions": lib_versions()}
    for split in ["train", "test"]:
        info = load_records(cfg, split, ["entity_id", "src", "country"])
        n_total = len(info)
        cand = load_candidates(cfg, split)
        keep_uid = None
        if split == "train" and cfg.feat_s1_frac < 1.0:
            s1 = info[info["src"] == 1]
            keep_ids = s1_subsample(s1["entity_id"], cfg.feat_s1_frac, cfg.seed, "feat")
            keep_uid = s1.index.values[s1["entity_id"].isin(keep_ids).values]
        d = feat_dir(cfg, split)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.parquet"):
            f.unlink()
        n_rows = 0
        ctry_of_q = info["country"].values[cand["q"].values]
        del info
        for ctry in sorted(pd.unique(ctry_of_q)):
            c = cand[ctry_of_q == ctry].reset_index(drop=True)
            with Timer(f"features {split} [{ctry}] {len(c):,} pairs"):
                sub = load_records(cfg, split, REC_COLS, country=ctry)
                A = build_arrays(sub)
                lut = np.full(n_total, -1, np.int64)
                lut[sub["uid"].values] = np.arange(len(sub))
                _IDF["name"] = idf_table(sub["name_core"])
                _IDF["addr"] = idf_table(sub["addr_core"])
                _A.clear()
                _A.update(A)
                local = pd.DataFrame({"q": lut[c["q"].values], "p": lut[c["p"].values], "cos_dense": c["cos_dense"].values})
                assert (local[["q", "p"]].values >= 0).all()
                ctx = context_features(local, A, cfg.n_jobs)
                c = pd.concat([c, ctx], axis=1)
                if keep_uid is not None:
                    c = c[np.isin(c["q"].values, keep_uid)].reset_index(drop=True)
                for i, s in enumerate(range(0, len(c), CHUNK)):
                    ch = c.iloc[s:s + CHUNK].reset_index(drop=True)
                    ql, pl = lut[ch["q"].values], lut[ch["p"].values]
                    feats = pd.concat([ch, string_features(A, ql, pl, cfg.n_jobs), py_features(ql, pl, cfg.n_jobs)], axis=1)
                    for col in feats.columns:
                        if feats[col].dtype == np.float64:
                            feats[col] = feats[col].astype(np.float32)
                    save_parquet(feats, d / f"{country_slug(ctry)}_{i:04d}.parquet")
                    n_rows += len(feats)
                del sub, A
                _A.clear()
        meta[f"{split}_rows"] = n_rows
        LOG.info("features %s: %s rows", split, f"{n_rows:,}")
    mark_done(out, meta)


def load_features(cfg: Config, split: str, columns: list[str] | None = None, s1_filter: set | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted(feat_dir(cfg, split).glob("*.parquet")):
        df = pd.read_parquet(f, columns=columns)
        if s1_filter is not None:
            df = df[df["q"].isin(s1_filter)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def iter_feature_parts(cfg: Config, split: str, columns: list[str] | None = None):
    """Yield (row_offset, DataFrame) per part file, in the same order as load_features."""
    off = 0
    for f in sorted(feat_dir(cfg, split).glob("*.parquet")):
        df = pd.read_parquet(f, columns=columns)
        yield off, df
        off += len(df)


NON_FEATURES = {"q", "p", "y", "fold"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]
