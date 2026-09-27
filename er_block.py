"""Blocking: for each S1, union over S2 u S3 (same group) of
  (a)  char_wb 2-4 TF-IDF on normalised core names           -> top K_TFIDF
  (a2) word TF-IDF on core name + address                     -> top K_COMBO
  (b)  dense cosine in MLX on the retrieval embedding         -> top K_DENSE
  (c)  rare tokens / postal-code keys (df <= RARE_MAX_DF)      -> top K_RARE
then exact scores of every retriever for every union pair. Groups = country string when
SAME_COUNTRY (open set: whatever strings exist), else one global group.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RET = ("tfidf", "combo", "dense", "rare")


# ---------------------------------------------------------------- sparse top-k
def prune_rows(X, m):
    """Keep the m largest entries of each CSR row (query-side pruning: rare, high-idf grams)."""
    X = X.tocsr()
    lens = np.diff(X.indptr)
    if lens.max(initial=0) <= m:
        return X
    rows = np.repeat(np.arange(X.shape[0]), lens)
    order = np.lexsort((-X.data, rows))
    rank = np.arange(len(order)) - np.repeat(X.indptr[:-1], lens)
    keep = np.sort(order[rank < m])
    return sparse.csr_matrix((X.data[keep], X.indices[keep], np.r_[0, np.cumsum(np.minimum(lens, m))]),
                             shape=X.shape)


def csr_topk(S, k):
    S = S.tocsr()
    lens = np.diff(S.indptr)
    rows = np.repeat(np.arange(S.shape[0], dtype=np.int32), lens)
    order = np.lexsort((-S.data, rows))
    rank = np.arange(len(order)) - np.repeat(S.indptr[:-1], lens)
    sel = order[rank < k]
    r = (rank[rank < k] + 1).astype(np.int16)
    return rows[sel], S.indices[sel].astype(np.int32), S.data[sel].astype(np.float32), r


def sparse_topk(Q, TT, k, n_jobs, chunk=256):
    """Q: csr [nq, V]; TT: csr [V, nt] (= T.T). -> (qi, ti, score, rank). Threads (sparsetools drops the GIL)."""
    if k <= 0 or Q.shape[0] == 0:
        e = np.zeros(0, np.int32)
        return e, e, np.zeros(0, np.float32), np.zeros(0, np.int16)

    def run(a):
        qi, ti, s, r = csr_topk(Q[a:a + chunk] @ TT, k)
        return qi + a, ti, s, r

    res = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run)(a) for a in range(0, Q.shape[0], chunk))
    return tuple(np.concatenate(x) for x in zip(*res))


def rowwise_dot(A, B, qi, ti, chunk=500_000):
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        out[s:s + chunk] = np.asarray(A[qi[s:s + chunk]].multiply(B[ti[s:s + chunk]]).sum(1)).ravel()
    return out


# ---------------------------------------------------------------- dense (MLX)
def dense_topk(Eq, Et, k, score_budget=1.5e8):
    """Eq [nq, D], Et [nt, D] float16 numpy -> (qi, ti, score, rank). Chunked matmul + argpartition on GPU."""
    import mlx.core as mx
    nq, nt = len(Eq), len(Et)
    k = min(k, nt)
    if k <= 0 or nq == 0:
        e = np.zeros(0, np.int32)
        return e, e, np.zeros(0, np.float32), np.zeros(0, np.int16)
    T = mx.array(Et)
    rows = int(np.clip(score_budget // nt, 16, 2048))
    QI, TI, SC, RK = [], [], [], []
    for a in range(0, nq, rows):
        q = mx.array(Eq[a:a + rows])
        s = q @ T.T
        idx = mx.argpartition(-s, kth=k - 1, axis=1)[:, :k]
        val = mx.take_along_axis(s, idx, axis=1)
        o = mx.argsort(-val, axis=1)
        idx, val = mx.take_along_axis(idx, o, axis=1), mx.take_along_axis(val, o, axis=1)
        mx.eval(idx, val)
        n = idx.shape[0]
        QI.append(np.repeat(np.arange(a, a + n, dtype=np.int32), k))
        TI.append(np.array(idx).astype(np.int32).ravel())
        SC.append(np.array(val.astype(mx.float32)).ravel())
        RK.append(np.tile(np.arange(1, k + 1, dtype=np.int16), n))
    del T
    mx.clear_cache()
    return tuple(np.concatenate(x) for x in (QI, TI, SC, RK))


def gathered_cos(E, qi, ti, chunk=400_000):
    """Row-wise cosine of E[qi] and E[ti] (L2-normalised float16 memmap), dot product done in MLX."""
    import mlx.core as mx
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        a = mx.array(np.asarray(E[qi[s:s + chunk]])).astype(mx.float32)
        b = mx.array(np.asarray(E[ti[s:s + chunk]])).astype(mx.float32)
        c = (a * b).sum(axis=1)
        mx.eval(c)
        out[s:s + chunk] = np.array(c)
    return out


class ConcatEmb:
    """Retrieval embedding = [name, addr] / sqrt(2): cosine = mean(name cos, addr cos). Used when no 'combo' field."""
    def __init__(self, En, Ea):
        self.En, self.Ea = En, Ea
        self.shape = (En.shape[0], En.shape[1] + Ea.shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return (np.concatenate([np.asarray(self.En[idx]), np.asarray(self.Ea[idx])], axis=1)
                * np.float16(0.70710678))


# ---------------------------------------------------------------- one group
def _texts(rec, idx, cols):
    s = rec[cols[0]].iloc[idx].astype(str).to_numpy(dtype=object)
    for c in cols[1:]:
        s = s + " " + rec[c].iloc[idx].astype(str).to_numpy(dtype=object)
    return s.tolist()


def block_group(rec, q_idx, t_idx, E_ret, cfg, n_jobs, log=print):
    """q_idx / t_idx: global record indices of S1 / S2uS3 in this group. -> DataFrame of union pairs."""
    parts, t0 = {}, time.time()
    nq, nt = len(q_idx), len(t_idx)
    n_docs = nq + nt
    # max_df as an absolute count with a floor, so small (smoke) groups keep their common n-grams
    max_df_char = max(int(cfg["TFIDF_MAX_DF"] * n_docs), cfg["TFIDF_MAX_DF_MIN"])
    max_df_word = max(int(cfg["TFIDF_MAX_DF"] * 4 * n_docs), cfg["TFIDF_MAX_DF_MIN"])

    # (a) char TF-IDF on core names
    names = _texts(rec, np.r_[q_idx, t_idx], ["name_core"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_df=max_df_char,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(names).tocsr()
    del names
    Qc, Tc = X[:nq], X[nq:]
    parts["tfidf"] = sparse_topk(prune_rows(Qc, cfg["PRUNE_CHAR"]), Tc.T.tocsr(), cfg["K_TFIDF"], n_jobs)
    log(f"      tfidf  {time.time() - t0:,.0f}s  vocab={X.shape[1]:,}")

    # (a2) word TF-IDF on core name + address
    combo = _texts(rec, np.r_[q_idx, t_idx], ["name_core", "addr_n"])
    vec = TfidfVectorizer(analyzer=str.split, min_df=1, max_df=max_df_word, sublinear_tf=True,
                          dtype=np.float32)
    Xw = vec.fit_transform(combo).tocsr()
    Qw, Tw = Xw[:nq], Xw[nq:]
    parts["combo"] = sparse_topk(prune_rows(Qw, cfg["PRUNE_WORD"]), Tw.T.tocsr(), cfg["K_COMBO"], n_jobs)
    log(f"      combo  {time.time() - t0:,.0f}s  vocab={Xw.shape[1]:,}")

    # (c) rare tokens + postal keys: idf-weighted overlap on tokens with df_t <= RARE_MAX_DF
    pc = rec["postal"].iloc[np.r_[q_idx, t_idx]].astype(str).to_numpy(dtype=object)
    rare_txt = [c + (" pc_" + p if p else "") for c, p in zip(combo, pc)]
    del combo, pc
    cv = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
    B = cv.fit_transform(rare_txt).tocsr()
    del rare_txt
    df_t = np.asarray(B[nq:].sum(0)).ravel()
    keep = np.flatnonzero((df_t >= 1) & (df_t <= cfg["RARE_MAX_DF"]))
    idf = np.log((nt + 1) / (df_t[keep] + 1)).astype(np.float32) + 1
    Br = B[:, keep]
    Qr, Tr = (Br[:nq] @ sparse.diags(idf)).tocsr(), Br[nq:].tocsr()
    parts["rare"] = sparse_topk(Qr, Tr.T.tocsr(), cfg["K_RARE"], n_jobs)
    log(f"      rare   {time.time() - t0:,.0f}s  rare vocab={len(keep):,}")

    # (b) dense
    Eq = np.asarray(E_ret[q_idx])
    Et = np.asarray(E_ret[t_idx])
    parts["dense"] = dense_topk(Eq, Et, cfg["K_DENSE"])
    del Eq, Et
    log(f"      dense  {time.time() - t0:,.0f}s")

    # union + ranks per retriever (numpy keys; no pandas merge -> low memory)
    keys = {n: parts[n][0].astype(np.int64) * nt + parts[n][1] for n in RET}
    uk = np.unique(np.concatenate(list(keys.values())))
    u = pd.DataFrame(index=np.arange(len(uk)))
    for n in RET:
        r = np.zeros(len(uk), np.int16)
        r[np.searchsorted(uk, keys[n])] = parts[n][3]
        u[f"r_{n}"] = r
    del keys, parts
    qi, ti = uk // nt, uk % nt
    del uk

    # exact scores of every retriever on every union pair
    u["s_tfidf"] = rowwise_dot(Qc, Tc, qi, ti)
    u["s_combo"] = rowwise_dot(Qw, Tw, qi, ti)
    u["s_rare"] = rowwise_dot(Qr, Tr, qi, ti)
    u["s_dense"] = gathered_cos(E_ret, q_idx[qi], t_idx[ti])
    u["q"] = q_idx[qi].astype(np.int32)
    u["t"] = t_idx[ti].astype(np.int32)
    log(f"      union  {time.time() - t0:,.0f}s  pairs={len(u):,}  ({len(u) / max(nq, 1):.1f}/S1)")
    return u


# ---------------------------------------------------------------- recall report
def recall_report(pairs, rec, pos_keys, n_true, groups, ks=(5, 10, 20, 30, 50)):
    """pairs: union pairs of one split (q,t global). pos_keys: int64 q*NR+t of true pairs.
    n_true: np array [n_rec] true-match count per S1 row. Returns (table, recall_at_k table)."""
    NR = len(rec)
    key = pairs["q"].to_numpy(np.int64) * NR + pairs["t"].to_numpy(np.int64)
    is_pos = np.isin(key, pos_keys)
    ctry = rec["country"].to_numpy()
    rows, rows_k = [], []
    for g, (q_idx, t_idx) in groups.items():
        m_g = np.isin(pairs["q"].to_numpy(), q_idx)
        nq, nt = len(q_idx), len(t_idx)
        tot_pos = int(n_true[q_idx].sum())
        nonsing = q_idx[n_true[q_idx] > 0]
        for name in list(RET) + ["UNION"]:
            m = m_g if name == "UNION" else (m_g & (pairs[f"r_{name}"].to_numpy() > 0))
            found = np.bincount(pairs["q"].to_numpy()[m & is_pos], minlength=NR)
            rows.append({"group": g, "retriever": name, "n_s1": nq,
                         "pair_recall": found[q_idx].sum() / max(tot_pos, 1),
                         "s1_full_recall": float((found[nonsing] == n_true[nonsing]).mean()) if len(nonsing) else np.nan,
                         "mean_cands": m.sum() / max(nq, 1),
                         "reduction_ratio": 1 - m.sum() / max(nq * nt, 1)})
            if name != "UNION":
                r = pairs[f"r_{name}"].to_numpy()
                for k in ks:
                    mk = m_g & (r > 0) & (r <= k) & is_pos
                    rows_k.append({"group": g, "retriever": name, "k": k,
                                   "pair_recall": mk.sum() / max(tot_pos, 1)})
    return pd.DataFrame(rows), pd.DataFrame(rows_k)
