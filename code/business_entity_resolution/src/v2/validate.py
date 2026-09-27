"""Stage `lexval` (dev): does the 256-d lexical projection preserve exact TF-IDF retrieval?

Per train country and lexical field:
  * exact TF-IDF top-k (same hashed char 3-4 grams, same IDF, L2-normalised; sparse_dot_topn) vs
    lexvec top-k (GPU)  -> pair recall@k of the true matches
  * Spearman(projected cosine, exact cosine) on ~50k pairs (true + random + hard exact-top-10 non-matches)
  * union recall: v2 (sem + name_lex + addr_lex) vs v1-like (sem + exact name TF-IDF + exact addr TF-IDF)
Writes reports/v2_lexval_<profile>.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

from . import gpu
from .common import LOG, md_table, timed
from .config import Config
from .lexvec import field_texts, load_lex, weighted
from .records import load_index, load_truth
from .search import getter


def exact_topk(Xq: sparse.csr_matrix, Xp: sparse.csr_matrix, k: int):
    from sparse_dot_topn import sp_matmul_topn
    C = sp_matmul_topn(Xq, Xp.T.tocsr(), top_n=k, sort=True, n_threads=None, threshold=1e-9).tocsr()
    counts = np.diff(C.indptr)
    q = np.repeat(np.arange(C.shape[0]), counts)
    rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts) + 1
    return q.astype(np.int64), C.indices.astype(np.int64), rank.astype(np.int16)


def dense_hits(I: np.ndarray):
    nq, k = I.shape
    q = np.repeat(np.arange(nq, dtype=np.int64), k)
    t = I.ravel().astype(np.int64)
    rk = np.tile(np.arange(1, k + 1, dtype=np.int16), nq)
    ok = t >= 0
    return q[ok], t[ok], rk[ok]


def recall_at(hits, nt: int, tk: np.ndarray, k: int) -> tuple[float, float]:
    q, t, rk = hits
    m = rk <= k
    keys = q[m] * nt + t[m]
    return float(np.isin(tk, keys).mean()) if len(tk) else float("nan"), m.sum()


def union_recall(hit_list, nt: int, tk: np.ndarray, k: int, nq: int) -> tuple[float, float]:
    keys = np.unique(np.concatenate([h[0][h[2] <= k] * nt + h[1][h[2] <= k] for h in hit_list]))
    return float(np.isin(tk, keys).mean()), len(keys) / max(nq, 1)


def run(cfg: Config, force: bool = False) -> None:
    index = load_index(cfg)
    truth = load_truth(cfg)
    K = max(cfg.recall_ks)
    rec_rows, sp_rows, un_rows = [], [], []
    rng = np.random.default_rng(cfg.seed)
    for c, v in index["train"].items():
        nq, nt = v["n_s1"], v["n_pool"]
        tr = truth[(truth["s1_pos"] >= v["s1_lo"]) & (truth["s1_pos"] < v["s1_hi"])]
        tq, tt = tr["s1_pos"].to_numpy() - v["s1_lo"], tr["t_pos"].to_numpy() - v["pool_lo"]
        tk = tq * nt + tt
        gq, dim = getter(cfg, "train", "sem", v["s1_lo"])
        gt, _ = getter(cfg, "train", "sem", v["pool_lo"])
        sem = dense_hits(gpu.topk(gq, nq, gt, nt, dim, K, cfg, f"{c}/sem")[0])
        v2_list, v1_list = [sem], [sem]
        for name, col in cfg.lex_fields.items():
            with timed(f"lexval {c}/{name}"):
                idf = np.load(cfg.art / "lexvec" / f"idf_train_{name}_{c}.npy")
                X = normalize(weighted(field_texts(cfg, "train", c, col), idf, cfg)).tocsr()
                Xq, Xp = X[:nq], X[nq:]
                ex = exact_topk(Xq, Xp, K)
                L = load_lex(cfg, "train", name)
                lq = lambda a, b, lo=v["s1_lo"]: L[lo + a:lo + b]
                lt = lambda a, b, lo=v["pool_lo"]: L[lo + a:lo + b]
                lx = dense_hits(gpu.topk(lq, nq, lt, nt, cfg.lex_dim, K, cfg, f"{c}/{name}")[0])
                v2_list.append(lx)
                v1_list.append(ex)
                for k in cfg.recall_ks:
                    r_ex, n_ex = recall_at(ex, nt, tk, k)
                    r_lx, n_lx = recall_at(lx, nt, tk, k)
                    rec_rows.append({"country": c, "field": name, "k": k, "exact_tfidf_recall": round(r_ex, 4),
                                     "lexvec_recall": round(r_lx, 4), "delta": round(r_lx - r_ex, 4)})
                # Spearman on true + random + hard (exact top-10 non-matches) pairs
                n1 = min(len(tq), 20_000)
                i1 = rng.choice(len(tq), n1, replace=False)
                pq_ = [tq[i1], rng.integers(0, nq, 20_000)]
                pt_ = [tt[i1], rng.integers(0, nt, 20_000)]
                hard = np.flatnonzero((ex[2] <= 10) & ~np.isin(ex[0] * nt + ex[1], tk))
                hard = rng.choice(hard, min(len(hard), 10_000), replace=False)
                pq_.append(ex[0][hard]); pt_.append(ex[1][hard])
                a, b = np.concatenate(pq_), np.concatenate(pt_)
                exact_cos = np.asarray(Xq[a].multiply(Xp[b]).sum(1)).ravel()
                proj_cos = (np.asarray(L[v["s1_lo"] + a], np.float32) * np.asarray(L[v["pool_lo"] + b], np.float32)).sum(1)
                nz = (exact_cos != 0) | (proj_cos != 0)
                sp_rows.append({"country": c, "field": name, "pairs": int(len(a)),
                                "spearman_all": round(float(spearmanr(proj_cos, exact_cos)[0]), 4),
                                "spearman_nonzero": round(float(spearmanr(proj_cos[nz], exact_cos[nz])[0]), 4),
                                "spearman_true+hard": round(float(spearmanr(proj_cos[:n1].tolist() + proj_cos[n1 + 20_000:].tolist(),
                                                                            exact_cos[:n1].tolist() + exact_cos[n1 + 20_000:].tolist())[0]), 4)})
                del X, Xq, Xp
        for k in cfg.recall_ks:
            r2, m2 = union_recall(v2_list, nt, tk, k, nq)
            r1, m1 = union_recall(v1_list, nt, tk, k, nq)
            rs, ms = recall_at(sem, nt, tk, k)
            un_rows.append({"country": c, "k": k, "sem_only": round(rs, 4), "v2_union(sem+lexvec)": round(r2, 4),
                            "v2_cands/S1": round(m2, 1), "v1like_union(sem+exact_tfidf)": round(r1, 4),
                            "v1like_cands/S1": round(m1, 1)})
    rec, sp, un = pd.DataFrame(rec_rows), pd.DataFrame(sp_rows), pd.DataFrame(un_rows)
    lines = [f"# lexvec validation ({cfg.profile}, train)", "",
             f"lex_dim={cfg.lex_dim}, char_wb {tuple(cfg.lex_ngram)}, hashed 2^{int(np.log2(cfg.lex_n_features))}, max_df={cfg.lex_max_df}", "",
             "## Recall@k: lexvec vs exact TF-IDF (same features and weights)", md_table(rec), "",
             "## Spearman(projected cosine, exact cosine)", md_table(sp), "",
             "## Union recall: v2 (sem + lexvec) vs v1-like (sem + exact TF-IDF)", md_table(un), ""]
    cfg.reports.mkdir(parents=True, exist_ok=True)
    p = cfg.reports / f"v2_lexval_{cfg.profile}.md"
    p.write_text("\n".join(lines))
    LOG.info("lexval report -> %s\n%s\n%s\n%s", p, rec.to_string(index=False), sp.to_string(index=False), un.to_string(index=False))
