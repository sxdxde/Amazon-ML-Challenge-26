"""Stage `block`: candidate generation + recall diagnostics.

Per country string (EDA: 100% of true pairs share the country string, so candidates are restricted
to it — an unseen country such as France is simply another group), each S1 retrieves S2∪S3 records
via the UNION of:
  dense      exact cosine kNN on "{name} | {address}" embeddings (torch matmul; GPU if present)
  combo_word word 1-2-gram TF-IDF on name-skeleton + address tokens (strongest lexical retriever)
  combo_char char_wb 3-4-gram TF-IDF on core name + address (typo-robust)
  name_char  char_wb 2-4-gram TF-IDF on the legal-stripped, transliterated name
  addr_char  char_wb 3-4-gram TF-IDF on the address core
  name_word  word TF-IDF on consonant-skeleton name tokens (transliteration-robust, rare tokens)
  addr_word  word 1-2-gram TF-IDF on the address core
  key_*      exact match on (street number, street word) / postal code / core name / name skeleton;
             blocks larger than `key_block_cap` are skipped
Pass A (train, S1 subsample): retrieve at k_max, tune per-retriever k greedily (best marginal
recall per added candidate) until union pair recall >= target or the mean-candidate budget is hit.
Pass B (all train S1, all test S1): retrieve with the tuned k, union, cap per S1 by reciprocal-rank
fusion. The result is exactly what the matcher scores and what goes to candidate_pairs.tsv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import Config, resolve_device
from embed import load_emb
from io_utils import LOG, Timer, cand_path, is_done, lib_versions, mark_done, save_parquet
from prepare import load_records, load_truth

ABSENT = 9999
Q_CHUNK = 100_000


@dataclass
class Retriever:
    name: str
    field: str
    analyzer: str
    ngram: tuple[int, int]
    max_df: float


RETRIEVERS = [
    Retriever("combo_word", "combo_w", "word", (1, 2), 0.01),      # name skeleton + address tokens
    Retriever("combo_char", "combo", "char_wb", (3, 4), 0.01),     # name core + address, typo-robust
    Retriever("name_char", "name_core", "char_wb", (2, 4), 0.03),  # empty / garbled addresses
    Retriever("addr_char", "addr_core", "char_wb", (3, 4), 0.02),  # replaced (fake) names
    Retriever("name_word", "name_skel", "word", (1, 1), 0.01),
    Retriever("addr_word", "addr_core", "word", (1, 2), 0.01),
]
KEY_FIELDS = {"key_stnum": ("addr_stnum", "addr_stword"), "key_postal": ("addr_postal",),
              "key_name": ("name_core",), "key_skel": ("name_skel",)}
RET_NAMES = ["dense"] + [r.name for r in RETRIEVERS] + list(KEY_FIELDS)
REC_COLS = ["uid", "src", "country", "name_core", "name_skel", "addr_core", "addr_stnum",
            "addr_stword", "addr_postal"]
EMPTY_HITS = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))


# ------------------------------------------------------------------ primitives
def make_vectorizer(r: Retriever, n_docs: int, min_df: int, small_max_df: float,
                    max_df_override: float | None = None) -> TfidfVectorizer:
    base = r.max_df if max_df_override is None else max_df_override
    max_df = max(base, small_max_df) if n_docs < 50_000 else base  # tiny samples: relax
    extra = {"token_pattern": r"(?u)\b\w+\b"} if r.analyzer == "word" else {}
    return TfidfVectorizer(analyzer=r.analyzer, ngram_range=r.ngram, min_df=min(min_df, n_docs),
                           max_df=max(max_df, 2 / max(n_docs, 1)), sublinear_tf=True, dtype=np.float32,
                           lowercase=False, **extra)


# Sparse top-n runs single-threaded inside forked workers: several OpenMP runtimes (torch, sklearn,
# sparse_dot_topn) in one process can deadlock, while n_threads=None never enters an OMP region.
_PT: dict[str, sp.csr_matrix] = {}


def _topn_worker(args: tuple[str, sp.csr_matrix, int]) -> sp.csr_matrix:
    name, Q, k = args
    return sp_matmul_topn(Q, _PT[name], top_n=k, sort=True, n_threads=None, threshold=1e-6).tocsr()


def sparse_topk(Q: sp.csr_matrix, name: str, k: int, pool=None, piece: int = 4000):
    """Top-k inner product (rows L2-normalised -> cosine) of Q against _PT[name]. Local (qi, pi, rank, score)."""
    k = min(k, _PT[name].shape[1])
    if k <= 0 or Q.shape[0] == 0:
        return EMPTY_HITS
    jobs = [(name, Q[s:s + piece], k) for s in range(0, Q.shape[0], piece)]
    res = pool.map(_topn_worker, jobs, chunksize=1) if pool is not None and len(jobs) > 1 \
        else [_topn_worker(j) for j in jobs]
    C = sp.vstack(res).tocsr()
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0]), counts)
    ranks = np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts)
    return rows, C.indices.astype(np.int64), ranks, C.data.astype(np.float32)


def dense_topk(Eq: np.ndarray, P: torch.Tensor, k: int, device: str, q_chunk: int = 2048,
               p_chunk: int = 500_000):
    """Exact inner-product top-k. P (fp16) may live on the host: each p-chunk is moved to `device`
    once per call, so GPU memory ≈ queries + one pool chunk + one score block (a few GB)."""
    k = min(k, P.shape[0])
    if k <= 0 or len(Eq) == 0:
        return EMPTY_HITS
    dt = torch.float16 if device == "cuda" else torch.float32
    Q = torch.from_numpy(np.ascontiguousarray(Eq, dtype=np.float16)).to(device=device, dtype=dt)
    nq = Q.shape[0]
    best_v = torch.full((nq, k), -float("inf"), device=device)
    best_i = torch.zeros((nq, k), dtype=torch.long, device=device)
    for ps in range(0, P.shape[0], p_chunk):
        Pc = P[ps:ps + p_chunk].to(device=device, dtype=dt, non_blocking=True)
        for s in range(0, nq, q_chunk):
            sc = (Q[s:s + q_chunk] @ Pc.T).float()
            v, i = torch.topk(sc, min(k, sc.shape[1]), dim=1)
            cv = torch.cat([best_v[s:s + q_chunk], v], 1)
            ci = torch.cat([best_i[s:s + q_chunk], i + ps], 1)
            best_v[s:s + q_chunk], j = torch.topk(cv, k, dim=1)
            best_i[s:s + q_chunk] = torch.gather(ci, 1, j)
        del Pc
    qi = np.repeat(np.arange(nq), k)
    return qi, best_i.cpu().numpy().ravel(), np.tile(np.arange(k), nq), best_v.cpu().numpy().ravel()


def _key_series(df: pd.DataFrame, cols: tuple) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].fillna("")
    parts = [df[c].fillna("") for c in cols]
    ok = np.logical_and.reduce([(p != "").values for p in parts])
    joined = parts[0].str.cat(parts[1:], sep="|")
    return joined.where(ok, "")


class CountryIndex:
    """All retrieval structures for one country: TF-IDF matrices, dense pool tensor, key blocks.

    `sub` holds only this country's records (indexed by global uid). Local indices (qi / pi) are
    positions within the country's S1 rows / pool rows; self.q_idx / self.p_idx map them to uids.
    """

    def __init__(self, cfg: Config, sub: pd.DataFrame, emb: np.ndarray, device: str, rets: list[str]):
        self.cfg, self.device = cfg, device
        sub = sub.assign(combo=sub["name_core"] + " " + sub["addr_core"],
                         combo_w=sub["name_skel"] + " " + sub["addr_core"])
        is_q = sub["src"].values == 1
        q, p = sub[is_q], sub[~is_q]
        self.q_idx, self.p_idx = q["uid"].values, p["uid"].values
        self.emb_q = np.asarray(emb[self.q_idx], dtype=np.float16)
        # pool embeddings stay in host RAM (fp16) unless dense_pool_on_gpu: the GPU may be shared
        self.P_dense = None
        if len(self.p_idx):
            self.P_dense = torch.from_numpy(np.asarray(emb[self.p_idx], dtype=np.float16))
            if cfg.dense_pool_on_gpu and device == "cuda":
                self.P_dense = self.P_dense.to(device)
        self.Q, self.P = {}, {}
        _PT.clear()
        n_docs = len(q) + len(p)
        for r in RETRIEVERS:
            if r.name not in rets:
                continue
            tq = q[r.field].where(q[r.field] != "", q["name_core"]).tolist()
            tp = p[r.field].where(p[r.field] != "", p["name_core"]).tolist()
            vec = make_vectorizer(r, n_docs, cfg.tfidf_min_df, cfg.tfidf_max_df_frac, cfg.retriever_max_df.get(r.name))
            try:
                vec.fit(tq + tp)
            except ValueError:  # empty vocabulary on tiny samples
                vec = make_vectorizer(r, 1, 1, 1.0)
                vec.fit(tq + tp + ["x"])
            self.Q[r.name] = vec.transform(tq).tocsr()
            self.P[r.name] = vec.transform(tp).tocsr()
            _PT[r.name] = self.P[r.name].T.tocsr()
        self.keys = {}
        for key, cols in KEY_FIELDS.items():
            if key not in rets:
                continue
            kq, kp = _key_series(q, cols), _key_series(p, cols)
            dp = pd.DataFrame({"k": kp.values, "pi": np.arange(len(p))})
            dp = dp[dp["k"] != ""]
            sizes = dp["k"].value_counts()
            dp = dp[dp["k"].map(sizes).values <= cfg.key_block_cap]
            self.keys[key] = (kq.values, dp)
        self.pool = get_context("fork").Pool(cfg.n_jobs) if (cfg.n_jobs > 1 and self.Q) else None

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        _PT.clear()
        self.P_dense = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def query(self, q_local: np.ndarray, k: dict) -> pd.DataFrame:
        """Wide hit table [qi, pi, rank_<ret>...] (local indices) for a chunk of S1 rows."""
        hits = []
        if k.get("dense", 0) > 0 and self.P_dense is not None:
            qi, pi, rk, _ = dense_topk(self.emb_q[q_local], self.P_dense, k["dense"], self.device,
                                       self.cfg.dense_q_chunk, self.cfg.dense_p_chunk)
            hits.append(("dense", q_local[qi], pi, rk))
        for r in RETRIEVERS:
            if k.get(r.name, 0) > 0 and r.name in self.Q:
                qi, pi, rk, _ = sparse_topk(self.Q[r.name][q_local], r.name, k[r.name], self.pool)
                hits.append((r.name, q_local[qi], pi, rk))
        for key, (kq, dp) in self.keys.items():
            if k.get(key, 0) > 0:
                dq = pd.DataFrame({"k": kq[q_local], "qi": q_local})
                m = dq[dq["k"] != ""].merge(dp, on="k")
                hits.append((key, m["qi"].values, m["pi"].values, np.zeros(len(m), np.int64)))
        return hits_to_wide(hits, len(self.p_idx))

    @torch.inference_mode()
    def pair_sims(self, qi: np.ndarray, pi: np.ndarray, chunk: int = 250_000) -> dict[str, np.ndarray]:
        cd = np.empty(len(qi), np.float32)
        for s in range(0, len(qi), chunk):
            a = torch.from_numpy(self.emb_q[qi[s:s + chunk]]).float()
            b = self.P_dense[torch.from_numpy(pi[s:s + chunk]).to(self.P_dense.device)].float().cpu()
            cd[s:s + chunk] = (a * b).float().sum(1).cpu().numpy()
        out = {"cos_dense": cd}
        for name in self.Q:
            out[f"cos_{name}"] = pair_cos_sparse(self.Q[name], self.P[name], qi, pi)
        return out


def hits_to_wide(hits: list, n_p: int) -> pd.DataFrame:
    if not hits or sum(len(h[1]) for h in hits) == 0:
        return pd.DataFrame({"qi": np.zeros(0, np.int64), "pi": np.zeros(0, np.int64),
                             **{f"rank_{r}": np.zeros(0, np.int16) for r in RET_NAMES}})
    keys = np.concatenate([h[1].astype(np.int64) * n_p + h[2] for h in hits])
    uniq, inv = np.unique(keys, return_inverse=True)
    R = np.full((len(uniq), len(RET_NAMES)), ABSENT, np.int16)
    off = 0
    for name, qi, _, rk in hits:
        n = len(qi)
        R[inv[off:off + n], RET_NAMES.index(name)] = rk.astype(np.int16)  # (qi, pi) unique per retriever
        off += n
    df = pd.DataFrame(R, columns=[f"rank_{r}" for r in RET_NAMES])
    df.insert(0, "pi", uniq % n_p)
    df.insert(0, "qi", uniq // n_p)
    return df


def pair_cos_sparse(Q: sp.csr_matrix, P: sp.csr_matrix, qi: np.ndarray, pi: np.ndarray,
                    chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        out[s:s + chunk] = np.asarray(Q[qi[s:s + chunk]].multiply(P[pi[s:s + chunk]]).sum(1)).ravel()
    return out


# ------------------------------------------------------------------ selection
def select(wide: pd.DataFrame, kvec: dict) -> np.ndarray:
    m = np.zeros(len(wide), bool)
    for r, kk in kvec.items():
        if kk > 0:
            m |= wide[f"rank_{r}"].values < kk
    return m


def rrf(wide: pd.DataFrame, kvec: dict) -> np.ndarray:
    s = np.zeros(len(wide), np.float32)
    for r, kk in kvec.items():
        if kk > 0:
            rk = wide[f"rank_{r}"].values
            s += np.where(rk < kk, 1.0 / (10.0 + rk), 0.0).astype(np.float32)
    return s


def finalize(wide: pd.DataFrame, kvec: dict, cap: int) -> pd.DataFrame:
    c = wide[select(wide, kvec)].copy()
    c["rrf"] = rrf(c, kvec)
    c = c.sort_values(["qi", "rrf"], ascending=[True, False], kind="stable")
    return c[c.groupby("qi").cumcount().values < cap]


def greedy_k(wide: pd.DataFrame, y: np.ndarray, n_q: int, n_true: int, grid: dict, target: float,
             budget: float) -> tuple[dict, list]:
    """Greedy per-retriever k: repeatedly take the step with the best Δtrue/Δcandidates."""
    R = {r: wide[f"rank_{r}"].values for r in RET_NAMES}
    kvec = {r: 0 for r in RET_NAMES}
    sel = np.zeros(len(wide), bool)
    trace = []
    while True:
        tp, nc = int((sel & y).sum()), int(sel.sum())
        trace.append({**kvec, "recall": tp / max(n_true, 1), "mean_cands": nc / max(n_q, 1)})
        if tp / max(n_true, 1) >= target:
            break
        best = None
        for r in RET_NAMES:
            nxt = [g for g in grid[r] if g > kvec[r]]
            if not nxt:
                continue
            add = (~sel) & (R[r] < nxt[0])
            dpos, dc = int((add & y).sum()), int(add.sum())
            eff = dpos / dc if dc else 0.0
            if dc and (best is None or eff > best[0]):
                best = (eff, r, nxt[0], add, dc)
        if best is None or best[0] <= 0 or (nc + best[4]) / max(n_q, 1) > budget:
            break
        _, r, k2, add, _ = best
        kvec[r] = k2
        sel |= add
    return kvec, trace


# ------------------------------------------------------------------ diagnostics
def diagnostics(cand: pd.DataFrame, truth_uid: pd.DataFrame, s1_country: pd.Series,
                pool_n: pd.Series, kvec: dict) -> pd.DataFrame:
    """Pair recall, S1 full-recall rate, mean cands, reduction ratio: per retriever & union, per country."""
    rows = []
    tset = truth_uid
    for ctry in sorted(s1_country.unique()):
        qs = s1_country.index[s1_country.values == ctry]
        c = cand[cand["q"].isin(qs)]
        t = tset[tset["q"].isin(qs)]
        n_q, n_pool = len(qs), int(pool_n.get(ctry, 0))
        per_q_true = t.groupby("q").size()
        variants = {f"{r} (k={kvec.get(r, 0)})": (c[f"rank_{r}"] < kvec.get(r, 0)).values
                    for r in RET_NAMES if kvec.get(r, 0) > 0}
        variants["UNION"] = np.ones(len(c), bool)
        for name, m in variants.items():
            cc = c.loc[m, ["q", "p"]]
            hit = t.merge(cc, on=["q", "p"], how="inner")
            per_q_hit = hit.groupby("q").size().reindex(per_q_true.index, fill_value=0)
            rows.append({"country": ctry, "retriever": name, "n_s1": n_q,
                         "pair_recall": len(hit) / max(len(t), 1),
                         "s1_full_recall": float((per_q_hit == per_q_true).mean()) if len(per_q_true) else np.nan,
                         "mean_cands": len(cc) / max(n_q, 1),
                         "reduction_ratio": 1 - len(cc) / max(n_q * n_pool, 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ passes
def pass_tune(cfg: Config, meta: pd.DataFrame, emb, truth_uid: pd.DataFrame, device: str):
    """Retrieve at k_max for a subsample of train S1, then tune k greedily."""
    k_max = {**cfg.k_max, **{kf: 1 for kf in KEY_FIELDS}}
    grid = {r: sorted({g for g in cfg.k_grid if g <= k_max[r]} | {k_max[r]}) for r in cfg.k_max}
    grid |= {kf: [0, 1] for kf in KEY_FIELDS}
    rng = np.random.default_rng(cfg.seed)
    s1_all = meta.index[meta["src"].values == 1].values
    tune_q = np.sort(rng.choice(s1_all, min(len(s1_all), cfg.tune_s1_max), replace=False))
    parts = []
    for ctry in sorted(meta["country"].unique()):
        sub = load_records(cfg, "train", REC_COLS, country=ctry)
        n_q, n_p = int((sub["src"] == 1).sum()), int((sub["src"] != 1).sum())
        if n_q == 0 or n_p == 0:
            continue
        with Timer(f"block[tune] {ctry}: pool {n_p:,} at k_max"):
            ix = CountryIndex(cfg, sub, emb, device, RET_NAMES)
            q_sel = np.flatnonzero(np.isin(ix.q_idx, tune_q))
            for s in range(0, len(q_sel), Q_CHUNK):
                w = ix.query(q_sel[s:s + Q_CHUNK], k_max)
                w["q"], w["p"] = ix.q_idx[w["qi"].values], ix.p_idx[w["pi"].values]
                parts.append(w.drop(columns=["qi", "pi"]))
            ix.close()
        del sub
    wide = pd.concat(parts, ignore_index=True)
    tq = truth_uid[truth_uid["q"].isin(tune_q)]
    wide["y"] = pd.MultiIndex.from_frame(wide[["q", "p"]]).isin(pd.MultiIndex.from_frame(tq[["q", "p"]]))
    kvec, trace = greedy_k(wide, wide["y"].values, len(tune_q), len(tq), grid, cfg.target_recall,
                           cfg.max_mean_cands)
    pool_n = meta[meta["src"] != 1].groupby("country").size()
    diag_max = diagnostics(wide, tq, meta.loc[tune_q, "country"], pool_n, k_max)
    return kvec, diag_max, trace


def pass_build(cfg: Config, split: str, meta: pd.DataFrame, emb, kvec: dict, device: str) -> pd.DataFrame:
    """Retrieve with tuned k for every S1 of the split, union, cap, add pair similarities."""
    parts = []
    for ctry in sorted(meta["country"].unique()):
        sub = load_records(cfg, split, REC_COLS, country=ctry)
        n_q, n_p = int((sub["src"] == 1).sum()), int((sub["src"] != 1).sum())
        if n_q == 0 or n_p == 0:
            LOG.warning("%s country %r: %d S1 / %d pool — no candidates possible", split, ctry, n_q, n_p)
            continue
        with Timer(f"block[{split}] {ctry}: {n_q:,} S1 × {n_p:,} pool"):
            ix = CountryIndex(cfg, sub, emb, device, RET_NAMES)  # sims need every space
            for s in range(0, n_q, Q_CHUNK):
                w = finalize(ix.query(np.arange(s, min(s + Q_CHUNK, n_q)), kvec), kvec, cfg.cand_cap)
                for k_, v in ix.pair_sims(w["qi"].values, w["pi"].values).items():
                    w[k_] = v
                w["q"], w["p"] = ix.q_idx[w["qi"].values], ix.p_idx[w["pi"].values]
                parts.append(w.drop(columns=["qi", "pi"]))
            ix.close()
        del sub
    cand = pd.concat(parts, ignore_index=True)
    cand["q"], cand["p"] = cand["q"].astype(np.int32), cand["p"].astype(np.int32)
    return cand


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "block"
    if is_done(out) and not force:
        LOG.info("block: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg.reports.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)

    meta = load_records(cfg, "train", ["entity_id", "src", "country"])
    emb = load_emb(cfg, "train")
    truth = load_truth(cfg)
    uid = pd.Series(meta.index.values, index=meta["entity_id"].values)
    truth_uid = pd.DataFrame({"q": uid.reindex(truth["s1"]).values, "p": uid.reindex(truth["id"]).values})
    del uid, truth
    kpath = out / "k_choice.json"
    if kpath.exists() and not force:
        kc = json.loads(kpath.read_text())
        kvec, trace, diag_max = kc["k"], kc["trace"], pd.DataFrame(kc["diag_max"])
    else:
        kvec, diag_max, trace = pass_tune(cfg, meta, emb, truth_uid, device)
        kpath.write_text(json.dumps({"k": kvec, "trace": trace, "diag_max": diag_max.to_dict("records")}, indent=1))
    LOG.info("tuned k: %s", kvec)

    n_train = -1
    if not cand_path(cfg.art, "train").exists() or force:
        cand = pass_build(cfg, "train", meta, emb, kvec, device)
        pool_n = meta[meta["src"] != 1].groupby("country").size()
        diag = diagnostics(cand, truth_uid, meta.loc[meta["src"].values == 1, "country"], pool_n, kvec)
        LOG.info("train blocking (tuned k):\n%s", diag.round(4).to_string())
        write_report(cfg, diag, diag_max, kvec, trace)
        save_parquet(cand, cand_path(cfg.art, "train"))
        n_train = len(cand)
        del cand
    del meta, emb

    meta = load_records(cfg, "test", ["entity_id", "src", "country"])
    emb = load_emb(cfg, "test")
    cand_t = pass_build(cfg, "test", meta, emb, kvec, device)
    save_parquet(cand_t, cand_path(cfg.art, "test"))
    s1 = meta[meta["src"] == 1]
    ts = pd.DataFrame({"s1_total": s1.groupby("country").size()})
    cc = meta["country"].values[cand_t["q"].values]
    ts["pairs"] = pd.Series(cc).value_counts()
    ts["s1_with_cands"] = pd.Series(cc[~cand_t["q"].duplicated().values]).value_counts()
    ts = ts.fillna(0)
    ts["mean_cands"] = ts["pairs"] / ts["s1_total"]
    with open(cfg.reports / "blocking.md", "a") as f:
        f.write("\n## Test candidate statistics (same tuned k)\n\n" + ts.to_markdown() + "\n")
    LOG.info("test candidates per country:\n%s", ts.to_string())
    mark_done(out, {"k": kvec, "embed_model": cfg.embed.name, "versions": lib_versions(),
                    "train_pairs": int(n_train), "test_pairs": int(len(cand_t))})


def write_report(cfg: Config, diag: pd.DataFrame, diag_max: pd.DataFrame, kvec: dict, trace: list) -> None:
    lines = ["# Blocking diagnostics", "",
             f"Profile `{cfg.profile}`, embedder `{cfg.embed.name}`. Candidates restricted to the same country string.",
             f"Target union pair recall {cfg.target_recall:.3f}, budget {cfg.max_mean_cands} mean cands/S1, "
             f"hard cap {cfg.cand_cap}/S1 (reciprocal-rank fusion).", "", f"**Tuned k:** `{json.dumps(kvec)}`", "",
             "## Train, all S1, at tuned k (this is the final candidate set)", "",
             diag.round(4).to_markdown(index=False), "",
             "## Tuning subsample: each retriever at k_max (upper bounds)", "",
             diag_max.round(4).to_markdown(index=False), "",
             "## Greedy k trace (tuning subsample)", "", pd.DataFrame(trace).round(4).to_markdown(index=False), ""]
    (cfg.reports / "blocking.md").write_text("\n".join(lines))
