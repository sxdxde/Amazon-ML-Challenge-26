"""Stage 2 `lexvec`: dense lexical vectors that replace v1's CPU TF-IDF retrievers.

Per field (name_core -> name_lex, addr_core -> addr_lex), per split and country:
  1. char_wb 3-4-gram counts via HashingVectorizer (2**20 features, no sign flipping, no norm)
  2. sublinear tf (1+log tf) x IDF from the SAME split+country pool (hashed document frequency);
     n-grams present in > lex_max_df of the pool get weight 0
  3. x fixed-seed Gaussian projection (2**20 -> lex_dim), L2-normalise, float16
Output: lexvec/<split>_<field>.npy  [n_pos, lex_dim] float16 aligned to records `pos`.
Stage `lexval` (dev) compares projected cosine / recall against exact TF-IDF with the same weights.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .common import LOG, is_done, timed, write_manifest
from .config import Config
from .records import load_country, load_index


def _hv(cfg: Config) -> HashingVectorizer:
    return HashingVectorizer(analyzer="char_wb", ngram_range=tuple(cfg.lex_ngram), n_features=cfg.lex_n_features,
                             alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)


def _counts(texts: list[str], cfg: Config) -> sparse.csr_matrix:
    X = _hv(cfg).transform(texts).tocsr()
    X.sum_duplicates()
    return X


def _df_chunk(texts: list[str], cfg: Config) -> np.ndarray:
    X = _counts(texts, cfg)
    return np.bincount(X.indices, minlength=cfg.lex_n_features).astype(np.int64)


def weighted(texts: list[str], idf: np.ndarray, cfg: Config) -> sparse.csr_matrix:
    """Exact TF-IDF rows (sublinear tf x idf), un-normalised."""
    X = _counts(texts, cfg)
    X.data = (1.0 + np.log(X.data)) * idf[X.indices]
    X.eliminate_zeros()
    return X


def _proj_chunk(texts: list[str], idf: np.ndarray, R_path: str, cfg: Config) -> np.ndarray:
    R = np.load(R_path, mmap_mode="r")
    X = weighted(texts, idf, cfg)
    Y = np.asarray(X @ R, dtype=np.float32)
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = np.divide(Y, n, out=np.zeros_like(Y), where=n > 0)
    return Y.astype(np.float16)


def projection(cfg: Config):
    d = cfg.art.parent / "shared"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"R_{cfg.lex_n_features}_{cfg.lex_dim}_s{cfg.seed}.npy"
    if not p.exists():
        rng = np.random.default_rng(cfg.seed + 7)
        mm = np.lib.format.open_memmap(p.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32,
                                       shape=(cfg.lex_n_features, cfg.lex_dim))
        step = 1 << 16
        for s in range(0, cfg.lex_n_features, step):
            mm[s:s + step] = rng.standard_normal((min(step, cfg.lex_n_features - s), cfg.lex_dim), dtype=np.float32) / np.sqrt(cfg.lex_dim)
        mm.flush()
        del mm
        p.with_suffix(".tmp.npy").rename(p)
    return p


def idf_for(texts_pool: list[str], cfg: Config, par: Parallel) -> np.ndarray:
    n = len(texts_pool)
    ch = cfg.lex_chunk
    dfs = par(delayed(_df_chunk)(texts_pool[s:s + ch], cfg) for s in range(0, n, ch))
    df = np.sum(dfs, axis=0) if dfs else np.zeros(cfg.lex_n_features, np.int64)
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    idf[df > cfg.lex_max_df * n] = 0.0
    return idf


def field_texts(cfg: Config, split: str, country: str, col: str) -> list[str]:
    return load_country(cfg, split, country, [col])[col].astype(str).tolist()


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "lexvec"
    h = cfg.hash("profile", "seed", "lex_fields", "lex_ngram", "lex_n_features", "lex_max_df", "lex_dim")
    if is_done(out, h) and not force:
        LOG.info("lexvec: cached (%s), skipping", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    R_path = str(projection(cfg))
    index = load_index(cfg)
    stats = {}
    with Parallel(n_jobs=cfg.n_jobs, backend="loky", max_nbytes="1M") as par:
        for split, countries in index.items():
            n_all = max(v["hi"] for v in countries.values())
            for name, col in cfg.lex_fields.items():
                path = out / f"{split}_{name}.npy"
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n_all, cfg.lex_dim))
                for c, v in countries.items():
                    with timed(f"lexvec {split}/{c}/{name}") as st:
                        texts = field_texts(cfg, split, c, col)
                        pool = texts[v["pool_lo"] - v["lo"]:]
                        idf = idf_for(pool, cfg, par)
                        np.save(out / f"idf_{split}_{name}_{c}.npy", idf)
                        ch = cfg.lex_chunk
                        parts = par(delayed(_proj_chunk)(texts[s:s + ch], idf, R_path, cfg) for s in range(0, len(texts), ch))
                        mm[v["lo"]:v["hi"]] = np.concatenate(parts) if parts else np.zeros((0, cfg.lex_dim), np.float16)
                        zero = float((np.abs(mm[v["lo"]:v["hi"]]).sum(1) == 0).mean())
                    stats[f"{split}/{c}/{name}"] = {"n": v["hi"] - v["lo"], "zero_rows": round(zero, 4),
                                                    "kept_ngrams": int((idf > 0).sum()), **st}
                mm.flush()
                del mm
    write_manifest(out, h, stats=stats)


def load_lex(cfg: Config, split: str, name: str) -> np.ndarray:
    return np.load(cfg.art / "lexvec" / f"{split}_{name}.npy", mmap_mode="r")
