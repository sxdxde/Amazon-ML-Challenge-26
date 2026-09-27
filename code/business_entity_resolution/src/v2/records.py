"""Stage 1 `records`: per-record store built from v1's normalised prepare tables (read-only).

Output (per profile):
  records/<split>/country=<c>/part.parquet   rows ordered by (country, src, uid); `pos` = global position
  records/index.json                         per split/country: S1 pos range, pool pos range, counts
  records/<split>_rows.npy                   pos -> row in v1's bge-m3 memmap (== v1 uid)
  truth_train.parquet                        true pairs as (s1_pos, t_pos)
  folds.parquet                              S1-grouped fold per train S1 (reused by adapter + models)
Within a country the S1 rows come first, then S2, then S3, so S1 and pool are contiguous pos ranges.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import LOG, is_done, timed, write_manifest
from .config import Config

V1_COLS = ["uid", "entity_id", "country", "src", "name_norm", "name_core", "name_legal", "name_sorted",
           "name_skel", "name_nonlatin", "addr_norm", "addr_core", "addr_digits", "addr_postal", "addr_stnum",
           "addr_stword", "addr_landmark"]


def safe(c: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", c) or "_"


def v1_tune_uids(cfg: Config, src: np.ndarray) -> np.ndarray:
    """Exactly v1's tuning subsample: default_rng(seed).choice(S1 uids in uid order, tune_s1_max)."""
    rng = np.random.default_rng(cfg.seed)
    s1_all = np.flatnonzero(src == 1)
    return np.sort(rng.choice(s1_all, min(len(s1_all), cfg.v1_tune_s1_max), replace=False))


def dev_keep(cfg: Config, df: pd.DataFrame, truth: pd.DataFrame | None, split: str) -> np.ndarray:
    """dev: ~dev_s1_per_split S1 (every country kept) + their true matches + random pool slice per country."""
    rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))
    is_s1 = df["src"].to_numpy() == 1
    ctry = df["country"].to_numpy(dtype=object)
    frac = cfg.dev_s1_per_split / max(int(is_s1.sum()), 1)
    keep = np.zeros(len(df), bool)
    for c in np.unique(ctry):
        s1c = np.flatnonzero(is_s1 & (ctry == c))
        keep[rng.choice(s1c, max(1, min(len(s1c), round(len(s1c) * frac))), replace=False)] = True
        pc_ = np.flatnonzero(~is_s1 & (ctry == c))
        keep[rng.choice(pc_, min(len(pc_), cfg.dev_pool_per_country), replace=False)] = True
    if truth is not None:  # hash-based pandas isin (np.isin on 12M object strings is minutes)
        ids = df["entity_id"]
        chosen = ids[keep & is_s1]
        matched = truth.loc[truth["s1"].isin(chosen), "id"]
        keep |= ids.isin(matched).to_numpy(bool)
    return np.flatnonzero(keep)


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "records"
    h = cfg.hash("profile", "seed", "dev_s1_per_split", "dev_pool_per_country", "n_folds", "v1_profile", "v1_tune_s1_max")
    if is_done(out, h) and not force:
        LOG.info("records: cached (%s), skipping", out)
        return
    truth_v1 = pd.read_parquet(cfg.v1_art / "prepare" / "train_truth.parquet")  # columns s1, id (entity ids)
    index, counts = {}, {}
    for split in ("train", "test"):
        with timed(f"records {split}"):
            df = pd.read_parquet(cfg.v1_records_path(split), columns=V1_COLS, dtype_backend="pyarrow")
            assert (df["uid"].to_numpy() == np.arange(len(df))).all(), "v1 uid must equal the row index"
            src = df["src"].to_numpy()
            df["v1_tune"] = False
            if split == "train":
                df.loc[v1_tune_uids(cfg, src), "v1_tune"] = True
            LOG.info("  read %s rows", f"{len(df):,}")
            if cfg.profile == "dev":
                df = df.iloc[dev_keep(cfg, df, truth_v1 if split == "train" else None, split)]
                LOG.info("  dev sample %s rows", f"{len(df):,}")
            df = df.sort_values(["country", "src", "uid"], kind="stable").reset_index(drop=True)
            LOG.info("  sorted")
            df.insert(0, "pos", np.arange(len(df), dtype=np.int64))
            tab = pa.Table.from_pandas(df, preserve_index=False)
            tab = tab.append_column("name_tokens", pc.utf8_split_whitespace(tab["name_core"]))
            tab = tab.append_column("addr_tokens", pc.utf8_split_whitespace(tab["addr_core"]))
            tab = tab.append_column("digit_set", pc.utf8_split_whitespace(tab["addr_digits"]))
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{split}_rows.npy", df["uid"].to_numpy(np.int64))
            index[split] = {}
            ctry = df["country"].to_numpy(dtype=object)
            srcs = df["src"].to_numpy()
            for c in pd.unique(ctry):
                m = np.flatnonzero(ctry == c)
                lo, hi = int(m[0]), int(m[-1]) + 1
                n_s1 = int((srcs[lo:hi] == 1).sum())
                assert (srcs[lo:lo + n_s1] == 1).all() and (srcs[lo + n_s1:hi] != 1).all()
                index[split][c] = {"dir": f"country={safe(c)}", "lo": lo, "hi": hi, "s1_lo": lo, "s1_hi": lo + n_s1,
                                   "pool_lo": lo + n_s1, "pool_hi": hi, "n_s1": n_s1, "n_pool": hi - lo - n_s1}
                d = out / split / f"country={safe(c)}"
                d.mkdir(parents=True, exist_ok=True)
                pq.write_table(tab.slice(lo, hi - lo), d / "part.parquet", compression="zstd")
            counts[split] = {c: (v["n_s1"], v["n_pool"]) for c, v in index[split].items()}
            LOG.info("  %s: %s", split, counts[split])
            if split == "train":
                pos_of = pd.Series(df["pos"].to_numpy(), index=df["entity_id"].to_numpy(dtype=object))
                t = truth_v1[truth_v1["s1"].isin(pos_of.index) & truth_v1["id"].isin(pos_of.index)]
                tr = pd.DataFrame({"s1_pos": pos_of.reindex(t["s1"]).to_numpy(np.int64),
                                   "t_pos": pos_of.reindex(t["id"]).to_numpy(np.int64)})
                tr.to_parquet(cfg.art / "truth_train.parquet", index=False)
                # S1-grouped folds, stratified by country (every S1 of the split gets one; S1s are the groups)
                rng = np.random.default_rng(cfg.seed + 101)
                fold = np.full(len(df), -1, np.int8)
                for c, v in index[split].items():
                    s1 = np.arange(v["s1_lo"], v["s1_hi"])
                    fold[rng.permutation(s1)] = np.arange(len(s1)) % cfg.n_folds
                s1m = srcs == 1
                pd.DataFrame({"s1_pos": df["pos"].to_numpy()[s1m], "entity_id": df["entity_id"].to_numpy(dtype=object)[s1m],
                              "country": ctry[s1m], "fold": fold[s1m]}).to_parquet(cfg.art / "folds.parquet", index=False)
                LOG.info("  train truth pairs in profile: %s; folds: %s", f"{len(tr):,}",
                         np.bincount(fold[s1m], minlength=cfg.n_folds).tolist())
            del df, tab
    (out / "index.json").write_text(json.dumps(index, indent=1))
    write_manifest(out, h, counts=counts)


# ---------------------------------------------------------------- loaders for later stages
def load_index(cfg: Config) -> dict:
    return json.loads((cfg.art / "records" / "index.json").read_text())


def load_country(cfg: Config, split: str, country: str, columns: list[str] | None = None) -> pd.DataFrame:
    d = load_index(cfg)[split][country]["dir"]
    return pd.read_parquet(cfg.art / "records" / split / d / "part.parquet", columns=columns)


def load_rows(cfg: Config, split: str) -> np.ndarray:
    return np.load(cfg.art / "records" / f"{split}_rows.npy")


def load_truth(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.art / "truth_train.parquet")
