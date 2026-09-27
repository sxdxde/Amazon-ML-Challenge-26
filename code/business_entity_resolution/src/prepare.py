"""Stage `prepare`: load TSVs, (sub)sample per profile, normalise, write parquet.

Outputs (artifacts/<profile>/prepare/):
  {split}_records.parquet   one row per record (S1, S2, S3), row order = integer uid
  train_truth.parquet       true links [s1, id] restricted to kept S1s
  test_dir/                 (subsampled profiles only) test_source{1,2,3}.tsv for the validator
"""
from __future__ import annotations

import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from io_utils import LOG, Timer, is_done, lib_versions, mark_done, read_tsv, save_parquet
from normalize import ADDR_FIELDS, NAME_FIELDS, normalize_records

RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


def _norm_chunk(args: tuple[list, list, list]) -> dict:
    return normalize_records(*args)


def normalise_frame(df: pd.DataFrame, n_jobs: int, chunk: int = 20_000) -> pd.DataFrame:
    names, addrs, ctry = (df[c].tolist() for c in ["business_name", "business_address", "country"])
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], ctry[i:i + chunk]) for i in range(0, len(df), chunk)]
    cols: dict[str, list] = {k: [] for k in NAME_FIELDS + ADDR_FIELDS}
    if n_jobs > 1 and len(jobs) > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            results = pool.imap(_norm_chunk, jobs, chunksize=1)
            for r in results:
                for k in cols:
                    cols[k].extend(r[k])
    else:
        for j in jobs:
            r = _norm_chunk(j)
            for k in cols:
                cols[k].extend(r[k])
    out = df.reset_index(drop=True).copy()
    for k, v in cols.items():
        out[k] = v
    return out


def explode_truth(gt: pd.DataFrame) -> pd.DataFrame:
    g = gt[gt["matched_entity_ids"].str.strip() != ""]
    e = g.assign(id=g["matched_entity_ids"].str.split(","))[["source1_entity_id", "id"]].explode("id")
    e = e.rename(columns={"source1_entity_id": "s1"})
    e["id"] = e["id"].str.strip()
    return e.drop_duplicates().reset_index(drop=True)


def _sample_other(df: pd.DataFrame, keep_ids: set, frac: float, n: int | None, seed: int) -> pd.DataFrame:
    must = df["entity_id"].isin(keep_ids)
    rest = df[~must]
    if n is not None:
        rest = rest.sample(min(n, len(rest)), random_state=seed)
    elif frac < 1.0:
        rest = rest.sample(frac=frac, random_state=seed)
    return pd.concat([df[must], rest])


def _stratified_s1(s1: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n S1 rows, at least ceil(n / n_countries) per country string when available."""
    per = math.ceil(n / s1["country"].nunique())
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in s1.groupby("country")]
    return pd.concat(parts)


def load_split(cfg: Config, split: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    d = cfg.data_dir / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s2 = read_tsv(d / f"{split}_source2.tsv")
    s3 = read_tsv(d / f"{split}_source3.tsv")
    truth = None
    if split == "train":
        truth = explode_truth(read_tsv(d / "train_ground_truth.tsv"))
        if cfg.train_s1_n is not None:
            s1 = s1.sample(min(cfg.train_s1_n, len(s1)), random_state=cfg.seed)
            truth = truth[truth["s1"].isin(set(s1["entity_id"]))]
        keep = set(truth["id"])
        s2 = _sample_other(s2, keep, cfg.train_other_frac, cfg.train_other_n and cfg.train_other_n // 2, cfg.seed)
        s3 = _sample_other(s3, keep, cfg.train_other_frac, cfg.train_other_n and cfg.train_other_n // 2, cfg.seed)
    else:
        if cfg.test_s1_n is not None:
            s1 = _stratified_s1(s1, cfg.test_s1_n, cfg.seed)
        n_other = cfg.test_other_n and cfg.test_other_n // 2
        s2 = _sample_other(s2, set(), cfg.test_other_frac, n_other, cfg.seed)
        s3 = _sample_other(s3, set(), cfg.test_other_frac, n_other, cfg.seed)
    parts = []
    for src, df in [(1, s1), (2, s2), (3, s3)]:
        df = df[RAW_COLS].copy()
        df["src"] = np.int8(src)
        parts.append(df)
    rec = pd.concat(parts, ignore_index=True)
    # sort by country string so per-country parquet reads can skip row groups (uid = this order)
    rec = rec.sort_values(["country", "src"], kind="stable").reset_index(drop=True)
    return rec, truth


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "prepare"
    if is_done(out) and not force:
        LOG.info("prepare: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    meta = {"versions": lib_versions()}
    for split in ["train", "test"]:
        with Timer(f"prepare {split}"):
            rec, truth = load_split(cfg, split)
            LOG.info("%s: %s records; per src/country:\n%s", split, f"{len(rec):,}",
                     rec.groupby(["src", "country"]).size().to_string())
            rec = normalise_frame(rec, cfg.n_jobs)
            rec = rec.rename(columns={"business_name": "name", "business_address": "address"})
            rec.insert(0, "uid", np.arange(len(rec), dtype=np.int64))
            rec.to_parquet(out / f"{split}_records.parquet", index=False, row_group_size=250_000)
            meta[f"{split}_n"] = int(len(rec))
            if truth is not None:
                save_parquet(truth, out / "train_truth.parquet")
                meta["train_true_pairs"] = int(len(truth))
            if split == "test" and cfg.profile != "full":
                write_test_dir(rec, out / "test_dir")
    mark_done(out, meta)


def write_test_dir(rec: pd.DataFrame, d: Path) -> None:
    """Materialise the subsampled test sources so the official validator can run against them."""
    d.mkdir(parents=True, exist_ok=True)
    for src in (1, 2, 3):
        sub = rec.loc[rec["src"] == src, ["entity_id", "name", "address", "country"]]
        sub = sub.rename(columns={"name": "business_name", "address": "business_address"})
        sub.to_csv(d / f"test_source{src}.tsv", sep="\t", index=False, quoting=3, escapechar=None)


def test_dir(cfg: Config) -> Path:
    return cfg.data_dir / "test" if cfg.profile == "full" else cfg.art / "prepare" / "test_dir"


def load_records(cfg: Config, split: str, columns: list[str] | None = None, country: str | None = None) -> pd.DataFrame:
    """Records of a split (row order == uid). With `country`, only that country's rows, indexed by uid."""
    path = cfg.art / "prepare" / f"{split}_records.parquet"
    if country is None:
        return pd.read_parquet(path, columns=columns)
    cols = None if columns is None else list(dict.fromkeys(["uid"] + columns))
    df = pd.read_parquet(path, columns=cols, filters=[("country", "==", country)])
    return df.set_index("uid", drop=False)


def split_countries(cfg: Config, split: str) -> list[str]:
    return sorted(pd.read_parquet(cfg.art / "prepare" / f"{split}_records.parquet", columns=["country"])["country"].unique())


def load_truth(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.art / "prepare" / "train_truth.parquet")
