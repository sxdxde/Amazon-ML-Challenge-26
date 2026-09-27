"""TSV I/O, logging, artifact bookkeeping and submission writers."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger("ber")
MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    LOG.setLevel(level)
    LOG.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


class Timer:
    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        LOG.info("▶ %s", self.what)
        return self

    def __exit__(self, *exc) -> None:
        LOG.info("✔ %s (%.1fs)", self.what, time.time() - self.t0)


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a challenge TSV: all strings, empty stays '', no quote processing."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, encoding="utf-8")


def country_slug(country: str) -> str:
    """Filesystem-safe name for an arbitrary country string (open set)."""
    base = re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_")[:40] or "x"
    return f"{base}_{hashlib.md5(country.encode()).hexdigest()[:6]}"


# ------------------------------------------------------------------ artifacts
def done_path(stage_dir: Path) -> Path:
    return stage_dir / "_DONE.json"


def is_done(stage_dir: Path) -> bool:
    return done_path(stage_dir).exists()


def mark_done(stage_dir: Path, meta: dict) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    meta = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
    done_path(stage_dir).write_text(json.dumps(meta, indent=1, default=str))


def lib_versions() -> dict:
    out = {}
    for m in ["numpy", "pandas", "sklearn", "lightgbm", "rapidfuzz", "torch", "transformers", "sparse_dot_topn"]:
        try:
            out[m] = __import__(m).__version__
        except Exception:  # pragma: no cover
            out[m] = None
    return out


# ------------------------------------------------------------------ submissions
def id_lists(s1_ids: Iterable[str], pairs: pd.DataFrame, s1_col: str, id_col: str,
             order_col: str | None = None) -> pd.Series:
    """Build {s1 -> 'id,id,...'} for every s1 in s1_ids (empty string if none). Dedupes IDs."""
    p = pairs[[s1_col, id_col] + ([order_col] if order_col else [])]
    if order_col:
        p = p.sort_values([s1_col, order_col], ascending=[True, False])
    p = p.drop_duplicates([s1_col, id_col])
    joined = p.groupby(s1_col, sort=False)[id_col].agg(",".join)
    return pd.Series(list(s1_ids), dtype=object).map(joined).fillna("")


def write_id_file(path: Path, header: list[str], s1_ids: list[str], lists: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, lst in zip(s1_ids, lists):
            f.write(f"{s1}\t{lst}\n")


def check_submission(match_path: Path, cand_path: Path, required_s1: set[str],
                     valid_ids: set[str] | None = None) -> list[str]:
    """Internal re-implementation of the official rules (+ strict subset check). Returns errors."""
    errors: list[str] = []
    maps = {}
    for path, header in [(match_path, MATCH_HEADER), (cand_path, CAND_HEADER)]:
        with open(path, encoding="utf-8") as f:
            head = f.readline().rstrip("\n").split("\t")
            if head != header:
                errors.append(f"{path.name}: bad header {head}")
                continue
            m: dict[str, set[str]] = {}
            for ln, line in enumerate(f, 2):
                s1, tab, rest = line.rstrip("\n").partition("\t")
                if not tab:
                    errors.append(f"{path.name}:{ln} no tab")
                    continue
                if s1 in m:
                    errors.append(f"{path.name}: duplicate row {s1}")
                ids = rest.split(",") if rest else []
                if len(ids) != len(set(ids)):
                    errors.append(f"{path.name}: duplicate ids in list for {s1}")
                bad = [i for i in ids if not i.startswith(("S2-", "S3-"))]
                if bad:
                    errors.append(f"{path.name}: bad prefix {bad[:3]} for {s1}")
                if valid_ids is not None:
                    unk = [i for i in ids if i not in valid_ids]
                    if unk:
                        errors.append(f"{path.name}: unknown ids {unk[:3]} for {s1}")
                m[s1] = set(ids)
            if set(m) != required_s1:
                errors.append(f"{path.name}: S1 set mismatch (missing {len(required_s1 - set(m))}, "
                              f"extra {len(set(m) - required_s1)})")
            maps[path.name] = m
        if len(errors) > 20:
            break
    if len(maps) == 2:
        mm, cc = maps[match_path.name], maps[cand_path.name]
        not_sub = [s for s, ids in mm.items() if not ids <= cc.get(s, set())]
        if not_sub:
            errors.append(f"{len(not_sub)} S1 have matches outside candidates, e.g. {not_sub[:3]}")
    return errors


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def write_empty_submission(test_source1: Path, out_dir: Path) -> tuple[Path, Path]:
    """Full-size all-empty submission through the real writer path (every test S1 gets one row)."""
    s1_ids = read_tsv(test_source1)["entity_id"].tolist()
    empty = pd.Series([""] * len(s1_ids), dtype=object)
    m, c = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    write_id_file(m, MATCH_HEADER, s1_ids, empty)
    write_id_file(c, CAND_HEADER, s1_ids, empty)
    return m, c


def cand_path(art: Path, split: str) -> Path:
    return art / "block" / f"{split}_candidates.parquet"


def load_candidates(cfg, split: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(cand_path(cfg.art, split), columns=columns)
