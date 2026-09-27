"""Shared helpers for entity_resolution.ipynb: timing/RSS logging, PASS/FAIL lines,
stage caching, TSV loading and train/test sampling. CPU only, no country hard-coding."""
import hashlib
import json
import os
import resource
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

SECTION_LOG = []  # rows for the cell-14 summary table


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def peak_rss_gb():
    # macOS ru_maxrss is in bytes (Linux: KiB)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


@contextmanager
def section(name):
    t0 = time.time()
    print(f"── {name} ── start (RSS {rss_gb():.2f} GB)", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        SECTION_LOG.append({"section": name, "wall_s": round(dt, 1), "rss_gb": round(rss_gb(), 2),
                            "peak_rss_gb": round(peak_rss_gb(), 2)})
        print(f"── {name} ── {dt:,.1f}s | RSS {rss_gb():.2f} GB | peak RSS {peak_rss_gb():.2f} GB", flush=True)


def check(ok, msg):
    print(("✅ PASS: " if ok else "❌ FAIL: ") + msg, flush=True)
    return bool(ok)


def paste(msg):
    print(">>> PASTE TO CLAUDE: " + msg, flush=True)


# ---------------------------------------------------------------- stage cache
def stage_dir(art, name, params, force=False):
    """Return (dir, fresh). fresh=True means cached outputs with identical params exist."""
    d = Path(art) / name
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
    meta = d / "_params.json"
    done = d / "_DONE"
    old_key = json.loads(meta.read_text()).get("_key") if meta.exists() else None
    fresh = (not force) and done.exists() and old_key == key
    if not fresh:
        done.unlink(missing_ok=True)
        if force or (old_key is not None and old_key != key):
            # params changed (or FORCE): drop stale partial outputs; same params -> keep them to resume
            for p in d.rglob("*"):
                if p.is_file():
                    p.unlink()
        meta.write_text(json.dumps({**params, "_key": key}, indent=1, default=str))
    return d, fresh


def mark_done(d):
    (Path(d) / "_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------- IO
def read_tsv(p, **kw):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, **kw)


def parse_ground_truth(gt):
    """-> DataFrame(s1, t) of positive pairs, and Series n_true per S1 (0 for singletons)."""
    lists = gt["matched_entity_ids"].str.split(",")
    ex = pd.DataFrame({"s1": gt["source1_entity_id"].to_numpy(), "t": lists}).explode("t")
    ex = ex[ex["t"].notna() & (ex["t"] != "")]
    n_true = ex.groupby("s1").size().reindex(gt["source1_entity_id"].to_numpy(), fill_value=0)
    return ex.reset_index(drop=True), n_true


def _segments(addr):
    """Address segments (split on commas), accent-stripped + casefolded. Country-agnostic."""
    s = addr.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.casefold().str.split(",")


def region_sample_train(s1, s23, pos, n_target, seed, seg_min=200, seg_max=20000):
    """Pick whole regions (frequent address segments = cities/states) until ~n_target S1s are covered.

    Keeps the local candidate density realistic: every S2/S3 record that shares a chosen region
    segment is kept (real competitors), plus all true matches of the chosen S1s.
    """
    rng = np.random.default_rng(seed)
    seg1 = _segments(s1["business_address"]).explode().str.strip()
    seg1 = seg1[seg1.str.len() >= 3]
    cnt = seg1.groupby(seg1).size()
    anchors = cnt[(cnt >= seg_min) & (cnt <= seg_max)].index.to_numpy()
    rng.shuffle(anchors)
    seg_rows = seg1.index.to_numpy()
    pos_by_seg = pd.Series(seg1.to_numpy()).groupby(seg1.to_numpy()).indices
    chosen, rows = [], set()
    for a in anchors:
        chosen.append(a)
        rows.update(seg_rows[pos_by_seg[a]].tolist())
        if len(rows) >= n_target:
            break
    s1_keep = s1.iloc[sorted(rows)]
    chosen = set(chosen)
    keep_ids = set(pos.loc[pos["s1"].isin(set(s1_keep["entity_id"])), "t"])
    parts = []
    for start in range(0, len(s23), 1_000_000):
        chunk = s23.iloc[start:start + 1_000_000]
        seg = _segments(chunk["business_address"]).explode().str.strip()
        hit = seg.isin(chosen).groupby(level=0).any()
        parts.append(chunk.loc[hit[hit].index])
    s23_keep = pd.concat(parts + [s23[s23["entity_id"].isin(keep_ids)]]).drop_duplicates("entity_id")
    return s1_keep.reset_index(drop=True), s23_keep.reset_index(drop=True), len(chosen)


def stratified_rows(df, n_total, rng, col="country", min_per_group=1):
    """~n_total rows, proportional per value of `col`, every value kept (open set)."""
    frac = min(1.0, n_total / max(len(df), 1))
    take = []
    for _, idx in df.groupby(col).indices.items():
        k = min(len(idx), max(min_per_group, int(round(len(idx) * frac))))
        take.append(rng.choice(idx, k, replace=False))
    return df.iloc[np.sort(np.concatenate(take))]


def random_sample_train(s1, s23, pos, n_s1, seed, distractor_mult=3):
    """Smoke sample: n_s1 S1 stratified by country + all their matches + distractor_mult x random others."""
    rng = np.random.default_rng(seed)
    s1_keep = stratified_rows(s1, n_s1, rng)
    keep_ids = set(pos.loc[pos["s1"].isin(set(s1_keep["entity_id"])), "t"])
    matched = s23[s23["entity_id"].isin(keep_ids)]
    others = s23[~s23["entity_id"].isin(keep_ids)]
    n_other = min(len(others), distractor_mult * max(len(matched), len(s1_keep)))
    others = others.iloc[rng.choice(len(others), n_other, replace=False)]
    return s1_keep.reset_index(drop=True), pd.concat([matched, others]).reset_index(drop=True)


def smoke_sample_test(t1, t23, n_s1, seed, n_random_per_country=3000, cap=40000):
    """Smoke test sample: n_s1 S1 (every country kept) + S2/S3 sharing a distinctive name token
    with them (realistic candidates) + random records per country."""
    rng = np.random.default_rng(seed)
    t1_keep = stratified_rows(t1, n_s1, rng)

    def toks(s):
        s = s.str.normalize("NFKD").str.casefold().str.replace(r"[^\w]+", " ", regex=True)
        return s.str.split()

    qt = toks(t1_keep["business_name"]).explode()
    qt = set(qt[qt.str.len() >= 5].dropna())
    hits = []
    for start in range(0, len(t23), 1_000_000):
        chunk = t23.iloc[start:start + 1_000_000]
        tk = toks(chunk["business_name"]).explode()
        hit = tk.isin(qt).groupby(level=0).any()
        hits.append(chunk.loc[hit[hit].index])
    hit_df = pd.concat(hits)
    if len(hit_df) > cap:
        hit_df = hit_df.iloc[rng.choice(len(hit_df), cap, replace=False)]
    rand = pd.concat([t23.iloc[rng.choice(idx, min(len(idx), n_random_per_country), replace=False)]
                      for _, idx in t23.groupby("country").indices.items()])
    t23_keep = pd.concat([hit_df, rand]).drop_duplicates("entity_id")
    return t1_keep.reset_index(drop=True), t23_keep.reset_index(drop=True)


def build_records(s1, s2, s3):
    """One table per split, global row index = position. src in {1,2,3}.

    Each source is sorted by entity_id so the row order (which every cached artifact — embeddings,
    candidate pairs, features, POS_KEYS — is indexed by) never depends on how the sample was drawn.
    """
    parts = []
    for src, df in ((1, s1), (2, s2), (3, s3)):
        d = df[["entity_id", "business_name", "business_address", "country"]].sort_values("entity_id", kind="stable").copy()
        d["src"] = np.int8(src)
        parts.append(d)
    rec = pd.concat(parts, ignore_index=True)
    rec["business_name"] = rec["business_name"].fillna("")
    rec["business_address"] = rec["business_address"].fillna("")
    return rec


def nfkd_ascii_fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()
