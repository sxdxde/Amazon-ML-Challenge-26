"""Phase-1 exploratory data analysis. Writes reports/eda.md.

Usage (from student_resource/):
    python code/business_entity_resolution/src/eda.py [--data-dir dataset] [--out reports/eda.md]
"""
from __future__ import annotations

import argparse
import gc
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

RNG = np.random.default_rng(42)
# Country-agnostic numeric patterns (no US/IN-specific regex).
RE_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")
RE_POSTAL_SPACED = re.compile(r"(?<!\d)\d{3}\s\d{3}(?!\d)")  # e.g. Indian "560 001"
RE_LEAD_NUM = re.compile(r"^\s*\d+[a-zA-Z]?\b")
RE_ANY_DIGIT = re.compile(r"\d")
RE_NONASCII = re.compile(r"[^\x00-\x7f]")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def simple_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^\w]+", " ", s).strip()


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.lines += ["", "#" * level + " " + text, ""]
        print("#" * level, text, flush=True)

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, df: pd.DataFrame, index: bool = True) -> None:
        self.lines.append(df.to_markdown(index=index))
        self.lines.append("")

    def code(self, text: str) -> None:
        self.lines += ["```", text, "```", ""]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def basic_stats(rep: Report, name: str, df: pd.DataFrame) -> None:
    rep.h(3, f"{name}: {len(df):,} rows")
    rep.p(f"Columns: {list(df.columns)}")
    empties = {c: int((df[c].str.strip() == "").sum()) for c in df.columns}
    rep.p(f"Empty/blank per column: {empties}")
    vc = df["country"].value_counts()
    rep.p("Country counts: " + ", ".join(f"`{k!r}`={v:,}" for k, v in vc.items()))
    dup_ids = int(df["entity_id"].duplicated().sum())
    prefixes = df["entity_id"].str[:3].value_counts().to_dict()
    rep.p(f"Duplicate entity_ids: {dup_ids}; ID prefixes: {prefixes}")
    exact_dup = int(df.duplicated(["business_name", "business_address", "country"]).sum())
    rep.p(f"Exact duplicate (name,address,country) rows within file: {exact_dup:,}")


def per_country_field_stats(rep: Report, name: str, df: pd.DataFrame) -> None:
    rows = []
    for c, g in df.groupby("country"):
        nm, ad = g["business_name"], g["business_address"]
        rows.append({
            "country": c,
            "n": len(g),
            "name_len_med": nm.str.len().median(),
            "name_len_p95": nm.str.len().quantile(0.95),
            "addr_len_med": ad.str.len().median(),
            "addr_len_p95": ad.str.len().quantile(0.95),
            "addr_empty%": 100 * (ad.str.strip() == "").mean(),
            "postal5-6%": 100 * ad.str.contains(RE_POSTAL).mean(),
            "postal_3+3%": 100 * ad.str.contains(RE_POSTAL_SPACED).mean(),
            "lead_street_num%": 100 * ad.str.contains(RE_LEAD_NUM).mean(),
            "addr_any_digit%": 100 * ad.str.contains(RE_ANY_DIGIT).mean(),
            "name_nonascii%": 100 * nm.str.contains(RE_NONASCII).mean(),
            "addr_nonascii%": 100 * ad.str.contains(RE_NONASCII).mean(),
        })
    rep.p(f"**{name}**")
    rep.table(pd.DataFrame(rows).round(1), index=False)


def script_profile(rep: Report, name: str, df: pd.DataFrame, n: int = 200_000) -> None:
    """% of rows containing each non-ASCII Unicode script (first word of the char name)."""
    sub = df.sample(min(n, len(df)), random_state=0)
    for c, g in sub.groupby("country"):
        for col in ["business_name", "business_address"]:
            cnt: Counter = Counter()
            for s in g[col]:
                blocks = {unicodedata.name(ch, "UNK").split()[0] for ch in s if ord(ch) > 127}
                cnt.update("LATIN-ACCENT" if b == "LATIN" else b for b in blocks)
            top = ", ".join(f"{k} {100*v/len(g):.1f}%" for k, v in cnt.most_common(9))
            rep.p(f"- {name} / {c} / {col}: {top or 'ASCII only'}")


def samples(rep: Report, name: str, df: pd.DataFrame, n: int = 15) -> None:
    for c, g in df.groupby("country"):
        s = g.sample(min(n, len(g)), random_state=42)
        rep.p(f"**{name} / {c}**")
        rep.table(s[["entity_id", "business_name", "business_address"]], index=False)


def explode_gt(gt: pd.DataFrame) -> pd.DataFrame:
    g = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    g["m"] = g["matched_entity_ids"].str.split(",")
    e = g[["source1_entity_id", "m"]].explode("m").rename(columns={"source1_entity_id": "s1", "m": "id"})
    e["id"] = e["id"].str.strip()
    return e.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="reports/eda.md")
    args = ap.parse_args()
    t0 = time.time()
    data = Path(args.data_dir)
    rep = Report()
    rep.p("# Phase 1 EDA — Business Entity Resolution")
    rep.p(f"Generated by `src/eda.py` on {time.strftime('%Y-%m-%d %H:%M')}. Seed 42.")

    # ------------------------------------------------------------ train load
    s1 = read_tsv(data / "train/train_source1.tsv")
    s2 = read_tsv(data / "train/train_source2.tsv")
    s3 = read_tsv(data / "train/train_source3.tsv")
    gt = read_tsv(data / "train/train_ground_truth.tsv")
    print(f"loaded train in {time.time()-t0:.0f}s", flush=True)

    rep.h(2, "1. File overview (train)")
    for nm, df in [("train_source1", s1), ("train_source2", s2), ("train_source3", s3)]:
        basic_stats(rep, nm, df)
    rep.h(3, f"train_ground_truth: {len(gt):,} rows")
    rep.p(f"Columns: {list(gt.columns)}; empty matched lists: {(gt['matched_entity_ids'].str.strip()=='').sum():,}")
    rep.p(f"Duplicate source1_entity_id rows in GT: {gt['source1_entity_id'].duplicated().sum()}")

    # ------------------------------------------------------------ test overview
    rep.h(2, "2. File overview (test)")
    test_frames = {}
    for nm in ["test_source1", "test_source2", "test_source3"]:
        df = read_tsv(data / f"test/{nm}.tsv")
        basic_stats(rep, nm, df)
        test_frames[nm] = df
    rep.h(3, "Test field statistics per country")
    for nm, df in test_frames.items():
        per_country_field_stats(rep, nm, df)
    rep.h(3, "Test: non-ASCII scripts per field (% of rows, 200k sample)")
    for nm, df in test_frames.items():
        script_profile(rep, nm, df)
    rep.h(3, "Test samples (15 per source per country)")
    for nm, df in test_frames.items():
        samples(rep, nm, df)
    del test_frames
    gc.collect()

    # ------------------------------------------------------------ train field stats + samples
    rep.h(2, "3. Train field statistics per country")
    for nm, df in [("train_source1", s1), ("train_source2", s2), ("train_source3", s3)]:
        per_country_field_stats(rep, nm, df)
    rep.h(3, "Train: non-ASCII scripts per field (% of rows, 200k sample)")
    for nm, df in [("train_source1", s1), ("train_source2", s2), ("train_source3", s3)]:
        script_profile(rep, nm, df)
    rep.h(2, "4. Train samples (15 per source per country)")
    for nm, df in [("train_source1", s1), ("train_source2", s2), ("train_source3", s3)]:
        samples(rep, nm, df)

    # ------------------------------------------------------------ ground truth
    rep.h(2, "5. Ground truth analysis")
    ex = explode_gt(gt)
    n_match = gt["matched_entity_ids"].apply(lambda x: 0 if not x.strip() else len(x.split(",")))
    rep.p(f"S1 entities in GT: {len(gt):,}; singletons (empty list): {(n_match==0).sum():,} "
          f"= **{100*(n_match==0).mean():.2f}%**")
    rep.p(f"Total true pairs: {len(ex):,}; mean matches per non-singleton: {n_match[n_match>0].mean():.2f}")
    dist = n_match.clip(upper=15).value_counts().sort_index()
    rep.p("Match-count distribution (15 = 15+):")
    rep.table(pd.DataFrame({"n_matches": dist.index, "n_s1": dist.values,
                            "pct": (100 * dist.values / len(gt)).round(2)}), index=False)

    ex["src"] = ex["id"].str[:2]
    comp = ex.groupby("s1")["src"].agg(lambda s: "".join(sorted(set(s))))
    comp_vc = comp.map({"S2": "S2-only", "S3": "S3-only", "S2S3": "both"}).value_counts()
    rep.p("Non-singleton composition: " + ", ".join(f"{k}={v:,} ({100*v/len(comp):.1f}%)" for k, v in comp_vc.items()))
    per_src = ex.groupby(["s1", "src"]).size().unstack(fill_value=0)
    rep.p(f"Among non-singletons: mean #S2 matches={per_src.get('S2', pd.Series(0)).mean():.2f}, "
          f"mean #S3 matches={per_src.get('S3', pd.Series(0)).mean():.2f}; "
          f"S1s with >1 S2 match: {(per_src.get('S2', 0) > 1).sum():,}; with >1 S3 match: {(per_src.get('S3', 0) > 1).sum():,}")

    s2_ids, s3_ids = set(s2["entity_id"]), set(s3["entity_id"])
    matched_ids = set(ex["id"])
    rep.p(f"Fraction of S2 matched to any S1: {100*len(matched_ids & s2_ids)/len(s2_ids):.2f}% "
          f"({len(matched_ids & s2_ids):,}/{len(s2_ids):,})")
    rep.p(f"Fraction of S3 matched to any S1: {100*len(matched_ids & s3_ids)/len(s3_ids):.2f}% "
          f"({len(matched_ids & s3_ids):,}/{len(s3_ids):,})")
    missing = matched_ids - s2_ids - s3_ids
    rep.p(f"GT IDs not found in S2/S3 files: {len(missing):,}; non S2/S3 prefix in GT: "
          f"{(~ex['id'].str.match(r'^S[23]-')).sum()}")

    rep.h(3, "Critical check (a): one-to-many assumption")
    cnt = ex["id"].value_counts()
    multi = cnt[cnt > 1]
    rep.p(f"S2/S3 IDs appearing under >1 S1: **{len(multi):,}** ({100*len(multi)/max(1,len(cnt)):.3f}% of matched IDs); "
          f"max S1s per ID: {cnt.max() if len(cnt) else 0}")
    if len(multi):
        rep.p("Examples: " + ", ".join(f"{i}×{c}" for i, c in multi.head(10).items()))
    dup_within = ex.duplicated(["s1", "id"]).sum()
    rep.p(f"Duplicate IDs inside a single GT list: {dup_within}")

    rep.h(3, "Critical check (b): do matched pairs share the country string?")
    country = pd.concat([s1, s2, s3])[["entity_id", "country"]].set_index("entity_id")["country"]
    ex["c1"] = ex["s1"].map(country)
    ex["c2"] = ex["id"].map(country)
    same = (ex["c1"] == ex["c2"])
    rep.p(f"Same country: {same.sum():,}/{len(ex):,} = **{100*same.mean():.4f}%**")
    if (~same).any():
        rep.table(ex.loc[~same, ["c1", "c2"]].value_counts().head(10).reset_index())

    rep.h(3, "Critical check (c): S1 coverage of GT")
    s1_ids, gt_ids = set(s1["entity_id"]), set(gt["source1_entity_id"])
    rep.p(f"S1 in file: {len(s1_ids):,}; in GT: {len(gt_ids):,}; S1 missing from GT: {len(s1_ids-gt_ids):,}; "
          f"GT S1 not in file: {len(gt_ids-s1_ids):,}")

    rep.h(3, "Per-country GT stats")
    gt["country"] = gt["source1_entity_id"].map(country)
    gt["n"] = n_match.values
    pc = gt.groupby("country").agg(n_s1=("n", "size"), singleton_pct=("n", lambda x: 100 * (x == 0).mean()),
                                   mean_matches=("n", "mean"), mean_nonzero=("n", lambda x: x[x > 0].mean()))
    rep.table(pc.round(2))
    for c in sorted(s2["country"].unique()):
        n2 = (s2["country"] == c).sum()
        n3 = (s3["country"] == c).sum()
        m2 = ((ex["c2"] == c) & (ex["src"] == "S2")).sum()
        m3 = ((ex["c2"] == c) & (ex["src"] == "S3")).sum()
        rep.p(f"{c}: S2 {n2:,} rows, matched pairs into S2 {m2:,}; S3 {n3:,} rows, matched pairs into S3 {m3:,}; "
              f"S1 rows {(s1['country']==c).sum():,}")

    # ------------------------------------------------------------ pair similarity + examples
    rep.h(2, "6. Matched pairs: similarity profile and examples")
    rec = pd.concat([s1, s2, s3]).set_index("entity_id")
    del s2, s3
    gc.collect()
    ex["n1"] = rec["business_name"].reindex(ex["s1"]).values
    ex["a1"] = rec["business_address"].reindex(ex["s1"]).values
    ex["n2"] = rec["business_name"].reindex(ex["id"]).values
    ex["a2"] = rec["business_address"].reindex(ex["id"]).values
    samp = ex.sample(min(400_000, len(ex)), random_state=42).copy()
    t1 = time.time()
    nn1 = [simple_norm(x) for x in samp["n1"]]
    nn2 = [simple_norm(x) for x in samp["n2"]]
    na1 = [simple_norm(x) for x in samp["a1"]]
    na2 = [simple_norm(x) for x in samp["a2"]]
    samp["name_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(nn1, nn2)]
    samp["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(nn1, nn2)]
    samp["addr_tset"] = [fuzz.token_set_ratio(a, b) if a and b else np.nan for a, b in zip(na1, na2)]
    samp["name_exact"] = [a == b for a, b in zip(nn1, nn2)]
    samp["script_mismatch"] = [bool(RE_NONASCII.search(a)) != bool(RE_NONASCII.search(b)) for a, b in zip(samp["n1"], samp["n2"])]
    samp["addr2_empty"] = [not b for b in na2]
    print(f"pair sims in {time.time()-t1:.0f}s", flush=True)
    prof = samp.groupby(["c1", "src"]).agg(
        n=("name_tset", "size"), name_tset_med=("name_tset", "median"), name_tset_p10=("name_tset", lambda x: x.quantile(.1)),
        name_ratio_med=("name_ratio", "median"), name_exact_pct=("name_exact", lambda x: 100 * x.mean()),
        addr_tset_med=("addr_tset", "median"), addr2_empty_pct=("addr2_empty", lambda x: 100 * x.mean()),
        script_mismatch_pct=("script_mismatch", lambda x: 100 * x.mean()))
    rep.p(f"On a random sample of {len(samp):,} true pairs (normalised strings, rapidfuzz 0-100):")
    rep.table(prof.round(1))

    cols = ["s1", "n1", "a1", "id", "n2", "a2"]
    rep.h(3, "20 random matched pairs")
    rep.table(samp.sample(20, random_state=1)[cols + ["name_tset"]], index=False)

    rep.h(3, "Ground-truth quality: suspicious positive labels")
    sus = samp[(samp["name_tset"] < 35) & ((samp["addr_tset"] < 40) | samp["addr_tset"].isna()) & ~samp["script_mismatch"]]
    rep.p(f"Pairs with name token_set < 35 AND address token_set < 40 (or missing) and same script: "
          f"{len(sus):,} / {len(samp):,} = {100*len(sus)/len(samp):.2f}% of sampled true pairs.")
    rep.table(sus.sample(min(25, len(sus)), random_state=0)[cols + ["name_tset", "addr_tset"]], index=False)
    lowname = samp[(samp["name_tset"] < 35) & ~samp["script_mismatch"]]
    rep.p(f"Name token_set < 35 alone (same script): {100*len(lowname)/len(samp):.2f}% of true pairs (DBA/trade names?). Examples:")
    rep.table(lowname.sample(min(15, len(lowname)), random_state=3)[cols + ["name_tset", "addr_tset"]], index=False)

    # ------------------------------------------------------------ hard negatives
    rep.h(2, "7. Hard non-matches (same/similar name, not a match)")
    s1_sub = s1.sample(min(300_000, len(s1)), random_state=7)
    s1_sub = s1_sub.assign(k=[simple_norm(x) for x in s1_sub["business_name"]])
    other = rec.loc[rec.index.str.startswith(("S2-", "S3-"))].reset_index()
    other = other.sample(min(3_000_000, len(other)), random_state=7)
    other["k"] = [simple_norm(x) for x in other["business_name"]]
    j = s1_sub.merge(other, on=["k", "country"], suffixes=("_1", "_2"))
    truth = set(zip(ex["s1"], ex["id"]))
    j["is_match"] = [(a, b) in truth for a, b in zip(j["entity_id_1"], j["entity_id_2"])]
    rep.p(f"Exact-normalised-name joins (300k S1 × 3M S2/S3 sample, same country): {len(j):,} pairs; "
          f"match rate {100*j['is_match'].mean():.1f}% → an identical name is NOT sufficient.")
    hn = j[~j["is_match"]]
    rep.table(hn.sample(min(20, len(hn)), random_state=2)[
        ["entity_id_1", "business_name_1", "business_address_1", "entity_id_2", "business_name_2", "business_address_2"]], index=False)

    # Name-frequency: how many S2/S3 records share an S1 normalised name
    kc = other["k"].value_counts()
    rep.p(f"Top repeated normalised names in S2/S3 sample: " + ", ".join(f"`{k}`×{v}" for k, v in kc.head(15).items()))

    # ------------------------------------------------------------ ordering/leakage sanity
    rep.h(2, "8. Sanity: ID/ordering leakage check (report only, not used)")
    pos1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    num1 = ex["s1"].str[3:].astype("int64")
    num2 = ex["id"].str[3:].astype("int64")
    corr = np.corrcoef(num1, num2)[0, 1]
    rep.p(f"Pearson corr between numeric S1 id and matched S2/S3 id: {corr:.4f} (≈0 means IDs are random).")
    gtpos = pd.Series(np.arange(len(gt)), index=gt["source1_entity_id"])
    rep.p(f"GT row order vs S1 file order Spearman: "
          f"{pd.Series(pos1.reindex(gt['source1_entity_id']).values).corr(pd.Series(gtpos.values), method='spearman'):.4f}")

    rep.save(Path(args.out))
    print(f"wrote {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
