"""Stage 3 `search`: GPU exact top-k per split per country + CPU key joins -> one long candidate table.

Retrievers
  sem        v1 bge-m3 "{name} | {address}" vectors (CLS, L2) read from v1's memmap (never recomputed)
  name_lex   stage-2 lexical vectors of the core name
  addr_lex   stage-2 lexical vectors of the core address
  key_postal exact (postal, street number) match           (blocks larger than key_block_cap skipped)
  key_stnum  exact (street number, first street word) match (v1's strongest key)
  rare_name  shared name token with pool df <= rare_df_cap, scored by summed idf, top k_rare
Reverse search (pool -> S1, top rev_k) for rev_retrievers gives rr_<r> = rank of this S1 among the
candidate's nearest S1s (0 = not in its top rev_k).

Output: cand/<split>/country=<c>/part.parquet with s1_pos, cand_pos, r_<ret> (rank, 0 = not retrieved),
s_<dense ret> (exact cosine for EVERY pair), s_rare, rr_<ret>, y (train: 1 = true match).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import gpu
from .common import LOG, is_done, md_table, timed, write_manifest
from .config import Config
from .lexvec import load_lex
from .records import load_country, load_index, load_rows, load_truth

KEYS = ("key_postal", "key_stnum", "rare_name")


def getter(cfg: Config, split: str, ret: str, lo: int):
    if ret == "sem":
        E = np.load(cfg.v1_emb_path(split), mmap_mode="r")
        rows = load_rows(cfg, split)
        return (lambda a, b: E[rows[lo + a:lo + b]]), E.shape[1]
    L = load_lex(cfg, split, ret)
    return (lambda a, b: L[lo + a:lo + b]), L.shape[1]


# ---------------------------------------------------------------- CPU key retrievers (local indices)
def _key(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    x, y = df[a].astype(str).to_numpy(dtype=object), df[b].astype(str).to_numpy(dtype=object)
    k = x + "|" + y
    k[(x == "") | (y == "")] = ""
    return k


def key_join(kq: np.ndarray, kp: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    dq = pd.DataFrame({"k": kq, "q": np.arange(len(kq))})
    dq = dq[dq["k"] != ""]
    dp = pd.DataFrame({"k": kp, "t": np.arange(len(kp))})
    dp = dp[dp["k"] != ""]
    size = dp["k"].map(dp["k"].value_counts())
    dp = dp[size.to_numpy() <= cap]
    m = dq.merge(dp, on="k")
    return m["q"].to_numpy(np.int64), m["t"].to_numpy(np.int64)


def _explode(s: pd.Series, col: str) -> pd.DataFrame:
    e = s.reset_index(drop=True).explode().dropna()
    return pd.DataFrame({"tok": e.to_numpy(dtype=object), col: e.index.to_numpy(np.int64)}).drop_duplicates()


def rare_join(tq: pd.Series, tp: pd.Series, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ep = _explode(tp, "t")
    df_tok = ep["tok"].value_counts()
    rare = df_tok[df_tok <= cfg.rare_df_cap]
    ep = ep[ep["tok"].isin(rare.index)]
    eq = _explode(tq, "q")
    eq = eq[eq["tok"].isin(rare.index)]
    m = eq.merge(ep, on="tok")
    if len(m) == 0:
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0, np.float32), np.zeros(0, np.int16)
    m["w"] = np.log(len(tp) / m["tok"].map(rare).to_numpy()).astype(np.float32)
    g = m.groupby(["q", "t"], sort=False)["w"].sum().reset_index()
    g = g.sort_values(["q", "w"], ascending=[True, False], kind="stable")
    g["rank"] = g.groupby("q").cumcount() + 1
    g = g[g["rank"] <= cfg.k_rare]
    return g["q"].to_numpy(np.int64), g["t"].to_numpy(np.int64), g["w"].to_numpy(np.float32), g["rank"].to_numpy(np.int16)


# ---------------------------------------------------------------- one split/country
def search_country(cfg: Config, split: str, c: str, v: dict, truth: pd.DataFrame | None) -> pd.DataFrame:
    nq, nt = v["n_s1"], v["n_pool"]
    hits, scores, rev = {}, {}, {}
    for r in cfg.dense_retrievers:
        gq, dim = getter(cfg, split, r, v["s1_lo"])
        gt, _ = getter(cfg, split, r, v["pool_lo"])
        I, V = gpu.topk(gq, nq, gt, nt, dim, cfg.k_wide, cfg, f"{split}/{c}/{r}")
        q = np.repeat(np.arange(nq, dtype=np.int64), I.shape[1])
        t = I.ravel().astype(np.int64)
        rk = np.tile(np.arange(1, I.shape[1] + 1, dtype=np.int16), nq)
        ok = t >= 0
        hits[r] = (q[ok], t[ok], rk[ok])
        if r in cfg.rev_retrievers:
            Ir, _ = gpu.topk(gt, nt, gq, nq, dim, cfg.rev_k, cfg, f"{split}/{c}/{r}/rev")
            tq = Ir.ravel().astype(np.int64)
            tt = np.repeat(np.arange(nt, dtype=np.int64), Ir.shape[1])
            rr = np.tile(np.arange(1, Ir.shape[1] + 1, dtype=np.int16), nt)
            ok = tq >= 0
            rev[r] = (tq[ok] * nt + tt[ok], rr[ok])
        del I, V
    rec = load_country(cfg, split, c, ["pos", "addr_postal", "addr_stnum", "addr_stword", "name_tokens"])
    qdf, pdf = rec.iloc[:nq], rec.iloc[nq:]
    q, t = key_join(_key(qdf, "addr_postal", "addr_stnum"), _key(pdf, "addr_postal", "addr_stnum"), cfg.key_block_cap)
    hits["key_postal"] = (q, t, np.ones(len(q), np.int16))
    q, t = key_join(_key(qdf, "addr_stnum", "addr_stword"), _key(pdf, "addr_stnum", "addr_stword"), cfg.key_block_cap)
    hits["key_stnum"] = (q, t, np.ones(len(q), np.int16))
    q, t, w, rk = rare_join(qdf["name_tokens"], pdf["name_tokens"], cfg)
    hits["rare_name"] = (q, t, rk)
    del rec, qdf, pdf

    keys = {r: h[0] * nt + h[1] for r, h in hits.items()}
    uk = np.unique(np.concatenate(list(keys.values())))
    out = pd.DataFrame({"s1_pos": (v["s1_lo"] + uk // nt).astype(np.int32), "cand_pos": (v["pool_lo"] + uk % nt).astype(np.int32)})
    for r, (_, _, rk) in hits.items():
        col = np.zeros(len(uk), np.int16)
        col[np.searchsorted(uk, keys[r])] = rk
        out[f"r_{r}"] = col
    s_rare = np.zeros(len(uk), np.float32)
    s_rare[np.searchsorted(uk, keys["rare_name"])] = w
    out["s_rare"] = s_rare
    uq, ut = uk // nt, uk % nt
    for r in cfg.dense_retrievers:  # exact cosine of every retriever for every union pair
        gq, dim = getter(cfg, split, r, v["s1_lo"])
        gt, _ = getter(cfg, split, r, v["pool_lo"])
        out[f"s_{r}"] = gpu.pair_scores(gq, nq, gt, nt, dim, uq, ut, cfg).astype(np.float16)
    for r, (rkeys, rr) in rev.items():
        col = np.zeros(len(uk), np.int16)
        pos = np.searchsorted(uk, rkeys)
        pos = np.minimum(pos, len(uk) - 1)
        m = uk[pos] == rkeys
        col[pos[m]] = rr[m]
        out[f"rr_{r}"] = col
    if truth is not None:
        tk = (truth["s1_pos"].to_numpy() - v["s1_lo"]) * nt + (truth["t_pos"].to_numpy() - v["pool_lo"])
        out["y"] = np.isin(uk, tk).astype(np.int8)
    return out


def run(cfg: Config, force: bool = False, splits=("train", "test")) -> None:
    out = cfg.art / "cand"
    h = cfg.hash("profile", "seed", "k_wide", "rev_k", "rev_retrievers", "dense_retrievers", "key_block_cap",
                 "rare_df_cap", "k_rare", "lex_fields", "lex_dim", "lex_max_df", "lex_ngram")
    mpath = out / "manifest.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else {}
    same_cfg = prev.get("config_hash") == h
    if prev and not same_cfg and not force:
        raise SystemExit(f"search: {out} was built with a different config; rerun with --force to rebuild")
    done_splits = set(prev.get("splits", [])) if same_cfg else set()
    if set(splits) <= done_splits and not force:
        LOG.info("search: cached for %s (%s), skipping", sorted(done_splits), out)
        return
    gpu.check_vram(cfg)
    index = load_index(cfg)
    truth = load_truth(cfg)
    stats = {}
    for split in splits:
        for c, v in index[split].items():
            d = out / split / v["dir"]
            f = d / "part.parquet"
            if f.exists() and not force:
                LOG.info("search %s/%s: exists, skipping", split, c)
                continue
            with timed(f"search {split}/{c}: {v['n_s1']:,} S1 x {v['n_pool']:,} pool") as st:
                tr = None
                if split == "train":
                    tr = truth[(truth["s1_pos"] >= v["s1_lo"]) & (truth["s1_pos"] < v["s1_hi"])]
                df = search_country(cfg, split, c, v, tr)
                d.mkdir(parents=True, exist_ok=True)
                df.to_parquet(f, index=False, compression="zstd")
            stats[f"{split}/{c}"] = {"pairs": len(df), "per_s1": round(len(df) / max(v["n_s1"], 1), 1), **st}
            LOG.info("  %s/%s: %s pairs (%.1f per S1)", split, c, f"{len(df):,}", len(df) / max(v["n_s1"], 1))
            del df
    stats = {**(prev.get("stats", {}) if same_cfg else {}), **stats}
    write_manifest(out, h, stats=stats, splits=sorted(done_splits | set(splits)))
    if "train" in splits:
        recall_report(cfg)


# ---------------------------------------------------------------- recall report
def _metrics(sel: np.ndarray, y: np.ndarray, q: np.ndarray, n_true: np.ndarray, s1: np.ndarray) -> dict:
    """sel: pair mask; q: s1_pos per pair; n_true[s1_pos]; s1: S1 positions being scored."""
    found = np.bincount(q[sel & (y == 1)], minlength=len(n_true))
    nt = n_true[s1]
    ns = nt > 0
    return {"pair_recall": found[s1].sum() / max(nt.sum(), 1),
            "s1_full_recall": float((found[s1][ns] == nt[ns]).mean()) if ns.any() else float("nan"),
            "mean_cands": np.bincount(q[sel], minlength=len(n_true))[s1].sum() / max(len(s1), 1)}


def recall_table(cfg: Config, df: pd.DataFrame, n_true: np.ndarray, s1: np.ndarray, label: str) -> list[dict]:
    q, y = df["s1_pos"].to_numpy(), df["y"].to_numpy()
    rows = []
    dense = [r for r in cfg.dense_retrievers if f"r_{r}" in df]
    for r in dense:
        rk = df[f"r_{r}"].to_numpy()
        for k in cfg.recall_ks:
            rows.append({"group": label, "retriever": r, "k": k, **_metrics((rk > 0) & (rk <= k), y, q, n_true, s1)})
    key_hit = np.zeros(len(df), bool)
    for r in KEYS:
        rk = df[f"r_{r}"].to_numpy()
        key_hit |= rk > 0
        rows.append({"group": label, "retriever": r, "k": "cap", **_metrics(rk > 0, y, q, n_true, s1)})
    for k in cfg.recall_ks:
        dm = np.zeros(len(df), bool)
        for r in dense:
            rk = df[f"r_{r}"].to_numpy()
            dm |= (rk > 0) & (rk <= k)
        rows.append({"group": label, "retriever": "UNION dense", "k": k, **_metrics(dm, y, q, n_true, s1)})
        rows.append({"group": label, "retriever": "UNION dense+keys", "k": k, **_metrics(dm | key_hit, y, q, n_true, s1)})
    return rows


def recall_report(cfg: Config) -> pd.DataFrame:
    index = load_index(cfg)
    truth = load_truth(cfg)
    n_all = max(v["hi"] for v in index["train"].values())
    n_true = np.bincount(truth["s1_pos"].to_numpy(), minlength=n_all)
    cols = ["s1_pos", "y"] + [f"r_{r}" for r in list(cfg.dense_retrievers) + list(KEYS)]
    rows, parts = [], []
    for c, v in index["train"].items():
        df = pd.read_parquet(cfg.art / "cand" / "train" / v["dir"] / "part.parquet", columns=cols)
        s1 = np.arange(v["s1_lo"], v["s1_hi"])
        rows += recall_table(cfg, df, n_true, s1, c)
        parts.append(df)
    allc = pd.concat(parts, ignore_index=True)
    s1_all = np.concatenate([np.arange(v["s1_lo"], v["s1_hi"]) for v in index["train"].values()])
    rows += recall_table(cfg, allc, n_true, s1_all, "ALL")
    # v1 reference: the same S1 subsample v1 tuned on (full profile only)
    ref_lines = []
    tune = pd.concat([load_country(cfg, "train", c, ["pos", "v1_tune"]) for c in index["train"]])
    s1_tune = tune.loc[tune["v1_tune"].to_numpy(bool), "pos"].to_numpy()
    if len(s1_tune) > 1000:
        rows += recall_table(cfg, allc, n_true, s1_tune, "v1-tune-sample")
        kc = cfg.v1_art / "block" / "k_choice.json"
        if kc.exists():
            k = json.loads(kc.read_text())
            last = k["trace"][-1]
            ref_lines.append(f"v1 final (tuned k, cap 80) on its tuning sample: pair recall {last['recall']:.4f} at {last['mean_cands']:.1f} candidates/S1")
            for d in k["diag_max"]:
                if d["retriever"] in ("UNION",) or d["retriever"].startswith("dense") or d["retriever"].startswith("combo_word"):
                    ref_lines.append(f"v1 {d['country']} {d['retriever']}: pair recall {d['pair_recall']:.4f}, "
                                     f"S1 full {d['s1_full_recall']:.4f}, {d['mean_cands']:.1f} cands/S1")
    tab = pd.DataFrame(rows)
    for col in ("pair_recall", "s1_full_recall", "mean_cands"):
        tab[col] = tab[col].astype(float).round(4)
    cfg.reports.mkdir(parents=True, exist_ok=True)
    name = "v2_recall.md" if cfg.profile == "full" else f"v2_recall_{cfg.profile}.md"
    lines = [f"# v2 recall ({cfg.profile}, train)", "", "Rank-k union: each dense retriever truncated at k; keys at their caps.", ""]
    lines += [md_table(tab), ""]
    if ref_lines:
        lines += ["## v1 reference (artifacts/full/block/k_choice.json)", ""] + [f"- {s}" for s in ref_lines]
    (cfg.reports / name).write_text("\n".join(lines))
    tab.to_csv(cfg.art / "cand" / "recall.csv", index=False)
    LOG.info("recall report -> %s", cfg.reports / name)
    show = tab[tab["retriever"].str.startswith("UNION") & tab["group"].isin(["ALL", "v1-tune-sample"])]
    LOG.info("\n%s", show.to_string(index=False))
    return tab
