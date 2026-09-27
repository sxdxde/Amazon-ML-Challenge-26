"""Dense-only reblock: recompute ONLY the dense retriever from a new set of embeddings and reuse every
lexical / key retrieval hit from an existing candidate file (those retrievers never read embeddings).
Same tuned k, RRF, cap and column layout as blocking.pass_build, so the result is identical to a full
`block` run except for S1s whose union hit the cap (6 train / 27 test S1s in production).

  --src-root  : root holding the existing block/{split}_candidates.parquet + k_choice.json
  --emb-root  : root whose artifacts/full/embed/{split}.npy is used for the dense retriever
  --out-root  : root to write artifacts/full/<out-subdir>/{split}_candidates.parquet (never production)
Self-test: --emb-root = --src-root = production, --countries France  -> must reproduce production.
"""
import argparse
import json
import sys
import time
from pathlib import Path

SRC = "/path/to/amazon_ml_2026/code/business_entity_resolution/src"
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import torch

from blocking import ABSENT, Q_CHUNK, RET_NAMES, REC_COLS, CountryIndex, dense_topk, pair_cos_sparse, rrf, select
from config import load_config, resolve_device
from io_utils import cand_path, lib_versions, mark_done, save_parquet
from prepare import load_records, load_truth

PROD_ROOT = Path("/path/to/amazon_ml_2026").resolve()
DATA = PROD_ROOT / "dataset"
RANK_COLS = [f"rank_{r}" for r in RET_NAMES]
LEX_RANK_COLS = [c for c in RANK_COLS if c != "rank_dense"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_country(cfg, split, ctry, old, emb, kvec, device, cols_order, n_check=200_000, rng=None):
    sub = load_records(cfg, split, REC_COLS, country=ctry)
    ix =CountryIndex(cfg, sub, emb, device, RET_NAMES)       # TF-IDF fit identical to pass_build; dense from `emb`
    q_idx, p_idx = ix.q_idx, ix.p_idx
    n_uid = int(max(q_idx.max(), p_idx.max())) + 1
    q_pos = np.full(n_uid, -1, np.int64); q_pos[q_idx] = np.arange(len(q_idx))
    p_pos = np.full(n_uid, -1, np.int64); p_pos[p_idx] = np.arange(len(p_idx))

    # guard: TF-IDF reproduction must match the stored lexical cosines exactly
    s = rng.choice(len(old), min(n_check, len(old)), replace=False)
    qi_s, pi_s = q_pos[old["q"].values[s]], p_pos[old["p"].values[s]]
    assert (qi_s >= 0).all() and (pi_s >= 0).all(), "old candidates reference uids outside this country"
    worst = 0.0
    for name in ix.Q:
        d = np.abs(pair_cos_sparse(ix.Q[name], ix.P[name], qi_s, pi_s) - old[f"cos_{name}"].values[s])
        worst = max(worst, float(d.max()))
    log(f"  [{ctry}] TF-IDF reproduction check on {len(s):,} stored pairs: max |cos diff| = {worst:.2e}")
    assert worst < 1e-5, "TF-IDF reproduction mismatch -- aborting"

    # new dense top-k (same call as CountryIndex.query)
    dq, dp, dr = [], [], []
    for st in range(0, len(q_idx), Q_CHUNK):
        ql = np.arange(st, min(st + Q_CHUNK, len(q_idx)))
        qi, pi, rk, _ = dense_topk(ix.emb_q[ql], ix.P_dense, kvec["dense"], device,
                                   cfg.dense_q_chunk, cfg.dense_p_chunk)
        dq.append(ql[qi]); dp.append(pi); dr.append(rk)
    dense = pd.DataFrame({"q": q_idx[np.concatenate(dq)], "p": p_idx[np.concatenate(dp)],
                          "rank_dense": np.concatenate(dr).astype(np.int16)})

    # lexical/key hits of the old run (any non-dense retriever within its k)
    kv_lex = {r: k for r, k in kvec.items() if r != "dense"}
    lex = old[select(old, kv_lex)].drop(columns=["rank_dense", "rrf", "cos_dense"])
    m = lex.merge(dense, on=["q", "p"], how="outer")
    for c in RANK_COLS:
        m[c] = m[c].fillna(ABSENT).astype(np.int16)
    m["rrf"] = rrf(m, kvec)
    # pass_build order: per S1 by rrf desc; ties in pool-index (== p uid) order; then cap
    m = m.sort_values(["q", "rrf", "p"], ascending=[True, False, True], kind="stable")
    m = m[m.groupby("q").cumcount().values < cfg.cand_cap].reset_index(drop=True)

    qi, pi = q_pos[m["q"].values], p_pos[m["p"].values]
    cd = np.empty(len(m), np.float32)
    for st in range(0, len(m), 250_000):              # identical to CountryIndex.pair_sims cos_dense
        a = torch.from_numpy(ix.emb_q[qi[st:st + 250_000]]).float()
        b = ix.P_dense[torch.from_numpy(pi[st:st + 250_000]).to(ix.P_dense.device)].float().cpu()
        cd[st:st + 250_000] = (a * b).float().sum(1).cpu().numpy()
    m["cos_dense"] = cd
    need = m["cos_combo_word"].isna().values          # rows only the NEW dense retriever found
    for name in ix.Q:
        col = f"cos_{name}"
        vals = m[col].values.astype(np.float32)
        if need.any():
            vals[need] = pair_cos_sparse(ix.Q[name], ix.P[name], qi[need], pi[need])
        m[col] = vals
    ix.close()
    n_new = int(need.sum())
    log(f"  [{ctry}] old pairs {len(old):,} -> new pairs {len(m):,} | dense-only new pairs {n_new:,} "
        f"| S1 at cap {int((m.groupby('q').size() >= cfg.cand_cap).sum())}")
    m["q"], m["p"] = m["q"].astype(np.int32), m["p"].astype(np.int32)
    return m[cols_order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-subdir", default="block")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--countries", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    assert out_root != PROD_ROOT, "refusing to write into the production root"
    cfg_src = load_config("full", root=args.src_root, data_dir=DATA)
    cfg_emb = load_config("full", root=args.emb_root, data_dir=DATA)
    cfg = load_config("full", root=out_root, data_dir=DATA)
    out_dir = cfg.art / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    kc = json.loads((cfg_src.art / "block" / "k_choice.json").read_text())
    kvec = kc["k"]
    device = resolve_device(cfg)
    rng = np.random.default_rng(0)
    log(f"kvec={kvec} cap={cfg.cand_cap} emb={cfg_emb.art / 'embed'} out={out_dir} device={device}")

    for split in args.splits.split(","):
        emb = np.load(cfg_emb.art / "embed" / f"{split}.npy", mmap_mode="r")
        meta = load_records(cfg, split, ["entity_id", "src", "country"])
        assert emb.shape[0] == len(meta), f"embedding rows {emb.shape[0]} != records {len(meta)}"
        old_all = pd.read_parquet(cand_path(cfg_src.art, split))
        cols_order = list(old_all.columns)
        ctry_q = meta["country"].values[old_all["q"].values]
        countries = sorted(meta["country"].unique())
        if args.countries:
            countries = [c for c in countries if c in args.countries.split(",")]
        parts = []
        for ctry in countries:
            t0 = time.time()
            old = old_all[ctry_q == ctry].reset_index(drop=True)
            parts.append(build_country(cfg, split, ctry, old, emb, kvec, device, cols_order, rng=rng))
            log(f"  [{ctry}] {split} done in {time.time() - t0:.0f}s")
        cand = pd.concat(parts, ignore_index=True)
        for c in cols_order:
            assert cand[c].dtype == old_all[c].dtype, f"dtype mismatch {c}: {cand[c].dtype} vs {old_all[c].dtype}"
        save_parquet(cand, out_dir / f"{split}_candidates.parquet")
        log(f"{split}: saved {len(cand):,} pairs (old {len(old_all):,})")

        if split == "train":
            truth = load_truth(cfg)
            uid = pd.Series(meta.index.values, index=meta["entity_id"].values)
            t = pd.DataFrame({"q": uid.reindex(truth["s1"]).values, "p": uid.reindex(truth["id"]).values})
            t = t.dropna().astype(np.int64)
            t["country"] = meta["country"].values[t["q"].values]
            for name, cc in (("OLD", old_all), ("NEW", cand)):
                key = cc["q"].values.astype(np.int64) * len(meta) + cc["p"].values
                hit = np.isin(t["q"].values * len(meta) + t["p"].values, key)
                per = pd.Series(hit).groupby(t["country"].values).mean().round(5).to_dict()
                log(f"TRAIN PAIR RECALL {name}: overall={hit.mean():.5f} per-country={per} | mean cands/S1="
                    f"{len(cc) / cc['q'].nunique():.2f}")
        del old_all, cand, parts

    (out_dir / "k_choice.json").write_text(json.dumps(kc, indent=1))
    if args.out_subdir == "block" and not args.countries:
        mark_done(out_dir, {"k": kvec, "embed_model": cfg.embed.name, "method": "fast_reblock (dense-only)",
                            "src_candidates": str(cfg_src.art / "block"), "versions": lib_versions()})
    log("DONE")


if __name__ == "__main__":
    main()
