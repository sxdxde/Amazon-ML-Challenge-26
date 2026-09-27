"""Stage `predict`: write output/matching_results.tsv and output/candidate_pairs.tsv, then validate.

candidate_pairs.tsv = exactly the pairs the matcher scored (blocking output, every pair of which
received a stage-1 GBM probability). matching_results.tsv = decide-stage selection, a subset.
Every test S1 (every country string, including ones unseen in train) gets exactly one row in both.
"""
from __future__ import annotations

import subprocess
import sys

import pandas as pd

from config import Config
from io_utils import load_candidates, CAND_HEADER, LOG, MATCH_HEADER, check_submission, id_lists, mark_done, write_id_file
from prepare import load_records, test_dir
from prune import kept_pairs


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rec = load_records(cfg, "test", ["entity_id", "src", "country"])
    ids = rec["entity_id"].values
    s1 = rec[rec["src"] == 1]
    s1_ids = s1["entity_id"].tolist()
    blocked = load_candidates(cfg, "test")[["q", "p", "rrf"]]
    match = pd.read_parquet(cfg.art / "decide" / "test_matches.parquet")
    stage1 = pd.read_parquet(cfg.art / "gbm" / "test_pred.parquet")[["q", "p"]]
    if len(stage1) != len(blocked) or not stage1.merge(blocked, on=["q", "p"]).shape[0] == len(blocked):
        raise RuntimeError("blocking candidates differ from the set scored by the stage-1 model")
    # candidate_pairs.tsv = the pruned set the final matcher runs inference over (prune.py)
    kept = kept_pairs(cfg, "test")
    cand = blocked.merge(kept, on=["q", "p"], how="inner")
    final_stage = "gbm2" if (cfg.art / "gbm2" / "_DONE.json").exists() else "gbm"
    scored = pd.read_parquet(cfg.art / final_stage / "test_pred.parquet")[["q", "p"]]
    if len(cand) != len(kept) or scored.merge(kept, on=["q", "p"]).shape[0] != len(kept):
        raise RuntimeError("pruned candidate set is not fully scored by the final matcher")
    LOG.info("candidate_pairs: %s pairs (%.2f per S1) out of %s blocked", f"{len(cand):,}",
             len(cand) / rec["src"].eq(1).sum(), f"{len(blocked):,}")
    cp = pd.DataFrame({"s1": ids[cand["q"].values], "id": ids[cand["p"].values], "o": cand["rrf"].values})
    mp = pd.DataFrame({"s1": ids[match["q"].values], "id": ids[match["p"].values]})
    missing = mp.merge(cp, on=["s1", "id"], how="left", indicator=True)
    if (missing["_merge"] != "both").any():
        raise RuntimeError("matched pairs outside the candidate set")
    cand_lists = id_lists(s1_ids, cp, "s1", "id", "o")
    match_lists = id_lists(s1_ids, mp, "s1", "id")
    m_path, c_path = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    write_id_file(m_path, MATCH_HEADER, s1_ids, match_lists)
    write_id_file(c_path, CAND_HEADER, s1_ids, cand_lists)

    stats = pd.DataFrame({"country": s1["country"].values,
                          "n_match": match_lists.map(lambda x: 0 if not x else x.count(",") + 1).values,
                          "n_cand": cand_lists.map(lambda x: 0 if not x else x.count(",") + 1).values})
    summ = stats.groupby("country").agg(n_s1=("n_match", "size"), mean_matches=("n_match", "mean"),
                                        empty_rate=("n_match", lambda x: (x == 0).mean()),
                                        mean_cands=("n_cand", "mean"), no_cand_rate=("n_cand", lambda x: (x == 0).mean()))
    LOG.info("test prediction summary:\n%s", summ.round(3).to_string())
    cfg.reports.mkdir(parents=True, exist_ok=True)
    (cfg.reports / "predict.md").write_text("# Test prediction summary\n\n" + summ.round(4).to_markdown() + "\n")

    valid_ids = set(ids[rec["src"].values != 1])
    errs = check_submission(m_path, c_path, set(s1_ids), valid_ids)
    if errs:
        raise RuntimeError("internal submission check failed:\n" + "\n".join(errs[:20]))
    LOG.info("internal submission check: OK")
    run_official_validator(cfg, m_path, c_path)
    mark_done(cfg.art / "predict", {"out": str(out), "n_s1": len(s1_ids), "n_matches": int(len(mp)),
                                    "n_cand_pairs": int(len(cp))})


def run_official_validator(cfg: Config, m_path, c_path) -> None:
    script = cfg.root / "utils" / "validate_submission.py"
    if not script.exists():
        LOG.warning("official validator not found at %s — internal check only", script)
        return
    cmd = [sys.executable, str(script), "--matching", str(m_path), "--candidate", str(c_path),
           "--test-dir", str(test_dir(cfg))]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cfg.root)
    LOG.info("official validator output:\n%s", r.stdout.strip())
    if r.returncode != 0 or "PASS" not in r.stdout:
        raise RuntimeError(f"OFFICIAL VALIDATOR FAILED:\n{r.stdout}\n{r.stderr}")
