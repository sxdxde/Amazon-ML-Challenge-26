"""Orchestrator.

Examples (run from the directory that contains dataset/):
  python code/business_entity_resolution/src/run_all.py --smoke            # tiny CPU run -> output_smoke/
  python code/business_entity_resolution/src/run_all.py                    # full run -> output/
  python code/business_entity_resolution/src/run_all.py --stage features --force
  python code/business_entity_resolution/src/run_all.py --set use_xenc=False

Every stage caches its artifacts under artifacts/<profile>/<stage>/ with a _DONE.json marker and is
skipped on re-runs unless --force is given (so an interrupted run resumes).

Each stage runs in a fresh Python subprocess (unless --inline): torch, scikit-learn, LightGBM and
sparse_dot_topn each ship an OpenMP runtime, and mixing initialised runtimes in one long-lived
process can deadlock. Process isolation also returns all memory between stages.
"""
from __future__ import annotations

import argparse
import ast
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we fork worker pools; avoid tokenizer deadlocks

from config import load_config, resolve_device
from io_utils import LOG, setup_logging

STAGES = ["prepare", "embed", "block", "features", "gbm", "xenc", "gbm2", "decide", "predict"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def run_stage(name: str, cfg, force: bool) -> None:
    mod = __import__({"gbm": "train_gbm", "xenc": "train_xenc", "gbm2": "train_gbm",
                      "block": "blocking"}.get(name, name))
    if name == "gbm2":
        mod.run_stage2(cfg, force)
    elif name == "gbm":
        mod.run_stage1(cfg, force)
    else:
        mod.run(cfg, force)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU profile (writes output_smoke/)")
    ap.add_argument("--profile", default=None, choices=["full", "dev", "smoke"])
    ap.add_argument("--stage", default=None, choices=STAGES, help="run only this stage")
    ap.add_argument("--from-stage", default=None, choices=STAGES, help="run this stage and all later ones")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--root", default=None, help="directory containing dataset/ (default: cwd)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--inline", action="store_true", help="run all stages in this process (debugging)")
    ap.add_argument("--log-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    profile = "smoke" if args.smoke else (args.profile or "full")
    cfg = load_config(profile, args.root, args.data_dir, args.device, parse_overrides(args.set))
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file or str(cfg.logs / f"run_{profile}_{ts}.log")
    setup_logging(Path(log_file))
    seed_everything(cfg.seed)
    LOG.info("profile=%s root=%s data=%s device=%s n_jobs=%d", profile, cfg.root, cfg.data_dir,
             resolve_device(cfg), cfg.n_jobs)
    cfg.art.mkdir(parents=True, exist_ok=True)
    (cfg.art / "config.json").write_text(cfg.to_json())

    stages = STAGES
    if args.stage:
        stages = [args.stage]
    elif args.from_stage:
        stages = STAGES[STAGES.index(args.from_stage):]
    if not cfg.use_xenc:
        stages = [s for s in stages if s != "xenc"]  # gbm2 still runs (probability-context stacking)
    t0 = time.time()
    for st in stages:
        t = time.time()
        LOG.info("=================== stage %s ===================", st)
        if args.inline or len(stages) == 1:
            run_stage(st, cfg, args.force)
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--profile", profile, "--stage", st,
                   "--device", args.device, "--log-file", log_file, "--root", str(cfg.root),
                   "--data-dir", str(cfg.data_dir)] + (["--force"] if args.force else []) \
                + (["--set", *args.set] if args.set else [])
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                LOG.error("stage %s FAILED (exit %d) — fix and re-run; finished stages are cached", st, rc)
                return rc
        LOG.info("stage %s finished in %.1f min", st, (time.time() - t) / 60)
    LOG.info("ALL DONE in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
