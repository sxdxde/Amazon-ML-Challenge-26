"""v2 CLI.  From code/business_entity_resolution/:

    nice -n 10 ../../.venv/bin/python -m src.v2.run --stage records --profile dev
    ... --stage lexvec | lexval | search | recall   [--splits train,test] [--force] [--set vram_gb=8 n_jobs=16]

Stages run in the order given (comma-separated). Each skips itself when its manifest matches the config.
"""
from __future__ import annotations

import argparse
import os
import time

# fewer fragmentation OOMs on a GPU shared with other jobs (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .common import LOG, setup_logging
from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="records,lexvec,lexval,search,recall (comma-separated)")
    ap.add_argument("--profile", default="dev", choices=["dev", "full"])
    ap.add_argument("--exp", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args()
    overrides = dict(kv.split("=", 1) for kv in a.set)
    cfg = load_config(a.profile, **overrides)
    from pathlib import Path
    log = Path(a.log_file) if a.log_file else cfg.root / "logs" / f"v2_{a.profile}_{a.stage.replace(',', '+')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log)
    LOG.info("v2 profile=%s stage=%s root=%s art=%s n_jobs=%d vram_gb=%.1f", a.profile, a.stage, cfg.root, cfg.art,
             cfg.n_jobs, cfg.vram_gb)
    splits = tuple(s for s in a.splits.split(",") if s)
    for st in a.stage.split(","):
        if st == "records":
            from . import records
            records.run(cfg, a.force)
        elif st == "lexvec":
            from . import lexvec
            lexvec.run(cfg, a.force)
        elif st == "lexval":
            from . import validate
            validate.run(cfg, a.force)
        elif st == "search":
            from . import search
            search.run(cfg, a.force, splits)
        elif st == "recall":
            from . import search
            search.recall_report(cfg)
        else:
            raise SystemExit(f"unknown stage {st}")
    LOG.info("done")


if __name__ == "__main__":
    main()
