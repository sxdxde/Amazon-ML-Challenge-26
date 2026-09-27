"""Shared helpers: logging, stage timer (wall / peak RSS / peak VRAM), manifests, done-markers."""
from __future__ import annotations

import json
import logging
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("v2")


def setup_logging(log_file: Path | None = None) -> None:
    if LOG.handlers:
        return
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    LOG.addHandler(h)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # Linux: KiB


def peak_vram_gb() -> float:
    if "torch" not in sys.modules:
        return 0.0
    import torch
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


@contextmanager
def timed(name: str):
    t0 = time.time()
    LOG.info("▶ %s", name)
    stats = {}
    ok = False
    try:
        yield stats
        ok = True
    finally:
        stats.update(wall_s=round(time.time() - t0, 1), peak_rss_gb=round(peak_rss_gb(), 2),
                     peak_vram_gb=round(peak_vram_gb(), 2))
        (LOG.info if ok else LOG.error)("%s %s  %.1fs  peak RSS %.2f GB  peak VRAM %.2f GB", "✔" if ok else "✖ FAILED",
                                        name, stats["wall_s"], stats["peak_rss_gb"], stats["peak_vram_gb"])


def is_done(d: Path, cfg_hash: str) -> bool:
    m = d / "manifest.json"
    return m.exists() and json.loads(m.read_text()).get("config_hash") == cfg_hash


def write_manifest(d: Path, cfg_hash: str, **info) -> None:
    d.mkdir(parents=True, exist_ok=True)
    info = {"config_hash": cfg_hash, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    (d / "manifest.json").write_text(json.dumps(info, indent=1, default=str))


def md_table(df) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def gpu_free_gb(device: str = "cuda") -> float:
    import torch
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 2 ** 30
