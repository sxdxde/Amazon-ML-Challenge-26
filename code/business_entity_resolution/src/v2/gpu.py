"""GPU exact inner-product top-k and pair scoring under a VRAM budget (torch fp16).

Vectors are L2-normalised, so inner product == cosine. Queries are kept resident in blocks; the target
side is streamed in tiles; per-tile top-k is merged into a running top-k on the GPU. Sizes derive from
cfg.vram_gb so the stage never exceeds its budget (other jobs share the A100).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from .common import LOG, gpu_free_gb

# read-only memmap slices are only read by torch; silence its "non-writable array" warning
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


def _plan(cfg, dim: int, nq: int, nt: int, vram_gb: float | None = None) -> tuple[int, int, int]:
    budget = (vram_gb or cfg.vram_gb) * 2 ** 30
    row = 2 * dim  # fp16 bytes per vector
    q_chunk = min(cfg.q_chunk, max(nq, 1), max(256, int(cfg.q_chunk * (vram_gb or cfg.vram_gb) / cfg.vram_gb)))
    q_block = int(min(nq, max(q_chunk, 0.30 * budget // row)))
    tile = int(min(nt, max(8192, 0.20 * budget // (2 * q_chunk)), max(8192, 0.15 * budget // row)))
    return max(q_block, 1), max(tile, 1), q_chunk


def check_vram(cfg) -> None:
    free = gpu_free_gb(cfg.device)
    LOG.info("GPU free %.1f GB (budget %.1f GB)", free, cfg.vram_gb)
    if free < cfg.vram_gb * 0.9:
        raise RuntimeError(f"only {free:.1f} GB VRAM free < budget {cfg.vram_gb} GB; lower --set vram_gb=... or wait")


def with_oom_retry(fn, cfg, label: str, *args, retries: int = 6, **kw):
    """Other jobs share the GPU and can grow at any time: on CUDA OOM, free memory, wait, and retry the
    whole call with a 40% smaller budget (never below 1.5 GB)."""
    import torch
    vram = cfg.vram_gb
    for attempt in range(retries + 1):
        try:
            return fn(*args, vram_gb=vram, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == retries:
                raise
            vram = max(1.5, vram * 0.6)
            LOG.warning("    CUDA OOM in %s (other jobs grew?); retry %d/%d in 60s with vram_gb=%.1f", label,
                        attempt + 1, retries, vram)
            time.sleep(60)


def topk(get_q, nq: int, get_t, nt: int, dim: int, k: int, cfg, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """get_q(lo, hi) / get_t(lo, hi) -> float16 arrays [hi-lo, dim]. Returns (idx int32 [nq,k], score f32 [nq,k]);
    idx indexes the target side (0..nt-1), -1 = padding when nt < k."""
    return with_oom_retry(_topk, cfg, label, get_q, nq, get_t, nt, dim, k, cfg, label)


def _topk(get_q, nq, get_t, nt, dim, k, cfg, label, vram_gb=None):
    import torch
    dev = torch.device(cfg.device)
    kk = min(k, nt)
    I = np.full((nq, k), -1, np.int32)
    V = np.full((nq, k), -np.inf, np.float32)
    if nq == 0 or nt == 0:
        return I, V
    q_block, tile, q_chunk = _plan(cfg, dim, nq, nt, vram_gb)
    t0 = time.time()
    with torch.inference_mode():
        for qb in range(0, nq, q_block):
            qe = min(nq, qb + q_block)
            Q = torch.from_numpy(np.ascontiguousarray(get_q(qb, qe), dtype=np.float16)).to(dev)
            bv = torch.full((qe - qb, kk), -float("inf"), device=dev, dtype=torch.float32)
            bi = torch.full((qe - qb, kk), -1, device=dev, dtype=torch.int32)
            for tb in range(0, nt, tile):
                te = min(nt, tb + tile)
                T = torch.from_numpy(np.ascontiguousarray(get_t(tb, te), dtype=np.float16)).to(dev)
                kt = min(kk, te - tb)
                for qc in range(0, qe - qb, q_chunk):
                    s = Q[qc:qc + q_chunk] @ T.T
                    v, i = torch.topk(s, kt, dim=1)
                    cv = torch.cat([bv[qc:qc + q_chunk], v.float()], 1)
                    ci = torch.cat([bi[qc:qc + q_chunk], (i + tb).int()], 1)
                    v2, j = torch.topk(cv, kk, dim=1)
                    bv[qc:qc + q_chunk] = v2
                    bi[qc:qc + q_chunk] = torch.gather(ci, 1, j)
                del T
            I[qb:qe, :kk] = bi.cpu().numpy()
            V[qb:qe, :kk] = bv.cpu().numpy()
            del Q, bv, bi
        torch.cuda.empty_cache()
    LOG.info("    topk %s: %s x %s (dim %d, k %d) in %.1fs [q_block %s, tile %s]", label, f"{nq:,}", f"{nt:,}",
             dim, k, time.time() - t0, f"{q_block:,}", f"{tile:,}")
    return I, V


def pair_scores(get_q, nq: int, get_t, nt: int, dim: int, qi: np.ndarray, ti: np.ndarray, cfg) -> np.ndarray:
    """Exact inner product for pairs (qi, ti) (local indices), streaming the same tiles as topk()."""
    return with_oom_retry(_pair_scores, cfg, "pair_scores", get_q, nq, get_t, nt, dim, qi, ti, cfg)


def _pair_scores(get_q, nq, get_t, nt, dim, qi, ti, cfg, vram_gb=None):
    import torch
    dev = torch.device(cfg.device)
    out = np.zeros(len(qi), np.float32)
    if len(qi) == 0:
        return out
    vram = vram_gb or cfg.vram_gb
    q_block, tile, _ = _plan(cfg, dim, nq, nt, vram)
    # gathered fp16 rows for both sides + fp16 product: 3 * n * dim * 2 bytes within ~20% of the budget
    pair_chunk = int(max(10_000, 0.20 * vram * 2 ** 30 // (6 * dim)))
    with torch.inference_mode():
        for qb in range(0, nq, q_block):
            qe = min(nq, qb + q_block)
            mq = np.flatnonzero((qi >= qb) & (qi < qe))
            if len(mq) == 0:
                continue
            mq = mq[np.argsort(ti[mq], kind="stable")]
            tsorted = ti[mq]
            Q = torch.from_numpy(np.ascontiguousarray(get_q(qb, qe), dtype=np.float16)).to(dev)
            for tb in range(0, nt, tile):
                te = min(nt, tb + tile)
                a, b = np.searchsorted(tsorted, [tb, te])
                if a == b:
                    continue
                T = torch.from_numpy(np.ascontiguousarray(get_t(tb, te), dtype=np.float16)).to(dev)
                for s in range(a, b, pair_chunk):
                    sel = mq[s:min(b, s + pair_chunk)]
                    qq = torch.from_numpy((qi[sel] - qb).astype(np.int64)).to(dev)
                    tt = torch.from_numpy((ti[sel] - tb).astype(np.int64)).to(dev)
                    out[sel] = (Q[qq] * T[tt]).sum(1, dtype=torch.float32).cpu().numpy()
                del T
            del Q
        torch.cuda.empty_cache()
    return out
