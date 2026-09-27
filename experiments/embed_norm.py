"""Embed "{name_norm} | {addr_norm}" (romanized + normalized text) with bge-m3 for every record.

Writes ONLY under an isolated root (never production artifacts). Same model, pooling, max_len,
fp16, batch size, shard size and .npy memmap format as src/embed.py, so downstream stages
(block, features, gbm, ...) run unchanged later with `run_all.py --root <ROM_ROOT>`.

  --preflight          : model/pipeline identity check against PRODUCTION embeddings; writes nothing
  --limit N --shard S  : dry run on the first N records per split (use with --out-subdir embed_dryrun)
  (default)            : full resumable run into <ROM_ROOT>/artifacts/full/embed/, verified before _DONE
"""
import os

os.environ["HF_HOME"] = "/path/to/amazon_ml_2026/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"          # pin refs/main snapshot (the one production used); no hub calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

SRC = "/path/to/amazon_ml_2026/code/business_entity_resolution/src"
sys.path.insert(0, SRC)

import numpy as np
import torch

from config import load_config, resolve_device
from embed import SHARD, encode, load_encoder, record_texts
from io_utils import lib_versions, mark_done
from prepare import load_records

PROD_ROOT = Path("/path/to/amazon_ml_2026").resolve()
DATA = PROD_ROOT / "dataset"
TEXT_COLS = ("name_norm", "addr_norm")
N_CHECK = 2000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rowcos(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return (a * b).sum(1)


def texts_for(cfg, split: str, cols, limit=None) -> list[str]:
    rec = load_records(cfg, split, list(cols))
    if limit:
        rec = rec.iloc[:limit]
    return record_texts(rec[cols[0]].tolist(), rec[cols[1]].tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--out-subdir", default="embed")
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    assert root != PROD_ROOT, "refusing to write into the production root"
    cfg = load_config("full", root=root, data_dir=DATA)
    prod = load_config("full", root=PROD_ROOT, data_dir=DATA)
    out_dir = cfg.art / args.out_subdir
    assert not str(out_dir.resolve()).startswith(str(prod.art.resolve())), "output would land in production artifacts"

    torch.set_num_threads(8)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg)
    em = cfg.embed
    tok, model = load_encoder(em.name, device, cfg.embed_fp16)
    dim = model.config.hidden_size
    log(f"model {em.name} pooling={em.pooling} max_len={em.max_len} batch={em.batch_size} "
        f"fp16={cfg.embed_fp16} dim={dim} device={device} | text cols {TEXT_COLS} | out {out_dir}")

    def enc(texts: list[str]) -> np.ndarray:
        bs = em.batch_size
        while True:
            try:
                return encode(texts, tok, model, device, em.pooling, em.max_len, bs, em.prefix)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(16, bs // 2)
                log(f"  CUDA OOM -> waiting 60s, retrying this shard with batch {bs}")
                time.sleep(60)

    rng = np.random.default_rng(0)
    splits = args.splits.split(",")

    # ------------------------------------------------------------ preflight: identity vs production
    if args.preflight:
        ok = True
        for split in splits:
            raw = texts_for(prod, split, ("name", "address"))
            nrm = texts_for(cfg, split, TEXT_COLS)
            assert len(raw) == len(nrm), "ROM prepare copy differs in length from production"
            P = np.load(prod.art / "embed" / f"{split}.npy", mmap_mode="r")
            assert P.shape == (len(raw), dim), f"production {split} embed shape {P.shape}"
            idx = np.sort(rng.choice(len(raw), N_CHECK, replace=False))
            c = rowcos(enc([raw[i] for i in idx]), P[idx])
            log(f"PREFLIGHT {split}: raw re-embed vs production rows (n={N_CHECK}): "
                f"mean cos={c.mean():.5f} min={c.min():.5f}")
            ok &= bool(c.min() > 0.995)
            for i in idx[:3]:
                log(f"   raw : {raw[i][:110]!r}")
                log(f"   norm: {nrm[i][:110]!r}")
        log("PREFLIGHT PASS" if ok else "PREFLIGHT FAIL")
        return 0 if ok else 1

    # ------------------------------------------------------------ run (resumable, same as embed.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        texts = texts_for(cfg, split, TEXT_COLS, args.limit)
        path = out_dir / f"{split}.npy"
        prog = out_dir / f"{split}.progress.json"
        done = set(json.loads(prog.read_text())) if (prog.exists() and path.exists()) else set()
        mm = np.lib.format.open_memmap(path, mode="r+" if done else "w+", dtype=np.float16,
                                       shape=(len(texts), dim))
        assert mm.shape == (len(texts), dim)
        t0 = time.time()
        log(f"embed {split}: {len(texts):,} texts, {len(done)} shards already done")
        for s in range(0, len(texts), args.shard):
            if s in done:
                continue
            mm[s:s + args.shard] = enc(texts[s:s + args.shard])
            mm.flush()
            done.add(s)
            prog.write_text(json.dumps(sorted(done)))
            log(f"  {split}: {min(s + args.shard, len(texts)):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
        del mm

    # ------------------------------------------------------------ verify before marking done
    ok = True
    for split in splits:
        texts = texts_for(cfg, split, TEXT_COLS, args.limit)
        mm = np.load(out_dir / f"{split}.npy", mmap_mode="r")
        lo, hi, nbad = 9.0, 0.0, 0
        for s in range(0, len(mm), 500_000):
            nr = np.linalg.norm(np.asarray(mm[s:s + 500_000], np.float32), axis=1)
            lo, hi = min(lo, float(nr.min())), max(hi, float(nr.max()))
            nbad += int(((nr < 0.99) | (nr > 1.01) | ~np.isfinite(nr)).sum())
        idx = np.sort(rng.choice(len(texts), min(N_CHECK, len(texts)), replace=False))
        c = rowcos(enc([texts[i] for i in idx]), mm[idx])
        log(f"VERIFY {split}: shape={mm.shape} norms[{lo:.4f},{hi:.4f}] bad_rows={nbad} | "
            f"re-encode vs stored (n={len(idx)}): mean cos={c.mean():.5f} min={c.min():.5f}")
        ok &= (nbad == 0) and bool(c.min() > 0.995)
    if not ok:
        log("VERIFY FAIL -- _DONE.json NOT written")
        return 1
    mark_done(out_dir, {"model": em.name, "pooling": em.pooling, "max_len": em.max_len, "dim": dim,
                        "device": device, "text": "name_norm | addr_norm", "limit": args.limit,
                        "hf_snapshot": (Path(os.environ["HF_HOME"]) / "hub/models--BAAI--bge-m3/refs/main").read_text().strip(),
                        "versions": lib_versions()})
    log("VERIFY PASS -- _DONE.json written. ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
