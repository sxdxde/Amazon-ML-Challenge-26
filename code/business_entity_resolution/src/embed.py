"""Stage `embed`: dense embeddings of "{name} | {address}" for every record (train & test).

* model: cfg.embed (default BAAI/bge-m3, CLS pooling, L2-normalised); fp16 on CUDA.
* length-sorted dynamic batching; written into a float16 .npy memmap in shards so an interrupted
  run resumes where it stopped.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import Config, resolve_device
from io_utils import LOG, Timer, is_done, lib_versions, mark_done
from prepare import load_records

SHARD = 200_000


def record_texts(names, addrs) -> list[str]:
    return [f"{n} | {a}" if a else n for n, a in zip(names, addrs)]


def load_encoder(name: str, device: str, fp16: bool):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval().to(device)
    if fp16 and device == "cuda":
        model.half()
    return tok, model


@torch.inference_mode()
def encode(texts: list[str], tok, model, device: str, pooling: str, max_len: int, batch: int,
           prefix: str = "") -> np.ndarray:
    order = np.argsort([len(t) for t in texts], kind="stable")[::-1]
    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float16)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([prefix + texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = F.normalize(v.float(), dim=-1)
        out[idx] = v.cpu().numpy().astype(np.float16)
    return out


def emb_path(cfg: Config, split: str):
    return cfg.art / "embed" / f"{split}.npy"


def run(cfg: Config, force: bool = False) -> None:
    out_dir = cfg.art / "embed"
    if is_done(out_dir) and not force:
        LOG.info("embed: cached, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    em = cfg.embed
    tok, model = load_encoder(em.name, device, cfg.embed_fp16)
    dim = model.config.hidden_size
    LOG.info("embed model %s on %s (dim %d)", em.name, device, dim)
    for split in ["train", "test"]:
        rec = load_records(cfg, split, ["name", "address"])
        texts = record_texts(rec["name"].tolist(), rec["address"].tolist())
        del rec
        path = emb_path(cfg, split)
        prog = out_dir / f"{split}.progress.json"
        done = set(json.loads(prog.read_text())) if (prog.exists() and path.exists() and not force) else set()
        mode = "r+" if done else "w+"
        mm = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=(len(texts), dim))
        with Timer(f"embed {split}: {len(texts):,} texts"):
            for s in range(0, len(texts), SHARD):
                if s in done:
                    continue
                mm[s:s + SHARD] = encode(texts[s:s + SHARD], tok, model, device, em.pooling, em.max_len,
                                         em.batch_size, em.prefix)
                mm.flush()
                done.add(s)
                prog.write_text(json.dumps(sorted(done)))
                LOG.info("  %s: %s/%s", split, f"{min(s + SHARD, len(texts)):,}", f"{len(texts):,}")
        del mm
    mark_done(out_dir, {"model": em.name, "pooling": em.pooling, "max_len": em.max_len, "dim": dim,
                        "device": device, "versions": lib_versions()})


def load_emb(cfg: Config, split: str) -> np.ndarray:
    return np.load(emb_path(cfg, split), mmap_mode="r")
