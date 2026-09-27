"""Stage `xenc`: cross-encoder pair classifier (cascade on the stage-1 uncertainty band).

Scoring every candidate pair with a 568M-parameter cross-encoder is infeasible at this scale
(~60M test pairs), so the cross-encoder is a cascade inside the matcher: it scores the pairs whose
stage-1 GBM probability lies in [xenc_band_lo, xenc_band_hi] (top `xenc_max_per_s1` per S1 by p1).
The stage-2 GBM receives the score as a feature (NaN outside the band, identically on train & test).

* model: cfg.xenc_model (default BAAI/bge-reranker-v2-m3, Apache-2.0), 1 logit, BCE loss
* text:  "{name} | {address}" (raw strings, any script) for both sides
* data:  band pairs; positives = ground truth, negatives = hard negatives from blocking
* folds: `xenc_folds` S1-grouped folds -> OOF logits for every train band pair;
         test = mean logit of the fold models ("all") or fold 0 only ("first")
* bf16 autocast on CUDA, AdamW + linear warmup/decay, grad clipping, periodic eval (AUC) with
  early stopping on a held-out slice of in-fold S1 groups; each fold is cached (resumable).
"""
from __future__ import annotations

import json
import math
import time
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import Config, resolve_device
from io_utils import LOG, Timer, is_done, lib_versions, mark_done, save_parquet
from prepare import load_records
from split import s1_folds


def band(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df[(df["p1"] >= cfg.xenc_band_lo) & (df["p1"] <= cfg.xenc_band_hi)]
    d = d.sort_values(["q", "p1"], ascending=[True, False])
    return d[d.groupby("q").cumcount().values < cfg.xenc_max_per_s1].reset_index(drop=True)


def texts(rec: pd.DataFrame, idx: np.ndarray) -> list[str]:
    n, a = rec["name"].values[idx], rec["address"].values[idx]
    return [f"{x} | {y}" if y else x for x, y in zip(n, a)]


class PairScorer:
    def __init__(self, name: str, device: str, max_len: int):
        self.device, self.max_len = device, max_len
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1,
                                                                        ignore_mismatched_sizes=True)
        self.model.to(device)
        self.amp = device == "cuda" and torch.cuda.is_bf16_supported()

    def batch(self, a: list[str], b: list[str]):
        return self.tok(a, b, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)

    def logits(self, enc) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            return self.model(**enc).logits.squeeze(-1).float()

    @torch.inference_mode()
    def score(self, a: list[str], b: list[str], bs: int) -> np.ndarray:
        self.model.eval()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])[::-1]
        out = np.empty(len(a), np.float32)
        for s in range(0, len(a), bs):
            idx = order[s:s + bs]
            out[idx] = self.logits(self.batch([a[i] for i in idx], [b[i] for i in idx])).cpu().numpy()
        return out

    def state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def train_fold(cfg: Config, sc: PairScorer, A: list[str], B: list[str], y: np.ndarray,
               Ae: list[str], Be: list[str], ye: np.ndarray, salt: str) -> dict:
    rng = np.random.default_rng(zlib.crc32(salt.encode()))
    n = len(y)
    steps_per_epoch = math.ceil(n / (cfg.xenc_batch * cfg.xenc_grad_accum))
    total = max(1, steps_per_epoch * cfg.xenc_epochs)
    opt = torch.optim.AdamW(sc.model.parameters(), lr=cfg.xenc_lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = {"auc": -1.0, "step": 0, "state": None}
    bad, step, t0 = 0, 0, time.time()

    def evaluate() -> float:
        if len(np.unique(ye)) < 2:
            return 0.5
        auc = roc_auc_score(ye, sc.score(Ae, Be, cfg.xenc_infer_batch))
        sc.model.train()
        return float(auc)

    sc.model.train()
    stop = False
    for ep in range(cfg.xenc_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, cfg.xenc_batch):
            idx = perm[s:s + cfg.xenc_batch]
            enc = sc.batch([A[i] for i in idx], [B[i] for i in idx])
            loss = lossf(sc.logits(enc), torch.tensor(y[idx], dtype=torch.float32, device=sc.device))
            (loss / cfg.xenc_grad_accum).backward()
            if (s // cfg.xenc_batch + 1) % cfg.xenc_grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(sc.model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.xenc_eval_every == 0 or step == total:
                auc = evaluate()
                LOG.info("  xenc %s step %d/%d loss %.4f eval_auc %.5f (%.0fs)", salt, step, total,
                         loss.item(), auc, time.time() - t0)
                if auc > best["auc"]:
                    best = {"auc": auc, "step": step, "state": sc.state()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= cfg.xenc_patience:
                        LOG.info("  early stop at step %d (best %d)", step, best["step"])
                        stop = True
                        break
        if stop:
            break
    if best["state"] is None or step % cfg.xenc_eval_every:
        auc = evaluate()
        if auc > best["auc"]:
            best = {"auc": auc, "step": step, "state": sc.state()}
    sc.model.load_state_dict(best["state"])
    return {"best_eval_auc": best["auc"], "best_step": best["step"], "steps": step}


def run(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "xenc"
    if is_done(out) and not force:
        LOG.info("xenc: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    rec_tr = load_records(cfg, "train", ["entity_id", "src", "name", "address"])
    oof1 = pd.read_parquet(cfg.art / "gbm" / "oof.parquet")
    bt = band(oof1, cfg)
    s1 = rec_tr.loc[rec_tr["src"].values == 1, "entity_id"]
    xf = s1_folds(s1, cfg.xenc_folds, cfg.seed, salt="xenc")
    bt["xfold"] = xf.reindex(rec_tr["entity_id"].values[bt["q"].values]).values
    LOG.info("xenc band: %s train pairs (%.2f/S1, pos rate %.3f) from %s S1", f"{len(bt):,}",
             len(bt) / max(bt["q"].nunique(), 1), bt["y"].mean() if len(bt) else 0, f"{bt['q'].nunique():,}")
    A_all, B_all = texts(rec_tr, bt["q"].values), texts(rec_tr, bt["p"].values)
    rec_te = load_records(cfg, "test", ["name", "address"])
    te1 = pd.read_parquet(cfg.art / "gbm" / "test_pred.parquet")
    btt = band(te1, cfg)
    At, Bt = texts(rec_te, btt["q"].values), texts(rec_te, btt["p"].values)
    LOG.info("xenc band: %s test pairs", f"{len(btt):,}")
    fold_meta = []
    test_scores = []
    n_test_models = 1 if cfg.xenc_test_models == "first" else cfg.xenc_folds
    for f in range(cfg.xenc_folds):
        fo, ft = out / f"fold{f}_oof.parquet", out / f"fold{f}_test.npy"
        if fo.exists() and (ft.exists() or f >= n_test_models) and not force:
            LOG.info("xenc fold %d cached", f)
            if f < n_test_models:
                test_scores.append(np.load(ft))
            continue
        tr_idx = np.flatnonzero(bt["xfold"].values != f)
        va_idx = np.flatnonzero(bt["xfold"].values == f)
        # split in-fold S1 groups: eval slice for early stopping, the rest sampled for training
        qs = np.unique(bt["q"].values[tr_idx])
        rng = np.random.default_rng(zlib.crc32(f"xenc{cfg.seed}{f}".encode()))
        rng.shuffle(qs)
        per_q = max(len(tr_idx) / max(len(qs), 1), 1)
        n_eval_q = min(len(qs) // 5, int(cfg.xenc_eval_pairs / per_q) + 1)
        eval_q, pool_q = qs[:n_eval_q], qs[n_eval_q:]
        n_train_q = min(len(pool_q), int(cfg.xenc_train_pairs_per_fold / per_q) + 1)
        tr_sel = tr_idx[np.isin(bt["q"].values[tr_idx], pool_q[:n_train_q])]
        ev_sel = tr_idx[np.isin(bt["q"].values[tr_idx], eval_q)]
        with Timer(f"xenc fold {f}: train {len(tr_sel):,} / eval {len(ev_sel):,} / score {len(va_idx):,} pairs"):
            sc = PairScorer(cfg.xenc_model, device, cfg.xenc_max_len)
            y = bt["y"].values
            info = {"fold": f, "n_train": int(len(tr_sel)), "n_eval": int(len(ev_sel))}
            if len(tr_sel) and len(np.unique(y[tr_sel])) == 2:
                info |= train_fold(cfg, sc, [A_all[i] for i in tr_sel], [B_all[i] for i in tr_sel], y[tr_sel],
                                   [A_all[i] for i in ev_sel], [B_all[i] for i in ev_sel], y[ev_sel], f"fold{f}")
            else:
                LOG.warning("xenc fold %d: not enough labelled pairs to fine-tune; using base model", f)
            s = sc.score([A_all[i] for i in va_idx], [B_all[i] for i in va_idx], cfg.xenc_infer_batch)
            save_parquet(bt.iloc[va_idx][["q", "p"]].assign(xenc=s), fo)
            if f < n_test_models:
                st = sc.score(At, Bt, cfg.xenc_infer_batch) if len(At) else np.zeros(0, np.float32)
                np.save(ft, st)
                test_scores.append(st)
            if len(np.unique(y[va_idx])) == 2:
                info["oof_auc"] = float(roc_auc_score(y[va_idx], s))
            LOG.info("xenc fold %d: %s", f, info)
            fold_meta.append(info)
            del sc
            if device == "cuda":
                torch.cuda.empty_cache()
    oof = pd.concat([pd.read_parquet(out / f"fold{f}_oof.parquet") for f in range(cfg.xenc_folds)], ignore_index=True)
    save_parquet(oof, out / "oof.parquet")
    tp = btt[["q", "p"]].assign(xenc=np.mean(test_scores, axis=0) if test_scores else np.zeros(len(btt), np.float32))
    save_parquet(tp, out / "test_pred.parquet")
    (out / "folds.json").write_text(json.dumps(fold_meta, indent=1))
    mark_done(out, {"model": cfg.xenc_model, "band": [cfg.xenc_band_lo, cfg.xenc_band_hi, cfg.xenc_max_per_s1],
                    "train_band_pairs": int(len(bt)), "test_band_pairs": int(len(btt)), "device": device,
                    "test_models": n_test_models, "versions": lib_versions()})
