"""Cheap validation: does embedding ROMANIZED/normalized text with bge-m3 separate true vs
random pairs better than embedding RAW (native-script) text? Runs briefly, small batch, to
avoid meaningfully disturbing the concurrent xenc job.

India, US: real ground-truth true pairs exist (train). Compare median cosine(true pairs) vs
median cosine(random pairs) under RAW text vs NORMALIZED (name_norm/addr_norm, already
romanized via translit_brahmic + accent-stripped) text.

France: no train ground truth (France is test-only, unseen). Proxy check instead: for the same
France test record, how much does bge-m3's embedding change between RAW (accented) and
NFKD-accent-stripped text? A large shift suggests normalization meaningfully changes the
representation (worth testing further); near-1.0 self-similarity suggests bge-m3 already treats
accented/unaccented forms almost identically, so normalization is unlikely to help much there.
"""
import os
import sys
sys.path.insert(0, ".")
os.environ.setdefault("HF_HOME", "/path/to/amazon_ml_2026/hf_cache")
import numpy as np
import pandas as pd
import torch

from config import load_config
from embed import load_encoder, encode
from prepare import load_records, load_truth

cfg = load_config(profile="full", root="/path/to/amazon_ml_2026",
                  data_dir="/path/to/amazon_ml_2026/dataset")
rng = np.random.default_rng(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading {cfg.embed.name} on {device} ...", flush=True)
tok, model = load_encoder(cfg.embed.name, device, fp16=True)


def emb(texts, batch=32):
    return encode(texts, tok, model, device, cfg.embed.pooling, cfg.embed.max_len, batch, cfg.embed.prefix)


def sep(cos_true, cos_rand):
    return float(np.median(cos_true) - np.median(cos_rand)), float(np.median(cos_true)), float(np.median(cos_rand))


N_PAIRS = 3000

# ---------------------------------------------------------------- India / US: real ground truth
rec = load_records(cfg, "train", ["entity_id", "country", "name", "address", "name_norm", "addr_norm"])
uid = pd.Series(rec.index.values, index=rec["entity_id"].values)
truth = load_truth(cfg)
ctry_of = rec["country"]

for country in ("India", "US"):
    ids_c = set(rec.loc[rec["country"] == country, "entity_id"])
    tc = truth[truth["s1"].isin(ids_c) & truth["id"].isin(ids_c)]
    tc = tc.sample(min(N_PAIRS, len(tc)), random_state=0)
    qa = uid.reindex(tc["s1"]).to_numpy()
    pa = uid.reindex(tc["id"]).to_numpy()
    all_c = rec.index[rec["country"] == country].to_numpy()
    ra = rng.choice(all_c, len(qa), replace=False)
    rb = rng.choice(all_c, len(qa), replace=False)

    def texts_for(field_n, field_a, idx):
        n = rec[field_n].to_numpy(dtype=object)[idx]
        a = rec[field_a].to_numpy(dtype=object)[idx]
        return [f"{x} | {y}" if y else x for x, y in zip(n, a)]

    all_idx = np.unique(np.r_[qa, pa, ra, rb])
    e_raw = emb(texts_for("name", "address", all_idx))
    e_norm = emb(texts_for("name_norm", "addr_norm", all_idx))
    loc = {v: i for i, v in enumerate(all_idx)}
    g = np.vectorize(loc.get)

    for label, E in (("RAW (native script)", e_raw), ("NORMALIZED (romanized+accent-stripped)", e_norm)):
        E = E.astype(np.float32)
        cos_true = (E[g(qa)] * E[g(pa)]).sum(1)
        cos_rand = (E[g(ra)] * E[g(rb)]).sum(1)
        d, mt, mr = sep(cos_true, cos_rand)
        print(f"{country:6s} | {label:42s} | n_pairs={len(qa):4d} | median_true={mt:.4f} "
             f"median_random={mr:.4f} | SEPARATION={d:.4f}", flush=True)
    print(flush=True)

del rec, truth

# ---------------------------------------------------------------- France: no ground truth, proxy check
tst = load_records(cfg, "test", ["entity_id", "country", "name", "address", "name_norm", "addr_norm"])
fr = tst[tst["country"] == "France"].sample(min(N_PAIRS, (tst["country"] == "France").sum()), random_state=0)
raw_txt = [f"{n} | {a}" if a else n for n, a in zip(fr["name"], fr["address"])]
norm_txt = [f"{n} | {a}" if a else n for n, a in zip(fr["name_norm"], fr["addr_norm"])]
e_raw = emb(raw_txt).astype(np.float32)
e_norm = emb(norm_txt).astype(np.float32)
self_cos = (e_raw * e_norm).sum(1)
print(f"France | self-similarity RAW-vs-NORMALIZED text, same record | n={len(fr):,} "
     f"| mean={self_cos.mean():.4f} median={np.median(self_cos):.4f} p10={np.quantile(self_cos, .1):.4f} "
     f"frac<0.95={np.mean(self_cos < 0.95):.4f}", flush=True)
print("(near 1.0 = bge-m3 already treats accented/unaccented forms almost identically;", flush=True)
print(" lower = normalization meaningfully shifts the embedding, worth testing further)", flush=True)
print("DONE", flush=True)
