"""Throughput of the MLX bge-m3 encoder on real record text, batch 32/64/128, length-sorted."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
import numpy as np, pandas as pd, mlx.core as mx
from er_mlx_encoder import load_encoder, encode

REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/bge-m3-mlx-fp16"
N = 6144
df = pd.read_csv(os.path.join(ROOT, "dataset/test/test_source2.tsv"), sep="\t", dtype=str, keep_default_na=False, nrows=200_000)
df = df.sample(N, random_state=0)
sets = {"name": df.business_name.tolist(), "address": df.business_address.tolist(),
        "name|address": (df.business_name + " | " + df.business_address).tolist()}
model, tok, cfg = load_encoder(REPO)
encode(model, tok, sets["name"][:512], batch_size=64)  # warm-up / kernel compile
res = {}
for field, texts in sets.items():
    for bs in (32, 64, 128):
        t = time.time(); encode(model, tok, texts, batch_size=bs); dt = time.time() - t
        res[(field, bs)] = N / dt
        print(f"{REPO.split('/')[-1]:>24} {field:>13} bs={bs:<4} {N/dt:8.0f} rec/s  peak_mlx={mx.get_peak_memory()/2**30:.2f}GB", flush=True)
best = {f: max((res[(f, b)], b) for b in (32, 64, 128)) for f in sets}
print("best:", {f: f"bs={b} {r:.0f}/s" for f, (r, b) in best.items()})
counts = {"train": 2_206_821 + 5_034_616 + 5_285_603, "test": 1_732_544 + 4_887_273 + 5_082_316}
for split, n in counts.items():
    hrs = {f: n / best[f][0] / 3600 for f in sets}
    print(f"projected {split} ({n:,} records): " + ", ".join(f"{f} {h:.1f}h" for f, h in hrs.items()) + f"  | total {sum(hrs.values()):.1f}h")
