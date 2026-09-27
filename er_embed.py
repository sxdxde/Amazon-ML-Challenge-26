"""Frozen text embeddings on Apple silicon.

Primary path: bge-m3 weights from `mlx-community/bge-m3-mlx-fp16`, run by the pure-MLX
XLM-R encoder in er_mlx_encoder.py (fp16, CLS pooling + L2 norm = BAAI/bge-m3 dense vector).
Fallback (USE_ST_MPS=True): sentence-transformers BAAI/bge-m3 on torch MPS in fp16.

Vectors are PCA-projected to EMB_DIM (fit on a train sample, centred, re-L2-normalised) so the
~12M test records fit in unified memory for dense search; stored as float16 .npy memmaps,
written chunk-by-chunk with a resume marker so an interrupted run continues where it stopped.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np

from er_common import check, paste


class Embedder:
    def __init__(self, repo, pooling="cls", max_len=96, prefix="", use_st_mps=False, st_ref="BAAI/bge-m3"):
        self.pooling, self.max_len, self.prefix, self.use_st = pooling, max_len, prefix, use_st_mps
        if use_st_mps:
            import torch
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(st_ref, device="mps")
            self.st.half()
            self.st.max_seq_length = max_len
            self.dim = self.st.get_sentence_embedding_dimension()
        else:
            import mlx.core as mx
            from er_mlx_encoder import load_encoder
            self.model, self.tok, cfg = load_encoder(repo, dtype=mx.float16)
            self.dim = cfg["hidden_size"]

    def embed(self, texts, batch_size=64):
        """-> float16 [n, dim], L2-normalised, original order (sorted by length internally)."""
        texts = [self.prefix + t for t in texts] if self.prefix else list(texts)
        if self.use_st:
            v = self.st.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
            return v.astype(np.float16)
        from er_mlx_encoder import encode
        return encode(self.model, self.tok, texts, batch_size=batch_size,
                      max_len=self.max_len, pooling=self.pooling)

    def close(self):
        for a in ("model", "tok", "st"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass


# ---------------------------------------------------------------- PCA projection
def fit_pca(vecs, dim):
    x = vecs.astype(np.float32)
    mu = x.mean(0)
    x -= mu
    cov = (x.T @ x) / len(x)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:dim]
    explained = float(w[order].sum() / w.sum())
    return mu.astype(np.float32), v[:, order].astype(np.float32), explained


def project(vecs, mu, W):
    if W is None:
        return vecs.astype(np.float16)
    y = (vecs.astype(np.float32) - mu) @ W
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y.astype(np.float16)


def pca_fidelity(full, mu, W, rng, n_pairs=20000):
    """Correlation of cosines before/after projection on random pairs and on nearest-neighbour pairs."""
    f = full.astype(np.float32)
    p = project(full, mu, W).astype(np.float32)
    n = len(f)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    rnd_full, rnd_proj = (f[i] * f[j]).sum(1), (p[i] * p[j]).sum(1)
    m = min(n, 3000)
    s = f[:m] @ f[:m].T
    np.fill_diagonal(s, -9)
    nn = s.argmax(1)
    nn_full = s[np.arange(m), nn]
    nn_proj = (p[:m] * p[nn]).sum(1)
    top_keep = float(((p[:m] @ p[:m].T - np.eye(m) * 9).argmax(1) == nn).mean())
    return {"rand_corr": float(np.corrcoef(rnd_full, rnd_proj)[0, 1]),
            "nn_corr": float(np.corrcoef(nn_full, nn_proj)[0, 1]),
            "nn_top1_kept": top_keep}


# ---------------------------------------------------------------- streaming to memmap
def embed_to_memmap(emb, texts, path, mu, W, batch_size, chunk=50_000, log=True):
    """Embed `texts` (list[str]) into float16 memmap `path` [N, D]. Resumable."""
    path = Path(path)
    prog = path.with_suffix(".progress")
    done_flag = path.with_suffix(".done")
    D = W.shape[1] if W is not None else emb.dim
    N = len(texts)
    if done_flag.exists():
        return np.load(path, mmap_mode="r")
    order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=N), kind="stable")
    n_chunks = (N + chunk - 1) // chunk
    start = int(prog.read_text()) if (prog.exists() and path.exists()) else 0
    mm = np.lib.format.open_memmap(path, mode="r+" if start else "w+", dtype=np.float16, shape=(N, D))
    t0, n_done = time.time(), 0
    for c in range(start, n_chunks):
        idx = order[c * chunk:(c + 1) * chunk]
        v = emb.embed([texts[i] for i in idx], batch_size=batch_size)
        mm[idx] = project(v, mu, W)
        mm.flush()
        prog.write_text(str(c + 1))
        n_done += len(idx)
        if log and (c % 10 == 0 or c == n_chunks - 1):
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            eta = (N - (c + 1) * chunk) / max(rate, 1e-9)
            print(f"    {path.name}: {min((c + 1) * chunk, N):,}/{N:,}  {rate:,.0f} rec/s  ETA {max(eta, 0) / 60:,.1f} min",
                  flush=True)
    del mm
    done_flag.write_text(json.dumps({"n": N, "dim": D}))
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------- cell 1a / 1b helpers
PARITY_STRINGS = [
    "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "B+ Retail Inc",
    "Christ Chapel | 2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "wilfordhancock.com",
    "LLC Moncada Léarning Center", "5780 Fawn Ct, Fort Worth, Texas", "Anderson, Hoover & Thomas Usa Inc",
    "Foot & Ankle Beacon Physicians Corp", "*** Creative Nutrition Laboratories [LP]", "31 1/2 KILLDEER LANE, FAIRPORT, NY",
    "Maure Hardin Vanguard Octave (L.L.C.) (L.L.C.)", "W617 SPENCER ROAD, PMB 4872, APPLETON, WI",
    "Jain Systems Pvt Ltd", "Plot No. F- 71, Shop No. 4, G/F Dilshad Colony, Shahadara, Delhi",
    "PGS Santoshi Private Limited", "C/O-Bhaktaram Putel, Keogan, Kalahandi, Golamunda, Orissa",
    "राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
    "आदित्य प्रॉपर्टीज एलएलपी", "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh",
    "गोल्डेन हसपिटालिटी Private Limited", "ਸ਼ਕਤੀ ਪ੍ਰੋਡਿਊਸਰ ਮੋਟਰਜ਼ ਲਿਮਟਿਡ", "NO 088 C/OYOGRAJ KHOSLA, DBN ROAD, OPP. SBI, GURDASPUR",
    "Prabhu Hospitality Pvt. Ltd.", "Flat No. A-4 604, Shree Gurudev Palace, Mira Road (E), Thane, Maharashtra",
    "Near SBI ATM, MG Road, Bengaluru", "शिवम कंसल्टेंट्स लिमिटेड", "H.no 910 A 3503, Mumbai, महाराष्ट्र",
    "Mediation Visages Fêtes EURL", "Dunkerque, 3 Rue de la Bienfaisance, Hauts-de-France",
    "Établissements Faire", "7 Avenue de la Reousse Dune l'Herbe, Lège-Cap-Ferret, Nouvelle-Aquitaine",
    "Trois & Frèrês Développement SA", "56 - RTE DES QUATRE VENTS, SAINT-NAZAIRE, Loire-Atlantique",
    "École primaire International SCI", "N° 14 R. DE BOULMGNE, TOURCOING", "Centre Médical Sainte Joseph SASU",
    "1053 GRANDE RUE DU PETIT COURGAIN, CALAIS, Pas-de-Calais", "Pharmacie Dame | 26 Rue Mercière, Bordeaux",
    "SCI Ptit Àmicale", "13 BOULEVARD DE L'OCÉAN, LA BAULE-ESCOUBLAC", "En face de la mairie, Place du Marché, Nantes",
    "Zephay Labs Inc", "IA, Iowa City, 1064 Newton Rd, Unit 11", "Red Perfect Trading",
    "Mirzapur, Ews 12, Uttar Pradesh, Awas Vikas Colony", "a", "12345", " ",
]


def parity_check(repo, st_ref, pooling, max_len):
    import mlx.core as mx
    from sentence_transformers import SentenceTransformer
    from er_mlx_encoder import encode, load_encoder
    model, tok, cfg = load_encoder(repo, dtype=mx.float16)
    ours = encode(model, tok, PARITY_STRINGS, batch_size=16, max_len=max_len, pooling=pooling).astype(np.float32)
    wrong = encode(model, tok, PARITY_STRINGS, batch_size=16, max_len=max_len,
                   pooling="mean" if pooling == "cls" else "cls").astype(np.float32)
    del model, tok
    st = SentenceTransformer(st_ref, device="cpu")
    st.max_seq_length = max_len
    st_pool = str(st[1]) if len(st) > 1 else "?"
    ref = st.encode(PARITY_STRINGS, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
    del st
    gc.collect()
    mx.clear_cache()
    cos = (ours * ref).sum(1)
    cos_wrong = (wrong * ref).sum(1)
    return {"n": len(PARITY_STRINGS), "mean_cos": float(cos.mean()), "min_cos": float(cos.min()),
            "worst": PARITY_STRINGS[int(cos.argmin())], "wrong_pool_mean_cos": float(cos_wrong.mean()),
            "st_pooling": st_pool, "dim": int(ours.shape[1])}


def benchmark(emb, texts_by_field, batch_sizes=(32, 64, 128), n=2048):
    """-> {field: {bs: rec/s}}. Texts are sorted by length inside embed(), like the real run."""
    res = {}
    emb.embed(texts_by_field[next(iter(texts_by_field))][:256], batch_size=64)  # warm-up / kernel build
    for field, texts in texts_by_field.items():
        res[field] = {}
        sample = texts[:n]
        for bs in batch_sizes:
            emb.embed(sample[:bs * 2], batch_size=bs)  # warm this shape bucket
            t0 = time.time()
            emb.embed(sample, batch_size=bs)
            res[field][bs] = len(sample) / (time.time() - t0)
    return res
