"""Minimal XLM-RoBERTa / BERT text encoder in pure MLX (mlx.nn), MIT-compatible.

Written here because `mlx-embeddings` is GPL-3.0 and therefore not allowed.
Parameter names mirror Hugging Face `XLMRobertaModel` / `BertModel`, so HF-format
safetensors (e.g. mlx-community/bge-m3-mlx-fp16) load with `load_weights` unchanged.

bge-m3 dense embedding = CLS token of the last hidden state, L2-normalised
(NOT mean pooling, despite some MLX model-card examples).
"""
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class _SelfAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)


class _Dense(nn.Module):
    def __init__(self, din, dout, eps=None):
        super().__init__()
        self.dense = nn.Linear(din, dout)
        if eps is not None:
            self.LayerNorm = nn.LayerNorm(dout, eps=eps)


class _Attention(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.self = _SelfAttn(d)
        self.output = _Dense(d, d, eps)


class _Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.n_heads = cfg["num_attention_heads"]
        self.attention = _Attention(d, eps)
        self.intermediate = _Dense(d, cfg["intermediate_size"])
        self.output = _Dense(cfg["intermediate_size"], d, eps)

    def __call__(self, x, bias):
        B, L, D = x.shape
        h, dh = self.n_heads, D // self.n_heads
        a = self.attention.self
        q = a.query(x).reshape(B, L, h, dh).transpose(0, 2, 1, 3)
        k = a.key(x).reshape(B, L, h, dh).transpose(0, 2, 1, 3)
        v = a.value(x).reshape(B, L, h, dh).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=dh ** -0.5, mask=bias)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, D)
        x = self.attention.output.LayerNorm(x + self.attention.output.dense(o))
        y = self.output.dense(nn.gelu(self.intermediate.dense(x)))
        return self.output.LayerNorm(x + y)


class _Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], d)
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], d)
        self.token_type_embeddings = nn.Embedding(cfg.get("type_vocab_size", 1), d)
        self.LayerNorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])


class _Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = [_Layer(cfg) for _ in range(cfg["num_hidden_layers"])]


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embeddings = _Embeddings(cfg)
        self.encoder = _Encoder(cfg)
        # XLM-R positions start at pad_token_id+1 and skip padding; BERT starts at 0.
        self.is_roberta = cfg["model_type"] in ("xlm-roberta", "roberta")
        self.pad_id = cfg.get("pad_token_id", 1)

    def __call__(self, ids, mask):
        """ids, mask: int32 [B, L]. Returns last hidden state [B, L, D]."""
        if self.is_roberta:
            pos = mx.cumsum(mask, axis=1) * mask + self.pad_id
        else:
            pos = mx.broadcast_to(mx.arange(ids.shape[1])[None], ids.shape)
        e = self.embeddings
        x = e.word_embeddings(ids) + e.position_embeddings(pos) + e.token_type_embeddings(mx.zeros_like(ids))
        x = e.LayerNorm(x)
        bias = ((1 - mask.astype(x.dtype)) * -1e4)[:, None, None, :].astype(x.dtype)
        for layer in self.encoder.layer:
            x = layer(x, bias)
        return x


def load_encoder(repo, revision=None, dtype=mx.float16):
    """Download (or reuse cache) an HF-format repo and build the MLX encoder + tokenizer."""
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    path = repo if os.path.isdir(repo) else snapshot_download(
        repo, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    cfg = json.load(open(os.path.join(path, "config.json")))
    model = TextEncoder(cfg)
    files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    weights = {}
    for f in files:
        weights.update(mx.load(os.path.join(path, f)))
    weights = {k.removeprefix("roberta.").removeprefix("bert.").removeprefix("model."): v
               for k, v in weights.items()}
    weights = {k: v.astype(dtype) for k, v in weights.items()
               if not k.startswith("pooler.") and "position_ids" not in k}
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    return model, tok, cfg


def encode(model, tok, texts, batch_size=64, max_len=128, pooling="cls", log_every=0):
    """Embed texts -> float16 numpy [N, D], L2-normalised.

    Texts are sorted by token length so each batch has little padding, then
    returned in the original order. mx.eval per batch keeps the lazy graph small.
    """
    pad_id = model.pad_id if model.is_roberta else 0
    tok.no_padding()
    tok.enable_truncation(max_len)
    enc = tok.encode_batch([t if t else " " for t in texts], add_special_tokens=True)
    ids_list = [e.ids for e in enc]
    lens = np.fromiter((len(x) for x in ids_list), dtype=np.int32, count=len(ids_list))
    order = np.argsort(lens, kind="stable")
    out = np.empty((len(texts), model.cfg["hidden_size"]), dtype=np.float16)
    t0 = time.time()
    for bi, s in enumerate(range(0, len(order), batch_size)):
        idx = order[s:s + batch_size]
        L = int(lens[idx].max())
        L = min(-(-L // 8) * 8, max_len)  # bucket to multiples of 8: fewer distinct shapes -> fewer kernel builds
        arr = np.full((len(idx), L), pad_id, dtype=np.int32)
        m = np.zeros((len(idx), L), dtype=np.int32)
        for r, j in enumerate(idx):
            arr[r, :lens[j]] = ids_list[j]
            m[r, :lens[j]] = 1
        ids_mx, m_mx = mx.array(arr), mx.array(m)
        h = model(ids_mx, m_mx)
        if pooling == "cls":
            v = h[:, 0, :]
        else:
            mf = m_mx[..., None].astype(h.dtype)
            v = (h * mf).sum(axis=1) / mf.sum(axis=1)
        v = v.astype(mx.float32)
        v = v / mx.maximum(mx.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        v = v.astype(mx.float16)
        mx.eval(v)
        out[idx] = np.array(v)
        if log_every and bi % log_every == 0 and bi:
            done = s + len(idx)
            print(f"  {done:,}/{len(texts):,}  {done / (time.time() - t0):,.0f} rec/s", flush=True)
    return out
