"""Pair classifiers: LightGBM on scalar features (CPU) and a Siamese-tower MLP in MLX (fp32),
with a PyTorch-MPS fallback of the same MLP."""
import time
from functools import partial

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_pr(y, p):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


# ================================================================ LightGBM
def lgb_params(seed, n_jobs, smoke):
    return {"objective": "binary", "learning_rate": 0.1 if smoke else 0.05, "num_leaves": 31 if smoke else 127,
            "min_data_in_leaf": 20 if smoke else 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": n_jobs, "seed": seed,
            "deterministic": True, "force_row_wise": True, "verbose": -1}


def lgb_cv(X, y, folds, params, rounds, es_rounds, log=print):
    import lightgbm as lgb
    full = lgb.Dataset(X, y, free_raw_data=False, params={"max_bin": 255, "verbose": -1}).construct()
    oof = np.zeros(len(y), np.float32)
    models, imps = [], []
    for f in np.unique(folds):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        t0 = time.time()
        m = lgb.train(params, full.subset(tr), rounds, valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        auc, pr = auc_pr(y[va], oof[va])
        log(f"    fold {f}: best_iter={m.best_iteration} auc={auc:.5f} pr_auc={pr:.5f} ({time.time() - t0:,.0f}s)")
        models.append(m)
        imps.append(m.feature_importance("gain"))
    return oof, models, np.mean(imps, 0)


def lgb_fit_fixed(X, y, params, rounds):
    import lightgbm as lgb
    return lgb.train(params, lgb.Dataset(X, y, params={"verbose": -1}), rounds)


# ================================================================ MLX Siamese MLP
def _mlx():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    return mx, nn, optim


def build_mlx_model(d_emb, n_scalar, hidden=256, p=0.1):
    mx, nn, _ = _mlx()

    class Tower(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin = nn.Linear(d_in, d_out)
            self.ln = nn.LayerNorm(d_out)
            self.drop = nn.Dropout(p)

        def __call__(self, x):
            return self.drop(self.ln(nn.gelu(self.lin(x))))

    class PairMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.name_tower = Tower(2 * d_emb + 1, hidden)
            self.addr_tower = Tower(2 * d_emb + 1, hidden)
            self.scalar = nn.Linear(n_scalar, 128)
            self.h1 = nn.Linear(2 * hidden + 128, 512)
            self.h2 = nn.Linear(512, 128)
            self.out = nn.Linear(128, 1)
            self.drop = nn.Dropout(p)

        @staticmethod
        def pair(a, b):
            a, b = a.astype(mx.float32), b.astype(mx.float32)
            return mx.concatenate([mx.abs(a - b), a * b, (a * b).sum(axis=-1, keepdims=True)], axis=-1)

        def __call__(self, an, bn, aa, ba, x):
            h = mx.concatenate([self.name_tower(self.pair(an, bn)), self.addr_tower(self.pair(aa, ba)),
                                nn.gelu(self.scalar(x))], axis=-1)
            h = self.drop(nn.gelu(self.h1(h)))
            h = nn.gelu(self.h2(h))
            return self.out(h).squeeze(-1)

    return PairMLP()


class Scaler:
    def fit(self, X):
        self.mu = np.nanmean(X, 0).astype(np.float32)
        self.sd = (np.nanstd(X, 0) + 1e-6).astype(np.float32)
        self.mu = np.nan_to_num(self.mu)
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)


def mlx_predict(model, EN, EA, q, t, Z, bs=16384):
    """EN/EA: mx.array tables (train) or numpy memmaps (test); q,t: global ids; Z: scaled scalars."""
    mx, nn, _ = _mlx()
    model.eval()
    out = np.empty(len(q), np.float32)
    is_mx = isinstance(EN, mx.array)
    for s in range(0, len(q), bs):
        qs, ts = q[s:s + bs], t[s:s + bs]
        if is_mx:
            qi, ti = mx.array(qs.astype(np.int32)), mx.array(ts.astype(np.int32))
            args = (mx.take(EN, qi, axis=0), mx.take(EN, ti, axis=0), mx.take(EA, qi, axis=0), mx.take(EA, ti, axis=0))
        else:
            args = tuple(mx.array(np.asarray(E[i])) for E, i in ((EN, qs), (EN, ts), (EA, qs), (EA, ts)))
        p = mx.sigmoid(model(*args, mx.array(Z[s:s + bs])))
        mx.eval(p)
        out[s:s + bs] = np.array(p)
    model.train()
    return out


def mlx_train_fold(EN, EA, q_tr, t_tr, Z_tr, y_tr, q_va, t_va, Z_va, y_va, *, epochs, bs=1024, lr=1e-3,
                   wd=1e-4, pos_weight=1.0, patience=2, seed=42, log=print):
    mx, nn, optim = _mlx()
    from mlx.utils import tree_flatten
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_mlx_model(EN.shape[1], Z_tr.shape[1])
    mx.eval(model.parameters())
    steps = max(1, len(y_tr) // bs)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps * epochs), weight_decay=wd)
    opt.init(model.trainable_parameters())  # materialise optimizer state before it is captured by mx.compile
    pw = float(pos_weight)

    def loss_fn(model, EN, EA, qi, ti, x, y):
        z = model(mx.take(EN, qi, axis=0), mx.take(EN, ti, axis=0),
                  mx.take(EA, qi, axis=0), mx.take(EA, ti, axis=0), x)
        zero = mx.zeros_like(z)
        # BCE with logits and positive weight: pw*y*softplus(-z) + (1-y)*softplus(z)
        return (pw * y * mx.logaddexp(-z, zero) + (1 - y) * mx.logaddexp(z, zero)).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(EN, EA, qi, ti, x, y):
        loss, grads = loss_and_grad(model, EN, EA, qi, ti, x, y)
        opt.update(model, grads)
        return loss

    Q, T = mx.array(q_tr.astype(np.int32)), mx.array(t_tr.astype(np.int32))
    Zm, Ym = mx.array(Z_tr), mx.array(y_tr.astype(np.float32))
    best, best_pr, bad, hist = None, -1.0, 0, []
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        perm = mx.array(rng.permutation(len(y_tr))[: steps * bs].astype(np.int32))
        tot = 0.0
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            loss = step(EN, EA, mx.take(Q, b), mx.take(T, b), mx.take(Zm, b, axis=0), mx.take(Ym, b))
            mx.eval(state)
            if s % 200 == 0:
                tot += loss.item()
        p_va = mlx_predict(model, EN, EA, q_va, t_va, Z_va)
        auc, pr = auc_pr(y_va, p_va)
        dt = time.time() - t0
        hist.append({"epoch": ep, "pr_auc": pr, "auc": auc, "sec": dt})
        log(f"      epoch {ep}: val auc={auc:.5f} pr_auc={pr:.5f}  {dt:,.1f}s/epoch")
        if pr > best_pr + 1e-5:
            best_pr, bad, best = pr, 0, tree_flatten(model.parameters())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_weights(best)
    mx.eval(model.parameters())
    return model, hist


# ================================================================ PyTorch MPS fallback (same model)
def torch_train_fold(EN, EA, q_tr, t_tr, Z_tr, y_tr, q_va, t_va, Z_va, y_va, *, epochs, bs=1024, lr=1e-3,
                     wd=1e-4, pos_weight=1.0, patience=2, seed=42, log=print):
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import torch.nn as tnn
    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = EN.shape[1]

    class Tower(tnn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.f = tnn.Sequential(tnn.Linear(din, dout), tnn.GELU(), tnn.LayerNorm(dout), tnn.Dropout(0.1))

        def forward(self, x):
            return self.f(x)

    class PairMLP(tnn.Module):
        def __init__(self):
            super().__init__()
            self.nt, self.at = Tower(2 * d + 1, 256), Tower(2 * d + 1, 256)
            self.sc = tnn.Linear(Z_tr.shape[1], 128)
            self.head = tnn.Sequential(tnn.Linear(640, 512), tnn.GELU(), tnn.Dropout(0.1),
                                       tnn.Linear(512, 128), tnn.GELU(), tnn.Linear(128, 1))

        @staticmethod
        def pair(a, b):
            a, b = a.float(), b.float()
            return torch.cat([(a - b).abs(), a * b, (a * b).sum(-1, keepdim=True)], -1)

        def forward(self, an, bn, aa, ba, x):
            h = torch.cat([self.nt(self.pair(an, bn)), self.at(self.pair(aa, ba)), tnn.functional.gelu(self.sc(x))], -1)
            return self.head(h).squeeze(-1)

    ENt, EAt = torch.from_numpy(np.asarray(EN)).to(dev), torch.from_numpy(np.asarray(EA)).to(dev)
    model = PairMLP().to(dev)
    steps = max(1, len(y_tr) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps * epochs)
    lossf = tnn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=dev))
    Q, T = torch.from_numpy(q_tr.astype(np.int64)).to(dev), torch.from_numpy(t_tr.astype(np.int64)).to(dev)
    Zt, Yt = torch.from_numpy(Z_tr).to(dev), torch.from_numpy(y_tr.astype(np.float32)).to(dev)
    rng = np.random.default_rng(seed)

    def predict(q, t, Z, bs_=16384):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(q), bs_):
                qi = torch.from_numpy(q[s:s + bs_].astype(np.int64)).to(dev)
                ti = torch.from_numpy(t[s:s + bs_].astype(np.int64)).to(dev)
                z = model(ENt[qi], ENt[ti], EAt[qi], EAt[ti], torch.from_numpy(Z[s:s + bs_]).to(dev))
                out.append(torch.sigmoid(z).float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    best, best_pr, bad, hist = None, -1.0, 0, []
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(len(y_tr))[: steps * bs]).to(dev)
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            qi, ti = Q[b], T[b]
            z = model(ENt[qi], ENt[ti], EAt[qi], EAt[ti], Zt[b])
            loss = lossf(z, Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        auc, pr = auc_pr(y_va, predict(q_va, t_va, Z_va))
        dt = time.time() - t0
        hist.append({"epoch": ep, "pr_auc": pr, "auc": auc, "sec": dt})
        log(f"      epoch {ep}: val auc={auc:.5f} pr_auc={pr:.5f}  {dt:,.1f}s/epoch (torch {dev})")
        if pr > best_pr + 1e-5:
            best_pr, bad = pr, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    model.predict_np = predict
    return model, hist


def torch_predict_test(model, EN_mm, EA_mm, q, t, Z, bs=16384):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = np.empty(len(q), np.float32)
    with torch.no_grad():
        for s in range(0, len(q), bs):
            g = lambda E, i: torch.from_numpy(np.asarray(E[i])).to(dev)
            qs, ts = q[s:s + bs], t[s:s + bs]
            z = model(g(EN_mm, qs), g(EN_mm, ts), g(EA_mm, qs), g(EA_mm, ts), torch.from_numpy(Z[s:s + bs]).to(dev))
            out[s:s + bs] = torch.sigmoid(z).float().cpu().numpy()
    return out
