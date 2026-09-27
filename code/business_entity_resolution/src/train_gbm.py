"""LightGBM matchers.

Stage 1 (`gbm`):  binary classifier on candidate-pair features. K-fold grouped by S1 (hash folds);
                  each fold model is FIT on a subsample (`gbm_fit_s1_frac`) of in-fold S1 groups, with
                  early stopping on a disjoint 10% of in-fold groups, and PREDICTS every held-out pair,
                  so OOF covers every featurised S1. Test = mean of fold models. Plus leave-one-country-out.
Stage 2 (`gbm2`): stacked GBM on stage-1 features + stage-1 OOF probability + probability-context
                  features (rank/gap/competition within the S1 and within the record) + cross-encoder
                  score (NaN outside the cascade band) when available. Same folds.
Rows are kept in feature-part order everywhere, so OOF/test arrays align positionally with
features.load_features / iter_feature_parts.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import gbm_backend
from config import Config
from decide import assign_exclusive, evaluate, fit_isotonic, select_expected_f
from features import _gap, _rank_desc, feature_columns, iter_feature_parts, load_features
from io_utils import LOG, Timer, is_done, lib_versions, mark_done, save_parquet
from prepare import load_records, load_truth
from prune import keep_mask
from split import s1_folds, s1_subsample


# ------------------------------------------------------------------ data helpers
def truth_keys(cfg: Config, rec: pd.DataFrame) -> np.ndarray:
    truth = load_truth(cfg)
    uid = pd.Series(rec.index.values, index=rec["entity_id"].values)
    return uid.reindex(truth["s1"]).values.astype(np.int64) * (1 << 32) + uid.reindex(truth["id"]).values


def label(q: np.ndarray, p: np.ndarray, tkeys: np.ndarray) -> np.ndarray:
    return np.isin(q.astype(np.int64) * (1 << 32) + p, tkeys).astype(np.int8)


def fold_of(cfg: Config, rec: pd.DataFrame, q: np.ndarray) -> np.ndarray:
    s1 = rec.loc[rec["src"].values == 1, "entity_id"]
    f = s1_folds(s1, cfg.n_folds, cfg.seed)
    return f.reindex(rec["entity_id"].values[q]).values.astype(np.int8)


def eval_s1(cfg: Config, rec: pd.DataFrame) -> np.ndarray:
    """Every train S1 whose pairs were featurised (all, or the feat_s1_frac subset)."""
    s1 = rec.loc[rec["src"].values == 1]
    if cfg.feat_s1_frac < 1.0:
        keep = s1_subsample(s1["entity_id"], cfg.feat_s1_frac, cfg.seed, "feat")
        s1 = s1[s1["entity_id"].isin(keep)]
    return s1.index.values


def _seed(cfg: Config, salt: str) -> int:
    return zlib.crc32(f"{cfg.seed}|{salt}".encode())


# ------------------------------------------------------------------ training
def fit_one(cfg: Config, X: np.ndarray, y: np.ndarray, groups: np.ndarray, feats: list[str], salt: str):
    """Fit on a group subsample with early stopping on a disjoint 10% of groups."""
    ug = np.unique(groups)
    rng = np.random.default_rng(_seed(cfg, salt))
    rng.shuffle(ug)
    n_val = max(1, int(0.1 * len(ug)))
    fit_pool = ug[n_val:]
    n_fit = max(1, int(cfg.gbm_fit_s1_frac * len(fit_pool)))
    mf, mv = np.isin(groups, fit_pool[:n_fit]), np.isin(groups, ug[:n_val])
    return gbm_backend.train(cfg.gbm_backend, cfg.gbm_params, X[mf], y[mf], X[mv], y[mv], feats,
                             cfg.gbm_rounds, cfg.gbm_early_stop, cfg.n_jobs, _seed(cfg, salt) % 100000)


def pair_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}


def cv_train(cfg: Config, X: np.ndarray, meta: pd.DataFrame, feats: list[str], tag: str, extra=None):
    """K-fold OOF + fold models. meta has q, y, fold aligned with X rows.
    extra = (X_e, y_e, groups_e): rows added to EVERY fold's training data (pseudo-labelled test pairs);
    they are never in any validation fold, so the OOF stays out-of-fold w.r.t. the train labels."""
    oof = np.zeros(len(X), np.float32)
    models, imps, stats = [], [], []
    y, fold, q = meta["y"].values, meta["fold"].values, meta["q"].values
    for f in range(cfg.n_folds):
        tr, va = fold != f, fold == f
        with Timer(f"{tag} fold {f}: in-fold rows {tr.sum():,}"):
            if extra is None:
                m = fit_one(cfg, X[tr], y[tr], q[tr], feats, f"{tag}{f}")
            else:
                Xe, ye, ge = extra
                m = fit_one(cfg, np.concatenate([X[tr], Xe]), np.concatenate([y[tr], ye]),
                            np.concatenate([q[tr], ge]), feats, f"{tag}{f}")
            oof[va] = m.predict(X[va], num_threads=cfg.n_jobs)
        st = {"fold": f, "best_iter": m.best_iteration, **pair_metrics(y[va], oof[va])}
        LOG.info("%s fold %d: %s", tag, f, {k: round(v, 5) if isinstance(v, float) else v for k, v in st.items()})
        stats.append(st)
        models.append(m)
        imps.append(pd.Series(m.feature_importance("gain"), index=feats))
    return oof, models, pd.concat(imps, axis=1).mean(1).sort_values(ascending=False), stats


def predict_models(models, X: np.ndarray, n_jobs: int) -> np.ndarray:
    return np.mean([m.predict(X, num_threads=n_jobs) for m in models], axis=0).astype(np.float32)


def per_country_metrics(rec: pd.DataFrame, meta: pd.DataFrame, score: np.ndarray) -> dict:
    ctry = rec["country"].values[meta["q"].values]
    return {c: pair_metrics(meta["y"].values[ctry == c], score[ctry == c]) for c in pd.unique(ctry)}


# ------------------------------------------------------------------ LOCO
def loco(cfg: Config, X: np.ndarray, meta: pd.DataFrame, feats: list[str], rec: pd.DataFrame,
         tk: np.ndarray, s1_all: np.ndarray) -> list[dict]:
    """Train on one country string, evaluate on another (proxy for the unseen test country)."""
    country = rec["country"].values
    mc = country[meta["q"].values]
    truth = pd.DataFrame({"q": tk // (1 << 32), "p": tk % (1 << 32)})
    out = []
    for a in sorted(pd.unique(mc)):
        for b in sorted(pd.unique(mc)):
            if a == b:
                continue
            ia, ib = np.flatnonzero(mc == a), np.flatnonzero(mc == b)
            qa = meta["q"].values[ia]
            ua = np.unique(qa)
            rng = np.random.default_rng(_seed(cfg, f"loco{a}"))
            cal = np.isin(qa, rng.choice(ua, max(1, int(0.2 * len(ua))), replace=False))
            with Timer(f"LOCO {a} -> {b}"):
                m = fit_one(cfg, X[ia[~cal]], meta["y"].values[ia[~cal]], qa[~cal], feats, f"loco{a}")
                iso = fit_isotonic(m.predict(X[ia[cal]], num_threads=cfg.n_jobs), meta["y"].values[ia[cal]])
                pb = m.predict(X[ib], num_threads=cfg.n_jobs)
            d = meta.iloc[ib][["q", "p"]].assign(pc=iso.predict(pb))
            pred = select_expected_f(assign_exclusive(d, "pc"), "pc", cfg.mc_max_cands, cfg.decide_chunk)
            sb = s1_all[country[s1_all] == b]
            ev = evaluate(pred, truth[truth["q"].isin(set(sb))], sb, None)["all"]
            pm = pair_metrics(meta["y"].values[ib], pb)
            row = {"train": a, "eval": b, "macro_F0.5": ev["f05"], "singleton_acc": ev["singleton_acc"],
                   "nonsingleton_F0.5": ev["nonsingleton_f05"], "pair_auc": pm.get("auc"),
                   "pair_pr_auc": pm.get("pr_auc")}
            LOG.info("LOCO %s", row)
            out.append(row)
    return out


# ------------------------------------------------------------------ stage 1
def load_train_matrix(cfg: Config, rec: pd.DataFrame, tk: np.ndarray):
    with Timer("load train features"):
        df = load_features(cfg, "train")
    feats = feature_columns(df)
    meta = df[["q", "p"]].copy()
    X = df[feats].to_numpy(np.float32)
    del df
    meta["y"] = label(meta["q"].values, meta["p"].values, tk)
    meta["fold"] = fold_of(cfg, rec, meta["q"].values)
    LOG.info("train pairs %s, positives %s, features %d", f"{len(meta):,}", f"{int(meta['y'].sum()):,}", len(feats))
    return X, meta, feats


def run_stage1(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "gbm"
    if is_done(out) and not force:
        LOG.info("gbm: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    rec = load_records(cfg, "train", ["entity_id", "src", "country"])
    tk = truth_keys(cfg, rec)
    X, meta, feats = load_train_matrix(cfg, rec, tk)
    LOG.info("gbm backend: %s", gbm_backend.resolve(cfg.gbm_backend))
    oof, models, imp, stats = cv_train(cfg, X, meta, feats, "gbm")
    save_parquet(meta.assign(p1=oof), out / "oof.parquet")
    s1_all = eval_s1(cfg, rec)
    np.save(out / "eval_s1.npy", s1_all)
    for i, m in enumerate(models):
        m.save_model(str(out / f"fold{i}.txt"))
    imp.to_csv(out / "importance.csv", header=["gain"])
    (out / "features.json").write_text(json.dumps(feats))
    loco_rows = []
    if cfg.loco and rec.loc[rec["src"] == 1, "country"].nunique() > 1:
        loco_rows = loco(cfg, X, meta, feats, rec, tk, s1_all)
        (out / "loco.json").write_text(json.dumps(loco_rows, indent=1))
    metrics = {"oof_overall": pair_metrics(meta["y"].values, oof),
               "oof_per_country": per_country_metrics(rec, meta, oof), "folds": stats, "loco": loco_rows}
    del X
    with Timer("predict test (stage 1)"):
        parts = []
        for _, df in iter_feature_parts(cfg, "test"):
            parts.append(df[["q", "p"]].assign(p1=predict_models(models, df[feats].to_numpy(np.float32), cfg.n_jobs)))
        save_parquet(pd.concat(parts, ignore_index=True), out / "test_pred.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_report(cfg, "gbm", metrics, imp)
    mark_done(out, {"n_features": len(feats), "versions": lib_versions(),
                    **{k: metrics["oof_overall"].get(k) for k in ("auc", "pr_auc")}})


# ------------------------------------------------------------------ stage 2
def prob_context(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Context features from a pair-level probability, per S1 (q) and per record (p)."""
    o = pd.DataFrame(index=d.index)
    o[f"{col}_rank_q"] = _rank_desc(d, "q", col)
    o[f"{col}_rank_p"] = _rank_desc(d, "p", col)
    o[f"{col}_gap_q"] = _gap(d, "q", col)
    o[f"{col}_gap_p"] = _gap(d, "p", col)
    o[f"{col}_sum_q"] = d.groupby("q")[col].transform("sum").values.astype(np.float32)
    o[f"{col}_sum_p"] = d.groupby("p")[col].transform("sum").values.astype(np.float32)
    o[f"{col}_n05_q"] = d.assign(h=d[col] > 0.5).groupby("q")["h"].transform("sum").values.astype(np.float32)
    o[f"{col}_share_p"] = (d[col] / o[f"{col}_sum_p"].clip(lower=1e-6)).values.astype(np.float32)
    return o


def stage2_extra(qp: pd.DataFrame, xenc: pd.DataFrame | None) -> pd.DataFrame:
    """Extra stage-2 columns for rows qp[q, p, p1] (kept in the same row order)."""
    ex = pd.concat([qp[["p1"]], prob_context(qp[["q", "p", "p1"]], "p1")], axis=1)
    if xenc is not None:
        x = qp[["q", "p"]].merge(xenc[["q", "p", "xenc"]], on=["q", "p"], how="left")["xenc"].values
        ex["xenc"] = x.astype(np.float32)
        has = ~np.isnan(x)
        sub = qp.loc[has, ["q", "p"]].assign(xenc=x[has])
        for name, fn in [("xenc_rank_q", _rank_desc), ("xenc_gap_q", _gap)]:
            v = np.full(len(qp), np.nan, np.float32)
            v[has] = fn(sub, "q", "xenc")
            ex[name] = v
    return ex.reset_index(drop=True)


def run_stage2(cfg: Config, force: bool = False) -> None:
    out = cfg.art / "gbm2"
    if is_done(out) and not force:
        LOG.info("gbm2: cached, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    rec = load_records(cfg, "train", ["entity_id", "src", "country"])
    tk = truth_keys(cfg, rec)
    xdir = cfg.art / "xenc"
    use_x = cfg.use_xenc and (xdir / "_DONE.json").exists()
    oof1 = pd.read_parquet(cfg.art / "gbm" / "oof.parquet")
    X1, meta, feats1 = load_train_matrix(cfg, rec, tk)
    assert (meta["q"].values == oof1["q"].values).all() and (meta["p"].values == oof1["p"].values).all()
    ex = stage2_extra(oof1[["q", "p", "p1"]], pd.read_parquet(xdir / "oof.parquet") if use_x else None)
    feats = feats1 + list(ex.columns)
    X = np.hstack([X1, ex.to_numpy(np.float32)])
    del X1
    LOG.info("stage 2: %d features (xenc=%s)", len(feats), use_x)

    # test matrix (all stage-1 candidates; needed for self-training and for the final inference)
    with Timer("build test matrix (stage 2)"):
        te1 = pd.read_parquet(cfg.art / "gbm" / "test_pred.parquet")
        ex_t = stage2_extra(te1[["q", "p", "p1"]], pd.read_parquet(xdir / "test_pred.parquet") if use_x else None)
        parts = []
        for off, df in iter_feature_parts(cfg, "test"):
            n = len(df)
            assert (df["q"].values == te1["q"].values[off:off + n]).all()
            parts.append(np.hstack([df[feats1].to_numpy(np.float32), ex_t.iloc[off:off + n].to_numpy(np.float32)]))
        Xt = np.vstack(parts)
        del parts, ex_t

    oof, models, imp, stats = cv_train(cfg, X, meta, feats, "gbm2")

    # ---- self-training for countries that never appear in train (open set, derived from the data)
    te_ctry = load_records(cfg, "test", ["country"])["country"].values[te1["q"].values]
    unseen = sorted(set(pd.unique(te_ctry)) - set(pd.unique(rec["country"].values)))
    pseudo = {"enabled": bool(cfg.pseudo_hi > 0 and unseen), "unseen_countries": unseen}
    if pseudo["enabled"]:
        with Timer("self-training: pseudo-label unseen-country test pairs"):
            pc = fit_isotonic(oof, meta["y"].values).predict(predict_models(models, Xt, cfg.n_jobs))
            conf = np.isin(te_ctry, unseen) & ((pc >= cfg.pseudo_hi) | (pc <= cfg.pseudo_lo))
            ips = np.flatnonzero(conf)
            yps = (pc[ips] >= cfg.pseudo_hi).astype(np.float32)
            gps = te1["q"].values[ips].astype(np.int64) + 10 ** 9      # test S1 groups, disjoint from train uids
            pseudo.update(n_rows=int(len(ips)), n_pos=int(yps.sum()), hi=cfg.pseudo_hi, lo=cfg.pseudo_lo,
                          frac_of_unseen=float(conf.sum() / max(np.isin(te_ctry, unseen).sum(), 1)))
            LOG.info("self-training on %s: %s pseudo-labelled pairs (%s positive, %.1f%% of their candidates)",
                     unseen, f"{len(ips):,}", f"{int(yps.sum()):,}", pseudo["frac_of_unseen"] * 100)
        oof, models, imp, stats = cv_train(cfg, X, meta, feats, "gbm2", extra=(Xt[ips], yps, gps))
    save_parquet(meta.assign(p2=oof), out / "oof.parquet")
    for i, m in enumerate(models):
        m.save_model(str(out / f"fold{i}.txt"))
    imp.to_csv(out / "importance.csv", header=["gain"])
    stand = {}
    if use_x:
        has = ~np.isnan(ex["xenc"].values)
        y = meta["y"].values
        stand = {"xenc_standalone_on_band": pair_metrics(y[has], ex["xenc"].values[has]),
                 "p1_on_band": pair_metrics(y[has], ex["p1"].values[has]),
                 "p2_on_band": pair_metrics(y[has], oof[has])}
        LOG.info("cross-encoder standalone vs stacked on band: %s", stand)
    metrics = {"oof_overall": pair_metrics(meta["y"].values, oof),
               "oof_per_country": per_country_metrics(rec, meta, oof), "folds": stats, "xenc": stand,
               "use_xenc": use_x, "self_training": pseudo}
    del X
    with Timer("predict test (stage 2) on the pruned candidate set"):
        keep = keep_mask(te1, cfg)                         # cascade: only these reach the final matcher
        preds = predict_models(models, Xt[keep], cfg.n_jobs)
        save_parquet(te1.loc[keep, ["q", "p"]].reset_index(drop=True).assign(p2=preds), out / "test_pred.parquet")
        LOG.info("final matcher scored %s of %s stage-1 candidates (%.2f per S1)", f"{int(keep.sum()):,}",
                 f"{len(te1):,}", keep.sum() / te1["q"].nunique())
    del Xt
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_report(cfg, "gbm2", metrics, imp)
    mark_done(out, {"n_features": len(feats), "use_xenc": use_x, "versions": lib_versions(),
                    **{k: metrics["oof_overall"].get(k) for k in ("auc", "pr_auc")}})


def write_report(cfg: Config, tag: str, metrics: dict, imp: pd.Series) -> None:
    cfg.reports.mkdir(parents=True, exist_ok=True)
    fmt = {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics["oof_overall"].items()}
    lines = [f"# {tag}: LightGBM pair classifier", "", f"OOF overall: `{json.dumps(fmt)}`", "",
             "## OOF per country", "", pd.DataFrame(metrics["oof_per_country"]).T.round(5).to_markdown(), "",
             "## Folds", "", pd.DataFrame(metrics["folds"]).round(5).to_markdown(index=False), ""]
    if metrics.get("loco"):
        lines += ["## Leave-one-country-out (train on one country, evaluate on the other)", "",
                  pd.DataFrame(metrics["loco"]).round(5).to_markdown(index=False), ""]
    if metrics.get("xenc"):
        lines += ["## Cross-encoder on the cascade band (pair level)", "",
                  pd.DataFrame(metrics["xenc"]).T.round(5).to_markdown(), ""]
    lines += ["## Top-40 features (mean gain over folds)", "", imp.head(40).round(1).to_frame("gain").to_markdown(), ""]
    (cfg.reports / f"{tag}.md").write_text("\n".join(lines))
