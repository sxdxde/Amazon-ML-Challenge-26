"""Gradient-boosting backend: LightGBM (MIT) or XGBoost (Apache-2.0) behind one interface.

LightGBM's Linux x86_64 wheels need glibc >= 2.28; on older systems (e.g. CentOS 7, glibc 2.17)
XGBoost's manylinux2014 wheel is used instead. `backend="auto"` picks LightGBM if importable.
Both grow leaf-wise trees with the same leaf budget / learning rate / sampling settings.
"""
from __future__ import annotations

import numpy as np


def resolve(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except ImportError:
        import xgboost  # noqa: F401
        return "xgboost"


class Model:
    """Uniform wrapper: predict(X, num_threads), best_iteration, feature_importance('gain'), save_model."""

    def __init__(self, kind: str, booster, feats: list[str]):
        self.kind, self.booster, self.feats = kind, booster, feats
        self.best_iteration = int(booster.best_iteration) if kind == "lightgbm" else int(booster.best_iteration) + 1

    def predict(self, X: np.ndarray, num_threads: int = 1) -> np.ndarray:
        if self.kind == "lightgbm":
            return self.booster.predict(X, num_threads=num_threads)
        import xgboost as xgb
        self.booster.set_param({"nthread": num_threads})
        return self.booster.predict(xgb.DMatrix(X, feature_names=self.feats, nthread=num_threads),
                                    iteration_range=(0, self.best_iteration))

    def feature_importance(self, kind: str = "gain") -> np.ndarray:
        if self.kind == "lightgbm":
            return self.booster.feature_importance(kind)
        sc = self.booster.get_score(importance_type="total_gain")
        return np.array([sc.get(f, 0.0) for f in self.feats])

    def save_model(self, path: str) -> None:
        self.booster.save_model(path if self.kind == "lightgbm" else path.replace(".txt", ".json"))


def train(backend: str, params: dict, Xf: np.ndarray, yf: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
          feats: list[str], rounds: int, early_stop: int, n_jobs: int, seed: int) -> Model:
    kind = resolve(backend)
    if kind == "lightgbm":
        import lightgbm as lgb
        p = {**params, "seed": seed, "num_threads": n_jobs, "deterministic": True, "force_col_wise": True}
        dtr = lgb.Dataset(Xf, yf, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)])
        return Model(kind, b, feats)
    import xgboost as xgb
    p = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
         "grow_policy": "lossguide", "max_depth": 0,
         "max_leaves": params.get("num_leaves", 255), "eta": params.get("learning_rate", 0.1),
         "subsample": params.get("bagging_fraction", 0.8), "colsample_bytree": params.get("feature_fraction", 0.8),
         "lambda": params.get("lambda_l2", 1.0), "max_bin": params.get("max_bin", 255),
         "min_child_weight": max(1.0, params.get("min_data_in_leaf", 100) / 100.0),
         "nthread": n_jobs, "seed": seed, "verbosity": 0}
    dtr = xgb.DMatrix(Xf, yf, feature_names=feats, nthread=n_jobs)
    dva = xgb.DMatrix(Xv, yv, feature_names=feats, nthread=n_jobs)
    b = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "valid")], early_stopping_rounds=early_stop,
                  verbose_eval=False)
    return Model(kind, b, feats)
