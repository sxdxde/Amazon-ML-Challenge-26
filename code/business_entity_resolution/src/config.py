"""Central configuration: paths, model names, k values, seeds, and profile overrides.

Profiles
--------
full  : the real run (remote A100). All train/test S1 are blocked and scored.
dev   : local development slice (a few thousand S1 + their matches + distractors).
smoke : tiny CPU run (<10 min) used by scripts/smoke_test.sh; writes to output_smoke/.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent  # code/business_entity_resolution


@dataclass
class EmbedModel:
    name: str
    pooling: str  # "cls" | "mean"
    prefix: str = ""
    max_len: int = 96
    batch_size: int = 256


@dataclass
class Config:
    profile: str = "full"
    root: Path = Path(".")
    data_dir: Path = Path("dataset")
    seed: int = 42
    device: str = "auto"  # auto -> cuda if available else cpu
    n_jobs: int = max(1, min(48, (os.cpu_count() or 2) - 1))  # capped: shared servers

    # ---------------------------------------------------------------- sampling
    # Train S1 entities kept (None = all). Their true matches are always kept.
    train_s1_n: int | None = None
    # Fraction / count of *other* S2/S3 records (not matching a kept S1) kept as distractors.
    train_other_frac: float = 1.0
    train_other_n: int | None = None
    test_s1_n: int | None = None
    test_other_frac: float = 1.0
    test_other_n: int | None = None
    # Fraction of train S1 groups whose pairs are featurised and used to FIT models
    # (blocking + OOF prediction always cover every train S1 that was featurised).
    feat_s1_frac: float = 1.0  # full profile needs ~100 GB RAM at 1.0; use 0.5 on <128 GB machines

    # ---------------------------------------------------------------- embeddings
    embed: EmbedModel = field(default_factory=lambda: EmbedModel("BAAI/bge-m3", "cls", "", 96, 512))
    embed_fp16: bool = True

    # ---------------------------------------------------------------- blocking
    # Max k retrieved per retriever (k is then tuned on train, <= k_max).
    k_max: dict = field(default_factory=lambda: {
        "dense": 50, "combo_word": 40, "combo_char": 30, "name_char": 30, "addr_char": 20,
        "name_word": 20, "addr_word": 20})
    k_grid: tuple = (0, 2, 5, 10, 15, 20, 30, 40, 50)
    target_recall: float = 0.995
    max_mean_cands: float = 40.0
    cand_cap: int = 80  # hard cap per S1 after union (kept by reciprocal-rank fusion)
    key_block_cap: int = 20  # exact-key retriever: skip blocks larger than this
    # per-retriever max document frequency (fraction); lower = faster sparse top-n, slightly lower recall
    retriever_max_df: dict = field(default_factory=lambda: {
        "combo_word": 0.01, "combo_char": 0.01, "name_char": 0.03, "addr_char": 0.02,
        "name_word": 0.01, "addr_word": 0.01})
    tfidf_min_df: int = 2
    tfidf_max_df_frac: float = 0.02  # drop char n-grams / words present in >2% of a country's docs
    block_chunk: int = 20000  # S1 rows per sparse top-n chunk
    # dense kNN chunking: GPU memory ≈ pool (N×dim fp16) + q_chunk×p_chunk×2 bytes (~14 GB for the
    # largest train country with these values; safe on a shared 40 GB A100 with ~22 GB free)
    dense_q_chunk: int = 1024
    dense_p_chunk: int = 500_000
    dense_pool_on_gpu: bool = False  # False: pool stays in RAM, streamed in p-chunks (~3 GB GPU peak)
    tune_s1_max: int = 150_000  # S1 used for greedy k tuning

    # ---------------------------------------------------------------- GBM
    n_folds: int = 5
    gbm_params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 255, "min_data_in_leaf": 300,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
        "max_bin": 255, "verbose": -1})
    gbm_backend: str = "auto"  # auto = lightgbm if importable, else xgboost (old-glibc Linux)
    gbm_rounds: int = 1500
    gbm_early_stop: int = 50
    gbm_fit_s1_frac: float = 0.3  # fraction of in-fold S1 groups used to FIT each fold model (all are predicted)
    loco: bool = True

    # ---------------------------------------------------------------- cross-encoder
    use_xenc: bool = True
    xenc_model: str = "BAAI/bge-reranker-v2-m3"
    xenc_folds: int = 3
    xenc_max_len: int = 128
    xenc_lr: float = 2e-5
    xenc_epochs: int = 1
    xenc_batch: int = 24
    xenc_grad_accum: int = 3  # effective batch 72; lower peak GPU memory on a shared A100
    xenc_infer_batch: int = 384
    xenc_train_pairs_per_fold: int = 800_000
    xenc_eval_pairs: int = 50_000
    xenc_eval_every: int = 2000  # optimizer steps
    xenc_patience: int = 3
    xenc_band_lo: float = 0.01  # stage-1 prob band sent to the cross-encoder (the uncertain pairs)
    xenc_band_hi: float = 0.998
    xenc_max_per_s1: int = 8
    xenc_test_models: str = "all"  # "all" = mean of fold models, "first" = fold-0 only

    # ---------------------------------------------------------------- decision
    mc_max_cands: int = 20  # candidates per S1 considered by the expected-F0.5 rule
    decide_chunk: int = 100_000

    # ---------------------------------------------------------------- cascade pruning (prune.py)
    # The stage-1 GBM is the last blocking filter: only candidates with stage-1 rank < prune_k AND
    # p1 >= prune_min_p1 reach the final matcher (gbm2 inference + decision). That set is what
    # candidate_pairs.tsv contains. 0 / 0.0 disables pruning.
    prune_k: int = 15
    prune_min_p1: float = 0.003

    # ---------------------------------------------------------------- self-training for unseen countries
    # gbm2 is refit with pseudo-labelled test pairs of countries absent from the training data (open set,
    # derived from the data): calibrated first-pass score >= pseudo_hi -> 1, <= pseudo_lo -> 0.
    # pseudo_hi = 0 disables it.
    pseudo_hi: float = 0.97
    pseudo_lo: float = 0.03

    # ------------------------------------------------------------------------
    @property
    def art(self) -> Path:
        return self.root / "artifacts" / self.profile

    @property
    def out_dir(self) -> Path:
        return self.root / ("output" if self.profile == "full" else f"output_{self.profile}")

    @property
    def reports(self) -> Path:
        return self.root / ("reports" if self.profile == "full" else f"reports/{self.profile}")

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        return json.dumps(d, default=str, indent=1)


def apply_profile(cfg: Config) -> Config:
    if cfg.profile == "smoke":
        cfg.train_s1_n = 300
        cfg.feat_s1_frac = 1.0
        cfg.train_other_n = 6000
        cfg.test_s1_n = 150
        cfg.test_other_n = 3000
        cfg.embed = EmbedModel("intfloat/multilingual-e5-small", "mean", "query: ", 64, 64)
        cfg.embed_fp16 = False
        cfg.k_max = {"dense": 15, "combo_word": 15, "combo_char": 10, "name_char": 10, "addr_char": 10,
                     "name_word": 10, "addr_word": 10}
        cfg.k_grid = (0, 2, 5, 10, 15)
        cfg.cand_cap = 30
        cfg.max_mean_cands = 30
        cfg.tfidf_min_df = 1
        cfg.tfidf_max_df_frac = 0.2
        cfg.n_folds = 3
        cfg.gbm_rounds = 60
        cfg.gbm_early_stop = 20
        cfg.gbm_params = {**cfg.gbm_params, "num_leaves": 15, "min_data_in_leaf": 20, "learning_rate": 0.1}
        cfg.gbm_fit_s1_frac = 1.0
        cfg.xenc_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
        cfg.xenc_folds = 2
        cfg.xenc_max_len = 64
        cfg.xenc_batch = 16
        cfg.xenc_infer_batch = 64
        cfg.xenc_train_pairs_per_fold = 400
        cfg.xenc_eval_pairs = 100
        cfg.xenc_eval_every = 10
        cfg.xenc_max_per_s1 = 3
        cfg.xenc_band_lo, cfg.xenc_band_hi = 0.0, 1.01
        cfg.n_jobs = min(cfg.n_jobs, 4)
    elif cfg.profile == "dev":
        cfg.train_s1_n = 20_000
        cfg.feat_s1_frac = 1.0
        cfg.train_other_frac = 0.1
        cfg.test_s1_n = 3_000
        cfg.test_other_frac = 0.01
        cfg.tune_s1_max = 20_000
        cfg.n_jobs = min(cfg.n_jobs, 8)
        cfg.embed = EmbedModel("intfloat/multilingual-e5-small", "mean", "query: ", 64, 128)
        cfg.embed_fp16 = False
        cfg.n_folds = 5
        cfg.gbm_rounds = 800
        cfg.gbm_params = {**cfg.gbm_params, "num_leaves": 63, "min_data_in_leaf": 50}
        cfg.gbm_fit_s1_frac = 1.0
        cfg.xenc_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
        cfg.xenc_folds = 2
        cfg.xenc_max_len = 96
        cfg.xenc_batch = 32
        cfg.xenc_train_pairs_per_fold = 20_000
        cfg.xenc_eval_pairs = 2_000
        cfg.xenc_eval_every = 100
    elif cfg.profile != "full":
        raise ValueError(f"unknown profile {cfg.profile}")
    return cfg


def load_config(profile: str = "full", root: str | Path | None = None, data_dir: str | Path | None = None,
                device: str = "auto", overrides: dict | None = None) -> Config:
    cfg = Config(profile=profile)
    cfg.root = Path(root or os.environ.get("BER_ROOT", ".")).resolve()
    cfg.data_dir = Path(data_dir).resolve() if data_dir else cfg.root / "dataset"
    cfg.device = device
    cfg = apply_profile(cfg)
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key {k}")
        setattr(cfg, k, type(getattr(cfg, k))(v) if getattr(cfg, k) is not None else v)
    return cfg


def resolve_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
