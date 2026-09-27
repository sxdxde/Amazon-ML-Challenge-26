"""v2 configuration: every path, k, budget and seed lives here.

Layout (ROOT = the project folder that holds dataset/, artifacts/, code/):
  ROOT/artifacts/full/...          v1 artifacts (READ-ONLY for v2: prepare tables, bge-m3 embeddings)
  ROOT/artifacts_v2/<profile>/...  v2 artifacts
  ROOT/experiments/                v2 model runs + leaderboard.csv
  ROOT/output_v2/                  v2 submission files
  ROOT/reports/v2_*.md             v2 reports
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[4]


@dataclass
class Config:
    profile: str = "full"
    root: Path = ROOT_DEFAULT
    seed: int = 42

    # ---------------------------------------------------------------- resources (v1 may be running)
    n_jobs: int = 16
    vram_gb: float = 10.0          # hard budget for v2 GPU stages (other jobs share the A100)
    device: str = "cuda"

    # ---------------------------------------------------------------- v1 inputs (read-only)
    v1_profile: str = "full"
    v1_tune_s1_max: int = 150_000  # v1's tuning subsample (reproduced for an apples-to-apples recall reference)

    # ---------------------------------------------------------------- dev profile sampling
    dev_s1_per_split: int = 20_000
    dev_pool_per_country: int = 150_000   # random pool slice per country (plus every true match of the dev S1)

    # ---------------------------------------------------------------- folds (shared by adapter + models)
    n_folds: int = 5

    # ---------------------------------------------------------------- stage 2: lexical vectors
    lex_fields: dict = field(default_factory=lambda: {"name_lex": "name_core", "addr_lex": "addr_core"})
    lex_ngram: tuple = (3, 4)
    lex_n_features: int = 2 ** 20
    lex_max_df: float = 0.005      # zero out hashed n-grams present in > 0.5% of the pool docs
    lex_dim: int = 256
    lex_chunk: int = 100_000

    # ---------------------------------------------------------------- stage 3: search
    k_wide: int = 50
    q_chunk: int = 4096
    rev_k: int = 10
    rev_retrievers: tuple = ("sem", "name_lex")
    dense_retrievers: tuple = ("sem", "name_lex", "addr_lex")
    key_block_cap: int = 20        # key retrievers skip keys shared by more than this many pool records
    rare_df_cap: int = 20          # "rare" name token: in <= this many pool records of the country
    k_rare: int = 10
    recall_ks: tuple = (5, 10, 20, 30, 50)

    # ---------------------------------------------------------------- derived paths
    @property
    def v1_art(self) -> Path:
        return self.root / "artifacts" / self.v1_profile

    @property
    def art(self) -> Path:
        return self.root / "artifacts_v2" / self.profile

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def output(self) -> Path:
        return self.root / ("output_v2" if self.profile == "full" else f"output_v2_{self.profile}")

    def v1_emb_path(self, split: str) -> Path:
        return self.v1_art / "embed" / f"{split}.npy"

    def v1_records_path(self, split: str) -> Path:
        return self.v1_art / "prepare" / f"{split}_records.parquet"

    def hash(self, *keys: str) -> str:
        d = asdict(self)
        d = {k: d[k] for k in keys} if keys else d
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(profile: str = "full", **overrides) -> Config:
    cfg = Config(profile=profile)
    if profile == "dev":
        cfg.k_wide = 50
    elif profile != "full":
        raise ValueError(f"unknown profile {profile}")
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key {k}")
        setattr(cfg, k, type(getattr(cfg, k))(v) if not isinstance(getattr(cfg, k), (dict, tuple)) else v)
    return cfg
