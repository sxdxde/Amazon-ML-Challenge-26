"""Checks the exact expected-F0.5 selector against Monte Carlo. Run directly or via pytest."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decide import expected_f_best_k  # noqa: E402


def mc_expected_f(p: np.ndarray, k: int, n: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(p))) < p
    tp = X[:, :k].sum(1)
    t = X.sum(1)
    if k == 0:
        return float((t == 0).mean())
    f = np.where(t > 0, 1.25 * tp / (0.25 * t + k), 0.0)
    return float(f.mean())


def test_against_monte_carlo():
    rng = np.random.default_rng(1)
    for _ in range(20):
        L = 6
        p = np.sort(rng.random(L) ** rng.uniform(0.3, 3))[::-1]
        k, e = expected_f_best_k(p[None, :])
        mc = [mc_expected_f(p, kk) for kk in range(L + 1)]
        assert abs(e[0] - max(mc)) < 0.01, (p, e, mc)
        assert abs(mc[k[0]] - max(mc)) < 0.01


def test_obvious_cases():
    k, _ = expected_f_best_k(np.array([[0.95, 0.9, 0.01, 0.0]]))
    assert k[0] == 2
    k, e = expected_f_best_k(np.array([[0.1, 0.05, 0.0, 0.0]]))
    assert k[0] == 0 and abs(e[0] - 0.9 * 0.95) < 1e-9


if __name__ == "__main__":
    test_obvious_cases()
    test_against_monte_carlo()
    print("ALL DECIDE TESTS PASSED")
