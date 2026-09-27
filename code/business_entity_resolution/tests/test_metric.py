"""Unit tests for metric.py. Run: python code/business_entity_resolution/tests/test_metric.py (or pytest)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metric import f_beta_single, macro_f05, per_entity_scores  # noqa: E402


def test_task_example():
    assert round(f_beta_single({"47", "193", "812"}, {"47", "812"}), 3) == 0.714


def test_singleton_empty_pred():
    assert f_beta_single(set(), set()) == 1.0


def test_singleton_any_pred():
    assert f_beta_single({"S2-1"}, set()) == 0.0


def test_matches_empty_pred():
    assert f_beta_single(set(), {"S2-1"}) == 0.0


def test_no_overlap():
    assert f_beta_single({"S2-9"}, {"S2-1"}) == 0.0


def test_macro_and_vectorised_agree():
    truth = {"a": {"47", "812"}, "b": set(), "c": {"1"}, "d": set()}
    pred = {"a": {"47", "193", "812"}, "b": set(), "c": set(), "d": {"5"}}
    expected = (0.7142857 + 1.0 + 0.0 + 0.0) / 4
    assert abs(macro_f05(pred, truth) - expected) < 1e-6
    pp = pd.DataFrame([(k, i) for k, v in pred.items() for i in v], columns=["s1", "id"])
    tp = pd.DataFrame([(k, i) for k, v in truth.items() for i in v], columns=["s1", "id"])
    sc = per_entity_scores(pp, tp, pd.Series(list(truth)))
    assert abs(sc.f.mean() - expected) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL METRIC TESTS PASSED")
