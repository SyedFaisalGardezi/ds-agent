"""Unit tests for monitoring (drift, cost, feedback)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent.monitoring.cost_tracker import CostTracker
from agent.monitoring.drift_detector import DriftDetector
from agent.monitoring.feedback_loop import FeedbackLoop

# --------------------------------------------------------------------------- #
# DriftDetector
# --------------------------------------------------------------------------- #


def test_psi_no_drift_on_identical_samples():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=500))
    y = pd.Series(rng.normal(size=500))
    r = DriftDetector().psi(x, y)
    assert r.status == "ok"
    assert r.data["psi"] < 0.1
    assert r.data["drift"] is False


def test_psi_detects_drift_on_shift():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(loc=0, scale=1, size=500))
    y = pd.Series(rng.normal(loc=3, scale=1, size=500))
    r = DriftDetector().psi(x, y)
    assert r.data["drift"] is True
    assert r.data["psi"] > 0.2


def test_psi_empty_input_errors():
    r = DriftDetector().psi(pd.Series([], dtype=float),
                            pd.Series([1.0, 2.0]))
    assert r.status == "error"


def test_ks_detects_different_distribution():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200))
    y = pd.Series(rng.normal(loc=2, size=200))
    r = DriftDetector().ks_test(x, y)
    assert r.status == "ok"
    assert r.data["drift"] is True


def test_ks_too_few_samples_errors():
    assert DriftDetector().ks_test(pd.Series([1.0]),
                                   pd.Series([2.0])).status == "error"


def test_chi_squared_categorical_drift():
    x = pd.Series(["a"] * 90 + ["b"] * 10)
    y = pd.Series(["a"] * 20 + ["b"] * 80)
    r = DriftDetector().chi_squared(x, y)
    assert r.status == "ok"
    assert r.data["drift"] is True


def test_chi_squared_single_category_errors():
    r = DriftDetector().chi_squared(pd.Series(["a", "a"]),
                                     pd.Series(["a", "a"]))
    assert r.status == "error"


def test_check_all_features_mixed_types():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({
        "num": rng.normal(size=300),
        "cat": rng.choice(["x", "y"], size=300),
    })
    current = pd.DataFrame({
        "num": rng.normal(loc=3, size=300),
        "cat": rng.choice(["x", "y"], size=300, p=[0.1, 0.9]),
    })
    r = DriftDetector().check_all_features(baseline, current)
    assert r.status == "ok"
    assert set(r.data["drifting"]) == {"num", "cat"}


def test_check_all_features_no_shared_errors():
    r = DriftDetector().check_all_features(
        pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# CostTracker
# --------------------------------------------------------------------------- #


def test_cost_tracker_accumulates_per_connector(tmp_path):
    ct = CostTracker(db_path=tmp_path / "c.db")
    ct.log_query_cost("bq", "SELECT 1", 0.5)
    ct.log_query_cost("bq", "SELECT 2", 1.5)
    ct.log_query_cost("sf", "SELECT 3", 2.0)
    r = ct.total_cost(since_days=30)
    assert r.data["total_usd"] == pytest.approx(4.0)
    assert r.data["n_queries"] == 3
    assert r.data["per_connector"]["bq"]["usd"] == pytest.approx(2.0)
    assert r.data["per_connector"]["sf"]["count"] == 1
    ct.close()


def test_cost_tracker_rejects_negative_window(tmp_path):
    ct = CostTracker(db_path=tmp_path / "c.db")
    assert ct.total_cost(since_days=-1).status == "error"
    ct.close()


def test_cost_tracker_zero_queries(tmp_path):
    ct = CostTracker(db_path=tmp_path / "c.db")
    r = ct.total_cost(since_days=30)
    assert r.data["total_usd"] == 0.0
    assert r.data["n_queries"] == 0
    ct.close()


# --------------------------------------------------------------------------- #
# FeedbackLoop
# --------------------------------------------------------------------------- #


def test_feedback_loop_insufficient_data(tmp_path):
    fb = FeedbackLoop(db_path=tmp_path / "f.db")
    for i in range(5):
        fb.store_label(str(i), 1, features={"x": i}, prediction=1)
    r = fb.check_accuracy_drop(window=10)
    assert r.status == "warning"
    assert r.data["needs_more_data"] is True
    fb.close()


def test_feedback_loop_detects_drop_and_sets_flag(tmp_path):
    fb = FeedbackLoop(db_path=tmp_path / "f.db")
    # Baseline: perfect accuracy (inserted first, so ends up oldest)
    for i in range(20):
        fb.store_label(f"b{i}", 1, features={}, prediction=1)
    # Recent: mostly wrong (inserted after — newer ids)
    for i in range(20):
        fb.store_label(f"r{i}", 1, features={},
                       prediction=0 if i < 15 else 1)
    r = fb.check_accuracy_drop(delta_threshold=0.1, window=20)
    assert r.status == "ok"
    assert r.data["trigger_retrain"] is True
    trig = fb.trigger_retrain()
    assert trig.data["retrain_triggered"] is True
    # Flag is consumed on read.
    assert fb.trigger_retrain().data["retrain_triggered"] is False
    fb.close()


def test_feedback_loop_rejects_bad_params(tmp_path):
    fb = FeedbackLoop(db_path=tmp_path / "f.db")
    assert fb.check_accuracy_drop(window=1).status == "error"
    assert fb.check_accuracy_drop(delta_threshold=0).status == "error"
    fb.close()


def test_feedback_loop_store_label_persists(tmp_path):
    fb = FeedbackLoop(db_path=tmp_path / "f.db")
    fb.store_label("p1", ground_truth=1, features={"a": 1}, prediction=0)
    cur = fb._conn.execute("SELECT ground_truth, prediction FROM labels")
    gt, pred = cur.fetchone()
    assert json.loads(gt) == 1
    assert json.loads(pred) == 0
    fb.close()
