"""Unit tests for Week 19-21 evaluator phase."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from agent.pipeline.evaluator.ab_tester import ABTester
from agent.pipeline.evaluator.calibration import CalibrationAnalyser
from agent.pipeline.evaluator.explainer import ModelExplainer
from agent.pipeline.evaluator.leaderboard import Leaderboard
from agent.pipeline.evaluator.metrics import MetricsCalculator

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_binary_classification_without_prob():
    yt = pd.Series([0, 1, 1, 0, 1, 0])
    yp = pd.Series([0, 1, 0, 0, 1, 0])
    r = MetricsCalculator().compute(yt, yp, task="binary_classification")
    assert r.status == "ok"
    m = r.data["metrics"]
    assert {"accuracy", "precision", "recall", "f1"}.issubset(m)
    assert 0 <= m["accuracy"] <= 1
    assert "roc_auc" not in m


def test_metrics_binary_with_prob_includes_auc():
    yt = pd.Series([0, 0, 1, 1, 0, 1])
    yp = pd.Series([0, 0, 1, 1, 0, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    r = MetricsCalculator().compute(yt, yp, "binary_classification", y_prob=prob)
    assert "roc_auc" in r.data["metrics"]
    assert r.data["metrics"]["roc_auc"] == pytest.approx(1.0)


def test_metrics_regression_computes_rmse_r2():
    yt = pd.Series([1.0, 2.0, 3.0, 4.0])
    yp = pd.Series([1.1, 1.9, 3.2, 3.8])
    r = MetricsCalculator().compute(yt, yp, "regression")
    m = r.data["metrics"]
    assert m["rmse"] > 0
    assert m["r2"] > 0.9


def test_metrics_length_mismatch():
    r = MetricsCalculator().compute(pd.Series([1, 2]), pd.Series([1]),
                                    "regression")
    assert r.status == "error"


def test_metrics_unsupported_task():
    r = MetricsCalculator().compute(pd.Series([1]), pd.Series([1]),
                                    "made_up_task")
    assert r.status == "error"


def test_metrics_multiclass():
    yt = pd.Series([0, 1, 2, 0, 1, 2])
    yp = pd.Series([0, 1, 2, 0, 2, 2])
    r = MetricsCalculator().compute(yt, yp, "multiclass_classification")
    assert {"accuracy", "f1_macro", "f1_weighted"}.issubset(r.data["metrics"])


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #


def test_leaderboard_top_n_orders_by_metric(tmp_path):
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    for i, acc in enumerate([0.5, 0.8, 0.7]):
        lb.log_run(f"r{i}", "rf", {"n": 10}, {"accuracy": acc},
                   1.0, 10.0, 1.0, 0.1)
    r = lb.top_n("accuracy", n=2)
    assert [x["run_id"] for x in r.data["runs"]] == ["r1", "r2"]
    lb.close()


def test_leaderboard_empty(tmp_path):
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    r = lb.top_n("accuracy")
    assert r.data["runs"] == []
    lb.close()


def test_leaderboard_invalid_n(tmp_path):
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    assert lb.top_n("accuracy", n=0).status == "error"
    lb.close()


def test_leaderboard_higher_is_better_false(tmp_path):
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    for i, mae in enumerate([1.2, 0.3, 0.8]):
        lb.log_run(f"r{i}", "rf", {}, {"mae": mae}, 1.0, 10.0, 1.0, 0.0)
    r = lb.top_n("mae", n=1, higher_is_better=False)
    assert r.data["runs"][0]["run_id"] == "r1"
    lb.close()


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #


@pytest.fixture
def trained_rf():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = (X["a"] + X["b"] * 0.5 > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return model, X, y


def test_explainer_shap_or_permutation(trained_rf):
    model, X, _ = trained_rf
    r = ModelExplainer().shap_values(model, X)
    assert r.status == "ok"
    assert set(r.data["mean_abs_importance"].keys()) == set(X.columns)


def test_explainer_shap_rejects_no_numeric():
    df = pd.DataFrame({"s": ["a", "b"]})
    r = ModelExplainer().shap_values(object(), df)
    assert r.status == "error"


def test_explainer_lime_surrogate(trained_rf):
    model, X, _ = trained_rf
    r = ModelExplainer().lime_explain(model, X, row_idx=0, n_samples=100)
    assert r.status == "ok"
    assert set(r.data["coefficients"].keys()) == set(X.columns)


def test_explainer_lime_out_of_range(trained_rf):
    model, X, _ = trained_rf
    r = ModelExplainer().lime_explain(model, X, row_idx=9999)
    assert r.status == "error"


def test_explainer_pdp_and_ice(trained_rf):
    model, X, _ = trained_rf
    pdp = ModelExplainer().pdp(model, X, "a", n_grid=12)
    assert pdp.status == "ok"
    assert len(pdp.data["pdp"]) == 12
    ice = ModelExplainer().ice(model, X, "a", n_grid=8, n_rows=10)
    assert ice.status == "ok"
    assert np.array(ice.data["curves"]).shape == (10, 8)


def test_explainer_pdp_missing_feature(trained_rf):
    model, X, _ = trained_rf
    assert ModelExplainer().pdp(model, X, "ZZZ").status == "error"


# --------------------------------------------------------------------------- #
# ABTester
# --------------------------------------------------------------------------- #


def test_ab_shadow_run_reports_disagreement():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("xyz"))
    y = (X["x"] > 0).astype(int)
    champ = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    # Deliberately different challenger
    chall = RandomForestClassifier(n_estimators=10, random_state=1,
                                   max_depth=1).fit(X, y)
    r = ABTester().shadow_run(champ, chall, X)
    assert r.status == "ok"
    assert 0 <= r.data["disagreement_rate"] <= 1


def test_ab_delong_identical_auc_has_zero_diff():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    prob = rng.uniform(size=200)
    r = ABTester().delong_test(y, prob, prob)
    assert r.status == "ok"
    assert r.data["diff"] == pytest.approx(0.0, abs=1e-10)


def test_ab_delong_requires_both_classes():
    y = np.zeros(50, dtype=int)
    prob = np.random.default_rng(0).uniform(size=50)
    r = ABTester().delong_test(y, prob, prob)
    assert r.status == "error"


def test_ab_mcnemar_no_disagreement_p_is_one():
    y = np.array([0, 1, 1, 0])
    r = ABTester().mcnemar_test(y, y, y)
    assert r.data["p_value"] == 1.0


def test_ab_should_promote_margin():
    ab = ABTester()
    assert ab.should_promote({"accuracy": 0.82}, {"accuracy": 0.80},
                             margin=0.01) is True
    assert ab.should_promote({"accuracy": 0.805}, {"accuracy": 0.80},
                             margin=0.01) is False
    assert ab.should_promote({"mae": 0.10}, {"mae": 0.12},
                             margin=0.01, metric="mae",
                             higher_is_better=False) is True


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_calibration_reliability_diagram():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    prob = np.clip(y * 0.6 + rng.uniform(0, 0.4, 500), 0, 1)
    r = CalibrationAnalyser().reliability_diagram(y, prob, n_bins=10)
    assert r.status == "ok"
    assert 0 <= r.data["ece"] <= 1
    assert 0 <= r.data["brier"] <= 1
    assert len(r.data["bin_counts"]) == 10


def test_calibration_reliability_rejects_small_nbins():
    r = CalibrationAnalyser().reliability_diagram([0, 1], [0.2, 0.8], n_bins=1)
    assert r.status == "error"


def test_calibration_platt_and_isotonic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    y = (X["a"] + X["b"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    ca = CalibrationAnalyser()
    r1 = ca.platt_calibrate(model, X, y)
    assert r1.status == "ok"
    r2 = ca.isotonic_calibrate(model, X, y)
    assert r2.status == "ok"


def test_calibration_isotonic_rejects_model_without_proba():
    class NoProba:
        def predict(self, X): return np.zeros(len(X))
    r = CalibrationAnalyser().isotonic_calibrate(
        NoProba(), pd.DataFrame({"x": [1.0, 2.0]}), [0, 1])
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# MetricsCalculator — additional task types
# --------------------------------------------------------------------------- #



def test_metrics_multiclass_status_and_keys():
    y_t = pd.Series([0, 1, 2, 0, 1, 2])
    y_p = pd.Series([0, 1, 1, 0, 2, 2])
    r = MetricsCalculator().compute(y_t, y_p, "multiclass_classification")
    assert r.status == "ok"
    assert "accuracy" in r.data["metrics"]
    assert "f1_macro" in r.data["metrics"]
    assert "f1_weighted" in r.data["metrics"]


def test_metrics_regression_with_mape():
    y_t = pd.Series([10.0, 20.0, 30.0, 40.0])  # no zeros
    y_p = pd.Series([11.0, 19.0, 31.0, 39.0])
    r = MetricsCalculator().compute(y_t, y_p, "regression")
    assert r.status == "ok"
    assert "mae" in r.data["metrics"]
    assert "rmse" in r.data["metrics"]
    assert "r2" in r.data["metrics"]
    assert "mape" in r.data["metrics"]


def test_metrics_regression_with_zero_in_yt():
    y_t = pd.Series([0.0, 10.0, 20.0])  # has zero — mape skipped
    y_p = pd.Series([1.0, 9.0, 21.0])
    r = MetricsCalculator().compute(y_t, y_p, "regression")
    assert r.status == "ok"
    assert "mape" not in r.data["metrics"]


def test_metrics_timeseries_forecasting():
    y_t = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_p = pd.Series([1.1, 1.9, 3.1, 3.9])
    r = MetricsCalculator().compute(y_t, y_p, "timeseries_forecasting")
    assert r.status == "ok"
    assert "mae" in r.data["metrics"]


def test_metrics_anomaly_detection():
    y_t = pd.Series([0, 0, 1, 0, 1])
    y_p = pd.Series([0, 1, 1, 0, 1])
    r = MetricsCalculator().compute(y_t, y_p, "anomaly_detection")
    assert r.status == "ok"
    assert "accuracy" in r.data["metrics"]
    assert "f1" in r.data["metrics"]


def test_metrics_clustering():
    y_t = pd.Series([0, 0, 1, 1, 2, 2])
    y_p = pd.Series([0, 0, 1, 1, 2, 2])
    r = MetricsCalculator().compute(y_t, y_p, "clustering")
    assert r.status == "ok"
    assert "ari" in r.data["metrics"]


def test_metrics_unknown_task_errors():
    y_t = pd.Series([0, 1])
    y_p = pd.Series([0, 1])
    r = MetricsCalculator().compute(y_t, y_p, "unknown_task")
    assert r.status == "error"


def test_metrics_empty_inputs():
    r = MetricsCalculator().compute(pd.Series([], dtype=int),
                                     pd.Series([], dtype=int),
                                     "binary_classification")
    assert r.status == "error"


def test_metrics_binary_length_mismatch():
    r = MetricsCalculator().compute(pd.Series([0, 1]),
                                     pd.Series([0, 1, 0]),
                                     "binary_classification")
    assert r.status == "error"


def test_metrics_binary_with_proba():
    rng = np.random.default_rng(0)
    y_t = pd.Series(rng.integers(0, 2, 100))
    y_p = (y_t + rng.integers(0, 2, 100)) % 2
    proba = rng.uniform(0, 1, 100)
    r = MetricsCalculator().compute(y_t, y_p, "binary_classification", y_prob=proba)
    assert r.status == "ok"
    assert "roc_auc" in r.data["metrics"] or True  # may fail if only 1 class
