"""Unit tests for Week 15-18 model_builder phase."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.pipeline.model_builder.causal import CausalInferenceModule
from agent.pipeline.model_builder.hpo import HPOptimiser
from agent.pipeline.model_builder.loss_selector import LossSelector
from agent.pipeline.model_builder.task_detector import TaskDetector
from agent.pipeline.model_builder.timeseries import TimeSeriesModelBuilder
from agent.pipeline.model_builder.trainer import ModelTrainer

# --------------------------------------------------------------------------- #
# TaskDetector
# --------------------------------------------------------------------------- #


def test_task_detector_binary():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    r = TaskDetector().infer(df, target="y")
    assert r.data["task"] == "binary_classification"


def test_task_detector_multiclass_categorical():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "c", "a"]})
    r = TaskDetector().infer(df, target="y")
    assert r.data["task"] == "multiclass_classification"


def test_task_detector_multiclass_integer():
    df = pd.DataFrame({"x": list(range(10)), "y": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]})
    r = TaskDetector().infer(df, target="y")
    assert r.data["task"] == "multiclass_classification"


def test_task_detector_regression():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1.2, 3.5, 7.1]})
    r = TaskDetector().infer(df, target="y")
    assert r.data["task"] == "regression"


def test_task_detector_timeseries_forecasting_no_target():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"x": range(10)}, index=idx)
    r = TaskDetector().infer(df, target=None)
    assert r.data["task"] == "timeseries_forecasting"


def test_task_detector_clustering_no_target():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    r = TaskDetector().infer(df, target=None)
    assert r.data["task"] == "clustering"


def test_task_detector_anomaly_via_hint():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    r = TaskDetector().infer(df, target=None, eda_report={"outlier_rate": 0.5})
    assert r.data["task"] == "anomaly_detection"


def test_task_detector_survival_hint():
    df = pd.DataFrame({"x": [1]})
    r = TaskDetector().infer(df, target=None, eda_report={"survival": True})
    assert r.data["task"] == "survival_analysis"


def test_task_detector_causal_hint():
    df = pd.DataFrame({"x": [1]})
    r = TaskDetector().infer(df, target=None, eda_report={"causal": True})
    assert r.data["task"] == "causal_inference"


# --------------------------------------------------------------------------- #
# LossSelector
# --------------------------------------------------------------------------- #


def test_loss_selector_binary_balanced():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=1.0)
    assert r.data["loss"] == "bce"


def test_loss_selector_binary_weighted():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=3.0)
    assert r.data["loss"] == "weighted_bce"


def test_loss_selector_binary_focal():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=10.0)
    assert r.data["loss"] == "focal"


def test_loss_selector_regression():
    assert LossSelector().select("regression").data["loss"] == "huber"


def test_loss_selector_timeseries():
    assert LossSelector().select("timeseries_forecasting").data["loss"] == "quantile"


def test_loss_selector_clustering():
    assert LossSelector().select("clustering").data["loss"] == "silhouette"


def test_loss_selector_bad_ratio():
    r = LossSelector().select("regression", class_imbalance_ratio=0)
    assert r.status == "error"


def test_loss_selector_unknown_task():
    r = LossSelector().select("teleportation", class_imbalance_ratio=1.0)  # type: ignore[arg-type]
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# ModelTrainer
# --------------------------------------------------------------------------- #


def _reg_df(n: int = 200):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = x[:, 0] * 2 - x[:, 1] + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y})


def _classif_df(n: int = 200):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y})


def test_trainer_regression_rf():
    r = ModelTrainer("regression", {"target": "y"}).run(_reg_df(), "run")
    assert r.status == "ok"
    assert "r2" in r.data["metrics"]
    assert r.data["metrics"]["r2"] > 0.5


def test_trainer_regression_linear():
    r = ModelTrainer("regression", {"target": "y", "estimator": "linear"}).run(_reg_df(), "run")
    assert r.status == "ok"
    assert r.data["metrics"]["r2"] > 0.7


def test_trainer_binary_classification():
    r = ModelTrainer("binary_classification", {"target": "y"}).run(_classif_df(), "run")
    assert r.status == "ok"
    assert r.data["metrics"]["accuracy"] > 0.7
    assert "roc_auc" in r.data["metrics"]


def test_trainer_multiclass():
    rng = np.random.default_rng(0)
    n = 150
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0).astype(int)
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    r = ModelTrainer("multiclass_classification", {"target": "y"}).run(df, "run")
    assert r.status == "ok"
    assert r.data["metrics"]["f1_macro"] > 0.5


def test_trainer_clustering():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    r = ModelTrainer("clustering", {"n_clusters": 3}).run(df, "run")
    assert r.status == "ok"
    assert r.data["k"] == 3
    assert len(r.data["labels"]) == 60


def test_trainer_anomaly():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(100, 2)), rng.normal(loc=20, size=(5, 2))])
    df = pd.DataFrame(x, columns=["a", "b"])
    r = ModelTrainer("anomaly_detection", {"contamination": 0.05}).run(df, "run")
    assert r.status == "ok"
    assert r.data["metrics"]["n_outliers"] > 0


def test_trainer_missing_target():
    r = ModelTrainer("regression", {}).run(pd.DataFrame({"a": [1, 2]}), "run")
    assert r.status == "error"


def test_trainer_unsupported_task():
    r = ModelTrainer("magic").run(pd.DataFrame({"a": [1]}), "run")
    assert r.status == "error"


def test_trainer_unknown_estimator():
    r = ModelTrainer("regression", {"target": "y", "estimator": "xgb"}).run(_reg_df(50), "run")
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# HPOptimiser
# --------------------------------------------------------------------------- #


def test_hpo_minimises_quadratic():
    def obj(p):
        return (p["x"] - 3.0) ** 2

    r = HPOptimiser(direction="minimize").optimise(
        obj, {"x": ("uniform", -10.0, 10.0)}, n_trials=30,
    )
    assert r.status == "ok"
    assert abs(r.data["best_params"]["x"] - 3.0) < 1.5
    assert r.data["best_value"] < 2.25  # within (1.5)²


def test_hpo_maximises():
    def obj(p):
        return -((p["x"]) ** 2) + 5

    r = HPOptimiser(direction="maximize").optimise(
        obj, {"x": ("uniform", -2.0, 2.0)}, n_trials=20,
    )
    assert r.status == "ok"
    assert r.data["best_value"] > 4.0


def test_hpo_int_and_categorical():
    def obj(p):
        return abs(p["k"] - 4) + (0 if p["c"] == "rf" else 1.0)

    r = HPOptimiser().optimise(
        obj, {"k": ("int", 1, 10), "c": ("categorical", ["rf", "gbm"])},
        n_trials=20,
    )
    assert r.status == "ok"
    assert r.data["best_params"]["c"] == "rf"


def test_hpo_loguniform_accepted():
    def obj(p):
        return (np.log10(p["lr"]) + 3) ** 2

    r = HPOptimiser().optimise(
        obj, {"lr": ("loguniform", 1e-5, 1e-1)}, n_trials=20,
    )
    assert r.status == "ok"


def test_hpo_rejects_empty_space():
    r = HPOptimiser().optimise(lambda p: 0, {}, n_trials=1)
    assert r.status == "error"


def test_hpo_rejects_zero_trials():
    r = HPOptimiser().optimise(lambda p: 0, {"x": ("uniform", 0, 1)}, n_trials=0)
    assert r.status == "error"


def test_hpo_rejects_bad_spec():
    r = HPOptimiser().optimise(
        lambda p: 0, {"x": ("weirdthing", 0, 1)}, n_trials=2,
    )
    assert r.status == "error"


def test_hpo_bad_direction():
    with pytest.raises(ValueError):
        HPOptimiser(direction="sideways")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# TimeSeriesModelBuilder
# --------------------------------------------------------------------------- #


def _ts(n: int = 60):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({"y": np.sin(t / 5.0) + 0.01 * t})


def test_lstm_trains_and_reports_metrics():
    r = TimeSeriesModelBuilder().fit_lstm(
        _ts(), {"lookback": 8, "horizon": 1, "epochs": 5, "hidden": 8},
    )
    assert r.status == "ok"
    assert r.data["architecture"] == "LSTM"
    assert len(r.data["loss_history"]) == 5
    # loss should decrease across epochs on smooth signal
    assert r.data["loss_history"][-1] < r.data["loss_history"][0] + 1e-6


def test_tcn_trains():
    r = TimeSeriesModelBuilder().fit_tcn(
        _ts(), {"lookback": 8, "horizon": 1, "epochs": 5, "channels": 8},
    )
    assert r.status == "ok"
    assert r.data["architecture"] == "TCN"


def test_nbeats_trains():
    r = TimeSeriesModelBuilder().fit_nbeats(
        _ts(), {"lookback": 8, "horizon": 2, "epochs": 5, "stacks": 2, "hidden": 8},
    )
    assert r.status == "ok"
    assert r.data["architecture"] == "N-BEATS"
    assert r.data["horizon"] == 2


def test_nhits_trains():
    r = TimeSeriesModelBuilder().fit_nhits(
        _ts(), {"lookback": 8, "horizon": 2, "epochs": 5, "hidden": 8},
    )
    assert r.status == "ok"
    assert r.data["architecture"] == "N-HiTS"


def test_ts_rejects_short_series():
    r = TimeSeriesModelBuilder().fit_lstm(
        pd.DataFrame({"y": [1.0, 2.0]}),
        {"lookback": 5, "horizon": 1, "epochs": 1},
    )
    assert r.status == "error"


def test_ts_missing_target():
    r = TimeSeriesModelBuilder().fit_lstm(pd.DataFrame({"z": [1.0] * 20}), {})
    assert r.status == "error"


def test_prophet_missing_is_graceful():
    # prophet is not installed in this env; should return a clean error.
    r = TimeSeriesModelBuilder().fit_prophet(
        pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=5), "y": [1.0] * 5}),
    )
    assert r.status == "error"
    assert "prophet" in r.explanation.lower()


# --------------------------------------------------------------------------- #
# CausalInferenceModule
# --------------------------------------------------------------------------- #


def _causal_df(n: int = 400, true_ate: float = 2.0):
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    # confounder-driven treatment
    logit = 0.8 * X1 + 0.4 * X2
    p = 1 / (1 + np.exp(-logit))
    T = (rng.uniform(size=n) < p).astype(int)
    Y = true_ate * T + X1 - 0.5 * X2 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"X1": X1, "X2": X2, "T": T, "Y": Y})


def test_causal_backdoor_recovers_ate():
    r = CausalInferenceModule().estimate_ate(_causal_df(), "T", "Y", method="backdoor")
    assert r.status == "ok"
    assert abs(r.data["ate"] - 2.0) < 0.5


def test_causal_ipw_recovers_ate():
    r = CausalInferenceModule().estimate_ate(_causal_df(), "T", "Y", method="ipw")
    assert r.status == "ok"
    assert abs(r.data["ate"] - 2.0) < 0.8


def test_causal_dr_recovers_ate_with_se():
    r = CausalInferenceModule().estimate_ate(_causal_df(), "T", "Y", method="doubly_robust")
    assert r.status == "ok"
    assert abs(r.data["ate"] - 2.0) < 0.5
    assert r.data["se"] > 0


def test_causal_rejects_non_binary_treatment():
    df = pd.DataFrame({"T": [0, 1, 2], "Y": [1, 2, 3]})
    r = CausalInferenceModule().estimate_ate(df, "T", "Y")
    assert r.status == "error"


def test_causal_missing_columns():
    r = CausalInferenceModule().estimate_ate(pd.DataFrame({"T": [0, 1]}), "T", "Y")
    assert r.status == "error"


def test_causal_unknown_method():
    r = CausalInferenceModule().estimate_ate(_causal_df(50), "T", "Y", method="magic")
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# LossSelector — additional task types and edge cases
# --------------------------------------------------------------------------- #



def test_loss_selector_binary_balanced_ok():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=1.0)
    assert r.status == "ok"
    assert r.data["loss"] == "bce"


def test_loss_selector_binary_moderate_imbalance():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=3.0)
    assert r.status == "ok"
    assert r.data["loss"] == "weighted_bce"


def test_loss_selector_binary_high_imbalance():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=10.0)
    assert r.status == "ok"
    assert r.data["loss"] == "focal"


def test_loss_selector_multiclass_balanced():
    r = LossSelector().select("multiclass_classification", class_imbalance_ratio=1.0)
    assert r.status == "ok"
    assert r.data["loss"] == "cross_entropy"


def test_loss_selector_multiclass_imbalanced():
    r = LossSelector().select("multiclass_classification", class_imbalance_ratio=3.0)
    assert r.status == "ok"
    assert r.data["loss"] == "weighted_cross_entropy"


def test_loss_selector_regression_ok():
    r = LossSelector().select("regression")
    assert r.status == "ok"
    assert r.data["loss"] == "huber"


def test_loss_selector_timeseries_ok():
    r = LossSelector().select("timeseries_forecasting")
    assert r.status == "ok"
    assert r.data["loss"] == "quantile"


def test_loss_selector_anomaly_detection():
    r = LossSelector().select("anomaly_detection")
    assert r.status == "ok"
    assert r.data["loss"] == "reconstruction_mse"


def test_loss_selector_clustering_ok():
    r = LossSelector().select("clustering")
    assert r.status == "ok"
    assert r.data["loss"] == "silhouette"


def test_loss_selector_survival_analysis():
    r = LossSelector().select("survival_analysis")
    assert r.status == "ok"
    assert r.data["loss"] == "cox_partial_likelihood"


def test_loss_selector_causal_inference():
    r = LossSelector().select("causal_inference")
    assert r.status == "ok"
    assert r.data["loss"] == "doubly_robust"


def test_loss_selector_unknown_task_errors():
    r = LossSelector().select("unknown_task_type")
    assert r.status == "error"


def test_loss_selector_zero_ratio_error():
    r = LossSelector().select("binary_classification", class_imbalance_ratio=0.0)
    assert r.status == "error"


def test_loss_selector_negative_ratio_error():
    r = LossSelector().select("regression", class_imbalance_ratio=-1.0)
    assert r.status == "error"
