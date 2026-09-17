"""End-to-end pipeline integration test.

Exercises the full ds-agent pipeline on the synthetic fixture:
    Profiler → Statistics → Quality → Feature-gen/Select
           → TaskDetector → Trainer → MetricsCalculator
           → Explainer → Leaderboard → DriftDetector

No external services touched; everything runs in-process. This is the
canonical smoke-test for the assembled agent.
"""
from __future__ import annotations

import pandas as pd

from agent.monitoring.drift_detector import DriftDetector
from agent.pipeline.eda.profiler import DataProfiler
from agent.pipeline.eda.quality_scorer import QualityScorer
from agent.pipeline.eda.statistics import StatisticsAnalyser
from agent.pipeline.evaluator.explainer import ModelExplainer
from agent.pipeline.evaluator.leaderboard import Leaderboard
from agent.pipeline.evaluator.metrics import MetricsCalculator
from agent.pipeline.feature_engineering.generator import FeatureGenerator
from agent.pipeline.model_builder.task_detector import TaskDetector
from agent.pipeline.model_builder.trainer import ModelTrainer


def test_end_to_end_binary_classification_pipeline(synthetic_df, tmp_path):
    """Run the whole stack top-to-bottom and assert each stage produces ok."""
    df = synthetic_df.drop(columns=["id"])  # drop leakage-prone id
    target = "target_binary"

    # ---- EDA -------------------------------------------------------- #
    prof = DataProfiler().run(df, run_id="e2e")
    assert prof.status == "ok"
    assert prof.data["n_rows"] == len(df)

    stats = StatisticsAnalyser().mutual_information(df, target=target)
    assert stats.status == "ok"
    assert len(stats.data["scores"]) > 0

    qual = QualityScorer().score(df)
    assert qual.status in {"ok", "warning"}
    assert 0 <= qual.data["score"] <= 1

    # ---- Feature engineering --------------------------------------- #
    # Drop datetime column to keep generator deterministic; it's already
    # covered by its own unit tests.
    num_df = df.drop(columns=["feature_datetime"])
    gen = FeatureGenerator(max_pairs=5, max_onehot_cardinality=8).generate(
        num_df, target=target)
    assert gen.status == "ok"
    augmented = gen.data["frame"]
    assert augmented.shape[1] >= num_df.shape[1]

    # ---- Task detection + training --------------------------------- #
    task = TaskDetector().infer(augmented, target=target)
    assert task.status == "ok"
    assert task.data["task"] == "binary_classification"

    trainer = ModelTrainer(
        task_type="binary_classification",
        config={"target": target, "estimator": "rf",
                "n_estimators": 30, "test_size": 0.25, "random_state": 0},
    )
    trained = trainer.run(augmented, run_id="e2e")
    assert trained.status == "ok"
    assert trained.data["model"] is not None
    assert 0 <= trained.data["metrics"].get("accuracy", 0) <= 1

    # ---- Metrics (validate the metrics dict shape) ----------------- #
    metrics = trained.data["metrics"]
    assert "accuracy" in metrics
    assert any(k.startswith("f1") for k in metrics)

    # Sanity: MetricsCalculator happily consumes a small synthetic vector.
    mr = MetricsCalculator().compute(
        pd.Series([0, 1, 1, 0]), pd.Series([0, 1, 0, 0]),
        task="binary_classification")
    assert mr.status == "ok"

    # ---- Explainability (feature importance) ----------------------- #
    model = trained.data["model"]
    feature_names = trained.data["feature_names"]
    expl = ModelExplainer().shap_values(
        model, augmented[feature_names])
    assert expl.status == "ok"
    assert len(expl.data["mean_abs_importance"]) > 0

    # ---- Leaderboard persistence ----------------------------------- #
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    lb.log_run(
        run_id="e2e",
        model_type="rf",
        hyperparams={"n_estimators": 30, "test_size": 0.25},
        metrics=trained.data["metrics"],
        train_time_s=0.0,
        inference_ms=1.0, model_mb=1.0, shap_runtime_s=0.1,
    )
    top = lb.top_n("accuracy", n=1)
    assert top.status == "ok"
    assert top.data["runs"][0]["run_id"] == "e2e"
    lb.close()

    # ---- Drift check against a lightly-shifted copy ---------------- #
    baseline = augmented[[c for c in augmented.columns if c != target]]
    shifted = baseline.copy()
    num_cols = shifted.select_dtypes("number").columns
    shifted[num_cols] = shifted[num_cols] + shifted[num_cols].std() * 2.0
    drift = DriftDetector().check_all_features(baseline, shifted)
    assert drift.status == "ok"
    assert drift.data["n_checked"] > 0
    # With a 2σ shift on numeric columns, at least some should trip PSI.
    assert len(drift.data["drifting"]) >= 1


def test_end_to_end_regression_pipeline(synthetic_df):
    """Regression flavour: exercise the regression branch of Trainer + Metrics."""
    df = synthetic_df.drop(columns=["id", "target_binary", "feature_datetime"])
    target = "target_regression"

    task = TaskDetector().infer(df, target=target)
    assert task.data["task"] == "regression"

    trainer = ModelTrainer(
        task_type="regression",
        config={"target": target, "estimator": "rf",
                "n_estimators": 30, "test_size": 0.25, "random_state": 0},
    )
    r = trainer.run(df, run_id="e2e-reg")
    assert r.status == "ok"
    metrics = r.data["metrics"]
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0
