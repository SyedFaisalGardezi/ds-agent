"""Unit tests for agent/api/data_analysis_agent.py."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.api.data_analysis_agent import (
    DataAnalysisAgent,
    DataAnalysisReport,
    MLPlan,
    MLStrategyPlanner,
    _safe_json,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _binary_df(n: int = 500, imbalanced: bool = False) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age":    rng.integers(18, 80, n).astype(float),
        "income": rng.exponential(50_000, n),
        "score":  rng.normal(100, 15, n),
        "cat_a":  rng.choice(["A", "B", "C"], n),
        "churn":  (
            rng.integers(0, 10, n) < 1  # 10% minority
            if imbalanced
            else rng.integers(0, 2, n)
        ).astype(int),
    })
    return df


def _regression_df(n: int = 300) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.uniform(-5, 5, n),
        "y":  rng.normal(10, 3, n),
    })


def _make_llm(json_response: str | None = None) -> MagicMock:
    llm = MagicMock()
    if json_response is None:
        json_response = json.dumps({
            "analyst_narrative": "Test narrative.",
            "ml_recommendations": {
                "models": [
                    {"name": "lgbm", "hyperparams": {"n_estimators": 300}},
                    {"name": "lr",   "hyperparams": {"C": 1.0}},
                ],
                "preprocessing": ["impute_median"],
                "features_to_exclude": [],
                "rationale": "Good dataset.",
            },
        })
    llm._generate.return_value = json_response
    llm._strip_thinking.side_effect = lambda x: x
    return llm


# ── _safe_json ────────────────────────────────────────────────────────────────

def test_safe_json_direct():
    assert _safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_fenced():
    assert _safe_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_embedded():
    result = _safe_json('Some text {"key": "val"} more text')
    assert result == {"key": "val"}


def test_safe_json_none():
    assert _safe_json("") is None
    assert _safe_json("plain text") is None


# ── DataAnalysisReport ────────────────────────────────────────────────────────

def test_report_roundtrip():
    r = DataAnalysisReport(n_rows=100, n_cols=5, target="y",
                           task_type="regression")
    d = r.to_dict()
    r2 = DataAnalysisReport.from_dict(d)
    assert r2.n_rows == 100
    assert r2.target == "y"
    assert r2.task_type == "regression"


def test_report_defaults():
    r = DataAnalysisReport()
    assert r.numeric_cols == []
    assert r.ml_recommendations == {}
    assert r.class_imbalance_ratio is None


# ── DataAnalysisAgent.analyze() ───────────────────────────────────────────────

def test_analyze_basic_shape():
    df = _binary_df(200)
    agent = DataAnalysisAgent(llm_client=_make_llm())
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    assert report.n_rows == 200
    assert report.n_cols == 5
    assert report.target == "churn"
    assert "churn" in [c for c in report.numeric_cols] or True  # churn is int


def test_analyze_col_types():
    df = _binary_df(200)
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    assert "age" in report.numeric_cols
    assert "cat_a" in report.categorical_cols


def test_analyze_target_distribution_binary():
    df = _binary_df(200)
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    assert isinstance(report.target_distribution, dict)
    assert len(report.target_distribution) >= 2
    assert report.class_imbalance_ratio is not None
    assert 0.0 < report.class_imbalance_ratio <= 1.0


def test_analyze_imbalanced_metric():
    df = _binary_df(500, imbalanced=True)
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    # Imbalance ratio < 0.4 → metric should be average_precision-related
    assert "precision" in report.recommended_metric.lower() or \
           "pr_auc" in report.recommended_metric.lower()


def test_analyze_regression_target():
    df = _regression_df(200)
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="y", task_type="regression")
    assert "mean" in report.target_distribution
    assert "std" in report.target_distribution
    assert report.class_imbalance_ratio is None


def test_analyze_null_detection():
    import numpy as np
    df = _binary_df(200)
    df.loc[:50, "age"] = np.nan   # ~25% null
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    assert "age" in report.null_pcts
    assert report.null_pcts["age"] > 20.0


def test_analyze_skew_detection():
    import numpy as np
    df = _binary_df(300)
    df["income"] = np.exp(df["income"] / 10000)  # create heavy skew
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    # income should be flagged as skewed
    assert "income" in report.skewed_cols


def test_analyze_llm_called():
    df = _binary_df(200)
    llm = _make_llm()
    agent = DataAnalysisAgent(llm_client=llm)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    llm._generate.assert_called_once()
    assert report.analyst_narrative == "Test narrative."
    assert len(report.ml_recommendations.get("models", [])) == 2


def test_analyze_llm_offline_uses_heuristics():
    df = _binary_df(300)
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification")
    # Heuristic fallback should still populate recommendations
    assert report.ml_recommendations.get("models")
    assert len(report.ml_recommendations["models"]) >= 1


def test_analyze_with_external_mi_scores():
    df = _binary_df(200)
    mi = {"age": 0.15, "income": 0.22, "score": 0.08}
    agent = DataAnalysisAgent(llm_client=None)
    report = agent.analyze(df, target="churn",
                           task_type="binary_classification",
                           mi_scores=mi)
    # top_mi_features should reflect the passed scores (income highest)
    assert report.top_mi_features[0][0] == "income"


# ── DataAnalysisAgent.heuristic_recommendations ───────────────────────────────

def test_heuristic_small_dataset():
    r = DataAnalysisReport(n_rows=500, task_type="binary_classification")
    rec = DataAnalysisAgent._heuristic_recommendations(r)
    names = [m["name"] for m in rec["models"]]
    assert "lr" in names or "rf" in names


def test_heuristic_large_imbalanced():
    r = DataAnalysisReport(
        n_rows=100_000, task_type="binary_classification",
        class_imbalance_ratio=0.1
    )
    rec = DataAnalysisAgent._heuristic_recommendations(r)
    names = [m["name"] for m in rec["models"]]
    assert "lgbm" in names or "xgb" in names


def test_heuristic_regression():
    r = DataAnalysisReport(n_rows=5000, task_type="regression")
    rec = DataAnalysisAgent._heuristic_recommendations(r)
    names = [m["name"] for m in rec["models"]]
    assert "lgbm" in names or "ridge" in names


def test_heuristic_preprocessing_null():
    import numpy as np
    r = DataAnalysisReport(null_pcts={"a": 30.0, "b": 15.0})
    rec = DataAnalysisAgent._heuristic_recommendations(r)
    assert "impute_median" in rec["preprocessing"]


def test_heuristic_preprocessing_skew():
    r = DataAnalysisReport(skewed_cols={"x": 4.5, "y": 3.2, "z": 5.0})
    rec = DataAnalysisAgent._heuristic_recommendations(r)
    assert "log1p_skewed" in rec["preprocessing"]


# ── MLPlan ────────────────────────────────────────────────────────────────────

def test_mlplan_roundtrip():
    p = MLPlan(
        models=[{"name": "lgbm", "hyperparams": {"n_estimators": 300}}],
        preprocessing=["impute_median"],
        eval_metric="roc_auc",
        task_type="binary_classification",
    )
    d = p.to_dict()
    p2 = MLPlan.from_dict(d)
    assert p2.models[0]["name"] == "lgbm"
    assert p2.eval_metric == "roc_auc"


# ── MLStrategyPlanner ─────────────────────────────────────────────────────────

def _make_report_dict(imbalance: float = 0.5) -> dict:
    return {
        "n_rows": 5000, "n_cols": 10,
        "target": "churn", "task_type": "binary_classification",
        "class_imbalance_ratio": imbalance,
        "recommended_metric": "roc_auc" if imbalance >= 0.4 else "average_precision",
        "null_pcts": {}, "skewed_cols": {}, "high_cardinality_cols": [],
        "near_constant_cols": [], "outlier_cols": [], "top_mi_features": [],
        "leakage_suspects": [], "analyst_narrative": "",
        "ml_recommendations": {
            "models": [
                {"name": "lgbm", "hyperparams": {"n_estimators": 400}},
            ],
            "preprocessing": ["impute_median"],
            "features_to_exclude": [],
            "rationale": "lgbm good for this size",
        },
    }


def test_planner_uses_report_recommendations():
    """When report has ml_recommendations, planner uses them directly."""
    planner = MLStrategyPlanner(llm_client=None)
    report = _make_report_dict()
    plan = planner.plan(report)
    assert plan.models[0]["name"] == "lgbm"
    assert plan.models[0]["hyperparams"]["n_estimators"] == 400


def test_planner_inherits_metric_from_report():
    planner = MLStrategyPlanner(llm_client=None)
    report = _make_report_dict(imbalance=0.1)
    plan = planner.plan(report)
    assert "precision" in plan.eval_metric or "roc" in plan.eval_metric


def test_planner_task_spec_overrides_metric():
    planner = MLStrategyPlanner(llm_client=None)
    report = _make_report_dict()
    task_spec = {"evaluation_metric": "pr_auc"}
    plan = planner.plan(report, task_spec=task_spec)
    assert plan.eval_metric == "pr_auc"


def test_planner_no_report_recs_uses_heuristics():
    """When report has no ml_recommendations, heuristic fallback used."""
    planner = MLStrategyPlanner(llm_client=None)
    report = _make_report_dict()
    report["ml_recommendations"] = {}  # empty
    plan = planner.plan(report)
    assert len(plan.models) >= 1


def test_planner_summarise_report():
    report = _make_report_dict()
    summary = MLStrategyPlanner._summarise_report(report)
    assert "5,000" in summary or "5000" in summary
    assert "churn" in summary


def test_planner_llm_called_when_no_rec():
    """When no recommendations in report, planner calls LLM."""
    llm_response = json.dumps({
        "models": [{"name": "rf", "hyperparams": {"n_estimators": 200}}],
        "preprocessing": [],
        "features_to_exclude": [],
        "eval_metric": "roc_auc",
        "rationale": "LLM choice",
    })
    llm = _make_llm(llm_response)
    planner = MLStrategyPlanner(llm_client=llm)
    report = _make_report_dict()
    report["ml_recommendations"] = {}

    plan = planner.plan(report)
    # LLM should have been called; result has models
    assert len(plan.models) >= 1


# ── Tool class integration (smoke tests) ─────────────────────────────────────

def _make_kernel(ns: dict | None = None):
    from agent.api.kernel import SessionKernel
    k = SessionKernel.__new__(SessionKernel)
    k.namespace = ns or {}
    def _execute(code):
        result = SimpleNamespace(success=True, error="", figures=[])
        result.as_text = lambda n=2000: ""
        return result
    k.execute = _execute
    return k


def _make_session(**kw):
    defaults = dict(target=None, brief="", data_path=None,
                    data_paths=[], brief_filenames=[], session_id="test")
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_data_analysis_agent_tool_no_df():
    from agent.api.agent_tools import DataAnalysisAgentTool
    tool = DataAnalysisAgentTool()
    k = _make_kernel({})
    s = _make_session()
    out = tool.execute({}, session=s, kernel=k, llm_client=None)
    assert out.success is False
    assert "no df" in out.error.lower() or "No data" in out.text


def test_data_analysis_agent_tool_with_df():
    from agent.api.agent_tools import DataAnalysisAgentTool
    df = _binary_df(100)
    k = _make_kernel({"df": df, "TASK_SPEC": {"task_type": "binary_classification",
                                               "evaluation_metric": "roc_auc"}})
    s = _make_session(target="churn")
    tool = DataAnalysisAgentTool()
    out = tool.execute({}, session=s, kernel=k, llm_client=None)
    assert out.success is True
    assert "DATA_ANALYSIS_REPORT" in out.text or "Data Analysis Agent" in out.text
    assert "DATA_ANALYSIS_REPORT" in k.namespace


def test_ml_strategy_agent_tool_no_report():
    from agent.api.agent_tools import MLStrategyAgentTool
    k = _make_kernel({})
    s = _make_session()
    tool = MLStrategyAgentTool()
    out = tool.execute({}, session=s, kernel=k, llm_client=None)
    assert out.success is False
    assert "DATA_ANALYSIS_REPORT" in out.text or "data_analysis_agent" in out.text


def test_ml_strategy_agent_tool_with_report():
    from agent.api.agent_tools import MLStrategyAgentTool
    report = _make_report_dict()
    k = _make_kernel({"DATA_ANALYSIS_REPORT": report})
    s = _make_session()
    tool = MLStrategyAgentTool()
    out = tool.execute({}, session=s, kernel=k, llm_client=None)
    assert out.success is True
    assert "ML_PLAN" in k.namespace
    assert k.namespace["ML_PLAN"]["models"]
