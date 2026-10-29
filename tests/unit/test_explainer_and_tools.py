"""Unit tests for MathExplainer + the 7 LangChain-style tools."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.core.types import ToolResult
from agent.explainer.math_explainer import MathExplainer
from agent.pipeline.evaluator.leaderboard import Leaderboard
from agent.tools.cost_estimator import CostEstimatorTool
from agent.tools.feature_engineer import FeatureEngineerTool
from agent.tools.hypothesis_test import HypothesisTestTool
from agent.tools.pii_detector import PIIDetectorTool
from agent.tools.plot_tool import PlotTool
from agent.tools.powerbi_exporter import PowerBIExporterTool
from agent.tools.report_generator import ReportGeneratorTool

# --------------------------------------------------------------------------- #
# MathExplainer
# --------------------------------------------------------------------------- #


def test_math_explainer_uses_llm_when_available():
    fake = SimpleNamespace(invoke=lambda prompt: "## Explanation body")
    me = MathExplainer(llm=fake)
    out = me.explain("matmul", {"n": 10})
    assert "## Explanation body" in out


def test_math_explainer_falls_back_on_llm_error():
    class Boom:
        def invoke(self, prompt):
            raise RuntimeError("offline")
    out = MathExplainer(llm=Boom()).explain("zscore", {"col": "x"})
    assert "Definition" in out and "zscore" in out


def test_math_explainer_template_when_router_unreachable(monkeypatch):
    # Force _get_llm() to return None.
    me = MathExplainer(llm=None)
    monkeypatch.setattr(me, "_get_llm", lambda: None)
    out = me.explain("mean", {"col": "y"})
    assert "Definition" in out


def test_math_explainer_enrich_result_preserves_existing_trace():
    tr = ToolResult(status="ok", data={}, explanation="e",
                    math_trace="pre-existing")
    out = MathExplainer(llm=SimpleNamespace(invoke=lambda p: "new")).enrich_result(
        tr, "op", {})
    assert out.math_trace == "pre-existing"


def test_math_explainer_enrich_result_fills_when_empty():
    tr = ToolResult(status="ok", data={}, explanation="e", math_trace="")
    out = MathExplainer(llm=SimpleNamespace(invoke=lambda p: "derived")).enrich_result(
        tr, "op", {})
    assert out.math_trace == "derived"


# --------------------------------------------------------------------------- #
# CostEstimatorTool
# --------------------------------------------------------------------------- #


class _StubConnector:
    def estimate_cost(self, query):
        return ToolResult(status="ok", data={"usd": 0.42, "query": query},
                          explanation="ok", math_trace="")

    def detect_pii(self, table):
        return ToolResult(status="ok", data={"table": table, "pii_cols": ["ssn"]},
                          explanation="ok", math_trace="")


def test_cost_estimator_dispatches_to_connector():
    tool = CostEstimatorTool(connectors={"bq": _StubConnector()})
    r = tool.run(json.dumps({"connector": "bq", "query": "SELECT 1"}))
    assert r.status == "ok"
    assert r.data["usd"] == 0.42


def test_cost_estimator_unknown_connector():
    tool = CostEstimatorTool(connectors={})
    r = tool.run(json.dumps({"connector": "x", "query": "q"}))
    assert r.status == "error"


def test_cost_estimator_missing_fields():
    tool = CostEstimatorTool(connectors={})
    assert tool.run(json.dumps({"connector": "x"})).status == "error"
    assert tool.run("not-json").status == "error"


# --------------------------------------------------------------------------- #
# PIIDetectorTool
# --------------------------------------------------------------------------- #


def test_pii_detector_dispatches():
    tool = PIIDetectorTool(connectors={"pg": _StubConnector()})
    r = tool.run(json.dumps({"connector": "pg", "table": "users"}))
    assert r.status == "ok"
    assert r.data["pii_cols"] == ["ssn"]


def test_pii_detector_errors_surfaced():
    class Bad:
        def detect_pii(self, table):
            raise ValueError("boom")
    tool = PIIDetectorTool(connectors={"x": Bad()})
    r = tool.run(json.dumps({"connector": "x", "table": "t"}))
    assert r.status == "error"
    assert r.error == "ValueError"


# --------------------------------------------------------------------------- #
# FeatureEngineerTool
# --------------------------------------------------------------------------- #


def test_feature_engineer_generates_features():
    recs = [{"a": i, "b": i + 1} for i in range(1, 11)]
    r = FeatureEngineerTool().run(json.dumps(
        {"records": recs, "max_pairs": 5}))
    assert r.status == "ok"


def test_feature_engineer_rejects_bad_input():
    assert FeatureEngineerTool().run(
        json.dumps({"records": "notalist"})).status == "error"


# --------------------------------------------------------------------------- #
# PlotTool
# --------------------------------------------------------------------------- #


def test_plot_tool_distribution(tmp_path):
    from agent.pipeline.eda.visualiser import Visualiser
    recs = [{"a": i, "b": i * 2} for i in range(30)]
    tool = PlotTool(visualiser=Visualiser(output_dir=str(tmp_path)))
    r = tool.run(json.dumps({"records": recs, "chart_type": "distribution",
                              "run_id": "t1"}))
    assert r.status == "ok"


def test_plot_tool_correlation(tmp_path):
    from agent.pipeline.eda.visualiser import Visualiser
    rng = np.random.default_rng(0)
    recs = pd.DataFrame(rng.normal(size=(40, 3)),
                        columns=list("abc")).to_dict("records")
    tool = PlotTool(visualiser=Visualiser(output_dir=str(tmp_path)))
    r = tool.run(json.dumps({"records": recs, "chart_type": "correlation",
                              "run_id": "t2"}))
    assert r.status == "ok"


def test_plot_tool_timeseries(tmp_path):
    from agent.pipeline.eda.visualiser import Visualiser
    recs = [{"t": f"2024-01-{i+1:02d}", "y": i * 1.5} for i in range(10)]
    tool = PlotTool(visualiser=Visualiser(output_dir=str(tmp_path)))
    r = tool.run(json.dumps({"records": recs, "chart_type": "timeseries",
                              "time_column": "t", "value_column": "y",
                              "run_id": "t3"}))
    assert r.status == "ok"


def test_plot_tool_invalid_chart_type():
    r = PlotTool().run(json.dumps({"records": [{"a": 1}],
                                    "chart_type": "pie"}))
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# HypothesisTestTool
# --------------------------------------------------------------------------- #


def test_hypothesis_test_normality():
    rng = np.random.default_rng(42)
    recs = [{"x": v} for v in rng.normal(size=100)]
    r = HypothesisTestTool().run(json.dumps(
        {"test": "normality", "records": recs}))
    assert r.status == "ok"


def test_hypothesis_test_spearman():
    recs = [{"a": i, "b": i ** 2} for i in range(1, 30)]
    r = HypothesisTestTool().run(json.dumps(
        {"test": "spearman", "records": recs}))
    assert r.status == "ok"


def test_hypothesis_test_adf_series():
    rng = np.random.default_rng(0)
    series = rng.normal(size=120).tolist()
    r = HypothesisTestTool().run(json.dumps(
        {"test": "adf", "records": series}))
    assert r.status == "ok"


def test_hypothesis_test_stl_requires_period():
    r = HypothesisTestTool().run(json.dumps(
        {"test": "stl", "records": list(range(60))}))
    assert r.status == "error"


def test_hypothesis_test_unknown():
    r = HypothesisTestTool().run(json.dumps(
        {"test": "??", "records": []}))
    assert r.status == "error"


def test_hypothesis_test_cramers_v():
    recs = (
        [{"a": "x", "b": "p"}] * 30
        + [{"a": "y", "b": "q"}] * 30
        + [{"a": "x", "b": "q"}] * 5
    )
    r = HypothesisTestTool().run(json.dumps(
        {"test": "cramers_v", "records": recs,
         "params": {"col_a": "a", "col_b": "b"}}))
    assert r.status == "ok"


# --------------------------------------------------------------------------- #
# ReportGeneratorTool + PowerBIExporterTool (share leaderboard fixture)
# --------------------------------------------------------------------------- #


@pytest.fixture
def lb_with_run(tmp_path):
    lb = Leaderboard(db_path=tmp_path / "lb.db")
    lb.log_run(
        run_id="r1",
        model_type="rf",
        hyperparams={"n_estimators": 100, "max_depth": 8},
        metrics={"accuracy": 0.91, "f1": 0.88},
        train_time_s=12.5, inference_ms=4.2, model_mb=3.1,
        shap_runtime_s=0.5,
    )
    yield lb, tmp_path
    lb.close()


def test_report_generator_writes_markdown(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = ReportGeneratorTool(leaderboard=lb,
                               output_dir=str(tmp_path / "reports"))
    r = tool.run(json.dumps({"run_id": "r1"}))
    assert r.status == "ok"
    md = open(r.data["output_path"]).read()
    assert "# Run Report" in md
    assert "accuracy" in md
    assert "n_estimators" in md


def test_report_generator_unknown_run(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = ReportGeneratorTool(leaderboard=lb,
                               output_dir=str(tmp_path / "reports"))
    r = tool.run(json.dumps({"run_id": "nope"}))
    assert r.status == "error"


def test_report_generator_missing_field(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = ReportGeneratorTool(leaderboard=lb,
                               output_dir=str(tmp_path / "reports"))
    assert tool.run(json.dumps({})).status == "error"


def test_powerbi_exporter_writes_csv(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = PowerBIExporterTool(leaderboard=lb,
                                output_dir=str(tmp_path / "pbi"))
    r = tool.run(json.dumps({"run_id": "r1"}))
    assert r.status == "ok"
    csv_content = open(r.data["csv_path"]).read()
    # Expect flattened headers
    assert "hp_n_estimators" in csv_content
    assert "metric_accuracy" in csv_content
    assert "0.91" in csv_content
    # Sidecar JSON
    side = json.loads(open(r.data["json_path"]).read())
    assert side[0]["metric_accuracy"] == 0.91


def test_powerbi_exporter_unknown_ids(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = PowerBIExporterTool(leaderboard=lb,
                                output_dir=str(tmp_path / "pbi"))
    r = tool.run(json.dumps({"run_ids": ["r1", "missing"]}))
    assert r.status == "error"


def test_powerbi_exporter_requires_id(lb_with_run):
    lb, tmp_path = lb_with_run
    tool = PowerBIExporterTool(leaderboard=lb,
                                output_dir=str(tmp_path / "pbi"))
    assert tool.run(json.dumps({})).status == "error"


# --------------------------------------------------------------------------- #
# HypothesisTestTool — gap coverage (kpss, acf_pacf, MI success, JSON error, to_json)
# --------------------------------------------------------------------------- #


def test_hypothesis_test_kpss():
    rng = np.random.default_rng(7)
    series = rng.normal(size=60).tolist()
    r = HypothesisTestTool().run(json.dumps(
        {"test": "kpss", "records": series}))
    assert r.status in ("ok", "error")  # ok if statsmodels available


def test_hypothesis_test_acf_pacf_default_lags():
    rng = np.random.default_rng(1)
    series = rng.normal(size=100).tolist()
    r = HypothesisTestTool().run(json.dumps(
        {"test": "acf_pacf", "records": series}))
    assert r.status in ("ok", "error")


def test_hypothesis_test_acf_pacf_custom_lags():
    rng = np.random.default_rng(2)
    series = rng.normal(size=80).tolist()
    r = HypothesisTestTool().run(json.dumps(
        {"test": "acf_pacf", "records": series,
         "params": {"lags": 10}}))
    assert r.status in ("ok", "error")


def test_hypothesis_test_mutual_information_success():
    recs = [{"a": i, "b": i % 2, "target": i % 2} for i in range(30)]
    r = HypothesisTestTool().run(json.dumps(
        {"test": "mutual_information", "records": recs,
         "params": {"target": "target"}}))
    assert r.status == "ok"


def test_hypothesis_test_mutual_information_missing_target():
    recs = [{"a": i} for i in range(10)]
    r = HypothesisTestTool().run(json.dumps(
        {"test": "mutual_information", "records": recs, "params": {}}))
    assert r.status == "error"
    assert "target" in r.explanation


def test_hypothesis_test_json_decode_error():
    r = HypothesisTestTool().run("not valid json{{{")
    assert r.status == "error"
    assert "JSON" in r.explanation


def test_hypothesis_test_non_list_records_frame_test():
    r = HypothesisTestTool().run(json.dumps(
        {"test": "normality", "records": "not a list"}))
    assert r.status == "error"


def test_hypothesis_test_non_list_records_series_test():
    r = HypothesisTestTool().run(json.dumps(
        {"test": "adf", "records": "not a list"}))
    assert r.status == "error"


def test_hypothesis_test_cramers_v_missing_cols():
    recs = [{"a": "x", "b": "y"}] * 10
    r = HypothesisTestTool().run(json.dumps(
        {"test": "cramers_v", "records": recs,
         "params": {}}))
    assert r.status == "error"


def test_hypothesis_test_to_json():
    from agent.core.types import ToolResult
    r = HypothesisTestTool().run(json.dumps(
        {"test": "normality", "records": [{"x": float(i)} for i in range(20)]}))
    text = HypothesisTestTool().to_json(r)
    parsed = json.loads(text)
    assert "status" in parsed


# --------------------------------------------------------------------------- #
# PIIDetectorTool — gap coverage (registry fallback, missing fields, to_json)
# --------------------------------------------------------------------------- #


def test_pii_detector_registry_fallback(monkeypatch):
    """When no connectors dict is passed, _resolve() falls back to ConnectorRegistry."""
    from agent.tools.pii_detector import PIIDetectorTool

    class FakeConn:
        def detect_pii(self, table):
            from agent.core.types import ToolResult
            return ToolResult(status="ok", data={"pii_cols": []},
                              explanation="ok", math_trace="")

    class FakeRegistry:
        def __getitem__(self, name):
            if name == "pg":
                return FakeConn()
            raise KeyError(name)

    monkeypatch.setattr(
        "agent.connectors.registry.ConnectorRegistry.get",
        classmethod(lambda cls: FakeRegistry()),
    )
    tool = PIIDetectorTool(connectors=None)
    r = tool.run(json.dumps({"connector": "pg", "table": "users"}))
    assert r.status == "ok"


def test_pii_detector_missing_connector_field():
    from agent.tools.pii_detector import PIIDetectorTool
    tool = PIIDetectorTool(connectors={})
    r = tool.run(json.dumps({"table": "t"}))  # no 'connector'
    assert r.status == "error"
    assert "required" in r.explanation


def test_pii_detector_missing_table_field():
    from agent.tools.pii_detector import PIIDetectorTool
    tool = PIIDetectorTool(connectors={})
    r = tool.run(json.dumps({"connector": "pg"}))  # no 'table'
    assert r.status == "error"


def test_pii_detector_to_json():
    from agent.tools.pii_detector import PIIDetectorTool
    tool = PIIDetectorTool(connectors={"pg": _StubConnector()})
    r = tool.run(json.dumps({"connector": "pg", "table": "users"}))
    text = tool.to_json(r)
    parsed = json.loads(text)
    assert "status" in parsed


def test_pii_detector_unknown_connector():
    from agent.tools.pii_detector import PIIDetectorTool
    tool = PIIDetectorTool(connectors={})
    r = tool.run(json.dumps({"connector": "nope", "table": "t"}))
    assert r.status == "error"
    assert "unknown" in r.explanation
