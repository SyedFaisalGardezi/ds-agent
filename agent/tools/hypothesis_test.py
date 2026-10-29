"""HypothesisTestTool — dispatch StatisticsAnalyser methods.

Supported tests:
  - normality        (Shapiro-Wilk over numeric columns)
  - adf              (Augmented Dickey-Fuller stationarity)
  - kpss             (KPSS trend-stationarity)
  - cramers_v        (categorical association, needs col_a, col_b)
  - spearman         (rank correlation matrix)
  - mutual_information (needs 'target')
  - stl              (STL decomposition, needs 'period')
  - acf_pacf         (needs optional 'lags')
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.eda.statistics import StatisticsAnalyser


def _err(msg: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=msg,
                      math_trace="", error=error)


_FRAME_TESTS = {"normality", "spearman", "mutual_information", "cramers_v"}
_SERIES_TESTS = {"adf", "kpss", "stl", "acf_pacf"}


class HypothesisTestTool:
    name: str = "hypothesis_test"
    description: str = (
        "Runs hypothesis tests via StatisticsAnalyser. "
        "Input: JSON with 'test', 'records' (list[dict] for frame tests; "
        "list[float] for series tests), 'params' (optional)."
    )
    task_type: str = "math"

    def __init__(self, analyser: StatisticsAnalyser | None = None) -> None:
        self._an = analyser or StatisticsAnalyser()

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")

        test = (payload.get("test") or "").lower()
        records = payload.get("records")
        params = payload.get("params") or {}

        if test in _FRAME_TESTS:
            if not isinstance(records, list):
                return _err("'records' must be a list of row dicts.", "ValueError")
            try:
                df = pd.DataFrame(records)
            except Exception as exc:  # noqa: BLE001
                return _err(f"DataFrame error: {exc}", type(exc).__name__)
            if test == "normality":
                return self._an.normality_test(df)
            if test == "spearman":
                return self._an.spearman_correlation(df)
            if test == "mutual_information":
                target = params.get("target")
                if not target:
                    return _err("'target' required in params.", "ValueError")
                return self._an.mutual_information(df, target)
            if test == "cramers_v":
                a, b = params.get("col_a"), params.get("col_b")
                if not a or not b:
                    return _err("'col_a' and 'col_b' required.", "ValueError")
                return self._an.cramers_v(df, a, b)

        if test in _SERIES_TESTS:
            if not isinstance(records, list):
                return _err("'records' must be a list of floats.", "ValueError")
            try:
                series = pd.Series(records, dtype=float)
            except Exception as exc:  # noqa: BLE001
                return _err(f"Series error: {exc}", type(exc).__name__)
            if test == "adf":
                return self._an.adf_test(series)
            if test == "kpss":
                return self._an.kpss_test(series)
            if test == "stl":
                period = int(params.get("period", 0))
                if period < 2:
                    return _err("'period' (>=2) required in params.", "ValueError")
                return self._an.stl_decompose(series, period)
            if test == "acf_pacf":
                return self._an.acf_pacf(series, int(params.get("lags", 40)))

        return _err(
            f"Unknown test {test!r}. "
            f"Supported: {sorted(_FRAME_TESTS | _SERIES_TESTS)}.",
            "ValueError",
        )

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result), default=str)
