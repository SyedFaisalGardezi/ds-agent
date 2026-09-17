"""PlotTool — thin wrapper over Visualiser."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.eda.visualiser import Visualiser


def _err(msg: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=msg,
                      math_trace="", error=error)


_CHART_TYPES = {"distribution", "correlation", "timeseries"}


class PlotTool:
    name: str = "plot_tool"
    description: str = (
        "Renders matplotlib plots and writes PNGs under the run output dir. "
        "Input: JSON with 'records' (list[dict]), 'chart_type' "
        f"({'|'.join(sorted(_CHART_TYPES))}), optional 'run_id', "
        "optional 'time_column', 'value_column' (for timeseries)."
    )
    task_type: str = "code"

    def __init__(self, visualiser: Visualiser | None = None) -> None:
        self._vis = visualiser or Visualiser()

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")

        records = payload.get("records")
        if not isinstance(records, list):
            return _err("'records' must be a list.", "ValueError")
        chart_type = (payload.get("chart_type") or "").lower()
        if chart_type not in _CHART_TYPES:
            return _err(
                f"'chart_type' must be one of {sorted(_CHART_TYPES)}.",
                "ValueError",
            )
        run_id = payload.get("run_id", "adhoc")

        try:
            df = pd.DataFrame(records)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not build DataFrame: {exc}", type(exc).__name__)

        try:
            if chart_type == "distribution":
                return self._vis.distribution_plots(df, run_id)
            if chart_type == "correlation":
                return self._vis.correlation_heatmap(df, run_id)
            # timeseries
            value_col = payload.get("value_column")
            if not value_col or value_col not in df.columns:
                return _err(
                    "timeseries requires 'value_column' present in records.",
                    "ValueError",
                )
            time_col = payload.get("time_column")
            if time_col and time_col in df.columns:
                series = pd.Series(
                    df[value_col].values,
                    index=pd.to_datetime(df[time_col], errors="coerce"),
                    name=value_col,
                )
            else:
                series = df[value_col]
            return self._vis.time_series_plot(series, run_id)
        except Exception as exc:  # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}", type(exc).__name__)

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result), default=str)
