"""TaskDetector — infer an ML task type from the frame and (optional) EDA report.

Heuristics (evaluated in order):

1. No target + datetime-indexed numeric frame → ``timeseries_forecasting``.
2. No target at all                            → ``clustering``.
3. Target boolean / 2 unique values            → ``binary_classification``.
4. Target categorical or int with ≤20 uniques  → ``multiclass_classification``.
5. Target numeric + EDA hint ``time_index``    → ``timeseries_forecasting``.
6. Target numeric                              → ``regression``.
7. EDA hint ``outlier_rate`` > 0.3             → ``anomaly_detection``.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from agent.core.types import ToolResult

TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "timeseries_forecasting",
    "anomaly_detection",
    "clustering",
    "survival_analysis",
    "causal_inference",
]


class TaskDetector:
    """Rule-based task inference."""

    def __init__(self, multiclass_threshold: int = 20) -> None:
        self._mc_threshold = int(multiclass_threshold)

    def infer(
        self,
        df: pd.DataFrame,
        target: str | None,
        eda_report: dict | None = None,
    ) -> ToolResult:
        hints = eda_report or {}
        reasons: list[str] = []

        def _ok(task: TaskType, reason: str) -> ToolResult:
            reasons.append(reason)
            return ToolResult(
                status="ok",
                data={"task": task, "reasons": reasons, "target": target},
                explanation=f"Inferred task: {task}.",
                math_trace="Rule-based dispatch on (target dtype, cardinality, EDA hints).",
            )

        # Explicit EDA override
        if hints.get("survival"):
            return _ok("survival_analysis", "EDA hint: survival columns present.")
        if hints.get("causal"):
            return _ok("causal_inference", "EDA hint: treatment/outcome present.")

        if target is None or target not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex) and df.select_dtypes("number").shape[1] > 0:
                return _ok("timeseries_forecasting", "No target + datetime index.")
            if hints.get("outlier_rate", 0.0) > 0.3:
                return _ok("anomaly_detection", "EDA hint: outlier_rate > 0.3.")
            return _ok("clustering", "No target provided.")

        y = df[target]
        non_null = y.dropna()
        if pd.api.types.is_bool_dtype(y) or non_null.nunique() == 2:
            return _ok("binary_classification", "Target has 2 unique values.")
        if y.dtype == object or str(y.dtype) == "category":
            return _ok("multiclass_classification", "Target is categorical.")
        if pd.api.types.is_integer_dtype(y) and non_null.nunique() <= self._mc_threshold:
            return _ok("multiclass_classification",
                       f"Integer target with ≤{self._mc_threshold} uniques.")
        # Float columns that are actually integer categories (happen when column has NaN)
        if pd.api.types.is_float_dtype(y) and len(non_null) > 0:
            rounded = non_null.round()
            if (non_null == rounded).all() and non_null.nunique() <= self._mc_threshold:
                return _ok(
                    "multiclass_classification" if non_null.nunique() > 2 else "binary_classification",
                    f"Float column with integer-only values and ≤{self._mc_threshold} unique classes.",
                )
        if pd.api.types.is_numeric_dtype(y):
            if isinstance(df.index, pd.DatetimeIndex) or hints.get("time_index"):
                return _ok("timeseries_forecasting", "Numeric target + time index.")
            if hints.get("outlier_rate", 0.0) > 0.3:
                return _ok("anomaly_detection", "Numeric target + high outlier rate.")
            return _ok("regression", "Numeric target.")
        return ToolResult(
            status="error", data=None,
            explanation=f"Cannot infer task for target of dtype {y.dtype}.",
            math_trace="", error="UnsupportedTargetDType",
        )
