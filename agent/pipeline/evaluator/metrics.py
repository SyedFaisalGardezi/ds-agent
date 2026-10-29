"""MetricsCalculator — task-aware metric suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from agent.core.types import ToolResult
from agent.pipeline.model_builder.task_detector import TaskType


class MetricsCalculator:
    """Compute metrics appropriate to the task type."""

    def compute(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        task: TaskType,
        y_prob: pd.Series | np.ndarray | None = None,
    ) -> ToolResult:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        if yt.shape[0] != yp.shape[0]:
            return ToolResult(status="error", data=None,
                              explanation="length mismatch",
                              math_trace="", error="ValueError")
        if yt.size == 0:
            return ToolResult(status="error", data=None,
                              explanation="empty inputs",
                              math_trace="", error="ValueError")

        m: dict[str, float] = {}
        math_trace = ""

        if task == "binary_classification":
            m["accuracy"] = float(accuracy_score(yt, yp))
            m["precision"] = float(precision_score(yt, yp, zero_division=0))
            m["recall"] = float(recall_score(yt, yp, zero_division=0))
            m["f1"] = float(f1_score(yt, yp, zero_division=0))
            if y_prob is not None:
                prob = np.asarray(y_prob)
                try:
                    m["roc_auc"] = float(roc_auc_score(yt, prob))
                    m["pr_auc"] = float(average_precision_score(yt, prob))
                    m["log_loss"] = float(log_loss(yt, np.clip(prob, 1e-7, 1 - 1e-7)))
                except Exception:
                    pass
            math_trace = "P, R, F1 = 2PR/(P+R); ROC-AUC integrates TPR over FPR."
        elif task == "multiclass_classification":
            m["accuracy"] = float(accuracy_score(yt, yp))
            m["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
            m["f1_weighted"] = float(f1_score(yt, yp, average="weighted", zero_division=0))
            math_trace = "macro-F1 = (1/K)·Σ_k F1_k; weighted uses class support."
        elif task in ("regression", "timeseries_forecasting"):
            m["mae"] = float(mean_absolute_error(yt, yp))
            m["rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
            m["r2"] = float(r2_score(yt, yp))
            if not (yt == 0).any():
                m["mape"] = float(mean_absolute_percentage_error(yt, yp))
            math_trace = "MAE=E|e|, RMSE=√E[e²], R²=1 − SSR/SST."
        elif task == "anomaly_detection":
            m["accuracy"] = float(accuracy_score(yt, yp))
            m["f1"] = float(f1_score(yt, yp, zero_division=0))
            math_trace = "accuracy and F1 on anomaly labels."
        elif task == "clustering":
            from sklearn.metrics import adjusted_rand_score
            try:
                m["ari"] = float(adjusted_rand_score(yt, yp))
            except Exception:
                pass
            math_trace = "ARI adjusted for chance."
        else:
            return ToolResult(status="error", data=None,
                              explanation=f"Unsupported task {task!r}",
                              math_trace="", error="UnsupportedTask")

        return ToolResult(
            status="ok",
            data={"metrics": m, "task": task, "n": int(yt.size)},
            explanation=f"Computed {len(m)} metric(s) for {task}.",
            math_trace=math_trace,
        )
