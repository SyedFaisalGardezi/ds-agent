"""CalibrationAnalyser — reliability diagrams, Platt, isotonic calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from agent.core.types import ToolResult


class CalibrationAnalyser:
    def reliability_diagram(self, y_true, y_prob, n_bins: int = 10) -> ToolResult:
        yt = np.asarray(y_true).astype(int)
        yp = np.asarray(y_prob).astype(float)
        if yt.shape != yp.shape:
            return ToolResult(status="error", data=None,
                              explanation="shape mismatch",
                              math_trace="", error="ValueError")
        if n_bins < 2:
            return ToolResult(status="error", data=None,
                              explanation="n_bins must be ≥ 2",
                              math_trace="", error="ValueError")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins - 1)
        bin_mean_p: list[float] = []
        bin_frac_pos: list[float] = []
        bin_counts: list[int] = []
        ece = 0.0
        n = len(yt)
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                bin_mean_p.append(float((edges[b] + edges[b + 1]) / 2))
                bin_frac_pos.append(0.0)
                bin_counts.append(0)
                continue
            mp = float(yp[mask].mean())
            fp = float(yt[mask].mean())
            cnt = int(mask.sum())
            bin_mean_p.append(mp); bin_frac_pos.append(fp); bin_counts.append(cnt)
            ece += (cnt / n) * abs(fp - mp)
        brier = float(np.mean((yp - yt) ** 2))
        return ToolResult(
            status="ok",
            data={"bin_mean_prob": bin_mean_p, "bin_frac_pos": bin_frac_pos,
                  "bin_counts": bin_counts, "ece": float(ece),
                  "brier": brier, "n_bins": n_bins},
            explanation=f"Reliability: ECE={ece:.4f}, Brier={brier:.4f}.",
            math_trace=("ECE = Σ_b (|B_b|/N)·|acc(B_b) − conf(B_b)|; "
                        "Brier = E[(p − y)²]."),
        )

    def platt_calibrate(self, model, X_val: pd.DataFrame, y_val) -> ToolResult:
        X = np.asarray(X_val if not isinstance(X_val, pd.DataFrame)
                       else X_val.select_dtypes("number").fillna(0.0).to_numpy())
        y = np.asarray(y_val).astype(int)
        if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
            return ToolResult(status="error", data=None,
                              explanation="model lacks predict_proba / decision_function",
                              math_trace="", error="AttributeError")
        try:
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(X, y)
        except Exception:  # noqa: BLE001
            # fallback: fit 1-D logistic on raw predict_proba
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
            else:
                raw = model.decision_function(X).reshape(-1, 1)
            lr = LogisticRegression().fit(raw, y)
            calibrated = ("1d-platt", lr)
        return ToolResult(
            status="ok",
            data={"calibrated_model": calibrated, "method": "platt"},
            explanation="Platt scaling fit on validation set.",
            math_trace="p_cal = 1 / (1 + exp(A·f(x) + B)); A, B via logistic MLE.",
        )

    def isotonic_calibrate(self, model, X_val: pd.DataFrame, y_val) -> ToolResult:
        X = np.asarray(X_val if not isinstance(X_val, pd.DataFrame)
                       else X_val.select_dtypes("number").fillna(0.0).to_numpy())
        y = np.asarray(y_val).astype(int)
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
        else:
            return ToolResult(status="error", data=None,
                              explanation="model lacks predict_proba / decision_function",
                              math_trace="", error="AttributeError")
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        return ToolResult(
            status="ok",
            data={"calibrator": iso, "method": "isotonic"},
            explanation="Isotonic regression calibrator fit on validation set.",
            math_trace="m̂(f) = argmin_{g non-decreasing} Σ (y_i − g(f_i))².",
        )
