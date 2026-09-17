"""ModelExplainer — SHAP / LIME-style / PDP / ICE (shap optional)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from agent.core.types import ToolResult


class ModelExplainer:
    def __init__(self, random_state: int = 0) -> None:
        self._rs = int(random_state)

    def shap_values(self, model, df: pd.DataFrame) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if X.shape[1] == 0:
            return ToolResult(status="error", data=None,
                              explanation="no numeric features",
                              math_trace="", error="ValueError")
        try:
            import shap  # type: ignore
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X.to_numpy())
            if isinstance(vals, list):
                arr = np.stack([np.asarray(v) for v in vals], axis=0).mean(axis=0)
            else:
                arr = np.asarray(vals)
            # Newer shap may return (n_samples, n_features, n_classes);
            # collapse the class axis so we end up with (n_samples, n_features).
            if arr.ndim == 3:
                arr = np.abs(arr).mean(axis=-1)
            mean_abs = np.abs(arr).mean(axis=0)
            method = "shap"
        except Exception:
            try:
                pred = model.predict(X.to_numpy())
            except Exception as exc:  # noqa: BLE001
                return ToolResult(status="error", data=None,
                                  explanation=f"model.predict failed: {exc}",
                                  math_trace="", error=type(exc).__name__)
            pi = permutation_importance(model, X.to_numpy(), pred,
                                        n_repeats=5, random_state=self._rs, n_jobs=1)
            mean_abs = pi.importances_mean
            method = "permutation_importance"
        ranking = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
        return ToolResult(
            status="ok",
            data={"mean_abs_importance": ranking.to_dict(), "method": method,
                  "top": ranking.index[0]},
            explanation=f"Computed global importance via {method} for {X.shape[1]} feature(s).",
            math_trace="E[|φ_j|] (SHAP) or E[Δloss on π_j] (permutation).",
        )

    def lime_explain(self, model, df: pd.DataFrame, row_idx: int,
                     n_samples: int = 500, sigma: float = 0.25) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if not 0 <= row_idx < len(X):
            return ToolResult(status="error", data=None,
                              explanation="row_idx out of range",
                              math_trace="", error="IndexError")
        x0 = X.iloc[row_idx].to_numpy()
        rng = np.random.default_rng(self._rs)
        std = X.std(ddof=0).to_numpy() + 1e-9
        samples = x0 + rng.normal(0, sigma, size=(n_samples, X.shape[1])) * std
        try:
            y_local = model.predict(samples)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"model.predict failed: {exc}",
                              math_trace="", error=type(exc).__name__)
        dists = np.linalg.norm((samples - x0) / std, axis=1)
        w = np.exp(-(dists ** 2))
        surrogate = Ridge(alpha=1.0, random_state=self._rs)
        surrogate.fit(samples, y_local, sample_weight=w)
        coefs = pd.Series(surrogate.coef_, index=X.columns)
        return ToolResult(
            status="ok",
            data={"coefficients": coefs.to_dict(),
                  "intercept": float(surrogate.intercept_),
                  "row_idx": int(row_idx), "n_samples": int(n_samples)},
            explanation=f"LIME linear surrogate for row {row_idx} ({n_samples} perturbations).",
            math_trace="g(z) = βᵀz; β = argmin Σ w(z)·(f(z) − g(z))² + α‖β‖².",
        )

    def _grid(self, x: pd.Series, n: int = 50) -> np.ndarray:
        x_num = x.dropna().astype(float).to_numpy()
        if x_num.size == 0:
            return np.array([])
        lo, hi = float(np.min(x_num)), float(np.max(x_num))
        if lo == hi:
            return np.array([lo])
        return np.linspace(lo, hi, n)

    def pdp(self, model, df: pd.DataFrame, feature: str,
            n_grid: int = 50) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if feature not in X.columns:
            return ToolResult(status="error", data=None,
                              explanation=f"{feature!r} not numeric or missing",
                              math_trace="", error="KeyError")
        grid = self._grid(X[feature], n_grid)
        if grid.size == 0:
            return ToolResult(status="error", data=None,
                              explanation="empty feature",
                              math_trace="", error="ValueError")
        X_arr = X.to_numpy()
        col = list(X.columns).index(feature)
        pdp_vals: list[float] = []
        for v in grid:
            Xm = X_arr.copy()
            Xm[:, col] = v
            pdp_vals.append(float(model.predict(Xm).mean()))
        return ToolResult(
            status="ok",
            data={"grid": grid.tolist(), "pdp": pdp_vals, "feature": feature},
            explanation=f"PDP for {feature!r} over {len(grid)} grid point(s).",
            math_trace="PDP_j(v) = E_{X_{−j}}[ f(v, X_{−j}) ].",
        )

    def ice(self, model, df: pd.DataFrame, feature: str,
            n_grid: int = 50, n_rows: int | None = None) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if feature not in X.columns:
            return ToolResult(status="error", data=None,
                              explanation=f"{feature!r} not numeric or missing",
                              math_trace="", error="KeyError")
        grid = self._grid(X[feature], n_grid)
        if grid.size == 0:
            return ToolResult(status="error", data=None,
                              explanation="empty feature",
                              math_trace="", error="ValueError")
        if n_rows is not None:
            X = X.sample(n=min(n_rows, len(X)), random_state=self._rs)
        X_arr = X.to_numpy()
        col = list(X.columns).index(feature)
        curves = np.zeros((X_arr.shape[0], grid.size))
        for i, v in enumerate(grid):
            Xm = X_arr.copy()
            Xm[:, col] = v
            curves[:, i] = model.predict(Xm)
        return ToolResult(
            status="ok",
            data={"grid": grid.tolist(), "curves": curves.tolist(),
                  "feature": feature, "n_rows": int(X_arr.shape[0])},
            explanation=f"ICE for {feature!r}, {X_arr.shape[0]} row(s) × {grid.size} grid point(s).",
            math_trace="ICE_i(v) = f(v, x_{i,−j}).",
        )
