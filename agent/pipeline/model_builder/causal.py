"""CausalInferenceModule — simple ATE estimators.

Implements three methods without external causal libraries:

- ``backdoor``           : outcome regression on [treatment, covariates]; ATE is
                            the treatment coefficient (binary treatment → mean
                            of Ŷ(T=1) − Ŷ(T=0) over the sample).
- ``ipw``                : inverse-propensity weighting with a LogisticRegression
                            propensity model.
- ``doubly_robust``      : combines the two via AIPW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from agent.core.types import ToolResult


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation,
                      math_trace="", error=error)


class CausalInferenceModule:
    """ATE estimation via backdoor / IPW / doubly-robust."""

    def __init__(self, random_state: int = 0,
                 propensity_clip: float = 0.02) -> None:
        self._rs = int(random_state)
        self._clip = float(propensity_clip)

    def estimate_ate(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: str = "backdoor",
    ) -> ToolResult:
        if treatment not in df.columns:
            return _err(f"treatment {treatment!r} missing", "KeyError")
        if outcome not in df.columns:
            return _err(f"outcome {outcome!r} missing", "KeyError")
        T = df[treatment].astype(float).to_numpy()
        if set(np.unique(T)) - {0.0, 1.0}:
            return _err("treatment must be binary 0/1", "ValueError")
        Y = df[outcome].astype(float).to_numpy()
        X = df.drop(columns=[treatment, outcome]).select_dtypes("number").fillna(0.0)

        if method == "backdoor":
            return self._backdoor(X, T, Y)
        if method == "ipw":
            return self._ipw(X, T, Y)
        if method == "doubly_robust":
            return self._dr(X, T, Y)
        return _err(f"unknown method {method!r}", "ValueError")

    # ---- backdoor ---------------------------------------------------- #

    def _backdoor(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> ToolResult:
        rf = RandomForestRegressor(n_estimators=200, random_state=self._rs, n_jobs=1)
        if X.shape[1] == 0:
            # Fall back to T-only regression (mean-difference).
            ate = float(Y[T == 1].mean() - Y[T == 0].mean()) if ((T == 1).any() and (T == 0).any()) else 0.0
            return ToolResult(
                status="ok",
                data={"ate": ate, "method": "backdoor",
                      "n_treated": int((T == 1).sum()),
                      "n_control": int((T == 0).sum())},
                explanation=f"ATE (no covariates) = {ate:.6f}.",
                math_trace="ATE = E[Y|T=1] − E[Y|T=0].",
            )
        X_aug = np.column_stack([T.reshape(-1, 1), X.to_numpy()])
        rf.fit(X_aug, Y)
        X1 = np.column_stack([np.ones_like(T).reshape(-1, 1), X.to_numpy()])
        X0 = np.column_stack([np.zeros_like(T).reshape(-1, 1), X.to_numpy()])
        ate = float((rf.predict(X1) - rf.predict(X0)).mean())
        return ToolResult(
            status="ok",
            data={"ate": ate, "method": "backdoor",
                  "n_treated": int((T == 1).sum()),
                  "n_control": int((T == 0).sum())},
            explanation=f"ATE via backdoor adjustment = {ate:.6f}.",
            math_trace="ATE = E_X[ m̂(X, 1) − m̂(X, 0) ]; m̂ is a RandomForest.",
        )

    # ---- IPW --------------------------------------------------------- #

    def _propensity(self, X: pd.DataFrame, T: np.ndarray) -> np.ndarray:
        if X.shape[1] == 0:
            # No covariates → marginal propensity.
            p = float(T.mean())
            return np.full_like(T, fill_value=max(self._clip, min(1 - self._clip, p)))
        lr = LogisticRegression(max_iter=500, random_state=self._rs)
        lr.fit(X.to_numpy(), T.astype(int))
        e = lr.predict_proba(X.to_numpy())[:, 1]
        return np.clip(e, self._clip, 1 - self._clip)

    def _ipw(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> ToolResult:
        e = self._propensity(X, T)
        w1 = T / e
        w0 = (1 - T) / (1 - e)
        ate = float((w1 * Y).mean() - (w0 * Y).mean())
        return ToolResult(
            status="ok",
            data={"ate": ate, "method": "ipw",
                  "propensity": e.tolist(),
                  "n_treated": int((T == 1).sum()),
                  "n_control": int((T == 0).sum())},
            explanation=f"ATE via IPW = {ate:.6f}.",
            math_trace=("ATE = E[TY/e(X)] − E[(1−T)Y/(1−e(X))]; "
                        f"e clipped to [{self._clip}, 1−{self._clip}]."),
        )

    # ---- Doubly robust ---------------------------------------------- #

    def _dr(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> ToolResult:
        e = self._propensity(X, T)
        if X.shape[1] == 0:
            # No covariates → DR collapses to IPW−style mean difference.
            m1 = np.full_like(Y, Y[T == 1].mean() if (T == 1).any() else 0.0)
            m0 = np.full_like(Y, Y[T == 0].mean() if (T == 0).any() else 0.0)
        else:
            rf1 = RandomForestRegressor(n_estimators=200, random_state=self._rs, n_jobs=1)
            rf0 = RandomForestRegressor(n_estimators=200, random_state=self._rs, n_jobs=1)
            if (T == 1).any():
                rf1.fit(X.to_numpy()[T == 1], Y[T == 1])
                m1 = rf1.predict(X.to_numpy())
            else:
                m1 = np.zeros_like(Y)
            if (T == 0).any():
                rf0.fit(X.to_numpy()[T == 0], Y[T == 0])
                m0 = rf0.predict(X.to_numpy())
            else:
                m0 = np.zeros_like(Y)
        psi = (m1 - m0) + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
        ate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi))) if len(psi) > 1 else 0.0
        return ToolResult(
            status="ok",
            data={"ate": ate, "method": "doubly_robust",
                  "se": se,
                  "ci95": (ate - 1.96 * se, ate + 1.96 * se),
                  "propensity": e.tolist(),
                  "n_treated": int((T == 1).sum()),
                  "n_control": int((T == 0).sum())},
            explanation=f"ATE via doubly robust = {ate:.6f} (SE={se:.4g}).",
            math_trace=("ψ = m̂₁(X) − m̂₀(X) + T(Y − m̂₁)/ê − "
                        "(1−T)(Y − m̂₀)/(1−ê); ATE = E[ψ]."),
        )
