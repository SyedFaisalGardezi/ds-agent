"""FeatureSelector — wraps several selection strategies.

- ``rfe``          : sklearn Recursive Feature Elimination over a linear model.
- ``boruta``       : minimal shadow-feature Boruta implementation using
                     RandomForest importances (no external ``boruta_py``).
- ``shap_selection`` : permutation-importance based global ranking. If the
                     optional ``shap`` package is installed we use TreeSHAP;
                     otherwise we fall back to sklearn ``permutation_importance``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge

from agent.core.types import ToolResult


def _split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, bool]:
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in frame")
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes("number").fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("no numeric features available")
    is_classif = (y.dtype == object) or str(y.dtype) == "category" or y.nunique() <= 20
    return X, y, is_classif


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation,
                      math_trace="", error=error)


class FeatureSelector:
    """Unified entry-point for RFE / Boruta / permutation importance."""

    def __init__(self, random_state: int = 0) -> None:
        self._rs = int(random_state)

    # ---- RFE --------------------------------------------------------- #

    def rfe(self, df: pd.DataFrame, target: str, n_features: int) -> ToolResult:
        if n_features < 1:
            return _err("n_features must be ≥ 1", "ValueError")
        try:
            X, y, is_classif = _split_xy(df, target)
        except (KeyError, ValueError) as exc:
            return _err(str(exc), type(exc).__name__)
        n_features = min(n_features, X.shape[1])
        estimator = (LogisticRegression(max_iter=500, random_state=self._rs)
                     if is_classif else Ridge(random_state=self._rs))
        y_fit = pd.factorize(y)[0] if is_classif else y.astype(float).to_numpy()
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X.to_numpy(), y_fit)
        selected = [c for c, keep in zip(X.columns, rfe.support_) if keep]
        ranks = dict(zip(X.columns, rfe.ranking_.tolist()))
        return ToolResult(
            status="ok",
            data={"selected": selected, "ranking": ranks,
                  "n_features": n_features, "is_classif": is_classif},
            explanation=f"RFE kept {len(selected)} / {X.shape[1]} feature(s).",
            math_trace=(
                "Iteratively drop feature with smallest |coef| from "
                "the fitted linear model until k remain."
            ),
        )

    # ---- Boruta ------------------------------------------------------ #

    def boruta(self, df: pd.DataFrame, target: str,
               n_iter: int = 20, alpha: float = 0.05) -> ToolResult:
        if n_iter < 5:
            return _err("n_iter must be ≥ 5", "ValueError")
        try:
            X, y, is_classif = _split_xy(df, target)
        except (KeyError, ValueError) as exc:
            return _err(str(exc), type(exc).__name__)
        rng = np.random.default_rng(self._rs)
        features = list(X.columns)
        y_fit = pd.factorize(y)[0] if is_classif else y.astype(float).to_numpy()
        hits = np.zeros(len(features), dtype=int)

        for _ in range(n_iter):
            shadow = X.apply(lambda col: col.sample(frac=1, random_state=rng.integers(1 << 31)).to_numpy())
            shadow.columns = [f"shadow_{c}" for c in X.columns]
            augmented = pd.concat([X.reset_index(drop=True),
                                   shadow.reset_index(drop=True)], axis=1)
            model = (RandomForestClassifier if is_classif else RandomForestRegressor)(
                n_estimators=50, random_state=self._rs, n_jobs=1,
            )
            model.fit(augmented.to_numpy(), y_fit)
            importances = model.feature_importances_
            real_imp = importances[: len(features)]
            shadow_imp = importances[len(features):]
            threshold = shadow_imp.max()
            hits += (real_imp > threshold).astype(int)

        # Binomial test with p=0.5 per iteration; feature is confirmed if
        # its hit count exceeds the (1 − α) quantile of Binomial(n_iter, 0.5).
        from scipy.stats import binom
        cutoff = binom.ppf(1 - alpha, n_iter, 0.5)
        confirmed = [f for f, h in zip(features, hits) if h > cutoff]
        hit_map = dict(zip(features, hits.tolist()))
        return ToolResult(
            status="ok",
            data={"confirmed": confirmed, "hits": hit_map,
                  "n_iter": n_iter, "cutoff": float(cutoff), "alpha": alpha},
            explanation=(
                f"Boruta confirmed {len(confirmed)} / {len(features)} "
                f"feature(s) over {n_iter} iterations (α={alpha})."
            ),
            math_trace=(
                "Shuffle each feature to build shadows; feature confirmed iff "
                "hits > Binomial(n, 0.5) upper α-quantile. "
                f"cutoff={float(cutoff)}."
            ),
        )

    # ---- SHAP / permutation importance ------------------------------- #

    def shap_selection(self, df: pd.DataFrame, target: str,
                       top_k: int | None = None,
                       method: Literal["auto", "shap", "permutation"] = "auto",
                       n_repeats: int = 5) -> ToolResult:
        try:
            X, y, is_classif = _split_xy(df, target)
        except (KeyError, ValueError) as exc:
            return _err(str(exc), type(exc).__name__)
        y_fit = pd.factorize(y)[0] if is_classif else y.astype(float).to_numpy()
        model = (RandomForestClassifier if is_classif else RandomForestRegressor)(
            n_estimators=100, random_state=self._rs, n_jobs=1,
        )
        model.fit(X.to_numpy(), y_fit)
        used_method = method
        scores: np.ndarray
        if method in ("auto", "shap"):
            try:
                import shap  # type: ignore
                explainer = shap.TreeExplainer(model)
                vals = explainer.shap_values(X.to_numpy())
                if isinstance(vals, list):
                    # multiclass: mean over classes
                    stacked = np.abs(np.stack(vals, axis=0)).mean(axis=0)
                    scores = stacked.mean(axis=0)
                else:
                    scores = np.abs(vals).mean(axis=0)
                used_method = "shap"
            except Exception:
                if method == "shap":
                    return _err("shap not available", "ImportError: shap")
                used_method = "permutation"
                scores = permutation_importance(
                    model, X.to_numpy(), y_fit,
                    n_repeats=n_repeats, random_state=self._rs, n_jobs=1,
                ).importances_mean
        else:
            scores = permutation_importance(
                model, X.to_numpy(), y_fit,
                n_repeats=n_repeats, random_state=self._rs, n_jobs=1,
            ).importances_mean
            used_method = "permutation"

        ranking = (pd.Series(scores, index=X.columns)
                   .sort_values(ascending=False))
        k = min(top_k, len(ranking)) if top_k else len(ranking)
        selected = ranking.index[:k].tolist()
        return ToolResult(
            status="ok",
            data={"selected": selected,
                  "scores": ranking.to_dict(),
                  "method": used_method,
                  "is_classif": is_classif},
            explanation=(
                f"{used_method.upper()} selected top {k} / {X.shape[1]} feature(s)."
            ),
            math_trace=(
                "shap: E[|φ_j|]; permutation: score_j = E[L(y, ĝ(X)) − L(y, ĝ(X^π_j))]."
            ),
        )
