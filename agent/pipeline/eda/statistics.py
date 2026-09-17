"""StatisticsAnalyser — hypothesis tests and time-series diagnostics.

All methods return ``ToolResult`` so they compose inside the pipeline. Heavy
statsmodels calls are imported lazily; if unavailable we fall back to a
clear error ToolResult.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from agent.core.types import ToolResult


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation,
                      math_trace="", error=error)


class StatisticsAnalyser:
    # ----- MI ---------------------------------------------------------- #

    def mutual_information(self, df: pd.DataFrame, target: str) -> ToolResult:
        if target not in df.columns:
            return _err(f"target {target!r} not found", "KeyError")
        y = df[target]
        X = df.drop(columns=[target]).select_dtypes("number").fillna(0)
        if X.shape[1] == 0:
            return _err("No numeric features for MI.", "ValueError")
        discrete = (y.dtype == object) or str(y.dtype) == "category" or y.nunique() < 20
        fn = mutual_info_classif if discrete else mutual_info_regression
        y_arr = pd.factorize(y)[0] if discrete else y.astype(float).to_numpy()
        scores = fn(X.to_numpy(), y_arr, random_state=0)
        s = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        return ToolResult(
            status="ok", data={"scores": s.to_dict(), "top": s.index[0]},
            explanation=f"MI computed against {target!r} ({'classif' if discrete else 'regress'}).",
            math_trace="I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]; k-NN estimator (Kraskov).",
        )

    # ----- correlation ------------------------------------------------- #

    def spearman_correlation(self, df: pd.DataFrame) -> ToolResult:
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            return _err("Need ≥2 numeric columns for Spearman.", "ValueError")
        rho = num.corr(method="spearman")
        return ToolResult(
            status="ok", data={"matrix": rho},
            explanation=f"Spearman ρ over {num.shape[1]} numeric column(s).",
            math_trace="ρ = 1 − 6·Σd_i² / [n(n²−1)] where d_i = rank(x_i) − rank(y_i).",
        )

    # ----- Cramér's V -------------------------------------------------- #

    def cramers_v(self, df: pd.DataFrame, col_a: str, col_b: str) -> ToolResult:
        if col_a not in df.columns or col_b not in df.columns:
            return _err("columns missing", "KeyError")
        table = pd.crosstab(df[col_a], df[col_b])
        if table.size == 0:
            return _err("empty contingency table", "ValueError")
        chi2, p, _, _ = _scipy_stats.chi2_contingency(table.to_numpy(), correction=False)
        n = int(table.to_numpy().sum())
        r, k = table.shape
        denom = n * (min(r, k) - 1) if min(r, k) > 1 else np.nan
        v = float(np.sqrt(chi2 / denom)) if denom and not np.isnan(denom) else 0.0
        return ToolResult(
            status="ok",
            data={"v": v, "chi2": float(chi2), "p_value": float(p),
                  "dof": (r - 1) * (k - 1)},
            explanation=f"Cramér's V({col_a!r}, {col_b!r}) = {v:.4f} (p={p:.3g}).",
            math_trace="V = √(χ²/(n·min(r−1,k−1))); χ² from contingency table.",
        )

    # ----- normality --------------------------------------------------- #

    def normality_test(self, df: pd.DataFrame) -> ToolResult:
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            return _err("No numeric columns.", "ValueError")
        out: dict[str, dict] = {}
        for c in num.columns:
            x = num[c].dropna().to_numpy()
            if x.size < 3:
                out[c] = {"stat": None, "p_value": None, "normal": None}
                continue
            # Shapiro has a practical upper limit ≈5000
            sample = x if x.size <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
            stat, p = _scipy_stats.shapiro(sample)
            out[c] = {"stat": float(stat), "p_value": float(p), "normal": bool(p > 0.05)}
        return ToolResult(
            status="ok", data={"tests": out},
            explanation=f"Shapiro-Wilk normality over {len(out)} column(s).",
            math_trace="W = (Σ a_i·x_(i))² / Σ (x_i − x̄)²; reject H₀(normal) if p ≤ 0.05.",
        )

    # ----- ADF / KPSS -------------------------------------------------- #

    def adf_test(self, series: pd.Series) -> ToolResult:
        try:
            from statsmodels.tsa.stattools import adfuller
        except Exception as exc:  # noqa: BLE001
            return _err("statsmodels missing", f"{type(exc).__name__}: {exc}")
        x = pd.Series(series).dropna().to_numpy()
        if x.size < 10:
            return _err("ADF requires ≥10 observations.", "ValueError")
        stat, p, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
        return ToolResult(
            status="ok",
            data={"stat": float(stat), "p_value": float(p), "lags": int(lags),
                  "nobs": int(nobs), "critical_values": {k: float(v) for k, v in crit.items()},
                  "stationary": bool(p < 0.05)},
            explanation=f"ADF stat={stat:.4f}, p={p:.3g}.",
            math_trace="Δy_t = α + βt + γy_{t−1} + Σ δ_i Δy_{t−i} + ε_t; H₀: γ=0 (unit root).",
        )

    def kpss_test(self, series: pd.Series) -> ToolResult:
        try:
            from statsmodels.tsa.stattools import kpss
        except Exception as exc:  # noqa: BLE001
            return _err("statsmodels missing", f"{type(exc).__name__}: {exc}")
        x = pd.Series(series).dropna().to_numpy()
        if x.size < 10:
            return _err("KPSS requires ≥10 observations.", "ValueError")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lags, crit = kpss(x, regression="c", nlags="auto")
        return ToolResult(
            status="ok",
            data={"stat": float(stat), "p_value": float(p), "lags": int(lags),
                  "critical_values": {k: float(v) for k, v in crit.items()},
                  "stationary": bool(p > 0.05)},
            explanation=f"KPSS stat={stat:.4f}, p≈{p:.3g}.",
            math_trace="y_t = r_t + βt + ε_t, r_t = r_{t−1} + u_t; H₀: σ²_u=0 (stationary).",
        )

    # ----- STL --------------------------------------------------------- #

    def stl_decompose(self, series: pd.Series, period: int) -> ToolResult:
        try:
            from statsmodels.tsa.seasonal import STL
        except Exception as exc:  # noqa: BLE001
            return _err("statsmodels missing", f"{type(exc).__name__}: {exc}")
        if period < 2:
            return _err("period must be ≥2", "ValueError")
        s = pd.Series(series).dropna()
        if s.size < 2 * period:
            return _err("need ≥2·period observations", "ValueError")
        res = STL(s, period=period, robust=True).fit()
        total_var = float(s.var(ddof=0)) or 1.0
        strength_trend = max(0.0, 1.0 - float(res.resid.var(ddof=0)) /
                             float((res.trend + res.resid).var(ddof=0) or 1.0))
        strength_season = max(0.0, 1.0 - float(res.resid.var(ddof=0)) /
                              float((res.seasonal + res.resid).var(ddof=0) or 1.0))
        return ToolResult(
            status="ok",
            data={"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid,
                  "strength_trend": strength_trend,
                  "strength_seasonal": strength_season,
                  "period": int(period), "total_var": total_var},
            explanation=f"STL(period={period}): trend-strength {strength_trend:.3f}, "
                        f"seasonal-strength {strength_season:.3f}.",
            math_trace="y_t = T_t + S_t + R_t; strength = max(0, 1 − Var(R)/Var(T+R | S+R)).",
        )

    # ----- ACF / PACF -------------------------------------------------- #

    def acf_pacf(self, series: pd.Series, lags: int = 40) -> ToolResult:
        try:
            from statsmodels.tsa.stattools import acf, pacf
        except Exception as exc:  # noqa: BLE001
            return _err("statsmodels missing", f"{type(exc).__name__}: {exc}")
        x = pd.Series(series).dropna().to_numpy()
        if x.size < lags + 2:
            return _err(f"need ≥{lags + 2} observations", "ValueError")
        a = acf(x, nlags=lags, fft=True)
        p = pacf(x, nlags=lags, method="ywm")
        return ToolResult(
            status="ok",
            data={"acf": a.tolist(), "pacf": p.tolist(), "lags": int(lags)},
            explanation=f"ACF/PACF up to lag {lags}.",
            math_trace=(
                "ACF(k) = Cov(y_t, y_{t−k}) / Var(y_t); "
                "PACF(k) via Yule-Walker (method='ywm')."
            ),
        )
