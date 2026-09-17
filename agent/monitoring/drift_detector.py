"""DriftDetector — PSI, KS, χ² drift tests + bulk feature sweep."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats

from agent.core.types import ToolResult

_PSI_THRESHOLD = 0.2
_N_BINS = 10


class DriftDetector:
    def psi(self, expected: pd.Series, actual: pd.Series,
            n_bins: int = _N_BINS, threshold: float = _PSI_THRESHOLD) -> ToolResult:
        e = pd.Series(expected).dropna().astype(float).to_numpy()
        a = pd.Series(actual).dropna().astype(float).to_numpy()
        if e.size == 0 or a.size == 0:
            return ToolResult(status="error", data=None,
                              explanation="empty input(s)",
                              math_trace="", error="ValueError")
        edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        # collapse duplicate edges
        edges = np.unique(edges)
        if edges.size < 3:
            return ToolResult(status="warning", data={"psi": 0.0, "drift": False},
                              explanation="not enough distinct bin edges",
                              math_trace="|bins|<2.")
        e_hist, _ = np.histogram(e, bins=edges)
        a_hist, _ = np.histogram(a, bins=edges)
        e_p = np.clip(e_hist / e.size, 1e-6, None)
        a_p = np.clip(a_hist / a.size, 1e-6, None)
        psi = float(((a_p - e_p) * np.log(a_p / e_p)).sum())
        return ToolResult(
            status="ok",
            data={"psi": psi, "drift": bool(psi > threshold),
                  "threshold": threshold, "n_bins": int(edges.size - 1)},
            explanation=f"PSI={psi:.4f} (drift={psi > threshold}, τ={threshold}).",
            math_trace="PSI = Σ_b (a_b − e_b)·log(a_b/e_b).",
        )

    def ks_test(self, expected: pd.Series, actual: pd.Series,
                alpha: float = 0.05) -> ToolResult:
        e = pd.Series(expected).dropna().to_numpy()
        a = pd.Series(actual).dropna().to_numpy()
        if e.size < 2 or a.size < 2:
            return ToolResult(status="error", data=None,
                              explanation="need ≥2 samples each",
                              math_trace="", error="ValueError")
        stat, p = _stats.ks_2samp(e, a)
        return ToolResult(
            status="ok",
            data={"stat": float(stat), "p_value": float(p),
                  "drift": bool(p < alpha), "alpha": alpha},
            explanation=f"KS={stat:.4f}, p={p:.4g}, drift={p < alpha}.",
            math_trace="D = sup_x |F_e(x) − F_a(x)|.",
        )

    def chi_squared(self, expected: pd.Series, actual: pd.Series,
                    alpha: float = 0.05) -> ToolResult:
        e_vc = pd.Series(expected).dropna().value_counts()
        a_vc = pd.Series(actual).dropna().value_counts()
        cats = sorted(set(e_vc.index) | set(a_vc.index))
        if len(cats) < 2:
            return ToolResult(status="error", data=None,
                              explanation="need ≥2 categories",
                              math_trace="", error="ValueError")
        e_arr = np.array([e_vc.get(c, 0) for c in cats], dtype=float)
        a_arr = np.array([a_vc.get(c, 0) for c in cats], dtype=float)
        # scale expected to actual total
        if e_arr.sum() == 0 or a_arr.sum() == 0:
            return ToolResult(status="error", data=None,
                              explanation="empty class", math_trace="",
                              error="ValueError")
        e_scaled = e_arr * (a_arr.sum() / e_arr.sum())
        e_scaled = np.clip(e_scaled, 1e-6, None)
        chi2 = float(((a_arr - e_scaled) ** 2 / e_scaled).sum())
        dof = len(cats) - 1
        p = float(1 - _stats.chi2.cdf(chi2, df=dof))
        return ToolResult(
            status="ok",
            data={"chi2": chi2, "p_value": p, "dof": dof,
                  "drift": bool(p < alpha), "categories": cats},
            explanation=f"χ²={chi2:.4f}, p={p:.4g}, drift={p < alpha}.",
            math_trace="χ² = Σ (O_i − E_i)²/E_i; rescale E to match Σ O.",
        )

    def check_all_features(self, df_baseline: pd.DataFrame,
                           df_current: pd.DataFrame,
                           threshold: float = _PSI_THRESHOLD) -> ToolResult:
        shared = [c for c in df_baseline.columns if c in df_current.columns]
        if not shared:
            return ToolResult(status="error", data=None,
                              explanation="no shared columns", math_trace="",
                              error="ValueError")
        results: dict[str, dict] = {}
        drifting: list[str] = []
        for c in shared:
            base = df_baseline[c]
            curr = df_current[c]
            if pd.api.types.is_numeric_dtype(base) and pd.api.types.is_numeric_dtype(curr):
                r = self.psi(base, curr, threshold=threshold)
                if r.status != "ok":
                    continue
                results[c] = {"method": "psi", **{k: r.data[k] for k in ("psi", "drift")}}
            else:
                r = self.chi_squared(base, curr)
                if r.status != "ok":
                    continue
                results[c] = {"method": "chi2", "p_value": r.data["p_value"],
                              "drift": r.data["drift"]}
            if results[c]["drift"]:
                drifting.append(c)
        return ToolResult(
            status="ok",
            data={"per_feature": results, "drifting": drifting,
                  "n_checked": len(results), "threshold": threshold},
            explanation=f"Checked {len(results)} feature(s); "
                        f"{len(drifting)} drifting.",
            math_trace="numeric→PSI; non-numeric→χ² on value counts.",
        )
