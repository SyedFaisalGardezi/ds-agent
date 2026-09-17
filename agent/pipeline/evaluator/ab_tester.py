"""ABTester — champion vs challenger: shadow runs, DeLong, McNemar, gating."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats

from agent.core.types import ToolResult


class ABTester:
    def shadow_run(self, champion, challenger, X: pd.DataFrame) -> ToolResult:
        Xn = X.select_dtypes("number").fillna(0.0).to_numpy()
        try:
            y_c = champion.predict(Xn)
            y_h = challenger.predict(Xn)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"predict failed: {exc}",
                              math_trace="", error=type(exc).__name__)
        disagree = float(np.mean(np.asarray(y_c) != np.asarray(y_h)))
        return ToolResult(
            status="ok",
            data={"champion_pred": np.asarray(y_c).tolist(),
                  "challenger_pred": np.asarray(y_h).tolist(),
                  "disagreement_rate": disagree,
                  "n": int(Xn.shape[0])},
            explanation=f"Shadow run: disagreement {disagree:.4f} on {Xn.shape[0]} row(s).",
            math_trace="disagreement = (1/N)·Σ 𝟙[ĉ_i ≠ ĥ_i].",
        )

    @staticmethod
    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        sorted_x = x[order]
        n = len(x)
        i = 0
        while i < n:
            j = i
            while j < n and sorted_x[j] == sorted_x[i]:
                j += 1
            avg = 0.5 * (i + j - 1) + 1
            ranks[order[i:j]] = avg
            i = j
        return ranks

    def delong_test(self, y_true, y_prob_a, y_prob_b) -> ToolResult:
        y = np.asarray(y_true).astype(int)
        a = np.asarray(y_prob_a).astype(float)
        b = np.asarray(y_prob_b).astype(float)
        if y.shape != a.shape or y.shape != b.shape:
            return ToolResult(status="error", data=None,
                              explanation="shape mismatch", math_trace="",
                              error="ValueError")
        pos = y == 1
        neg = ~pos
        m = int(pos.sum()); n = int(neg.sum())
        if m == 0 or n == 0:
            return ToolResult(status="error", data=None,
                              explanation="need both classes present",
                              math_trace="", error="ValueError")

        def _auc_and_v(p: np.ndarray):
            Xp = p[pos]; Xn = p[neg]
            r_all = self._midrank(np.concatenate([Xp, Xn]))
            r_pos = r_all[:m]; r_neg = r_all[m:]
            auc = (r_pos.sum() - m * (m + 1) / 2.0) / (m * n)
            r10 = self._midrank(Xp); r01 = self._midrank(Xn)
            v10 = (r_pos - r10) / n
            v01 = 1.0 - (r_neg - r01) / m
            return auc, v10, v01

        auc_a, v10_a, v01_a = _auc_and_v(a)
        auc_b, v10_b, v01_b = _auc_and_v(b)
        s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
        s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
        S = s10 / m + s01 / n
        diff = auc_a - auc_b
        var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
        if var <= 0:
            z = 0.0; p = 1.0
        else:
            z = float(diff / np.sqrt(var))
            p = float(2 * (1 - _stats.norm.cdf(abs(z))))
        return ToolResult(
            status="ok",
            data={"auc_a": float(auc_a), "auc_b": float(auc_b),
                  "diff": float(diff), "z": z, "p_value": p, "variance": var},
            explanation=f"DeLong: AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, p={p:.4g}.",
            math_trace="Z = (AUC_A − AUC_B) / √Var(AUC_A − AUC_B); ~ N(0,1).",
        )

    def mcnemar_test(self, pred_a, pred_b, y_true) -> ToolResult:
        a = np.asarray(pred_a); b = np.asarray(pred_b); y = np.asarray(y_true)
        if a.shape != b.shape or a.shape != y.shape:
            return ToolResult(status="error", data=None,
                              explanation="shape mismatch", math_trace="",
                              error="ValueError")
        ca = (a == y); cb = (b == y)
        b10 = int(((ca) & (~cb)).sum())
        b01 = int(((~ca) & (cb)).sum())
        total = b10 + b01
        if total == 0:
            chi2 = 0.0; p = 1.0
        else:
            chi2 = (abs(b10 - b01) - 1) ** 2 / total
            p = float(1 - _stats.chi2.cdf(chi2, df=1))
        return ToolResult(
            status="ok",
            data={"b10": b10, "b01": b01, "chi2": float(chi2), "p_value": p},
            explanation=f"McNemar: b10={b10}, b01={b01}, p={p:.4g}.",
            math_trace="χ² = (|b10 − b01| − 1)² / (b10 + b01); Yates corrected.",
        )

    def should_promote(self, challenger_metrics: dict, champion_metrics: dict,
                       margin: float, metric: str = "accuracy",
                       higher_is_better: bool = True) -> bool:
        if metric not in challenger_metrics or metric not in champion_metrics:
            return False
        delta = challenger_metrics[metric] - champion_metrics[metric]
        if not higher_is_better:
            delta = -delta
        return bool(delta > margin)
