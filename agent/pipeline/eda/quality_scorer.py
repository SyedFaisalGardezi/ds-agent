"""QualityScorer — weighted 5-dimension data-quality rubric.

    score = Σ w_d · s_d    where d ∈ {completeness, validity, uniqueness,
                                       timeliness, distribution}

Each dimension returns a value in [0, 1]. Baseline frame (when provided)
calibrates the `distribution` dimension via KS distance.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from agent.core.types import ToolResult

_WEIGHTS = {"completeness": 0.2, "validity": 0.2, "uniqueness": 0.2,
            "timeliness": 0.2, "distribution": 0.2}
_WARN_THRESHOLD = 0.6


class QualityScorer:
    """Score a DataFrame on five dimensions, optionally against a baseline."""

    def __init__(self, weights: dict[str, float] | None = None,
                 warn_threshold: float = _WARN_THRESHOLD) -> None:
        w = weights or _WEIGHTS
        missing = set(_WEIGHTS) - set(w)
        if missing:
            raise ValueError(f"weights missing dimensions: {sorted(missing)}")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weight sum must be positive")
        self._weights = {k: v / total for k, v in w.items()}
        self._warn_threshold = float(warn_threshold)

    # ---- dimensions --------------------------------------------------- #

    @staticmethod
    def _completeness(df: pd.DataFrame) -> float:
        if df.size == 0:
            return 0.0
        return float(1.0 - df.isna().mean().mean())

    @staticmethod
    def _validity(df: pd.DataFrame) -> float:
        """Fraction of numeric cells that are finite."""
        num = df.select_dtypes("number")
        if num.size == 0:
            return 1.0
        arr = num.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(arr).sum()
        return float(finite / arr.size)

    @staticmethod
    def _uniqueness(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        dups = int(df.duplicated().sum())
        return float(1.0 - dups / len(df))

    @staticmethod
    def _timeliness(df: pd.DataFrame) -> float:
        """If there is a datetime column, reward monotonically-increasing recency."""
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not dt_cols or len(df) == 0:
            return 1.0  # Can't penalise what we can't measure.
        s = df[dt_cols[0]].dropna()
        if s.empty:
            return 0.5
        monotonic = float(s.is_monotonic_increasing)
        span = (s.max() - s.min()).total_seconds() if len(s) > 1 else 0.0
        span_score = min(1.0, span / (365 * 24 * 3600)) if span > 0 else 0.0
        return float(0.5 * monotonic + 0.5 * span_score)

    @staticmethod
    def _distribution(df: pd.DataFrame, baseline: pd.DataFrame | None) -> float:
        if baseline is None:
            return 1.0
        num = df.select_dtypes("number")
        base_num = baseline.select_dtypes("number")
        shared = [c for c in num.columns if c in base_num.columns]
        if not shared:
            return 1.0
        distances: list[float] = []
        for c in shared:
            a = num[c].dropna().to_numpy()
            b = base_num[c].dropna().to_numpy()
            if a.size < 2 or b.size < 2:
                continue
            stat, _ = ks_2samp(a, b)
            distances.append(float(stat))
        if not distances:
            return 1.0
        return float(1.0 - np.mean(distances))  # KS ∈ [0,1]; subtract

    # ---- score -------------------------------------------------------- #

    def score(self, df: pd.DataFrame, baseline: pd.DataFrame | None = None) -> ToolResult:
        dims = {
            "completeness": self._completeness(df),
            "validity": self._validity(df),
            "uniqueness": self._uniqueness(df),
            "timeliness": self._timeliness(df),
            "distribution": self._distribution(df, baseline),
        }
        total = float(sum(self._weights[k] * v for k, v in dims.items()))
        status: Literal["ok", "warning"] = (
            "warning" if total < self._warn_threshold else "ok"
        )
        return ToolResult(
            status=status,
            data={"score": total, "dimensions": dims, "weights": dict(self._weights)},
            explanation=(
                f"Quality score {total:.3f} "
                f"(warn < {self._warn_threshold}): "
                + ", ".join(f"{k}={v:.2f}" for k, v in dims.items())
            ),
            math_trace=(
                "score = Σ_d w_d · s_d; "
                f"weights = {self._weights}; "
                f"dims = {dims}."
            ),
        )
