"""Transformer stage — nine composable DataFrame transformers.

Each transformer follows an sklearn-style fit / transform / fit_transform
contract but returns :class:`ToolResult` from ``transform`` so a transformer
can be dropped into a :class:`PipelineStage` without adaptation. Math traces
are relational-algebra-style formulas so the orchestrator's provenance log
captures both *what* ran and *why* it was mathematically sound.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class BaseTransformer(ABC):
    name: str = ""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseTransformer: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> ToolResult: ...

    def fit_transform(self, df: pd.DataFrame) -> ToolResult:
        return self.fit(df).transform(df)

    # Adapter so a transformer can act as a pipeline stage.
    def as_stage(self) -> TransformerStage:
        return TransformerStage(self)


class TransformerStage(PipelineStage):
    """Wrap a :class:`BaseTransformer` so it slots into the orchestrator."""

    def __init__(self, transformer: BaseTransformer, fit: bool = True) -> None:
        self._t = transformer
        self._fit = fit
        self.name = f"transform:{transformer.name}"

    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        if self._fit:
            return self._t.fit_transform(df)
        return self._t.transform(df)


def _null_rate(df: pd.DataFrame) -> float:
    return float(df.isna().mean().mean()) if df.size else 0.0


# --------------------------------------------------------------------------- #
# 1. MedianImputer
# --------------------------------------------------------------------------- #


class MedianImputer(BaseTransformer):
    """Fill numeric NaNs with the per-column median learned in ``fit``."""

    name = "MedianImputer"

    def fit(self, df: pd.DataFrame) -> MedianImputer:
        self._medians = df.median(numeric_only=True)
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        if not hasattr(self, "_medians"):
            return ToolResult(status="error", data=df, explanation="Not fitted.",
                              math_trace="", error="NotFittedError")
        out = df.copy()
        filled = 0
        for col, med in self._medians.items():
            if col in out.columns:
                before = out[col].isna().sum()
                out[col] = out[col].fillna(med)
                filled += int(before)
        return ToolResult(
            status="ok", data=out,
            explanation=f"Imputed {filled} NaN cell(s) with per-column median.",
            math_trace=(
                f"x_ij ← x_ij if ¬NaN else median(X_.j); cells_imputed={filled}, "
                f"null_rate: {_null_rate(df):.4f} → {_null_rate(out):.4f}."
            ),
        )


# --------------------------------------------------------------------------- #
# 2. IQRClipper
# --------------------------------------------------------------------------- #


class IQRClipper(BaseTransformer):
    """Clip each numeric column to Tukey's [Q1 − 1.5·IQR, Q3 + 1.5·IQR]."""

    name = "IQRClipper"

    def fit(self, df: pd.DataFrame) -> IQRClipper:
        q1 = df.quantile(0.25, numeric_only=True)
        q3 = df.quantile(0.75, numeric_only=True)
        iqr = q3 - q1
        self._lower = q1 - 1.5 * iqr
        self._upper = q3 + 1.5 * iqr
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        if not hasattr(self, "_lower"):
            return ToolResult(status="error", data=df, explanation="Not fitted.",
                              math_trace="", error="NotFittedError")
        out = df.copy()
        clipped = 0
        for col in self._lower.index:
            if col in out.columns:
                mask = (out[col] < self._lower[col]) | (out[col] > self._upper[col])
                clipped += int(mask.sum())
                out[col] = out[col].clip(self._lower[col], self._upper[col])
        return ToolResult(
            status="ok", data=out,
            explanation=f"Clipped {clipped} outlier cell(s) at Tukey fences.",
            math_trace=(
                "x_ij ← clip(x_ij, Q1−1.5·IQR, Q3+1.5·IQR); "
                f"cells_clipped={clipped}."
            ),
        )


# --------------------------------------------------------------------------- #
# 3. ZScoreClipper
# --------------------------------------------------------------------------- #


class ZScoreClipper(BaseTransformer):
    """Clip numeric values whose z-score exceeds ±k (default k=3)."""

    name = "ZScoreClipper"

    def __init__(self, k: float = 3.0) -> None:
        self._k = float(k)

    def fit(self, df: pd.DataFrame) -> ZScoreClipper:
        self._mean = df.mean(numeric_only=True)
        self._std = df.std(numeric_only=True).replace(0, np.nan)
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        if not hasattr(self, "_mean"):
            return ToolResult(status="error", data=df, explanation="Not fitted.",
                              math_trace="", error="NotFittedError")
        out = df.copy()
        clipped = 0
        for col in self._mean.index:
            if col in out.columns and not np.isnan(self._std.get(col, np.nan)):
                lo = self._mean[col] - self._k * self._std[col]
                hi = self._mean[col] + self._k * self._std[col]
                mask = (out[col] < lo) | (out[col] > hi)
                clipped += int(mask.sum())
                out[col] = out[col].clip(lo, hi)
        return ToolResult(
            status="ok", data=out,
            explanation=f"Clipped {clipped} cell(s) beyond ±{self._k}σ.",
            math_trace=f"x_ij ← clip(x_ij, μ−{self._k}σ, μ+{self._k}σ); cells_clipped={clipped}.",
        )


# --------------------------------------------------------------------------- #
# 4. TargetEncoder
# --------------------------------------------------------------------------- #


class TargetEncoder(BaseTransformer):
    """Mean-target encoding with additive smoothing.

    encoded(c) = (n_c · mean_c + m · μ) / (n_c + m)

    where μ is the global target mean, n_c the count for category *c*,
    mean_c the target mean within *c*, and *m* the smoothing strength.
    """

    name = "TargetEncoder"

    def __init__(self, target: str, columns: list[str] | None = None,
                 smoothing: float = 10.0) -> None:
        self._target = target
        self._columns = columns
        self._m = float(smoothing)

    def fit(self, df: pd.DataFrame) -> TargetEncoder:
        if self._target not in df.columns:
            raise KeyError(f"target {self._target!r} not in frame")
        cols = self._columns or [
            c for c in df.columns
            if c != self._target and (df[c].dtype == object or str(df[c].dtype) == "category")
        ]
        self._fit_columns = cols
        self._global = float(df[self._target].mean())
        self._maps: dict[str, dict] = {}
        for c in cols:
            grouped = df.groupby(c)[self._target]
            means = grouped.mean()
            counts = grouped.count()
            self._maps[c] = (
                (counts * means + self._m * self._global) / (counts + self._m)
            ).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        if not hasattr(self, "_maps"):
            return ToolResult(status="error", data=df, explanation="Not fitted.",
                              math_trace="", error="NotFittedError")
        out = df.copy()
        for c, mapping in self._maps.items():
            if c in out.columns:
                out[c] = out[c].map(mapping).fillna(self._global)
        return ToolResult(
            status="ok", data=out,
            explanation=f"Target-encoded {len(self._maps)} column(s) with m={self._m}.",
            math_trace=(
                f"enc(c) = (n_c·mean_c + m·μ) / (n_c + m), "
                f"μ={self._global:.6f}, m={self._m}."
            ),
        )


# --------------------------------------------------------------------------- #
# 5. LagFeature
# --------------------------------------------------------------------------- #


class LagFeature(BaseTransformer):
    """Append x_{t−k} columns for every (column, lag) pair."""

    name = "LagFeature"

    def __init__(self, lags: list[int], columns: list[str]) -> None:
        if any(k <= 0 for k in lags):
            raise ValueError("lags must be positive")
        self._lags = list(lags)
        self._columns = list(columns)

    def fit(self, df: pd.DataFrame) -> LagFeature:
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        out = df.copy()
        added = 0
        for col in self._columns:
            if col not in out.columns:
                continue
            for k in self._lags:
                out[f"{col}_lag{k}"] = out[col].shift(k)
                added += 1
        return ToolResult(
            status="ok", data=out,
            explanation=f"Added {added} lag feature(s).",
            math_trace=(
                f"x^(k)_t = x_{{t−k}} for k ∈ {self._lags}, cols={self._columns}; "
                f"added={added}."
            ),
        )


# --------------------------------------------------------------------------- #
# 6. RollingStats
# --------------------------------------------------------------------------- #


class RollingStats(BaseTransformer):
    """Rolling window mean/std/min/max/sum."""

    name = "RollingStats"

    _ALLOWED = {"mean", "std", "min", "max", "sum", "median"}

    def __init__(self, window: int, columns: list[str],
                 stats: list[str] | None = None, min_periods: int | None = None) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._window = int(window)
        self._columns = list(columns)
        self._stats = stats or ["mean", "std"]
        bad = set(self._stats) - self._ALLOWED
        if bad:
            raise ValueError(f"unsupported stats: {sorted(bad)}")
        self._min_periods = min_periods

    def fit(self, df: pd.DataFrame) -> RollingStats:
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        out = df.copy()
        added = 0
        for col in self._columns:
            if col not in out.columns:
                continue
            roll = out[col].rolling(self._window, min_periods=self._min_periods)
            for stat in self._stats:
                out[f"{col}_roll{self._window}_{stat}"] = getattr(roll, stat)()
                added += 1
        return ToolResult(
            status="ok", data=out,
            explanation=f"Added {added} rolling feature(s) (window={self._window}).",
            math_trace=(
                f"y_t = f({{x_{{t−w+1}}, …, x_t}}), w={self._window}, "
                f"f ∈ {self._stats}, added={added}."
            ),
        )


# --------------------------------------------------------------------------- #
# 7. PolynomialFeatures
# --------------------------------------------------------------------------- #


class PolynomialFeatures(BaseTransformer):
    """Generate polynomial (and optional interaction) features up to degree d."""

    name = "PolynomialFeatures"

    def __init__(self, degree: int = 2, columns: list[str] | None = None,
                 interaction_only: bool = False, include_bias: bool = False) -> None:
        if degree < 2:
            raise ValueError("degree must be ≥ 2")
        self._degree = int(degree)
        self._columns = columns
        self._interaction_only = bool(interaction_only)
        self._include_bias = bool(include_bias)

    def fit(self, df: pd.DataFrame) -> PolynomialFeatures:
        self._fit_columns = self._columns or df.select_dtypes("number").columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        if not hasattr(self, "_fit_columns"):
            return ToolResult(status="error", data=df, explanation="Not fitted.",
                              math_trace="", error="NotFittedError")
        out = df.copy()
        cols = [c for c in self._fit_columns if c in out.columns]
        added = 0
        if self._include_bias:
            out["poly_bias"] = 1.0
            added += 1
        for d in range(2, self._degree + 1):
            combos = combinations_with_replacement(cols, d)
            for combo in combos:
                if self._interaction_only and len(set(combo)) != len(combo):
                    continue
                name = "·".join(combo)
                series = out[combo[0]].astype(float).copy()
                for c in combo[1:]:
                    series = series * out[c].astype(float)
                out[f"poly({name})"] = series
                added += 1
        return ToolResult(
            status="ok", data=out,
            explanation=f"Added {added} polynomial feature(s) up to degree {self._degree}.",
            math_trace=(
                f"φ(x) = {{∏_{{i∈S}} x_i : |S|≤{self._degree}, "
                f"interaction_only={self._interaction_only}}}; added={added}."
            ),
        )


# --------------------------------------------------------------------------- #
# 8. FFTFeatures
# --------------------------------------------------------------------------- #


class FFTFeatures(BaseTransformer):
    """Per-column FFT summary: total spectral energy plus top-k peak frequencies.

    For each column we compute the one-sided real FFT, report total energy
    Σ|X_k|², and the frequencies of the k largest magnitude bins.
    """

    name = "FFTFeatures"

    def __init__(self, columns: list[str], top_k: int = 3,
                 sample_rate: float = 1.0) -> None:
        if top_k < 1:
            raise ValueError("top_k must be ≥ 1")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._columns = list(columns)
        self._top_k = int(top_k)
        self._fs = float(sample_rate)

    def fit(self, df: pd.DataFrame) -> FFTFeatures:
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        out = df.copy()
        added = 0
        for col in self._columns:
            if col not in out.columns:
                continue
            x = out[col].astype(float).fillna(0.0).to_numpy()
            n = len(x)
            if n < 2:
                continue
            spectrum = np.fft.rfft(x)
            mag = np.abs(spectrum)
            freqs = np.fft.rfftfreq(n, d=1.0 / self._fs)
            energy = float((mag ** 2).sum())
            out[f"fft({col})_energy"] = energy
            added += 1
            # top-k bins excluding DC
            if mag.size > 1:
                ranked = np.argsort(mag[1:])[::-1] + 1
                for i in range(self._top_k):
                    idx = ranked[i] if i < len(ranked) else ranked[-1]
                    out[f"fft({col})_peak{i + 1}_freq"] = float(freqs[idx])
                    out[f"fft({col})_peak{i + 1}_mag"] = float(mag[idx])
                    added += 2
        return ToolResult(
            status="ok", data=out,
            explanation=f"Computed FFT summary for {len(self._columns)} column(s).",
            math_trace=(
                f"X_k = Σ_{{n=0}}^{{N−1}} x_n·e^(−2πikn/N), energy=Σ|X_k|², "
                f"top_k={self._top_k}, fs={self._fs}; added={added}."
            ),
        )


# --------------------------------------------------------------------------- #
# 9. WaveletFeatures
# --------------------------------------------------------------------------- #


def _haar_decompose(x: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return the detail coefficients at each of ``levels`` Haar levels."""
    details: list[np.ndarray] = []
    a = x.astype(float)
    for _ in range(levels):
        if a.size < 2:
            break
        if a.size % 2 == 1:
            a = np.append(a, a[-1])  # reflect-pad last sample
        even = a[0::2]
        odd = a[1::2]
        d = (even - odd) / np.sqrt(2.0)
        a = (even + odd) / np.sqrt(2.0)
        details.append(d)
    return details


class WaveletFeatures(BaseTransformer):
    """Haar wavelet detail-energy features per column and level."""

    name = "WaveletFeatures"

    def __init__(self, columns: list[str], levels: int = 3) -> None:
        if levels < 1:
            raise ValueError("levels must be ≥ 1")
        self._columns = list(columns)
        self._levels = int(levels)

    def fit(self, df: pd.DataFrame) -> WaveletFeatures:
        return self

    def transform(self, df: pd.DataFrame) -> ToolResult:
        out = df.copy()
        added = 0
        for col in self._columns:
            if col not in out.columns:
                continue
            x = out[col].astype(float).fillna(0.0).to_numpy()
            details = _haar_decompose(x, self._levels)
            for lvl, d in enumerate(details, start=1):
                out[f"wavelet({col})_L{lvl}_energy"] = float((d ** 2).sum())
                added += 1
        return ToolResult(
            status="ok", data=out,
            explanation=f"Computed Haar wavelet energies for {len(self._columns)} column(s).",
            math_trace=(
                f"(a, d)_ℓ = Haar(a_{{ℓ−1}}); energy_ℓ = Σ d_ℓ²; "
                f"levels={self._levels}; added={added}."
            ),
        )


__all__ = [
    "BaseTransformer",
    "TransformerStage",
    "MedianImputer",
    "IQRClipper",
    "ZScoreClipper",
    "TargetEncoder",
    "LagFeature",
    "RollingStats",
    "PolynomialFeatures",
    "FFTFeatures",
    "WaveletFeatures",
]
