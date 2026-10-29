"""TimeSeriesFeatureEngineer — domain-agnostic and seismic-specific features.

- ``fourier_features`` : sin/cos harmonics for a seasonal index.
- ``seismic_features`` : waveform descriptors (RMS, peak-to-peak, zero-cross
                         rate, spectral centroid, dominant frequency, band
                         energies). Works on any numeric time-series column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent.core.types import ToolResult


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation,
                      math_trace="", error=error)


class TimeSeriesFeatureEngineer:
    # ---- fourier ----------------------------------------------------- #

    def fourier_features(
        self,
        series: pd.Series,
        periods: list[int],
        n_harmonics: int = 3,
    ) -> ToolResult:
        if not periods:
            return _err("periods must be non-empty", "ValueError")
        if any(p < 2 for p in periods):
            return _err("every period must be ≥ 2", "ValueError")
        if n_harmonics < 1:
            return _err("n_harmonics must be ≥ 1", "ValueError")
        s = pd.Series(series)
        n = len(s)
        if n == 0:
            return _err("empty series", "ValueError")
        t = np.arange(n, dtype=float)
        cols: dict[str, np.ndarray] = {}
        for p in periods:
            for h in range(1, n_harmonics + 1):
                omega = 2 * np.pi * h / p
                cols[f"fourier_p{p}_h{h}_sin"] = np.sin(omega * t)
                cols[f"fourier_p{p}_h{h}_cos"] = np.cos(omega * t)
        out = pd.DataFrame(cols, index=s.index)
        return ToolResult(
            status="ok",
            data={"frame": out, "periods": list(periods), "n_harmonics": n_harmonics},
            explanation=(
                f"Generated {out.shape[1]} Fourier feature(s) for "
                f"{len(periods)} period(s) × {n_harmonics} harmonic(s)."
            ),
            math_trace=(
                "For each (p, h): sin(2πht/p), cos(2πht/p); "
                f"|cols|={out.shape[1]}."
            ),
        )

    # ---- seismic ----------------------------------------------------- #

    @staticmethod
    def _zero_cross_rate(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        signs = np.sign(x - np.mean(x))
        signs[signs == 0] = 1
        return float(np.sum(signs[1:] != signs[:-1]) / (x.size - 1))

    @staticmethod
    def _band_energies(mag: np.ndarray, freqs: np.ndarray,
                       edges: tuple[float, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            band = float((mag[mask] ** 2).sum())
            out[f"band_{lo:g}_{hi:g}Hz"] = band
        return out

    def seismic_features(
        self,
        df: pd.DataFrame,
        sample_rate: float,
        columns: list[str] | None = None,
        bands_hz: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0, 25.0),
    ) -> ToolResult:
        if sample_rate <= 0:
            return _err("sample_rate must be > 0", "ValueError")
        if len(bands_hz) < 2:
            return _err("bands_hz must have ≥ 2 edges", "ValueError")
        cols = columns or df.select_dtypes("number").columns.tolist()
        if not cols:
            return _err("no numeric columns", "ValueError")

        rows: list[dict] = []
        for col in cols:
            if col not in df.columns:
                continue
            x = df[col].astype(float).dropna().to_numpy()
            if x.size < 4:
                continue
            rms = float(np.sqrt(np.mean(x ** 2)))
            peak_to_peak = float(np.ptp(x))
            zcr = self._zero_cross_rate(x)
            spectrum = np.fft.rfft(x)
            mag = np.abs(spectrum)
            freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
            total_energy = float((mag ** 2).sum())
            if mag.size > 1:
                centroid = float((freqs * mag).sum() / (mag.sum() or 1.0))
                dom_idx = int(np.argmax(mag[1:]) + 1)
                dom_freq = float(freqs[dom_idx])
                dom_mag = float(mag[dom_idx])
            else:
                centroid = dom_freq = dom_mag = 0.0
            entry = {
                "column": col,
                "rms": rms,
                "peak_to_peak": peak_to_peak,
                "zero_cross_rate": zcr,
                "spectral_centroid": centroid,
                "dominant_freq": dom_freq,
                "dominant_mag": dom_mag,
                "total_energy": total_energy,
            }
            entry.update(self._band_energies(mag, freqs, bands_hz))
            rows.append(entry)

        if not rows:
            return _err("no usable numeric series (each < 4 samples)", "ValueError")
        features = pd.DataFrame(rows).set_index("column")
        return ToolResult(
            status="ok",
            data={"features": features, "sample_rate": float(sample_rate),
                  "bands_hz": list(bands_hz)},
            explanation=(
                f"Computed seismic features for {len(features)} column(s) "
                f"(fs={sample_rate} Hz, {len(bands_hz) - 1} band(s))."
            ),
            math_trace=(
                "RMS = √(E[x²]); ZCR = mean(𝟙[sign(x_t)≠sign(x_{t−1})]); "
                "centroid = Σf·|X|/Σ|X|; band_E = Σ_{f∈[lo,hi)} |X_f|²."
            ),
        )
