"""Visualiser — headless matplotlib plots for EDA artefacts.

All methods write files under ``output_dir`` namespaced by ``run_id`` and
return a :class:`ToolResult` carrying the list of absolute paths written.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

# Force a non-interactive backend so tests run headless in CI/Docker.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent.core.types import ToolResult  # noqa: E402


class Visualiser:
    """Render distribution plots, correlation heatmaps, and time-series plots."""

    def __init__(self, output_dir: str | os.PathLike = "outputs/plots") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers ------------------------------------------------------ #

    def _run_dir(self, run_id: str) -> Path:
        d = self._dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- plots -------------------------------------------------------- #

    def distribution_plots(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            return ToolResult(status="warning", data={"paths": []},
                              explanation="No numeric columns to plot.",
                              math_trace="|numeric cols|=0.")
        out_dir = self._run_dir(run_id)
        paths: list[str] = []
        for col in num.columns:
            s = num[col].dropna()
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(s.to_numpy(), bins=min(50, max(5, int(np.sqrt(len(s))))))
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            path = out_dir / f"dist_{col}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=90)
            plt.close(fig)
            paths.append(str(path))
        return ToolResult(
            status="ok", data={"paths": paths},
            explanation=f"Wrote {len(paths)} distribution plot(s) to {out_dir}.",
            math_trace="For each numeric col j: hist(x_.j) with bins=min(50, ⌈√N⌉).",
        )

    def correlation_heatmap(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            return ToolResult(status="warning", data={"path": None},
                              explanation="Need ≥2 numeric columns.",
                              math_trace=f"|numeric cols|={num.shape[1]}.")
        corr = num.corr().to_numpy()
        out_dir = self._run_dir(run_id)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(num.shape[1]))
        ax.set_yticks(range(num.shape[1]))
        ax.set_xticklabels(num.columns, rotation=45, ha="right")
        ax.set_yticklabels(num.columns)
        fig.colorbar(im, ax=ax)
        ax.set_title("Pearson correlation")
        path = out_dir / "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return ToolResult(
            status="ok", data={"path": str(path)},
            explanation=f"Wrote correlation heatmap to {path}.",
            math_trace="ρ_ij = Cov(x_i, x_j) / (σ_i σ_j).",
        )

    def time_series_plot(self, series: pd.Series, run_id: str) -> ToolResult:
        s = pd.Series(series).dropna()
        if s.empty:
            return ToolResult(status="warning", data={"path": None},
                              explanation="Empty series.", math_trace="|s|=0.")
        out_dir = self._run_dir(run_id)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(s.index, s.to_numpy())
        ax.set_title(s.name or "time series")
        ax.set_xlabel(s.index.name or "t")
        ax.set_ylabel(s.name or "value")
        name = (s.name or "series").replace("/", "_")
        path = out_dir / f"ts_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return ToolResult(
            status="ok", data={"path": str(path)},
            explanation=f"Wrote time-series plot to {path}.",
            math_trace="Plot {(t_i, y_i)} for i=1..N.",
        )
