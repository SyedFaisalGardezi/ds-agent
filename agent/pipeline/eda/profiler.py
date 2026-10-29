"""DataProfiler — per-column summary statistics and health flags.

Returns one row per column plus overall frame metrics. Pure-pandas so it runs
in every environment.
"""
from __future__ import annotations

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage


class DataProfiler(PipelineStage):
    """Profile every column: dtype, null count, cardinality, min/max, skew, kurt."""

    name = "eda_profile"

    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        n_rows, n_cols = df.shape
        if n_rows == 0:
            return ToolResult(
                status="warning", data={"columns": [], "n_rows": 0, "n_cols": n_cols},
                explanation="Empty DataFrame; nothing to profile.",
                math_trace="|rows|=0.",
            )

        rows: list[dict] = []
        for col in df.columns:
            s = df[col]
            nulls = int(s.isna().sum())
            dtype = str(s.dtype)
            entry: dict = {
                "column": col,
                "dtype": dtype,
                "null_count": nulls,
                "null_rate": nulls / n_rows,
                "n_unique": int(s.nunique(dropna=True)),
                "unique_rate": float(s.nunique(dropna=True) / n_rows),
            }
            if pd.api.types.is_numeric_dtype(s):
                non_null = s.dropna()
                entry.update({
                    "min": float(non_null.min()) if not non_null.empty else None,
                    "max": float(non_null.max()) if not non_null.empty else None,
                    "mean": float(non_null.mean()) if not non_null.empty else None,
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
                    "kurtosis": float(non_null.kurt()) if len(non_null) > 3 else 0.0,
                })
            else:
                # top-value frequency for categoricals
                vc = s.value_counts(dropna=True)
                entry.update({
                    "top": vc.index[0] if not vc.empty else None,
                    "top_freq": int(vc.iloc[0]) if not vc.empty else 0,
                })
            rows.append(entry)

        profile = pd.DataFrame(rows)
        overall_null_rate = float(df.isna().mean().mean())
        duplicate_rows = int(df.duplicated().sum())
        memory_bytes = int(df.memory_usage(deep=True).sum())

        return ToolResult(
            status="ok",
            data={
                "profile": profile,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "overall_null_rate": overall_null_rate,
                "duplicate_rows": duplicate_rows,
                "memory_bytes": memory_bytes,
            },
            explanation=(
                f"Profiled {n_cols} column(s) × {n_rows} row(s); "
                f"null rate {overall_null_rate:.3%}, duplicates {duplicate_rows}."
            ),
            math_trace=(
                "For each column j: null_j = Σ 𝟙[NaN], unique_j = |distinct(x_.j)|, "
                "μ_j, σ_j, skew_j = E[(x−μ)³]/σ³, kurt_j = E[(x−μ)⁴]/σ⁴ − 3."
            ),
        )
