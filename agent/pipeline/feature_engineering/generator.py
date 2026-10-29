"""FeatureGenerator — automatic candidate-feature synthesis.

Proposes new columns from an input DataFrame without any external deps:
- pairwise ratios and differences for numeric columns (bounded fan-out)
- log1p, square, and sqrt unary transforms for positive numeric columns
- datetime decomposition (year / month / day / dayofweek / hour) for datetime
- cardinality-aware one-hot for low-cardinality categoricals

Designed to run before a FeatureSelector. Returns a DataFrame plus a
provenance dict listing each synthesised column's recipe.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from agent.core.types import ToolResult


class FeatureGenerator:
    """Synthesise candidate features via bounded unary / binary templates."""

    def __init__(
        self,
        max_pairs: int = 25,
        max_onehot_cardinality: int = 10,
        numeric_columns: list[str] | None = None,
    ) -> None:
        if max_pairs < 0:
            raise ValueError("max_pairs must be ≥ 0")
        if max_onehot_cardinality < 2:
            raise ValueError("max_onehot_cardinality must be ≥ 2")
        self._max_pairs = int(max_pairs)
        self._max_onehot = int(max_onehot_cardinality)
        self._numeric_columns = numeric_columns

    def generate(self, df: pd.DataFrame, target: str | None = None) -> ToolResult:
        if df.empty:
            return ToolResult(
                status="warning", data={"frame": df.copy(), "recipes": {}},
                explanation="Empty DataFrame; nothing to synthesise.",
                math_trace="|rows|=0.",
            )
        out = df.copy()
        recipes: dict[str, str] = {}

        num_cols = [
            c for c in (self._numeric_columns or out.select_dtypes("number").columns)
            if c != target and pd.api.types.is_numeric_dtype(out[c])
        ]

        # ---- unary transforms ---------------------------------------- #
        for c in num_cols:
            s = out[c]
            if (s > 0).all() and s.notna().any():
                out[f"log1p({c})"] = np.log1p(s)
                recipes[f"log1p({c})"] = f"log(1 + {c})"
            out[f"sq({c})"] = s ** 2
            recipes[f"sq({c})"] = f"{c}²"
            if (s >= 0).all():
                out[f"sqrt({c})"] = np.sqrt(s)
                recipes[f"sqrt({c})"] = f"√{c}"

        # ---- pairwise ratios and differences (bounded) --------------- #
        pair_count = 0
        for a, b in combinations(num_cols, 2):
            if pair_count >= self._max_pairs:
                break
            denom = out[b].replace(0, np.nan)
            out[f"ratio({a}/{b})"] = out[a] / denom
            recipes[f"ratio({a}/{b})"] = f"{a} / {b} (0→NaN)"
            out[f"diff({a}-{b})"] = out[a] - out[b]
            recipes[f"diff({a}-{b})"] = f"{a} − {b}"
            pair_count += 1

        # ---- datetime decomposition ---------------------------------- #
        for c in out.columns:
            if c == target:
                continue
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                base = out[c]
                out[f"dt_year({c})"] = base.dt.year
                out[f"dt_month({c})"] = base.dt.month
                out[f"dt_day({c})"] = base.dt.day
                out[f"dt_dow({c})"] = base.dt.dayofweek
                out[f"dt_hour({c})"] = base.dt.hour
                for k in ("year", "month", "day", "dow", "hour"):
                    recipes[f"dt_{k}({c})"] = f"{c}.{k}"

        # ---- low-cardinality one-hot --------------------------------- #
        for c in out.columns:
            if c == target:
                continue
            s = df[c] if c in df.columns else None
            if s is None:
                continue
            if s.dtype == object or str(s.dtype) == "category":
                nunique = s.nunique(dropna=True)
                if 2 <= nunique <= self._max_onehot:
                    dummies = pd.get_dummies(s, prefix=f"oh({c})", dummy_na=False)
                    for col in dummies.columns:
                        out[col] = dummies[col].astype(int)
                        recipes[col] = f"𝟙[{c} = {col.split('=', 1)[-1]}]"

        new_cols = [c for c in out.columns if c not in df.columns]
        return ToolResult(
            status="ok",
            data={"frame": out, "recipes": recipes, "new_columns": new_cols,
                  "base_shape": df.shape, "output_shape": out.shape},
            explanation=(
                f"Synthesised {len(new_cols)} feature(s) from "
                f"{df.shape[1]} base column(s) "
                f"({pair_count} pair expansion(s))."
            ),
            math_trace=(
                f"φ ∈ {{log1p, x², √x, a/b, a−b, dt_parts, one-hot}}; "
                f"|new|={len(new_cols)}, max_pairs={self._max_pairs}, "
                f"max_onehot={self._max_onehot}."
            ),
        )
