"""Loader — terminal ETL stage. Writes a DataFrame to disk in parquet / csv /
json / jsonl format.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage


class Loader(PipelineStage):
    """Persist the DataFrame at ``output_path`` in the inferred or requested format."""

    name = "load"

    _SUPPORTED = {"parquet", "csv", "json", "jsonl"}

    def __init__(
        self,
        output_path: str | Path,
        format: str | None = None,
        index: bool = False,
    ) -> None:
        self._output_path = Path(output_path)
        self._index = bool(index)
        fmt = (format or self._infer_format(self._output_path)).lower()
        if fmt not in self._SUPPORTED:
            raise ValueError(f"Unsupported format {fmt!r}; expected one of {sorted(self._SUPPORTED)}")
        self._format = fmt

    @staticmethod
    def _infer_format(path: Path) -> str:
        suf = path.suffix.lower().lstrip(".")
        if suf == "pq":
            return "parquet"
        return suf

    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            if self._format == "parquet":
                df.to_parquet(self._output_path, index=self._index)
            elif self._format == "csv":
                df.to_csv(self._output_path, index=self._index)
            elif self._format == "json":
                df.to_json(self._output_path, orient="records")
            elif self._format == "jsonl":
                df.to_json(self._output_path, orient="records", lines=True)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=df,
                explanation=f"Failed to write {self._format} file: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        size = self._output_path.stat().st_size if self._output_path.exists() else 0
        return ToolResult(
            status="ok",
            data=df,
            explanation=(
                f"Wrote {len(df)} row(s) × {len(df.columns)} column(s) "
                f"to {self._output_path} ({self._format}, {size} bytes)."
            ),
            math_trace=(
                f"Load: |rows|={len(df)}, |cols|={len(df.columns)}, "
                f"bytes={size}, format={self._format}."
            ),
        )
