"""Extractor — first ETL stage. Runs a query through a connector and
normalises the response into a `pandas.DataFrame`.
"""
from __future__ import annotations

import pandas as pd

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage


class Extractor(PipelineStage):
    """Execute `connector.query(query)` and return records as a DataFrame."""

    name = "extract"

    def __init__(
        self,
        connector: DataConnector,
        query: str,
        params: dict | None = None,
    ) -> None:
        self._connector = connector
        self._query = query
        self._params = params

    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        if self._params is not None:
            result = self._connector.query(self._query, params=self._params)
        else:
            result = self._connector.query(self._query)
        if result.status != "ok":
            return result
        data = result.data or {}
        records = data.get("records")
        if records is None:
            # REST-style: body + heuristic list extraction
            body = data.get("body")
            if isinstance(body, list):
                records = body
            elif isinstance(body, dict):
                for k in ("items", "results", "data"):
                    if isinstance(body.get(k), list):
                        records = body[k]
                        break
                else:
                    records = [body]
            else:
                records = []
        out = pd.DataFrame.from_records(records)
        null_rate = float(out.isna().mean().mean()) if out.size else 0.0
        return ToolResult(
            status="ok",
            data=out,
            explanation=(
                f"Extracted {len(out)} row(s) × {len(out.columns)} column(s) "
                f"from {type(self._connector).__name__} "
                f"(null rate {null_rate:.3%})."
            ),
            math_trace=(
                f"Extract: |records| = {len(out)}, |cols| = {len(out.columns)}, "
                f"null_rate = {null_rate:.4f}."
            ),
        )
