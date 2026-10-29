"""FeatureEngineerTool — thin wrapper over FeatureGenerator.generate."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.feature_engineering.generator import FeatureGenerator


def _err(msg: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=msg,
                      math_trace="", error=error)


class FeatureEngineerTool:
    name: str = "feature_engineer"
    description: str = (
        "Synthesises candidate features via ratios, differences, unary transforms, "
        "datetime decomposition, and one-hot encoding. "
        "Input: JSON with 'records' (list[dict]), optional 'target', "
        "'max_pairs' (int), 'max_onehot_cardinality' (int)."
    )
    task_type: str = "code"

    def __init__(self, generator: FeatureGenerator | None = None) -> None:
        self._generator = generator  # lazy per-call if None so kwargs can override

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")
        records = payload.get("records")
        if not isinstance(records, list):
            return _err("'records' must be a list of row dicts.", "ValueError")
        try:
            df = pd.DataFrame(records)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not build DataFrame: {exc}", type(exc).__name__)
        gen = self._generator or FeatureGenerator(
            max_pairs=int(payload.get("max_pairs", 25)),
            max_onehot_cardinality=int(payload.get("max_onehot_cardinality", 10)),
        )
        return gen.generate(df, target=payload.get("target"))

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result), default=str)
