"""PIIDetectorTool — dispatch `detect_pii` to a registered connector."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from agent.core.types import ToolResult


def _err(msg: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=msg,
                      math_trace="", error=error)


class PIIDetectorTool:
    name: str = "pii_detector"
    description: str = (
        "Detects PII columns in a table via the connector's detect_pii. "
        "Input: JSON with 'connector' and 'table'."
    )
    task_type: str = "default"

    def __init__(self, connectors: Mapping[str, Any] | None = None) -> None:
        self._connectors = connectors

    def _resolve(self, name: str):
        if self._connectors is not None:
            return self._connectors.get(name)
        try:
            from agent.connectors.registry import ConnectorRegistry
            return ConnectorRegistry.get()[name]
        except Exception:
            return None

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")
        connector_name = payload.get("connector")
        table = payload.get("table")
        if not connector_name or not table:
            return _err("'connector' and 'table' are required.", "ValueError")
        conn = self._resolve(connector_name)
        if conn is None:
            return _err(f"unknown connector {connector_name!r}.", "KeyError")
        try:
            return conn.detect_pii(table)
        except Exception as exc:  # noqa: BLE001
            return _err(f"{type(exc).__name__}: {exc}", type(exc).__name__)

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result))
