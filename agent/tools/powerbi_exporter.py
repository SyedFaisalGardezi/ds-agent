"""PowerBIExporterTool — flatten a leaderboard run to a Power-BI-ready CSV.

Power BI ingests CSV with minimal fuss; we flatten the JSON `metrics` and
`hyperparams` columns into a wide row so the artefact drops straight into a
table visual. A JSON metadata side-car is also written for programmatic use.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from agent.core.types import ToolResult
from agent.pipeline.evaluator.leaderboard import Leaderboard


def _err(msg: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=msg,
                      math_trace="", error=error)


def _fetch_run(leaderboard: Leaderboard, run_id: str) -> dict | None:
    cur = leaderboard._conn.execute(  # noqa: SLF001
        "SELECT * FROM runs WHERE run_id = ?", (run_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip(cols, row))


def _flatten(rec: dict) -> dict:
    flat = {
        "run_id": rec["run_id"],
        "model_type": rec.get("model_type"),
        "train_time_s": rec.get("train_time_s"),
        "inference_ms": rec.get("inference_ms"),
        "model_mb": rec.get("model_mb"),
        "shap_runtime_s": rec.get("shap_runtime_s"),
    }
    for k, v in (json.loads(rec.get("hyperparams") or "{}")).items():
        flat[f"hp_{k}"] = v
    for k, v in (json.loads(rec.get("metrics") or "{}")).items():
        flat[f"metric_{k}"] = v
    return flat


class PowerBIExporterTool:
    name: str = "powerbi_exporter"
    description: str = (
        "Exports leaderboard runs to Power-BI-ready CSV + JSON sidecar. "
        "Input: JSON with 'run_id' (single) or 'run_ids' (list), "
        "optional 'output_path'."
    )
    task_type: str = "default"

    def __init__(self, leaderboard: Leaderboard | None = None,
                 output_dir: str | Path = "outputs/powerbi") -> None:
        self._leaderboard = leaderboard
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")

        run_ids = payload.get("run_ids")
        if not run_ids:
            rid = payload.get("run_id")
            if not rid:
                return _err("'run_id' or 'run_ids' is required.", "ValueError")
            run_ids = [rid]
        if not isinstance(run_ids, list):
            return _err("'run_ids' must be a list.", "ValueError")

        lb = self._leaderboard or Leaderboard()
        try:
            raw = [_fetch_run(lb, rid) for rid in run_ids]
        except sqlite3.DatabaseError as exc:
            return _err(f"leaderboard query failed: {exc}", "DatabaseError")
        missing = [rid for rid, rec in zip(run_ids, raw) if rec is None]
        if missing:
            return _err(f"unknown run_id(s): {missing}", "KeyError")

        flat = [_flatten(rec) for rec in raw]  # type: ignore[arg-type]
        # Union of keys across rows so every column appears in the header.
        header: list[str] = []
        seen: set[str] = set()
        for row in flat:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    header.append(k)

        out = Path(payload.get("output_path") or (self._dir / "runs.csv"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for row in flat:
                w.writerow({k: row.get(k, "") for k in header})

        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps(flat, default=str, indent=2), encoding="utf-8")

        return ToolResult(
            status="ok",
            data={"csv_path": str(out), "json_path": str(sidecar),
                  "rows": len(flat), "columns": len(header),
                  "run_ids": run_ids},
            explanation=(
                f"Exported {len(flat)} run(s) × {len(header)} cols to {out.name}."
            ),
            math_trace=f"flatten(runs) → csv[{len(flat)}×{len(header)}].",
        )

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result), default=str)
