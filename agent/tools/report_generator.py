"""ReportGeneratorTool — emit a Markdown summary for a leaderboard run.

Reads from the SQLite Leaderboard (or a caller-provided dict). The output is
a deterministic Markdown file so CI can diff it; a future extension could
wrap this in a pandoc/weasyprint call to produce PDF/PPTX.
"""
from __future__ import annotations

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
    cur = leaderboard._conn.execute(  # noqa: SLF001 — intentional: private db
        "SELECT * FROM runs WHERE run_id = ?", (run_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if row is None:
        return None
    rec = dict(zip(cols, row))
    rec["hyperparams"] = json.loads(rec.get("hyperparams") or "{}")
    rec["metrics"] = json.loads(rec.get("metrics") or "{}")
    return rec


def _render(rec: dict) -> str:
    lines = [f"# Run Report — `{rec['run_id']}`", ""]
    lines.append(f"- **Model type**: {rec.get('model_type')}")
    lines.append(f"- **Train time (s)**: {rec.get('train_time_s')}")
    lines.append(f"- **Inference (ms)**: {rec.get('inference_ms')}")
    lines.append(f"- **Model size (MB)**: {rec.get('model_mb')}")
    lines.append(f"- **SHAP runtime (s)**: {rec.get('shap_runtime_s')}")
    lines.append("")
    lines.append("## Hyperparameters")
    for k, v in (rec.get("hyperparams") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Metrics")
    for k, v in (rec.get("metrics") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


class ReportGeneratorTool:
    name: str = "report_generator"
    description: str = (
        "Generates a Markdown report from a leaderboard run. "
        "Input: JSON with 'run_id' and optional 'output_path'."
    )
    task_type: str = "default"

    def __init__(self, leaderboard: Leaderboard | None = None,
                 output_dir: str | Path = "outputs/reports") -> None:
        self._leaderboard = leaderboard
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")
        run_id = payload.get("run_id")
        if not run_id:
            return _err("'run_id' is required.", "ValueError")
        lb = self._leaderboard or Leaderboard()
        try:
            rec = _fetch_run(lb, run_id)
        except sqlite3.DatabaseError as exc:
            return _err(f"leaderboard query failed: {exc}", "DatabaseError")
        if rec is None:
            return _err(f"run_id {run_id!r} not found.", "KeyError")
        out = Path(payload.get("output_path") or (self._dir / f"{run_id}.md"))
        out.parent.mkdir(parents=True, exist_ok=True)
        md = _render(rec)
        out.write_text(md, encoding="utf-8")
        return ToolResult(
            status="ok",
            data={"output_path": str(out), "bytes": out.stat().st_size,
                  "run_id": run_id},
            explanation=f"Wrote Markdown report for run {run_id} → {out}.",
            math_trace=f"leaderboard[{run_id!r}] → Markdown ({out.stat().st_size} B).",
        )

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result), default=str)
