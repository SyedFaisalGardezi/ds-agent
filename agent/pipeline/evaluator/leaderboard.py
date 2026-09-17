"""Leaderboard — SQLite-backed run registry with top-N queries."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.core.types import ToolResult

_DEFAULT_DB = Path("outputs/reports/leaderboard.db")


class Leaderboard:
    """Persist run metrics and rank by an arbitrary metric key."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI/TestClient call endpoints from worker threads while the
        # Leaderboard is constructed on the main thread — relax sqlite's
        # thread check (writes remain serialised by the GIL + commit()).
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                model_type TEXT,
                hyperparams JSON,
                metrics JSON,
                train_time_s REAL,
                inference_ms REAL,
                model_mb REAL,
                shap_runtime_s REAL
            )
        """)
        self._conn.commit()

    def log_run(self, run_id: str, model_type: str, hyperparams: dict, metrics: dict,
                train_time_s: float, inference_ms: float, model_mb: float,
                shap_runtime_s: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
            (run_id, model_type, json.dumps(hyperparams), json.dumps(metrics),
             float(train_time_s), float(inference_ms), float(model_mb),
             float(shap_runtime_s)),
        )
        self._conn.commit()

    def top_n(self, metric: str, n: int = 10,
              higher_is_better: bool = True) -> ToolResult:
        if n < 1:
            return ToolResult(status="error", data=None,
                              explanation="n must be ≥ 1",
                              math_trace="", error="ValueError")
        cur = self._conn.execute("SELECT * FROM runs")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        if not rows:
            return ToolResult(status="ok", data={"runs": []},
                              explanation="Leaderboard is empty.",
                              math_trace="|runs|=0.")
        scored: list[tuple[float, dict]] = []
        for r in rows:
            metrics = json.loads(r["metrics"] or "{}")
            if metric not in metrics:
                continue
            scored.append((float(metrics[metric]), r))
        scored.sort(key=lambda t: t[0], reverse=higher_is_better)
        top = [r for _, r in scored[:n]]
        return ToolResult(
            status="ok",
            data={"runs": top, "metric": metric,
                  "higher_is_better": bool(higher_is_better)},
            explanation=f"Top {len(top)} run(s) by {metric}.",
            math_trace=f"argsort(metrics[{metric!r}], desc={higher_is_better})[:{n}].",
        )

    def close(self) -> None:
        self._conn.close()
