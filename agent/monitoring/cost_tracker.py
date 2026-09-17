"""CostTracker — SQLite-backed ledger of query costs."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.core.types import ToolResult

_DEFAULT_DB = Path("outputs/reports/cost_tracker.db")


class CostTracker:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS query_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts DATETIME,
                connector TEXT,
                query TEXT,
                estimated_usd REAL
            )
        """)
        self._conn.commit()

    def log_query_cost(self, connector: str, query: str,
                       estimated_usd: float) -> None:
        self._conn.execute(
            "INSERT INTO query_costs (ts, connector, query, estimated_usd) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(),
             connector, query, float(estimated_usd)),
        )
        self._conn.commit()

    def total_cost(self, since_days: int = 30) -> ToolResult:
        if since_days < 0:
            return ToolResult(status="error", data=None,
                              explanation="since_days must be ≥ 0",
                              math_trace="", error="ValueError")
        cutoff = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
        cur = self._conn.execute(
            "SELECT connector, SUM(estimated_usd), COUNT(*) "
            "FROM query_costs WHERE ts >= ? GROUP BY connector", (cutoff,),
        )
        per_conn = {}
        total = 0.0
        n = 0
        for conn, usd, count in cur.fetchall():
            usd = float(usd or 0.0)
            per_conn[conn] = {"usd": usd, "count": int(count)}
            total += usd
            n += int(count)
        return ToolResult(
            status="ok",
            data={"total_usd": float(total), "n_queries": n,
                  "since_days": since_days, "per_connector": per_conn},
            explanation=f"${total:.4f} across {n} query/queries in last {since_days}d.",
            math_trace="total = Σ_q estimated_usd_q for q with ts ≥ now − Δ.",
        )

    def close(self) -> None:
        self._conn.close()
