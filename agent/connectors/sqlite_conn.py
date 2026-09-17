"""SQLite connector.

Uses the stdlib `sqlite3` module. Read-only: attaches the DB with
`mode=ro` and blocks DDL/DML at the SQL-guard layer. The `path` may be a
file on disk or `":memory:"`.
"""
from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|"
    r"REVOKE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b(e?_?mail|email_addr)\b", re.IGNORECASE),
    "phone": re.compile(r"\b(phone|mobile|telephone|msisdn)\b", re.IGNORECASE),
    "ssn": re.compile(r"\b(ssn|social_security|national_id|nin)\b", re.IGNORECASE),
    "name": re.compile(r"\b(first_?name|last_?name|full_?name|surname|given_?name)\b", re.IGNORECASE),
    "address": re.compile(r"\b(address|street|postal_?code|zip_?code|postcode)\b", re.IGNORECASE),
    "dob": re.compile(r"\b(dob|date_of_birth|birth_?date)\b", re.IGNORECASE),
    "ip": re.compile(r"\b(ip_?address|client_ip)\b", re.IGNORECASE),
    "cc": re.compile(r"\b(card_?number|credit_?card|cc_num|pan)\b", re.IGNORECASE),
    "geo": re.compile(r"\b(lat|latitude|lon|longitude|geo_point)\b", re.IGNORECASE),
}


class SQLiteConnector(DataConnector):
    """Read-only SQLite connector."""

    name = "sqlite"

    def __init__(
        self,
        path: str | None = None,
        connect_fn: Callable[..., sqlite3.Connection] | None = None,
    ) -> None:
        self._path = path or os.environ.get("SQLITE_PATH")
        self._connect_fn = connect_fn or sqlite3.connect
        self._conn: sqlite3.Connection | None = None

    def _require_connected(self) -> ToolResult | None:
        if self._conn is None:
            return ToolResult(
                status="error",
                data=None,
                explanation="Not connected. Call connect() first.",
                math_trace="",
                error="ConnectionError: no active connection",
            )
        return None

    @staticmethod
    def _guard_sql(sql: str) -> ToolResult | None:
        if _FORBIDDEN_SQL.search(sql):
            return ToolResult(
                status="error",
                data=None,
                explanation="Destructive SQL keyword detected. Connector is read-only.",
                math_trace="",
                error="SafetyError: destructive statement",
            )
        return None

    def connect(self) -> ToolResult:
        if not self._path:
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing SQLite path. Provide path= or SQLITE_PATH env.",
                math_trace="",
                error="ConfigError: missing path",
            )
        try:
            if self._path == ":memory:":
                self._conn = self._connect_fn(":memory:")
            else:
                p = Path(self._path)
                if not p.exists():
                    return ToolResult(
                        status="error",
                        data=None,
                        explanation=f"Path not found: {self._path!r}.",
                        math_trace="",
                        error="FileNotFoundError: path",
                    )
                # read-only URI connection
                uri = f"file:{p.resolve()}?mode=ro"
                self._conn = self._connect_fn(uri, uri=True)
            self._conn.row_factory = sqlite3.Row
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open SQLite: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(
            status="ok",
            data={"path": str(self._path)},
            explanation=f"Connected to SQLite at {self._path!r} (read-only).",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        assert self._conn is not None  # narrowed by _require_connected
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        names = [row[0] for row in cur.fetchall()]
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} table(s).",
            math_trace=f"sqlite_master scan → {len(names)} tables.",
        )

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        if not _SAFE_IDENT.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe table name: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table})")
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"describe_table failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        if not rows:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unknown table: {table!r}.",
                math_trace="",
                error="KeyError: table",
            )
        cols = [
            {"name": r[1], "type": r[2], "nullable": not bool(r[3])}
            for r in rows
        ]
        return ToolResult(
            status="ok",
            data={"table": table, "columns": cols},
            explanation=f"{table!r}: {len(cols)} column(s).",
            math_trace=f"PRAGMA table_info({table}) → {len(cols)} columns.",
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        sql = (query or "").strip()
        if not sql:
            return ToolResult(
                status="error",
                data=None,
                explanation="Empty query.",
                math_trace="",
                error="ValueError: query is required",
            )
        if (guard := self._guard_sql(sql)) is not None:
            return guard
        assert self._conn is not None
        try:
            cur = self._conn.cursor()
            cur.execute(sql, params or {})
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
            records = [dict(zip(cols, row)) for row in rows]
        except sqlite3.Error as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Query failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(
            status="ok",
            data={"records": records, "rowcount": len(records), "columns": cols},
            explanation=f"Query returned {len(records)} row(s).",
            math_trace=f"π[{','.join(cols)}] → {len(records)} rows.",
        )

    def estimate_cost(self, query: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        sql = (query or "").strip()
        if not sql:
            return ToolResult(
                status="error",
                data=None,
                explanation="Empty query.",
                math_trace="",
                error="ValueError: query is required",
            )
        if (guard := self._guard_sql(sql)) is not None:
            return guard
        assert self._conn is not None
        try:
            cur = self._conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {sql}", {})
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"EXPLAIN failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        plan = [tuple(r) for r in rows]
        uses_scan = any("SCAN" in str(step) for step in plan)
        return ToolResult(
            status="ok",
            data={
                "plan": plan,
                "full_scan_detected": uses_scan,
                "rows_estimate": None,  # sqlite doesn't expose cardinality estimates here
            },
            explanation=(
                f"EXPLAIN QUERY PLAN returned {len(plan)} step(s); "
                f"full scan detected: {uses_scan}."
            ),
            math_trace=(
                f"sqlite EXPLAIN QUERY PLAN steps={len(plan)}; "
                f"scan={'yes' if uses_scan else 'no'}."
            ),
        )

    def detect_pii(self, table: str) -> ToolResult:
        desc = self.describe_table(table)
        if desc.status != "ok":
            return desc
        flagged: dict[str, list[str]] = {}
        for col in desc.data["columns"]:
            colname = col["name"]
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(colname):
                    flagged.setdefault(pii_type, []).append(colname)
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "pii_columns": flagged,
                "columns_scanned": len(desc.data["columns"]),
            },
            explanation=(
                f"Scanned {len(desc.data['columns'])} column(s) on {table}; "
                f"flagged {sum(len(v) for v in flagged.values())} as PII across "
                f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'}."
            ),
            math_trace=f"Heuristic regex over {len(_PII_PATTERNS)} PII categories.",
        )

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


def register() -> SQLiteConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = SQLiteConnector()
    ConnectorRegistry.get().register(conn)
    return conn
