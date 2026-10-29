"""PostgreSQL connector.

Mirrors the design of `snowflake_conn.py`:
  * DB-API 2.0 surface, so the whole code path is testable by injecting a fake
    `connect_fn`. `psycopg[binary]` (v3) is the expected default driver.
  * Read-only: `query()` / `estimate_cost()` refuse DDL/DML via the shared
    regex guard.
  * `estimate_cost()` uses `EXPLAIN (FORMAT JSON)` — Postgres' plan node
    exposes `Plan Rows`, `Plan Width`, and `Total Cost`, which map naturally
    to the rows / bytes / cost fields of the ToolResult contract.
  * `detect_pii()` is a column-name heuristic shared with Snowflake.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|MERGE|GRANT|REVOKE|CREATE)\b",
    re.IGNORECASE,
)

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

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.\"]*$")


class PostgresConnector(DataConnector):
    """Read-only PostgreSQL connector conforming to the DataConnector ABC.

    Parameters
    ----------
    connect_fn
        DB-API 2.0 connection factory. Defaults to `psycopg.connect`.
    **config
        libpq kwargs: `host`, `port`, `user`, `password`, `dbname`, `sslmode`,
        `options`. Missing fields fall back to `POSTGRES_<NAME>` env vars. A
        full conninfo `dsn` string may also be supplied.
    """

    name = "postgres"

    _ENV_KEYS = ("host", "port", "user", "password", "dbname", "sslmode", "options", "dsn")
    _REQUIRED_EITHER = ("dsn", "host")

    def __init__(
        self,
        connect_fn: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._connect_fn = connect_fn
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._conn: Any = None

    # ---- helpers -----------------------------------------------------------

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get(f"POSTGRES_{key.upper()}")
            if env_val:
                cfg[key] = env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_connect_fn(self) -> Callable[..., Any]:
        if self._connect_fn is not None:
            return self._connect_fn
        import psycopg  # local import → driver is optional for unit tests
        return psycopg.connect

    @staticmethod
    def _rows_to_records(cursor: Any) -> list[dict[str, Any]]:
        if cursor.description is None:
            return []
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def _require_connection(self) -> ToolResult | None:
        if self._conn is None:
            return ToolResult(
                status="error",
                data=None,
                explanation="Not connected. Call connect() first.",
                math_trace="",
                error="ConnectionError: no active connection",
            )
        return None

    # ---- DataConnector API -------------------------------------------------

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        if not any(cfg.get(k) for k in self._REQUIRED_EITHER):
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing Postgres config: provide either 'dsn' or 'host'.",
                math_trace="",
                error="ConfigError: missing dsn/host",
            )
        try:
            if "dsn" in cfg:
                self._conn = self._resolve_connect_fn()(cfg["dsn"])
            else:
                kwargs = {k: v for k, v in cfg.items() if k != "dsn"}
                self._conn = self._resolve_connect_fn()(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Postgres connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {k: ("***" if k in ("password", "dsn") else v) for k, v in cfg.items()}
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=f"Connected to Postgres host={cfg.get('host', '(dsn)')!r}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connection()) is not None:
            return err
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog', 'information_schema') "
                "AND schema_name NOT LIKE 'pg_toast%' "
                "AND schema_name NOT LIKE 'pg_temp%' "
                "ORDER BY schema_name"
            )
            names = [r[0] for r in cur.fetchall()]
            return ToolResult(
                status="ok",
                data=names,
                explanation=f"Listed {len(names)} user schema(s).",
                math_trace=f"information_schema.schemata → {len(names)} rows.",
            )
        finally:
            cur.close()

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connection()) is not None:
            return err
        if not _SAFE_IDENT.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe table identifier: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        parts = table.split(".")
        if len(parts) == 2:
            schema, tbl = parts
        elif len(parts) == 1:
            schema, tbl = "public", parts[0]
        else:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Expected schema.table or table, got {table!r}.",
                math_trace="",
                error="ValueError: table",
            )
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s "
                "ORDER BY ordinal_position",
                (schema, tbl),
            )
            rows = cur.fetchall()
            cols = [
                {"name": r[0], "type": r[1], "nullable": r[2] == "YES"}
                for r in rows
            ]
            return ToolResult(
                status="ok",
                data={"table": table, "columns": cols},
                explanation=f"{table}: {len(cols)} column(s).",
                math_trace=f"information_schema.columns → {len(cols)} rows.",
            )
        finally:
            cur.close()

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        if (err := self._require_connection()) is not None:
            return err
        sql = query.strip().rstrip(";")
        if not sql:
            return ToolResult(
                status="error",
                data=None,
                explanation="Empty query.",
                math_trace="",
                error="ValueError: query is required",
            )
        if _FORBIDDEN_SQL.search(sql):
            return ToolResult(
                status="error",
                data=None,
                explanation="Refusing to run destructive SQL. Connector is read-only.",
                math_trace="",
                error="SafetyError: destructive statement",
            )
        cur = self._conn.cursor()
        try:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            records = self._rows_to_records(cur)
            cols = [d[0] for d in cur.description] if cur.description else []
            return ToolResult(
                status="ok",
                data={"records": records, "columns": cols, "rowcount": len(records)},
                explanation=f"Returned {len(records)} row(s) × {len(cols)} col(s).",
                math_trace=f"Query rowcount: {len(records)}; columns: {len(cols)}.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Query failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            cur.close()

    def estimate_cost(self, query: str) -> ToolResult:
        if (err := self._require_connection()) is not None:
            return err
        sql = query.strip().rstrip(";")
        if not sql:
            return ToolResult(
                status="error",
                data=None,
                explanation="Empty query.",
                math_trace="",
                error="ValueError: query is required",
            )
        if _FORBIDDEN_SQL.search(sql):
            return ToolResult(
                status="error",
                data=None,
                explanation="Refusing to estimate cost of destructive SQL.",
                math_trace="",
                error="SafetyError: destructive statement",
            )
        cur = self._conn.cursor()
        try:
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            rows = cur.fetchall()
            plan_raw = rows[0][0] if rows else None
            # psycopg3 already parses JSON (-> list[dict]); psycopg2 returns str.
            if isinstance(plan_raw, str):
                try:
                    plan = json.loads(plan_raw)
                except json.JSONDecodeError:
                    plan = [{"raw": plan_raw}]
            else:
                plan = plan_raw if plan_raw is not None else []

            top = plan[0].get("Plan", {}) if plan and isinstance(plan[0], dict) else {}
            rows_est = top.get("Plan Rows")
            width = top.get("Plan Width")
            total_cost = top.get("Total Cost")
            bytes_est = (rows_est or 0) * (width or 0) if rows_est and width else None

            math_trace = (
                f"Estimated rows: {rows_est}. Estimated row width (bytes): {width}. "
                f"Estimated total cost (planner units): {total_cost}.\n"
                f"Estimated bytes scanned ≈ rows × width = {bytes_est}."
            )
            return ToolResult(
                status="ok",
                data={
                    "rows_estimate": rows_est,
                    "row_width_bytes": width,
                    "bytes_estimate": bytes_est,
                    "total_cost": total_cost,
                    "plan": plan,
                },
                explanation=(
                    f"Planner estimate: rows≈{rows_est}, bytes≈{bytes_est}, "
                    f"cost≈{total_cost}."
                ),
                math_trace=math_trace,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"EXPLAIN failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            cur.close()

    def detect_pii(self, table: str) -> ToolResult:
        desc = self.describe_table(table)
        if desc.status != "ok":
            return desc
        flagged: dict[str, list[str]] = {}
        for col in desc.data.get("columns", []):
            colname = col.get("name") or ""
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(colname):
                    flagged.setdefault(pii_type, []).append(colname)
        return ToolResult(
            status="ok",
            data={"table": table, "pii_columns": flagged, "columns_scanned": len(desc.data.get("columns", []))},
            explanation=(
                f"Scanned {len(desc.data.get('columns', []))} columns on {table}; "
                f"flagged {sum(len(v) for v in flagged.values())} as potential PII across "
                f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'}."
            ),
            math_trace=(
                "Heuristic: column-name regex over "
                f"{len(_PII_PATTERNS)} PII categories. No value-level sampling."
            ),
        )

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


def register() -> PostgresConnector:
    """Register the connector in the global ConnectorRegistry (no connect())."""
    from agent.connectors.registry import ConnectorRegistry
    conn = PostgresConnector()
    ConnectorRegistry.get().register(conn)
    return conn
