"""MySQL connector.

Uses an injectable `connect_fn` that returns a DB-API 2.0 connection. In
production the default is `pymysql.connect`; in tests we pass a duck-typed
fake. Read-only — destructive SQL is blocked by regex guard. Identifier
safety: schema / table names are validated before interpolating into SQL.
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
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|"
    r"REVOKE|RENAME|LOAD|HANDLER|LOCK|UNLOCK)\b",
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


class MySQLConnector(DataConnector):
    """Read-only MySQL connector."""

    name = "mysql"

    _ENV_KEYS = ("host", "port", "user", "password", "database")

    def __init__(
        self,
        connect_fn: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._connect_fn = connect_fn
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._conn: Any = None

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get(f"MYSQL_{key.upper()}")
            if env_val is not None:
                cfg[key] = int(env_val) if key == "port" else env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_factory(self) -> Callable[..., Any]:
        if self._connect_fn is not None:
            return self._connect_fn
        import pymysql  # pragma: no cover
        return pymysql.connect

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
        cfg = self._effective_config()
        if not cfg.get("host") and not cfg.get("database"):
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing MySQL config: host or database is required.",
                math_trace="",
                error="ConfigError: missing host/database",
            )
        try:
            self._conn = self._resolve_factory()(**cfg)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open MySQL connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {k: ("***" if k == "password" else v) for k, v in cfg.items()}
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=f"Connected to MySQL at {cfg.get('host')!r} db={cfg.get('database')!r}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN "
                "('mysql','sys','information_schema','performance_schema') "
                "ORDER BY schema_name"
            )
            rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"list_schemas failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        names = [r[0] for r in rows]
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} schema(s).",
            math_trace=f"information_schema.schemata → {len(names)} rows.",
        )

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        parts = table.split(".")
        if len(parts) == 1:
            schema, name = None, parts[0]
        elif len(parts) == 2:
            schema, name = parts
        else:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Too many dots in identifier: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        if not _SAFE_IDENT.match(name) or (schema and not _SAFE_IDENT.match(schema)):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe identifier: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        cur = self._conn.cursor()
        try:
            if schema:
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
                    (schema, name),
                )
            else:
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_name=%s AND table_schema=DATABASE() "
                    "ORDER BY ordinal_position",
                    (name,),
                )
            rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
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
            {"name": r[0], "type": r[1], "nullable": str(r[2]).upper() == "YES"}
            for r in rows
        ]
        return ToolResult(
            status="ok",
            data={"table": table, "columns": cols},
            explanation=f"{table}: {len(cols)} column(s).",
            math_trace=f"information_schema.columns → {len(cols)} rows.",
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
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Query failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        records = [dict(zip(cols, row)) for row in rows]
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
        cur = self._conn.cursor()
        try:
            cur.execute(f"EXPLAIN FORMAT=JSON {sql}")
            row = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"EXPLAIN failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        plan_json = row[0][0] if row else None
        if isinstance(plan_json, (bytes, bytearray)):
            plan_json = plan_json.decode()
        plan: Any = plan_json
        if isinstance(plan_json, str):
            try:
                plan = json.loads(plan_json)
            except json.JSONDecodeError:
                plan = plan_json
        rows_estimate = _dig(plan, "rows_examined_per_scan") or _dig(plan, "rows")
        cost = _dig(plan, "query_cost") or _dig(plan, "cost_info")
        try:
            rows_estimate = int(rows_estimate) if rows_estimate is not None else None
        except (TypeError, ValueError):
            rows_estimate = None
        return ToolResult(
            status="ok",
            data={
                "rows_estimate": rows_estimate,
                "cost": cost,
                "plan": plan,
            },
            explanation=f"EXPLAIN estimate: rows≈{rows_estimate}, cost={cost}.",
            math_trace=f"MySQL EXPLAIN FORMAT=JSON → rows≈{rows_estimate}, cost={cost}.",
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


def _dig(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _dig(v, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _dig(item, key)
            if found is not None:
                return found
    return None


def register() -> MySQLConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = MySQLConnector()
    ConnectorRegistry.get().register(conn)
    return conn
