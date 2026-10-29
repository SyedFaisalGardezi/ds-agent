"""Snowflake connector.

Design notes
------------
* The Snowflake driver (`snowflake-connector-python`) is a DB-API 2.0 client.
  Every SQL path in this class goes through `cursor.execute` → `fetchall`, so the
  whole thing is testable by injecting a fake `connect_fn` that returns a
  duck-typed connection. No Snowflake account is required for unit tests.
* Configuration is read from environment variables by default
  (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, ...), but explicit kwargs to
  `__init__` win. Missing required creds are only checked at `connect()` time so
  the class can be instantiated for introspection / registry binding without
  credentials present.
* Read-only: `query()` refuses DDL/DML via a regex pre-scan that mirrors the
  pattern used in `agent.tools.nl2sql`. Cost estimates use `EXPLAIN USING JSON`
  and surface partition/byte stats when Snowflake returns them.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

# Matches the destructive-SQL guard in nl2sql — keep them in sync.
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|MERGE|GRANT|REVOKE|CREATE)\b",
    re.IGNORECASE,
)

# Heuristic PII column-name patterns — deliberately conservative. A dedicated
# PII-detection tool will expand this with value-level sampling later.
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


class SnowflakeConnector(DataConnector):
    """Read-only Snowflake connector conforming to the DataConnector ABC.

    Parameters
    ----------
    connect_fn
        Callable returning an already-open DB-API 2.0 connection. Defaults to
        `snowflake.connector.connect`. Override in tests.
    **config
        Standard Snowflake kwargs (`account`, `user`, `password`, `warehouse`,
        `database`, `schema`, `role`). Any that are omitted are pulled from
        `SNOWFLAKE_<NAME>` env vars at `connect()` time.
    """

    name = "snowflake"

    _ENV_KEYS = ("account", "user", "password", "warehouse", "database", "schema", "role")
    _REQUIRED_KEYS = ("account", "user", "password")

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
            env_val = os.environ.get(f"SNOWFLAKE_{key.upper()}")
            if env_val:
                cfg[key] = env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_connect_fn(self) -> Callable[..., Any]:
        if self._connect_fn is not None:
            return self._connect_fn
        import snowflake.connector as sf  # local import so driver is optional for unit tests
        return sf.connect

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
        missing = [k for k in self._REQUIRED_KEYS if not cfg.get(k)]
        if missing:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Missing required Snowflake config: {missing}.",
                math_trace="",
                error="ConfigError: missing " + ",".join(missing),
            )
        try:
            self._conn = self._resolve_connect_fn()(**cfg)
        except Exception as exc:  # noqa: BLE001 — surface driver errors verbatim
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Snowflake connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {k: ("***" if k == "password" else v) for k, v in cfg.items()}
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=f"Connected to Snowflake account {cfg.get('account')!r}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connection()) is not None:
            return err
        cur = self._conn.cursor()
        try:
            cur.execute("SHOW SCHEMAS")
            records = self._rows_to_records(cur)
            # SHOW SCHEMAS returns many metadata columns; expose only the names.
            names = [r.get("name") or r.get("NAME") for r in records]
            return ToolResult(
                status="ok",
                data=names,
                explanation=f"Listed {len(names)} schema(s).",
                math_trace=f"SHOW SCHEMAS → {len(names)} rows.",
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
        cur = self._conn.cursor()
        try:
            cur.execute(f"DESCRIBE TABLE {table}")
            records = self._rows_to_records(cur)
            cols = [
                {"name": r.get("name") or r.get("NAME"), "type": r.get("type") or r.get("TYPE")}
                for r in records
            ]
            return ToolResult(
                status="ok",
                data={"table": table, "columns": cols},
                explanation=f"{table}: {len(cols)} column(s).",
                math_trace=f"DESCRIBE TABLE {table} → {len(cols)} columns.",
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
            cur.execute(sql, params or {})
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
            cur.execute(f"EXPLAIN USING JSON {sql}")
            rows = cur.fetchall()
            plan_text = rows[0][0] if rows else "{}"
            try:
                plan = json.loads(plan_text) if isinstance(plan_text, str) else plan_text
            except (TypeError, json.JSONDecodeError):
                plan = {"raw": plan_text}

            partitions = _dig(plan, "partitionsTotal") or _dig(plan, "partitions")
            bytes_scanned = _dig(plan, "bytesAssigned") or _dig(plan, "bytes")
            rows_est = _dig(plan, "rowCount") or _dig(plan, "rows")

            math_trace = (
                f"Estimated partitions: {partitions}. Estimated bytes: {bytes_scanned}. "
                f"Estimated rows: {rows_est}.\n"
                f"Cost \\\\propto bytes\\\\_scanned — see Snowflake credit pricing."
            )
            return ToolResult(
                status="ok",
                data={
                    "partitions": partitions,
                    "bytes": bytes_scanned,
                    "rows_estimate": rows_est,
                    "plan": plan,
                },
                explanation=(
                    f"EXPLAIN estimate: rows≈{rows_est}, bytes≈{bytes_scanned}, "
                    f"partitions≈{partitions}."
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


_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.\"]*$")


def _dig(obj: Any, key: str) -> Any:
    """Recursively search a nested dict/list for the first occurrence of `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _dig(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _dig(item, key)
            if found is not None:
                return found
    return None


def register() -> SnowflakeConnector:
    """Register the connector in the global ConnectorRegistry (no connect())."""
    from agent.connectors.registry import ConnectorRegistry
    conn = SnowflakeConnector()
    ConnectorRegistry.get().register(conn)
    return conn
