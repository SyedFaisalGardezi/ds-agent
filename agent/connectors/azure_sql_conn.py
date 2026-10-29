"""Azure SQL Database (T-SQL / MS SQL Server) connector.

Uses an injectable `connect_fn` returning a DB-API 2.0 connection. In
production the default is `pyodbc.connect` with an ODBC connection string
built from discrete kwargs. Read-only — destructive T-SQL is blocked by
regex guard. Identifier safety: schema / table are validated before
interpolation. `estimate_cost` uses `SET SHOWPLAN_XML ON` (T-SQL's plan
surface) and falls back gracefully if the driver doesn't support it.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|MERGE|REPLACE|GRANT|"
    r"REVOKE|RENAME|BULK|EXEC|EXECUTE|BACKUP|RESTORE|SHUTDOWN)\b",
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


def _build_odbc_dsn(cfg: dict[str, Any]) -> str:
    driver = cfg.get("driver") or "{ODBC Driver 18 for SQL Server}"
    parts = [f"DRIVER={driver}"]
    if cfg.get("server"):
        port = cfg.get("port")
        parts.append(f"SERVER={cfg['server']}" + (f",{port}" if port else ""))
    if cfg.get("database"):
        parts.append(f"DATABASE={cfg['database']}")
    if cfg.get("uid") or cfg.get("user"):
        parts.append(f"UID={cfg.get('uid') or cfg.get('user')}")
    if cfg.get("pwd") or cfg.get("password"):
        parts.append(f"PWD={cfg.get('pwd') or cfg.get('password')}")
    if cfg.get("encrypt", True):
        parts.append("Encrypt=yes")
    if cfg.get("trust_server_certificate"):
        parts.append("TrustServerCertificate=yes")
    return ";".join(parts) + ";"


class AzureSQLConnector(DataConnector):
    """Read-only Azure SQL / MS SQL Server connector."""

    name = "azure_sql"

    _ENV_KEYS = (
        "server", "port", "database", "user", "password", "driver",
        "encrypt", "trust_server_certificate", "dsn",
    )

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
            env_val = os.environ.get(f"AZURE_SQL_{key.upper()}")
            if env_val is not None:
                if key == "port":
                    cfg[key] = int(env_val)
                elif key in ("encrypt", "trust_server_certificate"):
                    cfg[key] = env_val.lower() in ("1", "true", "yes")
                else:
                    cfg[key] = env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_factory(self) -> Callable[..., Any]:
        if self._connect_fn is not None:
            return self._connect_fn
        import pyodbc  # pragma: no cover
        return pyodbc.connect

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
                explanation="Destructive T-SQL keyword detected. Connector is read-only.",
                math_trace="",
                error="SafetyError: destructive statement",
            )
        return None

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        if not cfg.get("dsn") and not (cfg.get("server") and cfg.get("database")):
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing Azure SQL config: provide 'dsn' or 'server'+'database'.",
                math_trace="",
                error="ConfigError: missing dsn or server+database",
            )
        try:
            factory = self._resolve_factory()
            dsn = cfg.get("dsn") or _build_odbc_dsn(cfg)
            self._conn = factory(dsn)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Azure SQL connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {
            k: ("***" if k in ("password", "pwd", "dsn") else v)
            for k, v in cfg.items()
        }
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=(
                f"Connected to Azure SQL "
                f"server={cfg.get('server')!r} db={cfg.get('database')!r}."
            ),
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
                "('sys','INFORMATION_SCHEMA','guest','db_owner','db_accessadmin',"
                "'db_securityadmin','db_ddladmin','db_backupoperator',"
                "'db_datareader','db_datawriter','db_denydatareader',"
                "'db_denydatawriter') "
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
            schema, name = "dbo", parts[0]
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
        if not _SAFE_IDENT.match(name) or not _SAFE_IDENT.match(schema):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe identifier: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? "
                "ORDER BY ordinal_position",
                (schema, name),
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
            data={"table": f"{schema}.{name}", "columns": cols},
            explanation=f"{schema}.{name}: {len(cols)} column(s).",
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
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
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
        """Use `SET SHOWPLAN_XML ON` to surface the estimated plan cost."""
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
        plan_xml: str | None = None
        try:
            cur.execute("SET SHOWPLAN_XML ON")
            cur.execute(sql)
            rows = cur.fetchall()
            if rows:
                plan_xml = str(rows[0][0])
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"SHOWPLAN failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                cur.execute("SET SHOWPLAN_XML OFF")
            except Exception:  # noqa: BLE001
                pass

        rows_estimate = None
        total_subtree_cost = None
        if plan_xml:
            m = re.search(r'StatementEstRows="([0-9.eE+\-]+)"', plan_xml)
            if m:
                try:
                    rows_estimate = int(float(m.group(1)))
                except (TypeError, ValueError):
                    rows_estimate = None
            m = re.search(r'StatementSubTreeCost="([0-9.eE+\-]+)"', plan_xml)
            if m:
                try:
                    total_subtree_cost = float(m.group(1))
                except (TypeError, ValueError):
                    total_subtree_cost = None
        return ToolResult(
            status="ok",
            data={
                "rows_estimate": rows_estimate,
                "total_subtree_cost": total_subtree_cost,
                "plan_xml": plan_xml,
            },
            explanation=(
                f"SHOWPLAN estimate: rows≈{rows_estimate}, "
                f"subtree_cost={total_subtree_cost}."
            ),
            math_trace=(
                f"T-SQL SHOWPLAN_XML → StatementEstRows={rows_estimate}, "
                f"StatementSubTreeCost={total_subtree_cost}."
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


def register() -> AzureSQLConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = AzureSQLConnector()
    ConnectorRegistry.get().register(conn)
    return conn
