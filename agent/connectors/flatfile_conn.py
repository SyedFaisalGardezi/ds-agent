"""Flat-file connector.

Treats a directory of files as a read-only "database" where each supported file
is a table named after its stem (`sales.csv` → table `sales`). A single file
path is also accepted and exposed as a one-table database.

SQL support is provided via an in-memory SQLite DB: referenced tables are
loaded lazily into sqlite on each `query()`. This keeps the implementation
dependency-light (no duckdb / pandasql needed) at the cost of SQLite dialect
quirks (no `FULL OUTER JOIN`, etc.). A future upgrade path is to swap the
SQL engine for DuckDB behind the same interface.

Safety model:
  * All paths resolve under `DS_AGENT_FILE_ROOT` (default `./outputs`). Escape
    attempts raise `PathEscapeError` from `agent.tools.file_io`.
  * `query()` blocks DDL/DML via the same regex as the DB connectors.
  * File size cap (200 MB) applies per table load; per-query cost estimate
    aggregates referenced-table sizes.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult
from agent.tools.file_io import PathEscapeError  # re-use the existing guard type

_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|MERGE|GRANT|REVOKE|CREATE|ATTACH|DETACH|PRAGMA)\b",
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

# file-extension → (pandas loader name, format label)
_LOADERS: dict[str, tuple[str, str]] = {
    ".csv": ("read_csv", "csv"),
    ".tsv": ("read_csv", "tsv"),
    ".parquet": ("read_parquet", "parquet"),
    ".pq": ("read_parquet", "parquet"),
    ".json": ("read_json", "json"),
    ".jsonl": ("read_json", "jsonl"),
}

_MAX_FILE_BYTES = 200 * 1024 * 1024

# Extract identifier-like table names following FROM / JOIN tokens.
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

_DEFAULT_ROOT = Path(os.environ.get("DS_AGENT_FILE_ROOT", "outputs")).resolve()


class FlatFileConnector(DataConnector):
    """Directory-of-files or single-file connector with sqlite-backed SQL."""

    name = "flatfile"

    def __init__(self, path: str | Path = "", root: Path | None = None) -> None:
        self._root = (root or _DEFAULT_ROOT).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._path_arg = str(path) if path else ""
        self._resolved: Path | None = None
        self._tables: dict[str, Path] = {}  # name → file path

    # ---- helpers -----------------------------------------------------------

    def _resolve_under_root(self, rel: str) -> Path:
        candidate = (self._root / rel).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise PathEscapeError(
                f"Path {rel!r} escapes the sandbox root {self._root}."
            ) from exc
        return candidate

    def _discover_tables(self, base: Path) -> dict[str, Path]:
        if base.is_file():
            return {base.stem: base} if base.suffix.lower() in _LOADERS else {}
        tables: dict[str, Path] = {}
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix.lower() in _LOADERS:
                # Name collisions resolve to the first-seen file (sorted order).
                tables.setdefault(p.stem, p)
        return tables

    def _load_table(self, name: str) -> tuple[pd.DataFrame | None, ToolResult | None]:
        file = self._tables.get(name)
        if file is None:
            return None, ToolResult(
                status="error",
                data=None,
                explanation=f"Unknown table {name!r}. Known: {sorted(self._tables)}.",
                math_trace="",
                error="ValueError: unknown table",
            )
        size = file.stat().st_size
        if size > _MAX_FILE_BYTES:
            return None, ToolResult(
                status="error",
                data=None,
                explanation=f"Table {name!r} exceeds {_MAX_FILE_BYTES} byte cap ({size} bytes).",
                math_trace="",
                error="SizeLimitExceeded",
            )
        loader_name, _fmt = _LOADERS[file.suffix.lower()]
        loader = getattr(pd, loader_name)
        try:
            if file.suffix.lower() == ".tsv":
                df = loader(file, sep="\t")
            elif file.suffix.lower() == ".jsonl":
                df = loader(file, lines=True)
            else:
                df = loader(file)
        except Exception as exc:  # noqa: BLE001
            return None, ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to load {name!r}: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return df, None

    def _require_connected(self) -> ToolResult | None:
        if self._resolved is None:
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
        if not self._path_arg:
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing path. Provide a file or directory path (relative to the sandbox root).",
                math_trace="",
                error="ConfigError: missing path",
            )
        try:
            resolved = self._resolve_under_root(self._path_arg)
        except PathEscapeError as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation=str(exc),
                math_trace="",
                error="PathEscapeError",
            )
        if not resolved.exists():
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Path not found: {self._path_arg!r}.",
                math_trace="",
                error="FileNotFoundError",
            )
        tables = self._discover_tables(resolved)
        if not tables:
            return ToolResult(
                status="error",
                data=None,
                explanation=(
                    f"No supported files at {self._path_arg!r}. "
                    f"Supported extensions: {sorted(_LOADERS)}."
                ),
                math_trace="",
                error="NoTablesFound",
            )
        self._resolved = resolved
        self._tables = tables
        return ToolResult(
            status="ok",
            data={"path": str(resolved.relative_to(self._root)), "tables": sorted(tables)},
            explanation=f"Connected to {resolved.name!r} with {len(tables)} table(s).",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        # For flatfile there is one "schema" (the directory) whose tables ARE
        # the schemas for registry purposes. Return the table list directly so
        # the ConnectorRegistry indexer can describe each.
        if (err := self._require_connected()) is not None:
            return err
        names = sorted(self._tables)
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} table(s).",
            math_trace=f"File discovery → {len(names)} tables.",
        )

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        df, err = self._load_table(table)
        if err is not None:
            return err
        assert df is not None
        rows, cols = df.shape
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        file = self._tables[table]
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "file": str(file.relative_to(self._root)),
                "columns": [{"name": c, "type": dtypes[c], "nullable": bool(df[c].isna().any())} for c in df.columns],
                "shape": [rows, cols],
                "bytes": file.stat().st_size,
            },
            explanation=f"{table}: {rows} rows × {cols} cols ({file.stat().st_size} bytes).",
            math_trace=_tabular_math_trace(df),
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        if (err := self._require_connected()) is not None:
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

        referenced = _extract_tables(sql)
        if not referenced:
            return ToolResult(
                status="error",
                data=None,
                explanation="No FROM/JOIN tables found in query.",
                math_trace="",
                error="ValueError: no tables",
            )

        unknown = [t for t in referenced if t not in self._tables]
        if unknown:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unknown table(s): {unknown}. Known: {sorted(self._tables)}.",
                math_trace="",
                error="ValueError: unknown table",
            )

        # Load referenced tables into a fresh in-memory sqlite DB.
        con = sqlite3.connect(":memory:")
        try:
            total_bytes = 0
            total_rows = 0
            for t in referenced:
                df, err = self._load_table(t)
                if err is not None:
                    return err
                assert df is not None
                df.to_sql(t, con, index=False, if_exists="replace")
                total_rows += len(df)
                total_bytes += self._tables[t].stat().st_size
            try:
                cur = con.execute(sql, tuple(params.values()) if isinstance(params, dict) else params or ())
            except sqlite3.Error as exc:
                return ToolResult(
                    status="error",
                    data=None,
                    explanation=f"SQL execution failed (SQLite dialect): {exc}",
                    math_trace="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            records = [dict(zip(cols, r)) for r in rows]
            return ToolResult(
                status="ok",
                data={"records": records, "columns": cols, "rowcount": len(records)},
                explanation=(
                    f"Returned {len(records)} row(s) × {len(cols)} col(s) "
                    f"from tables {sorted(referenced)}."
                ),
                math_trace=(
                    f"Loaded {len(referenced)} table(s), {total_rows} row(s) total, "
                    f"{total_bytes} input bytes. SQL engine: SQLite (stdlib)."
                ),
            )
        finally:
            con.close()

    def estimate_cost(self, query: str) -> ToolResult:
        """Approximate cost: sum of referenced-table file sizes.

        Rationale: flat-file scans cost ≈ O(bytes_read), and we load the full
        table before querying. No planner, no statistics — the math_trace
        reflects that honestly.
        """
        if (err := self._require_connected()) is not None:
            return err
        sql = query.strip()
        if not sql:
            return ToolResult(
                status="error",
                data=None,
                explanation="Empty query.",
                math_trace="",
                error="ValueError: query is required",
            )
        referenced = _extract_tables(sql)
        unknown = [t for t in referenced if t not in self._tables]
        if unknown:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unknown table(s): {unknown}.",
                math_trace="",
                error="ValueError: unknown table",
            )
        per_table = {t: self._tables[t].stat().st_size for t in referenced}
        total = sum(per_table.values())
        return ToolResult(
            status="ok",
            data={"bytes_estimate": total, "per_table_bytes": per_table, "tables": sorted(referenced)},
            explanation=(
                f"Estimated scan: {total} bytes across {len(referenced)} table(s)."
                if referenced
                else "No tables referenced."
            ),
            math_trace=(
                "Flat-file cost ≈ Σ file_size(table_i) for tables referenced via FROM/JOIN. "
                f"Σ = {total} bytes."
            ),
        )

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
            data={
                "table": table,
                "pii_columns": flagged,
                "columns_scanned": len(desc.data.get("columns", [])),
            },
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
        self._resolved = None
        self._tables = {}


# ---- helpers ---------------------------------------------------------------


def _extract_tables(sql: str) -> list[str]:
    """Pull identifier tokens following FROM / JOIN."""
    # De-duplicate but preserve first-seen order.
    seen: dict[str, None] = {}
    for m in _TABLE_REF.finditer(sql):
        seen.setdefault(m.group(1), None)
    return list(seen)


def _tabular_math_trace(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    try:
        mem = int(df.memory_usage(deep=True).sum())
    except Exception:  # noqa: BLE001
        mem = -1
    numeric = [c for c, t in df.dtypes.items() if "int" in str(t) or "float" in str(t)]
    return (
        f"Shape: {rows} × {cols}. In-memory bytes: {mem}. "
        f"Numeric columns: {len(numeric)} / {cols}."
    )


def register(path: str = "") -> FlatFileConnector:
    """Register a FlatFileConnector in the global ConnectorRegistry.

    `path` is optional at registration; the caller should later call
    `connect()` before any operations. Useful for the registry indexer
    once a working dataset path is known.
    """
    from agent.connectors.registry import ConnectorRegistry
    conn = FlatFileConnector(path=path)
    ConnectorRegistry.get().register(conn)
    return conn
