"""BigQuery connector.

Uses an injectable `client_factory` returning a `google.cloud.bigquery.Client`
-alike. In production the default is `google.cloud.bigquery.Client`; in
tests we pass a duck-typed fake. The `[bigquery]` extra is required at
runtime but the unit tests never import it.

Mapping to the `DataConnector` ABC
----------------------------------
* **"schema" ≡ dataset** — `list_schemas()` returns dataset IDs within the
  configured GCP project.
* **"table" ≡ fully-qualified table** (`dataset.table` or
  `project.dataset.table`).
* **`query(sql)`** runs a read-only SQL job. `dry_run=True` is used for
  `estimate_cost` to get bytes-processed without running the query.
* Read-only: destructive SQL is regex-blocked.
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
    r"REVOKE|EXPORT|LOAD|COPY)\b",
    re.IGNORECASE,
)

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# GCP project IDs legally contain hyphens (e.g. "my-proj-123"); dataset and
# table names do not.
_SAFE_PROJECT = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*$")

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

# BigQuery on-demand pricing: $5 per TB processed (rounded to nearest MB,
# minimum 10 MB per query). This is a public figure used for estimate_cost;
# not a commitment. https://cloud.google.com/bigquery/pricing
_BQ_PRICE_PER_TB_USD = 5.0
_BQ_MIN_QUERY_MB = 10


class BigQueryConnector(DataConnector):
    """Read-only BigQuery connector."""

    name = "bigquery"

    _ENV_KEYS = ("project", "location", "credentials_path")

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._client_factory = client_factory
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._client: Any = None
        self._project: str | None = None
        self._location: str | None = None
        self._bigquery_module: Any = None

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get(f"BIGQUERY_{key.upper()}")
            if env_val is not None:
                cfg[key] = env_val
        if os.environ.get("GOOGLE_CLOUD_PROJECT") and "project" not in cfg:
            cfg["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        cfg.update(self._user_config)
        return cfg

    def _resolve_factory(self) -> Callable[..., Any]:
        if self._client_factory is not None:
            return self._client_factory
        from google.cloud import bigquery  # pragma: no cover
        self._bigquery_module = bigquery
        return bigquery.Client

    def _require_connected(self) -> ToolResult | None:
        if self._client is None:
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
                explanation=(
                    "Destructive SQL keyword detected. Connector is read-only."
                ),
                math_trace="",
                error="SafetyError: destructive statement",
            )
        return None

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        if not cfg.get("project"):
            return ToolResult(
                status="error",
                data=None,
                explanation=(
                    "Missing BigQuery config: 'project' is required "
                    "(set BIGQUERY_PROJECT or GOOGLE_CLOUD_PROJECT)."
                ),
                math_trace="",
                error="ConfigError: missing project",
            )
        try:
            factory = self._resolve_factory()
            kwargs: dict[str, Any] = {"project": cfg["project"]}
            if cfg.get("location"):
                kwargs["location"] = cfg["location"]
            self._client = factory(**kwargs)
            self._project = cfg["project"]
            self._location = cfg.get("location")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open BigQuery client: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {
            k: ("***" if k == "credentials_path" else v)
            for k, v in cfg.items()
        }
        return ToolResult(
            status="ok",
            data={"config": masked, "project": self._project},
            explanation=(
                f"Connected to BigQuery project={self._project!r} "
                f"location={self._location!r}."
            ),
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        try:
            datasets = list(self._client.list_datasets(project=self._project))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"list_datasets failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        names = sorted(
            getattr(d, "dataset_id", None) or str(d) for d in datasets
        )
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} dataset(s) in {self._project!r}.",
            math_trace=f"BigQuery.list_datasets → {len(names)} rows.",
        )

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        parts = table.split(".")
        if len(parts) == 2:
            project = self._project
            dataset, name = parts
        elif len(parts) == 3:
            project, dataset, name = parts
        else:
            return ToolResult(
                status="error",
                data=None,
                explanation=(
                    f"BigQuery table must be 'dataset.table' or "
                    f"'project.dataset.table'; got {table!r}."
                ),
                math_trace="",
                error="ValueError: invalid identifier",
            )
        if not _SAFE_PROJECT.match(project or "p") or not _SAFE_IDENT.match(dataset) or not _SAFE_IDENT.match(name):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe identifier in {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        try:
            tbl = self._client.get_table(f"{project}.{dataset}.{name}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"get_table failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        schema = getattr(tbl, "schema", []) or []
        cols = []
        for f in schema:
            nullable = getattr(f, "mode", "NULLABLE") != "REQUIRED"
            cols.append({
                "name": getattr(f, "name", None),
                "type": getattr(f, "field_type", None),
                "nullable": nullable,
                "mode": getattr(f, "mode", None),
            })
        num_rows = getattr(tbl, "num_rows", None)
        num_bytes = getattr(tbl, "num_bytes", None)
        return ToolResult(
            status="ok",
            data={
                "table": f"{project}.{dataset}.{name}",
                "columns": cols,
                "num_rows": num_rows,
                "num_bytes": num_bytes,
            },
            explanation=(
                f"{project}.{dataset}.{name}: {len(cols)} column(s), "
                f"{num_rows} row(s), {num_bytes} byte(s)."
            ),
            math_trace=(
                f"tables.get({project}.{dataset}.{name}) → "
                f"{len(cols)} cols, {num_rows} rows."
            ),
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
        try:
            job = self._client.query(sql)
            result = job.result()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Query failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        records: list[dict[str, Any]] = []
        for row in result:
            if isinstance(row, dict):
                records.append(row)
            elif hasattr(row, "keys"):
                records.append({k: row[k] for k in row.keys()})
            elif hasattr(row, "_asdict"):
                records.append(row._asdict())
            else:
                records.append({"value": str(row)})
        bytes_billed = getattr(job, "total_bytes_billed", None)
        bytes_processed = getattr(job, "total_bytes_processed", None)
        return ToolResult(
            status="ok",
            data={
                "records": records,
                "rowcount": len(records),
                "total_bytes_processed": bytes_processed,
                "total_bytes_billed": bytes_billed,
            },
            explanation=(
                f"Query returned {len(records)} row(s); "
                f"{bytes_processed} byte(s) processed."
            ),
            math_trace=(
                f"BigQuery job → {len(records)} rows; "
                f"bytes_processed={bytes_processed}, billed={bytes_billed}."
            ),
        )

    def estimate_cost(self, query: str) -> ToolResult:
        """Use `dry_run=True` to get bytes-processed estimate without running."""
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
        try:
            # Build a dry-run job config. If the real bigquery module isn't
            # available (tests), accept any truthy duck-typed job_config the
            # fake client understands, or pass None.
            job_config = self._make_dry_run_config()
            job = self._client.query(sql, job_config=job_config)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Dry-run failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        bytes_processed = getattr(job, "total_bytes_processed", None) or 0
        try:
            bytes_processed = int(bytes_processed)
        except (TypeError, ValueError):
            bytes_processed = 0
        # Apply BigQuery's 10 MB minimum and convert to USD.
        billed_mb = max(_BQ_MIN_QUERY_MB, bytes_processed / (1024 * 1024))
        usd_estimate = (billed_mb / (1024 * 1024)) * _BQ_PRICE_PER_TB_USD
        return ToolResult(
            status="ok",
            data={
                "bytes_estimate": bytes_processed,
                "billed_mb": billed_mb,
                "usd_estimate": usd_estimate,
            },
            explanation=(
                f"Dry-run estimate: {bytes_processed} bytes will be scanned; "
                f"billed ≈ {billed_mb:.2f} MB ≈ ${usd_estimate:.6f} USD."
            ),
            math_trace=(
                f"cost_usd = max({_BQ_MIN_QUERY_MB}, bytes/1MB) / 1TB × "
                f"${_BQ_PRICE_PER_TB_USD:.2f} = ${usd_estimate:.6f}."
            ),
        )

    def _make_dry_run_config(self) -> Any:
        """Build `QueryJobConfig(dry_run=True)` if google-cloud-bigquery is
        importable; otherwise return a plain sentinel dict for fakes."""
        if self._bigquery_module is not None:
            try:
                return self._bigquery_module.QueryJobConfig(
                    dry_run=True, use_query_cache=False
                )
            except Exception:  # noqa: BLE001
                pass
        return {"dry_run": True, "use_query_cache": False}

    def detect_pii(self, table: str) -> ToolResult:
        desc = self.describe_table(table)
        if desc.status != "ok":
            return desc
        flagged: dict[str, list[str]] = {}
        for col in desc.data["columns"]:
            colname = col["name"] or ""
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
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
            finally:
                self._client = None
                self._project = None
                self._location = None


def register() -> BigQueryConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = BigQueryConnector()
    ConnectorRegistry.get().register(conn)
    return conn
