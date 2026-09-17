"""MongoDB connector.

Design
------
* Mapping to the `DataConnector` ABC:
    - "schema" ≡ **collection** inside the configured MongoDB database.
      (Mongo's "database" is the connector's single configured db; its
      "collection" is the analogue of a SQL table, so `list_schemas` and
      `describe_table` operate on collections.)
* `query(spec)` accepts a JSON spec, *not* SQL — Mongo has no SQL. The spec
  maps to either `find()` or `aggregate()`:

      {
        "collection": "users",
        "filter":   {"active": true},      # optional, find()
        "projection": {"_id": 0, "email": 1},
        "sort": [["_id", 1]],
        "limit": 100,
        "pipeline": [ {"$match": ...}, ... ]  # if present, use aggregate()
      }

* Read-only: aggregation pipelines are scanned for destructive stages
  (`$out`, `$merge`). Admin / write ops are not exposed.
* Mock-friendly: `client_factory` is injectable. Tests pass in a duck-typed
  `MongoClient`-alike with minimal methods (`__getitem__`, `list_collection_names`,
  `aggregate`, `find`, `command`). No `mongomock` dependency.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult


# ObjectId / Decimal128 / datetime show up in sampled docs; coerce to str so
# the ToolResult payload stays JSON-serialisable for the LLM.
def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


_DESTRUCTIVE_STAGES = ("$out", "$merge")

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

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.\-]*$")


class MongoDBConnector(DataConnector):
    """Read-only MongoDB connector conforming to the DataConnector ABC.

    Parameters
    ----------
    client_factory
        Callable returning a `pymongo.MongoClient`-alike. Defaults to
        `pymongo.MongoClient`. Override in tests.
    **config
        Driver config: `uri`, or the discrete kwargs `host`/`port`/`username`/
        `password`/`authSource`. Plus `database` (required) — the working
        database against which all ops run. Falls back to `MONGO_<NAME>` env
        vars.
    """

    name = "mongodb"

    _ENV_KEYS = ("uri", "host", "port", "username", "password", "authsource", "database")

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._client_factory = client_factory
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._client: Any = None
        self._db: Any = None
        self._db_name: str | None = None

    # ---- helpers -----------------------------------------------------------

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get(f"MONGO_{key.upper()}")
            if env_val:
                cfg[key] = env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_factory(self) -> Callable[..., Any]:
        if self._client_factory is not None:
            return self._client_factory
        import pymongo
        return pymongo.MongoClient

    def _require_connected(self) -> ToolResult | None:
        if self._db is None:
            return ToolResult(
                status="error",
                data=None,
                explanation="Not connected. Call connect() first.",
                math_trace="",
                error="ConnectionError: no active connection",
            )
        return None

    @staticmethod
    def _infer_field_types(docs: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
        """Union of type names seen per top-level field across sampled docs."""
        types: dict[str, set[str]] = {}
        for doc in docs:
            for k, v in doc.items():
                types.setdefault(k, set()).add(type(v).__name__)
        return types

    # ---- DataConnector API -------------------------------------------------

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        db_name = cfg.get("database")
        if not db_name:
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing Mongo config: 'database' is required.",
                math_trace="",
                error="ConfigError: missing database",
            )
        try:
            factory = self._resolve_factory()
            if "uri" in cfg:
                self._client = factory(cfg["uri"])
            else:
                kwargs = {k: v for k, v in cfg.items() if k != "database"}
                self._client = factory(**kwargs)
            self._db = self._client[db_name]
            self._db_name = db_name
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Mongo connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {k: ("***" if k in ("password", "uri") else v) for k, v in cfg.items()}
        return ToolResult(
            status="ok",
            data={"config": masked, "database": db_name},
            explanation=f"Connected to Mongo database {db_name!r}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        try:
            names = sorted(self._db.list_collection_names())
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"list_collection_names failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} collection(s) in {self._db_name!r}.",
            math_trace=f"db.listCollections → {len(names)} collections.",
        )

    def describe_table(self, table: str, sample_size: int = 50) -> ToolResult:
        """Sample `sample_size` documents to infer the top-level schema."""
        if (err := self._require_connected()) is not None:
            return err
        if not _SAFE_IDENT.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe collection name: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        try:
            coll = self._db[table]
            cursor = coll.aggregate([{"$sample": {"size": int(sample_size)}}])
            docs = list(cursor)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Sampling {table!r} failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        types = self._infer_field_types(docs)
        cols = [
            {"name": k, "types": sorted(v), "type": "|".join(sorted(v))}
            for k, v in sorted(types.items())
        ]
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "columns": cols,
                "sampled": len(docs),
            },
            explanation=(
                f"Inferred schema of {table!r} from {len(docs)} sampled document(s): "
                f"{len(cols)} top-level field(s)."
            ),
            math_trace=(
                f"Schema inference: union of observed types across {len(docs)} "
                f"$sample docs. Fields detected: {len(cols)}."
            ),
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        """Execute a find()/aggregate() against the configured db.

        `query` is a JSON spec (string or dict). See module docstring for the schema.
        """
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        if params:
            spec = {**spec, **params}

        table = spec.get("collection")
        if not table or not isinstance(table, str) or not _SAFE_IDENT.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing or invalid 'collection'.",
                math_trace="",
                error="ValueError: collection",
            )

        pipeline = spec.get("pipeline")
        try:
            if pipeline is not None:
                if not isinstance(pipeline, list):
                    return ToolResult(
                        status="error",
                        data=None,
                        explanation="'pipeline' must be a list of stage dicts.",
                        math_trace="",
                        error="TypeError: pipeline",
                    )
                for stage in pipeline:
                    if not isinstance(stage, Mapping):
                        return ToolResult(
                            status="error",
                            data=None,
                            explanation="Each pipeline stage must be a dict.",
                            math_trace="",
                            error="TypeError: stage",
                        )
                    for op in stage.keys():
                        if op in _DESTRUCTIVE_STAGES:
                            return ToolResult(
                                status="error",
                                data=None,
                                explanation=(
                                    f"Pipeline stage {op!r} writes data. "
                                    "Connector is read-only."
                                ),
                                math_trace="",
                                error="SafetyError: destructive stage",
                            )
                cursor = self._db[table].aggregate(pipeline)
                docs = [_jsonable(d) for d in cursor]
                return ToolResult(
                    status="ok",
                    data={"records": docs, "rowcount": len(docs)},
                    explanation=(
                        f"Aggregation on {table!r} returned {len(docs)} document(s) "
                        f"through {len(pipeline)} stage(s)."
                    ),
                    math_trace=f"Aggregation stages: {len(pipeline)}; returned rows: {len(docs)}.",
                )
            # find() path
            filter_ = spec.get("filter") or {}
            projection = spec.get("projection")
            sort = spec.get("sort")
            limit = int(spec.get("limit", 1000))
            skip = int(spec.get("skip", 0))

            cursor = self._db[table].find(filter_, projection)
            if sort:
                cursor = cursor.sort([tuple(p) for p in sort])
            if skip:
                cursor = cursor.skip(skip)
            cursor = cursor.limit(limit)
            docs = [_jsonable(d) for d in cursor]
            return ToolResult(
                status="ok",
                data={"records": docs, "rowcount": len(docs)},
                explanation=f"find() on {table!r} returned {len(docs)} document(s) (limit {limit}).",
                math_trace=(
                    f"find(filter keys: {list(filter_)}, limit={limit}, skip={skip}) "
                    f"→ {len(docs)} rows."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Query failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

    def estimate_cost(self, query: str) -> ToolResult:
        """Use Mongo's `explain` to surface docs-examined / index-used stats."""
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        table = spec.get("collection")
        if not table or not isinstance(table, str) or not _SAFE_IDENT.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing or invalid 'collection'.",
                math_trace="",
                error="ValueError: collection",
            )
        try:
            # Build an explain command so the same call covers find + aggregate.
            if spec.get("pipeline") is not None:
                explain = self._db.command({
                    "explain": {"aggregate": table, "pipeline": spec["pipeline"], "cursor": {}},
                    "verbosity": "executionStats",
                })
            else:
                explain = self._db.command({
                    "explain": {
                        "find": table,
                        "filter": spec.get("filter") or {},
                        "limit": int(spec.get("limit", 0)) or None,
                    },
                    "verbosity": "executionStats",
                })
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"explain failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        docs_examined = _dig(explain, "totalDocsExamined")
        keys_examined = _dig(explain, "totalKeysExamined")
        nreturned = _dig(explain, "nReturned") or _dig(explain, "nreturned")
        winning_plan_stage = _dig(explain, "stage")
        math_trace = (
            f"Estimated docs examined: {docs_examined}. Keys examined: {keys_examined}. "
            f"Rows returned (est): {nreturned}. Plan stage: {winning_plan_stage}.\n"
            f"Cost \\propto totalDocsExamined for a full-collection scan."
        )
        return ToolResult(
            status="ok",
            data={
                "docs_examined": docs_examined,
                "keys_examined": keys_examined,
                "rows_estimate": nreturned,
                "plan_stage": winning_plan_stage,
                "explain": _jsonable(explain),
            },
            explanation=(
                f"explain stats: docs={docs_examined}, keys={keys_examined}, "
                f"returned≈{nreturned}, stage={winning_plan_stage}."
            ),
            math_trace=math_trace,
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
                f"Scanned {len(desc.data.get('columns', []))} fields on {table}; "
                f"flagged {sum(len(v) for v in flagged.values())} as potential PII across "
                f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'}."
            ),
            math_trace=(
                "Heuristic: field-name regex over "
                f"{len(_PII_PATTERNS)} PII categories (top-level fields from sampled docs)."
            ),
        )

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._db = None
                self._db_name = None


# ---- helpers ---------------------------------------------------------------


def _parse_spec(query: Any) -> dict | ToolResult:
    if isinstance(query, Mapping):
        return dict(query)
    if not isinstance(query, str):
        return ToolResult(
            status="error",
            data=None,
            explanation=f"Query must be a JSON spec (str or dict), got {type(query).__name__}.",
            math_trace="",
            error="TypeError: query",
        )
    s = query.strip()
    if not s:
        return ToolResult(
            status="error",
            data=None,
            explanation="Empty query.",
            math_trace="",
            error="ValueError: query is required",
        )
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as exc:
        return ToolResult(
            status="error",
            data=None,
            explanation=f"Query is not valid JSON: {exc}",
            math_trace="",
            error=f"JSONDecodeError: {exc}",
        )
    if not isinstance(parsed, Mapping):
        return ToolResult(
            status="error",
            data=None,
            explanation="Query JSON must be an object/dict.",
            math_trace="",
            error="TypeError: query",
        )
    return dict(parsed)


def _dig(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
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


def register() -> MongoDBConnector:
    """Register the connector in the global ConnectorRegistry (no connect())."""
    from agent.connectors.registry import ConnectorRegistry
    conn = MongoDBConnector()
    ConnectorRegistry.get().register(conn)
    return conn
