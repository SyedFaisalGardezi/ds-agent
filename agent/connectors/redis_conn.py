"""Redis connector.

Redis is a key-value / structured cache — mapping to the `DataConnector` ABC
is deliberate:

* **"schema" ≡ key namespace (prefix before `:`)** — `list_schemas()` scans
  the keyspace and returns unique namespace prefixes.
* **"table" ≡ namespace** — `describe_table(ns)` samples keys under the
  namespace, reports type distribution + a few sample key names.
* **`query(spec)`** accepts a JSON spec (not a Redis command):

      {
        "op": "get" | "mget" | "hgetall" | "lrange" | "smembers" |
              "zrange" | "scan",
        "key": "users:1",
        "keys": ["users:1","users:2"],      # for mget
        "start": 0, "stop": -1,              # for lrange / zrange
        "match": "users:*",                  # for scan
        "count": 100                         # for scan
      }

* Read-only: the connector never exposes writes (`SET`, `DEL`, `FLUSH`,
  `EXPIRE`, `EVAL`, `CONFIG`, `SHUTDOWN` etc. are not callable). The `op`
  whitelist is the enforcement layer.
* Mock-friendly: `client_factory` is injectable. Tests pass a duck-typed
  `redis.Redis`-alike that implements only the read ops we dispatch to.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

# Whitelisted read-only operations. Any op outside this set → SafetyError.
_READ_OPS = frozenset(
    {"get", "mget", "hgetall", "hget", "lrange", "smembers", "zrange", "scan", "type", "ttl"}
)

_SAFE_KEY = re.compile(r"^[A-Za-z0-9_:\-.\*\?\[\]\{\}]+$")

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


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.hex()
    if isinstance(v, Mapping):
        return {_jsonable(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


class RedisConnector(DataConnector):
    """Read-only Redis connector."""

    name = "redis"

    _ENV_KEYS = ("host", "port", "db", "password", "username")

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._client_factory = client_factory
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._client: Any = None

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get(f"REDIS_{key.upper()}")
            if env_val is not None:
                if key in ("port", "db"):
                    cfg[key] = int(env_val)
                else:
                    cfg[key] = env_val
        cfg.update(self._user_config)
        return cfg

    def _resolve_factory(self) -> Callable[..., Any]:
        if self._client_factory is not None:
            return self._client_factory
        import redis  # pragma: no cover
        return redis.Redis

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

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        if not cfg.get("host"):
            cfg["host"] = "localhost"
        cfg.setdefault("port", 6379)
        cfg.setdefault("db", 0)
        try:
            self._client = self._resolve_factory()(**cfg)
            # ping on connect so we surface connection errors immediately
            ping = getattr(self._client, "ping", None)
            if callable(ping):
                ping()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Redis connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {k: ("***" if k == "password" else v) for k, v in cfg.items()}
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=f"Connected to Redis at {cfg['host']}:{cfg['port']} db={cfg['db']}.",
            math_trace="",
        )

    def list_schemas(self, sample_keys: int = 1000) -> ToolResult:
        """Return sorted unique key namespace prefixes (key before first ':')."""
        if (err := self._require_connected()) is not None:
            return err
        namespaces: dict[str, int] = {}
        scanned = 0
        try:
            for key in self._client.scan_iter(match="*", count=200):
                if scanned >= sample_keys:
                    break
                scanned += 1
                k = key.decode() if isinstance(key, bytes) else str(key)
                ns = k.split(":", 1)[0] if ":" in k else k
                namespaces[ns] = namespaces.get(ns, 0) + 1
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"scan_iter failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        names = sorted(namespaces.keys())
        return ToolResult(
            status="ok",
            data=names,
            explanation=(
                f"Found {len(names)} namespace(s) across {scanned} sampled key(s)."
            ),
            math_trace=(
                f"Namespace discovery by SCAN ≤ {sample_keys} keys; "
                f"unique prefixes = {len(names)}."
            ),
        )

    def describe_table(self, table: str, sample_size: int = 50) -> ToolResult:
        """Sample keys under `table` namespace; report type mix + samples."""
        if (err := self._require_connected()) is not None:
            return err
        if not _SAFE_KEY.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe namespace: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        pattern = table if ("*" in table or "?" in table) else f"{table}:*"
        keys: list[str] = []
        try:
            for raw in self._client.scan_iter(match=pattern, count=200):
                if len(keys) >= sample_size:
                    break
                k = raw.decode() if isinstance(raw, bytes) else str(raw)
                keys.append(k)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"scan_iter failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        type_counts: dict[str, int] = {}
        hash_fields: set[str] = set()
        for k in keys:
            try:
                t = self._client.type(k)
                t = t.decode() if isinstance(t, bytes) else str(t)
            except Exception:  # noqa: BLE001
                t = "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
            if t == "hash":
                try:
                    h = self._client.hgetall(k)
                    hash_fields.update(
                        (f.decode() if isinstance(f, bytes) else str(f)) for f in h.keys()
                    )
                except Exception:  # noqa: BLE001
                    pass
        cols = [
            {"name": f, "type": "hash_field"} for f in sorted(hash_fields)
        ]
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "pattern": pattern,
                "sampled_keys": len(keys),
                "type_counts": type_counts,
                "sample_keys": keys[:10],
                "columns": cols,
            },
            explanation=(
                f"Namespace {table!r}: sampled {len(keys)} key(s); "
                f"type mix = {type_counts}; hash-field vocab = {len(hash_fields)}."
            ),
            math_trace=(
                f"SCAN MATCH {pattern} → {len(keys)} keys; "
                f"TYPE+HGETALL union over hashes → {len(hash_fields)} fields."
            ),
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        """Execute a whitelisted read op described by a JSON spec."""
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        if params:
            spec = {**spec, **params}

        op = str(spec.get("op", "")).lower()
        if not op:
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing 'op'.",
                math_trace="",
                error="ValueError: op",
            )
        if op not in _READ_OPS:
            return ToolResult(
                status="error",
                data=None,
                explanation=(
                    f"Op {op!r} is not in the read-only whitelist "
                    f"({sorted(_READ_OPS)})."
                ),
                math_trace="",
                error="SafetyError: op not allowed",
            )

        try:
            result: Any
            if op == "get":
                result = self._client.get(spec["key"])
            elif op == "mget":
                result = self._client.mget(spec.get("keys") or [])
            elif op == "hget":
                result = self._client.hget(spec["key"], spec["field"])
            elif op == "hgetall":
                result = self._client.hgetall(spec["key"])
            elif op == "lrange":
                result = self._client.lrange(
                    spec["key"], int(spec.get("start", 0)), int(spec.get("stop", -1))
                )
            elif op == "smembers":
                result = self._client.smembers(spec["key"])
            elif op == "zrange":
                result = self._client.zrange(
                    spec["key"], int(spec.get("start", 0)), int(spec.get("stop", -1))
                )
            elif op == "type":
                result = self._client.type(spec["key"])
            elif op == "ttl":
                result = self._client.ttl(spec["key"])
            elif op == "scan":
                match = spec.get("match", "*")
                count = int(spec.get("count", 100))
                keys = []
                for k in self._client.scan_iter(match=match, count=count):
                    keys.append(k.decode() if isinstance(k, bytes) else str(k))
                    if len(keys) >= count:
                        break
                result = keys
            else:  # pragma: no cover
                raise AssertionError("unreachable")
        except KeyError as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Spec missing required field: {exc}.",
                math_trace="",
                error=f"KeyError: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Redis op {op!r} failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

        decoded = _jsonable(result)
        if isinstance(decoded, list):
            rowcount = len(decoded)
        elif isinstance(decoded, dict):
            rowcount = len(decoded)
        elif decoded is None:
            rowcount = 0
        else:
            rowcount = 1
        return ToolResult(
            status="ok",
            data={"op": op, "result": decoded, "rowcount": rowcount},
            explanation=f"{op}() returned {rowcount} item(s).",
            math_trace=f"Redis {op} → {rowcount} items.",
        )

    def estimate_cost(self, query: str) -> ToolResult:
        """Estimate by DBSIZE for scan ops, or O(1) for direct key lookups."""
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        op = str(spec.get("op", "")).lower()
        if op not in _READ_OPS:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Op {op!r} is not a recognised read op.",
                math_trace="",
                error="SafetyError: op not allowed",
            )
        try:
            dbsize = self._client.dbsize() if hasattr(self._client, "dbsize") else None
        except Exception:  # noqa: BLE001
            dbsize = None
        if op in ("scan",):
            rows_estimate = int(spec.get("count", 100))
            complexity = "O(N) where N = dbsize"
        elif op in ("mget",):
            rows_estimate = len(spec.get("keys") or [])
            complexity = "O(k) where k = len(keys)"
        else:
            rows_estimate = 1
            complexity = "O(1) key lookup"
        return ToolResult(
            status="ok",
            data={
                "op": op,
                "rows_estimate": rows_estimate,
                "dbsize": dbsize,
                "complexity": complexity,
            },
            explanation=(
                f"op={op}: {complexity}; dbsize={dbsize}; rows≈{rows_estimate}."
            ),
            math_trace=f"Cost model: {complexity}. Estimate ≈ {rows_estimate}.",
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
                f"Scanned {len(desc.data.get('columns', []))} hash-field(s) on "
                f"{table}; flagged {sum(len(v) for v in flagged.values())} as PII "
                f"across {len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'}."
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


def register() -> RedisConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = RedisConnector()
    ConnectorRegistry.get().register(conn)
    return conn
