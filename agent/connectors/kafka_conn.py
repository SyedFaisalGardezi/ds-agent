"""Kafka connector.

Design
------
Kafka is a streaming log, not a database — the mapping to the
`DataConnector` ABC is deliberate:

* **"schema" ≡ topic** — `list_schemas()` returns topic names.
* **"table" ≡ topic** — `describe_table(topic)` samples recent messages and
  infers the top-level field schema (analogous to the Mongo
  `$sample`-based describe).
* **`query(spec)`** accepts a JSON spec (not SQL) describing a bounded
  consume:

      {
        "topic": "events",
        "partition": 0,          # optional
        "offset": "latest" | "earliest" | int,  # default: latest
        "limit": 100,            # max messages to return
        "timeout_ms": 2000,      # per-poll timeout
        "decode": "json" | "utf8" | "raw"  # default: json w/ utf8 fallback
      }

* Read-only. The connector never produces. Admin-write operations
  (create/delete topic, reset offsets server-side) are not exposed.
* Mock-friendly: `consumer_factory` and `admin_factory` are injectable.
  Tests pass duck-typed fakes — no real `confluent-kafka` dependency is
  exercised in unit tests.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


_SAFE_TOPIC = re.compile(r"^[A-Za-z0-9._\-]+$")

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


class KafkaConnector(DataConnector):
    """Read-only Kafka connector conforming to the DataConnector ABC."""

    name = "kafka"

    _ENV_KEYS = (
        "bootstrap.servers",
        "group.id",
        "security.protocol",
        "sasl.mechanism",
        "sasl.username",
        "sasl.password",
    )

    def __init__(
        self,
        consumer_factory: Callable[..., Any] | None = None,
        admin_factory: Callable[..., Any] | None = None,
        **config: Any,
    ) -> None:
        self._consumer_factory = consumer_factory
        self._admin_factory = admin_factory
        self._user_config = {k: v for k, v in config.items() if v is not None}
        self._consumer: Any = None
        self._admin: Any = None
        self._config: dict[str, Any] = {}

    # ---- helpers -------------------------------------------------------

    def _effective_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for key in self._ENV_KEYS:
            env_val = os.environ.get("KAFKA_" + key.replace(".", "_").upper())
            if env_val:
                cfg[key] = env_val
        cfg.update(self._user_config)
        cfg.setdefault("group.id", "ds-agent-readonly")
        cfg.setdefault("enable.auto.commit", False)
        cfg.setdefault("auto.offset.reset", "latest")
        return cfg

    def _resolve_consumer_factory(self) -> Callable[..., Any]:
        if self._consumer_factory is not None:
            return self._consumer_factory
        from confluent_kafka import Consumer  # pragma: no cover
        return Consumer

    def _resolve_admin_factory(self) -> Callable[..., Any] | None:
        if self._admin_factory is not None:
            return self._admin_factory
        try:
            from confluent_kafka.admin import AdminClient  # pragma: no cover
            return AdminClient
        except ImportError:  # pragma: no cover
            return None

    def _require_connected(self) -> ToolResult | None:
        if self._consumer is None:
            return ToolResult(
                status="error",
                data=None,
                explanation="Not connected. Call connect() first.",
                math_trace="",
                error="ConnectionError: no active connection",
            )
        return None

    @staticmethod
    def _decode(value: Any, mode: str) -> Any:
        if value is None:
            return None
        if mode == "raw":
            return _jsonable(value)
        if mode == "utf8":
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8")
                except UnicodeDecodeError:
                    return value.hex()
            return value
        # json (with utf8 fallback)
        raw = value
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.hex()
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return _jsonable(raw)

    # ---- DataConnector API --------------------------------------------

    def connect(self) -> ToolResult:
        cfg = self._effective_config()
        if not cfg.get("bootstrap.servers"):
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing Kafka config: 'bootstrap.servers' is required.",
                math_trace="",
                error="ConfigError: missing bootstrap.servers",
            )
        try:
            self._consumer = self._resolve_consumer_factory()(cfg)
            admin_factory = self._resolve_admin_factory()
            self._admin = admin_factory(cfg) if admin_factory is not None else None
            self._config = cfg
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open Kafka connection: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        masked = {
            k: ("***" if "password" in k or "secret" in k else v) for k, v in cfg.items()
        }
        return ToolResult(
            status="ok",
            data={"config": masked},
            explanation=f"Connected to Kafka at {cfg.get('bootstrap.servers')!r}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        try:
            meta_source = self._admin if self._admin is not None else self._consumer
            meta = meta_source.list_topics(timeout=10)
            topics_map = getattr(meta, "topics", None) or {}
            names = sorted(
                t for t in topics_map.keys() if not str(t).startswith("__")
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"list_topics failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(
            status="ok",
            data=names,
            explanation=f"Listed {len(names)} topic(s) (internal __ topics hidden).",
            math_trace=f"AdminClient.list_topics → {len(names)} topics.",
        )

    def describe_table(self, table: str, sample_size: int = 20) -> ToolResult:
        """Sample recent messages from `table` (topic) and infer field types."""
        if (err := self._require_connected()) is not None:
            return err
        if not _SAFE_TOPIC.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe topic name: {table!r}.",
                math_trace="",
                error="ValueError: invalid identifier",
            )
        try:
            meta_source = self._admin if self._admin is not None else self._consumer
            meta = meta_source.list_topics(timeout=10)
            topics_map = getattr(meta, "topics", None) or {}
            if table not in topics_map:
                return ToolResult(
                    status="error",
                    data=None,
                    explanation=f"Unknown topic: {table!r}.",
                    math_trace="",
                    error="KeyError: topic",
                )
            topic_meta = topics_map[table]
            n_partitions = len(getattr(topic_meta, "partitions", {}) or {})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Metadata lookup failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

        sample_spec = {
            "topic": table,
            "offset": "earliest",
            "limit": int(sample_size),
            "timeout_ms": 2000,
            "decode": "json",
        }
        sampled_messages: list[Any] = []
        sub = self.query(json.dumps(sample_spec))
        if sub.status == "ok":
            sampled_messages = sub.data.get("records", []) or []

        types: dict[str, set[str]] = {}
        for msg in sampled_messages:
            payload = msg.get("value") if isinstance(msg, Mapping) else None
            if isinstance(payload, Mapping):
                for k, v in payload.items():
                    types.setdefault(str(k), set()).add(type(v).__name__)
        cols = [
            {"name": k, "types": sorted(v), "type": "|".join(sorted(v))}
            for k, v in sorted(types.items())
        ]
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "partitions": n_partitions,
                "columns": cols,
                "sampled": len(sampled_messages),
            },
            explanation=(
                f"Topic {table!r}: {n_partitions} partition(s); inferred schema "
                f"from {len(sampled_messages)} sampled message(s) → "
                f"{len(cols)} top-level field(s)."
            ),
            math_trace=(
                f"Partitions: {n_partitions}. Schema union over "
                f"{len(sampled_messages)} decoded JSON payloads."
            ),
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        """Execute a bounded consume on a topic. See module docstring for spec."""
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        if params:
            spec = {**spec, **params}

        topic = spec.get("topic")
        if not topic or not isinstance(topic, str) or not _SAFE_TOPIC.match(topic):
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing or invalid 'topic'.",
                math_trace="",
                error="ValueError: topic",
            )

        limit = int(spec.get("limit", 100))
        timeout_ms = int(spec.get("timeout_ms", 2000))
        timeout_s = max(0.01, timeout_ms / 1000.0)
        decode = spec.get("decode", "json")
        if decode not in ("json", "utf8", "raw"):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unknown decode mode: {decode!r}.",
                math_trace="",
                error="ValueError: decode",
            )

        offset = spec.get("offset", "latest")
        partition = spec.get("partition")

        try:
            self._consumer.subscribe([topic])
        except Exception:
            pass

        records: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            seek = getattr(self._consumer, "seek", None)
            if callable(seek) and offset is not None:
                try:
                    seek(topic, partition, offset)
                except TypeError:
                    pass

            polls_without_message = 0
            max_empty_polls = 3
            while len(records) < limit:
                msg = self._consumer.poll(timeout_s)
                if msg is None:
                    polls_without_message += 1
                    if polls_without_message >= max_empty_polls:
                        break
                    continue
                polls_without_message = 0
                err = msg.error() if hasattr(msg, "error") else None
                if err:
                    errors.append(str(err))
                    continue
                records.append({
                    "topic": msg.topic() if hasattr(msg, "topic") else topic,
                    "partition": msg.partition() if hasattr(msg, "partition") else partition,
                    "offset": msg.offset() if hasattr(msg, "offset") else None,
                    "key": self._decode(
                        msg.key() if hasattr(msg, "key") else None, "utf8"
                    ),
                    "value": self._decode(
                        msg.value() if hasattr(msg, "value") else None, decode
                    ),
                    "timestamp": (msg.timestamp() if hasattr(msg, "timestamp") else None),
                })
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Consume failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

        return ToolResult(
            status="ok",
            data={
                "records": records,
                "rowcount": len(records),
                "errors": errors,
            },
            explanation=(
                f"Consumed {len(records)} message(s) from {topic!r} "
                f"(offset={offset!r}, limit={limit})."
            ),
            math_trace=(
                f"bounded_consume(topic={topic}, limit={limit}, "
                f"timeout_ms={timeout_ms}) → {len(records)} messages, "
                f"{len(errors)} error(s)."
            ),
        )

    def estimate_cost(self, query: str) -> ToolResult:
        """Estimate using per-partition low/high watermark offsets."""
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        topic = spec.get("topic")
        if not topic or not isinstance(topic, str) or not _SAFE_TOPIC.match(topic):
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing or invalid 'topic'.",
                math_trace="",
                error="ValueError: topic",
            )

        try:
            meta_source = self._admin if self._admin is not None else self._consumer
            meta = meta_source.list_topics(timeout=10)
            topics_map = getattr(meta, "topics", None) or {}
            if topic not in topics_map:
                return ToolResult(
                    status="error",
                    data=None,
                    explanation=f"Unknown topic: {topic!r}.",
                    math_trace="",
                    error="KeyError: topic",
                )
            partitions = list(
                (getattr(topics_map[topic], "partitions", {}) or {}).keys()
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Metadata lookup failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

        per_partition: dict[int, dict[str, int]] = {}
        total = 0
        get_wm = getattr(self._consumer, "get_watermark_offsets", None)
        for p in partitions:
            if not callable(get_wm):
                per_partition[p] = {"low": 0, "high": 0, "lag": 0}
                continue
            try:
                low, high = get_wm((topic, p))
            except TypeError:
                low, high = get_wm(topic, p)
            lag = max(0, int(high) - int(low))
            per_partition[p] = {"low": int(low), "high": int(high), "lag": lag}
            total += lag

        requested = int(spec.get("limit", 100))
        rows_estimate = min(total, requested) if total else requested

        return ToolResult(
            status="ok",
            data={
                "topic": topic,
                "partitions": len(partitions),
                "total_messages_available": total,
                "rows_estimate": rows_estimate,
                "per_partition_offsets": per_partition,
            },
            explanation=(
                f"Topic {topic!r} has {total} message(s) across "
                f"{len(partitions)} partition(s); bounded consume will read "
                f"≤ {rows_estimate}."
            ),
            math_trace=(
                "cost ≈ Σ_p (high_p - low_p), clamped by spec.limit. "
                f"Σ lag = {total}, limit = {requested}, estimate = {rows_estimate}."
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
                f"Scanned {len(desc.data.get('columns', []))} field(s) on "
                f"{table}; flagged {sum(len(v) for v in flagged.values())} as "
                f"potential PII across {len(flagged)} categor"
                f"{'y' if len(flagged) == 1 else 'ies'}."
            ),
            math_trace=(
                "Heuristic: field-name regex over "
                f"{len(_PII_PATTERNS)} PII categories (top-level fields from sampled messages)."
            ),
        )

    def close(self) -> None:
        if self._consumer is not None:
            try:
                self._consumer.close()
            finally:
                self._consumer = None
                self._admin = None
                self._config = {}


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


def register() -> KafkaConnector:
    """Register the connector in the global ConnectorRegistry (no connect())."""
    from agent.connectors.registry import ConnectorRegistry
    conn = KafkaConnector()
    ConnectorRegistry.get().register(conn)
    return conn
