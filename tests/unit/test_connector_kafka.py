"""Unit tests for KafkaConnector — duck-typed fakes, no confluent-kafka."""
from __future__ import annotations

import json

from agent.connectors.kafka_conn import KafkaConnector, register

# ---- fake Kafka stack --------------------------------------------------


class FakeMessage:
    def __init__(self, value, key=None, topic="t", partition=0, offset=0, err=None):
        self._v = value
        self._k = key
        self._t = topic
        self._p = partition
        self._o = offset
        self._e = err

    def value(self):
        return self._v

    def key(self):
        return self._k

    def topic(self):
        return self._t

    def partition(self):
        return self._p

    def offset(self):
        return self._o

    def error(self):
        return self._e

    def timestamp(self):
        return (1, 1700000000000)


class FakeTopicMeta:
    def __init__(self, n_partitions=1):
        self.partitions = {i: object() for i in range(n_partitions)}


class FakeMeta:
    def __init__(self, topics):
        self.topics = topics


class FakeConsumer:
    def __init__(self, messages=None, topics_meta=None, watermarks=None):
        self._messages = list(messages or [])
        self._topics_meta = topics_meta or {}
        self._watermarks = watermarks or {}
        self.subscribed = None
        self.closed = False
        self.seek_called = False

    def subscribe(self, topics):
        self.subscribed = list(topics)

    def poll(self, timeout):
        if self._messages:
            return self._messages.pop(0)
        return None

    def seek(self, topic, partition, offset):
        self.seek_called = True

    def get_watermark_offsets(self, tp):
        if isinstance(tp, tuple):
            return self._watermarks.get(tp, (0, 0))
        return (0, 0)

    def list_topics(self, timeout=10):
        return FakeMeta(self._topics_meta)

    def close(self):
        self.closed = True


class FakeAdmin:
    def __init__(self, topics_meta):
        self._topics_meta = topics_meta

    def list_topics(self, timeout=10):
        return FakeMeta(self._topics_meta)


def _make(consumer=None, admin=None, **config):
    cfg = {"bootstrap.servers": "h:9092", **config}
    cons = consumer or FakeConsumer()
    c = KafkaConnector(
        consumer_factory=lambda *a, **kw: cons,
        admin_factory=(lambda *a, **kw: admin) if admin else None,
        **cfg,
    )
    return c, cons


# ---- connect -----------------------------------------------------------


def test_connect_requires_bootstrap(monkeypatch):
    for k in ("BOOTSTRAP_SERVERS", "GROUP_ID"):
        monkeypatch.delenv(f"KAFKA_{k}", raising=False)
    c = KafkaConnector(consumer_factory=lambda *a, **kw: FakeConsumer())
    r = c.connect()
    assert r.status == "error"
    assert "bootstrap.servers" in r.explanation


def test_connect_masks_password():
    c, _ = _make(**{"sasl.password": "secret123"})
    r = c.connect()
    assert r.status == "ok"
    assert r.data["config"]["sasl.password"] == "***"
    assert r.data["config"]["bootstrap.servers"] == "h:9092"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "envhost:9092")
    captured = {}

    def factory(cfg):
        captured.update(cfg)
        return FakeConsumer()

    c = KafkaConnector(consumer_factory=factory)
    r = c.connect()
    assert r.status == "ok"
    assert captured["bootstrap.servers"] == "envhost:9092"


def test_connect_surfaces_driver_error():
    def boom(cfg):
        raise RuntimeError("no brokers")

    c = KafkaConnector(consumer_factory=boom, **{"bootstrap.servers": "h:9092"})
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- require-connect ---------------------------------------------------


def test_operations_require_connect():
    c = KafkaConnector(
        consumer_factory=lambda cfg: FakeConsumer(),
        **{"bootstrap.servers": "h:9092"},
    )
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", '{"topic":"t"}'),
        ("estimate_cost", '{"topic":"t"}'),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"
        assert "Not connected" in r.explanation


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_filters_internal():
    topics = {"events": FakeTopicMeta(), "orders": FakeTopicMeta(), "__consumer_offsets": FakeTopicMeta()}
    c, _ = _make(consumer=FakeConsumer(topics_meta=topics))
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert set(r.data) == {"events", "orders"}


def test_list_schemas_uses_admin_when_provided():
    topics = {"a": FakeTopicMeta(), "b": FakeTopicMeta()}
    c, _ = _make(admin=FakeAdmin(topics))
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["a", "b"]


# ---- describe_table ----------------------------------------------------


def test_describe_table_rejects_unsafe_topic():
    c, _ = _make()
    c.connect()
    r = c.describe_table("bad topic!")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_unknown_topic():
    c, _ = _make(consumer=FakeConsumer(topics_meta={}))
    c.connect()
    r = c.describe_table("nope")
    assert r.status == "error"
    assert "Unknown topic" in r.explanation


def test_describe_table_infers_from_sampled_json():
    payloads = [
        json.dumps({"email": "a@x", "age": 30}).encode(),
        json.dumps({"email": "b@x", "age": 25, "city": "LA"}).encode(),
    ]
    messages = [FakeMessage(v) for v in payloads]
    topics = {"events": FakeTopicMeta(n_partitions=3)}
    cons = FakeConsumer(messages=messages, topics_meta=topics)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.describe_table("events", sample_size=10)
    assert r.status == "ok"
    names = {col["name"] for col in r.data["columns"]}
    assert names == {"email", "age", "city"}
    assert r.data["partitions"] == 3
    assert r.data["sampled"] == 2


# ---- query -------------------------------------------------------------


def test_query_consumes_bounded():
    messages = [FakeMessage(b'{"n":1}', offset=0), FakeMessage(b'{"n":2}', offset=1)]
    cons = FakeConsumer(messages=messages)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "events", "limit": 10, "timeout_ms": 100}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"][0]["value"] == {"n": 1}
    assert cons.subscribed == ["events"]


def test_query_accepts_dict_spec():
    cons = FakeConsumer(messages=[FakeMessage(b'"hi"')])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query({"topic": "events", "limit": 1, "timeout_ms": 50})
    assert r.status == "ok"
    assert r.data["records"][0]["value"] == "hi"


def test_query_respects_limit():
    messages = [FakeMessage(b'{"n":%d}' % i, offset=i) for i in range(20)]
    cons = FakeConsumer(messages=messages)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 5, "timeout_ms": 50}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 5


def test_query_decode_utf8():
    cons = FakeConsumer(messages=[FakeMessage(b"not-json-text")])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 1, "timeout_ms": 50, "decode": "utf8"}')
    assert r.status == "ok"
    assert r.data["records"][0]["value"] == "not-json-text"


def test_query_decode_raw_bytes_hex_fallback():
    # json decode on non-utf8 bytes → hex fallback
    cons = FakeConsumer(messages=[FakeMessage(b"\xff\xfe\x00")])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 1, "timeout_ms": 50}')
    assert r.status == "ok"
    assert r.data["records"][0]["value"] == "fffe00"


def test_query_json_decode_falls_back_to_string():
    cons = FakeConsumer(messages=[FakeMessage(b"not json")])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 1, "timeout_ms": 50}')
    assert r.status == "ok"
    assert r.data["records"][0]["value"] == "not json"


def test_query_rejects_unsafe_topic():
    c, _ = _make()
    c.connect()
    r = c.query('{"topic": "a b"}')
    assert r.status == "error"
    assert "topic" in r.error


def test_query_rejects_empty():
    c, _ = _make()
    c.connect()
    r = c.query("   ")
    assert r.status == "error"
    assert "query is required" in r.error


def test_query_rejects_invalid_json():
    c, _ = _make()
    c.connect()
    r = c.query("{not json")
    assert r.status == "error"
    assert "JSONDecodeError" in r.error


def test_query_rejects_bad_decode_mode():
    c, _ = _make()
    c.connect()
    r = c.query('{"topic": "t", "decode": "protobuf"}')
    assert r.status == "error"
    assert "decode" in r.error


def test_query_stops_on_empty_polls():
    # No messages — poll returns None repeatedly — connector should exit.
    cons = FakeConsumer(messages=[])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 100, "timeout_ms": 10}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 0


def test_query_surfaces_poll_error():
    class BoomConsumer(FakeConsumer):
        def poll(self, t):
            raise RuntimeError("broker down")

    c, _ = _make(consumer=BoomConsumer())
    c.connect()
    r = c.query('{"topic": "t", "timeout_ms": 10}')
    assert r.status == "error"
    assert "RuntimeError" in r.error


def test_query_captures_message_errors():
    msg = FakeMessage(None, err="offset out of range")
    cons = FakeConsumer(messages=[msg])
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.query('{"topic": "t", "limit": 1, "timeout_ms": 50}')
    assert r.status == "ok"
    # The error is captured; record not emitted
    assert r.data["rowcount"] == 0
    assert any("offset out of range" in e for e in r.data["errors"])


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_sums_watermark_lag():
    topics = {"events": FakeTopicMeta(n_partitions=2)}
    wm = {("events", 0): (10, 110), ("events", 1): (0, 50)}
    cons = FakeConsumer(topics_meta=topics, watermarks=wm)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.estimate_cost('{"topic": "events", "limit": 10000}')
    assert r.status == "ok"
    assert r.data["total_messages_available"] == 150  # 100 + 50
    assert r.data["partitions"] == 2
    assert r.data["rows_estimate"] == 150  # clamped by limit=10000


def test_estimate_cost_clamped_by_limit():
    topics = {"events": FakeTopicMeta(n_partitions=1)}
    wm = {("events", 0): (0, 1_000_000)}
    cons = FakeConsumer(topics_meta=topics, watermarks=wm)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.estimate_cost('{"topic": "events", "limit": 50}')
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 50


def test_estimate_cost_unknown_topic():
    cons = FakeConsumer(topics_meta={})
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.estimate_cost('{"topic": "nope"}')
    assert r.status == "error"


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_fields():
    payload = json.dumps({
        "email": "a@x", "first_name": "A", "ip_address": "1.1.1.1", "age": 30,
    }).encode()
    topics = {"users": FakeTopicMeta(n_partitions=1)}
    cons = FakeConsumer(messages=[FakeMessage(payload)], topics_meta=topics)
    c, _ = _make(consumer=cons)
    c.connect()
    r = c.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears_state():
    c, cons = _make()
    c.connect()
    c.close()
    assert cons.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["kafka"] is tool


# ---- _jsonable helper -------------------------------------------------


def test_jsonable_bytes_valid_utf8():
    from agent.connectors.kafka_conn import _jsonable
    assert _jsonable(b"hello") == "hello"


def test_jsonable_bytes_invalid_utf8():
    from agent.connectors.kafka_conn import _jsonable
    result = _jsonable(b"\xff\xfe")
    assert isinstance(result, str)
    assert "ff" in result.lower()


def test_jsonable_mapping():
    from agent.connectors.kafka_conn import _jsonable
    result = _jsonable({"key": b"val"})
    assert result == {"key": "val"}


def test_jsonable_list():
    from agent.connectors.kafka_conn import _jsonable
    result = _jsonable([1, b"x"])
    assert result == [1, "x"]


def test_jsonable_tuple():
    from agent.connectors.kafka_conn import _jsonable
    result = _jsonable((1, 2))
    assert result == [1, 2]


def test_jsonable_unknown_type():
    from agent.connectors.kafka_conn import _jsonable

    class Custom:
        def __str__(self):
            return "custom_obj"

    assert _jsonable(Custom()) == "custom_obj"


# ---- _decode utf8 UnicodeDecodeError ----------------------------------


def test_decode_utf8_unicode_error():
    c, _ = _make()
    result = c._decode(b"\xff\xfe", mode="utf8")
    assert isinstance(result, str)
    assert "ff" in result.lower()


# ---- list_schemas admin error path ------------------------------------


def test_list_schemas_admin_list_topics_exception():
    class BrokenAdmin:
        def list_topics(self, timeout=10):
            raise RuntimeError("broker unavailable")

    c, _ = _make(admin=BrokenAdmin())
    c.connect()
    r = c.list_schemas()
    assert r.status == "error"
    assert "list_topics failed" in r.explanation


# ---- list_schemas consumer metadata error ---------------------------


def test_list_schemas_consumer_metadata_exception():
    class BrokenConsumer(FakeConsumer):
        def list_topics(self, timeout=10):
            raise RuntimeError("meta explosion")

    c = KafkaConnector(
        consumer_factory=lambda *a, **kw: BrokenConsumer(),
        **{"bootstrap.servers": "h:9092"},
    )
    c.connect()
    r = c.list_schemas()
    assert r.status == "error"
