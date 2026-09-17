"""Unit tests for RedisConnector — duck-typed client, no redis-py dep."""
from __future__ import annotations

import json

from agent.connectors.redis_conn import RedisConnector, register


class FakeRedis:
    """Minimal duck type implementing the read subset used by the connector."""

    def __init__(self, data=None, hashes=None, lists=None, sets=None, zsets=None, types=None):
        self._data = data or {}       # str -> bytes/str
        self._hashes = hashes or {}   # str -> dict
        self._lists = lists or {}     # str -> list
        self._sets = sets or {}       # str -> set
        self._zsets = zsets or {}     # str -> list (sorted)
        self._types = types or {}     # override type if needed
        self.closed = False
        self.pinged = False

    # --- introspection ----
    def ping(self):
        self.pinged = True
        return True

    def dbsize(self):
        return (
            len(self._data) + len(self._hashes) + len(self._lists)
            + len(self._sets) + len(self._zsets)
        )

    def _all_keys(self):
        keys = set(self._data) | set(self._hashes) | set(self._lists) | set(self._sets) | set(self._zsets)
        return sorted(keys)

    def scan_iter(self, match="*", count=100):
        import fnmatch
        for k in self._all_keys():
            if fnmatch.fnmatchcase(k, match):
                yield k.encode()

    def type(self, key):
        k = key.decode() if isinstance(key, bytes) else key
        if k in self._types:
            return self._types[k]
        if k in self._hashes:
            return b"hash"
        if k in self._lists:
            return b"list"
        if k in self._sets:
            return b"set"
        if k in self._zsets:
            return b"zset"
        if k in self._data:
            return b"string"
        return b"none"

    def ttl(self, key):
        return -1

    # --- reads ----
    def get(self, key):
        return self._data.get(key)

    def mget(self, keys):
        return [self._data.get(k) for k in keys]

    def hget(self, key, field):
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return {k.encode(): v.encode() if isinstance(v, str) else v
                for k, v in self._hashes.get(key, {}).items()}

    def lrange(self, key, start, stop):
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    def smembers(self, key):
        return set(self._sets.get(key, set()))

    def zrange(self, key, start, stop):
        lst = self._zsets.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    def close(self):
        self.closed = True


def _make(fake=None, **config):
    fake = fake or FakeRedis()
    return RedisConnector(client_factory=lambda **kw: fake, host="h", **config), fake


# ---- connect -----------------------------------------------------------


def test_connect_defaults_host():
    c, fake = _make()
    r = c.connect()
    assert r.status == "ok"
    assert fake.pinged is True
    assert r.data["config"]["host"] == "h"
    assert r.data["config"]["port"] == 6379


def test_connect_masks_password():
    c, _ = _make(password="s3cret")
    r = c.connect()
    assert r.status == "ok"
    assert r.data["config"]["password"] == "***"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "envhost")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "3")
    captured = {}

    def factory(**kw):
        captured.update(kw)
        return FakeRedis()

    c = RedisConnector(client_factory=factory)
    r = c.connect()
    assert r.status == "ok"
    assert captured["host"] == "envhost"
    assert captured["port"] == 6380
    assert captured["db"] == 3


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("refused")

    c = RedisConnector(client_factory=boom)
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- require-connect ---------------------------------------------------


def test_operations_require_connect():
    c = RedisConnector(client_factory=lambda **kw: FakeRedis())
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", '{"op":"get","key":"x"}'),
        ("estimate_cost", '{"op":"get"}'),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"
        assert "Not connected" in r.explanation


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_returns_namespaces():
    fake = FakeRedis(
        data={"users:1": b"a", "users:2": b"b", "cache:x": b"c"},
        hashes={"orders:99": {"amount": "5"}},
    )
    c, _ = _make(fake=fake)
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert set(r.data) == {"users", "cache", "orders"}


# ---- describe_table ----------------------------------------------------


def test_describe_table_samples_keys_and_types():
    fake = FakeRedis(
        data={"users:1": b"a"},
        hashes={"users:2": {"email": "a@x", "first_name": "A"}},
        lists={"users:3": [b"x"]},
    )
    c, _ = _make(fake=fake)
    c.connect()
    r = c.describe_table("users", sample_size=10)
    assert r.status == "ok"
    assert r.data["sampled_keys"] == 3
    assert set(r.data["type_counts"].keys()) >= {"string", "hash", "list"}
    names = {col["name"] for col in r.data["columns"]}
    assert names == {"email", "first_name"}


def test_describe_table_rejects_unsafe_key():
    c, _ = _make()
    c.connect()
    r = c.describe_table("users; DROP")
    assert r.status == "error"
    assert "invalid identifier" in r.error


# ---- query -------------------------------------------------------------


def test_query_get():
    fake = FakeRedis(data={"users:1": b"alice"})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "get", "key": "users:1"}')
    assert r.status == "ok"
    assert r.data["result"] == "alice"
    assert r.data["rowcount"] == 1


def test_query_mget():
    fake = FakeRedis(data={"u:1": b"a", "u:2": b"b"})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "mget", "keys": ["u:1", "u:2", "u:404"]}')
    assert r.status == "ok"
    assert r.data["result"] == ["a", "b", None]
    assert r.data["rowcount"] == 3


def test_query_hgetall():
    fake = FakeRedis(hashes={"u:1": {"email": "a@x", "age": "30"}})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "hgetall", "key": "u:1"}')
    assert r.status == "ok"
    assert r.data["result"] == {"email": "a@x", "age": "30"}


def test_query_lrange():
    fake = FakeRedis(lists={"log": [b"a", b"b", b"c", b"d"]})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "lrange", "key": "log", "start": 0, "stop": 1}')
    assert r.status == "ok"
    assert r.data["result"] == ["a", "b"]


def test_query_smembers():
    fake = FakeRedis(sets={"tags": {b"python", b"redis"}})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "smembers", "key": "tags"}')
    assert r.status == "ok"
    assert set(r.data["result"]) == {"python", "redis"}


def test_query_scan():
    fake = FakeRedis(data={"users:1": b"a", "users:2": b"b", "cache:x": b"c"})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "scan", "match": "users:*", "count": 100}')
    assert r.status == "ok"
    assert set(r.data["result"]) == {"users:1", "users:2"}


def test_query_blocks_write_op():
    c, _ = _make()
    c.connect()
    for bad_op in ("set", "del", "flushall", "eval", "config", "shutdown", "expire"):
        r = c.query(json.dumps({"op": bad_op, "key": "x"}))
        assert r.status == "error", bad_op
        assert r.error.startswith("SafetyError"), bad_op


def test_query_rejects_empty():
    c, _ = _make()
    c.connect()
    r = c.query("   ")
    assert r.status == "error"


def test_query_rejects_invalid_json():
    c, _ = _make()
    c.connect()
    r = c.query("{not json")
    assert r.status == "error"
    assert "JSONDecodeError" in r.error


def test_query_rejects_missing_op():
    c, _ = _make()
    c.connect()
    r = c.query("{}")
    assert r.status == "error"
    assert "op" in r.explanation.lower()


def test_query_missing_required_field():
    fake = FakeRedis()
    c, _ = _make(fake=fake)
    c.connect()
    r = c.query('{"op": "get"}')  # no 'key'
    assert r.status == "error"
    assert "KeyError" in r.error


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_get_is_o1():
    fake = FakeRedis(data={"a": b"1", "b": b"2"})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.estimate_cost('{"op": "get", "key": "a"}')
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 1
    assert r.data["dbsize"] == 2
    assert "O(1)" in r.data["complexity"]


def test_estimate_cost_scan_reports_n():
    fake = FakeRedis(data={"a": b"1"})
    c, _ = _make(fake=fake)
    c.connect()
    r = c.estimate_cost('{"op": "scan", "count": 500}')
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 500
    assert "O(N)" in r.data["complexity"]


def test_estimate_cost_blocks_write_op():
    c, _ = _make()
    c.connect()
    r = c.estimate_cost('{"op": "set"}')
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_hash_fields():
    fake = FakeRedis(hashes={
        "users:1": {"email": "a@x", "first_name": "A", "ip_address": "1.1.1.1", "age": "30"},
    })
    c, _ = _make(fake=fake)
    c.connect()
    r = c.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears_client():
    c, fake = _make()
    c.connect()
    c.close()
    assert fake.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["redis"] is tool


# ---- _jsonable helper -------------------------------------------------


def test_redis_jsonable_bytes_valid():
    from agent.connectors.redis_conn import _jsonable
    assert _jsonable(b"hello") == "hello"


def test_redis_jsonable_bytes_invalid_utf8():
    from agent.connectors.redis_conn import _jsonable
    result = _jsonable(b"\xff\xfe")
    assert isinstance(result, str)
    assert "ff" in result.lower()


def test_redis_jsonable_mapping():
    from agent.connectors.redis_conn import _jsonable
    assert _jsonable({"k": b"v"}) == {"k": "v"}


def test_redis_jsonable_list():
    from agent.connectors.redis_conn import _jsonable
    assert _jsonable([b"a", 1]) == ["a", 1]


def test_redis_jsonable_unknown():
    from agent.connectors.redis_conn import _jsonable

    class Obj:
        def __str__(self):
            return "obj"

    assert _jsonable(Obj()) == "obj"


# ---- list_schemas scan_iter error -------------------------------------


def test_list_schemas_scan_error():
    class BrokenRedis(FakeRedis):
        def scan_iter(self, match="*", count=100):
            raise RuntimeError("scan boom")

    c, _ = _make(fake=BrokenRedis())
    c.connect()
    r = c.list_schemas()
    assert r.status == "error"
    assert "scan_iter failed" in r.explanation


# ---- describe_table scan_iter error -----------------------------------


def test_describe_table_scan_error():
    class BrokenRedis(FakeRedis):
        def scan_iter(self, match="*", count=100):
            raise RuntimeError("describe scan boom")

    c, _ = _make(fake=BrokenRedis())
    c.connect()
    r = c.describe_table("users")
    assert r.status == "error"
    assert "scan_iter failed" in r.explanation


# ---- query general exception ------------------------------------------


def test_query_general_exception():
    class BrokenRedis(FakeRedis):
        def get(self, key):
            raise ConnectionError("oops")

    c, _ = _make(fake=BrokenRedis(data={"k": "v"}))
    c.connect()
    r = c.query('{"op": "get", "key": "k"}')
    assert r.status == "error"
    assert "failed" in r.explanation
