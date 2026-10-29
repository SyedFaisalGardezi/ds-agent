"""Unit tests for MySQLConnector — duck-typed DB-API mock."""
from __future__ import annotations

import json

from agent.connectors.mysql_conn import MySQLConnector, register


class FakeCursor:
    def __init__(self, script):
        self._script = list(script)
        self.description = None
        self._rows: list[tuple] = []
        self.last_sql: str | None = None
        self.last_params = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params
        if not self._script:
            raise RuntimeError(f"FakeCursor out of script: {sql}")
        desc, rows = self._script.pop(0)
        self.description = desc
        self._rows = rows

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursors):
        self._cursors = list(cursors)
        self.closed = False

    def cursor(self):
        return self._cursors.pop(0)

    def close(self):
        self.closed = True


def _make(cursors, **config):
    cfg = {"host": "h", "user": "u", "password": "p", "database": "d", **config}
    conn = FakeConnection(cursors)
    return MySQLConnector(connect_fn=lambda **kw: conn, **cfg), conn


def test_connect_requires_host_or_db(monkeypatch):
    for k in ("HOST", "DATABASE", "PORT", "USER", "PASSWORD"):
        monkeypatch.delenv(f"MYSQL_{k}", raising=False)
    c = MySQLConnector(connect_fn=lambda **kw: FakeConnection([]))
    r = c.connect()
    assert r.status == "error"
    assert "host" in r.explanation.lower()


def test_connect_masks_password():
    c, _ = _make([])
    r = c.connect()
    assert r.status == "ok"
    assert r.data["config"]["password"] == "***"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "envhost")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return FakeConnection([])

    c = MySQLConnector(connect_fn=fake)
    r = c.connect()
    assert r.status == "ok"
    assert captured["host"] == "envhost"
    assert captured["port"] == 3307


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("refused")

    c = MySQLConnector(connect_fn=boom, host="h", database="d")
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


def test_operations_require_connect():
    c = MySQLConnector(connect_fn=lambda **kw: FakeConnection([]), host="h", database="d")
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", "SELECT 1"),
        ("estimate_cost", "SELECT 1"),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"


def test_list_schemas_filters_system():
    cur = FakeCursor([(
        [("schema_name",)],
        [("appdb",), ("analytics",)],
    )])
    c, _ = _make([cur])
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["appdb", "analytics"]
    assert "information_schema.schemata" in cur.last_sql


def test_describe_table_default():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [("id", "int", "NO"), ("email", "varchar", "YES")],
    )])
    c, _ = _make([cur])
    c.connect()
    r = c.describe_table("users")
    assert r.status == "ok"
    assert cur.last_params == ("users",)
    assert r.data["columns"][0] == {"name": "id", "type": "int", "nullable": False}
    assert r.data["columns"][1]["nullable"] is True


def test_describe_table_schema_qualified():
    cur = FakeCursor([([("column_name",), ("data_type",), ("is_nullable",)], [("id", "int", "NO")])])
    c, _ = _make([cur])
    c.connect()
    c.describe_table("analytics.users")
    assert cur.last_params == ("analytics", "users")


def test_describe_table_rejects_unsafe():
    c, _ = _make([])
    c.connect()
    r = c.describe_table("users; drop")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_rejects_triple_dotted():
    c, _ = _make([])
    c.connect()
    r = c.describe_table("a.b.c")
    assert r.status == "error"


def test_describe_table_unknown():
    cur = FakeCursor([([("column_name",), ("data_type",), ("is_nullable",)], [])])
    c, _ = _make([cur])
    c.connect()
    r = c.describe_table("ghost")
    assert r.status == "error"
    assert "Unknown table" in r.explanation


def test_query_happy_path():
    cur = FakeCursor([(
        [("id",), ("name",)],
        [(1, "alice"), (2, "bob")],
    )])
    c, _ = _make([cur])
    c.connect()
    r = c.query("SELECT id, name FROM users")
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"][0] == {"id": 1, "name": "alice"}


def test_query_blocks_destructive():
    c, _ = _make([])
    c.connect()
    for bad in (
        "DELETE FROM users",
        "DROP TABLE users",
        "UPDATE users SET x=1",
        "INSERT INTO u VALUES (1)",
        "TRUNCATE u",
        "LOAD DATA INFILE '/etc/passwd' INTO TABLE x",
        "LOCK TABLES u WRITE",
    ):
        r = c.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError"), bad


def test_query_empty():
    c, _ = _make([])
    c.connect()
    r = c.query("   ")
    assert r.status == "error"


def test_query_surfaces_error():
    class Boom(FakeCursor):
        def execute(self, sql, params=None):
            raise ValueError("no such table")

    c, _ = _make([Boom([])])
    c.connect()
    r = c.query("SELECT * FROM nope")
    assert r.status == "error"
    assert "ValueError" in r.error


def test_estimate_cost_parses_explain_json():
    plan = {
        "query_block": {
            "cost_info": {"query_cost": "42.5"},
            "table": {"rows_examined_per_scan": 1000},
        }
    }
    cur = FakeCursor([([("EXPLAIN",)], [(json.dumps(plan),)])])
    c, _ = _make([cur])
    c.connect()
    r = c.estimate_cost("SELECT * FROM big")
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 1000


def test_estimate_cost_handles_non_json():
    cur = FakeCursor([([("EXPLAIN",)], [("not-json",)])])
    c, _ = _make([cur])
    c.connect()
    r = c.estimate_cost("SELECT 1")
    assert r.status == "ok"
    assert r.data["rows_estimate"] is None


def test_estimate_cost_blocks_destructive():
    c, _ = _make([])
    c.connect()
    r = c.estimate_cost("DELETE FROM users")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_detect_pii_flags():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [
            ("id", "int", "NO"),
            ("email", "varchar", "YES"),
            ("first_name", "varchar", "YES"),
            ("ip_address", "varchar", "YES"),
        ],
    )])
    c, _ = _make([cur])
    c.connect()
    r = c.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


def test_close_clears():
    c, raw = _make([])
    c.connect()
    c.close()
    assert raw.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["mysql"] is tool
