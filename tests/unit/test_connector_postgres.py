"""Unit tests for PostgresConnector.

Unit tests use a duck-typed DB-API 2.0 mock — no real Postgres. The fixture at
the bottom optionally runs against the `deploy-postgres-1` service when psycopg
is installed and the host is reachable (skipped otherwise).
"""
from __future__ import annotations

import json
import os

import pytest

from agent.connectors.postgres_conn import PostgresConnector, register


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
            raise RuntimeError(f"FakeCursor has no scripted response for: {sql}")
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
    cfg = {"host": "h", "user": "u", "password": "p", "dbname": "d", **config}
    conn = FakeConnection(cursors)
    return PostgresConnector(connect_fn=lambda *a, **k: conn, **cfg), conn


# ---- connect ------------------------------------------------------------


def test_connect_requires_dsn_or_host(monkeypatch):
    for k in ("HOST", "DSN", "USER", "PASSWORD", "DBNAME"):
        monkeypatch.delenv(f"POSTGRES_{k}", raising=False)
    c = PostgresConnector(connect_fn=lambda *a, **k: FakeConnection([]))
    r = c.connect()
    assert r.status == "error"
    assert "dsn" in r.explanation.lower()


def test_connect_with_dsn_positional():
    captured = {}
    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeConnection([])
    c = PostgresConnector(connect_fn=fake, dsn="postgresql://u:p@h/d")
    r = c.connect()
    assert r.status == "ok"
    assert captured["args"] == ("postgresql://u:p@h/d",)
    assert r.data["config"]["dsn"] == "***"


def test_connect_with_kwargs():
    captured = {}
    def fake(**kw):
        captured.update(kw)
        return FakeConnection([])
    c = PostgresConnector(connect_fn=fake, host="h", user="u", password="p", dbname="d")
    r = c.connect()
    assert r.status == "ok"
    assert captured["host"] == "h"
    assert r.data["config"]["password"] == "***"


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("connection refused")
    c = PostgresConnector(connect_fn=boom, host="h")
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- operations require connect ----------------------------------------


def test_operations_require_connect():
    c = PostgresConnector(connect_fn=lambda **kw: FakeConnection([]), host="h")
    for method, arg in (("list_schemas", None), ("describe_table", "t"), ("query", "SELECT 1"), ("estimate_cost", "SELECT 1")):
        r = getattr(c, method)(arg) if arg is not None else getattr(c, method)()
        assert r.status == "error"


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_filters_system():
    cur = FakeCursor([(
        [("schema_name",)],
        [("public",), ("analytics",)],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.list_schemas()
    assert r.status == "ok"
    assert r.data == ["public", "analytics"]
    assert "information_schema.schemata" in cur.last_sql


# ---- describe_table ----------------------------------------------------


def test_describe_table_default_schema_public():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [("id", "integer", "NO"), ("email", "text", "YES")],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.describe_table("users")
    assert r.status == "ok"
    assert cur.last_params == ("public", "users")
    assert r.data["columns"][0] == {"name": "id", "type": "integer", "nullable": False}
    assert r.data["columns"][1]["nullable"] is True


def test_describe_table_schema_qualified():
    cur = FakeCursor([([("column_name",), ("data_type",), ("is_nullable",)], [])])
    conn, _ = _make([cur])
    conn.connect()
    conn.describe_table("analytics.users")
    assert cur.last_params == ("analytics", "users")


def test_describe_table_rejects_unsafe_identifier():
    conn, _ = _make([])
    conn.connect()
    r = conn.describe_table("users; DROP TABLE x")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_rejects_triple_dotted():
    conn, _ = _make([])
    conn.connect()
    r = conn.describe_table("a.b.c")
    assert r.status == "error"


# ---- query -------------------------------------------------------------


def test_query_happy_path():
    cur = FakeCursor([(
        [("id",), ("name",)],
        [(1, "alice"), (2, "bob")],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.query("SELECT id, name FROM users")
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"] == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]


def test_query_blocks_destructive():
    conn, _ = _make([])
    conn.connect()
    for bad in ("DELETE FROM users", "DROP TABLE users", "UPDATE users SET x=1", "INSERT INTO u VALUES (1)", "TRUNCATE u"):
        r = conn.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError")


def test_query_passes_params():
    cur = FakeCursor([(None, [])])
    conn, _ = _make([cur])
    conn.connect()
    conn.query("SELECT * FROM t WHERE id = %(id)s", params={"id": 7})
    assert cur.last_params == {"id": 7}


def test_query_surfaces_execution_error():
    class BoomCursor(FakeCursor):
        def execute(self, sql, params=None):
            raise ValueError("relation does not exist")
    conn, _ = _make([BoomCursor([])])
    conn.connect()
    r = conn.query("SELECT * FROM nope")
    assert r.status == "error"
    assert "ValueError" in r.error


def test_query_empty():
    conn, _ = _make([])
    conn.connect()
    r = conn.query("   ")
    assert r.status == "error"


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_parses_explain_plan_object():
    # psycopg3 style: already-parsed JSON.
    plan = [{"Plan": {"Plan Rows": 1_000_000, "Plan Width": 64, "Total Cost": 12345.67}}]
    cur = FakeCursor([([("QUERY PLAN",)], [(plan,)])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.estimate_cost("SELECT * FROM big")
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 1_000_000
    assert r.data["row_width_bytes"] == 64
    assert r.data["bytes_estimate"] == 64_000_000
    assert r.data["total_cost"] == 12345.67
    assert "rows × width" in r.math_trace


def test_estimate_cost_parses_explain_plan_string():
    # psycopg2 style: JSON as a string.
    plan = [{"Plan": {"Plan Rows": 50, "Plan Width": 16, "Total Cost": 3.5}}]
    cur = FakeCursor([([("QUERY PLAN",)], [(json.dumps(plan),)])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.estimate_cost("SELECT 1")
    assert r.status == "ok"
    assert r.data["bytes_estimate"] == 800


def test_estimate_cost_blocks_destructive():
    conn, _ = _make([])
    conn.connect()
    r = conn.estimate_cost("DELETE FROM users")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_estimate_cost_handles_malformed_plan():
    cur = FakeCursor([([("QUERY PLAN",)], [("not-json",)])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.estimate_cost("SELECT 1")
    assert r.status == "ok"  # graceful fallback
    assert r.data["rows_estimate"] is None


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_columns():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [
            ("id", "integer", "NO"),
            ("email", "text", "YES"),
            ("last_name", "text", "YES"),
            ("postal_code", "text", "YES"),
            ("ip_address", "inet", "YES"),
            ("created_at", "timestamptz", "NO"),
        ],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "last_name" in flagged["name"]
    assert "postal_code" in flagged["address"]
    assert "ip_address" in flagged["ip"]


# ---- close + registry --------------------------------------------------


def test_close_clears_connection():
    cur = FakeCursor([])
    c, raw = _make([cur])
    c.connect()
    c.close()
    assert raw.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["postgres"] is tool


# ---- integration against deploy-postgres-1 (optional) ------------------


def _psycopg_available():
    try:
        import psycopg  # noqa: F401
        return True
    except ImportError:
        return False


def _postgres_reachable(host, port, user, password, dbname):
    if not _psycopg_available():
        return False
    import psycopg
    try:
        with psycopg.connect(host=host, port=port, user=user, password=password, dbname=dbname, connect_timeout=3):
            return True
    except Exception:
        return False


PG_HOST = os.environ.get("TEST_PG_HOST", "postgres")  # compose service name inside network
PG_PORT = int(os.environ.get("TEST_PG_PORT", "5432"))
PG_USER = os.environ.get("TEST_PG_USER", "mlflow")
PG_PASSWORD = os.environ.get("TEST_PG_PASSWORD", "mlflow")
PG_DB = os.environ.get("TEST_PG_DB", "mlflow")


@pytest.mark.skipif(
    not _postgres_reachable(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB),
    reason="Real Postgres not reachable — skipping integration test.",
)
def test_integration_real_postgres():
    c = PostgresConnector(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB)
    r = c.connect()
    assert r.status == "ok", r.error
    try:
        r = c.list_schemas()
        assert r.status == "ok"
        assert "public" in r.data

        r = c.query("SELECT 1 AS one, 'hi' AS greeting")
        assert r.status == "ok"
        assert r.data["records"] == [{"one": 1, "greeting": "hi"}]

        r = c.query("DROP TABLE foo")
        assert r.status == "error"
        assert r.error.startswith("SafetyError")

        r = c.estimate_cost("SELECT generate_series(1, 1000)")
        assert r.status == "ok"
        assert r.data["rows_estimate"] is not None
    finally:
        c.close()
