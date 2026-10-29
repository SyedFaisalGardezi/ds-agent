"""Unit tests for the Snowflake connector.

The real `snowflake.connector.connect` is never called — we inject a fake
`connect_fn` that returns a duck-typed DB-API 2.0 connection. This lets the
full code path (identifier validation, forbidden-SQL guard, describe_table,
query, EXPLAIN parsing, PII heuristic) run without any Snowflake credentials.
"""
from __future__ import annotations

import json

from agent.connectors.snowflake_conn import SnowflakeConnector, register


class FakeCursor:
    """Minimal DB-API 2.0 cursor stub. Results queued as (description, rows)."""

    def __init__(self, script: list[tuple[list[tuple[str, ...]] | None, list[tuple]]]):
        # `script` is a list of (description, rows) pairs, consumed in order.
        self._script = list(script)
        self.description = None
        self._rows: list[tuple] = []
        self.last_sql: str | None = None
        self.last_params: dict | None = None

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
    def __init__(self, cursors: list[FakeCursor]):
        self._cursors = list(cursors)
        self.closed = False

    def cursor(self):
        if not self._cursors:
            raise RuntimeError("FakeConnection ran out of cursors.")
        return self._cursors.pop(0)

    def close(self):
        self.closed = True


def _make(cursors: list[FakeCursor], **config):
    cfg = {"account": "acc", "user": "u", "password": "p", **config}
    conn = FakeConnection(cursors)
    return SnowflakeConnector(connect_fn=lambda **kw: conn, **cfg), conn


# ---- connect() ----------------------------------------------------------


def test_connect_requires_credentials(monkeypatch):
    # Ensure env is clean so the connector has nothing to fall back on.
    for key in ("ACCOUNT", "USER", "PASSWORD"):
        monkeypatch.delenv(f"SNOWFLAKE_{key}", raising=False)
    c = SnowflakeConnector(connect_fn=lambda **kw: FakeConnection([]))
    r = c.connect()
    assert r.status == "error"
    assert "Missing required" in r.explanation


def test_connect_reads_env_vars(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env-acc")
    monkeypatch.setenv("SNOWFLAKE_USER", "env-user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "env-pw")
    captured = {}
    def fake(**kw):
        captured.update(kw)
        return FakeConnection([])
    c = SnowflakeConnector(connect_fn=fake)
    r = c.connect()
    assert r.status == "ok"
    assert captured["account"] == "env-acc"
    # Masked in data payload.
    assert r.data["config"]["password"] == "***"


def test_connect_explicit_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env-acc")
    monkeypatch.setenv("SNOWFLAKE_USER", "env-user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "env-pw")
    captured = {}
    c = SnowflakeConnector(
        connect_fn=lambda **kw: (captured.update(kw), FakeConnection([]))[1],
        account="kw-acc",
    )
    r = c.connect()
    assert r.status == "ok"
    assert captured["account"] == "kw-acc"


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("network down")
    c = SnowflakeConnector(connect_fn=boom, account="a", user="u", password="p")
    r = c.connect()
    assert r.status == "error"
    assert r.error.startswith("RuntimeError")


# ---- operations require prior connect() --------------------------------


def test_operations_require_connect():
    c = SnowflakeConnector(connect_fn=lambda **kw: FakeConnection([]), account="a", user="u", password="p")
    for method, arg in (("list_schemas", None), ("describe_table", "T"), ("query", "SELECT 1"), ("estimate_cost", "SELECT 1")):
        r = getattr(c, method)(arg) if arg is not None else getattr(c, method)()
        assert r.status == "error"
        assert "Not connected" in r.explanation


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_extracts_names():
    cur = FakeCursor([(
        [("created_on",), ("name",), ("is_default",)],
        [("2024-01-01", "PUBLIC", "Y"), ("2024-01-02", "ANALYTICS", "N")],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.list_schemas()
    assert r.status == "ok"
    assert r.data == ["PUBLIC", "ANALYTICS"]
    assert cur.last_sql == "SHOW SCHEMAS"


# ---- describe_table ----------------------------------------------------


def test_describe_table_returns_columns():
    cur = FakeCursor([(
        [("name",), ("type",), ("kind",)],
        [("ID", "NUMBER", "COLUMN"), ("EMAIL", "VARCHAR", "COLUMN")],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.describe_table("ANALYTICS.USERS")
    assert r.status == "ok"
    assert r.data["columns"][0] == {"name": "ID", "type": "NUMBER"}
    assert "ANALYTICS.USERS" in cur.last_sql


def test_describe_table_rejects_unsafe_identifier():
    conn, _ = _make([])
    conn.connect()
    r = conn.describe_table("users; DROP TABLE x")
    assert r.status == "error"
    assert "invalid identifier" in r.error


# ---- query -------------------------------------------------------------


def test_query_returns_records():
    cur = FakeCursor([(
        [("id",), ("email",)],
        [(1, "a@x"), (2, "b@x"), (3, "c@x")],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.query("SELECT id, email FROM users")
    assert r.status == "ok"
    assert r.data["rowcount"] == 3
    assert r.data["columns"] == ["id", "email"]
    assert r.data["records"][0] == {"id": 1, "email": "a@x"}


def test_query_blocks_destructive_sql():
    conn, _ = _make([])
    conn.connect()
    for bad in ("DROP TABLE users", "DELETE FROM users", "UPDATE users SET x=1", "INSERT INTO users VALUES (1)"):
        r = conn.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError")


def test_query_rejects_empty():
    conn, _ = _make([])
    conn.connect()
    r = conn.query("   ")
    assert r.status == "error"
    assert "query is required" in r.error


def test_query_surfaces_execution_error():
    class BoomCursor(FakeCursor):
        def execute(self, sql, params=None):
            raise ValueError("bad syntax")
    conn, _ = _make([BoomCursor([])])
    conn.connect()
    r = conn.query("SELECT xyz")
    assert r.status == "error"
    assert "ValueError" in r.error


def test_query_passes_params():
    cur = FakeCursor([(None, [])])  # no description → no results
    conn, _ = _make([cur])
    conn.connect()
    conn.query("SELECT * FROM t WHERE id = %(id)s", params={"id": 42})
    assert cur.last_params == {"id": 42}


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_parses_explain_json():
    plan = {
        "GlobalStats": {"partitionsTotal": 12, "bytesAssigned": 999999},
        "Operations": [{"rowCount": 1_000_000}],
    }
    cur = FakeCursor([([("plan",)], [(json.dumps(plan),)])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.estimate_cost("SELECT count(*) FROM big_table")
    assert r.status == "ok"
    assert r.data["partitions"] == 12
    assert r.data["bytes"] == 999_999
    assert r.data["rows_estimate"] == 1_000_000
    assert "partitions" in r.math_trace.lower()


def test_estimate_cost_blocks_destructive():
    conn, _ = _make([])
    conn.connect()
    r = conn.estimate_cost("DROP TABLE t")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_estimate_cost_handles_non_json_plan():
    cur = FakeCursor([([("plan",)], [("not-json",)])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.estimate_cost("SELECT 1")
    # Fields will be None, but status should still be ok and data.plan is the raw string wrapper.
    assert r.status == "ok"
    assert r.data["plan"] == {"raw": "not-json"}


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_by_column_name():
    cur = FakeCursor([(
        [("name",), ("type",)],
        [
            ("id", "NUMBER"),
            ("email", "VARCHAR"),
            ("first_name", "VARCHAR"),
            ("ip_address", "VARCHAR"),
            ("created_at", "TIMESTAMP"),
        ],
    )])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.detect_pii("USERS")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]
    assert "id" not in {c for cols in flagged.values() for c in cols}


def test_detect_pii_empty_table():
    cur = FakeCursor([([("name",), ("type",)], [])])
    conn, _ = _make([cur])
    conn.connect()
    r = conn.detect_pii("T")
    assert r.status == "ok"
    assert r.data["pii_columns"] == {}


# ---- close + registry -------------------------------------------------


def test_close_clears_connection():
    cur = FakeCursor([])
    conn, raw = _make([cur])
    conn.connect()
    conn.close()
    assert raw.closed is True
    # Subsequent operations must now fail with "Not connected".
    r = conn.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    """Only checks binding; does not attempt to connect."""
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["snowflake"] is tool
