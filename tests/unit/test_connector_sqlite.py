"""Unit tests for SQLiteConnector — uses a real on-disk sqlite file."""
from __future__ import annotations

import sqlite3

import pytest

from agent.connectors.sqlite_conn import SQLiteConnector, register


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, first_name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
        INSERT INTO users VALUES (1,'a@x','A'),(2,'b@x','B');
        INSERT INTO orders VALUES (10,1,9.99),(11,2,4.00);
        CREATE INDEX idx_orders_user ON orders(user_id);
        """
    )
    con.commit()
    con.close()
    return p


@pytest.fixture
def connected(db_path):
    c = SQLiteConnector(path=str(db_path))
    r = c.connect()
    assert r.status == "ok"
    return c


# ---- connect -----------------------------------------------------------


def test_connect_requires_path(monkeypatch):
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    c = SQLiteConnector()
    r = c.connect()
    assert r.status == "error"
    assert "path" in r.explanation.lower()


def test_connect_missing_file(tmp_path):
    c = SQLiteConnector(path=str(tmp_path / "nope.db"))
    r = c.connect()
    assert r.status == "error"
    assert "Path not found" in r.explanation


def test_connect_reads_env(monkeypatch, db_path):
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    c = SQLiteConnector()
    r = c.connect()
    assert r.status == "ok"


def test_connect_is_readonly(connected):
    r = connected.query("INSERT INTO users VALUES (99,'x','y')")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


# ---- operations require connect ---------------------------------------


def test_operations_require_connect(tmp_path):
    c = SQLiteConnector(path=str(tmp_path / "x.db"))
    for method, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", "SELECT 1"),
        ("estimate_cost", "SELECT 1"),
    ):
        r = getattr(c, method)(arg) if arg is not None else getattr(c, method)()
        assert r.status == "error"


# ---- list_schemas ------------------------------------------------------


def test_list_schemas(connected):
    r = connected.list_schemas()
    assert r.status == "ok"
    assert set(r.data) == {"users", "orders"}


# ---- describe_table ----------------------------------------------------


def test_describe_table(connected):
    r = connected.describe_table("users")
    assert r.status == "ok"
    names = [c["name"] for c in r.data["columns"]]
    assert names == ["id", "email", "first_name"]
    email_col = next(c for c in r.data["columns"] if c["name"] == "email")
    assert email_col["nullable"] is False


def test_describe_table_unknown(connected):
    r = connected.describe_table("nope")
    assert r.status == "error"


def test_describe_table_rejects_unsafe(connected):
    r = connected.describe_table("users; drop")
    assert r.status == "error"
    assert "invalid identifier" in r.error


# ---- query -------------------------------------------------------------


def test_query_select(connected):
    r = connected.query("SELECT id, email FROM users ORDER BY id")
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"][0] == {"id": 1, "email": "a@x"}


def test_query_join(connected):
    r = connected.query(
        "SELECT u.email, SUM(o.amount) AS total "
        "FROM users u JOIN orders o ON u.id=o.user_id GROUP BY u.email"
    )
    assert r.status == "ok"
    assert r.data["rowcount"] == 2


def test_query_blocks_destructive(connected):
    for bad in (
        "DELETE FROM users",
        "UPDATE users SET email='x'",
        "DROP TABLE users",
        "ATTACH DATABASE 'evil' AS e",
        "PRAGMA foreign_keys=OFF",
    ):
        r = connected.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError"), bad


def test_query_rejects_empty(connected):
    r = connected.query("   ")
    assert r.status == "error"


def test_query_surfaces_syntax_error(connected):
    r = connected.query("SELECT frobnicate(*) FROM users")
    assert r.status == "error"


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_returns_plan(connected):
    r = connected.estimate_cost("SELECT * FROM users")
    assert r.status == "ok"
    assert len(r.data["plan"]) >= 1
    assert r.data["full_scan_detected"] is True  # no index


def test_estimate_cost_blocks_destructive(connected):
    r = connected.estimate_cost("DELETE FROM users")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


# ---- detect_pii --------------------------------------------------------


def test_detect_pii(connected):
    r = connected.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears_connection(connected):
    connected.close()
    r = connected.list_schemas()
    assert r.status == "error"


def test_memory_db():
    c = SQLiteConnector(path=":memory:")
    r = c.connect()
    assert r.status == "ok"
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == []


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["sqlite"] is tool
