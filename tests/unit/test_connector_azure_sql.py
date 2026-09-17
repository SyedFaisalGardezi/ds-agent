"""Unit tests for AzureSQLConnector — duck-typed DB-API mock."""
from __future__ import annotations

from agent.connectors.azure_sql_conn import AzureSQLConnector, register


class FakeCursor:
    def __init__(self, script):
        self._script = list(script)
        self.description = None
        self._rows: list[tuple] = []
        self.last_sql: str | None = None
        self.last_params = None
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params
        self.executed.append((sql, params))
        if self._script:
            desc, rows = self._script.pop(0)
            self.description = desc
            self._rows = rows
        else:
            self.description = None
            self._rows = []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


# pyodbc.connect takes a positional DSN string; capture it.
class FakeConnection:
    def __init__(self, cursors):
        self._cursors = list(cursors)
        self.closed = False

    def cursor(self):
        return self._cursors.pop(0)

    def close(self):
        self.closed = True


from typing import Any  # noqa: E402


def _make(cursors, **config):
    cfg = {"server": "srv", "database": "db", "user": "u", "password": "p", **config}
    conn = FakeConnection(cursors)
    captured: dict[str, Any] = {}

    def fake(dsn):
        captured["dsn"] = dsn
        return conn

    return (
        AzureSQLConnector(connect_fn=fake, **cfg),
        conn,
        captured,
    )


# ---- connect -----------------------------------------------------------


def test_connect_requires_config(monkeypatch):
    for k in ("SERVER", "DATABASE", "DSN", "USER", "PASSWORD", "DRIVER", "PORT",
              "ENCRYPT", "TRUST_SERVER_CERTIFICATE"):
        monkeypatch.delenv(f"AZURE_SQL_{k}", raising=False)
    c = AzureSQLConnector(connect_fn=lambda dsn: FakeConnection([]))
    r = c.connect()
    assert r.status == "error"
    assert "server" in r.explanation.lower() or "dsn" in r.explanation.lower()


def test_connect_builds_odbc_dsn():
    c, _, captured = _make([])
    r = c.connect()
    assert r.status == "ok"
    assert "SERVER=srv" in captured["dsn"]
    assert "DATABASE=db" in captured["dsn"]
    assert "UID=u" in captured["dsn"]
    assert "PWD=p" in captured["dsn"]
    assert "Encrypt=yes" in captured["dsn"]


def test_connect_with_explicit_dsn():
    captured: dict[str, Any] = {}

    def fake(dsn):
        captured["dsn"] = dsn
        return FakeConnection([])

    c = AzureSQLConnector(connect_fn=fake, dsn="DSN=prod;UID=x;PWD=y;")
    r = c.connect()
    assert r.status == "ok"
    assert captured["dsn"] == "DSN=prod;UID=x;PWD=y;"
    assert r.data["config"]["dsn"] == "***"


def test_connect_masks_password():
    c, _, _ = _make([])
    r = c.connect()
    assert r.data["config"]["password"] == "***"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("AZURE_SQL_SERVER", "envsrv")
    monkeypatch.setenv("AZURE_SQL_DATABASE", "envdb")
    monkeypatch.setenv("AZURE_SQL_PORT", "1433")
    captured: dict[str, Any] = {}

    def fake(dsn):
        captured["dsn"] = dsn
        return FakeConnection([])

    c = AzureSQLConnector(connect_fn=fake)
    r = c.connect()
    assert r.status == "ok"
    assert "SERVER=envsrv,1433" in captured["dsn"]


def test_connect_surfaces_driver_error():
    def boom(dsn):
        raise RuntimeError("login failed")

    c = AzureSQLConnector(connect_fn=boom, server="s", database="d")
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- operations require connect ---------------------------------------


def test_operations_require_connect():
    c = AzureSQLConnector(connect_fn=lambda dsn: FakeConnection([]), server="s", database="d")
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", "SELECT 1"),
        ("estimate_cost", "SELECT 1"),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_filters_system():
    cur = FakeCursor([(
        [("schema_name",)],
        [("dbo",), ("analytics",)],
    )])
    c, _, _ = _make([cur])
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["dbo", "analytics"]
    assert "information_schema.schemata" in cur.last_sql


# ---- describe_table ----------------------------------------------------


def test_describe_table_default_schema_dbo():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [("id", "int", "NO"), ("email", "nvarchar", "YES")],
    )])
    c, _, _ = _make([cur])
    c.connect()
    r = c.describe_table("users")
    assert r.status == "ok"
    assert cur.last_params == ("dbo", "users")
    assert r.data["columns"][0] == {"name": "id", "type": "int", "nullable": False}
    assert r.data["columns"][1]["nullable"] is True


def test_describe_table_schema_qualified():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [("id", "int", "NO")],
    )])
    c, _, _ = _make([cur])
    c.connect()
    c.describe_table("analytics.users")
    assert cur.last_params == ("analytics", "users")


def test_describe_table_rejects_unsafe():
    c, _, _ = _make([])
    c.connect()
    r = c.describe_table("users; DROP TABLE x")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_rejects_triple_dotted():
    c, _, _ = _make([])
    c.connect()
    r = c.describe_table("a.b.c")
    assert r.status == "error"


def test_describe_table_unknown():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [],
    )])
    c, _, _ = _make([cur])
    c.connect()
    r = c.describe_table("ghost")
    assert r.status == "error"
    assert "Unknown table" in r.explanation


# ---- query -------------------------------------------------------------


def test_query_happy_path():
    cur = FakeCursor([(
        [("id",), ("name",)],
        [(1, "alice"), (2, "bob")],
    )])
    c, _, _ = _make([cur])
    c.connect()
    r = c.query("SELECT id, name FROM users")
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"][0] == {"id": 1, "name": "alice"}


def test_query_blocks_destructive():
    c, _, _ = _make([])
    c.connect()
    for bad in (
        "DELETE FROM users",
        "DROP TABLE users",
        "UPDATE users SET x=1",
        "INSERT INTO u VALUES (1)",
        "TRUNCATE TABLE u",
        "MERGE INTO t USING s ON t.id=s.id WHEN MATCHED THEN UPDATE SET a=1",
        "EXEC sp_something",
        "BACKUP DATABASE x TO DISK='y'",
    ):
        r = c.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError"), bad


def test_query_passes_params():
    cur = FakeCursor([(None, [])])
    c, _, _ = _make([cur])
    c.connect()
    c.query("SELECT * FROM t WHERE id = ?", params=(7,))
    assert cur.last_params == (7,)


def test_query_empty():
    c, _, _ = _make([])
    c.connect()
    r = c.query("   ")
    assert r.status == "error"


def test_query_surfaces_error():
    class Boom(FakeCursor):
        def execute(self, sql, params=None):
            raise ValueError("login timeout")

    c, _, _ = _make([Boom([])])
    c.connect()
    r = c.query("SELECT * FROM t")
    assert r.status == "error"
    assert "ValueError" in r.error


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_parses_showplan_xml():
    plan_xml = (
        '<ShowPlanXML><Statements><StmtSimple '
        'StatementEstRows="1234.5" StatementSubTreeCost="0.25" />'
        '</Statements></ShowPlanXML>'
    )
    # Script: SET SHOWPLAN_XML ON (no rows), EXEC sql (1 row with plan), SET ... OFF
    cur = FakeCursor([
        (None, []),
        ([("XML",)], [(plan_xml,)]),
        (None, []),
    ])
    c, _, _ = _make([cur])
    c.connect()
    r = c.estimate_cost("SELECT * FROM big")
    assert r.status == "ok"
    assert r.data["rows_estimate"] == 1234
    assert abs(r.data["total_subtree_cost"] - 0.25) < 1e-9


def test_estimate_cost_blocks_destructive():
    c, _, _ = _make([])
    c.connect()
    r = c.estimate_cost("DELETE FROM users")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_estimate_cost_handles_missing_plan():
    cur = FakeCursor([
        (None, []),
        (None, []),  # no plan row
        (None, []),
    ])
    c, _, _ = _make([cur])
    c.connect()
    r = c.estimate_cost("SELECT 1")
    assert r.status == "ok"
    assert r.data["rows_estimate"] is None
    assert r.data["plan_xml"] is None


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_columns():
    cur = FakeCursor([(
        [("column_name",), ("data_type",), ("is_nullable",)],
        [
            ("id", "int", "NO"),
            ("email", "nvarchar", "YES"),
            ("first_name", "nvarchar", "YES"),
            ("ip_address", "nvarchar", "YES"),
        ],
    )])
    c, _, _ = _make([cur])
    c.connect()
    r = c.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears():
    c, raw, _ = _make([])
    c.connect()
    c.close()
    assert raw.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["azure_sql"] is tool
