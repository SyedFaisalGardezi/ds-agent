from __future__ import annotations

import json

import pandas as pd
import pytest

from agent.connectors.flatfile_conn import FlatFileConnector, register


@pytest.fixture
def data_dir(tmp_path):
    """A tmp directory with three tables in three different formats."""
    users = pd.DataFrame({
        "id": [1, 2, 3],
        "email": ["a@x", "b@x", "c@x"],
        "first_name": ["A", "B", "C"],
    })
    orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 2],
        "amount": [9.99, 19.50, 4.00],
    })
    logs = pd.DataFrame({
        "ts": ["2024-01-01", "2024-01-02"],
        "ip_address": ["10.0.0.1", "10.0.0.2"],
    })
    users.to_csv(tmp_path / "users.csv", index=False)
    orders.to_parquet(tmp_path / "orders.parquet", index=False)
    (tmp_path / "logs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in logs.to_dict(orient="records"))
    )
    return tmp_path


@pytest.fixture
def connected(data_dir):
    c = FlatFileConnector(path=".", root=data_dir)
    r = c.connect()
    assert r.status == "ok"
    return c


# ---- connect ------------------------------------------------------------


def test_connect_requires_path(tmp_path):
    c = FlatFileConnector(path="", root=tmp_path)
    r = c.connect()
    assert r.status == "error"
    assert "Missing path" in r.explanation


def test_connect_rejects_path_escape(tmp_path):
    c = FlatFileConnector(path="../../etc", root=tmp_path)
    r = c.connect()
    assert r.status == "error"
    assert r.error == "PathEscapeError"


def test_connect_missing_path(tmp_path):
    c = FlatFileConnector(path="nope", root=tmp_path)
    r = c.connect()
    assert r.status == "error"
    assert "Path not found" in r.explanation


def test_connect_empty_dir_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    c = FlatFileConnector(path="empty", root=tmp_path)
    r = c.connect()
    assert r.status == "error"
    assert r.error == "NoTablesFound"


def test_connect_single_file(data_dir):
    c = FlatFileConnector(path="users.csv", root=data_dir)
    r = c.connect()
    assert r.status == "ok"
    assert r.data["tables"] == ["users"]


def test_connect_directory_discovers_all_formats(connected):
    r = connected.list_schemas()
    assert set(r.data) == {"users", "orders", "logs"}


# ---- describe_table -----------------------------------------------------


def test_describe_table_reports_shape_and_dtypes(connected):
    r = connected.describe_table("users")
    assert r.status == "ok"
    assert r.data["shape"] == [3, 3]
    col_names = [c["name"] for c in r.data["columns"]]
    assert "email" in col_names
    assert "Shape: 3 × 3" in r.math_trace


def test_describe_table_unknown(connected):
    r = connected.describe_table("nonexistent")
    assert r.status == "error"
    assert "Unknown table" in r.explanation


# ---- query --------------------------------------------------------------


def test_query_simple_select(connected):
    r = connected.query("SELECT id, email FROM users ORDER BY id")
    assert r.status == "ok"
    assert r.data["rowcount"] == 3
    assert r.data["records"][0] == {"id": 1, "email": "a@x"}


def test_query_join_across_files(connected):
    r = connected.query(
        "SELECT u.email, SUM(o.amount) AS total "
        "FROM users u JOIN orders o ON u.id = o.user_id "
        "GROUP BY u.email ORDER BY u.email"
    )
    assert r.status == "ok"
    assert r.data["rowcount"] == 2  # user 3 had no orders
    by_email = {row["email"]: row["total"] for row in r.data["records"]}
    assert abs(by_email["a@x"] - 9.99) < 1e-6
    assert abs(by_email["b@x"] - 23.50) < 1e-6


def test_query_blocks_destructive(connected):
    for bad in (
        "DROP TABLE users",
        "DELETE FROM users",
        "UPDATE users SET email='x'",
        "INSERT INTO users VALUES (9, 'x', 'Y')",
        "ATTACH DATABASE 'evil.db' AS e",
    ):
        r = connected.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError"), bad


def test_query_rejects_no_tables(connected):
    r = connected.query("SELECT 1")
    assert r.status == "error"
    assert "no tables" in r.error.lower()


def test_query_rejects_empty(connected):
    r = connected.query("   ")
    assert r.status == "error"
    assert "query is required" in r.error


def test_query_unknown_table(connected):
    r = connected.query("SELECT * FROM ghosts")
    assert r.status == "error"
    assert "Unknown table" in r.explanation


def test_query_surfaces_sqlite_syntax_error(connected):
    r = connected.query("SELECT frobnicate(*) FROM users")
    assert r.status == "error"
    assert r.error.startswith("OperationalError") or "sqlite" in r.error.lower()


def test_query_requires_connect(tmp_path):
    c = FlatFileConnector(path="users.csv", root=tmp_path)
    r = c.query("SELECT * FROM users")
    assert r.status == "error"
    assert "Not connected" in r.explanation


# ---- estimate_cost ------------------------------------------------------


def test_estimate_cost_sums_file_sizes(connected, data_dir):
    r = connected.estimate_cost("SELECT * FROM users JOIN orders ON users.id=orders.user_id")
    assert r.status == "ok"
    expected = (data_dir / "users.csv").stat().st_size + (data_dir / "orders.parquet").stat().st_size
    assert r.data["bytes_estimate"] == expected
    assert set(r.data["per_table_bytes"]) == {"users", "orders"}


def test_estimate_cost_unknown_table(connected):
    r = connected.estimate_cost("SELECT * FROM ghosts")
    assert r.status == "error"


# ---- detect_pii ---------------------------------------------------------


def test_detect_pii_flags_known_columns(connected):
    r = connected.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]


def test_detect_pii_on_logs(connected):
    r = connected.detect_pii("logs")
    assert r.status == "ok"
    assert "ip_address" in r.data["pii_columns"]["ip"]


# ---- lifecycle / registry ----------------------------------------------


def test_close_clears_state(connected):
    connected.close()
    r = connected.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["flatfile"] is tool
