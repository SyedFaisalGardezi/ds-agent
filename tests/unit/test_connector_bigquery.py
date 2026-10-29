"""Unit tests for BigQueryConnector — duck-typed fakes, no google-cloud-bigquery."""
from __future__ import annotations

from agent.connectors.bigquery_conn import BigQueryConnector, register

# ---- fake BQ stack -----------------------------------------------------


class FakeRow(dict):
    """Dict subclass so `.keys()` iteration yields columns (as BQ Row does)."""


class FakeJob:
    def __init__(self, records=None, bytes_processed=0, bytes_billed=0):
        self._records = records or []
        self.total_bytes_processed = bytes_processed
        self.total_bytes_billed = bytes_billed

    def result(self):
        return list(self._records)


class FakeField:
    def __init__(self, name, field_type, mode="NULLABLE"):
        self.name = name
        self.field_type = field_type
        self.mode = mode


class FakeTable:
    def __init__(self, schema, num_rows=0, num_bytes=0):
        self.schema = schema
        self.num_rows = num_rows
        self.num_bytes = num_bytes


class FakeDataset:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id


class FakeClient:
    def __init__(self, datasets=None, tables=None, query_job=None):
        self._datasets = datasets or []
        self._tables = tables or {}
        self._query_job = query_job
        self.closed = False
        self.last_query_sql = None
        self.last_query_job_config = None
        self.last_get_table = None

    def list_datasets(self, project=None):
        return list(self._datasets)

    def get_table(self, fqn):
        self.last_get_table = fqn
        if fqn not in self._tables:
            raise ValueError(f"table not found: {fqn}")
        return self._tables[fqn]

    def query(self, sql, job_config=None):
        self.last_query_sql = sql
        self.last_query_job_config = job_config
        return self._query_job or FakeJob()

    def close(self):
        self.closed = True


def _make(client=None, **config):
    cfg = {"project": "my-proj", **config}
    client = client or FakeClient()
    return (
        BigQueryConnector(client_factory=lambda **kw: client, **cfg),
        client,
    )


# ---- connect -----------------------------------------------------------


def test_connect_requires_project(monkeypatch):
    for k in ("BIGQUERY_PROJECT", "BIGQUERY_LOCATION", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    c = BigQueryConnector(client_factory=lambda **kw: FakeClient())
    r = c.connect()
    assert r.status == "error"
    assert "project" in r.explanation.lower()


def test_connect_reads_bigquery_env(monkeypatch):
    monkeypatch.setenv("BIGQUERY_PROJECT", "env-proj")
    monkeypatch.setenv("BIGQUERY_LOCATION", "US")
    captured = {}

    def factory(**kw):
        captured.update(kw)
        return FakeClient()

    c = BigQueryConnector(client_factory=factory)
    r = c.connect()
    assert r.status == "ok"
    assert captured["project"] == "env-proj"
    assert captured["location"] == "US"


def test_connect_reads_google_cloud_project_env(monkeypatch):
    monkeypatch.delenv("BIGQUERY_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "adc-proj")
    c = BigQueryConnector(client_factory=lambda **kw: FakeClient())
    r = c.connect()
    assert r.status == "ok"
    assert r.data["project"] == "adc-proj"


def test_connect_masks_credentials():
    c, _ = _make(credentials_path="/secret/key.json")
    r = c.connect()
    assert r.status == "ok"
    assert r.data["config"]["credentials_path"] == "***"


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("auth fail")

    c = BigQueryConnector(client_factory=boom, project="p")
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- operations require connect ---------------------------------------


def test_operations_require_connect():
    c = BigQueryConnector(client_factory=lambda **kw: FakeClient(), project="p")
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "d.t"),
        ("query", "SELECT 1"),
        ("estimate_cost", "SELECT 1"),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"
        assert "Not connected" in r.explanation


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_returns_dataset_ids():
    datasets = [FakeDataset("analytics"), FakeDataset("raw")]
    c, _ = _make(client=FakeClient(datasets=datasets))
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["analytics", "raw"]


# ---- describe_table ----------------------------------------------------


def test_describe_table_two_part():
    schema = [FakeField("id", "INT64", "REQUIRED"), FakeField("email", "STRING", "NULLABLE")]
    table = FakeTable(schema, num_rows=1_000_000, num_bytes=64_000_000)
    client = FakeClient(tables={"my-proj.ds.users": table})
    c, _ = _make(client=client)
    c.connect()
    r = c.describe_table("ds.users")
    assert r.status == "ok"
    assert client.last_get_table == "my-proj.ds.users"
    assert r.data["num_rows"] == 1_000_000
    id_col = next(c for c in r.data["columns"] if c["name"] == "id")
    assert id_col["nullable"] is False
    email_col = next(c for c in r.data["columns"] if c["name"] == "email")
    assert email_col["nullable"] is True


def test_describe_table_three_part():
    # Project IDs with hyphens are legal in GCP and must be accepted.
    schema = [FakeField("id", "INT64")]
    client = FakeClient(tables={"other-proj.ds.t": FakeTable(schema)})
    c, _ = _make(client=client)
    c.connect()
    r = c.describe_table("other-proj.ds.t")
    assert r.status == "ok"
    assert client.last_get_table == "other-proj.ds.t"


def test_describe_table_rejects_hyphen_in_dataset():
    # Dataset and table names cannot contain hyphens — only project IDs can.
    c, _ = _make()
    c.connect()
    r = c.describe_table("bad-dataset.t")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_rejects_one_part():
    c, _ = _make()
    c.connect()
    r = c.describe_table("users")
    assert r.status == "error"
    assert "dataset.table" in r.explanation


def test_describe_table_rejects_four_part():
    c, _ = _make()
    c.connect()
    r = c.describe_table("a.b.c.d")
    assert r.status == "error"


def test_describe_table_rejects_unsafe():
    c, _ = _make()
    c.connect()
    r = c.describe_table("ds.users; drop")
    assert r.status == "error"
    assert "invalid identifier" in r.error


def test_describe_table_surfaces_not_found():
    client = FakeClient(tables={})
    c, _ = _make(client=client)
    c.connect()
    r = c.describe_table("ds.ghost")
    assert r.status == "error"
    assert "not found" in r.error.lower() or "ValueError" in r.error


# ---- query -------------------------------------------------------------


def test_query_happy_path():
    rows = [FakeRow({"id": 1, "name": "alice"}), FakeRow({"id": 2, "name": "bob"})]
    job = FakeJob(records=rows, bytes_processed=1024, bytes_billed=2048)
    client = FakeClient(query_job=job)
    c, _ = _make(client=client)
    c.connect()
    r = c.query("SELECT id, name FROM ds.users")
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["records"][0] == {"id": 1, "name": "alice"}
    assert r.data["total_bytes_processed"] == 1024
    assert r.data["total_bytes_billed"] == 2048


def test_query_blocks_destructive():
    c, _ = _make()
    c.connect()
    for bad in (
        "DELETE FROM ds.users",
        "DROP TABLE ds.users",
        "UPDATE ds.users SET x=1",
        "INSERT INTO ds.u VALUES (1)",
        "TRUNCATE TABLE ds.u",
        "MERGE INTO t USING s ON 1=1 WHEN MATCHED THEN UPDATE SET a=1",
        "EXPORT DATA OPTIONS(uri='gs://b/*') AS SELECT 1",
        "LOAD DATA INTO ds.t FROM FILES ('gs://b/*.csv')",
    ):
        r = c.query(bad)
        assert r.status == "error", bad
        assert r.error.startswith("SafetyError"), bad


def test_query_empty():
    c, _ = _make()
    c.connect()
    r = c.query("   ")
    assert r.status == "error"


def test_query_surfaces_error():
    class Boom(FakeClient):
        def query(self, sql, job_config=None):
            raise ValueError("syntax error")

    c, _ = _make(client=Boom())
    c.connect()
    r = c.query("SELECT frobnicate(*) FROM t")
    assert r.status == "error"
    assert "ValueError" in r.error


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_dry_run_applies_minimum_10mb():
    # 1 byte processed → billed = 10 MB minimum
    job = FakeJob(bytes_processed=1)
    client = FakeClient(query_job=job)
    c, _ = _make(client=client)
    c.connect()
    r = c.estimate_cost("SELECT 1")
    assert r.status == "ok"
    assert r.data["bytes_estimate"] == 1
    assert abs(r.data["billed_mb"] - 10.0) < 1e-9
    # 10 MB at $5/TB = 10 / 1_048_576 × 5 ≈ 4.768e-5
    assert r.data["usd_estimate"] > 0
    assert r.data["usd_estimate"] < 1e-3


def test_estimate_cost_dry_run_above_minimum():
    # 1 TB processed → ~ $5
    one_tb = 1024 * 1024 * 1024 * 1024
    job = FakeJob(bytes_processed=one_tb)
    client = FakeClient(query_job=job)
    c, _ = _make(client=client)
    c.connect()
    r = c.estimate_cost("SELECT * FROM huge")
    assert r.status == "ok"
    assert r.data["bytes_estimate"] == one_tb
    assert abs(r.data["usd_estimate"] - 5.0) < 1e-6


def test_estimate_cost_passes_dry_run_job_config():
    job = FakeJob(bytes_processed=0)
    client = FakeClient(query_job=job)
    c, _ = _make(client=client)
    c.connect()
    c.estimate_cost("SELECT 1")
    cfg = client.last_query_job_config
    # Fake path → dict sentinel
    assert isinstance(cfg, dict)
    assert cfg["dry_run"] is True
    assert cfg["use_query_cache"] is False


def test_estimate_cost_blocks_destructive():
    c, _ = _make()
    c.connect()
    r = c.estimate_cost("DELETE FROM ds.users")
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_estimate_cost_handles_none_bytes():
    # Job reports None → treated as 0, still applies minimum.
    job = FakeJob(bytes_processed=None)
    client = FakeClient(query_job=job)
    c, _ = _make(client=client)
    c.connect()
    r = c.estimate_cost("SELECT 1")
    assert r.status == "ok"
    assert r.data["bytes_estimate"] == 0
    assert abs(r.data["billed_mb"] - 10.0) < 1e-9


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_columns():
    schema = [
        FakeField("id", "INT64", "REQUIRED"),
        FakeField("email", "STRING"),
        FakeField("first_name", "STRING"),
        FakeField("ip_address", "STRING"),
    ]
    client = FakeClient(tables={"my-proj.ds.users": FakeTable(schema)})
    c, _ = _make(client=client)
    c.connect()
    r = c.detect_pii("ds.users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears():
    c, client = _make()
    c.connect()
    c.close()
    assert client.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["bigquery"] is tool
