"""Unit tests for MongoDBConnector.

Uses a duck-typed MongoClient-alike — no pymongo or mongomock dependency.
"""
from __future__ import annotations

import json

from agent.connectors.mongodb_conn import MongoDBConnector, register

# ---- fake Mongo stack --------------------------------------------------


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._sort = None
        self._skip = 0
        self._limit = None

    def sort(self, key_dir):
        self._sort = key_dir
        return self

    def skip(self, n):
        self._skip = n
        return self

    def limit(self, n):
        self._limit = n
        return self

    def __iter__(self):
        docs = list(self._docs)
        if self._sort:
            for k, direction in reversed(self._sort):
                docs.sort(key=lambda d: d.get(k), reverse=(direction == -1))
        docs = docs[self._skip:]
        if self._limit:
            docs = docs[: self._limit]
        return iter(docs)


class FakeCollection:
    def __init__(self, docs):
        self.docs = list(docs)
        self.last_find = None
        self.last_aggregate = None

    def find(self, filter_=None, projection=None):
        self.last_find = (filter_, projection)
        return FakeCursor(self.docs)

    def aggregate(self, pipeline):
        self.last_aggregate = pipeline
        # $sample: just return first N
        if pipeline and "$sample" in pipeline[0]:
            size = pipeline[0]["$sample"]["size"]
            return iter(self.docs[:size])
        return iter(self.docs)


class FakeDB:
    def __init__(self, collections=None, command_result=None):
        self._collections = collections or {}
        self._command_result = command_result or {}
        self.last_command = None

    def __getitem__(self, name):
        return self._collections.setdefault(name, FakeCollection([]))

    def list_collection_names(self):
        return list(self._collections.keys())

    def command(self, cmd):
        self.last_command = cmd
        return self._command_result


class FakeClient:
    def __init__(self, db):
        self._db = db
        self.closed = False

    def __getitem__(self, name):
        return self._db

    def close(self):
        self.closed = True


def _make(db, **config):
    cfg = {"database": "mydb", "host": "h", **config}
    return MongoDBConnector(client_factory=lambda *a, **kw: FakeClient(db), **cfg)


# ---- connect -----------------------------------------------------------


def test_connect_requires_database(monkeypatch):
    for k in ("URI", "HOST", "PORT", "USERNAME", "PASSWORD", "AUTHSOURCE", "DATABASE"):
        monkeypatch.delenv(f"MONGO_{k}", raising=False)
    c = MongoDBConnector(client_factory=lambda **kw: FakeClient(FakeDB()))
    r = c.connect()
    assert r.status == "error"
    assert "database" in r.explanation.lower()


def test_connect_with_uri():
    captured = {}

    def factory(*args, **kw):
        captured["args"] = args
        captured["kwargs"] = kw
        return FakeClient(FakeDB())

    c = MongoDBConnector(client_factory=factory, uri="mongodb://u:p@h/mydb", database="mydb")
    r = c.connect()
    assert r.status == "ok"
    assert captured["args"] == ("mongodb://u:p@h/mydb",)
    assert r.data["config"]["uri"] == "***"
    assert r.data["database"] == "mydb"


def test_connect_with_kwargs():
    captured = {}

    def factory(**kw):
        captured.update(kw)
        return FakeClient(FakeDB())

    c = MongoDBConnector(
        client_factory=factory,
        host="h",
        port=27017,
        username="u",
        password="p",
        database="mydb",
    )
    r = c.connect()
    assert r.status == "ok"
    assert captured["host"] == "h"
    assert "database" not in captured  # database is not a driver kwarg
    assert r.data["config"]["password"] == "***"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("MONGO_HOST", "envhost")
    monkeypatch.setenv("MONGO_DATABASE", "envdb")
    captured = {}

    def factory(**kw):
        captured.update(kw)
        return FakeClient(FakeDB())

    c = MongoDBConnector(client_factory=factory)
    r = c.connect()
    assert r.status == "ok"
    assert captured["host"] == "envhost"
    assert r.data["database"] == "envdb"


def test_connect_surfaces_driver_error():
    def boom(**kw):
        raise RuntimeError("refused")

    c = MongoDBConnector(client_factory=boom, host="h", database="d")
    r = c.connect()
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- require_connected -------------------------------------------------


def test_operations_require_connect():
    c = MongoDBConnector(client_factory=lambda **kw: FakeClient(FakeDB()), database="d")
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "t"),
        ("query", '{"collection":"t"}'),
        ("estimate_cost", '{"collection":"t"}'),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"
        assert "Not connected" in r.explanation


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_returns_collections():
    db = FakeDB({"users": FakeCollection([]), "orders": FakeCollection([])})
    c = _make(db)
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert set(r.data) == {"users", "orders"}


# ---- describe_table ----------------------------------------------------


def test_describe_table_infers_types_from_sample():
    docs = [
        {"_id": 1, "email": "a@x", "age": 30},
        {"_id": 2, "email": "b@x", "age": 25, "note": "hi"},
    ]
    db = FakeDB({"users": FakeCollection(docs)})
    c = _make(db)
    c.connect()
    r = c.describe_table("users", sample_size=10)
    assert r.status == "ok"
    names = {col["name"] for col in r.data["columns"]}
    assert names == {"_id", "email", "age", "note"}
    assert r.data["sampled"] == 2


def test_describe_table_rejects_unsafe_identifier():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    r = c.describe_table("users; drop")
    assert r.status == "error"
    assert "invalid identifier" in r.error


# ---- query -------------------------------------------------------------


def test_query_find_returns_records():
    docs = [{"_id": 1, "name": "a"}, {"_id": 2, "name": "b"}]
    coll = FakeCollection(docs)
    db = FakeDB({"users": coll})
    c = _make(db)
    c.connect()
    r = c.query('{"collection": "users", "filter": {"active": true}, "limit": 5}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert coll.last_find[0] == {"active": True}


def test_query_accepts_dict_spec():
    coll = FakeCollection([{"_id": 1}])
    db = FakeDB({"users": coll})
    c = _make(db)
    c.connect()
    r = c.query({"collection": "users"})
    assert r.status == "ok"


def test_query_aggregate_runs_pipeline():
    docs = [{"_id": 1}, {"_id": 2}, {"_id": 3}]
    coll = FakeCollection(docs)
    db = FakeDB({"users": coll})
    c = _make(db)
    c.connect()
    r = c.query(json.dumps({
        "collection": "users",
        "pipeline": [{"$match": {"_id": {"$gte": 1}}}],
    }))
    assert r.status == "ok"
    assert r.data["rowcount"] == 3
    assert coll.last_aggregate == [{"$match": {"_id": {"$gte": 1}}}]


def test_query_blocks_out_stage():
    db = FakeDB({"users": FakeCollection([])})
    c = _make(db)
    c.connect()
    r = c.query(json.dumps({"collection": "users", "pipeline": [{"$out": "archive"}]}))
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_query_blocks_merge_stage():
    db = FakeDB({"users": FakeCollection([])})
    c = _make(db)
    c.connect()
    r = c.query(json.dumps({
        "collection": "users",
        "pipeline": [{"$match": {}}, {"$merge": {"into": "other"}}],
    }))
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_query_rejects_empty():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    r = c.query("   ")
    assert r.status == "error"
    assert "query is required" in r.error


def test_query_rejects_invalid_json():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    r = c.query("{not json")
    assert r.status == "error"
    assert "JSONDecodeError" in r.error


def test_query_rejects_missing_collection():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    r = c.query("{}")
    assert r.status == "error"
    assert "collection" in r.explanation.lower()


def test_query_rejects_bad_pipeline_type():
    db = FakeDB({"users": FakeCollection([])})
    c = _make(db)
    c.connect()
    r = c.query(json.dumps({"collection": "users", "pipeline": "not-a-list"}))
    assert r.status == "error"
    assert r.error.startswith("TypeError")


def test_query_coerces_objectid_like_to_str():
    class Weird:
        def __str__(self):
            return "oid:abc"

    coll = FakeCollection([{"_id": Weird(), "n": 1}])
    db = FakeDB({"users": coll})
    c = _make(db)
    c.connect()
    r = c.query('{"collection": "users"}')
    assert r.status == "ok"
    assert r.data["records"][0]["_id"] == "oid:abc"


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_find_uses_explain():
    explain = {
        "executionStats": {
            "totalDocsExamined": 1000,
            "totalKeysExamined": 50,
            "nReturned": 10,
        },
        "queryPlanner": {"winningPlan": {"stage": "IXSCAN"}},
    }
    db = FakeDB({"users": FakeCollection([])}, command_result=explain)
    c = _make(db)
    c.connect()
    r = c.estimate_cost('{"collection": "users", "filter": {"x": 1}}')
    assert r.status == "ok"
    assert r.data["docs_examined"] == 1000
    assert r.data["keys_examined"] == 50
    assert r.data["rows_estimate"] == 10
    assert r.data["plan_stage"] == "IXSCAN"
    assert "find" in db.last_command["explain"]


def test_estimate_cost_aggregate_uses_explain():
    explain = {"stages": [{"$cursor": {"executionStats": {"totalDocsExamined": 5}}}]}
    db = FakeDB({"users": FakeCollection([])}, command_result=explain)
    c = _make(db)
    c.connect()
    r = c.estimate_cost(json.dumps({"collection": "users", "pipeline": [{"$match": {}}]}))
    assert r.status == "ok"
    assert r.data["docs_examined"] == 5
    assert "aggregate" in db.last_command["explain"]


def test_estimate_cost_rejects_bad_collection():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    r = c.estimate_cost('{"collection": "users; drop"}')
    assert r.status == "error"


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_flags_fields():
    docs = [
        {"_id": 1, "email": "a@x", "first_name": "A", "ip_address": "1.1.1.1", "age": 30},
    ]
    db = FakeDB({"users": FakeCollection(docs)})
    c = _make(db)
    c.connect()
    r = c.detect_pii("users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]
    assert "ip_address" in flagged["ip"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears_connection():
    db = FakeDB({})
    c = _make(db)
    c.connect()
    client = c._client
    c.close()
    assert client.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["mongodb"] is tool
