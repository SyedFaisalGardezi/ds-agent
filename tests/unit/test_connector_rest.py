"""Unit tests for RESTConnector — duck-typed HTTP client, no httpx dep."""
from __future__ import annotations

import json

from agent.connectors.rest_conn import RESTConnector, register


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeClient:
    def __init__(self, responses=None, default=None):
        # `responses` keyed by (METHOD, path-suffix); else returns `default`
        self._responses = responses or {}
        self._default = default or FakeResponse(200, {"ok": True})
        self.requests: list[tuple[str, str, dict]] = []
        self.closed = False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        for (m, suffix), resp in self._responses.items():
            if method == m and url.endswith(suffix):
                return resp
        return self._default

    def close(self):
        self.closed = True


def _make(client=None, **config):
    client = client or FakeClient()
    return (
        RESTConnector(
            base_url=config.pop("base_url", "https://api.example.com"),
            http_client=client,
            **config,
        ),
        client,
    )


# ---- connect -----------------------------------------------------------


def test_connect_requires_base_url(monkeypatch):
    monkeypatch.delenv("REST_BASE_URL", raising=False)
    c = RESTConnector(http_client=FakeClient())
    r = c.connect()
    assert r.status == "error"
    assert "base_url" in r.explanation


def test_connect_masks_token():
    c, _ = _make(auth_config={"type": "bearer", "token": "s3cret"})
    r = c.connect()
    assert r.status == "ok"
    assert r.data["auth"]["token"] == "***"


def test_connect_reads_env(monkeypatch):
    monkeypatch.setenv("REST_BASE_URL", "https://env.example.com")
    c = RESTConnector(http_client=FakeClient())
    r = c.connect()
    assert r.status == "ok"
    assert r.data["base_url"] == "https://env.example.com"


def test_connect_loads_openapi():
    spec = {"paths": {"/users": {"get": {"parameters": []}}, "/orders": {"get": {}}}}
    client = FakeClient(responses={("GET", "/openapi.json"): FakeResponse(200, spec)})
    c = RESTConnector(
        base_url="https://api.example.com",
        openapi_url="/openapi.json",
        http_client=client,
    )
    r = c.connect()
    assert r.status == "ok"
    assert r.data["openapi_loaded"] is True


# ---- operations require connect ---------------------------------------


def test_operations_require_connect():
    c = RESTConnector(base_url="https://x", http_client=FakeClient())
    for name, arg in (
        ("list_schemas", None),
        ("describe_table", "/u"),
        ("query", '{"path":"/u"}'),
        ("estimate_cost", '{"path":"/u"}'),
    ):
        r = getattr(c, name)(arg) if arg is not None else getattr(c, name)()
        assert r.status == "error"


# ---- list_schemas ------------------------------------------------------


def test_list_schemas_from_openapi():
    spec = {"paths": {"/users": {}, "/orders": {}}}
    client = FakeClient(responses={("GET", "/openapi.json"): FakeResponse(200, spec)})
    c = RESTConnector(
        base_url="https://api.example.com",
        openapi_url="/openapi.json",
        http_client=client,
    )
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["/orders", "/users"]


def test_list_schemas_from_config():
    c, _ = _make(endpoints=["/a", "/b"])
    c.connect()
    r = c.list_schemas()
    assert r.status == "ok"
    assert r.data == ["/a", "/b"]


# ---- describe_table ----------------------------------------------------


def test_describe_table_from_openapi():
    spec = {"paths": {"/users": {"get": {"parameters": [
        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
        {"name": "email", "in": "query", "schema": {"type": "string"}},
    ]}}}}
    client = FakeClient(responses={("GET", "/openapi.json"): FakeResponse(200, spec)})
    c = RESTConnector(
        base_url="https://api.example.com",
        openapi_url="/openapi.json",
        http_client=client,
    )
    c.connect()
    r = c.describe_table("/users")
    assert r.status == "ok"
    assert r.data["methods"] == ["GET"]
    names = {col["name"] for col in r.data["columns"]}
    assert names == {"limit", "email"}
    assert r.data["source"] == "openapi"


def test_describe_table_probes_when_no_openapi():
    client = FakeClient(default=FakeResponse(200, [{"id": 1, "email": "a@x"}]))
    c, _ = _make(client=client)
    c.connect()
    r = c.describe_table("/users")
    assert r.status == "ok"
    assert r.data["source"] == "probe"
    names = {col["name"] for col in r.data["columns"]}
    assert {"id", "email"} <= names


def test_describe_table_rejects_unsafe_path():
    c, _ = _make()
    c.connect()
    r = c.describe_table("/users?q=1;drop table`")
    assert r.status == "error"
    assert "invalid path" in r.error


# ---- query -------------------------------------------------------------


def test_query_get_returns_list():
    client = FakeClient(default=FakeResponse(200, [{"id": 1}, {"id": 2}]))
    c, _ = _make(client=client)
    c.connect()
    r = c.query('{"path": "/users"}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 2
    assert r.data["status_code"] == 200


def test_query_counts_wrapped_list_in_dict():
    client = FakeClient(default=FakeResponse(200, {"items": [1, 2, 3], "page": 1}))
    c, _ = _make(client=client)
    c.connect()
    r = c.query('{"path": "/users"}')
    assert r.status == "ok"
    assert r.data["rowcount"] == 3


def test_query_accepts_dict_spec():
    client = FakeClient(default=FakeResponse(200, {"ok": True}))
    c, _ = _make(client=client)
    c.connect()
    r = c.query({"path": "/ping"})
    assert r.status == "ok"


def test_query_sends_auth_bearer_header():
    client = FakeClient(default=FakeResponse(200, {"ok": True}))
    c, _ = _make(client=client, auth_config={"type": "bearer", "token": "abc"})
    c.connect()
    c.query('{"path": "/u"}')
    _, _, kwargs = client.requests[-1]
    assert kwargs["headers"]["Authorization"] == "Bearer abc"


def test_query_sends_api_key_header():
    client = FakeClient(default=FakeResponse(200, {}))
    c, _ = _make(client=client, auth_config={"type": "api_key", "header": "X-Token", "key": "K"})
    c.connect()
    c.query('{"path": "/u"}')
    _, _, kwargs = client.requests[-1]
    assert kwargs["headers"]["X-Token"] == "K"


def test_query_basic_auth():
    client = FakeClient(default=FakeResponse(200, {}))
    c, _ = _make(client=client, auth_config={"type": "basic", "username": "u", "password": "p"})
    c.connect()
    c.query('{"path": "/u"}')
    _, _, kwargs = client.requests[-1]
    assert kwargs["auth"] == ("u", "p")


def test_query_blocks_write_method_by_default():
    c, _ = _make()
    c.connect()
    r = c.query('{"path": "/u", "method": "POST", "json_body": {"x": 1}}')
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_query_always_blocks_delete_and_trace():
    c, _ = _make(allow_writes=True)  # even with writes enabled
    c.connect()
    for m in ("DELETE", "TRACE", "CONNECT"):
        r = c.query(json.dumps({"path": "/u", "method": m}))
        assert r.status == "error", m
        assert r.error.startswith("SafetyError"), m


def test_query_allows_post_when_explicitly_enabled():
    client = FakeClient(default=FakeResponse(201, {"id": 1}))
    c, _ = _make(client=client, allow_writes=True)
    c.connect()
    r = c.query('{"path": "/u", "method": "POST", "json_body": {"x": 1}}')
    assert r.status == "ok"
    _, _, kwargs = client.requests[-1]
    assert kwargs["json"] == {"x": 1}


def test_query_reports_4xx_as_error():
    client = FakeClient(default=FakeResponse(404, {"detail": "not found"}))
    c, _ = _make(client=client)
    c.connect()
    r = c.query('{"path": "/missing"}')
    assert r.status == "error"
    assert "404" in r.error


def test_query_handles_non_json_response():
    client = FakeClient(default=FakeResponse(200, None, text="plain-text"))
    c, _ = _make(client=client)
    c.connect()
    r = c.query('{"path": "/u"}')
    assert r.status == "ok"
    assert r.data["body"] == "plain-text"


def test_query_rejects_empty():
    c, _ = _make()
    c.connect()
    r = c.query("   ")
    assert r.status == "error"


def test_query_rejects_missing_path():
    c, _ = _make()
    c.connect()
    r = c.query("{}")
    assert r.status == "error"
    assert "path" in r.explanation.lower()


def test_query_surfaces_transport_error():
    class Boom(FakeClient):
        def request(self, m, u, **kw):
            raise RuntimeError("dns fail")

    c, _ = _make(client=Boom())
    c.connect()
    r = c.query('{"path": "/u"}')
    assert r.status == "error"
    assert "RuntimeError" in r.error


# ---- estimate_cost -----------------------------------------------------


def test_estimate_cost_uses_head_content_length():
    client = FakeClient(default=FakeResponse(200, None, headers={"Content-Length": "2048"}))
    c, _ = _make(client=client)
    c.connect()
    r = c.estimate_cost('{"path": "/u"}')
    assert r.status == "ok"
    assert r.data["bytes_estimate"] == 2048
    assert r.data["requests_estimate"] == 1


def test_estimate_cost_blocks_bad_method():
    c, _ = _make()
    c.connect()
    r = c.estimate_cost('{"path": "/u", "method": "DELETE"}')
    assert r.status == "error"
    assert r.error.startswith("SafetyError")


def test_estimate_cost_handles_missing_header():
    client = FakeClient(default=FakeResponse(200, None, headers={}))
    c, _ = _make(client=client)
    c.connect()
    r = c.estimate_cost('{"path": "/u"}')
    assert r.status == "ok"
    assert r.data["bytes_estimate"] is None


# ---- detect_pii --------------------------------------------------------


def test_detect_pii_from_openapi_params():
    spec = {"paths": {"/users": {"get": {"parameters": [
        {"name": "email", "in": "query", "schema": {"type": "string"}},
        {"name": "first_name", "in": "query", "schema": {"type": "string"}},
        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
    ]}}}}
    client = FakeClient(responses={("GET", "/openapi.json"): FakeResponse(200, spec)})
    c = RESTConnector(
        base_url="https://api.example.com",
        openapi_url="/openapi.json",
        http_client=client,
    )
    c.connect()
    r = c.detect_pii("/users")
    assert r.status == "ok"
    flagged = r.data["pii_columns"]
    assert "email" in flagged["email"]
    assert "first_name" in flagged["name"]


# ---- lifecycle + registry ---------------------------------------------


def test_close_clears_client():
    c, client = _make()
    c.connect()
    c.close()
    assert client.closed is True
    r = c.list_schemas()
    assert r.status == "error"


def test_register_adds_to_registry():
    tool = register()
    from agent.connectors.registry import ConnectorRegistry
    assert ConnectorRegistry.get()["rest"] is tool
