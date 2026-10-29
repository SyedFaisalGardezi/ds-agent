"""REST connector.

Mapping to the `DataConnector` ABC
----------------------------------
* **"schema" ≡ endpoint** — `list_schemas()` returns the configured endpoint
  catalog (or inspects an OpenAPI spec if `openapi_url` is set).
* **"table" ≡ endpoint path** — `describe_table(path)` returns the endpoint
  definition (HTTP verbs, parameters, response schema) discovered from the
  OpenAPI spec if available, else just the declared config.
* **`query(spec)`** accepts a JSON spec: the path, optional query params,
  and (if allowed) HTTP method:

      {
        "path": "/users",
        "method": "GET",              # default GET; POST/PUT/DELETE blocked
        "params": {"limit": 10},      # query string
        "headers": {"X-Foo": "bar"},
        "json_body": {...}            # for POST/PATCH if method is allowed
      }

* **Read-only by default**: only GET / HEAD / OPTIONS are allowed unless the
  caller constructs with `allow_writes=True`. Even then, dangerous methods
  (DELETE, TRACE, CONNECT) remain blocked.
* **Auth** is supported via `auth_config`: `{"type": "bearer", "token": ...}`,
  `{"type": "basic", "username": ..., "password": ...}`, or
  `{"type": "api_key", "header": "X-API-Key", "key": ...}`.
* **Mock-friendly**: `http_client` is injectable — pass any object with a
  `.request(method, url, **kwargs)` returning a duck-typed response
  (`.status_code`, `.json()`, `.text`, `.headers`).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

from agent.connectors.base import DataConnector
from agent.core.types import ToolResult

_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_BLOCKED_METHODS = frozenset({"DELETE", "TRACE", "CONNECT"})

_SAFE_PATH = re.compile(r"^/?[A-Za-z0-9_\-./\{\}:~%]*$")

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b(e?_?mail|email_addr)\b", re.IGNORECASE),
    "phone": re.compile(r"\b(phone|mobile|telephone|msisdn)\b", re.IGNORECASE),
    "ssn": re.compile(r"\b(ssn|social_security|national_id|nin)\b", re.IGNORECASE),
    "name": re.compile(r"\b(first_?name|last_?name|full_?name|surname|given_?name)\b", re.IGNORECASE),
    "address": re.compile(r"\b(address|street|postal_?code|zip_?code|postcode)\b", re.IGNORECASE),
    "dob": re.compile(r"\b(dob|date_of_birth|birth_?date)\b", re.IGNORECASE),
    "ip": re.compile(r"\b(ip_?address|client_ip)\b", re.IGNORECASE),
    "cc": re.compile(r"\b(card_?number|credit_?card|cc_num|pan)\b", re.IGNORECASE),
    "geo": re.compile(r"\b(lat|latitude|lon|longitude|geo_point)\b", re.IGNORECASE),
}


def _extract_fields(obj: Any, max_depth: int = 2) -> set[str]:
    """Collect top-level + first-level nested field names from a JSON payload."""
    out: set[str] = set()

    def walk(x: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(x, Mapping):
            for k, v in x.items():
                out.add(str(k))
                walk(v, depth + 1)
        elif isinstance(x, list):
            for item in x[:5]:  # sample first few items
                walk(item, depth)

    walk(obj, 0)
    return out


class RESTConnector(DataConnector):
    """Read-only REST connector."""

    name = "rest"

    def __init__(
        self,
        base_url: str = "",
        auth_config: dict | None = None,
        endpoints: list[str] | None = None,
        openapi_url: str | None = None,
        http_client: Any = None,
        allow_writes: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or os.environ.get("REST_BASE_URL", "")
        self._auth_config = auth_config or {}
        self._endpoints = list(endpoints or [])
        self._openapi_url = openapi_url
        self._http_client = http_client
        self._allow_writes = bool(allow_writes)
        self._timeout = float(timeout)
        self._connected = False
        self._openapi_spec: dict | None = None

    def _resolve_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        import httpx  # pragma: no cover
        return httpx.Client(timeout=self._timeout)

    def _require_connected(self) -> ToolResult | None:
        if not self._connected:
            return ToolResult(
                status="error",
                data=None,
                explanation="Not connected. Call connect() first.",
                math_trace="",
                error="ConnectionError: no active connection",
            )
        return None

    def _auth_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"headers": {}}
        a = self._auth_config
        t = (a.get("type") or "").lower()
        if t == "bearer" and a.get("token"):
            kwargs["headers"]["Authorization"] = f"Bearer {a['token']}"
        elif t == "api_key" and a.get("key"):
            header = a.get("header", "X-API-Key")
            kwargs["headers"][header] = a["key"]
        elif t == "basic" and a.get("username") and a.get("password"):
            kwargs["auth"] = (a["username"], a["password"])
        return kwargs

    def _allowed_method(self, method: str) -> ToolResult | None:
        m = method.upper()
        if m in _BLOCKED_METHODS:
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Method {m!r} is never allowed (read-only connector).",
                math_trace="",
                error="SafetyError: method blocked",
            )
        if m not in _READ_METHODS and not self._allow_writes:
            return ToolResult(
                status="error",
                data=None,
                explanation=(
                    f"Method {m!r} requires allow_writes=True. "
                    f"Read-only methods: {sorted(_READ_METHODS)}."
                ),
                math_trace="",
                error="SafetyError: write method not enabled",
            )
        return None

    def connect(self) -> ToolResult:
        if not self._base_url:
            return ToolResult(
                status="error",
                data=None,
                explanation="Missing REST config: base_url is required.",
                math_trace="",
                error="ConfigError: missing base_url",
            )
        try:
            client = self._resolve_client()
            self._http_client = client
            if self._openapi_url:
                resp = client.request(
                    "GET",
                    urljoin(self._base_url, self._openapi_url),
                    **self._auth_kwargs(),
                )
                if getattr(resp, "status_code", 500) < 400:
                    try:
                        self._openapi_spec = resp.json()
                    except Exception:  # noqa: BLE001
                        self._openapi_spec = None
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Failed to open REST session: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )
        self._connected = True
        masked = {
            k: ("***" if k in ("token", "password", "key") else v)
            for k, v in self._auth_config.items()
        }
        return ToolResult(
            status="ok",
            data={
                "base_url": self._base_url,
                "auth": masked,
                "openapi_loaded": self._openapi_spec is not None,
                "endpoints_configured": len(self._endpoints),
            },
            explanation=f"Connected to REST API at {self._base_url}.",
            math_trace="",
        )

    def list_schemas(self) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        paths: list[str]
        if self._openapi_spec and isinstance(self._openapi_spec.get("paths"), Mapping):
            paths = sorted(self._openapi_spec["paths"].keys())
            source = "openapi"
        else:
            paths = sorted(self._endpoints)
            source = "config"
        return ToolResult(
            status="ok",
            data=paths,
            explanation=f"{len(paths)} endpoint(s) from {source}.",
            math_trace=f"Endpoint discovery source: {source}; count={len(paths)}.",
        )

    def describe_table(self, table: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        if not _SAFE_PATH.match(table):
            return ToolResult(
                status="error",
                data=None,
                explanation=f"Unsafe path: {table!r}.",
                math_trace="",
                error="ValueError: invalid path",
            )
        # Prefer OpenAPI-derived description.
        if self._openapi_spec:
            ops = (self._openapi_spec.get("paths") or {}).get(table)
            if ops:
                verbs = sorted(k.upper() for k in ops.keys() if isinstance(ops.get(k), Mapping))
                cols: list[dict[str, Any]] = []
                for verb in verbs:
                    meta = ops[verb.lower()]
                    for p in meta.get("parameters", []) or []:
                        cols.append({
                            "name": p.get("name"),
                            "type": (p.get("schema") or {}).get("type", "string"),
                            "in": p.get("in"),
                            "verb": verb,
                        })
                return ToolResult(
                    status="ok",
                    data={
                        "table": table,
                        "methods": verbs,
                        "columns": cols,
                        "source": "openapi",
                    },
                    explanation=(
                        f"{table}: {len(verbs)} method(s) from OpenAPI; "
                        f"{len(cols)} parameter(s)."
                    ),
                    math_trace=f"OpenAPI paths[{table}] → {verbs}.",
                )
        # Fallback: probe with GET and infer fields from the response payload.
        probe = self.query(json.dumps({"path": table, "method": "GET", "params": {"limit": 1}}))
        if probe.status != "ok":
            return probe
        body = (probe.data or {}).get("body")
        fields = sorted(_extract_fields(body))
        cols = [{"name": f, "type": "unknown", "in": "body", "verb": "GET"} for f in fields]
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "methods": ["GET"],
                "columns": cols,
                "source": "probe",
            },
            explanation=(
                f"{table}: probed with GET; inferred {len(cols)} top-level field(s)."
            ),
            math_trace=f"GET {table} → {len(cols)} fields (OpenAPI absent).",
        )

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        if params:
            spec = {**spec, **params}

        path = spec.get("path")
        if not path or not isinstance(path, str) or not _SAFE_PATH.match(path):
            return ToolResult(
                status="error",
                data=None,
                explanation="Spec missing or invalid 'path'.",
                math_trace="",
                error="ValueError: path",
            )
        method = str(spec.get("method", "GET")).upper()
        if (block := self._allowed_method(method)) is not None:
            return block

        url = urljoin(self._base_url.rstrip("/") + "/", path.lstrip("/"))
        auth_kw = self._auth_kwargs()
        headers = {**auth_kw.get("headers", {}), **(spec.get("headers") or {})}
        kwargs: dict[str, Any] = {"headers": headers, "params": spec.get("params")}
        if "auth" in auth_kw:
            kwargs["auth"] = auth_kw["auth"]
        if spec.get("json_body") is not None and method in ("POST", "PUT", "PATCH"):
            kwargs["json"] = spec["json_body"]

        try:
            resp = self._http_client.request(method, url, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                data=None,
                explanation=f"HTTP request failed: {exc}",
                math_trace="",
                error=f"{type(exc).__name__}: {exc}",
            )

        status_code = getattr(resp, "status_code", 0)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = getattr(resp, "text", "")
        resp_headers = dict(getattr(resp, "headers", {}) or {})

        if status_code >= 400:
            return ToolResult(
                status="error",
                data={"status_code": status_code, "body": body},
                explanation=f"{method} {path} returned HTTP {status_code}.",
                math_trace=f"{method} {url} → {status_code}.",
                error=f"HTTPError: {status_code}",
            )

        if isinstance(body, list):
            rowcount = len(body)
        elif isinstance(body, dict):
            # Heuristic: if there's a common list key, count it.
            for k in ("items", "results", "data"):
                if isinstance(body.get(k), list):
                    rowcount = len(body[k])
                    break
            else:
                rowcount = 1
        else:
            rowcount = 1

        return ToolResult(
            status="ok",
            data={
                "status_code": status_code,
                "headers": resp_headers,
                "body": body,
                "rowcount": rowcount,
            },
            explanation=f"{method} {path} → HTTP {status_code}; {rowcount} record(s).",
            math_trace=f"{method} {url} → {status_code}; rowcount={rowcount}.",
        )

    def estimate_cost(self, query: str) -> ToolResult:
        if (err := self._require_connected()) is not None:
            return err
        spec = _parse_spec(query)
        if isinstance(spec, ToolResult):
            return spec
        method = str(spec.get("method", "GET")).upper()
        if (block := self._allowed_method(method)) is not None:
            return block
        path = spec.get("path", "")
        if not _SAFE_PATH.match(str(path or "")):
            return ToolResult(
                status="error",
                data=None,
                explanation="Invalid 'path'.",
                math_trace="",
                error="ValueError: path",
            )
        # Best-effort estimate: one HEAD call → Content-Length if present.
        url = urljoin(self._base_url.rstrip("/") + "/", str(path).lstrip("/"))
        auth_kw = self._auth_kwargs()
        headers = {**auth_kw.get("headers", {}), **(spec.get("headers") or {})}
        bytes_estimate: int | None = None
        try:
            resp = self._http_client.request(
                "HEAD",
                url,
                headers=headers,
                params=spec.get("params"),
                **({"auth": auth_kw["auth"]} if "auth" in auth_kw else {}),
            )
            cl = (getattr(resp, "headers", {}) or {}).get("Content-Length") or \
                 (getattr(resp, "headers", {}) or {}).get("content-length")
            if cl is not None:
                bytes_estimate = int(cl)
        except Exception:  # noqa: BLE001
            bytes_estimate = None
        return ToolResult(
            status="ok",
            data={
                "method": method,
                "path": path,
                "bytes_estimate": bytes_estimate,
                "requests_estimate": 1,
            },
            explanation=(
                f"HEAD {path}: bytes≈{bytes_estimate}; one request will be issued."
            ),
            math_trace=(
                f"Cost ≈ 1 HTTP request. Bytes from Content-Length = {bytes_estimate}."
            ),
        )

    def detect_pii(self, table: str) -> ToolResult:
        desc = self.describe_table(table)
        if desc.status != "ok":
            return desc
        flagged: dict[str, list[str]] = {}
        for col in desc.data.get("columns", []):
            colname = str(col.get("name") or "")
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(colname):
                    flagged.setdefault(pii_type, []).append(colname)
        return ToolResult(
            status="ok",
            data={
                "table": table,
                "pii_columns": flagged,
                "columns_scanned": len(desc.data.get("columns", [])),
            },
            explanation=(
                f"Scanned {len(desc.data.get('columns', []))} field(s) on {table}; "
                f"flagged {sum(len(v) for v in flagged.values())} as PII across "
                f"{len(flagged)} categor{'y' if len(flagged) == 1 else 'ies'}."
            ),
            math_trace=f"Heuristic regex over {len(_PII_PATTERNS)} PII categories.",
        )

    def close(self) -> None:
        if self._http_client is not None:
            try:
                close = getattr(self._http_client, "close", None)
                if callable(close):
                    close()
            finally:
                self._http_client = None
                self._connected = False
                self._openapi_spec = None


def _parse_spec(query: Any) -> dict | ToolResult:
    if isinstance(query, Mapping):
        return dict(query)
    if not isinstance(query, str):
        return ToolResult(
            status="error",
            data=None,
            explanation=f"Query must be a JSON spec (str or dict), got {type(query).__name__}.",
            math_trace="",
            error="TypeError: query",
        )
    s = query.strip()
    if not s:
        return ToolResult(
            status="error",
            data=None,
            explanation="Empty query.",
            math_trace="",
            error="ValueError: query is required",
        )
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as exc:
        return ToolResult(
            status="error",
            data=None,
            explanation=f"Query is not valid JSON: {exc}",
            math_trace="",
            error=f"JSONDecodeError: {exc}",
        )
    if not isinstance(parsed, Mapping):
        return ToolResult(
            status="error",
            data=None,
            explanation="Query JSON must be an object/dict.",
            math_trace="",
            error="TypeError: query",
        )
    return dict(parsed)


def register() -> RESTConnector:
    from agent.connectors.registry import ConnectorRegistry
    conn = RESTConnector()
    ConnectorRegistry.get().register(conn)
    return conn
