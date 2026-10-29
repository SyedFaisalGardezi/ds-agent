"""Email verification-code authentication for the multi-user app.

This is a starting point for a small trusted team, not a full production auth
system. It seeds a fixed set of users, signs a stateless session token, and
sends a short-lived one-time code to the user's email.

Delivery requires configured SMTP. Missing configuration or delivery failure
returns an error; verification codes are never returned to callers or logged.

The gate is opt-in. It only requires login when DSAGENT_AUTH_ENABLED is set to
1, so the existing test suite and local single-user use are unaffected.

Environment variables:
  DSAGENT_AUTH_ENABLED   1 to require login (default 0)
  DSAGENT_AUTH_SECRET    HMAC secret for signing tokens (set a real one)
  DSAGENT_TOKEN_TTL_S    session token lifetime in seconds (default 86400)
  DSAGENT_CODE_TTL_S     verification code lifetime in seconds (default 600)
  DSAGENT_SMTP_HOST/PORT/USER/PASSWORD/FROM   SMTP settings for real email
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

log = logging.getLogger("dsagent.auth")

_USERS_FILE = Path(__file__).parent.parent.parent / "configs" / "users.yaml"
_TOKEN_TTL = int(os.environ.get("DSAGENT_TOKEN_TTL_S", "86400"))   # 24 hours
_CODE_TTL = int(os.environ.get("DSAGENT_CODE_TTL_S", "600"))       # 10 minutes
_CODE_MAX_ATTEMPTS = 5

# email -> {"code", "expires", "attempts"}. In-memory is fine for one process
# and a small team; move to a shared store if you run multiple workers.
_codes: dict[str, dict[str, Any]] = {}
_requests: dict[str, float] = {}


def _secret() -> bytes:
    value = os.environ.get("DSAGENT_AUTH_SECRET", "")
    if len(value) < 32:
        raise RuntimeError("DSAGENT_AUTH_SECRET must contain at least 32 characters")
    return value.encode()


def auth_enabled() -> bool:
    """Login is required only when DSAGENT_AUTH_ENABLED is 1. Read at call time
    so tests can toggle it and the running app can be configured at start."""
    return os.environ.get("DSAGENT_AUTH_ENABLED", "0") == "1"


def _load_users() -> dict[str, dict[str, Any]]:
    try:
        data = yaml.safe_load(_USERS_FILE.read_text(encoding="utf-8")) or {}
        return {u["email"].strip().lower(): u for u in data.get("users", [])
                if u.get("email")}
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load users from %s: %s", _USERS_FILE, exc)
        return {}


USERS = _load_users()


def get_user(email: str) -> dict[str, Any] | None:
    return USERS.get((email or "").strip().lower())


def _public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {"email": user["email"], "name": user.get("name", user["email"]),
            "role": user.get("role", "member")}


# ── verification codes ────────────────────────────────────────────────────────

def generate_code(email: str) -> str | None:
    """Create and store a code for a known user. None for an unknown email."""
    email = (email or "").strip().lower()
    if email not in USERS:
        return None
    code = f"{secrets.randbelow(1_000_000):06d}"
    _codes[email] = {"code": code, "expires": time.time() + _CODE_TTL,
                     "attempts": 0}
    return code


def verify_code(email: str, code: str) -> bool:
    """One-time check of a code: correct, unexpired, and under the attempt cap."""
    email = (email or "").strip().lower()
    rec = _codes.get(email)
    if not rec:
        return False
    if time.time() > rec["expires"]:
        _codes.pop(email, None)
        return False
    rec["attempts"] += 1
    if rec["attempts"] > _CODE_MAX_ATTEMPTS:
        _codes.pop(email, None)
        return False
    if not hmac.compare_digest(rec["code"], str(code or "").strip()):
        return False
    _codes.pop(email, None)   # single use
    return True


# ── session tokens (stateless, HMAC-signed) ──────────────────────────────────

def _b64(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def _unb64(raw: bytes) -> bytes:
    return base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))


def issue_token(email: str) -> str:
    payload = {"sub": email.strip().lower(), "exp": int(time.time()) + _TOKEN_TTL}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_secret(), body, hashlib.sha256).digest())
    return (body + b"." + sig).decode()


def verify_token(token: str | None) -> str | None:
    """Return the email if the token is valid and unexpired, else None."""
    if not token or "." not in token:
        return None
    try:
        body_b64, sig_b64 = token.encode().split(b".", 1)
        expected = _b64(hmac.new(_secret(), body_b64, hashlib.sha256).digest())
        if not hmac.compare_digest(sig_b64, expected):
            return None
        payload = json.loads(_unb64(body_b64))
        if int(payload.get("exp", 0)) < time.time():
            return None
        email = payload.get("sub")
        return email if email in USERS else None
    except Exception:  # noqa: BLE001
        return None


# ── email delivery ────────────────────────────────────────────────────────────

def smtp_configured() -> bool:
    return bool(os.environ.get("DSAGENT_SMTP_HOST"))


def _smtp_send(to_addr: str, subject: str, text: str) -> None:
    host = os.environ["DSAGENT_SMTP_HOST"]
    port = int(os.environ.get("DSAGENT_SMTP_PORT", "587"))
    user = os.environ.get("DSAGENT_SMTP_USER")
    password = os.environ.get("DSAGENT_SMTP_PASSWORD")
    sender = os.environ.get("DSAGENT_SMTP_FROM", user or "no-reply@ds-agent.local")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content(text)
    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)


def send_code(email: str, code: str) -> dict[str, Any]:
    """Deliver the code. Returns the channel used. In dev mode (no SMTP) the
    code is logged and returned so the caller can display it; with SMTP the code
    is sent by email and never returned."""
    subject = "Your DS Agent verification code"
    text = (f"Your DS Agent verification code is {code}.\n"
            f"It expires in {_CODE_TTL // 60} minutes.")
    if not smtp_configured():
        raise HTTPException(503, "Email delivery is not configured")
    try:
        _smtp_send(email, subject, text)
    except Exception as exc:
        log.error("Verification delivery failed: %s", type(exc).__name__)
        raise HTTPException(503, "Verification email could not be delivered") from exc
    return {"channel": "email"}


# ── FastAPI wiring ────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _token_from(request: Request,
                cred: HTTPAuthorizationCredentials | None) -> str | None:
    if cred and cred.credentials:
        return cred.credentials
    return request.cookies.get("ds_token")


def require_user(request: Request,
                 cred: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
                 ) -> dict[str, Any]:
    """Dependency: resolve the current user from the token, or 401. When auth is
    disabled this still requires a token, so use it only on endpoints that are
    meant to be user-aware."""
    email = verify_token(_token_from(request, cred))
    user = get_user(email) if email else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    request.state.user_email = user["email"]
    return user


def current_user_email(request: Request) -> str | None:
    """Best-effort current user for gated requests; set by the middleware."""
    return getattr(request.state, "user_email", None)


class RequestCodeBody(BaseModel):
    email: str


class VerifyCodeBody(BaseModel):
    email: str
    code: str


def make_auth_router() -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.get("/status")
    def status(request: Request,
               cred: HTTPAuthorizationCredentials | None = Depends(_bearer)):  # noqa: B008
        email = verify_token(_token_from(request, cred))
        user = get_user(email) if email else None
        return {"auth_required": auth_enabled(),
                "user": _public_user(user) if user else None}

    @router.post("/request-code")
    def request_code(body: RequestCodeBody):
        email = body.email.strip().lower()
        now = time.monotonic()
        if now - _requests.get(email, -60) < 60:
            raise HTTPException(429, "Wait before requesting another verification code")
        if len(_requests) > 10000:
            _requests.clear()
        _requests[email] = now
        code = generate_code(email)
        # Do not reveal whether an address is registered: always report sent.
        if code is None:
            return {"sent": True}
        result = send_code(body.email.strip().lower(), code)
        out: dict[str, Any] = {"sent": True, "channel": result["channel"]}
        if result.get("dev_code"):
            out["dev_code"] = result["dev_code"]
        return out

    @router.post("/verify-code")
    def verify(body: VerifyCodeBody):
        if not verify_code(body.email, body.code):
            raise HTTPException(status_code=401,
                                detail="Invalid or expired code")
        user = get_user(body.email)
        if user is None:
            raise HTTPException(401, "Unknown user")
        return {"token": issue_token(user["email"]), "user": _public_user(user)}

    @router.get("/me")
    def me(user: dict[str, Any] = Depends(require_user)):  # noqa: B008
        return _public_user(user)

    return router
