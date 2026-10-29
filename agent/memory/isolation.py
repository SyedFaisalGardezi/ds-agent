"""Memory isolation mode detector and manager.

Detects natural-language intent to enable/disable cross-session memory recall.
Pure regex — no LLM calls, no heavy imports.
"""
from __future__ import annotations

import re

# Phrases that enable isolation (ignore cross-session memory)
_ENABLE_PATTERNS = [
    r"\bignore\b.{0,20}\b(memory|context|history|chats?|sessions?)\b",
    r"\b(fresh|clean)\s+start\b",
    r"\bno\b.{0,15}\b(previous|past|other)\b.{0,15}\b(context|memory|sessions?|chats?)\b",
    r"\bforget\b.{0,20}\b(other|past|previous)\b.{0,20}\b(sessions?|chats?)\b",
    r"\bisolation\s+mode\b",
    r"\bdon.?t\b.{0,20}\b(use|recall|remember)\b.{0,20}\b(past|other|previous)\b",
    r"\bstart\s+fresh\b",
    r"\bclean\s+slate\b",
]

# Phrases that disable isolation (restore cross-session memory)
_DISABLE_PATTERNS = [
    r"\buse\b.{0,20}\b(memory|context|history)\b.{0,20}\b(again|back|other|all)\b",
    r"\brestore\b.{0,20}\b(memory|context)\b",
    r"\bremember\b.{0,20}\b(other|past|previous)\b.{0,20}\b(sessions?|chats?)\b",
    r"\bdisable\b.{0,15}\bisolation\b",
    r"\bturn\s+off\b.{0,20}\bisolation\b",
]

_ENABLE_RE  = [re.compile(p, re.IGNORECASE) for p in _ENABLE_PATTERNS]
_DISABLE_RE = [re.compile(p, re.IGNORECASE) for p in _DISABLE_PATTERNS]


def detect_isolation_intent(text: str) -> str | None:
    """Return 'enable', 'disable', or None."""
    for pattern in _ENABLE_RE:
        if pattern.search(text):
            return "enable"
    for pattern in _DISABLE_RE:
        if pattern.search(text):
            return "disable"
    return None


def apply_isolation_intent(text: str, session: object, db_store: object | None) -> str | None:
    """Detect isolation intent, mutate session, persist to DB if available.

    Returns a confirmation message if isolation intent was found, else None.
    `session` must have an `isolation_mode` attribute.
    `db_store` must have `set_isolation_mode(session_id, bool)` — optional.
    """
    intent = detect_isolation_intent(text)

    if intent == "enable":
        session.isolation_mode = True  # type: ignore[attr-defined]
        if db_store is not None:
            try:
                db_store.set_isolation_mode(session.session_id, True)  # type: ignore[attr-defined]
            except Exception:
                pass
        return (
            "Memory isolation enabled. Only context from this session will be used — "
            "no information from other sessions will be recalled."
        )

    if intent == "disable":
        session.isolation_mode = False  # type: ignore[attr-defined]
        if db_store is not None:
            try:
                db_store.set_isolation_mode(session.session_id, False)  # type: ignore[attr-defined]
            except Exception:
                pass
        return "Memory isolation disabled. Cross-session recall restored."

    return None
