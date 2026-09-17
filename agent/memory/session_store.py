"""Persistent SQLite-backed session and message store.

Replaces the in-memory SessionStore with a restart-proof equivalent.
All operations are thread-safe (single shared connection + lock + WAL mode).

Storage: ~/.ds-agent/memory/agent_memory.db
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".ds-agent" / "memory" / "agent_memory.db"


class PersistentSessionStore:
    """Thread-safe, restart-proof session and message store.

    Drop-in companion to the in-memory SessionStore — wrap it internally
    rather than replacing it so the in-memory cache stays fast.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; transactions managed manually
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._bootstrap()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    target          TEXT,
                    brief           TEXT,
                    brief_filenames TEXT,
                    data_paths      TEXT,
                    artifacts       TEXT,
                    notebook_path   TEXT,
                    isolation_mode  INTEGER DEFAULT 0,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
            """)

    # ── Session CRUD ─────────────────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (session_id, brief_filenames, data_paths, artifacts,
                    isolation_mode, created_at, updated_at)
                   VALUES (?, '[]', '[]', '{}', 0, ?, ?)""",
                (session_id, now, now),
            )

    def save_session(self, session: Any) -> None:
        """Persist mutable session fields.  Call after any state mutation."""
        with self._lock:
            self._conn.execute(
                """UPDATE sessions SET
                   target          = ?,
                   brief           = ?,
                   brief_filenames = ?,
                   data_paths      = ?,
                   artifacts       = ?,
                   notebook_path   = ?,
                   isolation_mode  = ?,
                   updated_at      = ?
                   WHERE session_id = ?""",
                (
                    session.target,
                    session.brief,
                    json.dumps(getattr(session, "brief_filenames", [])),
                    json.dumps([str(p) for p in getattr(session, "data_paths", [])]),
                    json.dumps(getattr(session, "artifacts", {})),
                    str(session.notebook_path) if session.notebook_path else None,
                    int(getattr(session, "isolation_mode", False)),
                    datetime.utcnow().isoformat(),
                    session.session_id,
                ),
            )

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def list_session_ids(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY updated_at DESC"
        )
        return [r[0] for r in cur.fetchall()]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )

    # ── Message CRUD ──────────────────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
                (session_id, role, content, datetime.utcnow().isoformat()),
            )

    def load_messages(self, session_id: str, n: int = 200) -> list[dict[str, str]]:
        """Return last *n* messages, oldest-first, as {role, content} dicts."""
        cur = self._conn.execute(
            """SELECT role, content FROM (
                   SELECT role, content, id
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (session_id, n),
        )
        return [{"role": r, "content": c} for r, c in cur.fetchall()]

    def message_count(self, session_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
        return cur.fetchone()[0]

    # ── Isolation mode ────────────────────────────────────────────────────────

    def set_isolation_mode(self, session_id: str, enabled: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET isolation_mode = ? WHERE session_id = ?",
                (int(enabled), session_id),
            )

    def get_isolation_mode(self, session_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT isolation_mode FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        return bool(row[0]) if row else False
