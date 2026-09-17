"""In-memory chat-session store for the ds-agent API.

A Session bundles everything a chat turn needs:
  - messages:       rolling conversation transcript
  - data_path:      primary CSV/parquet the agent will analyse
  - data_paths:     all uploaded data files (accumulates; data_path = first)
  - target:         the column the user (or PDF brief) named as target
  - brief:          free-text extracted from all uploaded briefs (concatenated)
  - brief_filename: first uploaded brief filename (backward compat)
  - brief_filenames:all uploaded brief filenames
  - run_id:         current pipeline run identifier
  - artifacts:      per-stage ToolResult dicts (kept JSON-friendly)
  - notebook:       the last-generated .ipynb path

Good enough for a single-process demo; swap for Redis if you need
multi-worker deployments.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class ChatMessage:
    role: str              # "user" | "assistant" | "system"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    messages: list[ChatMessage] = field(default_factory=list)
    # Primary data file (first uploaded); kept for backward compat.
    data_path: Path | None = None
    # All uploaded data files (data_path mirrors data_paths[0]).
    data_paths: list[Path] = field(default_factory=list)
    target: str | None = None
    # Concatenated text of all uploaded briefs.
    brief: str = ""
    # First brief filename (backward compat).
    brief_filename: str | None = None
    # All uploaded brief filenames.
    brief_filenames: list[str] = field(default_factory=list)
    run_id: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    notebook_path: Path | None = None
    work_dir: Path | None = None
    # Memory isolation — when True, cross-session LTM recall is suppressed.
    isolation_mode: bool = False

    def add_data_path(self, path: Path) -> None:
        """Append a data file, keeping data_path in sync with the first entry."""
        if path not in self.data_paths:
            self.data_paths.append(path)
        self.data_path = self.data_paths[0]

    def add_brief(self, text: str, filename: str) -> None:
        """Append brief text (with separator) and track filename."""
        if self.brief:
            self.brief += f"\n\n--- {filename} ---\n\n{text}"
        else:
            self.brief = text
        self.brief_filenames.append(filename)
        if self.brief_filename is None:
            self.brief_filename = filename

    def add(self, role: str, text: str) -> ChatMessage:
        msg = ChatMessage(role=role, text=text)
        self.messages.append(msg)
        return msg

    def to_public(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "data_path": str(self.data_path) if self.data_path else None,
            "data_paths": [str(p) for p in self.data_paths],
            "target": self.target,
            "brief_filename": self.brief_filename,
            "brief_filenames": self.brief_filenames,
            "brief_chars": len(self.brief),
            "run_id": self.run_id,
            "artifacts": sorted(self.artifacts.keys()),
            "notebook_ready": self.notebook_path is not None,
            "work_dir": str(self.work_dir) if self.work_dir else None,
            "message_count": len(self.messages),
        }


class SessionStore:
    """Thread-safe in-memory session store with optional SQLite persistence.

    On construction, tries to open PersistentSessionStore and restore any
    previously saved sessions.  If SQLite init fails (e.g., permissions),
    falls back silently to in-memory only — existing behaviour unchanged.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._db: Any = None  # PersistentSessionStore | None
        try:
            from agent.memory.session_store import PersistentSessionStore
            self._db = PersistentSessionStore()
            self._restore_from_db()
        except Exception:
            pass  # graceful fallback to pure in-memory

    def _restore_from_db(self) -> None:
        """Reload sessions from SQLite after a server restart."""
        if self._db is None:
            return
        for sid in self._db.list_session_ids():
            row = self._db.load_session(sid)
            if not row:
                continue
            s = Session(session_id=sid)
            s.target = row.get("target")
            s.brief = row.get("brief") or ""
            import json as _json
            s.brief_filenames = _json.loads(row.get("brief_filenames") or "[]")
            s.brief_filename = s.brief_filenames[0] if s.brief_filenames else None
            s.data_paths = [
                Path(p) for p in _json.loads(row.get("data_paths") or "[]")
            ]
            s.data_path = s.data_paths[0] if s.data_paths else None
            s.artifacts = _json.loads(row.get("artifacts") or "{}")
            s.isolation_mode = bool(row.get("isolation_mode", 0))
            nb = row.get("notebook_path")
            s.notebook_path = Path(nb) if nb else None
            # Restore last 200 messages
            for m in self._db.load_messages(sid, n=200):
                s.messages.append(ChatMessage(role=m["role"], text=m["content"]))
            with self._lock:
                self._sessions[sid] = s

    def persist_message(self, session_id: str, role: str, content: str) -> None:
        """Persist a single message.  No-op if SQLite unavailable."""
        if self._db is None:
            return
        try:
            self._db.save_message(session_id, role, content)
        except Exception:
            pass

    def persist_session(self, session: Session) -> None:
        """Persist full session state.  No-op if SQLite unavailable."""
        if self._db is None:
            return
        try:
            self._db.save_session(session)
        except Exception:
            pass

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[sid] = Session(session_id=sid)
            s = self._sessions[sid]
        if self._db is not None:
            try:
                self._db.create_session(sid)
            except Exception:
                pass
        return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(sid)

    def require(self, sid: str) -> Session:
        s = self.get(sid)
        if s is None:
            raise KeyError(f"session {sid!r} not found")
        return s

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())
