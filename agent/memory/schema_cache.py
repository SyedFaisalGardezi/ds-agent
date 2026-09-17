"""Dataset schema cache backed by SQLite.

Keyed by SHA-256 of first 64 KB + file size — fast and collision-resistant
for files up to several GB.

On your NVMe SSD: cache hit ~ 1 ms vs 5–30 s for full EDA re-profile.
No embeddings, no external services needed.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path.home() / ".ds-agent" / "memory" / "agent_memory.db"


class SchemaCache:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_cache (
                    file_hash    TEXT PRIMARY KEY,
                    filename     TEXT NOT NULL,
                    schema_json  TEXT NOT NULL,
                    row_count    INTEGER,
                    col_count    INTEGER,
                    cached_at    TEXT NOT NULL
                )
            """)
            self._conn.commit()

    @staticmethod
    def _hash(path: Path) -> str:
        """Hash first 64 KB + file size + mtime. Fast for large files.

        mtime is included so an in-place edit that changes bytes past the
        first 64 KB without changing the total size still re-keys — otherwise
        a stale schema (wrong dtypes/columns) would be served after such an
        edit. Files re-saved with identical content+size+mtime hit the cache.
        """
        st = path.stat()
        with open(path, "rb") as f:
            head = f.read(65536)
        fingerprint = f"{st.st_size}:{int(st.st_mtime)}".encode()
        return hashlib.sha256(head + fingerprint).hexdigest()[:20]

    def save(self, path: Path, df: pd.DataFrame) -> None:
        """Persist schema from a loaded DataFrame.

        Call immediately after `df = pd.read_csv(...)` in chat.py / kernel.py.
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        schema: dict[str, Any] = {
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "shape": list(df.shape),
            "nulls": {c: int(v) for c, v in df.isnull().sum().items()},
            "numeric_stats": df[numeric_cols].describe().round(4).to_dict()
                             if numeric_cols else {},
            "categorical": {
                c: df[c].value_counts().head(10).to_dict()
                for c in cat_cols[:20]
            },
        }

        fhash = self._hash(path)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO schema_cache
                   (file_hash, filename, schema_json, row_count, col_count, cached_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (fhash, path.name, json.dumps(schema), df.shape[0], df.shape[1]),
            )
            self._conn.commit()

    def load(self, path: Path) -> dict[str, Any] | None:
        """Return cached schema dict or None on miss."""
        try:
            fhash = self._hash(path)
        except (OSError, FileNotFoundError):
            return None
        cur = self._conn.execute(
            "SELECT schema_json FROM schema_cache WHERE file_hash = ?", (fhash,)
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def load_as_summary(self, path: Path) -> str | None:
        """Compact human-readable schema string for LLM context injection."""
        schema = self.load(path)
        if not schema:
            return None

        lines = [
            f"Dataset: {Path(path).name}",
            f"Shape: {schema['shape'][0]:,} rows × {schema['shape'][1]} cols",
            f"Columns: {', '.join(schema['columns'][:40])}",
        ]
        null_cols = {c: v for c, v in schema["nulls"].items() if v > 0}
        if null_cols:
            null_summary = ", ".join(f"{c}({v})" for c, v in list(null_cols.items())[:10])
            lines.append(f"Nulls: {null_summary}")
        return "\n".join(lines)

    def list_cached(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT filename, row_count, col_count, cached_at "
            "FROM schema_cache ORDER BY cached_at DESC"
        )
        return [
            {"filename": r[0], "rows": r[1], "cols": r[2], "cached_at": r[3]}
            for r in cur.fetchall()
        ]
