from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_DB_PATH = Path("outputs/datasets/provenance.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS provenance (
    run_id        TEXT,
    stage         TEXT,
    transformer   TEXT,
    params        JSON,
    input_shape   TEXT,
    output_shape  TEXT,
    null_rate_in  REAL,
    null_rate_out REAL,
    timestamp     DATETIME,
    math_trace    TEXT
);
"""


class ProvenanceStore:
    def __init__(self, db_path: Path = _DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()

    def log(
        self,
        run_id: str,
        stage: str,
        transformer: str,
        params: dict,
        input_shape: tuple,
        output_shape: tuple,
        null_rate_in: float,
        null_rate_out: float,
        math_trace: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO provenance VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, stage, transformer,
                json.dumps(params),
                str(input_shape), str(output_shape),
                null_rate_in, null_rate_out,
                datetime.now(UTC).isoformat(),
                math_trace,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM provenance WHERE run_id=?", (run_id,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
