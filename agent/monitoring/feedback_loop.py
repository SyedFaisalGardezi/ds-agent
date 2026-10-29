"""FeedbackLoop — store prediction/label pairs, watch accuracy, trigger retrain."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.core.types import ToolResult

_DEFAULT_DB = Path("outputs/reports/feedback.db")


class FeedbackLoop:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                ground_truth TEXT,
                prediction TEXT,
                features TEXT
            )
        """)
        self._conn.commit()
        self._retrain_flag = False

    def store_label(self, prediction_id: str, ground_truth,
                    features: dict, prediction=None) -> ToolResult:
        self._conn.execute(
            "INSERT INTO labels (prediction_id, ground_truth, prediction, features) "
            "VALUES (?, ?, ?, ?)",
            (str(prediction_id),
             json.dumps(ground_truth),
             json.dumps(prediction),
             json.dumps(features)),
        )
        self._conn.commit()
        return ToolResult(status="ok", data={"prediction_id": prediction_id},
                          explanation=f"Stored label for {prediction_id}.",
                          math_trace="INSERT.")

    def _recent_accuracy(self, limit: int) -> float | None:
        cur = self._conn.execute(
            "SELECT ground_truth, prediction FROM labels "
            "ORDER BY id DESC LIMIT ?", (limit,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        correct = sum(1 for g, p in rows if g == p and p is not None)
        return correct / len(rows)

    def check_accuracy_drop(self, delta_threshold: float = 0.03,
                            window: int = 100) -> ToolResult:
        if window < 2 or delta_threshold <= 0:
            return ToolResult(status="error", data=None,
                              explanation="bad window/threshold",
                              math_trace="", error="ValueError")
        cur = self._conn.execute("SELECT COUNT(*) FROM labels")
        n_total = int(cur.fetchone()[0])
        if n_total < 2 * window:
            return ToolResult(
                status="warning",
                data={"drop": None, "n_total": n_total,
                      "needs_more_data": True},
                explanation=f"only {n_total} label(s); need ≥ {2 * window}.",
                math_trace="insufficient history.",
            )
        cur = self._conn.execute(
            "SELECT ground_truth, prediction FROM labels ORDER BY id DESC",
        )
        rows = cur.fetchall()
        recent = rows[:window]
        baseline = rows[window:2 * window]
        def _acc(xs):
            return sum(1 for g, p in xs if g == p and p is not None) / len(xs)
        acc_recent = _acc(recent)
        acc_base = _acc(baseline)
        drop = acc_base - acc_recent
        trigger = drop > delta_threshold
        if trigger:
            self._retrain_flag = True
        return ToolResult(
            status="ok",
            data={"acc_baseline": acc_base, "acc_recent": acc_recent,
                  "drop": float(drop), "threshold": delta_threshold,
                  "trigger_retrain": bool(trigger),
                  "window": window},
            explanation=f"Acc {acc_base:.4f} → {acc_recent:.4f} "
                        f"(Δ={drop:+.4f}); retrain={trigger}.",
            math_trace="drop = acc(baseline) − acc(recent); trigger if drop > δ.",
        )

    def trigger_retrain(self) -> ToolResult:
        was_set = self._retrain_flag
        self._retrain_flag = False
        return ToolResult(
            status="ok",
            data={"retrain_triggered": bool(was_set)},
            explanation=f"Retrain flag consumed (was {was_set}).",
            math_trace="state transition: flag → False.",
        )

    def close(self) -> None:
        self._conn.close()
