"""Tiny in-process background-job runner for slow chat actions.

Why not Celery/RQ? This is a single-machine demo agent. A bounded
ThreadPoolExecutor + a thread-safe dict is enough — and keeps the
deployment surface to one process.

A Job has four states (pending → running → done | failed) and stores
its result (or error) once finished. The chat router returns a job_id
immediately; the client polls `GET /chat/sessions/{sid}/jobs/{jid}`.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# A job running longer than this is reported as timed out to pollers.
# The worker thread itself keeps running (Python threads can't be killed),
# but kernel-level exec timeouts (DSAGENT_EXEC_TIMEOUT_S) and LLM httpx
# timeouts bound the realistic hang sources underneath.
DEFAULT_JOB_TIMEOUT_S = float(os.environ.get("DSAGENT_JOB_TIMEOUT_S", "900"))
# Finished jobs older than this are pruned to stop unbounded dict growth.
FINISHED_JOB_TTL_S = float(os.environ.get("DSAGENT_JOB_TTL_S", "1800"))


@dataclass
class Job:
    job_id: str
    kind: str                          # e.g. "build_notebook", "llm_chat"
    state: str = "pending"             # pending | running | done | failed | timeout
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    timeout_s: float = DEFAULT_JOB_TIMEOUT_S

    def _effective_state(self) -> str:
        """Running jobs past their deadline are reported as 'timeout'."""
        if (self.state == "running" and self.started_at
                and time.time() - self.started_at > self.timeout_s):
            return "timeout"
        return self.state

    def to_public(self) -> dict[str, Any]:
        state = self._effective_state()
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                round((self.finished_at or time.time()) - (self.started_at or self.created_at), 2)
                if self.started_at else 0.0
            ),
            "result": self.result if state == "done" else None,
            "error": (
                self.error if state != "timeout"
                else f"Job exceeded {self.timeout_s:.0f}s deadline."
            ),
        }


class JobRunner:
    """Submit callables; each runs on a worker thread and stores its outcome."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="dsagent-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, func: Callable[[], Any],
               *, job_id: str | None = None,
               timeout_s: float | None = None) -> Job:
        jid = job_id or uuid.uuid4().hex[:10]
        job = Job(job_id=jid, kind=kind,
                  timeout_s=timeout_s or DEFAULT_JOB_TIMEOUT_S)
        with self._lock:
            self._prune_locked()
            self._jobs[jid] = job

        def _run() -> None:
            with self._lock:
                job.state = "running"
                job.started_at = time.time()
            try:
                job.result = func()
                with self._lock:
                    job.state = "done"
                    job.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job.state = "failed"
                    job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
                    job.finished_at = time.time()

        self._pool.submit(_run)
        return job

    def _prune_locked(self) -> None:
        """Drop finished jobs older than FINISHED_JOB_TTL_S. Caller holds lock."""
        cutoff = time.time() - FINISHED_JOB_TTL_S
        stale = [
            jid for jid, j in self._jobs.items()
            if j.finished_at is not None and j.finished_at < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
