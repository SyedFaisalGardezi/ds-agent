"""Per-job progress bus — thread-safe message queue for SSE streaming.

Agent runner calls emit() while executing; the SSE endpoint reads from
the queue and pushes events to the browser in real-time.
"""
from __future__ import annotations

import queue
import threading
import time

_queues: dict[str, queue.Queue] = {}
_lock = threading.Lock()


def create(job_id: str, maxsize: int = 500) -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    with _lock:
        _queues[job_id] = q
    return q


def emit(job_id: str, message: str, *,
         tool: str = "", level: str = "info") -> None:
    with _lock:
        q = _queues.get(job_id)
    if q:
        try:
            q.put_nowait({
                "type": "progress",
                "ts": round(time.time(), 3),
                "tool": tool,
                "level": level,
                "message": message,
            })
        except queue.Full:
            pass


def close(job_id: str) -> None:
    """Signal end-of-stream with a None sentinel, then remove the queue."""
    with _lock:
        q = _queues.get(job_id)
    if q:
        q.put(None)           # sentinel → SSE generator stops
    with _lock:
        _queues.pop(job_id, None)


def get_queue(job_id: str) -> queue.Queue | None:
    with _lock:
        return _queues.get(job_id)
