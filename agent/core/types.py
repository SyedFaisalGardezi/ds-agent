from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ToolResult:
    status: Literal["ok", "error", "warning"]
    data: Any
    explanation: str
    math_trace: str
    error: str | None = None


@dataclass
class PipelineResult:
    run_id: str
    stage: str
    status: Literal["ok", "error", "warning", "skipped"]
    tool_result: ToolResult
    elapsed_seconds: float = 0.0
    checkpoint_path: str | None = None


@dataclass
class JobStatus:
    job_id: str
    node: str
    scheduler: Literal["slurm", "pbs"]
    state: Literal["pending", "running", "completed", "failed", "cancelled"]
    stdout_path: str = ""
    stderr_path: str = ""
    exit_code: int | None = None
