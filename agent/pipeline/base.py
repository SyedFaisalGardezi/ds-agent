from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from agent.core.types import PipelineResult, ToolResult

_CHECKPOINT_ROOT = Path("outputs/datasets/.checkpoints")


class PipelineStage(ABC):
    name: str = ""

    @abstractmethod
    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult: ...

    def execute(self, df: pd.DataFrame, run_id: str) -> PipelineResult:
        start = time.perf_counter()
        ckpt = _CHECKPOINT_ROOT / run_id / f"{self.name}.parquet"
        if ckpt.exists():
            return PipelineResult(
                run_id=run_id,
                stage=self.name,
                status="skipped",
                tool_result=ToolResult(status="ok", data=pd.read_parquet(ckpt),
                                       explanation="Loaded from checkpoint.", math_trace=""),
                elapsed_seconds=0.0,
                checkpoint_path=str(ckpt),
            )
        result = self.run(df, run_id)
        elapsed = time.perf_counter() - start
        if result.status == "ok" and isinstance(result.data, pd.DataFrame):
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            result.data.to_parquet(ckpt, index=False)
        return PipelineResult(
            run_id=run_id,
            stage=self.name,
            status=result.status,
            tool_result=result,
            elapsed_seconds=elapsed,
            checkpoint_path=str(ckpt) if result.status == "ok" else None,
        )


class PipelineOrchestrator:
    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, df: pd.DataFrame) -> list[PipelineResult]:
        run_id = str(uuid.uuid4())[:8]
        results: list[PipelineResult] = []
        current = df
        for stage in self._stages:
            pr = stage.execute(current, run_id)
            results.append(pr)
            if pr.status == "error":
                break
            if isinstance(pr.tool_result.data, pd.DataFrame):
                current = pr.tool_result.data
        return results

    def resume(self, run_id: str, df: pd.DataFrame) -> list[PipelineResult]:
        return self.run(df)
