from __future__ import annotations

import pandas as pd

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage


class PassThroughStage(PipelineStage):
    name = "passthrough"

    def run(self, df: pd.DataFrame, run_id: str) -> ToolResult:
        return ToolResult(status="ok", data=df, explanation="Pass-through.", math_trace="")


def test_stage_execute_produces_pipeline_result(synthetic_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
    import agent.pipeline.base as base_mod
    monkeypatch.setattr(base_mod, "_CHECKPOINT_ROOT", tmp_path)
    stage = PassThroughStage()
    pr = stage.execute(synthetic_df, run_id="test001")
    assert pr.status == "ok"
    assert isinstance(pr.tool_result.data, pd.DataFrame)
