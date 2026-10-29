from __future__ import annotations

from agent.core.types import PipelineResult, ToolResult


def test_tool_result_ok() -> None:
    r = ToolResult(status="ok", data={"x": 1}, explanation="It worked.", math_trace="$x = 1$")
    assert r.status == "ok"
    assert r.error is None


def test_tool_result_error() -> None:
    r = ToolResult(status="error", data=None, explanation="Failed.", math_trace="", error="ValueError")
    assert r.status == "error"
    assert r.error == "ValueError"


def test_pipeline_result() -> None:
    tr = ToolResult(status="ok", data=None, explanation="ok", math_trace="")
    pr = PipelineResult(run_id="abc", stage="extract", status="ok", tool_result=tr)
    assert pr.stage == "extract"
