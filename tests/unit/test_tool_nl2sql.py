from __future__ import annotations

import json

from agent.core.tool_registry import ToolRegistry
from agent.core.types import ToolResult
from agent.tools.nl2sql import NL2SQLTool, register


def make_tool(llm_out: str, schema: list[str] | None = None, connector=None) -> NL2SQLTool:
    return NL2SQLTool(
        llm_invoke=lambda _prompt: llm_out,
        schema_search=lambda _q, _n=5: schema or [],
        connector_factory=(lambda _name: connector) if connector else None,
    )


def test_rejects_blank_question() -> None:
    tool = make_tool("SELECT 1")
    result = tool.run(json.dumps({"question": ""}))
    assert result.status == "error"
    assert "question is required" in (result.error or "")


def test_rejects_invalid_json() -> None:
    tool = make_tool("SELECT 1")
    result = tool.run("not-json{")
    assert result.status == "error"
    assert "JSONDecodeError" in (result.error or "")


def test_generates_sql_and_populates_math_trace() -> None:
    tool = make_tool("SELECT name, revenue FROM sales WHERE region = 'NZ'")
    result = tool.run(json.dumps({"question": "Show me NZ sales names and revenue"}))
    assert result.status == "ok"
    assert result.data["sql"].startswith("SELECT")
    assert "\\pi" in result.math_trace
    assert "\\sigma" in result.math_trace


def test_strips_markdown_fences() -> None:
    tool = make_tool("```sql\nSELECT * FROM t\n```")
    result = tool.run(json.dumps({"question": "all rows from t"}))
    assert result.status == "ok"
    assert result.data["sql"] == "SELECT * FROM t"


def test_blocks_destructive_sql() -> None:
    tool = make_tool("DROP TABLE users")
    result = tool.run(json.dumps({"question": "delete users"}))
    assert result.status == "error"
    assert "destructive" in result.explanation.lower()


def test_uses_schema_context() -> None:
    captured: list[str] = []

    def llm(prompt: str) -> str:
        captured.append(prompt)
        return "SELECT 1"

    tool = NL2SQLTool(
        llm_invoke=llm,
        schema_search=lambda _q, _n=5: ["sales.revenue:FLOAT", "sales.region:VARCHAR"],
    )
    tool.run(json.dumps({"question": "sum revenue"}))
    assert "sales.revenue" in captured[0]
    assert "sales.region" in captured[0]


def test_connector_validation_note_on_explain_failure() -> None:
    class FailingConnector:
        def query(self, _sql: str) -> ToolResult:
            return ToolResult(status="error", data=None, explanation="bad", math_trace="", error="SyntaxError")

    tool = make_tool("SELECT * FROM nope", connector=FailingConnector())
    result = tool.run(json.dumps({"question": "anything", "connector": "x"}))
    assert result.status == "ok"
    assert "EXPLAIN failed" in result.explanation


def test_register_adds_to_registry() -> None:
    ToolRegistry._instance = None
    register()
    assert "nl2sql" in ToolRegistry.get()
    assert ToolRegistry.get()["nl2sql"].task_type == "sql"
