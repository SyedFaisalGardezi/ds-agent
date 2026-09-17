from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from agent.core.types import ToolResult

_SQL_PROMPT_TEMPLATE = """You are an expert SQL generator. Given the schema context and a
natural-language question, produce ONLY a valid {dialect} SQL query. No prose, no markdown.

Schema context:
{schema_context}

Question: {question}

SQL:"""

_FORBIDDEN = re.compile(r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|GRANT|REVOKE)\b", re.IGNORECASE)


def _relational_algebra_trace(sql: str) -> str:
    sql_one = " ".join(sql.split()).upper()
    parts: list[str] = []

    cols_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_one)
    from_match = re.search(r"FROM\s+([A-Z0-9_\.]+)", sql_one)
    where_match = re.search(r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)", sql_one)
    group_match = re.search(r"GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|$)", sql_one)

    if from_match:
        relation = from_match.group(1)
        expr = relation
        if where_match:
            expr = f"\\sigma_{{{where_match.group(1).strip()}}}(" + expr + ")"
        if group_match:
            expr = f"{group_match.group(1).strip()}\\;\\mathcal{{G}}\\;" + expr
        if cols_match and cols_match.group(1).strip() != "*":
            expr = f"\\pi_{{{cols_match.group(1).strip()}}}(" + expr + ")"
        parts.append(f"$${expr}$$")

    parts.append(
        "Relational algebra: $\\pi$ = projection (SELECT columns), "
        "$\\sigma$ = selection (WHERE filter), $\\mathcal{G}$ = grouping."
    )
    return "\n".join(parts)


class NL2SQLTool:
    """Converts natural language to SQL using the schema cache and a code LLM.

    Kept dependency-light: LLM + schema search are injected, so unit tests can
    supply mocks without importing langchain/chromadb.
    """

    name: str = "nl2sql"
    description: str = (
        "Converts natural language to SQL using the schema cache. "
        "Input: JSON with 'question', optional 'dialect' (default 'ansi'), optional 'connector'."
    )
    task_type: str = "sql"

    def __init__(
        self,
        llm_invoke: Callable[[str], str] | None = None,
        schema_search: Callable[[str, int], list[str]] | None = None,
        connector_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._llm_invoke = llm_invoke
        self._schema_search = schema_search
        self._connector_factory = connector_factory

    def _default_llm_invoke(self, prompt: str) -> str:
        from agent.core.llm_router import router
        return router.get("sql").invoke(prompt)

    def _default_schema_search(self, query: str, n: int = 5) -> list[str]:
        from agent.connectors.registry import ConnectorRegistry
        return ConnectorRegistry.get().search_schema(query, n=n)

    def _default_connector_factory(self, name: str) -> Any:
        from agent.connectors.registry import ConnectorRegistry
        return ConnectorRegistry.get()[name]

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation="Input was not valid JSON.",
                math_trace="",
                error=f"JSONDecodeError: {exc}",
            )

        question = payload.get("question", "").strip()
        dialect = payload.get("dialect", "ansi")
        connector_name = payload.get("connector")

        if not question:
            return ToolResult(
                status="error",
                data=None,
                explanation="No question provided.",
                math_trace="",
                error="ValueError: question is required",
            )

        search = self._schema_search or self._default_schema_search
        schema_snippets = search(question, 5)
        schema_context = "\n".join(schema_snippets) if schema_snippets else "(no schema cache available)"

        prompt = _SQL_PROMPT_TEMPLATE.format(
            dialect=dialect, schema_context=schema_context, question=question
        )
        invoke = self._llm_invoke or self._default_llm_invoke
        raw = invoke(prompt)
        sql = _strip_fences(raw).strip().rstrip(";")

        if _FORBIDDEN.search(sql):
            return ToolResult(
                status="error",
                data={"sql": sql},
                explanation="Generated SQL contains a destructive statement. Refusing to return.",
                math_trace="",
                error="SafetyError: destructive statement",
            )

        validation_note = ""
        if connector_name:
            factory = self._connector_factory or self._default_connector_factory
            try:
                connector = factory(connector_name)
                explain = connector.query(f"EXPLAIN {sql}")
                if explain.status != "ok":
                    validation_note = (
                        f"EXPLAIN failed on connector '{connector_name}': {explain.error}. "
                        "SQL is returned without validation."
                    )
            except Exception as exc:  # noqa: BLE001 — surface all failures to caller
                validation_note = f"Connector validation skipped ({exc})."

        math_trace = _relational_algebra_trace(sql)
        explanation = (
            f"Generated {dialect.upper()} SQL from question: {question!r}. "
            f"Schema snippets used: {len(schema_snippets)}."
        )
        if validation_note:
            explanation += " " + validation_note

        return ToolResult(status="ok", data={"sql": sql}, explanation=explanation, math_trace=math_trace)

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def register() -> NL2SQLTool:
    """Register the tool in the global ToolRegistry. Safe to call multiple times."""
    from agent.core.tool_registry import ToolRegistry
    tool = NL2SQLTool()
    ToolRegistry.get()._tools[tool.name] = tool  # type: ignore[attr-defined]
    return tool
