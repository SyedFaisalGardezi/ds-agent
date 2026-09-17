"""MathExplainer — enrich a ToolResult with a rigorous derivation.

Uses the LLM router when a live model is available, but degrades gracefully
to a deterministic template string if the router raises (e.g. Ollama
unreachable in tests/CI).
"""
from __future__ import annotations

from agent.core.types import ToolResult

_TEMPLATE = (
    "Definition: operation {op!r}.\n"
    "Formula: (see math_trace of upstream tool).\n"
    "Context: {context}.\n"
    "Interpretation: the result summarises the transformation.\n"
    "Troubleshooting: verify shapes, nulls, and value ranges."
)


class MathExplainer:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from agent.core.llm_router import router
            return router.get("math")
        except Exception:
            return None

    def explain(self, operation: str, context: dict) -> str:
        llm = self._get_llm()
        if llm is None:
            return _TEMPLATE.format(op=operation, context=context)
        try:
            prompt = (
                f"Provide a rigorous mathematical derivation for: {operation}\n"
                f"Context: {context}\n"
                "Format: 1) Definition 2) Formula (LaTeX $...$) 3) Instantiation "
                "4) Interpretation 5) Troubleshooting"
            )
            return str(llm.invoke(prompt))
        except Exception:
            return _TEMPLATE.format(op=operation, context=context)

    def enrich_result(self, result: ToolResult, operation: str,
                      context: dict) -> ToolResult:
        if result.math_trace:
            return result
        result.math_trace = self.explain(operation, context)
        return result
