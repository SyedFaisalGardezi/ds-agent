"""Record every model call the agent makes, for evaluation and fine-tuning data.

Off by default. Set DSAGENT_TRACE_DIR to a folder and every call through the
session's LLM client (`_generate`, `_chat`, `chat_with_tools`) is appended to
<dir>/traces.jsonl as one JSON record: which job the call served, the exact
input the model received, what it returned, how long it took, and the turn it
belonged to. Tool schemas are large and repeat on every turn, so they are stored
once in <dir>/tools.jsonl and referenced by hash.

The job ("purpose") is read from the calling function, so call sites need no
changes: interpret -> intent, read_brief -> brief, WorkPlanTool -> plan_scope /
plan_approach, _critique_step -> critic, _tool_call_loop -> agent_turn, and so
on. A call from anywhere else is recorded as "other:<function>".

Traces contain whatever the model saw, including brief text and data samples.
Keep tracing off for confidential data; the fine-tuning pipeline builds its
dataset from synthetic tasks only.
"""
from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_TURN: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace_turn", default=None)
_LOCK = threading.Lock()

# Calling function (or "Class.method") -> the job the model call serves.
_PURPOSES = {
    "interpret": "intent",
    "check_action": "action_check",
    "read_brief": "brief",
    "_critique_step": "critic",
    "_tool_call_loop": "agent_turn",
    "_run_multiagent_loop": "orchestrator",
    "_synthesise": "synthesis",
    "run_reactive": "qa_answer",
    "WriteSqlTool.execute": "sql",
    "SqlQueryTool.execute": "sql",
    "InferTargetTool.execute": "infer_target",
    "_stage6_narrative": "eda_narrative",
}
_SKIP_FILES = ("llm.py", "trace.py")


def new_turn(session_id: str = "", instruction: str = "") -> str:
    """Start a turn: every call until the next new_turn shares its id."""
    turn_id = uuid.uuid4().hex[:12]
    _TURN.set({"turn_id": turn_id, "session_id": session_id,
               "instruction": instruction[:2000], "seq": 0})
    return turn_id


def _purpose(prompt_text: str) -> str:
    frame = inspect.currentframe()
    try:
        while frame is not None:
            name = frame.f_code.co_name
            if not frame.f_code.co_filename.endswith(_SKIP_FILES):
                owner = frame.f_locals.get("self")
                qualified = f"{type(owner).__name__}.{name}" if owner is not None else name
                if qualified in _PURPOSES:
                    return _PURPOSES[qualified]
                if name in _PURPOSES:
                    return _PURPOSES[name]
                if name == "_ask" or qualified.startswith("WorkPlanTool"):
                    return ("plan_approach" if "SECOND half of planning" in prompt_text
                            else "plan_scope")
            frame = frame.f_back
        return "other"
    finally:
        del frame


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Drop attached images: large, and not part of a text training example."""
    return [{k: v for k, v in m.items() if k != "images"} for m in messages or []]


class _Writer:
    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.traces = folder / "traces.jsonl"
        self.tools = folder / "tools.jsonl"
        self._seen_tools: set[str] = set()

    def tools_ref(self, tools: list[dict]) -> str:
        blob = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        ref = hashlib.sha256(blob.encode()).hexdigest()[:16]
        if ref not in self._seen_tools:
            with _LOCK, self.tools.open("a") as fh:
                fh.write(json.dumps({"ref": ref, "tools": tools}, ensure_ascii=False) + "\n")
            self._seen_tools.add(ref)
        return ref

    def write(self, record: dict) -> None:
        with _LOCK, self.traces.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def install(llm: Any, folder: str | Path | None = None) -> Any:
    """Wrap the client's model calls so each one is recorded. Returns `llm`.
    A no-op when no folder is given and DSAGENT_TRACE_DIR is unset."""
    folder = folder or os.environ.get("DSAGENT_TRACE_DIR")
    if not folder or getattr(llm, "_traced", False):
        return llm
    writer = _Writer(Path(folder))
    backend = type(llm).__name__

    def record(method: str, inputs: dict, prompt_text: str, call):
        turn = _TURN.get() or {}
        if turn:
            turn["seq"] += 1
        t0 = time.time()
        output: Any = None
        error = ""
        try:
            output = call()
            return output
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            writer.write({
                "id": uuid.uuid4().hex[:12], "ts": round(t0, 3),
                "turn_id": turn.get("turn_id", ""), "session_id": turn.get("session_id", ""),
                "seq": turn.get("seq", 0), "instruction": turn.get("instruction", ""),
                "purpose": _purpose(prompt_text), "method": method,
                "backend": backend, "model": getattr(llm, "model", ""),
                "input": inputs, "output": output, "error": error,
                "seconds": round(time.time() - t0, 2)})

    gen, chat, cwt = llm._generate, llm._chat, llm.chat_with_tools

    def _generate(prompt, **kw):
        inputs = {"prompt": prompt, "system": kw.get("system"),
                  "json_mode": kw.get("json_mode", False),
                  "max_tokens": kw.get("max_tokens"), "temperature": kw.get("temperature")}
        return record("generate", inputs, prompt or "", lambda: gen(prompt, **kw))

    def _chat(messages, **kw):
        inputs = {"messages": _clean_messages(messages), "system": kw.get("system"),
                  "max_tokens": kw.get("max_tokens")}
        return record("chat", inputs, json.dumps(messages[-1:], default=str),
                      lambda: chat(messages, **kw))

    def chat_with_tools(messages, tools, **kw):
        inputs = {"messages": _clean_messages(messages), "system": kw.get("system"),
                  "tools_ref": writer.tools_ref(tools), "max_tokens": kw.get("max_tokens")}
        return record("chat_with_tools", inputs, "", lambda: cwt(messages, tools, **kw))

    llm._generate, llm._chat, llm.chat_with_tools = _generate, _chat, chat_with_tools
    llm._traced = True
    return llm
