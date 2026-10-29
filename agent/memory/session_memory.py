"""Smart context window builder.

Replaces hard-truncation in _chat_messages() with:
  1. Summary buffer — older turns compressed by LLM, not dropped
  2. LTM injection — relevant past-session memories prepended as system context

Returns the same [{role, content}] format as Ollama /api/chat expects.
Falls back to simple truncation if summarisation or LTM fails.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.api.sessions import Session
    from agent.memory.long_term import LongTermMemory

# ── Tuneable constants ────────────────────────────────────────────────────────

RECENT_TURNS         = int(os.environ.get("DSAGENT_RECENT_TURNS", "8"))
USER_CHAR_LIMIT      = 800
ASSISTANT_CHAR_LIMIT = 2000
SUMMARY_MAX_TOKENS   = int(os.environ.get("DSAGENT_SUMMARY_TOKENS", "400"))
LTM_HITS             = int(os.environ.get("DSAGENT_LTM_HITS", "4"))
LTM_DISTANCE_THRESH  = float(os.environ.get("DSAGENT_LTM_DISTANCE", "0.55"))


def _trim(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def build_context_messages(
    session: Session,
    llm: Any,
    ltm: LongTermMemory | None = None,
    current_query: str = "",
) -> list[dict[str, str]]:
    """Build the messages list for an Ollama /api/chat call.

    Steps:
      1. Split messages into older (summarise) and recent (verbatim).
      2. If older turns exist, ask LLM to compress them into a summary block.
      3. Optionally retrieve relevant LTM chunks and inject as system context.
      4. Return: [summary_system_msg?, ltm_system_msg?, ...recent_turns]

    Falls back gracefully — any failure returns the recent turns verbatim.
    """
    messages = getattr(session, "messages", [])
    isolation = getattr(session, "isolation_mode", False)

    if not messages:
        return []

    # Filter to user/assistant only (skip system messages for context building)
    chat_msgs = [m for m in messages if m.role in ("user", "assistant")]

    if not chat_msgs:
        return []

    # ── Step 1: split ─────────────────────────────────────────────────────────
    if len(chat_msgs) <= RECENT_TURNS:
        older_msgs: list[Any] = []
        recent_msgs = chat_msgs
    else:
        older_msgs = chat_msgs[:-RECENT_TURNS]
        recent_msgs = chat_msgs[-RECENT_TURNS:]

    system_blocks: list[dict[str, str]] = []

    # ── Step 2: summarise older turns ─────────────────────────────────────────
    if older_msgs and llm is not None:
        try:
            older_text = "\n".join(
                f"{m.role.upper()}: {m.text[:600]}"
                for m in older_msgs
            )
            summary = llm._generate(
                prompt=(
                    "/no_think\n"
                    "Summarise this conversation in 4–6 sentences. "
                    "Preserve: task type, dataset name, key findings, "
                    "best model and its score, any errors, current goal.\n\n"
                    f"{older_text}"
                ),
                temperature=0.0,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            if summary and summary.strip():
                system_blocks.append({
                    "role": "system",
                    "content": f"[Earlier conversation summary]\n{summary.strip()}",
                })
        except Exception:
            pass  # summarisation failure is non-fatal

    # ── Step 3: LTM retrieval (skipped in isolation mode) ────────────────────
    if ltm is not None and current_query and not isolation:
        try:
            ltm_context = ltm.retrieve_as_context_string(
                query=current_query,
                n=LTM_HITS,
                session_id=getattr(session, "session_id", None),
                isolation_mode=isolation,
                distance_threshold=LTM_DISTANCE_THRESH,
            )
            if ltm_context:
                system_blocks.append({
                    "role": "system",
                    "content": ltm_context,
                })
        except Exception:
            pass  # LTM failure is non-fatal

    # ── Step 4: format recent turns ───────────────────────────────────────────
    recent_formatted = [
        {
            "role": m.role,
            "content": _trim(
                m.text,
                USER_CHAR_LIMIT if m.role == "user" else ASSISTANT_CHAR_LIMIT,
            ),
        }
        for m in recent_msgs
    ]

    return system_blocks + recent_formatted
