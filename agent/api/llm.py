"""Ollama-backed LLM client for the chat router.

Talks straight to the Ollama HTTP API (no LangChain wrapper) so the
dependency surface stays small. Used for two things:

  1. **Intent classification** — when the keyword router can't tell
     what the user wants, ask the LLM to pick a label from a fixed set.
  2. **Free-form replies** — when the user is just chatting (asking
     about the data, the brief, what to do next), generate a grounded
     answer using session context (data schema, brief, last metrics).

Falls back gracefully: if Ollama isn't reachable, every method returns
a sentinel that the caller can detect (`available=False`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

# Scope/planning model: deep reasoning on task briefs, intent, synthesis.
SCOPE_MODEL = os.environ.get(
    "DSAGENT_SCOPE_MODEL",
    "qwen3.8:27b-mlx",
)
# Code model: Python/SQL generation, numeric reasoning.
CODE_MODEL = os.environ.get("DSAGENT_CODE_MODEL", SCOPE_MODEL)
# Legacy env-var still honoured; falls back to SCOPE_MODEL.
DEFAULT_MODEL = os.environ.get("DSAGENT_LLM_MODEL", SCOPE_MODEL)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Context window sent with every request. The Modelfile pins num_ctx=8192,
# which silently head-truncates long prompts (system prompt goes FIRST).
# qwen3.8:27b-mlx supports 262K; 16K fits comfortably in 32 GB with the
# MLX (nvfp4) KV cache. Override via DSAGENT_NUM_CTX.
NUM_CTX = int(os.environ.get("DSAGENT_NUM_CTX", "16384"))

# Disable qwen3 thinking at the API level (saves 30-60s/call on M-series;
# the /no_think prompt directive alone is unreliable on distills).
# Set DSAGENT_THINK=1 to re-enable thinking.
DISABLE_THINK = os.environ.get("DSAGENT_THINK", "0") != "1"

INTENTS = ("build", "set_target", "status", "help", "qa", "unknown")

_INTENT_PROMPT = """\
You are an intent classifier for a data-science assistant.

Read the user message and pick exactly one intent label from this set:
  - build       : user explicitly wants to RUN something new — train a model, run EDA,
                  build a notebook, execute the pipeline, perform an analysis.
                  Examples: "train the model", "run EDA", "build the notebook", "start the pipeline".
  - set_target  : user is naming the target column (e.g. 'target is price', 'predict y').
  - status      : user is asking what's loaded / what's been done in this session.
  - help        : user is asking for help, commands, or usage instructions.
  - qa          : EVERYTHING ELSE — questions, explanations, plans, feedback, acknowledgments.
                  This is the DEFAULT label. Use it for:
                  "what is the task?", "how many rows?", "what columns are there?",
                  "plan the analysis", "outline the approach", "explain the methodology",
                  "what should I do?", "what does X column mean?", "what metric is used?",
                  "what were the results?", "ok thanks", "got it", "yes", "no",
                  "what is the target?", "tell me about the brief", "describe the problem",
                  "what does the brief say?", "summarise the task", "think about this".
  - unknown     : only when the message is completely unintelligible or unrelated to data science.

IMPORTANT: When in doubt between qa and unknown, choose qa. Most messages are qa.
Only use build when the user clearly wants computation to START (not just ask about it).

User message: {message}

Respond with a JSON object: {{"intent": "<label>", "target": "<column-or-null>"}}.
If the intent is set_target, fill 'target' with the column name they said. Otherwise leave it null.
Reply with ONLY the JSON, nothing else.
"""

_QA_SYSTEM = """\
/no_think
You are a senior data scientist assisting a user via a chat API.
You have access to a session that includes (in SESSION STATE below):
  - the full BRIEF text — read it carefully to answer task/requirement questions
  - the dataset schema: shape, columns, null stats
  - an extracted TASK_SPEC: task type, target, metric, aggregation
  - results from recent pipeline runs (metrics, feature importance, etc.)
  - recent conversation history

CRITICAL RULES:
1. Answer factual questions from the SESSION STATE — do NOT say "run a tool" when you already
   have the answer (e.g. "how many columns?" → count df.columns; "what's the target?" → read TASK_SPEC).
2. When asked about the task, brief, or requirements → read the BRIEF content in SESSION STATE
   and answer directly. Do not ask the user to run understand_task.
3. When asked about data stats (row count, null %, column names) → use df.shape and df.columns.
4. Only tell the user to run a tool when they explicitly ask to DO something new that requires
   computation (e.g. "train a model", "run EDA", "build the notebook").

If the user asks you to plan, outline, or strategise:
  - Write a numbered step-by-step plan grounded in the TASK_SPEC and BRIEF details.
  - Include: problem statement, data prep steps, modelling approach, evaluation metric,
    any special requirements (aggregation, SHAP, output file, etc.).
  - Be concrete — name the target column, metric, and model types to try.
  - Do NOT say "I will run tools" — write the plan as prose/numbered steps.

Conversational / short queries:
  - Acknowledge naturally in 1-2 sentences. No over-explanation.
  - "Thanks / ok / got it" → brief warm acknowledgment, offer next step.

ANSWER DISCIPLINE (important):
  - Answer the question ONCE. Do not restate it, and do not repeat an answer you
    already gave in a previous turn — the user can see the history.
  - Do NOT re-explain earlier definitions or re-print prior answers under new
    headings. If a follow-up only needs one line, give one line and stop.
  - No filler sections ("Why X?", "Summary") unless the user asked for detail.

Do NOT emit <think> blocks or internal reasoning. Output only the final response.
Simple factual answers: under 80 words, no headings. Plans/outlines: up to 800 words.
No preamble. No emoji.
"""


@dataclass
class IntentResult:
    intent: str
    target: str | None
    available: bool
    raw: str = ""


@dataclass
class ChatResult:
    text: str
    available: bool


class OllamaClient:
    """Minimal Ollama wrapper. One client per process is fine."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST,
                 timeout: float = 180.0, probe_ttl: float = 30.0,
                 code_model: str = CODE_MODEL) -> None:
        self.model = model                # scope / planning / reasoning model
        self.code_model = code_model      # coding / SQL / numeric model
        self.host = host.rstrip("/")
        self._timeout = timeout
        self._probe_ttl = probe_ttl
        self._last_ok: float = 0.0      # timestamp of last successful probe
        # Native function-calling support, learned on first use: None = unknown,
        # True = model accepted a `tools` payload, False = model/Ollama rejected
        # it (so callers fall back to the text ReAct loop without re-probing).
        self._tools_supported: bool | None = None

    # ---- liveness ---------------------------------------------------- #

    def is_available(self) -> bool:
        """Probe /api/tags. Positive results cached for `probe_ttl` seconds;
        negative results always retry so a late-starting Ollama is picked up.
        """
        import time
        if time.time() - self._last_ok < self._probe_ttl:
            return True
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            tags = r.json().get("models", []) if r.status_code == 200 else []
            names = {m.get("name") for m in tags}
            if self.model in names:
                self._last_ok = time.time()
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    # ---- raw generate ----------------------------------------------- #

    # Qwen/Llama chat-template tokens that must never appear in responses
    _STOP_TOKENS: list[str] = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|end_of_text|>", "<|eot_id|>",
    ]

    def _generate(self, prompt: str, *, system: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 512,
                  json_mode: bool = False, think: bool | None = None,
                  _model_override: str | None = None) -> str:
        body: dict[str, Any] = {
            "model": _model_override or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
                "stop": self._STOP_TOKENS,
            },
        }
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"
        # qwen3 thinking off at API level unless explicitly requested.
        if think is False or (think is None and DISABLE_THINK):
            body["think"] = False
        raw = self._post_llm("/api/generate", body, "response")
        return self._strip_special_tokens(raw)

    def _post_llm(self, path: str, body: dict[str, Any],
                  response_key: str) -> str:
        """POST to Ollama; if the model rejects the `think` field (older
        Ollama / non-thinking model → 400), retry once without it."""
        r = httpx.post(f"{self.host}{path}", json=body, timeout=self._timeout)
        if r.status_code == 400 and "think" in body:
            body = {k: v for k, v in body.items() if k != "think"}
            r = httpx.post(f"{self.host}{path}", json=body,
                           timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if response_key == "message":
            return (data.get("message", {}).get("content") or "").strip()
        return (data.get(response_key) or "").strip()

    @classmethod
    def _strip_special_tokens(cls, text: str) -> str:
        """Strip chat-template special tokens that leak into model output."""
        for tok in cls._STOP_TOKENS:
            # Everything from the first special token onwards is garbage
            idx = text.find(tok)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks that qwen3 models emit.

        Handles the truncated case too: if the model opened <think> but ran
        out of tokens before </think>, drop the whole thinking tail so we
        don't return raw reasoning to the user.
        """
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Unclosed <think> at end → strip from opener onwards.
        if "<think>" in cleaned and "</think>" not in cleaned:
            cleaned = cleaned.split("<think>", 1)[0]
        return cleaned.strip()

    # ---- intent ----------------------------------------------------- #

    def classify_intent(self, message: str) -> IntentResult:
        if not self.is_available():
            return IntentResult(intent="unknown", target=None,
                                available=False, raw="")
        try:
            raw = self._generate(_INTENT_PROMPT.format(message=message),
                                 temperature=0.0, max_tokens=512,
                                 json_mode=True)
        except Exception as exc:  # noqa: BLE001
            return IntentResult(intent="unknown", target=None,
                                available=False, raw=f"[error: {exc}]")
        # Strip qwen3 thinking blocks and markdown fences.
        cleaned = self._strip_thinking(raw).strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        # Find the first {...} block in case the model added prose.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        try:
            parsed = json.loads(cleaned)
            intent = parsed.get("intent", "unknown")
            if intent not in INTENTS:
                intent = "unknown"
            target = parsed.get("target")
            if isinstance(target, str) and not target.strip():
                target = None
            return IntentResult(intent=intent, target=target,
                                available=True, raw=raw)
        except json.JSONDecodeError:
            return IntentResult(intent="unknown", target=None,
                                available=True, raw=raw)

    # ---- free-form QA ----------------------------------------------- #

    def generate_code(self, prompt: str, *, system: str | None = None,
                      max_tokens: int = 1024) -> str:
        """Generate Python / SQL code using the dedicated code model."""
        if not self.is_available():
            return ""
        try:
            raw = self._generate(
                prompt, system=system, temperature=0.0,
                max_tokens=max_tokens,
                _model_override=self.code_model,
            )
            return self._strip_thinking(raw)
        except Exception:
            return ""

    def _chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Multi-turn chat via Ollama /api/chat.

        `messages` is a list of {"role": "user"|"assistant", "content": "..."}.
        The system prompt is passed separately and prepended by Ollama.
        """
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
                "stop": self._STOP_TOKENS,
            },
        }
        if DISABLE_THINK:
            body["think"] = False
        raw = self._post_llm("/api/chat", body, "message")
        raw = self._strip_special_tokens(raw)
        return self._strip_thinking(raw)

    def supports_tools(self) -> bool | None:
        """Tri-state cache of native function-calling support.

        None until the first `chat_with_tools` call has learned the answer.
        """
        return self._tools_supported

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        think: bool | None = None,
    ) -> dict | None:
        """One turn of native Ollama function-calling via /api/chat.

        `tools` is a list of Ollama tool schemas
        ({"type": "function", "function": {name, description, parameters}}).

        Returns the assistant message as
        {"content": str, "tool_calls": [ {"function": {"name", "arguments"}} ]}
        with `tool_calls` empty when the model chose to answer directly.

        Returns None when the model or the installed Ollama does not support
        the `tools` field (HTTP 400 / 'does not support tools'), so the caller
        can fall back to the text ReAct loop. `_tools_supported` is cached so
        the fallback only probes once per process.
        """
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": NUM_CTX,
                "stop": self._STOP_TOKENS,
            },
        }
        if think is False or (think is None and DISABLE_THINK):
            body["think"] = False

        try:
            r = httpx.post(f"{self.host}/api/chat", json=body,
                           timeout=self._timeout)
            # Retry once without `think` (older Ollama rejects the field).
            if r.status_code == 400 and "think" in body:
                body = {k: v for k, v in body.items() if k != "think"}
                r = httpx.post(f"{self.host}/api/chat", json=body,
                               timeout=self._timeout)
            # A 400 mentioning tools means this model can't do native calling.
            if r.status_code == 400 and "tool" in r.text.lower():
                self._tools_supported = False
                return None
            r.raise_for_status()
        except httpx.HTTPStatusError:
            # Non-tool 4xx/5xx: treat as unsupported so we degrade, not crash.
            self._tools_supported = False
            return None

        self._tools_supported = True
        msg = (r.json().get("message") or {})
        content = self._strip_thinking(
            self._strip_special_tokens(msg.get("content") or ""))
        return {"content": content, "tool_calls": msg.get("tool_calls") or []}

    def answer(
        self,
        question: str,
        *,
        context: str,
        messages: list[dict] | None = None,
    ) -> ChatResult:
        """Answer a question, optionally with multi-turn message history.

        When `messages` is provided (list of {"role", "content"} dicts),
        the full conversation is sent via /api/chat for proper memory.
        Without it, falls back to single-turn /api/generate.
        """
        if not self.is_available():
            return ChatResult(text="(LLM offline — start Ollama and pull "
                                   f"{self.model} for free-form chat.)",
                              available=False)
        try:
            if messages:
                # True multi-turn: send the full conversation history
                system = _QA_SYSTEM
                if context:
                    system = _QA_SYSTEM + f"\n\nSESSION STATE:\n{context}"
                text = self._chat(messages, system=system,
                                  temperature=0.3, max_tokens=2048)
            else:
                prompt = f"Context:\n{context}\n\nUser question: {question}"
                raw = self._generate(prompt, system=_QA_SYSTEM,
                                     temperature=0.3, max_tokens=2048)
                text = self._strip_thinking(raw)
            if not text:
                text = ("(The model emitted only internal reasoning. Try a "
                        "shorter or more direct question, or re-send — the "
                        "/no_think directive should take effect now.)")
            return ChatResult(text=text, available=True)
        except Exception as exc:  # noqa: BLE001
            return ChatResult(text=f"(LLM error: {exc})", available=False)
