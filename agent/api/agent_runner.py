"""Full agentic runner for ds-agent.

Architecture:
  1. InstructionParser — accept natural language from chat OR a document path
  2. TaskPlanner (LLM) — decompose into a JSON step list; each step names a tool
  3. Executor — run each tool in order, capture outputs, feed back to planner
  4. Re-planner — after each tool output the LLM may revise remaining steps
  5. Synthesizer (LLM) — write a concise final answer from all findings

The LLM never has to write code from scratch for standard DS tasks — it selects
tools from the MCP registry. For custom analysis it uses execute_code.
Everything runs locally via Ollama.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.api.agent_tools import TOOL_REGISTRY, ToolOutput, work_plan_is_stale
from agent.api.kernel import SessionKernel
from agent.api.sessions import Session

# Memory imports — all optional; fall back gracefully if unavailable
try:
    from agent.memory.session_memory import build_context_messages
    _MEMORY_OK = True
except Exception:
    _MEMORY_OK = False

MAX_STEPS = 10          # safety cap on plan length
MAX_REPLAN = 1          # how many times the LLM may revise the plan mid-run

# Prefer native Ollama function-calling for the agentic loop; fall back to the
# text ReAct loop when the model/Ollama can't do it. Set DSAGENT_NATIVE_TOOLS=0
# to force the text loop.
_NATIVE_TOOLS = os.environ.get("DSAGENT_NATIVE_TOOLS", "1") != "0"

# Reflexion-style critic: after each tool result, judge whether it moved toward
# the goal and steer the next turn (accept / retry / replan / stop). Set
# DSAGENT_CRITIC=0 to disable. Retries and replans are budgeted so the loop
# always terminates.
_CRITIC_ENABLED = os.environ.get("DSAGENT_CRITIC", "1") != "0"
_MAX_RETRIES_PER_STEP = int(os.environ.get("DSAGENT_MAX_RETRIES", "2"))
_MAX_LOOP_REPLANS = int(os.environ.get("DSAGENT_MAX_REPLANS", "2"))

# The critic costs a full LLM call per tool result, which on a local model is as
# expensive as the step it is judging. Retrieval tools either return the bytes
# asked for or an error, and the error path is caught by the rule-based check
# without an LLM call, so a successful one has nothing for the critic to weigh.
# Judge the tools whose output should actually change the plan: what the data
# looks like, how good it is, and how a model scored.
_CRITIC_SKIP_TOOLS = frozenset({
    "explore_directory", "read_file", "search_files", "read_instructions",
    "write_file", "edit_file", "build_notebook", "scaffold_cicd", "visualize",
})
# Set DSAGENT_CRITIC_ALL=1 to critique every tool result (slower, more thorough).
_CRITIC_ALL_TOOLS = os.environ.get("DSAGENT_CRITIC_ALL", "0") == "1"

# Multi-agent orchestration: a supervisor dispatches sub-goals to role-specialised
# agents. OFF by default. A data-science pipeline is a dependency chain —
# discover, understand, profile, plan, engineer, train — and that is the shape
# multi-agent handles worst: Anthropic report it suits "breadth-first queries
# pursuing multiple independent directions" and that "domains that require all
# agents to share the same context or involve many dependencies between agents
# are not a good fit". Measured here, the handoffs lose most of what they carry
# (see _run_multiagent_loop). Set DSAGENT_MULTIAGENT=1 to use it anyway.
_MULTIAGENT = os.environ.get("DSAGENT_MULTIAGENT", "0") != "0"
_MAX_ORCH_ROUNDS = int(os.environ.get("DSAGENT_MAX_ORCH_ROUNDS", "6"))

# Step budget for the agentic loop. This used to share `max_tool_calls` with the
# deterministic keyword plan, which left the single agent 12 steps while the
# multi-agent path took 6 rounds x 6 specialist steps = 36 and ignored the
# argument entirely. Those are different jobs, so they get different numbers.
_AGENT_MAX_STEPS = int(os.environ.get("DSAGENT_AGENT_STEPS", "24"))
_SPECIALIST_MAX_STEPS = int(os.environ.get("DSAGENT_SPECIALIST_STEPS", "6"))

# Sentinel returned by _classify_tools for open-ended exploration/analysis. It
# is not a real tool: run_reactive intercepts it and hands the turn to the
# adaptive tool loop so the LLM designs the analysis and reasons about each
# result, instead of executing a fixed keyword-chosen sequence.
_ADAPTIVE_SENTINEL = "__adaptive__"

# Output budget for one tool-calling turn. This has to hold the whole tool call,
# and when the agent writes a document the document IS an argument, so a small
# budget truncates the JSON mid-string. A truncated call does not parse, and
# _coerce_tool_args then yields {}, which surfaces as "needs a 'path'" rather
# than as "too long". 3000 tokens is roughly 2200 words of file content and
# still leaves room inside num_ctx=8192 beside the prompt and tool schemas.
# Raise DSAGENT_NUM_CTX too before raising this much further.
_TOOL_TURN_TOKENS = int(os.environ.get("DSAGENT_TOOL_TURN_TOKENS", "4000"))

# A reasoning model spends output tokens thinking before it answers, and the
# budget has to cover both or the reply is cut off before the JSON appears.
# Measured with gpt-oss:20b on this project's prompts: the router thinks ~590
# characters, the critic ~800. These were 60 and 280, which returned nothing.
_ROUTER_TOKENS = int(os.environ.get("DSAGENT_ROUTER_TOKENS", "500"))
_CRITIC_TOKENS = int(os.environ.get("DSAGENT_CRITIC_TOKENS", "900"))

# How much of a tool result the model gets to read. The newest result keeps the
# full budget because that is what the next decision rests on; older ones are
# trimmed to a gist so the conversation cannot outgrow num_ctx.
_OBS_CHARS = int(os.environ.get("DSAGENT_OBS_CHARS", "4000"))
_OBS_TAIL_CHARS = int(os.environ.get("DSAGENT_OBS_TAIL_CHARS", "700"))

# Show the model the plots it produces, so a distribution or a correlation
# heatmap is read rather than guessed at from summary statistics. OFF by
# default because the default model (gpt-oss:20b) has no vision, and a model
# that ignores images rather than rejecting them would silently burn tokens on
# every figure. Set DSAGENT_SEND_FIGURES=1 when running a vision model; it was
# verified working on qwen3.8:27b-mlx, which read a histogram correctly.
# Only the newest step's figures are attached, and only a couple of them.
_SEND_FIGURES = os.environ.get("DSAGENT_SEND_FIGURES", "0") != "0"
_MAX_FIGURES_SENT = int(os.environ.get("DSAGENT_MAX_FIGURES_SENT", "2"))
# Cap on the encoded size of an attached figure. The kernel saves at dpi=100,
# where a multi-panel EDA grid runs to about 48 KB, and on a 27B local model a
# 7 KB image already added around 90 s to a turn.
_MAX_FIGURE_CHARS = int(os.environ.get("DSAGENT_MAX_FIGURE_CHARS", "20000"))


def _shrink_figure(b64: str, max_chars: int = _MAX_FIGURE_CHARS) -> str | None:
    """Compress an encoded PNG to fit the budget, keeping it readable.

    Charts compress by colour count, not by pixel count. A plot is flat areas
    and a few line colours, which PNG stores cheaply; resampling it smaller
    anti-aliases every crisp edge into a gradient and makes the file BIGGER.
    Measured on a 12-panel histogram grid: 48,112 chars original, 18,140 after
    quantising at full resolution, but 28,512 after resizing to 75% and
    quantising. So quantise first and only resize if that is not enough, since
    resolution is what keeps axis labels legible.

    Returns None when it cannot be brought under the cap, so the caller skips
    the picture rather than spending minutes on it. Best-effort: Pillow ships
    with matplotlib, but a missing decoder only costs the image.
    """
    if len(b64) <= max_chars:
        return b64
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image
        img = Image.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB")

        def _encode(im, colors: int) -> str:
            buf = _io.BytesIO()
            im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(
                buf, format="PNG", optimize=True)
            return _b64.b64encode(buf.getvalue()).decode()

        for scale, colors in ((1.0, 64), (1.0, 32), (1.0, 16),
                              (0.7, 32), (0.5, 32), (0.5, 16)):
            im = img if scale == 1.0 else img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS)
            out = _encode(im, colors)
            if len(out) <= max_chars:
                return out
    except Exception:  # noqa: BLE001
        return None
    return None

# ── compact tool list shown to planner (much shorter than full MCP JSON) ──────

_TOOL_LIST = """\
- understand_task    : Parse the PDF brief to extract task spec: target definition, task type, evaluation metric, whether aggregation is needed, secondary datasets. Run FIRST when brief is loaded.
- aggregate_data     : Group incident/transaction rows into one row per entity (e.g. per establishment). Optionally loads a second dataset and derives a binary target from it. Args: group_by (str), target_source_path (str), target_condition (str), target_column_name (str).
- eda_profile        : Profile dataset — shape, dtypes, nulls, stats, task-aware feature-vs-target analysis (correlation, AUC, class KDEs). Args: none.
- quality_check      : Outliers, constants, high-cardinality, skew, VIF, leakage detection. Args: none.
- deep_eda           : Six-stage deep EDA — quality radar, univariate panorama, bivariate type-dispatched analysis, clustered association matrix, PCA + parallel coordinates, missingness map, LLM narrative. Args: depth (quick|standard|deep), sample_rows (int).
- infer_target       : Auto-select best target column for ML. Args: hint (str, optional).
- mutual_information : MI scores — which features predict the target. Args: none.
- feature_engineering: Encode, generate interactions, select top features. Args: top_k (int).
- train_model        : Compare RF/GBM/LR with 5-fold CV, balanced class weights, PR-AUC for imbalanced data. Args: task_type (auto|classification|regression), models (list), eval_metric (auto|average_precision|roc_auc|f1_macro).
- visualize          : Generate plots. Args: plot_type (distributions|correlation|target_dist|waveform|spectrogram|image_grid).
- execute_code       : Run any Python in the kernel. Args: code (str).
- build_notebook     : Build full .ipynb report and save to WORK_DIR. Args: none.
- read_instructions  : Read a PDF or text file as task instructions. Args: path (str).\
"""

# ── prompts ───────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
/no_think
You are a data science agent. The task instructions have ALREADY been read and understood.
Your job is to produce a precise JSON execution plan based on the extracted task specification below.

Tools available:
{tool_list}

Data context:
{data_context}

Task specification (extracted from instructions — use this to drive your plan):
{brief}

Working directory (save all outputs here): {work_dir}

Rules:
- DO NOT include understand_task in your plan — it has already run.
- DO NOT include read_instructions — the brief is already loaded.
- If task spec says aggregation_needed=true: include aggregate_data with the correct group_by key and target_condition BEFORE feature_engineering and train_model.
- If task spec says target_is_derived=true: aggregate_data must create the target column. Pass target_condition exactly as specified in the task spec.
- If task spec gives evaluation_metric: pass it as eval_metric to train_model (pr_auc→average_precision, roc_auc→roc_auc, f1_macro→f1_macro).
- DEFAULT pipeline (no aggregation):
    eda_profile → quality_check → infer_target → mutual_information →
    feature_engineering → train_model → visualize(distributions) → build_notebook
- AGGREGATION pipeline:
    eda_profile → quality_check →
    aggregate_data(group_by="<key>", target_condition="<expr>", target_column_name="<name>") →
    mutual_information → feature_engineering →
    train_model(eval_metric="<from spec>") → visualize(distributions) → build_notebook
- "EDA only" means: eda_profile → quality_check → visualize(distributions) → visualize(correlation)
- ERROR RECOVERY (user pasted a traceback): use ONLY execute_code steps — diagnose then fix.
- Max {max_steps} steps.

Reply with ONLY valid JSON — no markdown fences, no explanation:
{{"goal": "one sentence", "steps": [{{"step": 1, "tool": "tool_name", "args": {{}}, "reason": "why"}}]}}
"""

_REPLAN_SYSTEM = """\
/no_think
You are a data science agent. You just executed a tool and got the output below.
Decide whether to keep the remaining planned steps or revise them.

Remaining planned steps:
{remaining}

Tool just executed: {last_tool}
Output:
{last_output}

If the remaining steps are still correct, reply: KEEP
If you want to revise, reply with a JSON array of revised steps (same schema as before).
Reply with ONLY "KEEP" or a JSON array — no explanation.
"""

_SYNTH_SYSTEM = """\
/no_think
You are a senior data scientist. Summarise the following analysis findings in clear,
concise, professional language. Cite specific numbers, column names, and model metrics.
No preamble. No filler. Lead with the key insight.
"""

_REACT_SYSTEM = """\
/no_think
You are a senior data scientist assistant with full memory of this session. \
Follow the ReAct format exactly.

Thought: your reasoning about what to do next
Action: the tool name (exactly as listed below)
Action Input: {{"key": "value"}}
Observation: [system provides this after tool execution]
...repeat until done...
Final Answer: your complete response to the user

AVAILABLE TOOLS:
{tool_list}

CURRENT SESSION STATE:
{session_state}

RECENT CONVERSATION (most recent last):
{recent_history}

RULES:
1. Always start with Thought:.
2. Use only the exact tool names listed above — never invent names.
3. Action Input must be valid JSON (use {{}} for empty args).
4. After each Observation, write a new Thought before the next Action.
5. End with Final Answer: when you have enough information to respond.
6. The session state above tells you what task was extracted, what data is loaded, \
and what has been completed — use this as your memory of prior work.
7. Never say "I don't have access to the previous response" — it's in the session state above.
"""


# System prompt for the native function-calling loop. No format scaffolding is
# needed here — the model receives the tools as structured schemas and Ollama
# returns structured tool_calls, so we only give it role, state, and rules.
_TOOLCALL_SYSTEM = """\
You are a senior data scientist assistant with full memory of this session.
You have a set of tools. Decide which tools to call, in what order, to satisfy
the user's request, then write a final answer.

CURRENT SESSION STATE:
{session_state}

RECENT CONVERSATION (most recent last):
{recent_history}

RULES:
1. Call a tool only when it moves the task forward; do not call tools you do not need.
2. Read each tool result before deciding the next call, and adapt: let what a
   result reveals decide the next step instead of following a fixed sequence.
   Chain calls when a later step depends on an earlier one (e.g. understand_task
   before train_model).
3. When the session state shows a TASK FOCUS, prioritise exploring the features
   and relationships most relevant to that outcome after the general checks.
4. When the session state lists REQUIREMENTS TO SATISFY, treat them as a
   checklist. Address every one, using execute_code for anything the standard
   tools do not cover, before you write the final answer.
5. When a tool result is an error, either fix the arguments and retry, or choose
   a different tool. Do not repeat the identical failing call.
6. The session state above is your memory of prior work. Use it and never claim
   you cannot see previous results.
7. When every requirement is addressed and you have enough information, stop
   calling tools and write the final answer as plain text for the user.
8. For exploration, do not run a fixed checklist. If the session state names a
   task or focus (from a brief), let it set the depth and the kind of EDA and
   target the questions it raises; with no task known, do a broad general
   exploration of the whole dataset. Profile first, then READ that result and
   let it choose the next analysis. Go past per-column histograms to
   the relationships that explain the data: group-level aggregates, ratios
   between columns, per-entity comparisons against the peer median, and
   concentration between entities. Each plot or table must answer a specific
   question the previous step raised; if it does not, do not produce it. Use
   execute_code for any relationship the standard tools do not cover.
9. Never state a statistic, describe a plot, or say a file was saved unless a
   tool in THIS turn actually produced it. If you did not run it, do not claim
   it. When something could not run, say so plainly instead of inventing a
   result. A made-up number is worse than a missing one.
10. You cannot see charts; only printed text reaches you. Base every finding
   on printed numbers, and never say what a plot "shows" or "confirms". A file
   written in this turn is a proposal awaiting the user's approval, not saved.
11. When the session state shows an EDA AGENDA, it is your work list. Take the
   OPEN items in order of importance to the brief, and make each step settle
   one of them or a lead a result raised; use the ANSWERED findings to choose
   what to look at next. Stop exploring when no open item can be advanced.
   Do not start feature engineering or modelling unless the request asks for it.
12. The user sees the work plan separately, so do not restate it. Organise the
   final answer around the brief's questions (REQUIREMENTS TO SATISFY): answer
   each one with the numbers found, and say plainly which remain open.
"""


# ── result types ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step: int
    tool: str
    args: dict
    reason: str
    output: ToolOutput
    elapsed_s: float


@dataclass
class AgentRunResult:
    goal: str
    reply: str
    steps: list[StepResult] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str = ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _task_focus(session: Session, kernel: SessionKernel) -> str:
    """A compact statement of what the exploration should serve: the target to
    predict or analyse, the task type, the metric, and the stated goal. Returns
    an empty string when no task is known, so generic exploration is unaffected.
    """
    spec = kernel.namespace.get("TASK_SPEC")
    spec = spec if isinstance(spec, dict) else {}
    target = getattr(session, "target", None) or spec.get("target_column")
    task_type = spec.get("task_type")
    metric = spec.get("evaluation_metric")
    desc = spec.get("task_description") or ""
    parts: list[str] = []
    if target:
        parts.append(f"target `{target}`")
    elif _task_has_no_target(session, kernel):
        parts.append("no labelled target (unlabelled analysis)")
    if task_type and task_type != "auto":
        parts.append(f"task type {task_type}")
    if metric and metric != "auto":
        parts.append(f"optimise {metric}")
    focus = ", ".join(parts)
    if desc:
        focus = (focus + ". " if focus else "") + f"Goal: {desc[:200]}"
    return focus


def _agenda_block(agenda: list[dict]) -> str:
    """The EDA agenda as the model sees it: the open items, and what each
    answered item found, so the next step builds on the results so far
    instead of repeating or drifting."""
    open_items = [a for a in agenda if a.get("status") != "answered"]
    done = [a for a in agenda if a.get("status") == "answered"]
    out = [f"EDA AGENDA ({len(done)} of {len(agenda)} answered). Each step should "
           "settle an OPEN item or follow a lead a result raised:"]
    if open_items:
        out.append("  OPEN:")
        out += [f"    [{a['id']}] ({a.get('source', '')}) {a['question'][:200]}"
                for a in open_items[:14]]
    if done:
        out.append("  ANSWERED (build on these):")
        out += [f"    [{a['id']}] {a['question'][:110]} -> {a.get('finding', '')[:220]}"
                for a in done[-10:]]
    return "\n".join(out)


# Tools that cannot do anything without a supervised target column.
_TARGET_ONLY_TOOLS = frozenset({"infer_target", "mutual_information", "train_model"})


def _task_has_no_target(session: Session, kernel: SessionKernel) -> bool:
    """True once the brief has been read and neither it nor the data gave a
    target, i.e. the task is unlabelled. Offering target-only tools then only
    sends the agent hunting for a label that does not exist."""
    spec = kernel.namespace.get("TASK_SPEC")
    if not isinstance(spec, dict) or not spec:
        return False
    return not (spec.get("target_column") or getattr(session, "target", None)
                or kernel.namespace.get("TARGET"))


def _data_context(session: Session, kernel: SessionKernel,
                  *, compact: bool = False) -> str:
    """Compact description of what's loaded in the kernel.

    `compact` trims the parts a tool-calling agent does not need in every single
    turn — chiefly the long verbatim brief, whose actionable content is already
    carried by the TASK_SPEC and the REQUIREMENTS checklist. It saves roughly a
    thousand tokens per LLM call, which on a local model is the difference
    between a prompt that fits the context window and one that gets truncated
    from the front (taking the system prompt with it). The QA path passes
    compact=False because answering questions about the brief needs its text.
    """
    lines: list[str] = []
    # Task focus and the running exploration log make the agent adaptive and
    # task-directed: it explores what serves the stated outcome, and each turn
    # sees what earlier steps already revealed so it can decide the next step.
    focus = _task_focus(session, kernel)
    if focus:
        lines.append("TASK FOCUS (explore what serves this outcome): " + focus)
    if _task_has_no_target(session, kernel):
        lines.append(
            "NO TARGET: the brief and the data give no labelled outcome column. "
            "This is unlabelled analysis: do not look for or invent a target.")
    plan = kernel.namespace.get("WORK_PLAN")
    agenda = kernel.namespace.get("EDA_AGENDA")
    if isinstance(plan, dict):
        _tq = [q for q in (plan.get("eda_questions_task") or [])
               if isinstance(q, str) and q.strip()]
        _iq = [q for q in (plan.get("eda_questions_independent") or [])
               if isinstance(q, str) and q.strip()]
        lines.append(
            f"WORK PLAN version {plan.get('version', 1)} is already written"
            + (f" (objective: {str(plan.get('objective', ''))[:160]})"
               if plan.get("objective") else "")
            + ". Do not call work_plan again except with revise=true after the "
              "agenda is worked through.")
        if isinstance(agenda, list) and agenda:
            lines.append(_agenda_block(agenda))
        elif _tq or _iq:
            _pl = [f"EDA QUESTIONS FROM THE PLAN (v{plan.get('version', '?')}). "
                   "Answer these, and say what each answer implies:"]
            if _tq:
                _pl.append("  Needed for the brief:")
                _pl += [f"    - {q[:200]}" for q in _tq[:8]]
            if _iq:
                _pl.append("  Independent checks on the data (do these too, "
                           "they are not optional):")
                _pl += [f"    - {q[:200]}" for q in _iq[:8]]
            lines.append("\n".join(_pl))
        agenda_open = isinstance(agenda, list) and any(
            a.get("status") != "answered" for a in agenda)
        # With an agenda, revising is the step AFTER the questions are
        # answered; shown earlier, the nudge made the model revise after a
        # single profile with most of the agenda still open.
        if not plan.get("revised_after_eda") and not agenda_open:
            lines.append(
                "The plan has NOT yet been revised against EDA. Once the "
                "questions above are answered, call work_plan with revise=true "
                "before any feature engineering or model training.")

    log = kernel.namespace.get("EXPLORATION_LOG")
    if isinstance(log, list) and log:
        recent = "\n".join(f"  - {x}" for x in log[-8:])
        lines.append("WHAT WE HAVE LEARNED SO FAR:\n" + recent)
    # Workspace awareness: when a working directory has been mapped, show a
    # compact structure so the agent knows what files exist and can read/load
    # them on demand (read_file / explore_directory / connect_data).
    ws = kernel.namespace.get("WORKSPACE_MAP")
    if isinstance(ws, dict) and ws.get("root"):
        counts = ", ".join(f"{c} {n}" for c, n in (ws.get("counts") or {}).items())
        lines.append(f"WORKING DIRECTORY: {ws['root']}"
                     + (f" ({ws['n_files']} files: {counts})" if counts else ""))
        if not ws.get("n_files"):
            # Say it outright. Given only a bare path, the model invents
            # plausible names (data/, docs/, README.md) and burns the step
            # budget on reads that cannot succeed.
            lines.append(
                "This directory is EMPTY. There are no files to read here, so "
                "do not guess file names. Tell the user the working directory "
                "is empty and ask them to check the path.")
        tree = (ws.get("tree") or "").splitlines()
        if tree:
            _n = 12 if compact else 25
            shown = "\n".join(tree[:_n])
            more = f"\n  … +{len(tree) - _n} more" if len(tree) > _n else ""
            lines.append("Structure:\n" + shown + more)
        for ds in (ws.get("data_schemas") or [])[:5]:
            lines.append(f"  {ds['name']}: {ds['n_cols']} cols — {ds['columns'][:15]}")
    data_type = kernel.namespace.get("DATA_TYPE")
    data_fmt = kernel.namespace.get("DATA_FMT")
    if data_type:
        lines.append(f"data_type: {data_type} | format: {data_fmt}")
    df = kernel.namespace.get("df")
    if df is not None:
        try:
            # Say plainly that these are live variables, not a description of
            # files on disk. Read as metadata, the agent re-loads data that is
            # already in memory: it wrote pd.read_csv with guessed filenames,
            # failed, and reported "no data loaded yet" while df sat in the
            # namespace.
            lines.append(
                "DATA ALREADY LOADED IN THE PYTHON KERNEL. These are live "
                "variables: use them directly in execute_code. Do NOT read the "
                "CSVs again.")
            lines.append(f"df.shape: {df.shape}")
            lines.append(f"df.columns: {list(df.columns[:30])}")
            null_pct = (df.isna().mean() * 100).round(1)
            high_null = null_pct[null_pct > 10]
            if not high_null.empty:
                lines.append(f"High-null (>10%): {high_null.to_dict()}")
        except Exception:
            pass
    elif session.data_path:
        lines.append(f"data_path: {session.data_path} (not yet loaded into kernel)")
    else:
        lines.append("No data loaded.")
    # List any secondary data files already loaded
    for i in range(2, 10):
        sdf = kernel.namespace.get(f"df{i}")
        if sdf is None:
            break
        try:
            lines.append(f"df{i}.shape: {sdf.shape} — {list(sdf.columns[:15])}")
        except Exception:
            pass
    if session.target:
        lines.append(f"TARGET: {session.target}")
    # Include extracted task spec when available — critical for memory
    task_spec = kernel.namespace.get("TASK_SPEC")
    if task_spec and isinstance(task_spec, dict):
        ts_lines = ["Task spec (already extracted):"]
        for key in ("task_type", "target_column", "evaluation_metric",
                    "aggregation_needed", "aggregation_key", "task_description"):
            val = task_spec.get(key)
            if val not in (None, "", False):
                ts_lines.append(f"  {key}: {val}")
        lines.append("\n".join(ts_lines))
        # Surface the concrete instructions from the brief as a checklist. These
        # are short and actionable, so they stay in context even when the raw
        # brief is long, and the agent is told to address every one.
        _reqs = [r for r in (task_spec.get("specific_requirements") or [])
                 if isinstance(r, str) and r.strip()
                 and not r.strip().startswith("<")]
        if _reqs:
            req_lines = ["REQUIREMENTS TO SATISFY (from the brief; address every "
                         "one before finishing):"]
            req_lines += [f"  - {r[:200]}" for r in _reqs[:12]]
            lines.append("\n".join(req_lines))
    if session.artifacts:
        lines.append(f"Completed artifacts: {sorted(session.artifacts.keys())}")
    if session.notebook_path:
        lines.append(f"Notebook: {session.notebook_path}")
    # Include brief content so LLM can answer questions about the task
    # without needing to run understand_task first.
    _brief = getattr(session, "brief", None)
    if _brief and _brief.strip():
        _fnames = getattr(session, "brief_filenames", [])
        _label = ", ".join(_fnames) if _fnames else "yes"
        # Include a substantial excerpt of the brief text. The concrete
        # instructions are also surfaced separately as the REQUIREMENTS checklist
        # above, so a long brief does not lose its actionable parts here. In
        # compact mode the checklist carries them and the excerpt shrinks to a
        # short reminder, because repeating the whole brief on every tool call
        # is what pushes the prompt past the context window.
        _budget = 600 if compact else 4000
        _brief_excerpt = _brief[:_budget]
        if len(_brief) > _budget:
            _brief_excerpt += f"\n... [{len(_brief) - _budget:,} more chars"
            _brief_excerpt += (", call read_instructions for the full text]"
                               if compact else "]")
        lines.append(
            f"BRIEF ({_label}):\n{_brief_excerpt}"
        )
        if not task_spec:
            lines.append("(call understand_task to extract a structured task spec)")

    # Include last pipeline metrics/output for result questions
    _last_metrics = kernel.namespace.get("LAST_METRICS") or kernel.namespace.get("metrics")
    if _last_metrics:
        lines.append(f"LAST METRICS: {_last_metrics}")

    return "\n".join(lines) or "No data loaded."


_SPECIAL_TOKENS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|end_of_text|>", "<|eot_id|>",
)


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    elif "</think>" in cleaned:
        # A closing tag with no opener: the model started reasoning straight
        # away and only marked where it stopped. Everything before the tag is
        # that reasoning, and without this it is shown to the user as the answer.
        cleaned = cleaned.split("</think>", 1)[1]
    # Strip Qwen/Llama chat-template tokens that leak into responses
    for tok in _SPECIAL_TOKENS:
        idx = cleaned.find(tok)
        if idx != -1:
            cleaned = cleaned[:idx]
    cleaned = _deduplicate_repetition(cleaned)
    return cleaned.strip()


def _deduplicate_repetition(text: str) -> str:
    """Truncate at the point where the model starts repeating itself.

    When local models run out of meaningful content they often loop,
    repeating the same fragment. Two loop shapes are handled:
      1. A short fragment (any length) repeated 3+ times *consecutively*
         — e.g. "PR-AUC.\n\nPR-AUC.\n\nPR-AUC." — cut after the 1st.
      2. A longer sentence (>= 20 chars) that appears 3+ times anywhere.
    """
    # ── Shape 1: consecutive duplicate fragments (catches short-token loops) ──
    frags = re.split(r"(?<=[.!?\n])\s+", text)
    prev_key: str | None = None
    run_len = 0
    run_first_start = 0
    consumed = 0  # chars of `text` covered so far
    for frag in frags:
        key = frag.strip().lower()
        if key and key == prev_key:
            run_len += 1
            if run_len >= 3:
                # Keep the 1st occurrence, drop the rest of the run.
                return text[:run_first_start + len(frag)].strip()
        elif key:
            prev_key = key
            run_len = 1
            run_first_start = text.find(frag, consumed)
        else:
            prev_key = None
            run_len = 0
        consumed += len(frag) + 1

    # ── Shape 2: long sentence repeated 3+ times anywhere ──
    seen: dict[str, int] = {}
    pos = 0
    for sent in frags:
        key = sent.strip().lower()
        if len(key) >= 20:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                cut = text.find(sent, pos)
                if cut > 0:
                    return text[:cut].strip()
        pos += len(sent) + 1
    return text


def _trim_echoed_history(reply: str, session: Session) -> str:
    """Cut a trailing block that verbatim-echoes a prior assistant turn.

    The local model sometimes appends a large chunk of a previous assistant
    message after its real answer — context bleed from the multi-turn history
    fed to /api/chat. `_deduplicate_repetition` misses it because the block
    appears only once (not 3+ times). Detect it: if the opening ~60 chars of
    any earlier assistant reply reappear later in the new text, that later
    span is an echo — truncate there and keep the fresh answer before it.
    """
    if not reply:
        return reply

    def _content(m: Any) -> str:
        # ChatMessage stores the body in `.text`; some callers use `.content`.
        return getattr(m, "text", None) or getattr(m, "content", None) or ""

    prior = [
        _content(m) for m in getattr(session, "messages", [])
        if getattr(m, "role", "") == "assistant" and len(_content(m).strip()) >= 80
    ]
    cut = len(reply)
    for content in prior:
        head = content.strip()[:60]
        idx = reply.find(head)
        # idx > 0: real answer precedes the echo → cut it off.
        # idx == 0 means the whole reply is the echo; nothing better to keep.
        if idx > 0:
            cut = min(cut, idx)
    trimmed = reply[:cut].strip()
    return trimmed if trimmed else reply


def _extract_json(text: str) -> Any:
    """Find and parse the first {...} or [...] block in LLM output."""
    text = _strip_thinking(text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first JSON block
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _parse_plan(raw: str, max_steps: int) -> list[dict]:
    """Parse LLM planner output → list of step dicts."""
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        steps = obj.get("steps", [])
    elif isinstance(obj, list):
        steps = obj
    else:
        return []
    valid: list[dict] = []
    for s in steps[:max_steps]:
        if isinstance(s, dict) and s.get("tool") in TOOL_REGISTRY:
            valid.append({
                "step": s.get("step", len(valid) + 1),
                "tool": s["tool"],
                "args": s.get("args", {}),
                "reason": s.get("reason", ""),
            })
    return valid


def _is_error_message(instruction: str) -> bool:
    """Return True if the instruction looks like a Python error/traceback."""
    error_signals = [
        "Traceback (most recent call last)",
        "ValueError:", "TypeError:", "AttributeError:", "KeyError:",
        "IndexError:", "NameError:", "RuntimeError:", "ImportError:",
        "Cell In[", "----->", "File \"", ".py\", line ",
    ]
    return any(sig in instruction for sig in error_signals)


_ACT_OR_ANSWER_PROMPT = """\
/no_think
Decide whether this message can be answered from what is already known, or
whether it needs work doing.

WHAT IS ALREADY KNOWN (the session so far):
{state}

WHAT THIS AGENT CAN DO IF IT ACTS:
{tools}

MESSAGE FROM THE USER:
{message}

"answer" means everything needed is in the session state above and replying is
enough. Examples: what columns are there, what does the brief say, what were
the results, explain that again, thanks.

"act" means the message asks for something that the state above does not
already contain: something produced, computed, changed, looked up, or found
out. Anything that should leave a file, a model, a figure, a plan or a new
finding behind is "act". If a reply would have to DESCRIBE doing the work
instead of doing it, the answer is "act".

Reply with only this JSON: {{"decision": "answer" | "act", "why": "under 12 words"}}
"""


def _decide_action(instruction: str, session: Session, kernel: SessionKernel,
                   llm_client: Any) -> tuple[bool, str]:
    """Should this message be acted on, or just answered? Returns (act, why).

    The model makes this call knowing what tools exist and what the session
    already holds, because "do I need a tool?" cannot be answered by anything
    that has not been shown the tools. It replaces a topic classifier whose six
    labels collapsed into this one bit, so it costs no extra round trip.

    Two different failures, two different fallbacks. A reply that cannot be
    parsed means the model is working but was unclear, so act: the tool loop can
    still answer a question without calling anything, whereas answering a
    request to DO something only describes it, which is the failure a user
    notices. A raised exception means the model is unreachable, and the tool
    loop needs that same model, so fall back to answering and let the reply path
    use its own offline handling.
    """
    try:
        raw = llm_client._generate(
            _ACT_OR_ANSWER_PROMPT.format(
                state=_data_context(session, kernel, compact=True)[:1500],
                tools=", ".join(sorted(TOOL_REGISTRY)),
                message=instruction[:600]),
            temperature=0.0, max_tokens=_ROUTER_TOKENS, json_mode=True)
    except Exception:  # noqa: BLE001
        return False, "model unreachable; answering from session context"
    data = _extract_json(_strip_thinking(raw))
    if isinstance(data, dict) and data.get("decision") in ("answer", "act"):
        return data["decision"] == "act", str(data.get("why", ""))[:80]
    return True, "unclear request; acting is the safer default"


_EXPLORATION_PROMPT = """\
Decide whether the user is asking for open-ended DATA ANALYSIS or EXPLORATION of
the dataset that is already loaded — understanding the data, profiling it,
finding patterns, relationships, distributions, group differences, anomalies,
data-quality issues, or insights — phrased in ANY way, not only with the words
"EDA" or "explore".

TRUE examples (regardless of wording): "dig into these claims", "what is going
on in this data", "give the dataset a thorough look", "find anything unusual",
"analyse the data", "profile it", "what stands out", "understand the data",
"look for fraud signals in the data".

FALSE examples: train or evaluate a specific named model, build the notebook or
report, deploy or package a model, read the brief, or a factual question whose
answer is already in the session state (e.g. "how many rows", "what columns").

MESSAGE FROM THE USER:
{message}

Reply with only this JSON: {{"exploration": true|false}}
"""


def _is_exploration_request(instruction: str, session: Session,
                            kernel: SessionKernel, llm_client: Any) -> bool:
    """LLM decision: is this a request for data analysis / exploration?

    Routing must not be bound to a fixed keyword list — the user can ask for a
    "deeper EDA and complete analysis" in countless phrasings. The model reads
    the actual request and decides. Fail CLOSED: on any error or unparseable
    reply return False so the caller falls back to the explicit keyword routes,
    rather than sending a non-analysis request into the exploration loop.
    """
    if llm_client is None:
        return False
    try:
        raw = llm_client._generate(
            _EXPLORATION_PROMPT.format(message=instruction[:600]),
            temperature=0.0, max_tokens=_ROUTER_TOKENS, json_mode=True)
    except Exception:  # noqa: BLE001
        return False
    data = _extract_json(_strip_thinking(raw))
    return bool(isinstance(data, dict) and data.get("exploration") is True)


def _is_keyword_confident(instruction: str) -> bool:
    """Return True if keywords clearly indicate what to do — skip LLM planner."""
    # Error messages are always handled confidently via keyword plan
    if _is_error_message(instruction):
        return True
    lo = instruction.lower()
    strong_signals = [
        "pipeline", "full pipeline", "complete", "implement",
        "eda", "explore", "analyse", "analyze",
        "train", "model", "classify", "regression", "predict",
        "notebook", "ipynb", "build", "generate report",
        "feature", "mutual information", "visuali",
    ]
    # Also confident if instruction is long (user wrote detailed instructions)
    matches = sum(1 for s in strong_signals if s in lo)
    return matches >= 2 or len(instruction) > 80


def _keyword_goal(instruction: str) -> str:
    """Derive a short goal description from instruction keywords."""
    lo = instruction.lower()
    if "pipeline" in lo or ("model" in lo and "notebook" in lo):
        return "Full DS pipeline: EDA → target selection → feature engineering → model training → notebook"
    if "eda" in lo and "model" in lo:
        return "EDA + ML model training and evaluation"
    if "eda" in lo or "explore" in lo or "analyse" in lo:
        return "Exploratory data analysis"
    if "model" in lo or "train" in lo:
        return "ML model training and evaluation"
    if "notebook" in lo or "ipynb" in lo or "build" in lo:
        return "Build complete analysis notebook"
    return instruction[:100]


def _is_real_plan(steps: list[dict]) -> bool:
    """Reject LLM plans that are just a single trivial execute_code."""
    if not steps:
        return False
    if len(steps) == 1 and steps[0]["tool"] == "execute_code":
        code = steps[0].get("args", {}).get("code", "")
        # Trivial if it's just print statements
        if len(code) < 200 and "print(" in code and "df." not in code:
            return False
    return True


_TOOL_ALIASES: dict[str, str] = {
    # PDF / document reading
    "read_file": "understand_task",
    "read_pdf": "understand_task",
    "read_document": "understand_task",
    "read_brief": "understand_task",
    "load_pdf": "understand_task",
    "load_brief": "understand_task",
    "parse_pdf": "understand_task",
    "extract_pdf": "understand_task",
    "open_file": "understand_task",
    "open_pdf": "understand_task",
    "read_instructions_file": "read_instructions",
    "load_instructions": "read_instructions",
    # Data / EDA aliases
    "profile_data": "eda_profile",
    "load_data": "eda_profile",
    "explore_data": "eda_profile",
    "data_profile": "eda_profile",
    "run_eda": "eda_profile",
    "data_quality": "quality_check",
    "check_quality": "quality_check",
    "feature_selection": "feature_engineering",
    "engineer_features": "feature_engineering",
    # Model training aliases
    "train": "train_model",
    "fit_model": "train_model",
    "run_model": "train_model",
    "ml_pipeline": "train_model",
    "machine_learning": "train_model",
    # Misc
    "plot": "visualize",
    "visualise": "visualize",
    "run_code": "execute_code",
    "python": "execute_code",
    "exec_code": "execute_code",
    "notebook": "build_notebook",
    "generate_notebook": "build_notebook",
}


def _resolve_tool_name(name: str) -> str:
    """Map LLM-hallucinated tool names to actual registry names."""
    if name in TOOL_REGISTRY:
        return name
    return _TOOL_ALIASES.get(name, name)


def _parse_react_action(text: str) -> dict | None:
    """Parse the LLM's ReAct action — hybrid natural-language + JSON fallback.

    Priority order:
      1. CALL_TOOL: <name> / ARGS: {...}  ← natural-language marker (primary)
      2. {"action": "tool_call", ...}     ← canonical JSON (fallback)
      3. {"action": "respond", ...}       ← canonical JSON (fallback)
      4. Many JSON schema variants        ← compat with older format
      5. Plain prose                      ← treated as respond

    Returns None only when the text is truly empty or structureless.
    """
    text = _strip_thinking(text).strip()

    # ── 1. CALL_TOOL: marker (natural-language hybrid format) ───────────────
    _ct = re.search(
        r'CALL_TOOL:\s*(\S+)(?:\s*\nARGS:\s*(\{.*?\}))?',
        text, re.DOTALL | re.IGNORECASE,
    )
    if _ct:
        tool_name = _ct.group(1).strip()
        args_raw = (_ct.group(2) or "").strip()
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {}
        return {"action": "tool_call", "tool": tool_name, "args": args}

    obj = _extract_json(text)

    if isinstance(obj, dict):
        action = obj.get("action", "")

        # ── Canonical schema ─────────────────────────────────────────────────
        if action == "tool_call":
            return obj
        if action == "respond":
            return obj

        # ── Model omitted "action" but gave a "tool" key → tool_call ────────
        if not action and "tool" in obj and isinstance(obj.get("tool"), str):
            return {
                "action": "tool_call",
                "tool": obj["tool"],
                "args": obj.get("args") or obj.get("arguments") or obj.get("input") or {},
            }

        # ── Model used synonyms for "tool_call" ──────────────────────────────
        if action in ("function_call", "use_tool", "call_tool", "tool_use"):
            tool = obj.get("tool") or obj.get("name") or obj.get("function", "")
            args = obj.get("args") or obj.get("arguments") or obj.get("input") or {}
            if tool:
                return {"action": "tool_call", "tool": tool, "args": args}

        # ── Model used synonyms for "respond" ────────────────────────────────
        if action in ("response", "answer", "reply", "message", "text"):
            for key in ("text", "message", "response", "answer", "reply", "content"):
                if key in obj:
                    return {"action": "respond", "text": str(obj[key])}

        # ── No "action" key but has a text-like key → respond ───────────────
        if not action:
            for key in ("text", "message", "response", "answer", "reply", "content"):
                if key in obj and isinstance(obj[key], str) and len(obj[key]) > 5:
                    return {"action": "respond", "text": obj[key]}

        # ── Fallback: stringify the whole JSON as a respond ──────────────────
        stringified = str(obj)
        if len(stringified) > 20:
            return {"action": "respond", "text": stringified}

    # ── Non-JSON prose → treat as a direct text reply ───────────────────────
    if len(text) > 10 and not text.startswith("{"):
        return {"action": "respond", "text": text}

    return None


# ── Thought/Action/Observation ReAct parser ───────────────────────────────────

_ACTION_RE  = re.compile(r"Action\s*:\s*(.+)",                                    re.IGNORECASE)
_INPUT_RE   = re.compile(r"Action\s+Input\s*:\s*(\{.*?\})",                       re.IGNORECASE | re.DOTALL)
_FINAL_RE   = re.compile(r"Final\s+Answer\s*:\s*(.+)",                            re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?=Action|Final Answer|$)",         re.IGNORECASE | re.DOTALL)


@dataclass
class _ReactStep:
    thought: str
    action: str | None
    action_input: dict
    final_answer: str | None
    parse_error: str | None


def _parse_react_step(text: str) -> _ReactStep:
    """Parse one ReAct step from model output (think-tags already stripped)."""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = _FINAL_RE.search(text)
    if final_m:
        return _ReactStep(thought=thought, action=None, action_input={},
                          final_answer=final_m.group(1).strip(), parse_error=None)

    action_m = _ACTION_RE.search(text)
    if not action_m:
        return _ReactStep(thought=thought, action=None, action_input={},
                          final_answer=text.strip() or None,
                          parse_error="no Action or Final Answer found")

    action = action_m.group(1).strip()
    action_input: dict = {}
    parse_error: str | None = None
    input_m = _INPUT_RE.search(text)
    if input_m:
        try:
            action_input = json.loads(input_m.group(1))
        except json.JSONDecodeError as exc:
            json_m = re.search(r"\{.*\}", input_m.group(1), re.DOTALL)
            recovered = False
            if json_m:
                try:
                    action_input = json.loads(json_m.group())
                    recovered = True
                except json.JSONDecodeError:
                    pass
            if not recovered:
                # Surface the failure so the loop can ask the model to resend
                # valid JSON, instead of silently running the tool with {}.
                parse_error = (
                    f"Action Input was not valid JSON ({exc.msg}). "
                    f"Got: {input_m.group(1)[:200]}"
                )
    elif re.search(r"Action\s+Input\s*:", text, re.IGNORECASE):
        # An Action Input line exists but no {...} braces matched.
        parse_error = ("Action Input present but no JSON object found. "
                       "Provide args as a JSON object, e.g. {\"key\": \"value\"}.")

    return _ReactStep(thought=thought, action=action, action_input=action_input,
                      final_answer=None, parse_error=parse_error)


def _format_tool_line(name: str, tool: Any) -> str:
    """One line per tool for the ReAct system prompt: name, description, and
    the arg names from its input_schema so the model stops guessing arg keys.
    """
    desc = (getattr(tool, "description", "") or "")[:80]
    schema = getattr(tool, "input_schema", None) or {}
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
    if props:
        arg_parts = [
            f"{k}{'*' if k in required else ''}:{v.get('type', 'any')}"
            for k, v in list(props.items())[:8]
        ]
        args_str = f"  args({', '.join(arg_parts)})"
    else:
        args_str = "  args(none)"
    return f"  {name}: {desc}\n{args_str}"


def _build_ollama_tool_schemas(names: list[str] | None = None) -> list[dict]:
    """Convert the TOOL_REGISTRY into Ollama native function-call schemas.

    Each entry is {"type": "function", "function": {name, description,
    parameters}} where parameters is the tool's JSON-Schema input_schema.
    Pass `names` to expose only a subset (used to give each specialist agent a
    focused tool space).
    """
    items = (TOOL_REGISTRY.items() if names is None
             else [(n, TOOL_REGISTRY[n]) for n in names if n in TOOL_REGISTRY])
    schemas: list[dict] = []
    for name, tool in items:
        params = getattr(tool, "input_schema", None)
        if not (isinstance(params, dict) and params.get("type")):
            params = {"type": "object", "properties": {}, "required": []}
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (getattr(tool, "description", "") or "")[:1024],
                "parameters": params,
            },
        })
    return schemas


def _coerce_tool_args(raw: Any) -> dict:
    """Ollama usually returns arguments as a dict; some builds send a JSON
    string. Normalise to a dict, tolerating malformed payloads."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


_CRITIC_PROMPT = """\
/no_think
You are a data scientist reviewing the result of one step. Read what the result
reveals, judge it against the goal and the task focus, and decide the next step,
the way a person exploring data adapts after seeing each output.

USER GOAL:
{goal}

TASK FOCUS:
{task_focus}

OPEN EDA AGENDA ITEMS:
{agenda}

TOOL CALLED: {tool}
ARGUMENTS: {args}
RESULT (truncated):
{observation}

Reply with a single JSON object and nothing else:
{{"verdict": "accept" | "retry" | "replan", "goal_met": true | false,
  "finding": "one sentence on what this result reveals, quoting its key numbers",
  "answers": ["ids of the open agenda items this result settles, or none"],
  "new_question": "one follow-up question this result raises that is worth answering, or empty",
  "next_step": "one sentence on the best next step given this result",
  "reason": "at most 15 words"}}

Definitions:
- accept: the result is valid and useful; continue.
- retry: the result is wrong, empty, or off-target, but the same tool could work with different arguments.
- replan: this tool or approach cannot satisfy the goal; a different strategy is needed.
- goal_met: true only if no further tools are needed and the goal is now fully answerable.

Ground finding and next_step in the actual numbers in the result. When a task
focus is given, prefer next steps that explore the features and relationships
most relevant to that outcome. List an agenda item in answers only when the
result actually settles it. Ask a new_question only when the result shows
something unexpected (an outlier entity, a skew, a relationship) that no open
item covers.
"""


@dataclass
class Critique:
    verdict: str        # "accept" | "retry" | "replan"
    reason: str
    goal_met: bool
    finding: str = ""      # what the result revealed about the data or task
    next_step: str = ""    # the recommended next step, grounded in the result
    answers: list[str] = field(default_factory=list)   # agenda ids it settles
    new_question: str = ""  # a follow-up the result raised


def _call_signature(tool_name: str, args: dict) -> str:
    """Stable key for a (tool, args) pair so repeated identical calls are
    detected and escalated instead of looping forever."""
    try:
        return f"{tool_name}:{json.dumps(args, sort_keys=True)[:200]}"
    except (TypeError, ValueError):
        return f"{tool_name}:{str(args)[:200]}"


def _error_headline(err: str, limit: int = 220) -> str:
    """The line of a traceback that says what actually went wrong.

    A traceback opens with "Traceback (most recent call last):" and the frame
    that raised, and closes with the exception. Truncating from the front, as
    this used to, showed the user the header and nothing else: every failure
    read as `Traceback (most recent call last): File ".../kernel.py", line 163,
    in _work exec(`, which is the same text whatever went wrong.
    """
    lines = [ln.strip() for ln in (err or "").strip().splitlines() if ln.strip()]
    if not lines:
        return "(no error text)"
    last = lines[-1]
    # A SyntaxError prints the offending source and a caret before the message;
    # keep the message, which is the line that names the problem.
    if len(lines) > 1 and last.startswith("^"):
        last = lines[-2]
    return last[:limit]


def _is_timeout(exc: BaseException) -> bool:
    """True when a request ran out of time rather than failing outright.

    The loops retry a failed LLM call once, which is right for a refused
    connection but wrong for a timeout: the retry waits the full timeout again
    and usually ends the same way, so a single slow call costs twice the budget.
    Matching on the class name covers httpx, socket, and any wrapper without
    importing them here.
    """
    return "timeout" in type(exc).__name__.lower()


def _critique_step(
    goal: str,
    tool_name: str,
    args: dict,
    observation: str,
    success: bool,
    *,
    llm_client: Any,
    repeats: int,
    task_focus: str = "",
    agenda: list[dict] | None = None,
) -> Critique:
    """Reflexion-style judgement of one tool result.

    Besides the accept/retry/replan verdict, the critic reads what the result
    reveals (finding) and recommends the next step (next_step), so the loop can
    re-plan from each output. Rule-based fast paths handle the obvious cases
    (error / empty) without an LLM call; genuine successes are judged by a
    short, constrained LLM call. Fails OPEN: any error or unparseable reply
    yields "accept" so a flaky critic can never stall real progress.
    """
    obs = observation or ""
    # Repeated identical failures escalate from retry to replan.
    _escalate = "replan" if repeats >= _MAX_RETRIES_PER_STEP else "retry"

    if not success or obs.strip().upper().startswith("ERROR"):
        return Critique(_escalate, "tool returned an error", False)
    if not obs.strip():
        return Critique(_escalate, "empty result", False)

    # Ambiguous success → ask the model, but never block on it.
    try:
        # A standalone prompt on /api/generate, deliberately. Asking inside the
        # loop's own conversation was measured slower (50.8 s against 21.5 s),
        # because the critic then reprocesses the whole conversation, and it
        # bought nothing: Ollama keeps more than one cache slot, so a separate
        # critic call does not cost the loop its cached prefix (the next loop
        # turn after one still evaluated its prompt in 0.46 s).
        raw = llm_client._generate(
            _CRITIC_PROMPT.format(
                goal=goal, task_focus=(task_focus or "(none given)"),
                tool=tool_name, args=json.dumps(args)[:300],
                agenda=("\n".join(f"[{a['id']}] {a['question'][:200]}"
                                   for a in (agenda or [])
                                   if a.get("status") != "answered")[:2500]
                        or "(none)"),
                observation=obs[:2500]),
            temperature=0.0, max_tokens=_CRITIC_TOKENS, json_mode=True,
        )
        data = _extract_json(_strip_thinking(raw))
        if isinstance(data, dict) and data.get("verdict") in (
                "accept", "retry", "replan"):
            verdict = data["verdict"]
            # Guard: don't let the critic demand a retry past the budget.
            if verdict == "retry" and repeats >= _MAX_RETRIES_PER_STEP:
                verdict = "replan"
            return Critique(
                verdict, str(data.get("reason", ""))[:200],
                bool(data.get("goal_met", False)),
                finding=str(data.get("finding", ""))[:400],
                next_step=str(data.get("next_step", ""))[:280],
                answers=[str(i) for i in (data.get("answers") or [])
                         if isinstance(i, (str, int))][:6]
                if isinstance(data.get("answers"), list) else [],
                new_question=str(data.get("new_question") or "")[:300])
    except Exception:  # noqa: BLE001
        pass
    return Critique("accept", "", False)


def _execute_tool(name, args, *, session, kernel, llm_client):
    """Validate schema and semantic constraints at the execution boundary."""
    from jsonschema import validate

    from agent.api.intent import check_action
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return ToolOutput(False, "", error="Unknown tool")
    if getattr(kernel, "remaining_tool_calls", 1) <= 0 or time.monotonic() > getattr(kernel, "turn_deadline", float("inf")):
        return ToolOutput(False, "", error="Turn execution budget exhausted.")
    if hasattr(kernel, "remaining_tool_calls"):
        kernel.remaining_tool_calls -= 1
    try:
        validate(args, getattr(tool, "input_schema", {"type": "object"}))
    except Exception as exc:
        return ToolOutput(False, "", error=f"Invalid tool arguments: {exc}")
    intent = getattr(kernel, "task_intent", None)
    if intent is not None:
        if name not in intent.allowed_tools:
            # Mechanical boundary, applied to every tool: a capability the
            # interpreted request did not authorise never runs.
            return ToolOutput(False, "", error="This tool is outside the interpreted request.")
        if _needs_semantic_check(name):
            ok, reason = check_action(intent, instruction=kernel.task_instruction,
                                     brief=session.brief or "", name=name, args=args,
                                     llm=llm_client)
            if not ok:
                return ToolOutput(False, "", error=reason)
    return tool.execute(args, session=session, kernel=kernel, llm_client=llm_client)


# Tools whose effects are read-only or confined: they read files or data,
# compute in the host kernel without writing, or run in the isolated sandbox
# (no network, no host writes; any file it makes is only a staged proposal).
# For these the per-call LLM permission check bought no protection that the
# mechanical allowed_tools check, the sandbox and the approval queue do not
# already give, and it cost 9-180 s per call on the local model — enough that
# a 12-step exploration ran past the 900 s turn deadline. Every other tool
# (training, writing, deploying, external data, web) is still checked on each
# call. DSAGENT_CHECK_ALL_ACTIONS=1 restores checking every call.
_READ_ONLY_TOOLS = frozenset({
    "execute_code", "eda_profile", "deep_eda", "quality_check",
    "mutual_information", "visualize", "read_file", "read_instructions",
    "search_files", "explore_directory", "understand_task",
    "data_analysis_agent",
})


def _needs_semantic_check(name: str) -> bool:
    if os.environ.get("DSAGENT_CHECK_ALL_ACTIONS", "0") == "1":
        return True
    return name not in _READ_ONLY_TOOLS


def _grounded_reply(draft, steps, llm_client):
    """Keep the model's interpretation, but only what the evidence supports.

    Every number and file name in the draft must appear in the output of a
    tool that completed this turn, at the precision the draft states it (so
    "4.2" matches 4.211353 and "88%" matches 0.8755). A statement with an
    unsupported figure or file is removed deterministically; no model judges
    its own claims. Interpretation that states no figure is kept, which is
    what the user asked for when they asked "why". Failed steps are summarised
    in one line instead of being shown as tracebacks. If nothing supported
    remains, the completed tool outputs themselves are shown.
    """
    if not steps:
        return "No computation completed this turn. No results are available to report."
    ok_steps = [s for s in steps if s.output.success]
    failed = [s for s in steps if not s.output.success]
    evidence = "\n".join((s.output.text or "") for s in ok_steps)
    ev_numbers = [_as_number(m) for m in _NUMBER_RE.findall(evidence)]
    ev_numbers = [v for v in ev_numbers if v is not None]
    ev_lower = evidence.lower()

    kept_lines: list[str] = []
    removed = 0
    for line in (draft or "").splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= set("|-: "):
            kept_lines.append(line)
            continue
        whole = stripped.startswith(("|", "-", "*", "#")) or bool(re.match(r"\d+\.\s", stripped))
        units = [line] if whole else re.split(r"(?<=[.!?])\s+(?=[A-Z*(\[])", line)
        kept_units = []
        for unit in units:
            if _unit_supported(unit, ev_numbers, ev_lower):
                kept_units.append(unit)
            else:
                removed += 1
        if kept_units:
            kept_lines.append(" ".join(kept_units))
    narrative = re.sub(r"\n{3,}", "\n\n", _drop_empty_tables(kept_lines)).strip()

    evidence_block = "\n\n".join(
        f"[E{i + 1}] {s.tool}:\n{(s.output.text or '')[:1500]}"
        for i, s in enumerate(ok_steps))[:6000]
    if not re.search(r"[A-Za-z]{3}", narrative):
        narrative = evidence_block or "No analysis step completed this turn."
    elif evidence_block and not any(
            _is_metric(t) for t in _NUMBER_RE.findall(narrative)):
        # The text cites no computed figure ("Analysis complete."), so show
        # the results it rests on rather than leave the user without them.
        narrative += "\n\nEvidence:\n" + evidence_block

    notes = []
    if ok_steps:
        notes.append(f"Based on {len(ok_steps)} completed step(s): "
                     + ", ".join(dict.fromkeys(s.tool for s in ok_steps)) + ".")
    if failed:
        heads = "; ".join(dict.fromkeys(_error_headline(s.output.error, 80) for s in failed))
        notes.append(f"{len(failed)} step(s) failed and were not used: {heads}")
    if removed:
        notes.append(f"Removed {removed} statement(s) whose figures or files did not "
                     "appear in any completed tool output.")
    staged = [f for s in ok_steps for f in (s.output.artifacts or {}).get("scratch_outputs", [])]
    if staged:
        notes.append("Files produced but NOT yet in your folder (approve them under "
                     "Pending changes to save): " + ", ".join(dict.fromkeys(staged)) + ".")
    return narrative + ("\n\n---\n" + "\n".join(notes) if notes else "")


# A number as written in prose or tool output: 7,733 / 637263.45 / -1.7 / 47%.
# Not part of an identifier, so PRV020 or claim_id2 do not count as figures.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9_])")
_FILE_RE = re.compile(r"\b[\w\-]+\.(?:csv|png|xlsx|parquet|json|pkl|joblib|ipynb|html|pdf|txt|md)\b",
                      re.IGNORECASE)


def _drop_empty_tables(lines: list[str]) -> str:
    """Remove a markdown table left with a header and no rows after filtering."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_sep = bool(nxt) and nxt.startswith("|") and set(nxt) <= set("|-: ")
        if line.startswith("|") and is_sep:
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            if j > i + 2:
                out.extend(lines[i:j])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _as_number(token: str) -> float | None:
    try:
        return float(token.rstrip("%").replace(",", ""))
    except ValueError:
        return None


def _is_metric(token: str) -> bool:
    """A figure worth checking: not a small count like "top 5" or "2 providers"."""
    value = _as_number(token)
    if value is None:
        return False
    return not (value.is_integer() and abs(value) <= 20
                and "%" not in token and "." not in token)


def _figure_supported(token: str, ev_numbers: list[float]) -> bool:
    """True when some evidence value rounds to the figure as stated."""
    value = _as_number(token)
    if value is None or not _is_metric(token):
        return True
    digits = token.rstrip("%").replace(",", "")
    places = len(digits.split(".")[1]) if "." in digits else 0
    candidates = ev_numbers + ([e * 100 for e in ev_numbers] if token.endswith("%") else [])
    return any(round(e, places) == round(value, places) for e in candidates)


def _unit_supported(unit: str, ev_numbers: list[float], ev_lower: str) -> bool:
    if any(not _figure_supported(t, ev_numbers) for t in _NUMBER_RE.findall(unit)):
        return False
    return all(f.lower() in ev_lower for f in _FILE_RE.findall(unit))


def _tool_call_loop(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    max_steps: int = 8,
    emit: Any = None,
    tool_names: list[str] | None = None,
    system_prompt: str | None = None,
    prior_experience: str = "",
) -> tuple[str, list[StepResult], list[str], bool]:
    """Agentic loop driven by native Ollama function-calling.

    The model receives the tools as structured schemas and returns structured
    tool_calls. We execute each call against the session kernel, feed the result
    back as a `tool` message, and repeat until the model answers with plain text
    or the step budget is spent.

    `tool_names` restricts the exposed tools to a subset (a specialist's toolset);
    `system_prompt` overrides the default role instructions. Both default to the
    full registry and the generic data-scientist prompt.

    Returns (reply, step_results, figures, native_ok). native_ok is False when
    the model/Ollama does not support the `tools` field, signalling the caller
    to fall back to the text ReAct loop.
    """
    def _emit(msg: str, tool: str = "", level: str = "info") -> None:
        if emit:
            emit(msg, tool=tool, level=level)

    intent = getattr(kernel, "task_intent", None)
    if intent is not None:
        tool_names = [n for n in (tool_names or intent.allowed_tools)
                      if n in intent.allowed_tools]
    if _task_has_no_target(session, kernel):
        tool_names = [n for n in (tool_names if tool_names is not None
                                  else list(TOOL_REGISTRY))
                      if n not in _TARGET_ONLY_TOOLS]
    tools = _build_ollama_tool_schemas(tool_names)
    system = system_prompt or _TOOLCALL_SYSTEM.format(
        session_state=_data_context(session, kernel, compact=True),
        recent_history=_recent_history(session, n=6),
    )
    if prior_experience and system_prompt is None:
        system += (f"\n\nPRIOR EXPERIENCE (from past sessions — reuse what "
                   f"worked):\n{prior_experience}")
    messages: list[dict] = [{"role": "user", "content": instruction}]
    step_results: list[StepResult] = []
    all_figures: list[str] = []
    attempts: dict[str, int] = {}   # (tool, args) → count, for the critic budget
    replans = 0
    goal_met = False
    task_focus = _task_focus(session, kernel)

    for _step_n in range(max_steps):
        if getattr(kernel, "remaining_tool_calls", 1) <= 0 or time.monotonic() > getattr(kernel, "turn_deadline", float("inf")):
            break
        live_context = _data_context(session, kernel, compact=True)
        active_system = system + "\nCURRENT STATE:\n" + live_context
        if intent is not None:
            active_system += "\nBINDING TASK:\n" + intent.model_dump_json()
        resp = None
        last_exc: Exception | None = None
        for _attempt in range(2):  # one retry on transient LLM/HTTP failure
            try:
                resp = llm_client.chat_with_tools(
                    messages, tools, system=active_system,
                    temperature=0.0, max_tokens=_TOOL_TURN_TOKENS,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_timeout(exc):
                    break

        # None → native tool-calling unsupported: caller falls back.
        if resp is None:
            if llm_client.supports_tools() is False:
                return "", step_results, all_figures, False
            # Transient failure after retry: preserve any work done.
            if step_results:
                findings = _build_findings_context(
                    instruction, instruction, step_results)
                reply = _synthesise(
                    findings, llm_client,
                    task_context=_data_context(session, kernel),
                    history=_recent_history(session, n=4))
                return reply, step_results, all_figures, True
            return f"(LLM error: {last_exc})", step_results, all_figures, True

        content = resp.get("content", "") or ""
        calls = resp.get("tool_calls") or []

        # Record the assistant turn (with its tool_calls) so the model keeps
        # continuity across steps.
        assistant_msg: dict = {"role": "assistant", "content": content}
        if calls:
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        # No tool calls → the model answered directly. Done.
        if not calls:
            if content.strip():
                return content.strip(), step_results, all_figures, True
            break  # empty answer: fall through to synthesis from findings

        # Execute each requested tool call.
        for call in calls:
            if getattr(kernel, "remaining_tool_calls", 1) <= 0:
                break
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            raw_name = fn.get("name", "")
            args = _coerce_tool_args(fn.get("arguments"))
            resolved = _resolve_tool_name(raw_name)
            tool = TOOL_REGISTRY.get(resolved)

            call_success = False
            step_figures: list[str] = []   # figures from THIS call only
            if tool is None:
                obs = (f"ERROR: unknown tool '{raw_name}'. "
                       f"Choose from: {', '.join(TOOL_REGISTRY)}")
                _emit(f"Unknown tool '{raw_name}' — skipping", level="warn")
            else:
                _emit(f"Running {resolved}…", tool=resolved)
                t0 = time.time()
                try:
                    out: ToolOutput = _execute_tool(
                        resolved, args, session=session, kernel=kernel,
                        llm_client=llm_client)
                    call_success = out.success
                    obs = (out.text if out.success else
                           f"ERROR: {_error_headline(out.error, 400)}\n\n"
                           f"Full traceback:\n{out.error}")
                    all_figures.extend(out.figures)
                    step_figures = list(out.figures or [])
                    if out.success:
                        # A record of what actually ran, independent of the
                        # critic. EXPLORATION_LOG only exists when the critic
                        # is enabled and says nothing about which tool wrote an
                        # entry, so phase and staleness cannot be derived from
                        # it. Bounded: names are short and the tail is enough.
                        _run = kernel.namespace.setdefault("TOOLS_RUN", [])
                        if isinstance(_run, list):
                            _run.append(resolved)
                            if len(_run) > 100:
                                del _run[:-100]
                    elapsed = round(time.time() - t0, 2)
                    step_results.append(StepResult(
                        step=len(step_results) + 1, tool=resolved,
                        args=args, reason=content[:200],
                        output=out, elapsed_s=elapsed))
                    if out.success:
                        _emit(f"✓ {resolved} done ({elapsed}s)",
                              tool=resolved, level="ok")
                    else:
                        _emit(f"✗ {resolved} failed: {_error_headline(out.error)}",
                              tool=resolved, level="error")
                except Exception as exc:  # noqa: BLE001
                    obs = f"ERROR: tool crashed: {exc}"
                    _emit(f"✗ {resolved} crashed: {exc}",
                          tool=resolved, level="error")

            # The newest result is the one being reasoned about, so it gets the
            # full budget; older ones are squeezed to a gist. A flat 1500-char
            # cap meant an eda_profile over 26 columns was cut to roughly its
            # first third, and the model was planning from EDA it could not
            # read. Compressing the tail keeps the prompt inside num_ctx.
            for _m in messages:
                if _m.get("role") == "tool" and len(_m.get("content", "")) > _OBS_TAIL_CHARS:
                    _m["content"] = (_m["content"][:_OBS_TAIL_CHARS]
                                     + "\n… [earlier result trimmed]")
            messages.append({
                "role": "tool", "tool_name": resolved,
                "content": (obs or "")[:_OBS_CHARS],
            })

            # Let the model look at the plots it just produced. Only the newest
            # turn carries images: they are large, and the point is to read the
            # chart in front of it, not to re-read every earlier one.
            if _SEND_FIGURES and step_figures:
                for _m in messages:
                    _m.pop("images", None)
                if llm_client.supports_vision() is not False:
                    _imgs = [s for s in
                             (_shrink_figure(f)
                              for f in step_figures[:_MAX_FIGURES_SENT])
                             if s]
                    if _imgs:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The {resolved} step produced the "
                                f"{'figure' if len(_imgs) == 1 else 'figures'} "
                                "attached. Read them and say what they show "
                                "about the data before choosing the next step."),
                            "images": _imgs,
                        })

            # ── Critic: judge the result and steer the next turn ────────────
            if not _CRITIC_ENABLED:
                continue
            if (call_success and not _CRITIC_ALL_TOOLS
                    and resolved in _CRITIC_SKIP_TOOLS):
                # A successful retrieval needs no judgement; skipping it saves a
                # full LLM call per step. Failures still fall through below,
                # where the rule-based path handles them without an LLM call.
                continue
            sig = _call_signature(resolved, args)
            attempts[sig] = attempts.get(sig, 0) + 1
            crit = _critique_step(
                instruction, resolved, args, obs, call_success,
                llm_client=llm_client, repeats=attempts[sig],
                task_focus=task_focus,
                agenda=kernel.namespace.get("EDA_AGENDA"))
            if call_success and crit.verdict == "accept":
                from agent.api.agent_tools import add_lead, record_findings
                settled = record_findings(kernel.namespace, crit.answers, crit.finding)
                cells = getattr(kernel, "analysis_cells", None)
                if resolved == "execute_code" and isinstance(cells, list) and cells:
                    # The export shows each step with what it found.
                    cells[-1]["finding"] = crit.finding
                    cells[-1]["answers"] = settled
                if settled:
                    _emit(f"agenda: {', '.join(settled)} answered", tool="agenda",
                          level="ok")
                lead = add_lead(kernel.namespace, crit.new_question)
                if lead:
                    _emit(f"agenda: new lead {lead}: {crit.new_question[:140]}",
                          tool="agenda")
            _emit(f"critic: {crit.verdict}"
                  + (f" — {crit.reason}" if crit.reason else ""),
                  tool="critic",
                  level="ok" if crit.verdict == "accept" else "warn")

            # Record what the step revealed so later steps and the final answer
            # build on it (the running exploration log, kept on the kernel).
            if crit.finding:
                _log = kernel.namespace.setdefault("EXPLORATION_LOG", [])
                if isinstance(_log, list):
                    _log.append(f"[{resolved}] {crit.finding}")
                _emit(f"learned: {crit.finding}", tool="reflect")

            if crit.goal_met:
                # Goal satisfied: stop calling tools and let the loop synthesise.
                goal_met = True
                break
            if crit.verdict == "accept" and crit.next_step:
                # Feed the reflection back so the model re-plans from the output
                # rather than following a fixed sequence.
                messages.append({
                    "role": "user",
                    "content": (
                        f"Reflection: {crit.finding} Best next step: "
                        f"{crit.next_step} Decide the next tool call, or give the "
                        "final answer if the goal is met."),
                })
            elif crit.verdict == "retry":
                messages.append({
                    "role": "user",
                    "content": (
                        f"Reviewer: the {resolved} result is inadequate "
                        f"({crit.reason or 'off-target'}). Retry this step with "
                        "different arguments or a better-suited tool. Do not "
                        "repeat the identical call."),
                })
            elif crit.verdict == "replan":
                replans += 1
                messages.append({
                    "role": "user",
                    "content": (
                        f"Reviewer: the current approach is not working "
                        f"({crit.reason or 'wrong strategy'}). Rethink which "
                        "tools can meet the goal and change strategy."),
                })
                if replans > _MAX_LOOP_REPLANS:
                    _emit("Replan budget spent — stopping.", level="warn")
                    goal_met = True  # force exit; synthesise partial findings
                    break

        if goal_met:
            break

    # Budget spent or empty answer: synthesise from whatever we have.
    if step_results:
        findings = _build_findings_context(instruction, instruction, step_results)
        reply = _synthesise(
            findings, llm_client,
            task_context=_data_context(session, kernel),
            history=_recent_history(session, n=4))
        return reply, step_results, all_figures, True
    return (
        "(No response generated — please try again with more specific "
        "instructions.)",
        step_results, all_figures, True,
    )


# ── Phase 3: role-specialised sub-agents + orchestrator ───────────────────────

# Each role is a focused system prompt + a subset of the tool registry. A smaller
# tool space per agent measurably improves tool selection on local models, and
# the role prompt keeps each agent on its own part of the pipeline.
_AGENT_ROLES: dict[str, dict[str, Any]] = {
    "data_engineer": {
        "title": "Data Engineer",
        "purpose": "explore the working directory, connect to data sources, "
                   "load, join, aggregate, and quality-check data; define the "
                   "modelling frame and the prediction target",
        "tools": ["explore_directory", "read_file", "search_files",
                  "connect_data", "sql_query", "understand_task",
                  "read_instructions", "aggregate_data", "quality_check",
                  "infer_target", "work_plan"],
    },
    "analyst": {
        "title": "Analyst",
        "purpose": "run the EDA: answer the plan's task questions AND its "
                   "independent checks on the data, read each result before "
                   "choosing the next step, then revise the plan (work_plan "
                   "with revise=true) so the feature and model choices follow "
                   "what the data showed",
        "tools": ["eda_profile", "deep_eda", "mutual_information",
                  "data_analysis_agent", "visualize", "work_plan"],
    },
    "ml_engineer": {
        "title": "ML Engineer",
        "purpose": "engineer features, choose a modelling strategy, train and "
                   "evaluate models, and package the trained model for "
                   "deployment",
        "tools": ["infer_target", "feature_engineering", "ml_strategy_agent",
                  "train_model", "deploy_model"],
    },
    "coder": {
        "title": "Coder",
        "purpose": "search, read, write, and edit files in the working "
                   "directory, run Python for anything the other tools do not "
                   "cover, assemble the final notebook, and scaffold CI/CD",
        "tools": ["search_files", "read_file", "write_file", "edit_file",
                  "execute_code", "build_notebook", "scaffold_cicd"],
    },
    "researcher": {
        "title": "Researcher",
        "purpose": "research topics, methods, and libraries on the live web when "
                   "current information is needed — latest versions, new packages, "
                   "recent best practices — and report findings with source URLs",
        "tools": ["web_research", "execute_code"],
    },
}

_SPECIALIST_SYSTEM = """\
You are the {title} on a data-science team. Your job: {purpose}.

You have a focused set of tools. Use them to complete ONLY the sub-goal below,
then write a short plain-text report of what you did and what you found (include
any metric, path, or source URL a teammate would need). Do not attempt work
outside your role — the team handles the rest.

Read each tool result before choosing the next call, and adapt: let what a
result reveals decide the next step instead of following a fixed sequence. When
the session state lists REQUIREMENTS TO SATISFY, address every requirement that
falls within your role before you report, using execute_code for anything your
standard tools do not cover.

If the session state lists EDA QUESTIONS FROM THE PLAN, they are your agenda.
Answer the task questions AND the independent checks; the independent ones are
how problems the brief never mentions get caught, so do not skip them because
they look tangential. Use execute_code for anything the standard EDA tools do
not cover, and state what each answer implies for the work, not just the
number. A question you cannot settle is a finding too: say so and say why.

CURRENT SESSION STATE:
{session_state}

SUB-GOAL:
{subgoal}
"""

_ORCH_SYSTEM = """\
You are the orchestrator of a data-science team. Break the user's request into
sub-goals and delegate each to the right specialist by calling its tool with a
clear one-sentence `subgoal`. The specialists share the same data session, so
work accumulates across calls.

Team:
{team}

Rules:
1. Delegate in a sensible order — data must be loaded and understood before it is
   analysed or modelled; research any unfamiliar library or method before relying
   on it.
2. Call one specialist at a time and read its report before the next delegation.
3. Do not do the work yourself and do not repeat a delegation that already
   succeeded.
4. Plan in two passes, because a plan written before the data is seen is a
   guess. Early on, have a specialist call `work_plan` to set out the approach.
   After EDA has run, have the analyst call `work_plan` with revise=true so the
   feature-engineering and model choices are rewritten from what the data
   actually showed. Only then delegate the modelling.
5. When the request is fully handled, stop delegating and write the final answer
   to the user as plain text.
{prior_experience}
CURRENT SESSION STATE:
{session_state}
"""


def _run_specialist(
    role: str,
    subgoal: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    emit: Any = None,
) -> tuple[str, list[StepResult], list[str]]:
    """Run one role-specialised agent on a sub-goal. It is the native tool-call
    loop restricted to the role's toolset and given the role's system prompt, so
    the critic and all budgets apply unchanged. Returns (report, steps, figs)."""
    spec = _AGENT_ROLES[role]
    system = _SPECIALIST_SYSTEM.format(
        title=spec["title"], purpose=spec["purpose"],
        session_state=_data_context(session, kernel, compact=True),
        subgoal=subgoal)
    if emit:
        emit(f"→ {spec['title']}: {subgoal[:80]}", tool=role)
    reply, steps, figs, _native = _tool_call_loop(
        subgoal, session, kernel, llm_client,
        max_steps=_SPECIALIST_MAX_STEPS, emit=emit,
        tool_names=spec["tools"], system_prompt=system)
    return reply, steps, figs


def _build_handoff_schemas() -> list[dict]:
    """One Ollama function per role — the orchestrator's only tools. Each takes a
    single `subgoal` string, so delegation is a native tool call."""
    schemas: list[dict] = []
    for role, spec in _AGENT_ROLES.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": role,
                "description": f"Delegate to the {spec['title']}, who can "
                               f"{spec['purpose']}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subgoal": {
                            "type": "string",
                            "description": "One clear sentence describing what "
                                           "this specialist should do now.",
                        },
                    },
                    "required": ["subgoal"],
                },
            },
        })
    return schemas


def _run_multiagent_loop(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    max_rounds: int | None = None,
    emit: Any = None,
    prior: str = "",
) -> tuple[str, list[StepResult], list[str], bool]:
    """Supervisor loop: the orchestrator delegates sub-goals to specialists via
    native handoff tool-calls until the request is handled or the round budget is
    spent. Each specialist runs its own critic-guarded tool loop on the shared
    session kernel (the blackboard).

    Returns (reply, all_steps, all_figures, native_ok). native_ok is False when
    the model can't do native tool-calling, so the caller falls back.
    """
    def _emit(msg: str, tool: str = "", level: str = "info") -> None:
        if emit:
            emit(msg, tool=tool, level=level)

    max_rounds = max_rounds or _MAX_ORCH_ROUNDS
    team = "\n".join(f"  - {r} ({s['title']}): {s['purpose']}"
                     for r, s in _AGENT_ROLES.items())
    handoffs = _build_handoff_schemas()
    prior_block = (f"\nPRIOR EXPERIENCE (from past sessions — reuse what worked):\n"
                   f"{prior}\n" if prior else "")
    if prior:
        _emit("Recalled relevant prior work from memory.", tool="memory")
    system = _ORCH_SYSTEM.format(
        team=team, prior_experience=prior_block,
        session_state=_data_context(session, kernel, compact=True))
    messages: list[dict] = [{"role": "user", "content": instruction}]
    all_steps: list[StepResult] = []
    all_figures: list[str] = []
    done_delegations: set[str] = set()

    _emit("Orchestrator planning…", tool="orchestrator")
    for _round in range(max_rounds):
        system = _ORCH_SYSTEM.format(team=team, prior_experience=prior_block,
                                    session_state=_data_context(session, kernel, compact=True))
        resp = None
        for _attempt in range(2):
            try:
                resp = llm_client.chat_with_tools(
                    messages, handoffs, system=system,
                    temperature=0.0, max_tokens=700)
                break
            except Exception as exc:  # noqa: BLE001
                if _is_timeout(exc):
                    break
        if resp is None:
            # Native tool-calling unsupported → signal fallback (only if we have
            # not already delegated any work).
            if llm_client.supports_tools() is False and not all_steps:
                return "", all_steps, all_figures, False
            break

        content = resp.get("content", "") or ""
        calls = resp.get("tool_calls") or []
        assistant_msg: dict = {"role": "assistant", "content": content}
        if calls:
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not calls:
            if content.strip():
                return _grounded_reply(content, all_steps, llm_client), all_steps, all_figures, True
            break

        for call in calls:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            role = fn.get("name", "")
            if role not in _AGENT_ROLES:
                messages.append({
                    "role": "tool", "tool_name": role or "unknown",
                    "content": (f"ERROR: no such specialist '{role}'. "
                                f"Choose from: {', '.join(_AGENT_ROLES)}")})
                continue
            args = _coerce_tool_args(fn.get("arguments"))
            subgoal = (args.get("subgoal") or instruction).strip()

            # Modelling must not start from a plan written before the data was
            # seen. When EDA has produced findings the plan has not been
            # revised for, send the orchestrator back to the analyst instead of
            # letting it engineer features against assumptions.
            if role == "ml_engineer" and work_plan_is_stale(kernel):
                messages.append({
                    "role": "tool", "tool_name": role,
                    "content": (
                        "BLOCKED: EDA has produced findings that the work plan "
                        "has not been revised for, so the feature-engineering "
                        "and model choices are still assumptions. Delegate to "
                        "the analyst to call work_plan with revise=true first, "
                        "then delegate the modelling.")})
                _emit("Modelling held: the plan has not been revised since EDA.",
                      tool="orchestrator", level="warn")
                continue

            sig = _call_signature(role, {"subgoal": subgoal})
            if sig in done_delegations:
                messages.append({
                    "role": "tool", "tool_name": role,
                    "content": "Already completed this exact sub-goal. Delegate "
                               "the next distinct step or write the final answer."})
                continue
            done_delegations.add(sig)

            report, steps, figs = _run_specialist(
                role, subgoal + "\nPrior evidence:\n" + _build_findings_context(instruction, instruction, all_steps),
                session, kernel, llm_client, emit=emit)
            all_steps.extend(steps)
            all_figures.extend(figs)
            _emit(f"✓ {_AGENT_ROLES[role]['title']} done "
                  f"({len(steps)} tool call{'s' if len(steps) != 1 else ''})",
                  tool=role, level="ok")
            messages.append({
                "role": "tool", "tool_name": role,
                "content": f"{_AGENT_ROLES[role]['title']} report:\n"
                           f"{(report or '(no report)')[:1500]}"})

    # Round budget spent or empty answer: synthesise from accumulated work.
    if all_steps:
        findings = _build_findings_context(instruction, instruction, all_steps)
        reply = _synthesise(
            findings, llm_client,
            task_context=_data_context(session, kernel),
            history=_recent_history(session, n=4))
        return reply, all_steps, all_figures, True
    return (
        "(No response generated — please try again with more specific "
        "instructions.)",
        all_steps, all_figures, True,
    )


def _run_agentic_loop(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    max_steps: int | None = None,
    emit: Any = None,
    ltm: Any = None,
    multiagent: bool | None = None,
) -> tuple[str, list[StepResult], list[str]]:
    """Dispatch the build path: multi-agent orchestrator first, then the
    single-agent native loop, then the text ReAct loop — each falling back to the
    next when native tool-calling is unavailable.

    `ltm` (long-term memory) is queried once for relevant past work, which is
    injected into the orchestrator/agent so it can reuse prior solutions.

    `multiagent` overrides the DSAGENT_MULTIAGENT default per request (the web-UI
    toggle): True forces the orchestrator, False forces the single-agent loop,
    None uses the configured default.

    This is the single entry point run_reactive uses for build-style requests.
    """
    max_steps = max_steps or _AGENT_MAX_STEPS
    use_multiagent = _MULTIAGENT if multiagent is None else bool(multiagent)
    prior = _recall_from_ltm(ltm, instruction, session)
    if _NATIVE_TOOLS and llm_client.supports_tools() is not False:
        if use_multiagent:
            reply, steps, figs, native_ok = _run_multiagent_loop(
                instruction, session, kernel, llm_client, emit=emit, prior=prior)
            if native_ok:
                return reply, steps, figs
        else:
            reply, steps, figs, native_ok = _tool_call_loop(
                instruction, session, kernel, llm_client,
                max_steps=max_steps, emit=emit, prior_experience=prior)
            if native_ok:
                return reply, steps, figs
        if emit:
            emit("Model lacks native tool-calling — using text loop.",
                 level="warn")
    return _react_loop(instruction, session, kernel, llm_client,
                       max_steps=max_steps)


def _react_loop(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    max_steps: int = 8,
) -> tuple[str, list[StepResult], list[str]]:
    """Multi-turn Thought/Action/Observation loop.

    Called when _classify_tools() returns no confident tool plan.
    Returns (reply, step_results, figures).
    """
    tool_list = "\n".join(_format_tool_line(name, t)
                          for name, t in TOOL_REGISTRY.items())
    system = _REACT_SYSTEM.format(
        tool_list=tool_list,
        session_state=_data_context(session, kernel),
        recent_history=_recent_history(session, n=6),
    )

    messages: list[dict] = [{"role": "user", "content": instruction}]
    step_results: list[StepResult] = []
    all_figures: list[str] = []

    for step_n in range(max_steps):
        # Short messages (< 8 words) with no tool keywords → cap tokens low
        _word_count = len(instruction.split())
        _max_tok = 400 if _word_count <= 8 else 600

        raw = None
        for _attempt in range(2):  # one retry on transient LLM failure
            try:
                raw = llm_client._chat(
                    messages,
                    system=system,
                    temperature=0.0,
                    max_tokens=_max_tok,
                )
                break
            except Exception as exc:
                last_exc = exc
        if raw is None:
            # LLM unreachable after retry. Don't discard work done so far —
            # synthesise whatever findings we already have.
            if step_results:
                findings = _build_findings_context(
                    instruction, instruction, step_results)
                return (
                    _synthesise(findings, llm_client,
                                task_context=_data_context(session, kernel),
                                history=_recent_history(session, n=4)),
                    step_results, all_figures,
                )
            return f"(LLM error: {last_exc})", step_results, all_figures

        text = _strip_thinking(raw).strip()
        if not text:
            break

        messages.append({"role": "assistant", "content": text})
        parsed = _parse_react_step(text)

        if parsed.final_answer is not None:
            return parsed.final_answer, step_results, all_figures

        # Malformed Action Input → tell the model instead of running with {}.
        if parsed.action and parsed.parse_error:
            messages.append({
                "role": "user",
                "content": (f"Observation: ERROR — {parsed.parse_error} "
                            "Resend the Action and a valid JSON Action Input."),
            })
            continue

        if not parsed.action:
            return text, step_results, all_figures

        resolved = _resolve_tool_name(parsed.action)
        tool = TOOL_REGISTRY.get(resolved)
        if tool is None:
            obs = f"ERROR: Unknown tool '{parsed.action}'. Choose from: {', '.join(TOOL_REGISTRY)}"
        else:
            t0 = time.time()
            try:
                out: ToolOutput = _execute_tool(
                    resolved, parsed.action_input,
                    session=session, kernel=kernel, llm_client=llm_client,
                )
                obs = out.text if out.success else f"ERROR: {out.error}"
                all_figures.extend(out.figures)
                step_results.append(StepResult(
                    step=step_n + 1, tool=resolved,
                    args=parsed.action_input, reason=parsed.thought,
                    output=out, elapsed_s=round(time.time() - t0, 2),
                ))
            except Exception as exc:
                obs = f"ERROR: Tool crashed: {exc}"

        messages.append({"role": "user", "content": f"Observation: {obs[:1200]}"})

    if step_results:
        findings = _build_findings_context(instruction, instruction, step_results)
        return (
            _synthesise(findings, llm_client,
                        task_context=_data_context(session, kernel),
                        history=_recent_history(session, n=4)),
            step_results, all_figures,
        )
    return (
        "(No response generated — please try again with more specific instructions.)",
        [], [],
    )


def _error_recovery_plan(instruction: str) -> list[dict]:
    """Generate a debug plan when the user pastes a Python error/traceback."""
    # Extract the error type and message for diagnosis code
    error_line = ""
    for line in instruction.splitlines():
        if "Error:" in line or "Exception:" in line:
            error_line = line.strip()[:200]
            break

    diag_code = (
        "# Diagnose the error\n"
        "print('=== Kernel state diagnosis ===')\n"
        "import numpy as np\n"
        "for _name, _val in list(locals().items()) + list(globals().items()):\n"
        "    if hasattr(_val, 'shape'):\n"
        "        print(f'{_name}: shape={_val.shape}, dtype={getattr(_val, \"dtype\", \"?\")}', end='')\n"
        "        if hasattr(_val, '__len__') and len(_val.shape) == 1:\n"
        "            nan_count = np.isnan(_val.astype(float)).sum() if np.issubdtype(getattr(_val, 'dtype', object), np.number) else 0\n"
        "            print(f', NaN={nan_count}')\n"
        "        else:\n"
        "            print()\n"
        f"print('Error was: {error_line}')"
    )

    fix_code = (
        "# Fix: ensure target (y) has no NaN before training\n"
        "import pandas as pd, numpy as np\n"
        "# Identify and drop NaN rows in target\n"
        "if 'y' in dir() or 'y_enc' in dir():\n"
        "    _y = y_enc if 'y_enc' in dir() else y\n"
        "    _nan_mask = pd.isnull(_y)\n"
        "    if _nan_mask.any():\n"
        "        print(f'Found {_nan_mask.sum()} NaN values in target — fixing...')\n"
        "        if 'X_proc' in dir():\n"
        "            X_proc = X_proc[~_nan_mask.values] if hasattr(_nan_mask, 'values') else X_proc[~_nan_mask]\n"
        "        if 'y_enc' in dir():\n"
        "            y_enc = pd.to_numeric(y_enc, errors='coerce').fillna(pd.to_numeric(y_enc, errors='coerce').median())\n"
        "        elif 'y' in dir():\n"
        "            from sklearn.preprocessing import LabelEncoder\n"
        "            y_clean = y.fillna(y.mode()[0])\n"
        "            le = LabelEncoder()\n"
        "            y_enc = le.fit_transform(y_clean)\n"
        "        print('Fix applied. Shapes:', getattr(X_proc, 'shape', '?'), getattr(y_enc, 'shape', '?'))\n"
        "    else:\n"
        "        print('No NaN in target — issue may be elsewhere. Check X_proc shape.')\n"
        "else:\n"
        "    print('y / y_enc not found in kernel. Run preprocessing cell first.')"
    )

    plan = [
        {"step": 1, "tool": "execute_code",
         "args": {"code": diag_code},
         "reason": "Diagnose the error — inspect kernel variable shapes and NaN counts"},
    ]
    # Only append the NaN-target fix when the error is actually about NaN /
    # infinite / non-finite values. For other error classes (import, key,
    # shape-mismatch, type) that fix is irrelevant — emit diagnosis only and
    # let the LLM synthesise a targeted next step from the diagnosis output.
    _err_low = error_line.lower()
    _is_nan_error = any(k in _err_low for k in (
        "nan", "infinity", "infinite", "contains nan", "input contains",
        "null", "missing",
    ))
    if _is_nan_error:
        plan.append({
            "step": 2, "tool": "execute_code",
            "args": {"code": fix_code},
            "reason": "Apply fix — remove NaN from target and align feature matrix",
        })
    return plan


def _keyword_plan(instruction: str, session: Session) -> list[dict]:
    """Generate a sensible plan from keywords when the LLM planner returns nothing valid."""
    # Error recovery takes priority
    if _is_error_message(instruction):
        return _error_recovery_plan(instruction)

    lo = instruction.lower()

    wants_pipeline = any(w in lo for w in [
        "pipeline", "full", "complete", "end-to-end", "end to end",
        "ds pipeline", "data science pipeline", "implement", "perform",
    ])
    wants_eda = any(w in lo for w in [
        "eda", "explore", "explor", "analysis", "analyse", "analyze",
        "profile", "distribution", "overview", "descriptive",
    ])
    wants_model = any(w in lo for w in [
        "model", "train", "test", "predict", "classif", "regression", "ml",
        "machine learning", "random forest", "gradient boost", "logistic",
        "select ml", "select model",
    ])
    wants_notebook = any(w in lo for w in [
        "notebook", "ipynb", "save", "report", "build", "generate", "conclusion",
    ])
    wants_viz = any(w in lo for w in [
        "plot", "chart", "graph", "visual", "correlation", "heatmap",
    ])

    # If a brief is present but instruction is vague (mentions "instructions/brief/pdf"),
    # treat it as a full cross-dataset pipeline request — understand_task will clarify.
    has_brief = bool(session.brief and session.brief.strip())
    mentions_brief = any(w in lo for w in [
        "brief", "pdf", "instruction", "document", "task", "exercise",
    ])
    if has_brief and (mentions_brief or not (wants_eda or wants_model or wants_pipeline)):
        # Full pipeline through understand_task
        wants_pipeline = True
        wants_notebook = True

    steps: list[dict] = []
    n = 1

    def add(tool, args=None, reason=""):
        nonlocal n
        steps.append({"step": n, "tool": tool,
                      "args": args or {}, "reason": reason})
        n += 1

    # If brief is present, understand the task first — it may reveal cross-dataset
    # aggregation requirements that keyword matching would miss.
    if has_brief:
        add("understand_task",
            reason="Extract task spec from brief: target definition, metric, aggregation needs")

    # Always profile the raw data
    add("eda_profile", reason="Understand raw dataset structure")

    if wants_pipeline or wants_model:
        add("quality_check", reason="Identify data quality issues")

        # Check if we already know from brief that aggregation is needed
        _task_spec_placeholder: dict = {}  # will be populated at runtime; we plan conservatively  # noqa: F841
        # If brief mentions establishment/entity-level prediction, plan aggregate step
        agg_keywords = [
            "per establishment", "per customer", "per patient",
            "aggregate", "rollup", "group by", "each establishment",
            "entity level", "entity-level",
        ]
        brief_lo = (session.brief or "").lower()
        needs_agg = any(k in brief_lo for k in agg_keywords)
        if needs_agg:
            add("aggregate_data",
                {"group_by": "establishment_id"},  # LLM will override with correct key
                reason="Aggregate incident rows to one row per establishment")

        if not needs_agg:
            # Standard path: infer target from existing columns
            add("infer_target", reason="Auto-select target column from brief and EDA")

        add("mutual_information", reason="Rank features by predictive power")
        add("feature_engineering", {"top_k": 30}, reason="Encode and select features")
        add("train_model",
            {"task_type": "auto", "models": ["rf", "gbm", "lr"],
             "eval_metric": "auto"},
            reason="Compare ML models with auto metric (PR-AUC for imbalanced)")
        add("visualize", {"plot_type": "distributions"}, reason="Distribution plots")
        add("visualize", {"plot_type": "target_dist"}, reason="Target distribution")
        if wants_notebook or wants_pipeline:
            add("build_notebook", reason="Save complete .ipynb report")

    elif wants_eda or wants_viz:
        add("quality_check", reason="Data quality check")
        add("visualize", {"plot_type": "distributions"}, reason="Feature distributions")
        add("visualize", {"plot_type": "correlation"}, reason="Correlation heatmap")
        if wants_notebook:
            add("build_notebook", reason="Save EDA notebook")

    else:
        # Generic — just profile + correlation
        add("quality_check", reason="Data quality")
        add("visualize", {"plot_type": "correlation"}, reason="Correlations")

    return steps


def _build_brief_context(brief: str, task_spec: dict) -> str:
    """Combine raw brief text with extracted task_spec for the planner prompt."""
    raw = (brief or "")[:2000]
    if not task_spec:
        return raw or "(no brief — infer from data and instruction)"
    spec_lines = [
        "=== Extracted Task Specification (from understand_task) ===",
        f"Task        : {task_spec.get('task_description', '?')}",
        f"Type        : {task_spec.get('task_type', '?')}",
        f"Target      : {task_spec.get('target_definition', '?')}",
        f"Derived     : {task_spec.get('target_is_derived', False)}",
        f"Aggregate   : {task_spec.get('aggregation_needed', False)} "
        f"(key: {task_spec.get('aggregation_key', 'n/a')})",
        f"Metric      : {task_spec.get('evaluation_metric', '?')}",
        f"2nd datasets: {task_spec.get('secondary_datasets', [])}",
        f"Feature hints: {task_spec.get('feature_engineering_hints', [])[:5]}",
        "",
        "=== Raw Brief (first 1500 chars) ===",
        raw[:1500],
    ]
    return "\n".join(spec_lines)


def _keyword_plan_from_spec(instruction: str, session: Session,
                             task_spec: dict,
                             llm_client: Any = None) -> list[dict]:
    """Build a keyword plan that uses task_spec when available.

    This is smarter than _keyword_plan because it reads the already-extracted
    task specification to choose correct aggregation key, target condition,
    evaluation metric, etc.
    """
    # Error recovery takes priority
    if _is_error_message(instruction):
        return _error_recovery_plan(instruction)

    lo = instruction.lower()

    # Determine intent from both instruction keywords and task_spec
    has_spec = bool(task_spec)
    needs_agg = (
        task_spec.get("aggregation_needed", False)
        if has_spec
        else any(k in (session.brief or "").lower() for k in [
            "per establishment", "per customer", "aggregate", "group by",
            "entity level", "entity-level", "one row per",
        ])
    )
    agg_key = task_spec.get("aggregation_key") or "establishment_id"
    # Guard: industry/classification codes are not valid entity grouping keys
    _CAT_CODE_KWS = ("naics", "sic", "nace", "isco", "isic",
                     "industry_code", "sector_code", "zip_code", "postal_code")
    if any(kw in agg_key.lower() for kw in _CAT_CODE_KWS):
        agg_key = "establishment_id"
    target_def = task_spec.get("target_definition", "")
    target_derived = task_spec.get("target_is_derived", False)
    eval_metric = task_spec.get("evaluation_metric", "auto")
    task_type = task_spec.get("task_type", "auto")

    # Convert spec metric names → train_model arg values
    metric_map = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "accuracy": "f1_macro",
        "rmse": "neg_root_mean_squared_error",
    }
    train_metric = metric_map.get(eval_metric, "auto")

    wants_pipeline = any(w in lo for w in [
        "pipeline", "full", "complete", "end-to-end", "implement",
        "perform", "build", "generate", "run",
    ]) or has_spec   # if we have a spec, always run full pipeline
    wants_notebook = any(w in lo for w in [
        "notebook", "ipynb", "save", "report", "build", "generate",
    ]) or has_spec

    steps: list[dict] = []
    n = 1

    def add(tool, args=None, reason=""):
        nonlocal n
        steps.append({"step": n, "tool": tool,
                      "args": args or {}, "reason": reason})
        n += 1

    # Always start with EDA of the raw data
    add("eda_profile", reason="Profile raw dataset structure and distributions")
    add("quality_check", reason="Identify data quality issues before modelling")
    # Deep EDA — six-stage visual + statistical analysis with LLM narrative
    _eda_depth = "quick" if task_spec.get("n_samples", 0) > 50_000 else "standard"
    add("deep_eda", {"depth": _eda_depth},
        reason=f"Deep EDA: quality radar, bivariate analysis, PCA, missingness map, LLM narrative ({_eda_depth} mode)")

    # Detect risk-ranking requirement: severity hierarchy mentioned in brief
    _reqs_text = " ".join(task_spec.get("specific_requirements", [])).lower()
    _brief_lower = (session.brief or "").lower()
    _wants_risk_ranking = any(
        kw in _reqs_text or kw in _brief_lower
        for kw in ["risk rank", "risk score", "severity", "severity hierarchy",
                   "death", "days away", "harm score", "risk scoring"]
    )

    if wants_pipeline:
        # Aggregation step — args built from task_spec
        if needs_agg:
            agg_args: dict = {"group_by": agg_key}
            # Add target condition if target must be derived
            if target_derived and target_def:
                # Check if the expression looks usable (not just a description)
                if any(op in target_def for op in [
                    "isin", "==", ">=", "<=", ">", "<", " in ["
                ]):
                    agg_args["target_condition"] = target_def
                    agg_args["target_column_name"] = (
                        task_spec.get("target_column") or "target"
                    )
            add("aggregate_data", agg_args,
                reason=f"Aggregate incident rows to one row per {agg_key}"
                       + (" and derive binary target" if target_derived else ""))
        else:
            # No aggregation — infer target from existing columns
            add("infer_target",
                reason="Auto-select target column from brief and dataset stats")

        # Risk ranking step — severity-weighted score per entity
        if _wants_risk_ranking:
            add(
                "execute_code",
                {
                    "code": (
                        "# Risk ranking: severity-weighted score per entity\n"
                        "# Severity hierarchy: Death(1) > DAFW(2) > Job transfer(3) > Other(4)\n"
                        "import pandas as _pd\n"
                        "_df_risk = df.copy() if 'df' in dir() else data.copy()\n"
                        "_severity_map = {1: 4, 2: 3, 3: 2, 4: 1}  # higher score = more severe\n"
                        "_out_col = [c for c in _df_risk.columns if 'outcome' in c.lower()]\n"
                        "if _out_col:\n"
                        "    _df_risk['_sev_score'] = _df_risk[_out_col[0]].map(_severity_map).fillna(0)\n"
                        "    _risk_table = _df_risk.groupby(_df_risk.columns[0])['_sev_score'].sum().sort_values(ascending=False)\n"
                        "    _risk_table.name = 'risk_score'\n"
                        "    _risk_table = _risk_table.reset_index()\n"
                        "    _risk_table['rank'] = range(1, len(_risk_table)+1)\n"
                        "    print('Top 10 highest-risk entities:')\n"
                        "    print(_risk_table.head(10).to_string(index=False))\n"
                        "    RISK_TABLE = _risk_table\n"
                    )
                },
                reason="Compute severity-weighted risk score and top-10 risk ranking",
            )

        # Core modelling pipeline
        add("mutual_information",
            reason="Rank features by MI, IGR, and permutation importance against target")

        # Task-aware EDA visualizations — injected after MI so top features are known
        _is_cls_task = task_type in ("classification", "binary_classification") or (
            task_type == "auto" and "classif" in str(task_spec.get("task_type", "")).lower()
        )
        if _is_cls_task:
            add("visualize", {"plot_type": "feature_target"},
                reason="KDE distributions of top features by target class")
            add("visualize", {"plot_type": "class_separation"},
                reason="Univariate ROC-AUC per feature — which features best separate classes")
            add("visualize", {"plot_type": "correlation_target"},
                reason="Feature-target Pearson correlation ranking")

        add("feature_engineering", {"top_k": 30},
            reason="Encode categoricals, generate interactions, select top features")
        add("train_model",
            {
                "task_type": task_type if task_type in ("classification",
                                                         "regression") else "auto",
                "models": ["rf", "gbm", "lr"],
                "eval_metric": train_metric,
            },
            reason=f"Compare RF/GBM/LR with 5-fold CV — metric: {train_metric}")
        add("visualize", {"plot_type": "distributions"},
            reason="Feature distributions")
        add("visualize", {"plot_type": "target_dist"},
            reason="Target distribution and class balance")

        # Inject task-specific requirements from the brief as execute_code steps.
        # These are explicit analyses/steps named in the PDF that the standard
        # pipeline doesn't cover (e.g. specific feature analysis, model comparisons).
        _COVERED_KEYWORDS = {
            "eda", "profile", "quality", "missing", "null", "aggregate",
            "aggregat", "feature engineer", "mutual information", "train",
            "model", "rf", "visualiz", "distribut", "notebook", "ipynb",
        }
        import re as _re_req
        _PLACEHOLDER_PAT = _re_req.compile(r'^\s*<[^>]+>\s*$')
        for req in task_spec.get("specific_requirements", []):
            if _PLACEHOLDER_PAT.match(req):
                continue  # skip "<requirement 1>" template strings
            req_lower = req.lower()
            if not any(kw in req_lower for kw in _COVERED_KEYWORDS):
                # Generate actual code for the requirement via the code model
                _req_code = ""
                if llm_client is not None:
                    _cols = task_spec.get("columns", [])
                    _tgt_hint = task_spec.get("target_column", "")
                    _prompt = (
                        f"Write a self-contained Python snippet that runs on a pandas "
                        f"DataFrame called `df`. The task is:\n{req}\n\n"
                        f"Available columns (sample): {_cols[:20]}\n"
                        f"Target column: {_tgt_hint}\n\n"
                        f"Rules: use only pandas/numpy/matplotlib/sklearn. "
                        f"Print results. No markdown fences. No explanations."
                    )
                    _req_code = llm_client.generate_code(_prompt, max_tokens=1024)
                    # Strip markdown fences the model may emit despite the instruction
                    import re as _re_code
                    _req_code = _re_code.sub(
                        r'```(?:python)?\s*\n?', '', _req_code
                    ).strip().rstrip('`').strip()
                    # Validate syntax; fall back to stub on parse error
                    import ast as _ast_code
                    try:
                        _ast_code.parse(_req_code)
                    except SyntaxError:
                        _req_code = ""
                if not _req_code:
                    _req_code = f"# Requirement: {req}\nprint('Requirement: {req[:80]}')\n"
                add(
                    "execute_code",
                    {"code": _req_code},
                    reason=f"Brief requirement: {req[:100]}",
                )

        if wants_notebook:
            add("build_notebook",
                reason="Build complete .ipynb report with all findings")
    else:
        add("visualize", {"plot_type": "distributions"},
            reason="Feature distributions")
        add("visualize", {"plot_type": "correlation"},
            reason="Correlation heatmap")

    return steps


def _instruction_from_document(path: str, session: Session) -> str:
    """Read a document and return its text as the instruction."""
    from pathlib import Path as _Path
    p = _Path(path).expanduser().resolve()
    if not p.exists():
        return f"[document not found: {path}]"
    if p.suffix.lower() == ".pdf":
        from agent.api.pdf_reader import extract
        summary = extract(p.read_bytes())
        if summary.mentioned_target and not session.target:
            session.target = summary.mentioned_target
        return summary.text
    return p.read_text(errors="replace")


def _eda_scope_enrichment(
    task_spec: dict,
    df: Any,
    kernel: Any,
    emit_fn,
) -> dict:
    """Enrich TASK_SPEC with lightweight data statistics before planning.

    Computes n_samples, class imbalance, feature counts, and derives a
    recommended_models list and eval_metric override based on the actual data.
    The enriched spec is written back to kernel.namespace["TASK_SPEC"].
    """
    try:
        import numpy as _np

        n_samples: int = len(df)
        n_features: int = df.shape[1] - 1  # approximate (excl. target)
        has_nulls: bool = bool(df.isnull().any().any())

        numeric_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[_np.number]).columns.tolist()

        # ── Class imbalance ──────────────────────────────────────────────────
        target_col = task_spec.get("target_column", "")
        class_imbalance_ratio: float = 0.0
        minority_class_count: int = 0
        task_type = task_spec.get("task_type", "")

        if target_col and target_col in df.columns and "classif" in task_type:
            vc = df[target_col].value_counts()
            if len(vc) >= 2:
                minority_class_count = int(vc.iloc[-1])
                majority_class_count = int(vc.iloc[0])
                class_imbalance_ratio = round(
                    minority_class_count / majority_class_count, 4)

        # ── Model recommendation ─────────────────────────────────────────────
        # Priority: user hint from task_spec > data-driven defaults.
        # When aggregation is needed, the pre-aggregation target column does not
        # reflect post-aggregation cardinality — trust task_type from understand_task.
        _agg_needed = task_spec.get("aggregation_needed", False)
        recommended_models: list[str] = task_spec.get("recommended_models", [])
        if not recommended_models:
            if "regression" in task_type:
                if n_samples > 50_000:
                    recommended_models = ["lgbm", "xgb", "ridge", "elasticnet"]
                elif n_samples > 5_000:
                    recommended_models = ["lgbm", "rf", "ridge", "elasticnet"]
                else:
                    recommended_models = ["rf", "ridge", "elasticnet", "lasso"]
            elif "classif" in task_type:
                # Only count n_classes from raw data when aggregation is NOT needed
                if _agg_needed:
                    n_classes = 2 if "binary" in task_type else 3
                else:
                    n_classes = (
                        df[target_col].nunique()
                        if target_col and target_col in df.columns else 2
                    )
                if n_samples > 50_000:
                    recommended_models = ["lgbm", "xgb", "rf", "lr"]
                elif n_samples > 5_000:
                    recommended_models = ["lgbm", "rf", "lr", "extra_trees"]
                else:
                    recommended_models = ["rf", "lr", "extra_trees"]
                if n_classes > 2:
                    recommended_models = [m for m in recommended_models
                                          if m not in ("lgbm", "xgb")] + ["lgbm", "xgb"]
            else:
                recommended_models = ["rf", "lgbm", "lr"]

        # ── Metric override ──────────────────────────────────────────────────
        # Override LLM-extracted metric only when data makes it clearly wrong.
        # When aggregation is needed, skip raw n_classes check — the target's
        # pre-aggregation distribution is not the modelling target distribution.
        current_metric = task_spec.get("evaluation_metric", "")
        if "classif" in task_type and not _agg_needed:
            n_classes = (
                df[target_col].nunique()
                if target_col and target_col in df.columns else 2
            )
            if n_classes > 2:
                # Multiclass: ensure we don't use binary metrics
                if current_metric in ("pr_auc", "average_precision", "roc_auc"):
                    task_spec["evaluation_metric"] = "roc_auc_ovr_weighted"
            else:
                # Binary: if highly imbalanced and metric is accuracy, upgrade
                if (class_imbalance_ratio > 0 and class_imbalance_ratio < 0.2
                        and current_metric in ("accuracy", "", "auto")):
                    task_spec["evaluation_metric"] = "pr_auc"

        # ── Write enriched fields into task_spec ─────────────────────────────
        task_spec.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "has_nulls": has_nulls,
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(cat_cols),
            "class_imbalance_ratio": class_imbalance_ratio,
            "minority_class_count": minority_class_count,
            "recommended_models": recommended_models,
        })
        kernel.namespace["TASK_SPEC"] = task_spec

        # ── Emit scope summary ───────────────────────────────────────────────
        imb_note = ""
        if class_imbalance_ratio > 0:
            imb_note = (f"  Imbalance : minority={minority_class_count} "
                        f"({class_imbalance_ratio:.1%} of majority)\n")
        emit_fn(
            f"📊 EDA scope enrichment\n"
            f"  Samples   : {n_samples:,}  |  Features: {n_features}\n"
            f"  Nulls     : {'yes' if has_nulls else 'none'}  |  "
            f"Numeric: {len(numeric_cols)}  Cat: {len(cat_cols)}\n"
            f"{imb_note}"
            f"  Models    : {', '.join(recommended_models)}\n"
            f"  Metric    : {task_spec.get('evaluation_metric', '?')}",
            tool="eda_scope", level="ok",
        )
    except Exception as _exc:  # noqa: BLE001
        emit_fn(f"⚠ EDA scope enrichment skipped: {_exc}", tool="eda_scope", level="warn")

    return task_spec


# ── intent → tool routing ─────────────────────────────────────────────────────

# Tools whose failure invalidates every downstream step (they produce the
# TASK_SPEC / aggregated frame / feature set that later tools consume).
_DEPENDENCY_TOOLS = frozenset({
    "understand_task", "aggregate_data", "feature_engineering",
    "data_analysis_agent",
})

# Tools that operate on a loaded dataframe — meaningless without `df`.
_DATA_REQUIRING_TOOLS = frozenset({
    "eda_profile", "deep_eda", "quality_check", "aggregate_data",
    "infer_target", "mutual_information", "feature_engineering",
    "train_model", "visualize", "data_analysis_agent", "ml_strategy_agent",
})
# The subset whose cost (minutes on local hardware) justifies asking about the
# target/goal when the task spec is missing and the request is vague.
_HEAVY_BUILD_TOOLS = frozenset({
    "train_model", "aggregate_data", "feature_engineering",
    "ml_strategy_agent", "build_notebook",
})
# Words that signal the user already stated a goal/target — suppress the
# "what do you want" clarification when present.
_GOAL_HINT_WORDS = frozenset({
    "predict", "classify", "target", "forecast", "detect", "churn",
    "fraud", "price", "regression", "classification", "label", "outcome",
})


def _needs_clarification(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    tool_plan: list[tuple[str, dict]],
) -> str | None:
    """Return a clarifying question if an expensive build lacks inputs.

    Deterministic (no LLM). Fires only when the planned tools actually need
    data / a goal that isn't present — QA and status queries pass through
    untouched because their plans don't contain data-requiring tools.
    """
    if not tool_plan:
        return None
    planned = {name for name, _ in tool_plan}

    df = kernel.namespace.get("df")
    has_data = df is not None or bool(getattr(session, "data_paths", None))

    # Case 1: a data-requiring tool is planned but nothing is loaded.
    if planned & _DATA_REQUIRING_TOOLS and not has_data:
        return (
            "I don't see a dataset loaded yet. Upload a CSV/Parquet file "
            "(or set a work directory containing one), and tell me what you'd "
            "like to predict or analyse — then I'll get started."
        )

    # Case 2: heavy build planned, data present, but no task spec AND the
    # request gives no goal/target hint → ask rather than guess.
    if planned & _HEAVY_BUILD_TOOLS and has_data:
        has_spec = bool(kernel.namespace.get("TASK_SPEC"))
        has_target = bool(getattr(session, "target", None)) or bool(
            kernel.namespace.get("TARGET"))
        has_brief = bool(getattr(session, "brief", None))
        lower = instruction.lower()
        gives_goal = any(w in lower for w in _GOAL_HINT_WORDS)
        if not (has_spec or has_target or has_brief or gives_goal):
            cols = []
            try:
                cols = list(df.columns)[:15] if df is not None else []
            except Exception:
                pass
            col_hint = f" Available columns: {', '.join(map(str, cols))}." if cols else ""
            return (
                "Before I build a model — which column should I predict "
                "(the target), and is this classification or regression?"
                + col_hint
            )
    return None


def _classify_tools(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
) -> list[tuple[str, dict]]:
    """Map a natural-language instruction to an ordered list of (tool_name, args).

    Uses deterministic keyword matching first (fast, no LLM call needed for common
    requests), with a single constrained LLM call as fallback for ambiguous cases.
    """
    plan: list[tuple[str, dict]]
    txt = instruction.lower()

    has_brief    = bool((getattr(session, "brief", None) or "").strip())
    has_data     = kernel.namespace.get("df") is not None
    has_task_spec = bool(kernel.namespace.get("TASK_SPEC"))

    # ── keyword sets ──────────────────────────────────────────────────────────
    _scope_w  = {"read", "understand", "explain", "outline", "scope", "problem",
                 "brief", "pdf", "instruction", "document", "exercise", "task",
                 "requirement", "analyse this", "analyze this",
                 "what does it say", "what is the task", "what are we",
                 "this is", "here is", "i have uploaded"}
    _eda_w    = {"eda", "explore", "profile", "insight", "distribution",
                 "statistical", "what is in", "what's in", "tell me about the data",
                 "summarise the data", "summarize the data",
                 "analyse data", "analyze data", "analyse the data", "analyze the data",
                 "key insight", "read the data", "load the data",
                 "data overview", "explore the data", "profile the data"}
    _qual_w   = {"quality", "missing", "outlier", "skew", "check the data",
                 "data issue", "null", "data problem"}
    _mi_w     = {"mutual information", "mi score", "feature importance"}
    _fe_w     = {"feature engineer", "encode", "transform", "preprocess",
                 "select feature", "feature select"}
    _train_w  = {"train", "model", "predict", "classif", "regression", "ml",
                 "machine learning", "algorithm", "fit model", "evaluate",
                 "performance", "accuracy", "auc", "f1", "deep learning",
                 "neural", "xgboost", "random forest", "gradient boost"}
    _nb_w     = {"notebook", "build report", "generate report",
                 "final report", "summarise findings", "summarize findings"}
    _dep_w    = {"deploy", "ci/cd", "productionise", "productionize",
                 "package", "serving", "endpoint", "dockerfile",
                 "deployment", "export model", "save model", "deployment ready",
                 "deployment-ready"}
    _mon_w    = {"monitor", "drift", "degradation", "retrain", "alert",
                 "performance over time", "data shift", "continuously monitor"}

    def _hit(words: set) -> bool:
        return any(w in txt for w in words)

    # ── -3. Research trigger: live-web questions about topics/libraries/versions ─
    # Route to web_research when the user explicitly asks to search the web, or
    # asks about current/latest information that may post-date the model's
    # training cutoff. Guarded so build verbs ("train the latest model") still
    # go to the pipeline, not the researcher.
    _web_verbs = ("research ", "look up", "search the web", "search online",
                  "web search", "google ", "find online", "look online",
                  "browse the web", "on the internet", "latest news")
    _currency_w = ("latest", "newest", "most recent", "state of the art",
                   "state-of-the-art", "sota", "up to date", "up-to-date",
                   "current version", "recent advance", "as of 202",
                   "cutting edge", "cutting-edge", "new library", "new libraries")
    _research_nouns = ("librar", "package", "framework", " tool", "version",
                       "release", " api", "method", "technique", "paper",
                       "approach", "algorithm", "best practice")
    _build_guard = ("train ", "build ", " fit ", "deploy", "aggregate",
                    "feature engineer", "preprocess", "notebook")
    if any(v in txt for v in _web_verbs) or (
            any(c in txt for c in _currency_w)
            and any(nn in txt for nn in _research_nouns)
            and not any(b in txt for b in _build_guard)):
        return [("web_research", {"query": instruction})]

    # ── -2. File-role clarification pending → any reply re-triggers understand_task ─
    # When file-role reasoning couldn't assign roles confidently it stores a
    # FILE_ROLE_CLARIFICATION flag in the kernel and surfaces a question to the user.
    # Any subsequent user message should re-run understand_task with the answer
    # as user_hint so the extractor can finalise role assignment.
    if kernel.namespace.get("FILE_ROLE_CLARIFICATION", {}).get("clarification_pending"):
        kernel.namespace.pop("FILE_ROLE_CLARIFICATION", None)
        return [("understand_task", {})]

    # NOTE: there used to be a keyword gate here ("plan", "strategy", ...) that
    # returned [] whenever the message contained the word "plan", so it could
    # bypass tool execution entirely — including for "plan AND PERFORM the EDA
    # in depth", which described real work and still got answered as prose
    # with nothing run. Patching it with a second keyword list to out-vote the
    # first ("perform", "execute", "do the", ...) was the wrong fix: it is the
    # same brittleness in the other direction, and there will always be a
    # phrasing neither list anticipates. It is deleted rather than patched.
    # Whether this is a pure "write me a plan" request or a "do it" request is
    # exactly the question run_reactive's _decide_action already answers with
    # an LLM call when nothing below matches — that is where this ambiguity
    # belongs, not in a keyword list here. An explicit EDA/analysis keyword
    # below is a fast-path to the SAME adaptive execution _decide_action would
    # reach, not a different decision, so nothing is lost by removing this.

    # ── 0. File confirmation messages ("this is the second file", "use this") ─
    # User is confirming an upload or pointing at a file just loaded.
    # Re-run understand_task so it picks up any newly loaded secondary data.
    _confirm_w = {"second file", "data file", "training file", "test file",
                  "use this file", "use this data", "this is the file",
                  "this is my data", "additional file", "other file",
                  "loaded the file", "uploaded the file"}
    if has_brief and (_hit(_confirm_w) or (
        _hit({"this is", "here is", "that is"}) and len(txt.split()) <= 8
    )):
        # Always re-run understand_task when a file confirmation arrives —
        # secondary data (df2, df3) may now be available that wasn't before.
        kernel.namespace.pop("TASK_SPEC", None)
        return [("understand_task", {})]

    # ── 1. Scope / brief reading ──────────────────────────────────────────────
    if has_brief and _hit(_scope_w) and not has_task_spec:
        return [("understand_task", {})]

    # ── 2. EDA / exploration → adaptive LLM-driven loop ──────────────────────
    # Exploration is not a fixed sequence of tools. The agent designs the EDA
    # itself: it profiles, reads what the profile revealed, and decides the next
    # analysis from that result (relationships, group-level aggregates, ratios),
    # rather than always running the same canned steps. The sentinel routes
    # run_reactive into _run_agentic_loop, where the model picks each tool from
    # the accumulating observations. A bare quality check stays deterministic —
    # it is one specific, well-defined tool, not open-ended exploration.
    if has_data:
        _eda_kw = _hit(_eda_w)
        _qual_kw = _hit(_qual_w)
        # A narrow, explicit quality check is one well-defined tool — keep it
        # fast and deterministic.
        if _qual_kw and not _eda_kw:
            return [("quality_check", {})]
        # Exploration / analysis intent is NOT bound to a keyword list. A
        # keyword hit is a cheap fast-yes; for every other phrasing the LLM
        # reads the request and decides. Either way it goes to the adaptive
        # loop, which designs the EDA and reasons about each result. The LLM
        # check is skipped for explicit build verbs (train/feature/deploy/
        # notebook/monitor), which have their own deterministic routes below.
        _explicit_build = (_hit(_train_w) or _hit(_fe_w) or _hit(_mi_w)
                           or _hit(_nb_w) or _hit(_dep_w) or _hit(_mon_w))
        if _eda_kw or (not _explicit_build
                       and _is_exploration_request(
                           instruction, session, kernel, llm_client)):
            return [(_ADAPTIVE_SENTINEL, {})]

    # ── 2.5. Deep analysis pipeline ──────────────────────────────────────────
    # User explicitly requests specialist data analysis before model training.
    # data_analysis_agent → ml_strategy_agent produces DATA_ANALYSIS_REPORT +
    # ML_PLAN which train_model reads for informed model/hyperparam selection.
    _deep_analysis_w = {
        "deep analysis", "comprehensive analysis", "full analysis",
        "data analysis agent", "analyse then train", "analyze then train",
        "analysis agent", "ml strategy", "intelligent train",
        "smart train", "informed train", "data driven train",
        "analyse and train", "analyze and train",
    }
    if has_data and _hit(_deep_analysis_w):
        plan = []
        if not has_task_spec and has_brief:
            plan.append(("understand_task", {}))
        plan.extend([
            ("data_analysis_agent", {}),
            ("ml_strategy_agent", {}),
            ("feature_engineering", {"top_k": 20}),
            ("train_model", {"task_type": "auto"}),
        ])
        return plan

    # ── 3. Full ML pipeline ───────────────────────────────────────────────────
    # Standard pipeline now runs data_analysis_agent before train_model so that
    # model selection and hyperparameters are informed by the data characteristics.
    if has_data and _hit(_train_w):
        plan = []
        if not has_task_spec and has_brief:
            plan.append(("understand_task", {}))
        has_analysis = bool(kernel.namespace.get("DATA_ANALYSIS_REPORT"))
        if not has_analysis:
            plan.append(("data_analysis_agent", {}))
        plan.extend([
            ("feature_engineering", {"top_k": 20}),
            ("train_model", {"task_type": "auto"}),
        ])
        return plan

    # ── 4. Feature engineering only ──────────────────────────────────────────
    if has_data and _hit(_fe_w):
        return [("feature_engineering", {"top_k": 20})]

    # ── 5. Mutual information only ───────────────────────────────────────────
    if has_data and _hit(_mi_w):
        return [("mutual_information", {})]

    # ── 6. Notebook / final report ───────────────────────────────────────────
    if _hit(_nb_w):
        return [("build_notebook", {})]

    # ── 7. Deployment packaging ──────────────────────────────────────────────
    if _hit(_dep_w):
        deploy_code = (
            "from pathlib import Path\n"
            "import joblib, textwrap\n"
            "_model = globals().get('best_model') or globals().get('model')\n"
            "if _model is not None:\n"
            "    _out = WORK_DIR / 'deployment'\n"
            "    _out.mkdir(exist_ok=True)\n"
            "    joblib.dump(_model, _out / 'model.pkl')\n"
            "    print(f'Model saved → {_out / \"model.pkl\"}')\n"
            "    _script = textwrap.dedent(\"\"\"\n"
            "        import joblib, pandas as pd\n"
            "        model = joblib.load('model.pkl')\n"
            "        def predict(features: dict) -> dict:\n"
            "            df = pd.DataFrame([features])\n"
            "            return {'prediction': int(model.predict(df)[0]),\n"
            "                    'probability': float(model.predict_proba(df).max())}\n"
            "    \"\"\")\n"
            "    (_out / 'inference.py').write_text(_script.strip())\n"
            "    print('Inference script → deployment/inference.py')\n"
            "else:\n"
            "    print('No trained model in kernel — run train_model first.')\n"
        )
        return [("execute_code", {"code": deploy_code})]

    # ── 8. Monitoring / drift ────────────────────────────────────────────────
    if _hit(_mon_w):
        monitor_code = (
            "import pandas as pd\n"
            "_model = globals().get('best_model') or globals().get('model')\n"
            "if _model is not None and 'df' in dir():\n"
            "    import numpy as np\n"
            "    _num = df.select_dtypes(include='number').fillna(0)\n"
            "    _preds = _model.predict(_num)\n"
            "    print('=== Monitoring: prediction distribution ===')\n"
            "    print(pd.Series(_preds).value_counts().to_dict())\n"
            "    print('Compare against training distribution to detect drift.')\n"
            "else:\n"
            "    print('Train a model first, then run monitoring.')\n"
        )
        return [("execute_code", {"code": monitor_code})]

    # ── 9. Direct Q&A — factual/conversational queries → LLM answer, no tools ──
    # Any query that looks like a question or conversation but hasn't matched a
    # tool-triggering keyword above should be answered directly by the LLM
    # (via classify_intent → "qa" → answer() path in run_reactive).
    # This prevents _llm_classify_tools from recommending tools for pure questions.
    _question_w = {
        # Wh-questions
        "what is", "what are", "what was", "what were", "what does",
        "what do", "what did", "what will", "what's", "whats",
        "how many", "how much", "how does", "how do", "how did",
        "how would", "how should", "how can", "how come",
        "which", "when", "where", "why", "who",
        # Imperative informational requests
        "tell me", "show me", "explain", "describe",
        "can you tell", "can you explain", "could you", "would you",
        "give me", "list", "summarise what", "summarize what",
        # Memory / reflection
        "what did we", "what have we", "what happened", "remind me",
        "what was the result", "what were the results", "what did you",
        "what have you", "do you remember", "recall",
        # Session state questions
        "is the data", "is there a", "have you", "did you",
        "what target", "what metric", "what column",
        "what model", "what score", "what accuracy",
    }
    # Action verbs that indicate the user WANTS a tool to run (not Q&A)
    _action_imperatives = {
        "run", "execute", "start", "build", "train", "create",
        "generate", "produce", "perform", "do the", "do an",
        "analyse", "analyze", "explore the data", "profile the data",
        "fit", "deploy", "export", "save", "load the data",
    }
    if _hit(_question_w) and not _hit(_action_imperatives):
        return []

    # Short message (≤ 6 words) with no action verbs → treat as conversational
    _all_action_w = _action_imperatives | _train_w | _eda_w | _fe_w | _mi_w | _nb_w | _dep_w
    if len(txt.split()) <= 6 and not _hit(_all_action_w):
        return []

    # ── 10. LLM fallback for ambiguous instructions ───────────────────────────
    return _llm_classify_tools(instruction, session, kernel, llm_client)


def _llm_classify_tools(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
) -> list[tuple[str, dict]]:
    """Constrained LLM call: returns comma-separated tool names or 'none'."""
    has_data  = kernel.namespace.get("df") is not None
    has_brief = bool((getattr(session, "brief", None) or "").strip())

    ctx: list[str] = []
    if has_brief:
        ctx.append("A task brief/PDF is loaded.")
    if has_data:
        _df = kernel.namespace["df"]
        ctx.append(f"Data loaded: {_df.shape[0]} rows × {_df.shape[1]} cols.")
    context = " ".join(ctx) or "Nothing loaded yet."

    tool_names = sorted(TOOL_REGISTRY.keys())
    prompt = (
        "/no_think\n"
        f'User says: "{instruction}"\n'
        f"Context: {context}\n\n"
        "First decide: is this a question/conversation that can be answered directly from "
        "the session context, OR does it require running a data-science tool?\n\n"
        "Answer 'none' (no tools) when the user is:\n"
        "  - asking a factual question about the data, task, columns, metrics, or results\n"
        "  - having a conversation, giving feedback, or acknowledging something\n"
        "  - asking for a plan, strategy, explanation, or analysis overview\n"
        "  - asking 'what is...', 'how many...', 'which...', 'tell me about...'\n\n"
        "Only suggest tools when the user explicitly asks to:\n"
        "  - run an analysis, profile/explore the data, train/evaluate a model\n"
        "  - build a notebook/report, deploy, or generate new artefacts\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        "Reply with ONLY comma-separated tool names, or reply: none"
    )
    try:
        raw = llm_client._generate(prompt, temperature=0.0, max_tokens=60)
        raw = _strip_thinking(raw).strip().lower()
        if not raw or raw == "none":
            return []
        names = [n.strip() for n in re.split(r"[,\s]+", raw) if n.strip()]
        return [(n, {}) for n in names if n in TOOL_REGISTRY]
    except Exception:
        return []


def _ensure_task_understood(session: Session, kernel: SessionKernel,
                            llm_client: Any, emit: Any) -> list[StepResult]:
    """Read and understand the brief before exploring, when one exists.

    With a brief/instructions document loaded, the agent should know what the
    task is, and therefore what kind and depth of EDA to do, before it touches
    the data. So understand_task runs first and its spec lands in the kernel,
    where _data_context surfaces it to the exploration loop as the task focus.
    With no brief, this is a no-op and exploration starts straight on the data.
    """
    has_brief = bool((getattr(session, "brief", None) or "").strip())
    has_spec = bool(kernel.namespace.get("TASK_SPEC"))
    if not has_brief or has_spec:
        return []
    tool = TOOL_REGISTRY.get("understand_task")
    if tool is None:
        return []
    if emit:
        emit("📄 Reading the task instructions first, to shape the analysis…",
             tool="understand_task")
    t0 = time.time()
    try:
        out = tool.execute({}, session=session, kernel=kernel, llm_client=llm_client)
    except Exception as exc:  # noqa: BLE001
        out = ToolOutput(success=False, text="", error=f"Tool crashed: {exc}")
    elapsed = round(time.time() - t0, 2)
    if out.success:
        spec = kernel.namespace.get("TASK_SPEC", {})
        df = kernel.namespace.get("df")
        if df is not None and spec:
            _eda_scope_enrichment(spec, df, kernel, emit)
        if emit:
            desc = (spec or {}).get("task_description", "task")
            emit(f"✓ Task understood ({elapsed}s): {str(desc)[:120]}",
                 tool="understand_task", level="ok")
    elif emit:
        emit(f"⚠ Could not read the brief ({_error_headline(out.error)}); "
             "exploring the data directly.", tool="understand_task", level="warn")
    return [StepResult(step=0, tool="understand_task", args={},
                       reason="Understand the task before exploring",
                       output=out, elapsed_s=elapsed)]


def _ensure_work_planned(session: Session, kernel: SessionKernel,
                         llm_client: Any, emit: Any) -> list[StepResult]:
    """Plan the work from the brief and the data before any analysis runs.

    Whether a request needs a plan is the model's call, made when it interprets
    the request (TaskIntent.plan_first). When it does and no plan exists yet,
    work_plan runs here, so the EDA starts from questions the plan derived
    from the task and the actual columns (surfaced by _data_context) instead
    of the loop guessing its first steps, e.g. hunting for a target the task
    does not have. A failed plan is reported and exploration continues.
    """
    intent = getattr(kernel, "task_intent", None)
    if intent is None or not getattr(intent, "plan_first", False):
        return []
    if isinstance(kernel.namespace.get("WORK_PLAN"), dict):
        return []
    tool = TOOL_REGISTRY.get("work_plan")
    if tool is None or kernel.namespace.get("df") is None:
        return []
    if emit:
        emit("Planning the work from the instructions and the data before "
             "exploring…", tool="work_plan")
    t0 = time.time()
    try:
        out = tool.execute({}, session=session, kernel=kernel, llm_client=llm_client)
    except Exception as exc:  # noqa: BLE001
        out = ToolOutput(success=False, text="", error=f"Tool crashed: {exc}")
    elapsed = round(time.time() - t0, 2)
    if emit:
        if out.success:
            plan = kernel.namespace.get("WORK_PLAN") or {}
            n_q = (len(plan.get("eda_questions_task") or [])
                   + len(plan.get("eda_questions_independent") or []))
            emit(f"✓ Plan written ({elapsed}s): {str(plan.get('objective', ''))[:160]} "
                 f"({n_q} EDA questions)", tool="work_plan", level="ok")
        else:
            emit(f"⚠ Could not write a plan ({_error_headline(out.error)}); "
                 "exploring without one.", tool="work_plan", level="warn")
    return [StepResult(step=0, tool="work_plan", args={},
                       reason="Plan the work before exploring",
                       output=out, elapsed_s=elapsed)]


def _with_plan_and_agenda(reply: str, steps: list[StepResult],
                          kernel: SessionKernel) -> str:
    """Put the stored plan and the agenda's coverage in the reply.

    When a plan was written or revised this turn, the user gets that plan as
    stored, not the model's retelling of it, which differed from the plan the
    agent then followed. After any analysis, a closing line says how many
    agenda questions were answered and which are still open.
    """
    ns = kernel.namespace
    parts: list[str] = []
    plan = ns.get("WORK_PLAN")
    if isinstance(plan, dict) and any(s.tool == "work_plan" and s.output.success
                                      for s in steps):
        from agent.api.agent_tools import _plan_to_markdown
        raw_spec = ns.get("TASK_SPEC")
        md = _plan_to_markdown(plan, raw_spec if isinstance(raw_spec, dict) else {})
        if md.startswith("# Analysis plan"):
            md = md.split("\n", 1)[1].lstrip()
        parts += [f"## Work plan (version {plan.get('version', 1)})\n\n{md}",
                  f"## Results\n\n{reply}"]
    else:
        parts.append(reply)
    agenda = ns.get("EDA_AGENDA")
    analysed = any(s.output.success and s.tool not in ("work_plan", "understand_task")
                   for s in steps)
    if isinstance(agenda, list) and agenda and analysed:
        open_items = [a for a in agenda if a.get("status") != "answered"]
        line = (f"EDA agenda: {len(agenda) - len(open_items)} of {len(agenda)} "
                "questions answered.")
        if open_items:
            line += " Still open: " + "; ".join(
                f"[{a['id']}] {a['question'][:90]}" for a in open_items[:6])
        parts.append(line)
    return "\n\n".join(parts)


def _run_guarded_adaptive(
    instruction: str, session: Session, kernel: SessionKernel, llm_client: Any,
    *, emit: Any, ltm: Any, multiagent: bool | None,
) -> tuple[str, list[StepResult], list[str]]:
    """Read the task (if a brief exists) then run the adaptive tool loop.

    This is the ONE guarded execution path for "the user wants the analysis
    actually done": every caller that decides that — whether by an explicit
    keyword fast-path or by the LLM's own act/answer judgement in
    _decide_action — routes through here, so both get the same guarantees:
    the brief is read first so exploration is task-informed, and a reply is
    never presented as a real result unless a tool actually produced it.
    Before this was unified, the keyword fast-path got both guards and the
    LLM-decided path got neither, so identical requests behaved differently
    depending on which route happened to notice them.
    """
    pre_steps = _ensure_task_understood(session, kernel, llm_client, emit)
    if any(not step.output.success or step.output.artifacts.get("clarification_pending") for step in pre_steps):
        return "Task instructions could not be understood; execution stopped.", pre_steps, []
    pre_steps += _ensure_work_planned(session, kernel, llm_client, emit)
    if emit:
        emit("Exploring adaptively — the agent will design the analysis "
             "from what each step reveals.", tool="router")
    reply, loop_steps, loop_figs = _run_agentic_loop(
        instruction, session, kernel, llm_client,
        emit=emit, ltm=ltm, multiagent=multiagent,
    )
    all_steps = pre_steps + loop_steps
    if not reply.strip():
        reply = ("(No response generated — the analysis did not run. "
                 "Please try again.)")
    # No-fabrication guard: a request that produced no executed tool has not
    # analysed anything. Whatever the model wrote is ungrounded, so flag it
    # rather than letting invented numbers, plots or saved files read as real
    # results.
    elif not any(s.output.success for s in loop_steps):
        if emit:
            emit("No analysis tool ran this turn — flagging the reply as "
                 "not computed from the data.", tool="router", level="warn")
        reply = (
            "⚠ I could not run the analysis this turn, so nothing below was "
            "computed from your data and no files were written. Please try "
            "again.\n\n" + reply
        )
    return reply, all_steps, loop_figs


# ── conversational runner (intent-classify → execute → synthesise) ────────────

def run_reactive(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    max_tool_calls: int = 12,
    ltm: Any = None,
    multiagent: bool | None = None,
    allow_writes: bool | None = None,
) -> AgentRunResult:
    """Classify intent → execute tool plan → synthesise reply.

    The LLM is called for two focused, constrained tasks only:
      1. Intent classification (returns tool names — short, constrained output).
      2. Synthesis (writes a natural-language response from tool results).

    This replaces the open-ended ReAct loop which asked local models to produce
    structured tool-call JSON — a format they cannot reliably follow.
    """
    from agent.api import progress as _progress
    from agent.api.agent_tools import get_tool_schemas
    from agent.api.intent import interpret

    started = time.time()
    if getattr(session, "owner_id", None):
        ltm = None
    def _emit(msg: str, tool: str = "", level: str = "info") -> None:
        if job_id:
            _progress.emit(job_id, msg, tool=tool, level=level)

    from agent.api import trace
    trace.new_turn(getattr(session, "session_id", ""), instruction)
    kernel.remaining_tool_calls = max_tool_calls
    kernel.turn_deadline = time.monotonic() + float(os.environ.get("DSAGENT_JOB_TIMEOUT_S", "900"))
    # Each turn starts from the explicit UI policy, never a previous worker's mode.
    kernel.namespace["FILE_WRITE_MODE"] = "ask" if allow_writes else "suggest"
    _emit("Interpreting your request and constraints…", tool="intent")
    prior_intent = getattr(kernel, "task_intent", None)
    context = _data_context(session, kernel) + "\nFULL BRIEF:\n" + (session.brief or "")
    if prior_intent is not None:
        context += "\nPREVIOUS TASK (reconcile with latest request):\n" + prior_intent.model_dump_json()
    # Choosing capabilities needs each tool's name and purpose, not its full
    # parameter schema: that halved the prompt (~4.4K -> ~2.3K tokens) on the
    # call that runs before every turn. Arguments are validated at execution.
    intent = interpret(instruction, context=context,
                       history=[{"role": m.role, "content": m.text} for m in session.messages],
                       tools=[{"name": t["name"], "description": t.get("description", "")}
                              for t in get_tool_schemas()],
                       llm=llm_client)
    # Policy lives outside the executable Python namespace.
    kernel.task_intent = intent
    kernel.task_instruction = instruction
    if intent.isolation is not None:
        session.isolation_mode = intent.isolation
    steps: list[StepResult] = []
    figures: list[str] = []
    if intent.decision == "clarify":
        reply = intent.clarification or "Please clarify the intended outcome."
    elif intent.decision == "answer":
        context = _data_context(session, kernel)
        context += "\nDo not claim new work, metrics, or files. No tools ran this turn."
        reply = llm_client.answer(instruction, context=context,
                    messages=_build_qa_messages(session, llm_client, ltm, instruction)).text
    else:
        reply, steps, figures = _run_guarded_adaptive(
            instruction, session, kernel, llm_client, emit=_emit,
            ltm=ltm, multiagent=multiagent)
        reply = _grounded_reply(reply, steps, llm_client)
        reply = _with_plan_and_agenda(reply, steps, kernel)
    reply = _strip_thinking(reply)
    _emit("Done.", level="ok")
    # Persist computed evidence, not unchecked generated prose, in shared memory.
    _store_to_ltm(ltm, session, instruction, "", steps)
    return AgentRunResult(goal=intent.objective, reply=reply, steps=steps,
                          figures=figures, elapsed_s=round(time.time() - started, 2))


# ── main runner ───────────────────────────────────────────────────────────────

def run(
    instruction: str,
    session: Session,
    kernel: SessionKernel,
    llm_client: Any,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
) -> AgentRunResult:
    """Full Plan → Execute → [Re-plan] → Synthesise loop."""
    from agent.api import progress as _progress

    def _emit(msg: str, tool: str = "", level: str = "info") -> None:
        if job_id:
            _progress.emit(job_id, msg, tool=tool, level=level)

    started = time.time()
    run_id = run_id or uuid.uuid4().hex[:8]
    all_figures: list[str] = []
    step_results: list[StepResult] = []

    # ── 0. Parse instruction source ──────────────────────────────────────
    _stripped = instruction.strip()
    if (len(_stripped) < 300
            and _stripped.startswith("/")
            and "." in _stripped.split("/")[-1]):
        _emit("Reading instruction document…", tool="read_instructions")
        doc_text = _instruction_from_document(_stripped, session)
        instruction = f"Instructions from document:\n{doc_text[:3000]}"

    data_ctx = _data_context(session, kernel)
    work_dir_str = str(session.work_dir) if session.work_dir else "(not set)"

    # ── 1. Pre-plan phase: read & understand instructions first ──────────
    # If a brief / instructions document is loaded, ALWAYS read it before
    # planning so the plan reflects what the task actually asks for.
    # understand_task runs here — not as a pipeline step — so the planner
    # has the full task spec before it decides what steps to run.

    task_spec: dict = {}
    has_brief = bool(session.brief and session.brief.strip())

    if has_brief:
        _emit("📄 Reading task instructions…", tool="understand_task")
        understand_tool = TOOL_REGISTRY.get("understand_task")
        if understand_tool:
            t0 = time.time()
            ut_out = understand_tool.execute(
                {}, session=session, kernel=kernel, llm_client=llm_client)
            elapsed = round(time.time() - t0, 2)
            task_spec = kernel.namespace.get("TASK_SPEC", {})
            # Surface validation failures from the 4-layer pipeline
            _vfails = task_spec.get("validation_failures", [])
            if _vfails:
                for _vf in _vfails:
                    _emit(f"⚠ Spec validation: {_vf}", tool="understand_task", level="warn")
            if not task_spec.get("is_valid", True):
                _emit(
                    "⚠ Task spec has validation issues — plan may be degraded. "
                    "Check the warnings above and consider re-uploading a clearer brief.",
                    tool="understand_task", level="warn",
                )
            confidence = ut_out.artifacts.get("confidence", 0.5)
            brief_type = ut_out.artifacts.get("brief_type", "unknown")

            # Emit a human-readable summary of what the instructions say
            desc = task_spec.get("task_description", "?")
            ttype = task_spec.get("task_type", "?")
            metric = task_spec.get("evaluation_metric", "?")
            agg = task_spec.get("aggregation_needed", False)
            derived = task_spec.get("target_is_derived", False)
            conf_emoji = "✓" if confidence >= 0.7 else ("⚠" if confidence >= 0.5 else "✗")
            _emit(
                f"✓ Instructions understood ({elapsed}s)\n"
                f"  Brief type: {brief_type}  |  Confidence: {confidence:.0%} {conf_emoji}\n"
                f"  Task      : {desc}\n"
                f"  Type      : {ttype}  |  Metric: {metric}\n"
                f"  Aggregate : {'yes — group by ' + str(task_spec.get('aggregation_key','?')) if agg else 'no'}\n"
                f"  Target    : {'must be derived' if derived else task_spec.get('target_column','?')}",
                tool="understand_task", level="ok" if confidence >= 0.6 else "warn",
            )
            step_results.append(StepResult(
                step=0, tool="understand_task", args={},
                reason="Read and understand task instructions from brief/document",
                output=ut_out, elapsed_s=elapsed,
            ))
            all_figures.extend(ut_out.figures)

    # ── 1.5. EDA-informed scope enrichment ───────────────────────────────
    # After understand_task sets the initial TASK_SPEC from the brief, enrich
    # it with actual data statistics so the planner has the full picture before
    # deciding which steps, models, and metrics to use.
    # This runs even when no brief was loaded (data-only sessions).
    df_for_eda = kernel.namespace.get("df")
    if df_for_eda is not None:
        task_spec = _eda_scope_enrichment(task_spec, df_for_eda, kernel, _emit)

    # ── 2. Planning phase: build steps using task spec + LLM ─────────────
    # Now that we know what the task asks for, ask the LLM to build a proper
    # plan. The LLM receives the full task_spec so it can customise steps
    # (e.g. add aggregate_data with correct args, pick evaluation metric, etc.)
    # If LLM fails or produces a trivial plan, fall back to smart keyword plan.

    steps: list[dict] = []
    goal: str = instruction[:100]

    # Build an enriched brief snippet that includes extracted task spec
    brief_snip = _build_brief_context(session.brief, task_spec)

    # ── Planning strategy ────────────────────────────────────────────────
    # • task_spec available (understand_task ran)  → keyword plan from spec
    #   (most reliable — no LLM JSON round-trip, args are always correct)
    # • No spec + ambiguous instruction            → try LLM planner first
    # • No spec + clear keyword instruction        → keyword plan directly
    # We avoid the LLM planner for DS pipeline tasks because the local model
    # consistently produces trivial or malformed JSON plans for these prompts.

    _emit("Building execution plan…", tool="planner")

    if task_spec:
        # We know exactly what to do — build the plan from spec directly
        steps = _keyword_plan_from_spec(instruction, session, task_spec,
                                        llm_client=llm_client)
        goal = (f"DS pipeline: {task_spec.get('task_description', 'full analysis')}"
                f" ({task_spec.get('task_type', 'auto')})")

    elif not _is_keyword_confident(instruction):
        # Genuinely ambiguous — worth asking the LLM
        planner_prompt = (
            f"User instruction:\n{instruction}\n\n"
            f"Conversation history (last 4 turns):\n{_recent_history(session)}"
        )
        planner_system = _PLANNER_SYSTEM.format(
            tool_list=_TOOL_LIST,
            data_context=data_ctx,
            brief=brief_snip,
            work_dir=work_dir_str,
            max_steps=MAX_STEPS,
        )
        try:
            raw_plan = llm_client._generate(
                planner_prompt, system=planner_system,
                temperature=0.0, max_tokens=1200,
                json_mode=True,
            )
            plan_obj = _extract_json(_strip_thinking(raw_plan))
            if isinstance(plan_obj, dict):
                goal = plan_obj.get("goal", instruction[:80])
            llm_steps = _parse_plan(raw_plan, MAX_STEPS)
            if _is_real_plan(llm_steps):
                steps = llm_steps
                _emit(f"LLM plan: {len(steps)} steps", tool="planner")
        except Exception as exc:
            _emit(f"LLM planner error: {exc}", tool="planner", level="warn")

    # Fallback for clear keyword instructions or LLM failure
    if not _is_real_plan(steps):
        steps = _keyword_plan_from_spec(instruction, session, task_spec,
                                        llm_client=llm_client)
        goal = goal or _keyword_goal(instruction)

    if not steps:
        steps = [{"step": 1, "tool": "eda_profile",
                  "args": {}, "reason": "default: profile data"}]

    # Strip any understand_task that the LLM re-added (we already ran it above)
    steps = [s for s in steps if s["tool"] != "understand_task"]

    # Renumber cleanly
    for i, s in enumerate(steps):
        s["step"] = i + 1

    # Show the full plan
    _emit(f"Execution plan ({len(steps)} steps):", tool="planner")
    for s in steps:
        _emit(f"  Step {s['step']}: [{s['tool']}] {s['reason']}", tool="planner")

    # ── 3. Execute ───────────────────────────────────────────────────────
    _emit(f"Starting execution — {len(steps)} steps", tool="executor")
    replan_count = 0
    step_idx = 0

    while step_idx < len(steps):
        step = steps[step_idx]
        tool_name = step["tool"]
        tool = TOOL_REGISTRY.get(tool_name)

        if tool is None:
            _emit(f"Unknown tool '{tool_name}' — skipping", level="warn")
            step_idx += 1
            continue

        _emit(
            f"[{step_idx + 1}/{len(steps)}] Running {tool_name}: {step.get('reason', '')}",
            tool=tool_name,
        )
        t0 = time.time()
        try:
            out: ToolOutput = tool.execute(
                step.get("args", {}),
                session=session,
                kernel=kernel,
                llm_client=llm_client,
            )
        except Exception as exc:
            out = ToolOutput(success=False, text="",
                             error=f"Tool crashed: {exc}")

        elapsed = round(time.time() - t0, 2)

        if out.success:
            # Emit first line of output as a quick preview
            preview = (out.text or "").splitlines()[0][:120] if out.text else ""
            _emit(f"✓ {tool_name} done ({elapsed}s){' — ' + preview if preview else ''}",
                  tool=tool_name, level="ok")
        else:
            _emit(f"✗ {tool_name} failed ({elapsed}s): {_error_headline(out.error)}",
                  tool=tool_name, level="error")

        step_results.append(StepResult(
            step=step.get("step", step_idx + 1),
            tool=tool_name,
            args=step.get("args", {}),
            reason=step.get("reason", ""),
            output=out,
            elapsed_s=elapsed,
        ))
        all_figures.extend(out.figures)

        # ── 2a. Optional re-plan ──────────────────────────────────────────
        remaining = steps[step_idx + 1:]
        if remaining and replan_count < MAX_REPLAN and llm_client is not None:
            try:
                _emit("Checking if plan needs adjustment…", tool="planner")
                rp_prompt = _REPLAN_SYSTEM.format(
                    remaining=json.dumps(remaining, indent=2),
                    last_tool=tool_name,
                    last_output=out.text[:1500] if out.text else out.error[:500],
                )
                rp_raw = llm_client._generate(
                    rp_prompt, temperature=0.0, max_tokens=512)
                rp_clean = _strip_thinking(rp_raw).strip()
                if rp_clean.upper() != "KEEP" and rp_clean:
                    revised = _parse_plan(rp_clean, MAX_STEPS - step_idx - 1)
                    if revised:
                        steps = steps[:step_idx + 1] + revised
                        replan_count += 1
                        _emit(f"Plan revised — {len(revised)} steps updated",
                              tool="planner", level="warn")
                else:
                    _emit("Plan unchanged — continuing", tool="planner")
            except Exception:
                pass

        step_idx += 1

    # ── 3. Synthesise ────────────────────────────────────────────────────
    _emit("All steps done — synthesising final answer…", tool="synthesiser")
    findings = _build_findings_context(goal, instruction, step_results)
    reply = _synthesise(findings, llm_client)
    _emit("Done.", tool="synthesiser", level="ok")

    return AgentRunResult(
        goal=goal,
        reply=reply,
        steps=step_results,
        figures=all_figures,
        elapsed_s=round(time.time() - started, 2),
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _recent_history(session: Session, n: int = 10) -> str:
    """Return the last `n` messages as a readable string.

    Snippets are longer (800 chars) so key analysis outputs aren't truncated.
    """
    msgs = [m for m in session.messages if m.role in ("user", "assistant")][-n:]
    parts = []
    for m in msgs:
        role = "User" if m.role == "user" else "Assistant"
        snippet = m.text[:800] + "…" if len(m.text) > 800 else m.text
        parts.append(f"{role}: {snippet}")
    return "\n".join(parts) or "(start of conversation)"


def _chat_messages(session: Session, n: int = 12) -> list[dict]:
    """Build a proper Ollama /api/chat message array from session history.

    Returns the last `n` user/assistant turns as {"role", "content"} dicts.
    This is used for true multi-turn memory via the chat endpoint.
    """
    msgs = [m for m in session.messages if m.role in ("user", "assistant")][-n:]
    result = []
    for m in msgs:
        role = "user" if m.role == "user" else "assistant"
        # Keep more content for assistant messages — they hold analysis results
        max_len = 1200 if role == "assistant" else 400
        content = m.text[:max_len] + "…" if len(m.text) > max_len else m.text
        result.append({"role": role, "content": content})
    return result


def _build_qa_messages(session: Session, llm_client: Any, ltm: Any,
                       instruction: str) -> list[dict]:
    """Context messages for the QA path.

    Uses the memory-layer `build_context_messages` (summary buffer + LTM
    injection) when available; otherwise the fixed-window `_chat_messages`.
    Any failure degrades to the simple window — QA must never crash on
    memory issues.
    """
    if _MEMORY_OK:
        try:
            msgs = build_context_messages(
                session, llm_client, ltm=ltm, current_query=instruction)
            if msgs:
                return msgs
        except Exception:
            pass
    return _chat_messages(session)


def _recall_from_ltm(ltm: Any, instruction: str, session: Session) -> str:
    """Retrieve relevant past goals/solutions from long-term memory as a
    formatted block for prompt injection. Empty string when LTM is unavailable,
    isolation mode restricts it, or nothing relevant is found. Best-effort."""
    if ltm is None or getattr(session, "owner_id", None):
        return ""
    try:
        iso = bool(getattr(session, "isolation_mode", False))
        sid = getattr(session, "session_id", "") or ""
        return ltm.retrieve_as_context_string(
            query=instruction, session_id=sid, isolation_mode=iso)
    except Exception:  # noqa: BLE001
        return ""


def _store_to_ltm(ltm: Any, session: Session, instruction: str,
                  reply: str, step_results: list[StepResult]) -> None:
    """Persist this turn to long-term memory. Fully best-effort."""
    if ltm is None or getattr(session, "isolation_mode", False) or getattr(session, "owner_id", None):
        return
    sid = getattr(session, "session_id", "") or ""
    try:
        # Store the synthesised answer keyed to the question so future
        # semantic recall can surface it.
        if reply and not reply.startswith("("):
            ltm.store_synthesis(sid, f"Q: {instruction[:300]}\nA: {reply[:1500]}")
        # Store notable tool outputs (model results, task specs) for recall.
        for sr in step_results:
            if sr.output.success and sr.output.text and sr.tool in (
                "train_model", "understand_task", "data_analysis_agent",
                "ml_strategy_agent",
            ):
                ltm.store_tool_output(sid, sr.tool, sr.output.text[:1500])
    except Exception:
        pass  # LTM offline / embed model missing → non-fatal


def _build_findings_context(goal: str, instruction: str,
                            steps: list[StepResult]) -> str:
    parts = [f"Goal: {goal}", f"Instruction: {instruction}\n"]
    for sr in steps:
        status = "✓" if sr.output.success else "✗"
        parts.append(
            f"{status} [{sr.tool}] ({sr.elapsed_s}s)\n"
            f"  Reason: {sr.reason}\n"
            f"  Output: {(sr.output.text or sr.output.error)[:800]}"
        )
    return "\n\n".join(parts)


def _synthesise(
    findings: str,
    llm_client: Any,
    *,
    task_context: str = "",
    history: str = "",
) -> str:
    if llm_client is None:
        return findings
    try:
        system = _SYNTH_SYSTEM
        if task_context or history:
            extras: list[str] = []
            if task_context:
                extras.append(f"SESSION STATE:\n{task_context}")
            if history:
                extras.append(f"RECENT CONVERSATION:\n{history}")
            system = _SYNTH_SYSTEM + "\n" + "\n\n".join(extras)
        raw = llm_client._generate(
            findings, system=system,
            temperature=0.2, max_tokens=1024,
        )
        text = _strip_thinking(raw).strip()
        return text or findings[:600]
    except Exception:
        return findings[:600]


# ── serialisation helper for chat.py ─────────────────────────────────────────

def result_to_dict(result: AgentRunResult) -> dict:
    """Convert AgentRunResult to a JSON-serialisable dict for the job response."""
    return {
        "goal": result.goal,
        "reply": result.reply,
        "steps_run": [
            {
                "step": sr.step,
                "tool": sr.tool,
                "reason": sr.reason,
                "success": sr.output.success,
                "elapsed_s": sr.elapsed_s,
                "output_text": (sr.output.text or sr.output.error)[:1000],
                "figures": sr.output.figures,
            }
            for sr in result.steps
        ],
        "figures": result.figures,
        "elapsed_s": result.elapsed_s,
        "error": result.error,
    }
