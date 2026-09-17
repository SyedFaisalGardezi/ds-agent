# ds-agent Architecture & Recent Changes

> Updated: 2026-07-06 (see §13 for the audit-remediation batch)

---

## 1. Architecture Overview

```
User (browser / API)
       │
       ▼
  FastAPI server  (agent/api/chat.py)
       │  POST /chat/sessions/{sid}/message
       │  GET  /chat/sessions/{sid}/jobs/{jid}/stream  ← SSE progress
       │
       ▼
  run_reactive  (agent/api/agent_runner.py)   ← default chat runner
  ┌─────────────────────────────────────────────────────────────┐
  │  Fast path  — _classify_tools()                             │
  │    keyword match found → execute tool plan → synthesise     │
  │                                                             │
  │  QA path  — classify_intent() == "qa"                      │
  │    llm.answer(question, context=...,                        │
  │               messages=_chat_messages(session))             │
  │    → direct memory-aware reply via /api/chat                │
  │                                                             │
  │  ReAct path  — classify_intent() == "unknown"              │
  │    _react_loop() — Thought/Action/Observation loop          │
  │    LLM called via _chat() (multi-turn) each step            │
  └─────────────────────────────────────────────────────────────┘

  run  (agent/api/agent_runner.py)   ← full pipeline build
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase 0 — Pre-plan                                         │
  │    If brief loaded → understand_task runs FIRST             │
  │    Extracts TASK_SPEC: type, target, metric, agg            │
  │    _eda_scope_enrichment() enriches with data statistics    │
  │                                                             │
  │  Phase 1 — Planning                                         │
  │    TASK_SPEC available → _keyword_plan_from_spec()          │
  │    Else → LLM planner (JSON plan)                           │
  │                                                             │
  │  Phase 2 — Execution loop                                   │
  │    For each step:                                           │
  │      ToolRegistry.execute(tool_name, args)                  │
  │      Optional LLM re-plan after each step                   │
  │      Emit SSE progress events                               │
  │                                                             │
  │  Phase 3 — Synthesis                                        │
  │    LLM summarises all findings into final reply             │
  └─────────────────────────────────────────────────────────────┘
       │
       ▼
  Tool Registry  (agent/api/agent_tools.py)  — 12 tools
  ┌──────────────────────────────────────────────────────────────┐
  │  understand_task    → reads brief, extracts TASK_SPEC        │
  │                       scans df2/df3/… for secondary schema   │
  │  aggregate_data     → groups incident rows to entity level   │
  │  eda_profile        → shape, nulls, stats, domain EDA        │
  │  quality_check      → outliers (IQR), constants, skew        │
  │  infer_target       → auto-select target column (3-priority) │
  │  mutual_information → MI scores vs target                    │
  │  feature_engineering→ encode, interactions, RF top-K         │
  │  train_model        → RF/GBM/LR/LGBM/XGB/ElasticNet,        │
  │                       5-fold CV, class_weight=balanced,      │
  │                       PR-AUC for imbalanced                  │
  │  visualize          → distributions, correlation, target     │
  │  execute_code       → arbitrary Python in kernel             │
  │  build_notebook     → assemble full .ipynb report            │
  │  read_instructions  → load PDF/MD/TXT as task brief          │
  └──────────────────────────────────────────────────────────────┘
       │
       ▼
  SessionKernel  (agent/api/kernel.py)
    • Persistent Python namespace per session
    • df, df2, df3, …          (primary + secondary data files)
    • TARGET, TASK_SPEC, WORK_DIR, BRIEF
    • LEAKAGE_COLS             (set by aggregate_data; dropped by train_model)
    • Captures stdout + matplotlib figures as base64 PNG
       │
       ▼
  Orchestrator + Notebook Builder
    (agent/api/orchestrator.py, agent/api/notebook_builder.py)
    • Full pipeline: load → EDA → model → evaluate → drift
    • Assembles .ipynb from all step outputs
```

---

## 2. Key Design Decisions

### Two-phase Read-then-Plan
`understand_task` always runs first when a brief is loaded.
1. Reads brief via 4-layer pipeline (DocParser → TaskExtractor → SpecEnricher → Validator)
2. Extracts TASK_SPEC: task_type, target_column, aggregation_key, evaluation_metric
3. `_eda_scope_enrichment()` adds data-driven fields: n_samples, class_imbalance_ratio, recommended_models
4. Planner uses TASK_SPEC to build correct steps with right args — no guessing

### 5-Layer Memory System
`agent/memory/` is fully wired into the agentic loop:

1. **PersistentSessionStore** (`session_store.py`) — SQLite WAL at `~/.ds-agent/memory/agent_memory.db`; sessions + messages survive server restart. `SessionStore.__init__` restores all sessions on startup.
2. **LongTermMemory** (`long_term.py`) — ChromaDB + `nomic-embed-text` via Ollama; stores/retrieves tool outputs, model results, task specs, and synthesis across sessions. Silently disabled until `ollama pull nomic-embed-text` is run.
3. **SchemaCache** (`schema_cache.py`) — Pure SQLite; keyed by SHA-256(first 64 KB + file size); avoids re-profiling the same file on re-upload.
4. **build_context_messages** (`session_memory.py`) — Smart context window: summarises older turns via `_generate()` → system block; injects top-N LTM snippets → second system block; appends recent turns verbatim.
5. **isolation_mode** (`isolation.py`) — `Session.isolation_mode` flag; regex-detected intent (`"isolate this session"` / `"share context"`); suppresses cross-session LTM retrieval when enabled.

Storage: `~/.ds-agent/memory/agent_memory.db` (SQLite) + `~/.ds-agent/memory/ltm/` (ChromaDB)

Env vars:
- `DSAGENT_MEMORY_DIR` — override `~/.ds-agent/memory`
- `DSAGENT_EMBED_MODEL` — override `nomic-embed-text`
- `DSAGENT_LTM_HITS` — top-N LTM results (default 5)
- `DSAGENT_LTM_DISTANCE` — max cosine distance (default 0.6)
- `DSAGENT_RECENT_TURNS` — verbatim turns kept in context (default 6)
- `DSAGENT_SUMMARY_TOKENS` — token budget for old-turns summary (default 300)

**Graceful degradation**: every memory op is wrapped in `try/except`. Agent functions identically if SQLite / ChromaDB / Ollama-embed is unavailable (`_MEMORY_OK` flag in `agent_runner.py`).

### Multi-turn Memory
Every chat reply carries full conversation context:
- `_chat_messages(session, n=12)` → last 12 turns as `[{role, content}]` sent to `/api/chat`
- User messages capped at 400 chars; assistant at 1200 chars
- QA path: `llm.answer(q, context=..., messages=_chat_messages())` — direct reply with history
- ReAct path: `_react_loop()` calls `llm._chat(messages)` each step — loop has memory too
- `_recent_history(session, n=10)` → 800 chars each, used in synthesis + data context

### Intent Routing (run_reactive else-branch)
When `_classify_tools()` returns empty:
1. `classify_intent()` called — JSON mode, temperature 0.0
2. `qa` → `answer(messages=_chat_messages())` — fast direct reply, no loop
3. `unknown` → `_react_loop()` — Thought/Action/Observation, up to 8 steps

### Multi-file Support
- Primary file → `df` in kernel; `session.data_path`
- Additional uploads → `df2`, `df3`, … via `_load_secondary()` in `chat.py`
- `session.data_paths: list[Path]` — all uploaded files
- `set_workdir()` auto-scans directory: all data files loaded, first brief parsed
- `understand_task` scans kernel for `df2`, `df3` — includes secondary schema in extraction

### TASK_SPEC Lifecycle
- Created by `understand_task` → stored in `kernel.namespace["TASK_SPEC"]`
- Visible to all LLM calls via `_data_context()` (shown as "Task spec (already extracted)")
- Reset (`kernel.namespace.pop("TASK_SPEC", None)`) on every new brief upload
- Reset also on "this is the second file" / "additional file" confirmation messages

### Output Guardrails
- `_deduplicate_repetition()` — truncates at 3rd repeat of any sentence ≥20 chars; kills local-model garbage loops
- `_strip_thinking()` — strips `<think>…</think>` blocks + Qwen special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|eot_id|>`)
- LEAKAGE_COLS guard: `AggregateDataTool` sets `LEAKAGE_COLS`; `TrainModelTool` drops them before training
- Category-code guard: `naics_code`, `sic_code`, `nace` rejected as aggregation keys

### Data-driven Task Inference
When brief is a data dictionary (not task instructions), `understand_task` falls back to:
- Column name patterns for outcome columns (`incident_outcome`, `churn`, `default`)
- Rows-per-group ratio > 1.5 → aggregation needed
- Minority class detection → binary target → PR-AUC
- Zero LLM calls for this path

### Auto-target Fallback
`MutualInfoTool._ensure_target()` static method called by:
- `MutualInfoTool`, `FeatureEngineeringTool`, `TrainModelTool`

Calls `InferTargetTool` automatically if target not set rather than failing.

---

## 3. Session State

`agent/api/sessions.py`:

```
Session
  .session_id          str
  .messages            list[ChatMessage]   # all turns
  .data_path           Path | None         # first uploaded file (backward compat)
  .data_paths          list[Path]          # all uploaded files
  .target              str | None
  .brief               str | None          # concatenated brief text
  .brief_filename      str | None          # first brief file
  .brief_filenames     list[str]           # all brief files
  .work_dir            Path | None
  .run_id              str | None
  .artifacts           dict                # {stage: output}
  .notebook_path       Path | None
  .isolation_mode      bool = False        # True → cross-session LTM suppressed

Session.add_data_path(path)  — accumulates data_paths; data_path = first
Session.add_brief(text, fn)  — concatenates with "--- fn ---" separator
```

---

## 4. LLM Client (agent/api/llm.py)

```python
OllamaClient
  .model       = yanjia/Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled:q4km  (SCOPE_MODEL)
  .code_model  = same  (CODE_MODEL, overridable via DSAGENT_CODE_MODEL)

  .is_available()                                         → bool
  ._generate(prompt, *, system, temperature, max_tokens)  → str   # single-turn /api/generate
  ._chat(messages, *, system, temperature, max_tokens)    → str   # multi-turn /api/chat
  .classify_intent(message)                               → IntentResult
  .generate_code(prompt, *, system, max_tokens)           → str   # uses code_model
  .answer(question, *, context, messages=None)            → ChatResult
      # messages=None → single-turn _generate
      # messages=[…] → multi-turn _chat
```

`_strip_thinking()` and `_strip_special_tokens()` applied to all output before returning.

---

## 5. agent_runner.py Key Helpers

| Helper | Purpose |
|---|---|
| `_data_context(session, kernel)` | Compact state: df/df2/df3 shapes, TASK_SPEC, brief status |
| `_recent_history(session, n=10)` | Last 10 msgs as readable string, 800 chars each |
| `_chat_messages(session, n=12)` | Last 12 turns as `[{role,content}]` for `/api/chat` |
| `_react_loop(...)` | Thought/Action/Observation; calls `_chat()` per step |
| `_parse_react_step(text)` | Extracts Thought/Action/ActionInput/FinalAnswer |
| `_deduplicate_repetition(text)` | Truncates at 3rd repeat of sentence ≥20 chars |
| `_strip_thinking(text)` | Removes `<think>` blocks + Qwen special tokens |
| `_classify_tools(...)` | Keyword routing → `[(tool_name, args)]` |
| `_keyword_plan_from_spec(llm)` | Deterministic plan from TASK_SPEC |
| `_synthesise(findings, llm, ...)` | Summarises tool results to natural language |
| `_eda_scope_enrichment(spec, df, ...)` | Adds n_samples, imbalance ratio, recommended_models |

---

## 6. 4-Layer Instruction Understanding Pipeline

| Layer | File | What it does |
|---|---|---|
| 1 — DocumentParser | `agent/api/doc_parser.py` | Cleans/structures PDF/MD/TXT text |
| 2 — TaskExtractor | `agent/api/task_extractor.py` | 4 decomposed LLM calls + Phase 0 SCOPE ANALYSIS |
| 3 — SpecEnricher | `agent/api/spec_enricher.py` | Data-driven enrichment, zero LLM |
| 4 — TaskSpecValidator | `agent/api/spec_enricher.py` | Validation gate — blocks bad specs |

SCOPE ANALYSIS block fields: `TASK_TYPE`, `TARGET_COLUMN`, `AGGREGATION`, `AGGREGATION_KEY`, `METRIC`, `SECONDARY_FILE`, `SPECIFIC_STEPS`. Each extraction call (1–5) accepts priors from scope — skips LLM call if scope already committed a confident answer.

---

## 7. TrainModelTool — Model Keys

**Classification:** `rf`, `extra_trees`, `gbm`, `lgbm` (optional), `xgb` (optional), `lr`
**Regression:** `rf`, `extra_trees`, `gbm`, `lgbm`, `xgb`, `ridge`, `elasticnet`, `lasso`

`recommended_models` field in TASK_SPEC (from EDA scope enrichment) drives model selection. Falls back to `lgbm/rf/lr` if not set.

---

## 8. Data Flow — OSHA-style Scenario

```
User uploads:
  incidents_2023.csv  →  df  (14,533 rows × 26 cols)
  data_dictionary.pdf →  session.brief  (TASK_SPEC reset)

understand_task:
  • 4-layer pipeline reads brief + df statistics
  • _eda_scope_enrichment adds: n_samples=14533, imbalance detected
  • TASK_SPEC = {
      task_type: "binary_classification",
      aggregation_needed: True,
      aggregation_key: "establishment_id",
      target_definition: "incident_outcome in [1, 2, 3]",
      evaluation_metric: "pr_auc",
      recommended_models: ["lgbm", "xgb", "rf"]
    }

aggregate_data (reads TASK_SPEC):
  • 14,533 incident rows → ~200 establishment rows
  • Sets LEAKAGE_COLS = ["incident_outcome", "incident_date", …]
  • df_orig saved at incident level

train_model:
  • Drops LEAKAGE_COLS before training
  • class_weight="balanced" (imbalance detected)
  • eval_metric = average_precision (PR-AUC)
  • Runs lgbm, xgb, rf per recommended_models
```

---

## 9. Recent Changes (2026-04-24 → 2026-05-11)

| Change | File |
|---|---|
| 5-layer memory system wired (PersistentSessionStore, LongTermMemory, SchemaCache) | `agent/memory/` |
| Session persistence — SQLite WAL; sessions/messages survive server restart | `sessions.py`, `memory/session_store.py` |
| `Session.isolation_mode` + `apply_isolation_intent()` early-return in `run_reactive` | `sessions.py`, `agent_runner.py`, `memory/isolation.py` |
| SchemaCache save on data upload; message persisted on every chat turn | `chat.py` |
| `run_reactive(ltm=None)` param; memory imports with `_MEMORY_OK` flag | `agent_runner.py` |
| LTM + SchemaCache singletons in `create_app`; passed to `make_router` | `server.py`, `chat.py` |
| `memory/__init__.py` — all 5 layers exported | `agent/memory/__init__.py` |
| `test_memory.py` — 48 tests across all 5 memory layers; old sentence_transformers/LangChain tests replaced | `tests/unit/test_memory.py` |
| `run_reactive()` — 3-path router (keyword / QA / ReAct) | `agent_runner.py` |
| `_react_loop()` — Thought/Action/Observation; uses `_chat()` (multi-turn) | `agent_runner.py` |
| Intent-based QA routing: `qa` → `answer(messages=...)` | `agent_runner.py` |
| `_chat_messages()` — `[{role,content}]` array for `/api/chat`, last 12 turns | `agent_runner.py` |
| `_recent_history()` — n=10, 800 chars (was n=4, 300 chars) | `agent_runner.py` |
| `_data_context()` — includes df2/df3 shapes + TASK_SPEC | `agent_runner.py` |
| `_deduplicate_repetition()` + special-token stripping in `_strip_thinking()` | `agent_runner.py` |
| `_chat()` method — Ollama `/api/chat` multi-turn | `llm.py` |
| `answer()` accepts `messages=` param — uses `_chat()` for true multi-turn | `llm.py` |
| Model renamed: `yanjia/Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled:q4km` | `llm.py` |
| `DSAGENT_SCOPE_MODEL` / `DSAGENT_CODE_MODEL` env vars (both default same model) | `llm.py` |
| `_load_secondary()` — loads df2/df3/… from CSV/parquet/xlsx/tsv | `chat.py` |
| `set_workdir()` auto-scans directory for data + brief files | `chat.py` |
| New brief upload clears `TASK_SPEC` in kernel | `chat.py` |
| Multi-file upload: `data_paths[]`, `add_data_path()`, secondary → df2/df3 | `sessions.py`, `chat.py` |
| Multi-file brief: `brief_filenames[]`, `add_brief()` concatenates | `sessions.py`, `chat.py` |
| `UnderstandTaskTool` scans kernel df2/df3 — secondary schema in extraction | `agent_tools.py` |
| LEAKAGE_COLS guard — set post-aggregation, dropped before training | `agent_tools.py` |
| `TrainModelTool` expanded: lgbm, xgb, extra_trees, elasticnet, lasso | `agent_tools.py` |
| `_eda_scope_enrichment()` — pre-plan data profile → recommended_models | `agent_runner.py` |
| 4-layer instruction pipeline (DocParser → TaskExtractor → SpecEnricher → Validator) | `doc_parser.py`, `task_extractor.py`, `spec_enricher.py` |

---

## 10. File Map

| File | Purpose |
|---|---|
| `agent/api/agent_runner.py` | Main agentic loop; `run_reactive`, `run`, `_react_loop`, `_classify_tools`, `_synthesise` |
| `agent/api/agent_tools.py` | 12 tools; understand_task sees df2/df3; train_model expanded model set |
| `agent/api/chat.py` | FastAPI endpoints, SSE streaming; `_load_secondary()`; multi-file upload |
| `agent/api/sessions.py` | Session state; `add_data_path()`, `add_brief()` helpers |
| `agent/api/kernel.py` | Per-session Python kernel; df/df2/df3/TASK_SPEC/LEAKAGE_COLS namespaced |
| `agent/api/llm.py` | OllamaClient; `_chat()` for multi-turn; `answer(messages=)` |
| `agent/api/doc_parser.py` | Layer 1: DocumentParser |
| `agent/api/task_extractor.py` | Layer 2: TaskExtractor (4 decomposed calls + SCOPE ANALYSIS) |
| `agent/api/spec_enricher.py` | Layer 3+4: SpecEnricher + TaskSpecValidator |
| `agent/api/orchestrator.py` | Full pipeline runner for build_notebook |
| `agent/api/notebook_builder.py` | Assembles .ipynb cells from pipeline outputs |
| `agent/api/pdf_reader.py` | Extracts text from uploaded PDFs |
| `agent/api/data_loader.py` | Universal data loader (CSV, parquet, etc.) |
| `agent/api/progress.py` | SSE progress queue per job |
| `agent/api/jobs.py` | Background job runner (ThreadPoolExecutor, max_workers=2) |
| `agent/api/server.py` | FastAPI app factory, port 9090 |
| `agent/api/static/index.html` | Chat UI; multi-file inputs with `multiple` attribute |
| `agent/connectors/` | 10 data connectors (postgres, snowflake, bigquery, mongo, redis, kafka, …) |
| `agent/pipeline/` | eda/, etl/, feature_engineering/, model_builder/, evaluator/, deploy_packager/ |
| `agent/monitoring/` | CostTracker, DriftDetector, FeedbackLoop |
| `agent/memory/session_store.py` | Layer 1: PersistentSessionStore — SQLite WAL session + message persistence |
| `agent/memory/long_term.py` | Layer 2: LongTermMemory — ChromaDB + nomic-embed-text; cross-session semantic recall |
| `agent/memory/schema_cache.py` | Layer 3: SchemaCache — SQLite keyed by file SHA-256; avoids re-profiling |
| `agent/memory/session_memory.py` | Layer 4: build_context_messages — smart context window with LTM injection |
| `agent/memory/isolation.py` | Layer 5: detect/apply isolation intent; sets Session.isolation_mode |
| `agent/explainer/` | MathExplainer (LaTeX traces) |
| `configs/models.yaml` | Single model for all tasks |

---

## 11. Test Health (2026-07-05)

- **Tests:** 1189 passed, 3 skipped ✅
- **Coverage:** 80.39% ✅ (CI gate = 80%)
- **Ruff lint:** 0 errors ✅
- Run: `source .venv/bin/activate && python -m pytest tests/ --cov=agent --cov-fail-under=80 -q`

---

## 12. Recent Changes (2026-07-05)

### Two new specialist agents (15 tools total, was 12)
- **`data_analysis_agent`** (`agent/api/data_analysis_agent.py`, `DataAnalysisAgent`): deep statistical scan (nulls, skew, outliers, high-cardinality, MI, class imbalance) + LLM analyst narrative → writes `kernel["DATA_ANALYSIS_REPORT"]` with `ml_recommendations`. Heuristic fallback (`_heuristic_recommendations`) when LLM offline — size/imbalance-aware.
- **`ml_strategy_agent`** (`MLStrategyPlanner`): reads `DATA_ANALYSIS_REPORT` → writes `kernel["ML_PLAN"]` (models + hyperparams + preprocessing + features_to_exclude + eval_metric). Uses report recommendations directly, else calls LLM.
- **`deep_eda`**: deeper EDA tool.
- **Memory bridge:** `data_analysis_agent` → `DATA_ANALYSIS_REPORT` → `ml_strategy_agent` → `ML_PLAN` → `train_model` reads ML_PLAN and injects hyperparams via sklearn `set_params()`.

### File-role reasoning (understand_task Phase 0.5)
`TaskExtractor._call_file_roles()` maps each loaded df var to a role: `training_data` / `target_derivation_source` / `test_data` / `supplementary` / `unknown`, with confidence. If confidence <0.65 → sets `kernel["FILE_ROLE_CLARIFICATION"]` and asks the user which file is which. User reply re-triggers `understand_task` with `user_hint`. `RawSpec` gained `file_role_map` + `clarification_needed`. Replaces upload-order role assignment.

### Q&A routing (Claude-Code-style behavior)
Small questions no longer trigger tool execution:
- `_question_w` keyword set intercepts question-form queries before `_llm_classify_tools()` → returns `[]`.
- `run_reactive()`: `_react_loop()` fires ONLY for `intent="build"`; every other intent → `answer()` directly.
- `_deep_analysis_w` route → `[data_analysis_agent, ml_strategy_agent, feature_engineering, train_model]`.
- `_QA_SYSTEM` + `_INTENT_PROMPT` rewritten: `qa` is the DEFAULT intent; `build` only for explicit pipeline-run requests.

### Runtime / infra
- Python venv is **3.13.2**. Ruff per-file-ignore added for `agent/api/agent_tools.py` F821 (kernel code-template f-strings flagged as false positives under PEP 701; harmless at runtime, tests prove it).
- **LTM enabled**: `nomic-embed-text` pulled — cross-session semantic memory (Layer 2) now active.

### QA reply cleanup (2026-07-06 fix)
The multi-turn QA path (`answer(messages=_chat_messages())`) previously returned the raw `/api/chat` reply with no post-processing, so the local model's context-bleed leaked through. Fixes in `agent_runner.py`:
- QA reply now passes through `_strip_thinking()` (dedup + `<think>`/special-token strip) — was skipped before.
- New `_trim_echoed_history(reply, session)` — cuts a trailing block that verbatim-echoes a prior assistant turn (matches the 60-char head of any earlier assistant `.text` ≥80 chars). Reads `.text` (ChatMessage field), `.content` fallback.
- `_deduplicate_repetition()` extended: Shape 1 collapses a fragment repeated 3+ times *consecutively* (catches short-token loops like `"PR-AUC."` ×34 that the old ≥20-char sentence rule missed); Shape 2 = original long-sentence 3× rule.
- `_QA_SYSTEM` tightened: answer once, don't restate the question or re-print prior answers under new headings; factual answers <80 words, no headings.

Live-verified: ×34 short-token loop GONE; verbatim cross-turn block echo GONE; memory recall intact.
Residual (soft, local-model limitation): occasional 2× consecutive sentence repeat (under the 3× dedup threshold — lowering to 2 risks truncating real trailing content) and reworded restatements (not verbatim, so echo-trim can't match). Prompt tightening reduced but didn't eliminate these.

---

## 13. Audit Remediation (2026-07-06)

Ranked-audit fixes. Tests: 1146 pass / 3 skip, coverage 81.46%, ruff clean.

### Critical — code-execution safety
- **Kernel exec timeout + thread safety** (`kernel.py`). `SessionKernel.execute` now runs in a watchdog thread with a wall-clock cap (`DSAGENT_EXEC_TIMEOUT_S`, default 120s). On timeout it injects `KeyboardInterrupt` via `PyThreadState_SetAsyncExc` — breaks pure-Python infinite loops (the realistic LLM failure). Blocking C calls still leak the daemon thread but the caller is freed with a `TimeoutError` result. A process-global `_EXEC_LOCK` serialises execute() so the (necessarily global) stdout/stderr swap can't be raced by the 2 concurrent job threads. `_compile_with_repr` rewritten to evaluate the trailing expression exactly once (was double-executing → double side effects).
- **Job runner** (`jobs.py`). Per-job deadline (`DSAGENT_JOB_TIMEOUT_S`, default 900s) surfaced as a `timeout` state to pollers; finished jobs pruned after `DSAGENT_JOB_TTL_S` (default 1800s) to bound dict growth.

### High
- **num_ctx** (`llm.py`). Every `/api/generate` + `/api/chat` call now sends `num_ctx` (`DSAGENT_NUM_CTX`, default 16384). Previously unset → Modelfile's 8192 silently head-truncated long prompts (dropping the system prompt first). This was the likely root of prior context-bleed symptoms.
- **think:false** sent when `DSAGENT_THINK != 1`. NOTE: the Qwen3.6 *distill* ignores this (thinking is trained into the weights, not Ollama's structured channel) — verified accepted (HTTP 200) but still emits `<think>`. The `/no_think` directive + regex `_strip_thinking` remain the effective controls. `num_ctx` is the real win here.
- **Parse-error feedback** (`agent_runner._parse_react_step` / `_react_loop`). Malformed Action-Input JSON now sets `parse_error` and is fed back as an Observation asking the model to resend valid JSON, instead of silently running the tool with `{}` args.
- **LTM + summary buffer wired** (`agent_runner`). QA path uses `build_context_messages` (summary buffer + cross-session LTM injection) via new `_build_qa_messages`; `_store_to_ltm` writes the Q/A + notable tool outputs back after each turn. Both best-effort, isolation-aware. Previously the `ltm` singleton was passed in and never touched.
- **Clarification gate** (`_needs_clarification`). Deterministic: a data-requiring tool planned with no dataset → asks to upload; a heavy build (train/aggregate/FE) with data but no spec/target/goal-hint → asks for the target. Fires before the multi-minute pipeline.
- **Abort on dependency failure** (`_DEPENDENCY_TOOLS`). A failed `understand_task` / `aggregate_data` / `feature_engineering` / `data_analysis_agent` skips remaining plan steps rather than feeding garbage to `train_model`.

### Medium / hygiene
- **llm_router** (`core/llm_router.py`) fallback map now derives from `DSAGENT_SCOPE_MODEL` (Qwen3.6) instead of the deleted `gemma4:latest` / never-installed `qwen2.5-coder:7b`.
- **Schema cache** (`memory/schema_cache.py`) hash now includes mtime → in-place edits past the first 64 KB re-key (was serving stale schema).
- **LLM timeout retry + partial synthesis** (`_react_loop`): one retry on transient failure; on final failure, synthesise the findings gathered so far instead of discarding them.
- **Error-recovery plan** (`_error_recovery_plan`) now conditional — the hardcoded NaN-target fix only fires for NaN/non-finite errors; other error classes get diagnosis only.
- **ReAct tool list** now includes each tool's arg names + types from `input_schema` (`_format_tool_line`) — was `description[:80]` only, so the model guessed arg keys.
- **Dead code removed**: `agentic.py` (500 lines) + `code_agent.py` (273) + their tests — zero live callers; also eliminated 2 of the 4 duplicate `_strip_thinking` defs. Coverage rose to 81.46%.
- **`_StringIO`** dead class removed from `kernel.py`.

### Env vars added
`DSAGENT_EXEC_TIMEOUT_S` (120), `DSAGENT_JOB_TIMEOUT_S` (900), `DSAGENT_JOB_TTL_S` (1800), `DSAGENT_NUM_CTX` (16384), `DSAGENT_THINK` (0).

## 14. Model Swap + Native Tool-Calling Loop (2026-09-05)

### Model
Local model changed to `qwen3.8:27b-mlx` (Qwen3.5 arch, 27.8B dense, 262K ctx, nvfp4/MLX quant, vision + tools + thinking). The old `yanjia/Qwen3.6-35B-A3B-Opus4.7-Reasoning-Distilled:q4km` is deleted. Updated everywhere the name was hardcoded: `configs/models.yaml`, `agent/core/llm_router.py`, `agent/api/llm.py` (all default to the new tag; override with `DSAGENT_SCOPE_MODEL`), plus the offline notice in `chat.py`. Unlike the old distill, the new model supports native function-calling and honours `think:false` (both verified live).

`nomic-embed-text` is NOT installed, so LTM semantic retrieval (`agent/memory/long_term.py`) degrades to no-op until `ollama pull nomic-embed-text`. Schema cache and session memory use local MiniLM and are unaffected.

### Native tool-calling loop (Phase 1 of the multi-agent redesign)
The build path now uses native Ollama function-calling instead of hand-parsing tool-call JSON out of free text.
- **`OllamaClient.chat_with_tools(messages, tools, ...)`** (`llm.py`) — POSTs `/api/chat` with a `tools` payload, returns `{"content", "tool_calls"}`. Returns `None` when the model/Ollama rejects `tools` (HTTP 400), caching the verdict in `_tools_supported` so fallback only probes once. `supports_tools()` exposes the tri-state.
- **`_build_ollama_tool_schemas()`** (`agent_runner.py`) — turns `TOOL_REGISTRY` into Ollama function schemas from each tool's `input_schema`.
- **`_tool_call_loop(...)`** — the real iteration cycle: model picks tool + args → execute on session kernel → feed result back as a `tool` message → repeat until plain-text answer or step budget. Returns `(reply, steps, figures, native_ok)`.
- **`_run_agentic_loop(...)`** — dispatcher: native first, falls back to `_react_loop` when `native_ok` is False. Wired into `run_reactive` (build intent).
- **`_coerce_tool_args()`** — normalises dict-or-JSON-string arguments.
- **Flag:** `DSAGENT_NATIVE_TOOLS=0` forces the text loop. Verified live (model called `eda_profile`, executed, synthesised correct answer). 1153 tests pass, coverage 81.21%.

### Reflexion critic (Phase 2, SHIPPED 2026-09-05)
After every tool result inside `_tool_call_loop`, a critic judges it against the goal and steers the next turn.
- **`_critique_step(goal, tool, args, obs, success, *, llm_client, repeats)`** → `Critique(verdict, reason, goal_met)`. Verdicts: `accept` / `retry` / `replan`. Rule-based fast paths (error or empty output) skip the LLM; genuine successes get one short JSON-constrained `_generate(..., json_mode=True, max_tokens=200)` call. **Fails open** — any exception or unparseable reply → `accept`, so a flaky critic never stalls progress.
- **`Critique`** dataclass + **`_call_signature(tool, args)`** (stable key for repeat detection).
- Loop integration: each `(tool, args)` signature's attempts are counted; `retry` past `_MAX_RETRIES_PER_STEP` escalates to `replan`; `replan` count past `_MAX_LOOP_REPLANS` forces exit; `goal_met` ends the loop and synthesises. All budgets guarantee termination.
- **Flags:** `DSAGENT_CRITIC` (1), `DSAGENT_MAX_RETRIES` (2), `DSAGENT_MAX_REPLANS` (2).
- Verified live: eda_profile → critic `accept` → correct synthesis. 1162 tests pass, coverage 81.40%.

### LTM restored (2026-09-05)
`nomic-embed-text` pulled; `agent/memory/long_term.py` produces 768-dim embeddings again (roundtrip verified). Cross-session semantic recall is live.

### Multi-agent orchestrator + Researcher (Phase 3, SHIPPED 2026-09-05)
An orchestrator delegates sub-goals to five role-specialists via native handoff tool-calls. Registry is now **16 tools** (added `web_research`).
- **`_AGENT_ROLES`** (agent_runner.py): `data_engineer` (understand_task/read_instructions/aggregate_data/quality_check/infer_target), `analyst` (eda_profile/deep_eda/mutual_information/data_analysis_agent/visualize), `ml_engineer` (infer_target/feature_engineering/ml_strategy_agent/train_model), `coder` (execute_code/build_notebook), `researcher` (web_research/execute_code). Each role = focused system prompt + tool subset.
- **`_run_specialist(role, subgoal, ...)`** — runs `_tool_call_loop` restricted to the role's `tool_names` with the role `system_prompt` (critic + budgets apply unchanged). `_tool_call_loop` and `_build_ollama_tool_schemas` gained `tool_names`/`system_prompt` params.
- **`_build_handoff_schemas()`** — one Ollama function per role (arg: `subgoal`); the orchestrator's only tools.
- **`_run_multiagent_loop(...)`** — supervisor: orchestrator calls a role → `_run_specialist` runs → report fed back as a `tool` message; repeats until final text or `_MAX_ORCH_ROUNDS`. Skips duplicate `(role, subgoal)` delegations. Synthesises from accumulated steps if the round budget is hit. Returns `native_ok=False` (no work done) to signal fallback.
- **`_run_agentic_loop`** now dispatches: multi-agent (if `_MULTIAGENT`) → single `_tool_call_loop` → text `_react_loop`.
- **Flags:** `DSAGENT_MULTIAGENT` (1), `DSAGENT_MAX_ORCH_ROUNDS` (6), `DSAGENT_SPECIALIST_STEPS` (6).

### web_research tool (live web, no API key, no new deps)
`WebResearchTool` (agent_tools.py) with module helpers `_ddg_search` (DuckDuckGo HTML endpoint via httpx, parsed with lxml XPath — no cssselect/bs4), `_fetch_page_text` (fetch + strip script/style/nav, extract text), `_pypi_info` (official PyPI JSON for current version/summary). Args: `query` (required), `max_results` (5), `fetch_pages` (true). Returns search results + PyPI enrichment (from any pypi.org/project/<slug> hit, or a 1-2 word query) + extracted text from the top 2 pages, with source URLs in artifacts. All failures degrade gracefully. Also reachable deterministically: `_classify_tools` routes research/currency questions (research/search-web verbs, or latest/newest/current + a library/version/method noun, guarded against build verbs) straight to `web_research`.

### Memory-aware orchestration + connectors (Phase 4, SHIPPED 2026-09-05)
Registry is now **17 tools** (added `connect_data`). The build path recalls prior work, and specialists can pull external data into the blackboard.
- **LTM recall:** `_recall_from_ltm(ltm, instruction, session)` (agent_runner.py) queries `LongTermMemory.retrieve_as_context_string` (isolation-aware: passes `session_id` + `isolation_mode`), returns "" when LTM is off / nothing relevant / on any error. `_run_agentic_loop` gained an `ltm` param, computes the recall once, and injects it as a "PRIOR EXPERIENCE" block into the orchestrator (`_ORCH_SYSTEM` `{prior_experience}` slot via `_run_multiagent_loop(prior=...)`) or the single-agent loop (`_tool_call_loop(prior_experience=...)`). `run_reactive` passes `ltm=ltm`. Verified live: a fresh cross-session query surfaced a stored churn solution (GBM ROC-AUC 0.91) via nomic-embed. (The QA path already injected LTM via `_build_qa_messages`; this closes the build path.)
- **`connect_data` tool** (`ConnectDataTool`, agent_tools.py): connects to an external source and loads a query/table result into `kernel.namespace["df"]` (the blackboard). Sources via `_CONNECTOR_CLASSES` map: sqlite, flatfile/csv, postgres(ql), mysql, bigquery, snowflake, mongodb, redis, kafka, azure_sql, rest — lazy-imported, instantiated with the user's `config` kwargs. Discovery mode (no query/table) lists tables; `table` on a SQL source → `SELECT *` (identifier-guarded); results → DataFrame via `_connector_records_to_df`. Read-only (connectors' own SQL guard blocks DDL/DML). Connection always closed in `finally`. Added to the `data_engineer` role. Verified live on sqlite (discovery, table load into kernel, custom query, destructive-SQL + unsafe-identifier rejection).
- All four phases of the multi-agent redesign are now shipped. 1187 tests pass, coverage 81.02%.

### Web-UI multi-agent toggle (2026-09-05)
Multi-agent stays the default (`DSAGENT_MULTIAGENT=1`), but each message can override it. `MessageBody.multiagent: bool | None` (chat.py) → `_spawn_agent_job(..., multiagent)` → `run_reactive(multiagent=)` → `_run_agentic_loop(multiagent=)`, which resolves `use_multiagent = _MULTIAGENT if multiagent is None else bool(multiagent)`. The static UI (`agent/api/static/index.html`) has a "Multi-agent" switch by the Send button: state persists in `localStorage` (`ds_multiagent`), each `send()` posts `multiagent: maEnabled()`, and flipping it prints a system line. OFF routes to the single-agent `_tool_call_loop` (faster); ON uses the orchestrator. Verified live in-browser (toggle flips, POST /message returns 200). Unit-tested (`test_run_agentic_loop_multiagent_override`). 1188 tests pass.

## 15. Workspace Awareness (Claude-Code-style, 2026-09-06)
The agent can now be pointed at a folder and understand the whole tree, reading files on demand instead of ingesting everything. Registry is **19 tools** (added `explore_directory`, `read_file`).

**Technique:** a compact recursive map + on-demand reads (what Claude Code / aider / Cursor do), not full-content ingestion — this scales to large folders and stays inside the context window.

- **`scan_workspace(root, max_files=600, max_depth=6, profile_data=True)`** (agent_tools.py): ignore-aware walk (`_WS_IGNORE_DIRS`: .git/.venv/node_modules/__pycache__/… + dotfiles), classifies each file (`_WS_CATEGORY`: data/doc/code/notebook/config/image/other), builds a bounded `tree` string, captures column headers for up to 8 tabular files (`_ws_data_columns`, cheap header read), and returns a `summary` block for prompt injection. Content is NOT read here — cheap enough to run on every workdir change.
- **`explore_directory` tool**: runs/refreshes the scan (optional `subpath`, confined), stores `WORKSPACE_MAP` in the kernel, returns the map. First thing an agent runs to learn the folder.
- **`read_file` tool**: reads one file, confined to the working directory (`_ws_resolve_within` blocks path traversal + absolute-outside), size-bounded (`max_chars`, default 8000), type-aware — tabular → shape+columns+sample, notebook → markdown/code-cell headers, PDF → extracted text, image → note, else text. Added to the `data_engineer` role.
- **Context injection:** `_data_context` prepends a compact `WORKSPACE_MAP` block (root, counts, first 25 tree lines, data schemas) so every agent turn is workspace-aware without being asked.
- **`/workdir` endpoint** (chat.py) now runs `scan_workspace` recursively (was top-level data+brief only), stores the map, and reports file/folder counts. Top-level data auto-load into `df`/`df2` is unchanged; deeper files are known via the map and loaded on demand (`read_file` / `connect_data` / register).
- Verified live: agent called `explore_directory` → `read_file(README)` → answered the project goal and target column correctly. Path-traversal, absolute-outside, missing-file, and no-workdir cases all guarded and tested. 1196 tests pass, coverage 80.91%.

## 16. File Search + Editing, with a Suggest/Write Toggle (2026-09-06)
The agent can now search, write, and edit files in the working directory — with a per-message toggle between applying writes and only suggesting them (like Claude Code's edit modes). Registry is **22 tools** (added `search_files`, `write_file`, `edit_file`).

**Edit format:** exact string search/replace (`edit_file` = `old_string` → `new_string`, unique-match-or-fail), the most reliable format for local models — verifiable, minimal tokens, no whole-file rewrite. `write_file` creates/overwrites. This mirrors Claude Code's `str_replace`/create.

- **`search_files`** (agent_tools.py): pure-Python grep (ripgrep exists on the host but the agent runs as a subprocess where the `rg` shell shim doesn't apply, so Python is the robust choice). Ignore-aware walk (`_ws_iter_files`), skips binaries (`_is_probably_text`), regex or literal, optional `glob`, capped results, `file:line: match`.
- **`edit_file`**: exact `old_string`→`new_string`. 0 matches → error; >1 without `replace_all` → error (ask for more context); else apply. Confined to workdir (`_ws_resolve_within`).
- **`write_file`**: create or overwrite (needs `overwrite=true` to replace). Confined to workdir. No delete tool (destructive ops are out of scope).
- **Suggest/Write mode:** `_ws_write_mode(kernel)` returns `write` or `suggest`. In **suggest** mode `edit_file`/`write_file` return a unified diff and touch nothing; in **write** mode they apply. Default is **suggest** (safe). Set per turn via `FILE_WRITE_MODE` on the kernel, which `run_reactive(allow_writes=...)` sets from `MessageBody.allow_writes`; env default `DSAGENT_ALLOW_WRITES` (0). `search_files`/`read_file` always work regardless.
- Roles: `coder` gets `search_files`/`read_file`/`write_file`/`edit_file`/`execute_code`/`build_notebook`; `data_engineer` also gets `search_files`.
- **Web-UI toggle:** a red **Writes** switch beside the Multi-agent switch (`agent/api/static/index.html`), OFF by default, persisted in `localStorage` (`ds_allow_writes`), posts `allow_writes` per message; flipping it prints a system line. Verified in-browser (switch flips, ENABLED message shown).
- Verified: unit tests cover both modes, non-unique/missing guards, overwrite guard, and path confinement (traversal + absolute-outside rejected for write and edit). 1206 tests pass, coverage 80.91%.

## 17. SQL/Warehouse Queries + Model Deployment + CI/CD (2026-09-06)
The agent can now author and run SQL against warehouses (including Snowflake), package a trained model as a deployable service, and generate a CI/CD workflow. Registry is **25 tools** (`sql_query`, `deploy_model`, `scaffold_cicd`). These wire existing modules (`agent/tools/nl2sql.py` logic, `agent/pipeline/deploy_packager/`) into the agentic chat, plus a new CI/CD generator.

**Boundary (safety):** the agent GENERATES deployment and CI/CD artifacts and serialises the model. It does not run `docker push` or a live cloud deploy — those are irreversible, outward, and credential-bound, so they run from the generated CI/CD with the user's own secrets. Snowflake and every SQL source stay read-only (destructive statements refused at the connector guard and again in `sql_query`).

- **`sql_query`** (agent_tools.py): SQL sources only (snowflake, postgres, bigquery, mysql, sqlite, azure_sql). Give a natural-language `question` → it introspects the live schema (`_connector_schema_context`: `list_schemas` + `describe_table`), asks `llm_client.generate_code` for the SQL, refuses destructive statements (`_SQL_FORBIDDEN`), runs it read-only, returns SQL + rows. Or give explicit `sql` to run. Optional `save_to` writes a .sql file (respects the write toggle + workdir confinement). Added to `data_engineer`.
- **`deploy_model`** (wraps `deploy_packager.fastapi_builder.FastAPIBuilder`): finds `best_model` in the kernel, `joblib.dump`s it, and scaffolds `app.py` + `Dockerfile` + `requirements.txt` + `model.joblib` under `deploy/` in the working directory. Errors if no model or no workdir. Respects the write toggle (suggest → lists files + run hints, writes nothing). Prints local-run and `docker build` hints but executes neither. Added to `ml_engineer`.
- **`scaffold_cicd`**: renders a GitHub Actions (`.github/workflows/ci.yml`) or GitLab CI (`.gitlab-ci.yml`) workflow that lints, tests, builds the Docker image from `deploy/`, and has a gated/manual deploy step (push and deploy lines are commented or `when: manual`). Respects the write toggle + confinement. Added to `coder`.
- Verified: NL→SQL generated and ran against SQLite (grouped aggregate, 2 rows) and saved the .sql; destructive SQL and non-SQL sources refused; `deploy_model` serialised a real sklearn model and wrote all four files, and proposed-without-writing in suggest mode; `scaffold_cicd` wrote both providers and refused an unknown one. 1212 tests pass, coverage 80.94%.

## 18. Output-Aware, Task-Directed Exploration (2026-09-17)
The agentic loop now adapts to what each step reveals and steers exploration toward the specific task, rather than running a fixed sequence of generic checks. This extends the reflexion critic (§14) with two behaviours the user asked for: read the output and decide the next step, and explore what serves the expected outcome.

- **Task focus.** `_task_focus(session, kernel)` states, in one line, what the exploration should serve: the target, the task type, the metric, and the stated goal, taken from `TASK_SPEC` and `session.target`. It returns an empty string when no task is known, so generic exploration is unchanged. `_data_context` now prints a `TASK FOCUS` line, and both `_TOOLCALL_SYSTEM` and `_SPECIALIST_SYSTEM` instruct the agent to prioritise the features and relationships most relevant to that outcome after the general checks. The Analyst role purpose was rewritten to match.
- **Reflection, not just a gate.** `Critique` gained `finding` and `next_step`, and `_CRITIC_PROMPT` now asks the reviewer to read what the result reveals and recommend the next step, grounded in the actual numbers and the task focus. `_critique_step` takes a `task_focus` argument and parses the two new fields. The rule-based fast paths (error, empty) and the fail-open default still apply.
- **Feed the reflection back and remember it.** In `_tool_call_loop`, after each accepted step the finding is appended to a running `EXPLORATION_LOG` on the kernel, and the reflection (finding plus recommended next step) is fed back as a guidance turn so the model re-plans from the output instead of following a fixed order. `_data_context` surfaces the last several log entries under `WHAT WE HAVE LEARNED SO FAR`, so later steps in the same run, later turns, and the final synthesis all build on what earlier steps found. This holds for both the single-agent loop and every specialist in the multi-agent orchestrator, because both build their prompt from `_data_context`.

The deterministic pipeline path and `_eda_scope_enrichment` (one-shot data-aware model and metric selection) are unchanged; this change makes the reasoning loop adaptive without touching the tested pipeline. Verified: `_task_focus` renders from a spec and is empty without one; the critic returns finding and next_step and receives the task focus in its prompt; `_data_context` shows both blocks; the loop logs each finding and feeds the reflection back. 1217 tests pass, coverage 81.01%.

## 19. Instruction Handling Fixes (2026-09-17)
Four limits in how the agent carried brief and document instructions into preprocessing and EDA were fixed, after a run missed a task named in the brief.

- **Requirements now stay in context (was: brief truncated to 2000 chars).** `_data_context` surfaces the extracted `specific_requirements` as a short "REQUIREMENTS TO SATISFY" checklist, so the concrete instructions persist even when the raw brief is long, and it skips placeholder templates like `<requirement 3>`. The raw brief excerpt was widened from 2000 to 4000 characters.
- **The agentic loop now follows the checklist (was: requirement-to-code injection only on the deterministic path).** `_TOOLCALL_SYSTEM` and `_SPECIALIST_SYSTEM` tell the agent to treat the listed requirements as a checklist and address every one, using `execute_code` for anything the standard tools do not cover, before writing the final answer.
- **EDA now honours instructions through orchestration (was: fixed EDA tools ignore free text).** The EDA tools are unchanged, which keeps the tested pipeline intact. Instruction following comes from the requirements checklist plus the task focus steering which features to emphasise, so the agent runs `execute_code` for instruction-specific analysis the fixed tools do not cover.
- **Sub-folder documents are auto-loaded (was: only top-level brief files).** After the workspace scan, `set_workdir` in `chat.py` loads a bounded number (up to six) of brief-type documents (pdf, md, txt, rst) found in sub-folders that the top-level pass did not pick up, using the ignore-aware map so junk directories are skipped.

Verified: the requirements checklist renders and skips placeholders, the widened excerpt applies, and the full suite passes. 1217 tests pass, coverage 80.87%. A server restart is required for these to take effect.

### Optional next steps (beyond the plan)
Add a warehouse write path behind an explicit opt-in (currently read-only by design); a model-registry/versioning step in `deploy_model`; ONNX export; parallelise independent specialist calls; a pending-edits "apply all" action; a hard completion gate that refuses to finish while a listed requirement is unaddressed.
