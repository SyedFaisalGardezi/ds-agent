# AGENTS.md — Autonomous Data Science Agent
# Binding rules, contracts, and constraints

> Note: for the current architecture and setup, see README.md and
> AGENT_ARCHITECTURE.md, which are the maintained references. This file records
> the binding rules and contracts and may lag the newest capabilities.

> Read this file at the start of every session.
> Every rule and contract here is binding.
> Do not deviate without explicit user instruction.
> For architecture details see AGENT_ARCHITECTURE.md.
> For ops/quick-start see document.md.

---

## 0. Project Identity

**Project name:** `ds-agent`
**Working dir:** `/Users/ishtiasyed/My_Data/IPAD_sharing/DS_agent/ds-agent`
**Purpose:** Fully autonomous, locally-run AI agent acting as senior data scientist — connects to any data warehouse or database, runs the complete ML pipeline (ETL → deployment), explains every decision with mathematical rigour, exports to Power BI.
**Owner:** Faisal (PhD, AI/ML — seismic signal processing, deep generative models)
**Hardware:** MacBook Pro (MPS GPU available), remote HPC nodes via SSH (avida, raapoi, high01) managed by Slurm/PBS.
**Python version:** 3.11 only.
**Server:** `uvicorn agent.api.server:app --port 9090`

---

## 1. Absolute Rules

1. **No paid LLM APIs.** All LLMs run locally via `ollama`. Never suggest OpenAI, Anthropic API, or any cloud-hosted paid model.
2. **No hardcoded credentials.** Every secret comes from environment variables via `python-dotenv`.
3. **ToolOutput contract is sacred.** Every tool must return `ToolOutput` from `agent/api/agent_tools.py`. Fields: `success: bool`, `text: str`, `error: str`, `figures: list[str]`. No dict shortcuts.
4. **Math explainer is mandatory** for every statistical or ML decision. Populate `math_trace` with LaTeX — `$ $` for inline, `$$ $$` for display.
5. **MLflow logs every training run.**
6. **DVC tracks every dataset and model artifact.** `dvc repro --dry` must pass on every commit.
7. **Tests are not optional.** Every module has unit tests; every pipeline stage has integration tests.
8. **Ruff + mypy on every file.** 80% coverage is the CI hard gate — currently 80.55%.
9. **Never add `Co-Authored-By: Claude` to commit messages.**
10. **Single-model only.** All tasks (scope, plan, code, SQL, math, synthesis, intent) use the same model. No multi-model routing. Overridable per-role via env vars but default is always same model.

---

## 2. Current LLM Stack

All models served by `ollama` on `localhost:11434`.

| Role | Model | Env var |
|---|---|---|
| All tasks (scope, plan, code, SQL, intent, synthesis, QA) | `qwen3.8:27b-mlx` | `DSAGENT_SCOPE_MODEL` |
| Code / SQL override | same (defaults to SCOPE_MODEL) | `DSAGENT_CODE_MODEL` |

Pull: `ollama pull qwen3.8:27b-mlx`

Model config lives in `configs/models.yaml`. `agent/api/llm.py` reads `DSAGENT_SCOPE_MODEL` and `DSAGENT_CODE_MODEL` env vars.

---

## 3. Key Contracts

### ToolOutput (agent/api/agent_tools.py)
```python
@dataclass
class ToolOutput:
    success: bool
    text: str
    error: str = ""
    figures: list[str] = field(default_factory=list)
```
Every tool's `execute(args, *, session, kernel, llm_client) -> ToolOutput`.

### Tool base class
```python
class BaseTool:
    name: str
    description: str
    def execute(self, args: dict, *, session: Session,
                kernel: SessionKernel, llm_client: Any) -> ToolOutput: ...
```

### DataConnector methods
`connect`, `list_schemas`, `describe_table`, `query`, `estimate_cost`, `detect_pii`

### Session (agent/api/sessions.py)
```python
session.data_path        # Path | None  — first uploaded file
session.data_paths       # list[Path]   — all uploaded files
session.target           # str | None
session.brief            # str | None   — concatenated brief text
session.brief_filenames  # list[str]
session.work_dir         # Path | None
session.messages         # list[ChatMessage]
session.artifacts        # dict  — {stage: output}
session.isolation_mode   # bool  — True → cross-session LTM suppressed
session.add_data_path(p) # accumulates data_paths
session.add_brief(t, fn) # concatenates with separator
```

### Memory Layer Contracts
- **Never crash a chat turn on memory failure.** All memory ops must be wrapped in `try/except`. Use the `_MEMORY_OK` guard in `agent_runner.py`.
- **Isolation is binding.** When `session.isolation_mode=True`, cross-session LTM retrieval must be suppressed — never leak user data between sessions.
- **SchemaCache is hash-keyed.** Key = SHA-256(first 64 KB + file size). Do not serve stale schema for a re-uploaded file with the same name but different content.
- **LTM is optional.** Agent must function identically whether or not `nomic-embed-text` is pulled. Never make LTM a hard dependency.
- `tests/unit/test_memory.py` — 48 tests covering all 5 layers ✅

### SessionKernel namespace (agent/api/kernel.py)
```python
kernel.namespace["df"]          # primary dataframe
kernel.namespace["df2"]         # secondary data files
kernel.namespace["TARGET"]      # target column name
kernel.namespace["TASK_SPEC"]   # extracted task specification dict
kernel.namespace["LEAKAGE_COLS"]# columns to drop before training
kernel.namespace["WORK_DIR"]    # working directory path
kernel.namespace["BRIEF"]       # brief text
```

### TASK_SPEC dict keys
`task_type`, `target_column`, `evaluation_metric`, `aggregation_needed`, `aggregation_key`,
`task_description`, `n_samples`, `n_features`, `class_imbalance_ratio`, `recommended_models`

---

## 4. Chat Routing Contract (run_reactive)

Order is binding — do not change without good reason:

1. `_classify_tools()` — keyword routing. If match → execute + synthesise.
2. `classify_intent()` → `"qa"` — use `llm.answer(q, context, messages=_chat_messages())`. Multi-turn via `/api/chat`.
3. `classify_intent()` → `"unknown"` — use `_react_loop()`. Thought/Action/Observation, max 8 steps. Each step calls `_chat()`.

### Multi-turn memory rules
- `_chat_messages(session, n=12)` — last 12 turns, 400 chars user / 1200 chars assistant
- Always pass `messages=_chat_messages(session)` to `answer()` for QA paths
- `_react_loop()` must call `_chat(messages, system)` — never `_generate()` with flat concat

---

## 5. TASK_SPEC Lifecycle Rules

1. Created only by `understand_task` — never set manually in `agent_runner.py`
2. Reset (`kernel.namespace.pop("TASK_SPEC", None)`) on every new brief upload
3. Reset also when user confirms a new/second data file ("this is the second file", etc.)
4. Always include in LLM context via `_data_context()` — model must see what was extracted
5. Never skip `understand_task` when a brief is loaded and TASK_SPEC is absent

---

## 6. Output Guardrails (mandatory)

All LLM output must pass through in this order:
1. `_strip_thinking(text)` — removes `<think>…</think>` blocks
2. `_strip_special_tokens(text)` — removes `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|eot_id|>`
3. `_deduplicate_repetition(text)` — truncates at 3rd repeat of any sentence ≥20 chars

Applied automatically inside `OllamaClient._chat()`, `_generate()`, and `answer()`. Do not bypass.

---

## 7. Training Safety Rules

1. **LEAKAGE_COLS** — `AggregateDataTool` sets `LEAKAGE_COLS` in kernel after aggregation. `TrainModelTool` must drop them before training.
2. **Category-code guard** — `naics_code`, `sic_code`, `nace`, `isic`, `cpc` must never be used as aggregation keys. Enforced in both `agent_runner.py` and `AggregateDataTool`.
3. **class_weight="balanced"** — always set when imbalance detected.
4. **recommended_models** — if present in TASK_SPEC, use them. Fallback: `lgbm`, `rf`, `lr`.

---

## 8. Test & CI Rules

```bash
# Run before every commit
source .venv/bin/activate
python -m pytest tests/ --cov=agent --cov-fail-under=80 -q
ruff check agent/ tests/
```

Current state: **1107 passed, 3 skipped, 80.55% coverage, 0 ruff errors**.

- Every new function needs a test
- Coverage gate is 80% — adding code without tests will fail CI
- Connectors (10 files, all 0%) require docker-based test containers — don't write connector tests without containers
- `agent/memory/` is 0% — not yet wired into agentic loop

---

## 9. Build Status (as of 2026-05-11)

All 10 phases of the original build plan are complete or in progress:

| Phase | Status |
|---|---|
| Scaffold + tools (nl2sql, code_executor, file_io) | ✅ done |
| Connectors (postgres, snowflake, bigquery, mongo, redis, kafka, …) | ✅ built — not wired to tool registry |
| ETL pipeline | ✅ done |
| EDA pipeline | ✅ done |
| Feature engineering | ✅ done |
| Model builder (RF/GBM/LGBM/XGB/ElasticNet/Lasso + HPO) | ✅ done |
| Evaluator (metrics, AB test, calibration, explainer, leaderboard) | ✅ done |
| Deploy packager (Docker, FastAPI, HPC, Optimizer) | ✅ done |
| PowerBI exporter | ✅ done |
| Agentic chat layer (run_reactive, multi-turn memory, ReAct loop) | ✅ done |
| 4-layer instruction understanding (DocParser → Extractor → Enricher → Validator) | ✅ done |
| Memory layer (LongTerm, SchemaCache, SessionMemory) | ⏳ built, not wired |
| Connector → tool registry wiring | ⏳ pending |
| Session persistence (Redis) | ⏳ pending |
| Multi-table tool support (df2/df3 in EDA/MI/FE) | ⏳ pending |
