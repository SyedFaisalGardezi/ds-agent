# ds-agent — Architecture & Settings

> Note: for the current architecture and setup, see README.md and
> AGENT_ARCHITECTURE.md, which are the maintained references. This file is an
> earlier settings write-up and may lag the newest capabilities.

> Updated: 2026-05-11

A locally-run, chat-driven Data Science agent. Upload CSV/parquet files and optionally PDF/MD brief files, then ask the agent to "build a notebook" — it runs profiling → quality → MI → feature engineering → task detection → training → drift, logs the run to a leaderboard, and emits a `.ipynb`. Unrecognised messages route to a local LLM (Ollama) for free-form Q&A grounded in full session + conversation history.

## 1. Runtime settings

| Setting | Value | Source |
|---|---|---|
| API bind | `0.0.0.0:9090` | `agent/api/server.py` (`API_PORT = 9090`) |
| Entry point | `python -m agent.api.server` | `agent/api/server.py:main` |
| Scope / planning model | `qwen3.8:27b-mlx` | env `DSAGENT_SCOPE_MODEL` |
| Code / SQL model | same as scope model | env `DSAGENT_CODE_MODEL` (defaults to SCOPE_MODEL) |
| Legacy env var | `DSAGENT_LLM_MODEL` | still honoured; falls back to SCOPE_MODEL |
| Ollama host | `http://127.0.0.1:11434` | env `OLLAMA_HOST` |
| LLM timeout | 180 s (generate/chat), 2 s (probe) | `OllamaClient.__init__` |
| Probe TTL | 30 s positive cache; negative always re-probes | `is_available()` |
| Background workers | `ThreadPoolExecutor(max_workers=2)` | `JobRunner` |
| Pipeline sample cap | 20,000 rows (random_state=0) | `orchestrator.run_full_pipeline` |
| Notebook output dir | `outputs/notebooks/` | `orchestrator._NOTEBOOK_DIR` |
| Upload dir | `outputs/uploads/{session_id}/` | `chat._UPLOAD_DIR` |
| Leaderboard DB | SQLite, `check_same_thread=False` | `Leaderboard` |
| Memory DB | SQLite WAL at `~/.ds-agent/memory/agent_memory.db` | `PersistentSessionStore` |
| LTM store | ChromaDB at `~/.ds-agent/memory/ltm/` | `LongTermMemory` |
| Embedding model | `nomic-embed-text` via Ollama (optional) | env `DSAGENT_EMBED_MODEL` |
| Memory dir | `~/.ds-agent/memory/` | env `DSAGENT_MEMORY_DIR` |
| Python | ≥ 3.11 | `pyproject.toml` |

### Required external services

- **Ollama** running on `127.0.0.1:11434` with the model pulled. Verify: `ollama list` and `curl -s http://127.0.0.1:11434/api/tags`.
- Pull: `ollama pull qwen3.8:27b-mlx`
- If model is missing: keyword commands still work; free-form chat returns "LLM is offline" hint.
- **LTM (optional):** `ollama pull nomic-embed-text` — enables cross-session semantic memory (layer 2). Without it, layers 1, 3, 4, 5 still work; only semantic recall is silently disabled.

### Key dependencies (excerpt)

`fastapi`, `uvicorn`, `python-multipart`, `pypdf`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `httpx`, `pydantic`, `mlflow`, `dvc`. Full list in [pyproject.toml](pyproject.toml).

## 2. Component architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser UI (agent/api/static/index.html)                        │
│  • drag-drop multi-file CSV/parquet/xlsx/tsv                     │
│  • drag-drop multi-file PDF/MD/TXT brief                         │
│  • chat box • SSE progress panel • downloads .ipynb              │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (JSON + multipart)
┌────────────────────────────▼─────────────────────────────────────┐
│  FastAPI app  (agent/api/server.py :: create_app)                │
│  ├─ /                    → static index.html                     │
│  ├─ /health              → liveness + LLM availability           │
│  ├─ /stages, /tools      → registry introspection                │
│  ├─ /pipelines/*         → direct stage invocation               │
│  ├─ /tools/{name}        → dispatch into _TOOLS registry         │
│  ├─ /leaderboard         → top-N runs                            │
│  └─ /chat/...            → mounted chat router                   │
└──────┬──────────────────────────────────────────────┬────────────┘
       │                                              │
┌──────▼──────────────────────┐         ┌─────────────▼─────────────┐
│ Chat router (api/chat.py)   │         │ Pipeline stages           │
│ POST /chat/sessions/{sid}/  │         │ • DataProfiler            │
│  ├─ upload  multi-file      │         │ • QualityScorer           │
│  │   df loaded as df        │         │ • StatisticsAnalyser (MI) │
│  │   df2/df3… via           │         │ • FeatureGenerator        │
│  │   _load_secondary()      │         │ • TaskDetector            │
│  ├─ brief   multi-file PDF  │         │ • ModelTrainer            │
│  │   clears TASK_SPEC       │         │ • DriftDetector           │
│  ├─ workdir auto-scans dir  │         │ • Leaderboard (SQLite)    │
│  ├─ message → run_reactive  │         └───────────────────────────┘
│  └─ jobs/{jid}/stream (SSE) │
└──────┬──────────────────────┘
       │ spawns
       ▼
  JobRunner (ThreadPool×2)
       │
       ▼
  run_reactive (agent/api/agent_runner.py)   ← chat turns
  ┌───────────────────────────────────────────────┐
  │  1. _classify_tools() — keyword routing       │
  │     match → execute tool plan → synthesise    │
  │                                               │
  │  2. classify_intent() == "qa"                 │
  │     → answer(question, context,               │
  │              messages=_chat_messages())       │
  │       direct reply via /api/chat multi-turn   │
  │                                               │
  │  3. classify_intent() == "unknown"            │
  │     → _react_loop()                           │
  │       Thought/Action/Observation              │
  │       each step calls _chat() — multi-turn    │
  └───────────────────────────────────────────────┘
       │
       ▼
  OllamaClient (api/llm.py)
  ├─ _generate()     single-turn  /api/generate
  ├─ _chat()         multi-turn   /api/chat    ← used by answer() + _react_loop
  ├─ classify_intent()  JSON mode, temp 0.0
  ├─ generate_code()    code model override
  └─ answer(q, context, messages=[…])
       strips <think> + Qwen special tokens
       deduplicate_repetition on all output
       │
       ▼
  SessionKernel  (api/kernel.py)
    df, df2, df3, …   ← all uploaded data files
    TARGET, TASK_SPEC, WORK_DIR, BRIEF, LEAKAGE_COLS
    Captures stdout + matplotlib figures as base64 PNG
       │
       ▼
  SessionStore  (in-memory, locked)
    Session: data_path, data_paths[], target, brief,
             brief_filenames[], work_dir, artifacts,
             notebook_path, messages[]
```

## 3. File map (API layer)

| File | Role |
|---|---|
| [agent/api/server.py](agent/api/server.py) | FastAPI app factory; shared `Leaderboard`, `SessionStore`, `JobRunner`, `OllamaClient`. Serves UI at `/`. |
| [agent/api/chat.py](agent/api/chat.py) | `/chat/...` router. Multi-file upload; `_load_secondary()`; workdir auto-scan; spawns jobs. |
| [agent/api/agent_runner.py](agent/api/agent_runner.py) | `run_reactive()` + `run()`. All routing, ReAct loop, synthesis, helper functions. |
| [agent/api/sessions.py](agent/api/sessions.py) | `Session` dataclass + thread-safe `SessionStore`. `add_data_path()`, `add_brief()`. |
| [agent/api/jobs.py](agent/api/jobs.py) | `JobRunner` (ThreadPoolExecutor) + `Job` state machine. |
| [agent/api/llm.py](agent/api/llm.py) | Direct Ollama HTTP client. `_chat()` for multi-turn; `answer(messages=)`; `<think>` + special-token stripping. |
| [agent/api/doc_parser.py](agent/api/doc_parser.py) | Layer 1: Document cleaner/structurer. |
| [agent/api/task_extractor.py](agent/api/task_extractor.py) | Layer 2: 4 decomposed LLM calls + Phase 0 SCOPE ANALYSIS block. |
| [agent/api/spec_enricher.py](agent/api/spec_enricher.py) | Layer 3+4: Data-driven SpecEnricher + TaskSpecValidator gate. |
| [agent/api/orchestrator.py](agent/api/orchestrator.py) | Runs full pipeline stages; feeds `notebook_builder`. |
| [agent/api/notebook_builder.py](agent/api/notebook_builder.py) | Emits `.ipynb` (v4 JSON) directly — no `nbformat` dep. |
| [agent/api/pdf_reader.py](agent/api/pdf_reader.py) | `pypdf` extract + regex hints. |
| [agent/api/static/index.html](agent/api/static/index.html) | Single-page chat UI; multi-file inputs; SSE progress panel. |

## 4. Endpoints

### Top-level
- `GET  /`              — UI
- `GET  /health`        — `{status, service, version, llm:{model, available}}`
- `GET  /stages`        — pipeline stage names
- `GET  /tools`         — registered tool names
- `POST /pipelines/{stage}` — direct stage invocation
- `GET  /leaderboard?metric=&n=&higher_is_better=`
- `POST /tools/{name}`  — dispatch to `Tool.run(input_json)`

### Chat
- `POST   /chat/sessions` — create session (returns `session_id`)
- `GET    /chat/sessions` — list ids
- `GET    /chat/sessions/{sid}` — state + last 50 messages
- `DELETE /chat/sessions/{sid}` — drop session
- `POST   /chat/sessions/{sid}/data` — register server-side path `{path, target?}`
- `POST   /chat/sessions/{sid}/upload` — multipart; `files: list[UploadFile]`; primary → df, extras → df2/df3/…
- `POST   /chat/sessions/{sid}/brief` — multipart; `files: list[UploadFile]`; .pdf/.md/.txt/.rst; clears TASK_SPEC
- `POST   /chat/sessions/{sid}/workdir` — set work dir; auto-scans for data + brief files
- `POST   /chat/sessions/{sid}/message` — chat turn `{text}`; returns `{job_id}` or inline reply
- `GET    /chat/sessions/{sid}/jobs/{jid}/stream` — SSE real-time progress
- `GET    /chat/sessions/{sid}/notebook` — download last-built `.ipynb`

## 5. Chat intent routing (run_reactive)

1. **Keyword fast-path** — `_classify_tools()`: deterministic patterns for build, EDA, train, understand_task, aggregate, etc. Zero LLM calls. If match: execute tool plan → synthesise.

2. **Intent classification** — `classify_intent()` (Ollama JSON mode, temp 0.0). Labels: `build | set_target | status | help | qa | unknown`.

3. **QA path** — intent `qa` → `llm.answer(question, context=_data_context(), messages=_chat_messages())`. Full multi-turn memory via `/api/chat`. Returns direct conversational reply.

4. **ReAct path** — intent `unknown` → `_react_loop()`. Up to 8 Thought/Action/Observation steps. Each step calls `_chat()` with full message history. Parses `Final Answer:` to return.

5. **Hard fallback** — LLM offline → hint to use `help` / `target is` / `build notebook`.

## 6. Multi-turn memory

All chat paths carry conversation history:

| Path | Memory mechanism |
|---|---|
| QA | `_chat_messages(session, n=12)` → `[{role,content}]` → `/api/chat` |
| ReAct loop | `messages: list[dict]` built per step → `_chat()` per step |
| Synthesis | `_recent_history(session, n=10)` → 800 chars each → context string |
| `_data_context()` | TASK_SPEC + df/df2/df3 shapes always included |

User message max 400 chars; assistant max 1200 chars in `_chat_messages`.

## 7. Multi-file support

```
_DATA_EXTS  = {".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".feather"}
_BRIEF_EXTS = {".pdf", ".md", ".txt", ".rst"}
```

Upload flow:
1. First data file → `df` in kernel + `session.data_path`
2. Additional data files → `df2`, `df3`, … via `_load_secondary(kernel, path, var_name)`
3. All paths stored in `session.data_paths: list[Path]`
4. Brief files concatenated via `session.add_brief(text, filename)` with separator
5. `session.brief_filenames: list[str]` tracks all brief files

`set_workdir()` auto-scans: all matching data files loaded, first brief file parsed. `understand_task` scans kernel for `df2`, `df3` — includes their column schemas in extraction prompt.

## 8. Background-job lifecycle

`pending → running → done | failed`. Two kinds:
- `build_notebook` — runs full pipeline, writes notebook, mutates `session.artifacts` and `session.notebook_path`.
- `run_reactive` — chat turn; handles all intent routing and LLM calls.

Client receives SSE stream at `GET /chat/sessions/{sid}/jobs/{jid}/stream`. Each event: `{type, message, tool, level, ts}`. Final event: `{type: "done", reply, elapsed_s}`. Sessions and jobs are in-memory — lost on server restart.

## 9. Pipeline stages (orchestrator.run_full_pipeline)

| # | Stage | Class | artifact key |
|---|---|---|---|
| 1 | Profile | `DataProfiler` | `profile` |
| 2 | Quality | `QualityScorer` | `quality` |
| 3 | Mutual information | `StatisticsAnalyser.mutual_information` | `mi` |
| 4 | Feature generation | `FeatureGenerator` | `features` |
| 5 | Task detection | `TaskDetector.infer` | `task` |
| 6 | Train | `ModelTrainer` (RF/LGBM/XGB/etc, test=0.25) | `train` → leaderboard |
| 7 | Drift | `DriftDetector` (first vs second half) | `drift` |
| 8 | Notebook | `notebook_builder.build` | `outputs/notebooks/{run_id}.ipynb` |

All numeric outputs pass through `_sanitize` (numpy scalars → Python, NaN/Inf → None) before JSON serialisation.

## 10. Quick-start

```bash
cd /Users/ishtiasyed/My_Data/IPAD_sharing/DS_agent/ds-agent

# 1. confirm Ollama + model
ollama list | grep Qwen3.6
curl -s http://127.0.0.1:11434/api/tags | python -m json.tool | head

# 2. start the API
./.venv/bin/python -m agent.api.server          # binds 0.0.0.0:9090

# 3. UI
open http://127.0.0.1:9090/

# 4. scripted example
SID=$(curl -s -X POST http://127.0.0.1:9090/chat/sessions | jq -r .session_id)

# upload primary data + optional secondary
curl -s -F "files=@data.csv" -F "files=@lookup.csv" \
     http://127.0.0.1:9090/chat/sessions/$SID/upload

# upload brief (PDF or MD)
curl -s -F "files=@brief.pdf" \
     http://127.0.0.1:9090/chat/sessions/$SID/brief

# set target
curl -s -X POST http://127.0.0.1:9090/chat/sessions/$SID/message \
     -H 'content-type: application/json' -d '{"text":"target is price"}'

# build notebook (SSE stream)
JID=$(curl -s -X POST http://127.0.0.1:9090/chat/sessions/$SID/message \
     -H 'content-type: application/json' -d '{"text":"build notebook"}' | jq -r .job_id)

# stream progress
curl -N http://127.0.0.1:9090/chat/sessions/$SID/jobs/$JID/stream

# download notebook
curl -OJ http://127.0.0.1:9090/chat/sessions/$SID/notebook
```

## 11. Operational notes

- **Cold-load latency.** Loading `qwen3.8:27b-mlx` commonly takes 10–60 s on first request. The 180 s timeout accommodates this.
- **Probe caching.** `is_available()` caches positive probes for 30 s. Negative results never cached — late-starting Ollama picked up immediately on next call.
- **Multi-turn memory.** QA and ReAct paths use Ollama `/api/chat` for true conversation memory. Each turn carries last 12 messages (400 chars user / 1200 chars assistant).
- **Repetition guard.** `_deduplicate_repetition()` truncates output at 3rd repeat of any sentence ≥20 chars — prevents garbage loops from local model.
- **TASK_SPEC reset.** Every new brief upload resets TASK_SPEC so `understand_task` re-extracts from scratch. "This is the second file" messages also trigger a reset + re-extraction.
- **Single-process state.** `SessionStore` and `JobRunner` are in-memory. For multi-worker deployments swap for Redis + RQ/Celery.
- **Notebook dependencies bundled in cells** — the `.ipynb` runs anywhere Jupyter + pandas + sklearn + matplotlib are installed.
- **Secondary data in tools.** `understand_task` sees df2/df3 schemas. Other tools (`eda_profile`, `mutual_information`, `feature_engineering`, `train_model`) still operate on `df` only — multi-table join/merge not yet implemented.

## 12. Configuration knobs

| Env var | Default | Effect |
|---|---|---|
| `DSAGENT_SCOPE_MODEL` | `qwen3.8:27b-mlx` | Planning / intent / synthesis model |
| `DSAGENT_CODE_MODEL` | same as SCOPE_MODEL | Code / SQL / numeric generation model |
| `DSAGENT_LLM_MODEL` | same as SCOPE_MODEL | Legacy override; still honoured |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama HTTP base URL |

To change the bind port, edit `API_PORT` in `agent/api/server.py` or run `uvicorn agent.api.server:app --port <N>`.

## 13. Test health (2026-05-11)

```bash
source .venv/bin/activate
python -m pytest tests/ --cov=agent --cov-fail-under=80 -q
# → 1107 passed, 3 skipped | coverage 80.55% | ruff 0 errors
```
