# DS Agent

An autonomous data-science agent that runs locally. It behaves like a chat assistant for data analysis, machine learning, data engineering, and AI-API work. It plans, calls tools, reads each result, reflects, and adapts, and it can point at a working directory, read and edit files, query databases and warehouses, train models, and scaffold deployment and CI/CD.

The agent runs on a local model served through Ollama and a FastAPI server with a browser UI.

## Requirements

- macOS on Apple Silicon or a Linux machine. The default model targets Apple MLX; on Linux use the CUDA build.
- Python 3.11 or newer. Development was on 3.13.
- [Ollama](https://ollama.com) for local model serving.
- About 15 GB of free disk for the model, and 32 GB of memory.

## Setup

1. Install Ollama and pull the models the agent uses.

   ```bash
   ollama pull gpt-oss:20b
   ollama pull nomic-embed-text
   ```

   The chat model is `gpt-oss:20b` and `nomic-embed-text` powers long-term memory. Override the chat model with `DSAGENT_SCOPE_MODEL` if you use a different one. `gpt-oss:20b` is a mixture-of-experts model with about 3.6B active parameters per token, which is why it runs comfortably on 32 GB where a dense 27B swaps. It is also a reasoning model: it thinks before answering, so token budgets have to cover the thinking, and `think: false` must not be sent to it.

2. Create a virtual environment and install the package.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Create your environment file from the template and fill in the values you need.

   ```bash
   cp .env.example .env
   ```

   Only the Ollama settings are required for local use. The database, warehouse, cloud, and HPC entries are optional and used only when you connect to those sources.

## Run

Start Ollama if it is not already running.

```bash
ollama serve
```

In a second terminal, start the agent server.

```bash
source .venv/bin/activate
python -m uvicorn agent.api.server:app --port 9090
```

Open http://localhost:9090 in a browser. Check health with:

```bash
curl -s localhost:9090/health
```

The health check should report `llm.available: true`.

### UI controls

Two switches sit next to Send. Multi-agent is off by default: one agent keeps one continuous conversation, which suits a pipeline where each step depends on the last. Turn it on for work that splits into genuinely independent strands, bearing in mind that each handoff starts a specialist with only its sub-goal. Writes is off by default, so file edits are only suggested as diffs.

Set a working directory to give the agent awareness of the files in a project folder. The path must already exist. A path that does not exist is treated as a typo and rejected, with a suggestion when a similarly named folder sits beside it, because silently creating the folder gives the session an empty workspace that looks like it worked.

### Nothing is written without your approval

Turning Writes on lets the agent propose changes to disk. It does not let it make them. Every write, edit, and generated file is staged in the Pending changes panel with its diff, and a change that would create a new folder says so. You approve or reject each one, or all of them together, and only an approved change is applied.

The agent never creates a folder on its own. `write_file` refuses a path whose parent folder is missing unless it is called with `create_dirs=true`, which the tool is instructed to use only when you asked for a new folder. Deleting is not something the agent can do at all: there is no delete tool, and the code executor blocks `os.remove`, `os.unlink`, `os.rmdir`, `Path.unlink`, `shutil.rmtree`, and `shutil.move` so generated code cannot route around the approval queue.

The endpoints behind the panel are `GET /chat/sessions/{sid}/pending`, `POST /chat/sessions/{sid}/pending/apply`, and `POST /chat/sessions/{sid}/pending/discard`. Both POSTs take `{"ids": [...]}` for a subset or `{"ids": null}` for everything.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

The suite runs with a coverage gate of 80 percent.

## How it works

The request flow, the agent loop, the critic and reflection, the multi-agent orchestrator, the memory layers, and every tool are documented in [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md). A plan for fine-tuning the local model on data-science skills is in [FINETUNING_PLAN.md](FINETUNING_PLAN.md).

At a high level:

- A FastAPI server accepts messages and runs each as a background job.
- The agent classifies the request, then either runs a deterministic pipeline for well-specified builds or an adaptive loop for open-ended analysis.
- In the adaptive loop the model calls tools natively, a critic judges each result and returns a finding and a recommended next step, and the loop re-plans from that output. A task focus keeps exploration on the expected outcome.
- The multi-agent orchestrator delegates sub-goals to five role-specialists, each running the same critic loop on a subset of the tools over a shared session kernel.
- Long-term memory recalls relevant past work, and the running session is written back for next time.

## Configuration

The behaviour flags are read from the environment. The common ones:

| Variable | Default | Effect |
| --- | --- | --- |
| `DSAGENT_MULTIAGENT` | `0` | One agent, one conversation. Set `1` for the orchestrator, which suits independent strands of work rather than a pipeline. |
| `DSAGENT_AGENT_STEPS` | `24` | Tool-call budget for the agentic loop, separate from the deterministic plan's cap. |
| `DSAGENT_ALLOW_WRITES` | `0` | Suggest file edits as diffs. Set `1` to let the agent stage changes for your approval. It never applies them unattended. |
| `DSAGENT_CRITIC` | `1` | Enable the reflexion critic. |
| `DSAGENT_CRITIC_ALL` | `0` | Critique every tool result, including file reads. Slower; costs one extra LLM call per step. |
| `DSAGENT_NUM_CTX` | `65536` | Context window. Verified at 12 GB resident with a real 17K-token prompt on gpt-oss:20b. Drop to `8192` for a dense 27B, which thrashes above that. |
| `DSAGENT_LLM_TIMEOUT_S` | `600` | Per-request timeout. One tool-calling step on a large local model can take minutes. |
| `DSAGENT_KEEP_ALIVE` | `30m` | How long Ollama holds the model in memory. Shorten to give memory back sooner, `-1` to hold it indefinitely. |
| `DSAGENT_TOOL_TURN_TOKENS` | `4000` | Output budget for one tool-calling turn. A document the agent writes is an argument to the call, so this bounds its length. |
| `DSAGENT_SCOPE_TOKENS` | `2500` | Output budget for the task-scope call, covering its thinking as well as its answer. |
| `DSAGENT_ROUTER_TOKENS` | `500` | Budget for the act-or-answer decision. |
| `DSAGENT_CRITIC_TOKENS` | `900` | Budget for one critic verdict. |
| `DSAGENT_PLAN_TOKENS` | `8000` | Budget per planning stage. A reasoning model spends most of it thinking. |
| `DSAGENT_THINK` | `1` | Let the model think. Set `0` only for a non-reasoning model such as qwen3, where suppressing it saves time. |
| `DSAGENT_OBS_CHARS` | `4000` | How much of the newest tool result the model reads. Older results are trimmed to `DSAGENT_OBS_TAIL_CHARS` (700). |
| `DSAGENT_SEND_FIGURES` | `0` | Attach plots to the model. Off by default because gpt-oss:20b has no vision. Set `1` only on a vision model. |
| `DSAGENT_MAX_FIGURES_SENT` | `2` | Figures attached per step. |
| `DSAGENT_MAX_FIGURE_CHARS` | `20000` | Size budget per attached figure; larger ones are palette-quantised down to fit. |
| `DSAGENT_MAX_ORCH_ROUNDS` | `6` | Orchestrator delegation budget. |
| `DSAGENT_SPECIALIST_STEPS` | `6` | Steps per specialist. |

The per-message toggles in the UI override the server defaults for that message.

## Notes

- Data sources are read-only. Destructive SQL is refused.
- The agent generates deployment and CI/CD artifacts but does not push images or deploy to a live system. Those steps run from the generated CI/CD with your own credentials.
- Generated outputs, caches, the local memory store, and the `.env` file are not tracked. See `.gitignore`.


## September 2026 audit changes

The current default chat path interprets the complete request, conversation and
brief with the configured local Ollama model. It validates an answer/act/clarify
intent and checks each concrete tool call against the resulting constraints.
Keyword routing helpers are retained for compatibility but are not called by
`run_reactive`. Invalid intent/action decisions stop execution.

Use Python 3.11. Generated custom Python requires Docker and the local sandbox
image: `docker build -f docker/sandbox.Dockerfile -t ds-agent-sandbox:local .`.
It runs with no network, read-only inputs, resource limits and scratch output;
workspace artifacts require approval. There is no host-execution fallback.
Custom Python variables do not persist between container calls; built-in tool
state remains session-scoped. Notebook export records current results without
silently retraining. It is not currently a complete executable training notebook.

Supervised training reserves test rows before model selection, learns
preprocessing within validation folds and exports the fitted preprocessing/model
pipeline. Interactive runs use `DSAGENT_STATE_DIR` (default `~/.ds-agent`) for
MLflow records and DVC-tracked datasets/model manifests. `dvc repro` runs the
explicitly synthetic demonstration fixture in `configs/pipeline.yaml`; then
`dvc repro --dry` checks the declared stages. This is not the user's real data.

See [Thoughts from second Perspective](temp_review/DS_Agent_Review.html#second-perspective)
for implementation details, validation results and remaining limitations.
