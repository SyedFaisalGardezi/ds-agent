# DS Agent

An autonomous data-science agent that runs locally. It behaves like a chat assistant for data analysis, machine learning, data engineering, and AI-API work. It plans, calls tools, reads each result, reflects, and adapts, and it can point at a working directory, read and edit files, query databases and warehouses, train models, and scaffold deployment and CI/CD.

The agent runs on a local Qwen model served through Ollama and a FastAPI server with a browser UI.

## Requirements

- macOS on Apple Silicon or a Linux machine. The default model targets Apple MLX; on Linux use the CUDA build.
- Python 3.11 or newer. Development was on 3.13.
- [Ollama](https://ollama.com) for local model serving.
- About 20 GB of free disk for the model.

## Setup

1. Install Ollama and pull the models the agent uses.

   ```bash
   ollama pull qwen3.8:27b-mlx
   ollama pull nomic-embed-text
   ```

   The chat model is `qwen3.8:27b-mlx` and `nomic-embed-text` powers long-term memory. Override the chat model with `DSAGENT_SCOPE_MODEL` if you use a different one.

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

Two switches sit next to Send. Multi-agent is on by default and delegates work to specialist agents; turn it off for a faster single-agent loop. Writes is off by default, so file edits are only suggested as diffs; turn it on to let the agent apply edits to disk. Set a working directory to give the agent awareness of the files in a project folder.

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
| `DSAGENT_MULTIAGENT` | `1` | Use the multi-agent orchestrator. Set `0` for the single-agent loop. |
| `DSAGENT_ALLOW_WRITES` | `0` | Suggest file edits as diffs. Set `1` to apply edits to disk. |
| `DSAGENT_CRITIC` | `1` | Enable the reflexion critic. |
| `DSAGENT_NUM_CTX` | `16384` | Context window sent to Ollama. |
| `DSAGENT_MAX_ORCH_ROUNDS` | `6` | Orchestrator delegation budget. |
| `DSAGENT_SPECIALIST_STEPS` | `6` | Steps per specialist. |

The per-message toggles in the UI override the server defaults for that message.

## Notes

- Data sources are read-only. Destructive SQL is refused.
- The agent generates deployment and CI/CD artifacts but does not push images or deploy to a live system. Those steps run from the generated CI/CD with your own credentials.
- Generated outputs, caches, the local memory store, and the `.env` file are not tracked. See `.gitignore`.
