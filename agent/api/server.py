"""FastAPI server exposing ds-agent pipeline stages and tools.

Endpoints:
    GET  /health                      liveness probe
    GET  /stages                      list available pipeline stages
    GET  /tools                       list available tools
    POST /pipelines/profile           DataProfiler on records
    POST /pipelines/quality           QualityScorer on records
    POST /pipelines/statistics/mi     Mutual-information scores
    POST /pipelines/features          FeatureGenerator
    POST /pipelines/task              TaskDetector
    POST /pipelines/train             TaskDetector → ModelTrainer → Leaderboard
    POST /pipelines/drift             DriftDetector baseline vs current
    GET  /leaderboard                 top-N runs
    POST /tools/{name}                dispatch to a tool's run(input_json)

All responses share the ToolResult envelope (status/data/explanation/
math_trace/error). Models (trainer output) are not serialised — only their
metrics, feature-names, and run identifiers.
"""
from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.core.types import ToolResult
from agent.monitoring.drift_detector import DriftDetector
from agent.pipeline.eda.profiler import DataProfiler
from agent.pipeline.eda.quality_scorer import QualityScorer
from agent.pipeline.eda.statistics import StatisticsAnalyser
from agent.pipeline.evaluator.leaderboard import Leaderboard
from agent.pipeline.feature_engineering.generator import FeatureGenerator
from agent.pipeline.model_builder.task_detector import TaskDetector
from agent.pipeline.model_builder.trainer import ModelTrainer

# ---- tool registry ----------------------------------------------------- #
from agent.tools.cost_estimator import CostEstimatorTool
from agent.tools.feature_engineer import FeatureEngineerTool
from agent.tools.hypothesis_test import HypothesisTestTool
from agent.tools.pii_detector import PIIDetectorTool
from agent.tools.plot_tool import PlotTool
from agent.tools.powerbi_exporter import PowerBIExporterTool
from agent.tools.report_generator import ReportGeneratorTool

# Load .env into the process environment so DSAGENT_LLM_BACKEND, the Anthropic
# key, and the other settings take effect on `uvicorn agent.api.server:app`
# without the user having to export them by hand. Existing shell exports win
# (override=False), and this runs before create_app() builds the LLM client.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_TOOLS: dict[str, Any] = {
    "cost_estimator": CostEstimatorTool,
    "pii_detector": PIIDetectorTool,
    "feature_engineer": FeatureEngineerTool,
    "plot": PlotTool,
    "hypothesis_test": HypothesisTestTool,
    "report_generator": ReportGeneratorTool,
    "powerbi_exporter": PowerBIExporterTool,
}

_STAGES = [
    "profile", "quality", "statistics/mi", "features",
    "task", "train", "drift",
]

# ---- request models ---------------------------------------------------- #


class RecordsBody(BaseModel):
    """Generic body carrying a list-of-dicts table."""
    records: list[dict[str, Any]] = Field(..., description="row-oriented records")


class TargetBody(RecordsBody):
    target: str


class FeaturesBody(TargetBody):
    max_pairs: int = 5
    max_onehot_cardinality: int = 8


class TrainBody(TargetBody):
    estimator: str = "rf"
    n_estimators: int = 100
    test_size: float = 0.25
    random_state: int = 0
    run_id: str | None = None
    log_to_leaderboard: bool = True


class DriftBody(BaseModel):
    baseline: list[dict[str, Any]]
    current: list[dict[str, Any]]


class ToolBody(BaseModel):
    input_json: dict[str, Any] = Field(default_factory=dict)


# ---- helpers ----------------------------------------------------------- #


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None and coerce numpy scalars."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.head(1000).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _envelope(r: ToolResult) -> dict[str, Any]:
    """Serialise a ToolResult, scrubbing non-JSON-friendly payloads."""
    data = r.data
    # Drop non-serialisable objects (trained model, matplotlib figures, …).
    if isinstance(data, dict):
        data = {k: v for k, v in data.items()
                if k not in {"model", "figure", "figures", "frame"}}
        # If a DataFrame is present under 'frame', convert to records.
        if isinstance(r.data, dict) and isinstance(r.data.get("frame"), pd.DataFrame):
            data["frame_records"] = r.data["frame"].head(1000).to_dict(orient="records")
            data["frame_shape"] = list(r.data["frame"].shape)
    status_code = 200 if r.status != "error" else 400
    body = {
        "status": r.status,
        "data": _sanitize(data),
        "explanation": r.explanation,
        "math_trace": r.math_trace,
        "error": r.error,
    }
    return {"_status_code": status_code, "body": body}


def _respond(r: ToolResult) -> dict[str, Any]:
    env = _envelope(r)
    if env["_status_code"] >= 400:
        raise HTTPException(status_code=env["_status_code"], detail=env["body"])
    return env["body"]


def _df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=422, detail="records must be non-empty")
    return pd.DataFrame.from_records(records)


def _use_claude() -> bool:
    """Pick the LLM backend from a single toggle.

    USE_CLAUDE (yes/no/true/false/1/0/on/off) is the switch. When it is not set,
    fall back to the legacy DSAGENT_LLM_BACKEND=anthropic for compatibility.
    """
    raw = os.environ.get("USE_CLAUDE")
    if raw is not None:
        return raw.strip().lower() in ("yes", "y", "true", "1", "on", "claude")
    return os.environ.get("DSAGENT_LLM_BACKEND", "ollama").lower() == "anthropic"


def make_llm_client():
    """The LLM client the USE_CLAUDE toggle selects (shared by the server and
    the evaluation harness, so both run the same backend)."""
    from agent.api import trace
    from agent.api.llm import AnthropicClient, OllamaClient
    llm = AnthropicClient() if _use_claude() else OllamaClient()
    return trace.install(llm)   # records calls only when DSAGENT_TRACE_DIR is set


# ---- app factory ------------------------------------------------------- #


def create_app(leaderboard_db: Path | str | None = None) -> FastAPI:
    """Build the FastAPI app. `leaderboard_db` lets tests use a tmp DB."""
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from agent.api.chat import make_router as make_chat_router
    from agent.api.jobs import JobRunner
    from agent.api.sessions import SessionStore

    app = FastAPI(title="ds-agent", version="0.1.0")
    lb = Leaderboard(db_path=leaderboard_db)
    store = SessionStore()
    jobs = JobRunner(max_workers=2)
    # One toggle picks the backend: USE_CLAUDE=yes runs the Claude API (needs
    # ANTHROPIC_API_KEY and console credits, billed separately from a claude.ai
    # subscription); anything else runs local Ollama. The key can stay in .env
    # unused when the toggle is off, so nothing needs commenting out.
    llm = make_llm_client()


    # Memory singletons — graceful init; None means silently disabled
    ltm = None
    schema_cache = None
    try:
        from agent.memory.long_term import LongTermMemory
        ltm = LongTermMemory()
    except Exception:
        pass
    try:
        from agent.memory.schema_cache import SchemaCache
        schema_cache = SchemaCache()
    except Exception:
        pass

    app.include_router(make_chat_router(store=store, leaderboard=lb,
                                        jobs=jobs, llm=llm,
                                        ltm=ltm, schema_cache=schema_cache))

    # Multi-user auth. The router is always available; the gate below requires a
    # valid token only when DSAGENT_AUTH_ENABLED=1, so tests and single-user use
    # are unaffected.
    from agent.api.auth import auth_enabled, make_auth_router, verify_token
    if auth_enabled():
        from agent.api.auth import _secret, smtp_configured
        _secret()
        if not smtp_configured():
            raise RuntimeError("Authentication requires configured SMTP delivery")
    app.include_router(make_auth_router())

    _public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    _public_prefixes = ("/auth", "/static")

    @app.middleware("http")
    async def _auth_gate(request, call_next):  # noqa: ANN001, ANN202
        if not auth_enabled():
            return await call_next(request)
        path = request.url.path
        if path in _public_paths or any(path.startswith(p)
                                        for p in _public_prefixes):
            return await call_next(request)
        authz = request.headers.get("Authorization", "")
        token = authz[7:] if authz.startswith("Bearer ") else None
        if not token:
            token = request.cookies.get("ds_token")
        email = verify_token(token)
        if not email:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if path.startswith(("/pipelines", "/tools", "/leaderboard")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Use owner-scoped chat endpoints in multi-user mode"}, status_code=403)
        request.state.user_email = email
        return await call_next(request)

    app.state.session_store = store
    app.state.jobs = jobs
    app.state.llm = llm
    app.state.ltm = ltm
    app.state.schema_cache = schema_cache

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def root() -> FileResponse | dict[str, Any]:
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return {
            "service": "ds-agent", "version": "0.1.0", "docs": "/docs",
            "endpoints": {"chat": "/chat/sessions"},
        }

    @app.on_event("shutdown")
    def _shutdown() -> None:
        jobs.shutdown()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "ds-agent", "version": "0.1.0",
                "llm": {"model": llm.model, "available": llm.is_available()}}

    @app.get("/stages")
    def stages() -> dict[str, list[str]]:
        return {"stages": _STAGES}

    @app.get("/tools")
    def tools() -> dict[str, list[str]]:
        return {"tools": sorted(_TOOLS)}

    # ---- pipeline stages --------------------------------------------- #

    @app.post("/pipelines/profile")
    def profile(body: RecordsBody) -> dict[str, Any]:
        df = _df(body.records)
        run_id = f"api-{uuid.uuid4().hex[:8]}"
        return _respond(DataProfiler().run(df, run_id=run_id))

    @app.post("/pipelines/quality")
    def quality(body: RecordsBody) -> dict[str, Any]:
        return _respond(QualityScorer().score(_df(body.records)))

    @app.post("/pipelines/statistics/mi")
    def statistics_mi(body: TargetBody) -> dict[str, Any]:
        return _respond(
            StatisticsAnalyser().mutual_information(
                _df(body.records), target=body.target))

    @app.post("/pipelines/features")
    def features(body: FeaturesBody) -> dict[str, Any]:
        gen = FeatureGenerator(
            max_pairs=body.max_pairs,
            max_onehot_cardinality=body.max_onehot_cardinality,
        )
        return _respond(gen.generate(_df(body.records), target=body.target))

    @app.post("/pipelines/task")
    def task(body: TargetBody) -> dict[str, Any]:
        return _respond(TaskDetector().infer(_df(body.records), target=body.target))

    @app.post("/pipelines/train")
    def train(body: TrainBody) -> dict[str, Any]:
        df = _df(body.records)
        det = TaskDetector().infer(df, target=body.target)
        if det.status != "ok":
            raise HTTPException(status_code=400, detail=det.explanation)
        task_type = det.data["task"]
        trainer = ModelTrainer(
            task_type=task_type,
            config={
                "target": body.target,
                "estimator": body.estimator,
                "n_estimators": body.n_estimators,
                "test_size": body.test_size,
                "random_state": body.random_state,
            },
        )
        run_id = body.run_id or f"api-{uuid.uuid4().hex[:8]}"
        trained = trainer.run(df, run_id=run_id)
        if trained.status != "ok":
            raise HTTPException(status_code=400, detail=trained.explanation)
        metrics = trained.data.get("metrics", {}) or {}
        if body.log_to_leaderboard:
            lb.log_run(
                run_id=run_id, model_type=body.estimator,
                hyperparams={"n_estimators": body.n_estimators,
                             "test_size": body.test_size,
                             "random_state": body.random_state},
                metrics=metrics,
                train_time_s=0.0, inference_ms=0.0,
                model_mb=0.0, shap_runtime_s=0.0,
            )
        return {
            "status": "ok",
            "data": _sanitize({
                "run_id": run_id,
                "task": task_type,
                "metrics": metrics,
                "feature_names": trained.data.get("feature_names", []),
                "n_train": trained.data.get("n_train"),
                "n_test": trained.data.get("n_test"),
            }),
            "explanation": trained.explanation,
            "math_trace": trained.math_trace,
            "error": None,
        }

    @app.post("/pipelines/drift")
    def drift(body: DriftBody) -> dict[str, Any]:
        base = _df(body.baseline)
        cur = _df(body.current)
        return _respond(DriftDetector().check_all_features(base, cur))

    # ---- leaderboard ------------------------------------------------- #

    @app.get("/leaderboard")
    def leaderboard(metric: str = "accuracy", n: int = 10,
                    higher_is_better: bool = True) -> dict[str, Any]:
        return _respond(lb.top_n(metric, n=n, higher_is_better=higher_is_better))

    # ---- tools ------------------------------------------------------- #

    @app.post("/tools/{name}")
    def run_tool(name: str, body: ToolBody) -> dict[str, Any]:
        cls = _TOOLS.get(name)
        if cls is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown tool {name!r}. Available: {sorted(_TOOLS)}",
            )
        try:
            tool = cls()
            result = tool.run(body.input_json)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not isinstance(result, ToolResult):
            raise HTTPException(status_code=500,
                                detail="tool did not return a ToolResult")
        return _respond(result)

    # Expose leaderboard handle for graceful shutdown (tests call this).
    app.state.leaderboard = lb
    return app


# Default app instance for `uvicorn agent.api.server:app`.
app = create_app()

# Default bind port for ds-agent.
API_PORT = 9090


def main() -> None:
    """Run the API on port 9090. `python -m agent.api.server`."""
    import uvicorn

    uvicorn.run("agent.api.server:app", host="0.0.0.0", port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
