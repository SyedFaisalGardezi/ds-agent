"""FastAPIBuilder — scaffold a minimal FastAPI service around a trained model."""
from __future__ import annotations

from pathlib import Path

from agent.core.types import ToolResult

_APP_TEMPLATE = '''"""Auto-generated FastAPI service for {model_uri}."""
from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", {model_uri!r})
_model = joblib.load(MODEL_URI)

app = FastAPI(title="ds-agent model server")


class PredictIn(BaseModel):
    rows: list[dict[str, Any]]


class PredictOut(BaseModel):
    predictions: list[Any]


@app.get("/health")
def health() -> dict:
    return {{"status": "ok", "model_uri": MODEL_URI}}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> PredictOut:
    X = pd.DataFrame(payload.rows)
    if hasattr(_model, "feature_names_in_"):
        missing = set(_model.feature_names_in_) - set(X.columns)
        if missing:
            from fastapi import HTTPException
            raise HTTPException(422, "Missing features: " + ", ".join(sorted(missing)))
        X = X[list(_model.feature_names_in_)]
    preds = _model.predict(X).tolist()
    return PredictOut(predictions=preds)
'''

_DOCKERFILE_TEMPLATE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./app.py
COPY model.joblib ./model.joblib
ENV MODEL_URI=/app/model.joblib
EXPOSE 8000
CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
"""

_REQUIREMENTS = "fastapi>=0.110\nuvicorn[standard]>=0.29\njoblib>=1.4\nnumpy>=1.26\npydantic>=2.5\npandas>=2.2\nscipy>=1.13\nscikit-learn>=1.5\nlightgbm>=4\nxgboost>=2\n"


class FastAPIBuilder:
    def build(self, model_uri: str, output_dir: str = "deploy") -> ToolResult:
        if not model_uri:
            return ToolResult(status="error", data=None,
                              explanation="model_uri required",
                              math_trace="", error="ValueError")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "app.py").write_text(_APP_TEMPLATE.format(model_uri=model_uri))
        (out / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE)
        from importlib.metadata import version
        dependencies = ("fastapi", "uvicorn", "joblib", "numpy", "pydantic", "pandas",
                        "scipy", "scikit-learn", "lightgbm", "xgboost")
        (out / "requirements.txt").write_text("\n".join(
            f"{name}=={version(name)}" for name in dependencies) + "\n")
        return ToolResult(
            status="ok",
            data={"output_dir": str(out),
                  "files": ["app.py", "Dockerfile", "requirements.txt"],
                  "model_uri": model_uri},
            explanation=f"Scaffolded FastAPI service at {out}.",
            math_trace="Rendered 3 template file(s).",
        )
