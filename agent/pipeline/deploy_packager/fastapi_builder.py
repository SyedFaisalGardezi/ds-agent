"""FastAPIBuilder — scaffold a minimal FastAPI service around a trained model."""
from __future__ import annotations

from pathlib import Path

from agent.core.types import ToolResult

_APP_TEMPLATE = '''"""Auto-generated FastAPI service for {model_uri}."""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", {model_uri!r})
_model = joblib.load(MODEL_URI)

app = FastAPI(title="ds-agent model server")


class PredictIn(BaseModel):
    rows: list[list[float]]


class PredictOut(BaseModel):
    predictions: list[Any]


@app.get("/health")
def health() -> dict:
    return {{"status": "ok", "model_uri": MODEL_URI}}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> PredictOut:
    X = np.asarray(payload.rows, dtype=float)
    preds = _model.predict(X).tolist()
    return PredictOut(predictions=preds)
'''

_DOCKERFILE_TEMPLATE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./app.py
ENV MODEL_URI=/app/model.joblib
EXPOSE 8000
CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
"""

_REQUIREMENTS = "fastapi>=0.110\nuvicorn[standard]>=0.29\njoblib>=1.4\nnumpy>=1.26\npydantic>=2.5\n"


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
        (out / "requirements.txt").write_text(_REQUIREMENTS)
        return ToolResult(
            status="ok",
            data={"output_dir": str(out),
                  "files": ["app.py", "Dockerfile", "requirements.txt"],
                  "model_uri": model_uri},
            explanation=f"Scaffolded FastAPI service at {out}.",
            math_trace="Rendered 3 template file(s).",
        )
