from __future__ import annotations

import os
from pathlib import Path

import yaml
from langchain_ollama import OllamaLLM

# Fallback model for every role when configs/models.yaml is missing a key.
# Derived from the same env var the rest of the agent uses, so the router can
# never point at a model that isn't installed (previously hardcoded to the
# deleted gemma4:latest / never-installed qwen2.5-coder:7b).
_FALLBACK_MODEL = os.environ.get(
    "DSAGENT_SCOPE_MODEL",
    "qwen3.8:27b-mlx",
)

TASK_MODEL_MAP: dict[str, str] = {
    role: _FALLBACK_MODEL
    for role in ("plan", "route", "math", "code", "sql",
                 "long_ctx", "synthesise", "qa", "understand_task", "default")
}

_MODELS_YAML = Path(__file__).parent.parent.parent / "configs" / "models.yaml"


class LLMRouter:
    def __init__(self) -> None:
        self._map = self._load_map()
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._cache: dict[str, OllamaLLM] = {}

    def _load_map(self) -> dict[str, str]:
        if _MODELS_YAML.exists():
            with _MODELS_YAML.open() as f:
                data = yaml.safe_load(f) or {}
            return {**TASK_MODEL_MAP, **data.get("task_model_map", {})}
        return TASK_MODEL_MAP

    def get(self, task_type: str) -> OllamaLLM:
        model = self._map.get(task_type, self._map["default"])
        if model not in self._cache:
            self._cache[model] = OllamaLLM(model=model, base_url=self._base_url)
        return self._cache[model]

    def model_for(self, task_type: str) -> str:
        return self._map.get(task_type, self._map["default"])


router = LLMRouter()
