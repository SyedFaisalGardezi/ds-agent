"""Tests for agent/core/llm_router.py and agent/core/tool_registry.py."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.core.tool_registry import ToolRegistry, register_tool, tool

# ── ToolRegistry ────────────────────────────────────────────────────────────���─

@pytest.fixture(autouse=True)
def reset_registry():
    """Isolate each test — reset singleton between tests."""
    original = ToolRegistry._instance
    ToolRegistry._instance = None
    yield
    ToolRegistry._instance = original


def _make_tool(name, task="default"):
    t = SimpleNamespace(name=name, description="desc", task_type=task)
    return t


def test_tool_registry_get_creates_singleton():
    r1 = ToolRegistry.get()
    r2 = ToolRegistry.get()
    assert r1 is r2


def test_tool_registry_register_and_all():
    r = ToolRegistry.get()
    t = _make_tool("my_tool")
    r.register(t)
    assert t in r.all()


def test_tool_registry_getitem():
    r = ToolRegistry.get()
    t = _make_tool("lookup_tool")
    r.register(t)
    assert r["lookup_tool"] is t


def test_tool_registry_contains():
    r = ToolRegistry.get()
    t = _make_tool("exists_tool")
    r.register(t)
    assert "exists_tool" in r
    assert "nonexistent" not in r


def test_register_tool_function():
    t = _make_tool("func_reg_tool")
    result = register_tool(t)
    assert result is t
    assert "func_reg_tool" in ToolRegistry.get()


def test_tool_decorator_registers_instance():
    @tool(task_type="code")
    class MyTool:
        name = "decorator_tool"
        description = "test tool"

    r = ToolRegistry.get()
    assert "decorator_tool" in r
    assert r["decorator_tool"].task_type == "code"


def test_tool_registry_all_returns_list():
    r = ToolRegistry.get()
    r.register(_make_tool("a"))
    r.register(_make_tool("b"))
    tools = r.all()
    assert isinstance(tools, list)
    assert len(tools) >= 2


# ── LLMRouter ─────────────────────────────────────────────────────────────────

def test_llm_router_model_for_known_task(tmp_path, monkeypatch):
    """_load_map without YAML falls back to TASK_MODEL_MAP."""
    monkeypatch.setattr(
        "agent.core.llm_router._MODELS_YAML",
        tmp_path / "nonexistent.yaml",
    )
    from agent.core.llm_router import LLMRouter
    r = LLMRouter()
    assert r.model_for("default") != ""
    assert r.model_for("unknown_task") == r.model_for("default")


def test_llm_router_load_map_with_yaml(tmp_path, monkeypatch):
    """When YAML exists, its task_model_map is merged."""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml.dump({"task_model_map": {"plan": "custom-model:v1"}}))
    monkeypatch.setattr("agent.core.llm_router._MODELS_YAML", yaml_path)
    from agent.core.llm_router import LLMRouter
    r = LLMRouter()
    assert r.model_for("plan") == "custom-model:v1"


def test_llm_router_model_for_returns_default_for_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.core.llm_router._MODELS_YAML",
        tmp_path / "missing.yaml",
    )
    from agent.core.llm_router import LLMRouter
    r = LLMRouter()
    default = r.model_for("default")
    assert r.model_for("totally_unknown_key") == default


def test_llm_router_get_caches_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.core.llm_router._MODELS_YAML",
        tmp_path / "missing.yaml",
    )
    # Mock OllamaLLM so no real network call is made
    import agent.core.llm_router as llm_mod

    class FakeLLM:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(llm_mod, "OllamaLLM", FakeLLM)
    from agent.core.llm_router import LLMRouter
    r = LLMRouter()
    # Calling get() twice with same task type returns same instance
    llm1 = r.get("default")
    llm2 = r.get("default")
    assert llm1 is llm2
