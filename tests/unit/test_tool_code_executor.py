from __future__ import annotations

import json

import pytest

from agent.tools.code_executor import CodeExecutorTool, register


def _fake(out: str = "", err: str = "", rc: int = 0, timed_out: bool = False, elapsed: float = 0.01):
    """Build a fake executor callable with the same shape as _run_subprocess."""
    def _exec(code, timeout_s, workdir):
        return {
            "stdout": out,
            "stderr": err,
            "returncode": rc,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
        }
    return _exec


def test_rejects_invalid_json():
    tool = CodeExecutorTool(executor=_fake())
    r = tool.run("not-json")
    assert r.status == "error"
    assert r.error.startswith("JSONDecodeError")


def test_rejects_blank_code():
    tool = CodeExecutorTool(executor=_fake())
    r = tool.run(json.dumps({"code": "   "}))
    assert r.status == "error"
    assert "code is required" in r.error


@pytest.mark.parametrize("bad_code", [
    "import os; os.system('rm -rf /')",
    "import subprocess; subprocess.run(['ls'])",
    "x = __import__('os')",
    "eval('1+1')",
    "exec('x=1')",
    "import ctypes; ctypes.CDLL('libc.so.6')",
])
def test_rejects_forbidden_patterns(bad_code):
    tool = CodeExecutorTool(executor=_fake())
    r = tool.run(json.dumps({"code": bad_code}))
    assert r.status == "error"
    assert "forbidden" in r.error.lower()


def test_rejects_syntax_error():
    tool = CodeExecutorTool(executor=_fake())
    r = tool.run(json.dumps({"code": "def ("}))
    assert r.status == "error"
    assert r.error.startswith("SyntaxError")
    assert "parse" in r.explanation.lower()


def test_executes_and_returns_stdout():
    captured = {}
    def fake_exec(code, timeout_s, workdir):
        captured["code"] = code
        captured["timeout_s"] = timeout_s
        return {"stdout": "42\n", "stderr": "", "returncode": 0, "timed_out": False, "elapsed_seconds": 0.02}
    tool = CodeExecutorTool(executor=fake_exec)
    r = tool.run(json.dumps({"code": "print(21*2)", "timeout_s": 5}))
    assert r.status == "ok"
    assert r.data["stdout"] == "42\n"
    assert captured["timeout_s"] == 5
    assert "AST" in r.math_trace


def test_non_zero_exit_is_error():
    tool = CodeExecutorTool(executor=_fake(err="boom", rc=1))
    r = tool.run(json.dumps({"code": "x = 1"}))
    assert r.status == "error"
    assert "NonZeroExit: 1" in r.error
    assert r.data["stderr"] == "boom"


def test_timeout_is_error():
    tool = CodeExecutorTool(executor=_fake(timed_out=True, elapsed=30.0))
    r = tool.run(json.dumps({"code": "x = 1", "timeout_s": 2}))
    assert r.status == "error"
    assert r.error == "TimeoutExpired"


def test_timeout_is_clamped_to_max():
    captured = {}
    def fake(code, t, w):
        captured["t"] = t
        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "elapsed_seconds": 0.0}
    tool = CodeExecutorTool(executor=fake)
    tool.run(json.dumps({"code": "x=1", "timeout_s": 99999}))
    assert captured["t"] == 600


def test_timeout_clamped_to_min():
    captured = {}
    def fake(code, t, w):
        captured["t"] = t
        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "elapsed_seconds": 0.0}
    tool = CodeExecutorTool(executor=fake)
    tool.run(json.dumps({"code": "x=1", "timeout_s": 0}))
    assert captured["t"] == 1


def test_ast_trace_lists_imports_and_functions():
    tool = CodeExecutorTool(executor=_fake())
    code = "import math\nimport json\ndef foo():\n    return math.pi"
    r = tool.run(json.dumps({"code": code}))
    assert r.status == "ok"
    assert "math" in r.math_trace
    assert "json" in r.math_trace
    assert "foo" in r.math_trace


def test_invalid_timeout_falls_back_to_default():
    captured = {}
    def fake(code, t, w):
        captured["t"] = t
        return {"stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "elapsed_seconds": 0.0}
    tool = CodeExecutorTool(executor=fake)
    tool.run(json.dumps({"code": "x=1", "timeout_s": "not-a-number"}))
    assert captured["t"] == 30


def test_register_adds_to_registry():
    from agent.core.tool_registry import ToolRegistry
    tool = register()
    assert ToolRegistry.get()["code_executor"] is tool


def test_real_subprocess_execution():
    """Integration: actually spawns a Python subprocess (no mock)."""
    tool = CodeExecutorTool()
    r = tool.run(json.dumps({"code": "print('hello world')", "timeout_s": 10}))
    assert r.status == "ok", r.error
    assert r.data["stdout"].strip() == "hello world"
    assert r.data["returncode"] == 0


def test_real_subprocess_captures_stderr_and_nonzero_exit():
    tool = CodeExecutorTool()
    r = tool.run(json.dumps({"code": "import sys; sys.exit(2)", "timeout_s": 10}))
    assert r.status == "error"
    assert "NonZeroExit: 2" in r.error


def test_real_subprocess_timeout():
    tool = CodeExecutorTool()
    r = tool.run(json.dumps({"code": "while True: pass", "timeout_s": 1}))
    assert r.status == "error"
    assert r.error == "TimeoutExpired"
