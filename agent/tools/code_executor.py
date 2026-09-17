from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.core.types import ToolResult

# Pre-execution safety scan. Container isolation + timeout are the primary controls;
# this blocks obvious foot-guns before we even fork a subprocess.
# Note: leading \b prevents matching substrings inside longer identifiers
# (e.g. "xeval(" won't match). No trailing \b — the explicit "\." or "\(" already
# anchors the tail, and a trailing \b would fail after "(" because both "(" and
# a following "'" are non-word characters.
_FORBIDDEN_PATTERNS = re.compile(
    r"\b(?:"
    r"os\.system|os\.popen|os\.exec[lv]|os\.fork|os\.kill|"
    r"subprocess\.|shutil\.rmtree|socket\.(?:socket|create_connection)|"
    r"__import__\s*\(|compile\s*\(|eval\s*\(|exec\s*\(|"
    r"ctypes\.|multiprocessing\.|threading\.Thread"
    r")"
)

_DEFAULT_TIMEOUT_S = 30
_MAX_TIMEOUT_S = 600
_MAX_OUTPUT_BYTES = 256_000


def _ast_summary(code: str) -> tuple[str, dict[str, Any]]:
    """Walk the AST to build a math_trace-style summary; also surfaces SyntaxError early."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}", {"valid": False, "error": str(exc)}

    stmt_count = len(tree.body)
    imports: list[str] = []
    funcs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            funcs.append(node.name)

    imports_uniq = sorted(set(imports))
    lines = [
        f"AST: {stmt_count} top-level statement(s), "
        f"{len(imports_uniq)} import(s), {len(funcs)} function def(s)."
    ]
    if imports_uniq:
        lines.append("Imports: " + ", ".join(imports_uniq))
    if funcs:
        lines.append("Functions: " + ", ".join(funcs))
    return "\n".join(lines), {
        "valid": True,
        "statements": stmt_count,
        "imports": imports_uniq,
        "functions": funcs,
    }


def _run_subprocess(code: str, timeout_s: int, workdir: str) -> dict[str, Any]:
    """Execute `code` in a fresh Python subprocess. Captures stdout/stderr/rc/elapsed."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=workdir, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script = f.name
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=workdir,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return {
            "stdout": proc.stdout[:_MAX_OUTPUT_BYTES],
            "stderr": proc.stderr[:_MAX_OUTPUT_BYTES],
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 4),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
        return {
            "stdout": stdout[:_MAX_OUTPUT_BYTES],
            "stderr": stderr[:_MAX_OUTPUT_BYTES],
            "returncode": -1,
            "elapsed_seconds": float(timeout_s),
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


class CodeExecutorTool:
    """Executes Python code in an isolated subprocess with timeout + static safety scan.

    Safety model (defence in depth):
      1. Regex scan blocks the most dangerous patterns before fork.
      2. `ast.parse` catches syntax errors up front.
      3. Subprocess isolation + timeout via `subprocess.run(timeout=...)`.
      4. Work happens in a tempdir so any file writes are contained.
    Container isolation is the outer layer — this tool assumes the caller already
    runs the agent inside a sandboxed OCI container.

    The executor callable is injectable so unit tests can run without spawning processes.
    """

    name: str = "code_executor"
    description: str = (
        "Executes Python code in an isolated subprocess and returns stdout/stderr/returncode. "
        "Input: JSON with 'code' (required), 'timeout_s' (default 30, max 600), "
        "'workdir' (optional existing path; otherwise a temp dir is used)."
    )
    task_type: str = "code"

    def __init__(
        self,
        executor: Callable[[str, int, str], dict[str, Any]] | None = None,
    ) -> None:
        self._executor = executor or _run_subprocess

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return ToolResult(
                status="error",
                data=None,
                explanation="Input was not valid JSON.",
                math_trace="",
                error=f"JSONDecodeError: {exc}",
            )

        code = (payload.get("code") or "").strip()
        if not code:
            return ToolResult(
                status="error",
                data=None,
                explanation="No code provided.",
                math_trace="",
                error="ValueError: code is required",
            )

        try:
            timeout_s = int(payload.get("timeout_s", _DEFAULT_TIMEOUT_S))
        except (TypeError, ValueError):
            timeout_s = _DEFAULT_TIMEOUT_S
        timeout_s = max(1, min(timeout_s, _MAX_TIMEOUT_S))

        # 1. Forbidden-pattern scan.
        forbidden = _FORBIDDEN_PATTERNS.search(code)
        if forbidden:
            return ToolResult(
                status="error",
                data={"matched": forbidden.group(0)},
                explanation=(
                    f"Code contains forbidden pattern {forbidden.group(0)!r}. Refusing to execute."
                ),
                math_trace="",
                error="SafetyError: forbidden pattern",
            )

        # 2. AST summary / syntax check.
        math_trace, meta = _ast_summary(code)
        if not meta.get("valid"):
            return ToolResult(
                status="error",
                data=None,
                explanation="Code failed to parse.",
                math_trace=math_trace,
                error=f"SyntaxError: {meta.get('error', '')}",
            )

        # 3. Workdir: user-supplied existing dir, else ephemeral temp dir.
        workdir_arg = payload.get("workdir")
        cleanup: tempfile.TemporaryDirectory | None = None
        if workdir_arg and Path(workdir_arg).is_dir():
            workdir = str(workdir_arg)
        else:
            cleanup = tempfile.TemporaryDirectory(prefix="ds-exec-")
            workdir = cleanup.name

        try:
            result = self._executor(code, timeout_s, workdir)
        finally:
            if cleanup is not None:
                cleanup.cleanup()

        if result["timed_out"]:
            return ToolResult(
                status="error",
                data=result,
                explanation=f"Execution exceeded timeout of {timeout_s}s.",
                math_trace=math_trace,
                error="TimeoutExpired",
            )
        if result["returncode"] != 0:
            return ToolResult(
                status="error",
                data=result,
                explanation=f"Execution failed with return code {result['returncode']}.",
                math_trace=math_trace,
                error=f"NonZeroExit: {result['returncode']}",
            )

        return ToolResult(
            status="ok",
            data=result,
            explanation=(
                f"Executed {meta['statements']} statement(s) in "
                f"{result['elapsed_seconds']}s."
            ),
            math_trace=math_trace,
        )

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result))


def register() -> CodeExecutorTool:
    """Register the tool in the global ToolRegistry. Safe to call multiple times."""
    from agent.core.tool_registry import ToolRegistry
    tool = CodeExecutorTool()
    ToolRegistry.get()._tools[tool.name] = tool  # type: ignore[attr-defined]
    return tool
