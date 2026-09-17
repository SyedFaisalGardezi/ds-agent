"""DockerBuilder — thin wrapper around the `docker` CLI."""
from __future__ import annotations

import shutil
import subprocess

from agent.core.types import ToolResult


class DockerBuilder:
    """Build/push images via the local `docker` binary.

    Dependency injection: pass a custom ``runner`` (defaults to
    :func:`subprocess.run`) so tests can assert on commands without actually
    invoking docker.
    """

    def __init__(self, runner=None, which=None, context: str = ".",
                 dockerfile: str = "Dockerfile") -> None:
        self._runner = runner or subprocess.run
        self._which = which or shutil.which
        self._context = context
        self._dockerfile = dockerfile

    def _require_docker(self) -> ToolResult | None:
        if self._which("docker") is None:
            return ToolResult(status="error", data=None,
                              explanation="docker binary not found on PATH",
                              math_trace="", error="DockerNotInstalled")
        return None

    def build(self, tag: str, target: str = "runtime",
              build_args: dict[str, str] | None = None) -> ToolResult:
        if not tag:
            return ToolResult(status="error", data=None,
                              explanation="tag required",
                              math_trace="", error="ValueError")
        missing = self._require_docker()
        if missing:
            return missing
        args = ["docker", "build", "-t", tag, "--target", target,
                "-f", self._dockerfile]
        for k, v in (build_args or {}).items():
            args.extend(["--build-arg", f"{k}={v}"])
        args.append(self._context)
        try:
            result = self._runner(args, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error", data={"cmd": args, "returncode": exc.returncode,
                                      "stderr": getattr(exc, "stderr", "")},
                explanation=f"docker build failed (rc={exc.returncode}).",
                math_trace="", error=f"CalledProcessError: {exc}",
            )
        return ToolResult(
            status="ok",
            data={"tag": tag, "target": target, "cmd": args,
                  "stdout": getattr(result, "stdout", ""),
                  "returncode": getattr(result, "returncode", 0)},
            explanation=f"Built {tag} (target={target}).",
            math_trace=f"cmd = {' '.join(args)}.",
        )

    def push(self, tag: str, registry: str) -> ToolResult:
        if not tag or not registry:
            return ToolResult(status="error", data=None,
                              explanation="tag and registry required",
                              math_trace="", error="ValueError")
        missing = self._require_docker()
        if missing:
            return missing
        full = f"{registry.rstrip('/')}/{tag}"
        tag_cmd = ["docker", "tag", tag, full]
        push_cmd = ["docker", "push", full]
        try:
            self._runner(tag_cmd, check=True, capture_output=True, text=True)
            result = self._runner(push_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error",
                data={"cmd": push_cmd, "returncode": exc.returncode,
                      "stderr": getattr(exc, "stderr", "")},
                explanation=f"docker push failed (rc={exc.returncode}).",
                math_trace="", error=f"CalledProcessError: {exc}",
            )
        return ToolResult(
            status="ok",
            data={"pushed": full, "cmd": push_cmd,
                  "stdout": getattr(result, "stdout", "")},
            explanation=f"Pushed {full}.",
            math_trace=f"docker tag → docker push {full}.",
        )
