from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.core.types import ToolResult

# Maximum on-disk file size accepted for read (bytes).
_MAX_READ_BYTES = 200 * 1024 * 1024
# For tabular reads (csv / parquet), cap the rows returned in ToolResult.data.
# The full file is still read into pandas — only the JSON-serialisable preview is trimmed.
_TABULAR_PREVIEW_ROWS = 10_000
_DEFAULT_ROOT = Path(os.environ.get("DS_AGENT_FILE_ROOT", "outputs")).resolve()
_SUPPORTED_ACTIONS = {"read", "write", "list", "exists"}
_INFERRED_FORMATS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".txt": "txt",
    ".md": "txt",
    ".log": "txt",
}


class FileIOTool:
    """Read / write / list files under a sandboxed root directory.

    Safety model:
      - All paths must resolve (after symlink resolution) inside `root`. Attempts
        to escape via `..` or symlinks raise PathEscapeError.
      - The root defaults to `$DS_AGENT_FILE_ROOT` or `./outputs` and is created
        on first use.
      - Read size is capped at 200 MB; tabular reads return at most 10k rows in
        the ToolResult payload (full file still loaded server-side).
    Supported formats: csv, tsv, parquet, json, jsonl, txt/md/log. Format is
    inferred from extension unless explicitly supplied in the input.
    """

    name: str = "file_io"
    description: str = (
        "Reads, writes, lists, or checks files under a sandboxed output root. "
        "Input: JSON with 'action' (read|write|list|exists), 'path' (relative), "
        "optional 'format' (csv|parquet|json|jsonl|txt), optional 'data' (for write), "
        "optional 'options' dict passed to the underlying reader/writer."
    )
    task_type: str = "default"

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or _DEFAULT_ROOT).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ---- public API ---------------------------------------------------------

    def run(self, input_json: str) -> ToolResult:
        try:
            payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        except json.JSONDecodeError as exc:
            return _err("Input was not valid JSON.", f"JSONDecodeError: {exc}")

        action = (payload.get("action") or "").lower().strip()
        if action not in _SUPPORTED_ACTIONS:
            return _err(
                f"Unsupported action {action!r}. Expected one of: {sorted(_SUPPORTED_ACTIONS)}.",
                "ValueError: action",
            )

        rel_path = payload.get("path")
        if not rel_path:
            return _err("Missing 'path'.", "ValueError: path is required")

        try:
            resolved = self._resolve(rel_path)
        except PathEscapeError as exc:
            return _err(str(exc), "PathEscapeError")

        fmt = (payload.get("format") or _infer_format(resolved)).lower()
        options = payload.get("options") or {}

        try:
            if action == "read":
                return self._read(resolved, fmt, options)
            if action == "write":
                return self._write(resolved, fmt, payload.get("data"), options)
            if action == "list":
                return self._list(resolved, options)
            if action == "exists":
                return ToolResult(
                    status="ok",
                    data={"exists": resolved.exists(), "path": str(resolved.relative_to(self._root))},
                    explanation=f"Checked existence of {resolved.name!r}.",
                    math_trace="",
                )
        except FileNotFoundError as exc:
            return _err(f"File not found: {exc.filename}", "FileNotFoundError")
        except ValueError as exc:
            return _err(str(exc), f"ValueError: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface all backend failures to caller
            return _err(f"{type(exc).__name__}: {exc}", type(exc).__name__)

        # Unreachable given the action whitelist above, but keeps mypy happy.
        return _err("Unhandled action.", "InternalError")

    def to_json(self, result: ToolResult) -> str:
        return json.dumps(asdict(result))

    # ---- internals ----------------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        candidate = (self._root / rel).resolve()
        # Path.is_relative_to exists on 3.9+. Compare on resolved paths so symlinks are followed.
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise PathEscapeError(
                f"Path {rel!r} escapes the sandbox root {self._root}."
            ) from exc
        return candidate

    def _read(self, path: Path, fmt: str, options: dict[str, Any]) -> ToolResult:
        if not path.is_file():
            return _err(f"Not a file: {path.name}", "FileNotFoundError")
        size = path.stat().st_size
        if size > _MAX_READ_BYTES:
            return _err(
                f"File size {size} exceeds {_MAX_READ_BYTES} byte cap.",
                "SizeLimitExceeded",
            )

        if fmt == "csv":
            import pandas as pd
            df = pd.read_csv(path, **options)
            preview = df.head(_TABULAR_PREVIEW_ROWS).to_dict(orient="records")
            truncated = len(df) > _TABULAR_PREVIEW_ROWS
            return ToolResult(
                status="ok",
                data={
                    "records": preview,
                    "shape": list(df.shape),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                    "truncated": truncated,
                    "bytes": size,
                },
                explanation=(
                    f"Read CSV {path.name} ({df.shape[0]} rows × {df.shape[1]} cols, "
                    f"{size} bytes){' — preview truncated' if truncated else ''}."
                ),
                math_trace=_tabular_math_trace(df),
            )

        if fmt == "parquet":
            import pandas as pd
            df = pd.read_parquet(path, **options)
            preview = df.head(_TABULAR_PREVIEW_ROWS).to_dict(orient="records")
            truncated = len(df) > _TABULAR_PREVIEW_ROWS
            return ToolResult(
                status="ok",
                data={
                    "records": preview,
                    "shape": list(df.shape),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                    "truncated": truncated,
                    "bytes": size,
                },
                explanation=(
                    f"Read Parquet {path.name} ({df.shape[0]} rows × {df.shape[1]} cols, "
                    f"{size} bytes){' — preview truncated' if truncated else ''}."
                ),
                math_trace=_tabular_math_trace(df),
            )

        if fmt == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return ToolResult(
                status="ok",
                data={"content": data, "bytes": size},
                explanation=f"Read JSON {path.name} ({size} bytes).",
                math_trace="",
            )

        if fmt == "jsonl":
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return ToolResult(
                status="ok",
                data={"records": lines, "count": len(lines), "bytes": size},
                explanation=f"Read JSONL {path.name} ({len(lines)} records, {size} bytes).",
                math_trace="",
            )

        if fmt == "txt":
            text = path.read_text(encoding="utf-8")
            return ToolResult(
                status="ok",
                data={"text": text, "bytes": size},
                explanation=f"Read text {path.name} ({size} bytes).",
                math_trace="",
            )

        return _err(f"Unsupported format {fmt!r}.", "ValueError: format")

    def _write(
        self,
        path: Path,
        fmt: str,
        data: Any,
        options: dict[str, Any],
    ) -> ToolResult:
        if data is None:
            return _err("Missing 'data' for write.", "ValueError: data is required")

        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt in ("csv", "parquet"):
            import pandas as pd
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                return _err(
                    f"For {fmt}, data must be list[dict] or dict-of-lists, got {type(data).__name__}.",
                    "TypeError: data",
                )
            if fmt == "csv":
                df.to_csv(path, index=options.get("index", False))
            else:
                df.to_parquet(path, index=options.get("index", False))
            size = path.stat().st_size
            return ToolResult(
                status="ok",
                data={"path": str(path.relative_to(self._root)), "bytes": size, "shape": list(df.shape)},
                explanation=f"Wrote {df.shape[0]} rows × {df.shape[1]} cols to {path.name} ({size} bytes).",
                math_trace=_tabular_math_trace(df),
            )

        if fmt == "json":
            path.write_text(json.dumps(data, indent=options.get("indent", 2), default=str), encoding="utf-8")
        elif fmt == "jsonl":
            if not isinstance(data, list):
                return _err("For jsonl, data must be a list of records.", "TypeError: data")
            path.write_text(
                "\n".join(json.dumps(r, default=str) for r in data) + "\n",
                encoding="utf-8",
            )
        elif fmt == "txt":
            if not isinstance(data, str):
                return _err("For txt, data must be a string.", "TypeError: data")
            path.write_text(data, encoding="utf-8")
        else:
            return _err(f"Unsupported format {fmt!r}.", "ValueError: format")

        size = path.stat().st_size
        return ToolResult(
            status="ok",
            data={"path": str(path.relative_to(self._root)), "bytes": size},
            explanation=f"Wrote {fmt} to {path.name} ({size} bytes).",
            math_trace="",
        )

    def _list(self, path: Path, options: dict[str, Any]) -> ToolResult:
        if not path.exists():
            return _err(f"Not found: {path.name}", "FileNotFoundError")
        if not path.is_dir():
            return _err(f"Not a directory: {path.name}", "NotADirectoryError")
        glob = options.get("glob", "*")
        recursive = bool(options.get("recursive", False))
        iter_ = path.rglob(glob) if recursive else path.glob(glob)
        entries = []
        for p in sorted(iter_):
            if p == path:
                continue
            entries.append({
                "name": p.name,
                "path": str(p.relative_to(self._root)),
                "is_dir": p.is_dir(),
                "bytes": p.stat().st_size if p.is_file() else None,
            })
        return ToolResult(
            status="ok",
            data={"entries": entries, "count": len(entries)},
            explanation=f"Listed {len(entries)} entries under {path.name}/ (glob={glob!r}, recursive={recursive}).",
            math_trace="",
        )


# ---- helpers ---------------------------------------------------------------


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the sandbox root."""


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation, math_trace="", error=error)


def _infer_format(path: Path) -> str:
    return _INFERRED_FORMATS.get(path.suffix.lower(), "txt")


def _tabular_math_trace(df: Any) -> str:
    rows, cols = df.shape
    try:
        mem = int(df.memory_usage(deep=True).sum())
    except Exception:  # noqa: BLE001
        mem = -1
    numeric_cols = [c for c, t in df.dtypes.items() if "int" in str(t) or "float" in str(t)]
    lines = [
        f"Shape: {rows} × {cols}. Memory: {mem} bytes.",
        f"Numeric columns: {len(numeric_cols)} / {cols}.",
    ]
    if numeric_cols:
        try:
            desc = df[numeric_cols].describe().to_dict()
            # Keep it compact — LLM-readable summary, not full distribution.
            summary = {
                col: {
                    "mean": float(stats.get("mean", 0)),
                    "std": float(stats.get("std", 0)),
                    "min": float(stats.get("min", 0)),
                    "max": float(stats.get("max", 0)),
                }
                for col, stats in desc.items()
            }
            lines.append("Numeric summary: " + json.dumps(summary))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(lines)


def register() -> FileIOTool:
    """Register the tool in the global ToolRegistry. Safe to call multiple times."""
    from agent.core.tool_registry import ToolRegistry
    tool = FileIOTool()
    ToolRegistry.get()._tools[tool.name] = tool  # type: ignore[attr-defined]
    return tool
