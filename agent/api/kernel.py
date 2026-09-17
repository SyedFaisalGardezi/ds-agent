"""Per-session Python execution kernel.

Each chat session gets one kernel. The kernel holds a persistent namespace so
variables (df, model, X_train, …) survive across messages — exactly like a
running Jupyter kernel.

Execution captures:
  - stdout / stderr
  - matplotlib figures (saved as PNG base64)
  - the repr() of the last expression (like Jupyter's Out[n])
  - any exception with a clean traceback
"""
from __future__ import annotations

import base64
import ctypes
import io
import os
import sys
import textwrap
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Hard wall-clock cap on a single execute() call. Without it, LLM-generated
# `while True:` permanently occupies one of the two job-runner threads and
# the server stops serving chat. Override via DSAGENT_EXEC_TIMEOUT_S.
DEFAULT_EXEC_TIMEOUT_S = int(os.environ.get("DSAGENT_EXEC_TIMEOUT_S", "120"))


@dataclass
class CellOutput:
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    figures: list[str] = field(default_factory=list)   # base64 PNG strings
    last_repr: str = ""
    success: bool = True

    def is_empty(self) -> bool:
        return not any([self.stdout, self.stderr, self.error,
                        self.figures, self.last_repr])

    def as_text(self, max_chars: int = 3000) -> str:
        """Compact text representation for feeding back to the LLM."""
        parts = []
        if self.error:
            parts.append(f"ERROR:\n{self.error}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr[:500]}")
        if self.stdout:
            out = self.stdout
            if len(out) > max_chars:
                out = out[:max_chars] + f"\n… [{len(self.stdout) - max_chars} chars truncated]"
            parts.append(f"OUTPUT:\n{out}")
        if self.last_repr and not self.stdout:
            r = self.last_repr
            if len(r) > 1000:
                r = r[:1000] + "…"
            parts.append(f"RESULT:\n{r}")
        if self.figures:
            parts.append(f"[{len(self.figures)} figure(s) generated]")
        return "\n".join(parts) or "(no output)"


class SessionKernel:
    """Persistent Python namespace for a single chat session."""

    # Serialises execute() across ALL kernels: the stdout/stderr swap below is
    # process-global, so two concurrent job threads (even different sessions)
    # would interleave/steal each other's captured output without this.
    _EXEC_LOCK: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self._exec_count = 0
        self._boot()

    # ── pre-load common DS imports ─────────────────────────────────────

    def _boot(self) -> None:
        boot = textwrap.dedent("""\
            import warnings; warnings.filterwarnings('ignore')
            import pandas as pd
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            sns.set_theme(style='darkgrid')
            df         = None   # primary tabular view — always set by load_data()
            data       = None   # raw loaded object (xarray, ndarray, PIL Image, …)
            DATA_TYPE  = None   # 'tabular' | 'image' | 'audio' | 'scientific' | 'text'
            DATA_FMT   = None   # 'csv' | 'excel' | 'netcdf4' | 'image' | …
            TARGET     = None   # populated by set_context() or by the LLM
            BRIEF      = ''
            WORK_DIR   = None   # set via /workdir endpoint; use to save all outputs
        """)
        self._raw_exec(boot)

    def load_data(self, data_path: Path) -> str:
        """Detect file format and load into the kernel namespace via data_loader."""
        from agent.api.data_loader import describe_load_result, load
        result = load(data_path)
        if result.error and result.df is None:
            return f"Load error: {result.error}"
        # Inject all namespace variables produced by the loader.
        self.namespace.update(result.namespace)
        self.namespace["data"] = result.raw
        self.namespace["DATA_TYPE"] = result.data_type
        self.namespace["DATA_FMT"] = result.format_name
        # Ensure df is always set even when loader returns None.
        if self.namespace.get("df") is None and result.df is not None:
            self.namespace["df"] = result.df
        return describe_load_result(result)

    def set_context(self, target: str | None, brief: str) -> None:
        """Inject session metadata into the kernel namespace."""
        self.namespace["TARGET"] = target
        self.namespace["BRIEF"] = brief

    # ── execution ──────────────────────────────────────────────────────

    def _raw_exec(self, code: str) -> None:
        exec(compile(code, "<kernel>", "exec"), self.namespace)  # noqa: S102

    def execute(self, code: str,
                timeout_s: int | None = None) -> CellOutput:
        """Run `code` in the persistent namespace; capture all output.

        Runs the exec in a watchdog thread. If it exceeds `timeout_s`
        (default DSAGENT_EXEC_TIMEOUT_S = 120), a KeyboardInterrupt is
        injected into the worker via PyThreadState_SetAsyncExc — this
        breaks pure-Python infinite loops (the realistic LLM failure
        mode). Blocking C calls can't be interrupted this way; those
        still leak the worker thread but the caller gets a timeout
        result instead of hanging forever.
        """
        self._exec_count += 1
        deadline = timeout_s if timeout_s is not None else DEFAULT_EXEC_TIMEOUT_S

        with SessionKernel._EXEC_LOCK:
            out = CellOutput()

            # Capture stdout / stderr.
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # Close any open figures before running so we capture only new ones.
            try:
                import matplotlib.pyplot as _plt
                _plt.close("all")
            except Exception:
                pass

            def _work() -> None:
                try:
                    compiled = _compile_with_repr(code)
                    exec(compiled, self.namespace)  # noqa: S102
                    out.last_repr = str(
                        self.namespace.pop("__last_repr__", "") or "")
                except KeyboardInterrupt:
                    out.error = (
                        f"TimeoutError: execution exceeded {deadline}s "
                        "and was interrupted. Avoid unbounded loops; for "
                        "long training jobs raise DSAGENT_EXEC_TIMEOUT_S."
                    )
                    out.success = False
                except Exception:
                    out.error = traceback.format_exc(limit=8)
                    out.success = False

            worker = threading.Thread(target=_work, daemon=True,
                                      name="kernel-exec")
            try:
                worker.start()
                worker.join(deadline)
                if worker.is_alive():
                    # Inject KeyboardInterrupt into the worker thread.
                    _async_raise(worker.ident, KeyboardInterrupt)
                    worker.join(5.0)  # grace period for the except branch
                    if worker.is_alive():
                        # Uninterruptible (blocking C call). Report and move on;
                        # the daemon thread is leaked but the caller is free.
                        out.error = (
                            f"TimeoutError: execution exceeded {deadline}s and "
                            "could not be interrupted (blocking native call). "
                            "The kernel may be in an inconsistent state."
                        )
                        out.success = False
            finally:
                out.stdout = sys.stdout.getvalue()
                out.stderr = sys.stderr.getvalue()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Harvest matplotlib figures.
            try:
                import matplotlib.pyplot as _plt
                for fig_num in _plt.get_fignums():
                    fig = _plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    out.figures.append(base64.b64encode(buf.read()).decode())
                    _plt.close(fig)
            except Exception:
                pass

            return out


# ── helpers ────────────────────────────────────────────────────────────────


def _async_raise(thread_ident: int | None, exc_type: type) -> None:
    """Inject an exception into a running thread (CPython only).

    Standard PyThreadState_SetAsyncExc trick: the exception is raised the
    next time the target thread executes Python bytecode. No effect while
    it is inside a blocking C call.
    """
    if thread_ident is None:
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_ident), ctypes.py_object(exc_type))
    if res > 1:  # pragma: no cover — undo on multi-thread hit per CPython docs
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_ident), None)


def _compile_with_repr(code: str):
    """Compile code so the last expression's repr is captured."""
    import ast
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return compile(code, "<kernel>", "exec")

    if not (tree.body and isinstance(tree.body[-1], ast.Expr)):
        return compile(code, "<kernel>", "exec")

    try:
        # Replace the trailing expression statement with an assignment so it
        # is evaluated exactly ONCE. (Appending `__last_expr__ = (expr)` to the
        # original source would re-execute the expression — double prints,
        # double side effects.)
        last_src = ast.unparse(tree.body[-1].value)
        tree.body[-1] = ast.parse(
            f"__last_repr__ = repr({last_src})").body[0]
        return compile(ast.unparse(tree), "<kernel>", "exec")
    except Exception:
        return compile(code, "<kernel>", "exec")
