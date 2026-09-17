"""Chat-driven session router for ds-agent.

Every user message goes to a persistent per-session Python kernel via the
LLM code agent. The agent writes and executes Python code iteratively,
exactly like Claude Code, and can trigger the full ML pipeline notebook
by calling `build_notebook_report()` from within a code block.

Endpoints (mounted at /chat):
    POST   /sessions                       start a session
    GET    /sessions                       list session ids
    GET    /sessions/{sid}                 state + recent messages
    DELETE /sessions/{sid}                 drop a session
    POST   /sessions/{sid}/data            register a server-side path
    POST   /sessions/{sid}/upload          multipart CSV/parquet upload
    POST   /sessions/{sid}/brief           multipart PDF brief
    POST   /sessions/{sid}/workdir         set working directory
    POST   /sessions/{sid}/message         chat turn → background job
    GET    /sessions/{sid}/jobs/{jid}      poll background job
    GET    /sessions/{sid}/notebook        download generated .ipynb
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.api import agent_runner, orchestrator
from agent.api.agent_runner import result_to_dict
from agent.api.jobs import JobRunner
from agent.api.kernel import SessionKernel
from agent.api.llm import OllamaClient
from agent.api.pdf_reader import extract as pdf_extract
from agent.api.sessions import Session, SessionStore
from agent.pipeline.evaluator.leaderboard import Leaderboard

_UPLOAD_DIR = Path("outputs/uploads")

# Extensions auto-discovered when scanning a working directory
_DATA_EXTS  = {".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".feather"}
_BRIEF_EXTS = {".pdf", ".md", ".txt", ".rst"}


def _load_secondary(kernel: SessionKernel, path: Path, var_name: str) -> str:
    """Load a secondary data file into the kernel as `var_name` (e.g. df2)."""
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            code = (
                f"import pandas as pd\n"
                f"{var_name} = pd.read_csv(r'{path}')\n"
                f"print(f'{var_name}: {{len({var_name})}} rows × {{len({var_name}.columns)}} cols')"
            )
        elif ext == ".parquet":
            code = (
                f"import pandas as pd\n"
                f"{var_name} = pd.read_parquet(r'{path}')\n"
                f"print(f'{var_name}: {{len({var_name})}} rows × {{len({var_name}.columns)}} cols')"
            )
        elif ext in (".xlsx", ".xls"):
            code = (
                f"import pandas as pd\n"
                f"{var_name} = pd.read_excel(r'{path}')\n"
                f"print(f'{var_name}: {{len({var_name})}} rows × {{len({var_name}.columns)}} cols')"
            )
        elif ext == ".tsv":
            code = (
                f"import pandas as pd\n"
                f"{var_name} = pd.read_csv(r'{path}', sep='\\t')\n"
                f"print(f'{var_name}: {{len({var_name})}} rows × {{len({var_name}.columns)}} cols')"
            )
        else:
            return f"Unsupported format `{ext}` for secondary file."
        result = kernel.execute(code)
        if result.error:
            return f"Could not load `{path.name}`: {result.error}"
        return result.stdout.strip() or f"Loaded as `{var_name}`."
    except Exception as exc:
        return f"Could not load `{path.name}`: {exc}"


class DataBody(BaseModel):
    path: str = Field(..., description="absolute path to CSV or parquet on the server")
    target: str | None = Field(None, description="target column (optional)")


class WorkDirBody(BaseModel):
    path: str = Field(..., description="absolute path to the session working directory")
    create: bool = Field(True, description="create the directory if it doesn't exist")


class MessageBody(BaseModel):
    text: str
    # Per-message override of the multi-agent orchestrator (web-UI toggle).
    # None → use the server default (DSAGENT_MULTIAGENT).
    multiagent: bool | None = None
    # Per-message file-write mode (web-UI toggle): True → apply edits/writes,
    # False → suggest-only (diffs). None → server default (DSAGENT_ALLOW_WRITES).
    allow_writes: bool | None = None


def make_router(*, store: SessionStore, leaderboard: Leaderboard,
                jobs: JobRunner, llm: OllamaClient,
                ltm: Any = None, schema_cache: Any = None) -> APIRouter:
    router = APIRouter(prefix="/chat", tags=["chat"])

    # -------- per-session kernel store ---------------------------------- #

    _kernels: dict[str, SessionKernel] = {}
    _k_lock = threading.Lock()

    def _get_kernel(sid: str) -> SessionKernel:
        with _k_lock:
            if sid not in _kernels:
                _kernels[sid] = SessionKernel()
            return _kernels[sid]

    def _drop_kernel(sid: str) -> None:
        with _k_lock:
            _kernels.pop(sid, None)

    def _inject_helpers(kernel: SessionKernel, s: Session) -> None:
        """Inject session-aware helpers into the kernel so the LLM can call them."""
        def build_notebook_report() -> str:
            result = orchestrator.run_full_pipeline(
                s, leaderboard=leaderboard,
                llm_client=llm if llm.is_available() else None,
            )
            if result.get("notebook"):
                s.notebook_path = Path(result["notebook"])
            return result.get("reply", "Pipeline complete.")

        kernel.namespace["build_notebook_report"] = build_notebook_report
        # Always keep WORK_DIR in sync with session
        if s.work_dir:
            kernel.namespace["WORK_DIR"] = s.work_dir

    # -------- session lifecycle ----------------------------------------- #

    @router.post("/sessions")
    def create_session() -> dict[str, Any]:
        s = store.create()
        _get_kernel(s.session_id)  # warm up kernel eagerly
        s.add("system", "Session ready. Upload a CSV/parquet then ask me anything.")
        return s.to_public()

    @router.get("/sessions")
    def list_sessions() -> dict[str, list[str]]:
        return {"sessions": store.list_ids()}

    @router.get("/sessions/{sid}")
    def get_session(sid: str) -> dict[str, Any]:
        s = store.get(sid)
        if s is None:
            raise HTTPException(404, f"session {sid!r} not found")
        return {
            **s.to_public(),
            "messages": [{"role": m.role, "text": m.text, "ts": m.ts}
                         for m in s.messages[-50:]],
        }

    @router.delete("/sessions/{sid}")
    def delete_session(sid: str) -> dict[str, str]:
        s = store.get(sid)
        if s is None:
            raise HTTPException(404, f"session {sid!r} not found")
        store._sessions.pop(sid, None)  # noqa: SLF001
        _drop_kernel(sid)
        return {"deleted": sid}

    # -------- data + brief registration --------------------------------- #

    @router.post("/sessions/{sid}/workdir")
    def set_workdir(sid: str, body: WorkDirBody) -> dict[str, Any]:
        s = store.require(sid)
        p = Path(body.path).expanduser().resolve()
        if body.create:
            p.mkdir(parents=True, exist_ok=True)
        elif not p.is_dir():
            raise HTTPException(400, f"not a directory: {p}")
        s.work_dir = p
        kernel = _get_kernel(sid)
        kernel.namespace["WORK_DIR"] = p
        kernel.execute(f"WORK_DIR = __import__('pathlib').Path(r'{p}')")

        # Auto-discover data and brief files in the directory
        data_files = sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _DATA_EXTS
        )
        brief_files = sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _BRIEF_EXTS
        )

        loaded_data: list[str] = []
        for i, df_path in enumerate(data_files):
            if i == 0 and not s.data_paths:
                s.add_data_path(df_path)
                load_msg = kernel.load_data(df_path)
                loaded_data.append(f"`{df_path.name}` → `df` ({load_msg.split(chr(10))[0]})")
            else:
                var_name = f"df{len(s.data_paths) + 1}"
                s.add_data_path(df_path)
                msg = _load_secondary(kernel, df_path, var_name)
                loaded_data.append(f"`{df_path.name}` → `{var_name}` ({msg})")
        if len(s.data_paths) > 1:
            kernel.execute(
                f"SECONDARY_DATA_PATHS = {[str(q) for q in s.data_paths[1:]]!r}"
            )

        loaded_briefs: list[str] = []
        for bf in brief_files:
            text = bf.read_text(errors="replace") if bf.suffix.lower() != ".pdf" else ""
            if bf.suffix.lower() == ".pdf":
                from agent.api.pdf_reader import extract as _pdf_extract
                summary = _pdf_extract(bf.read_bytes())
                text = summary.text
            s.add_brief(text, bf.name)
            loaded_briefs.append(f"`{bf.name}`")
        if loaded_briefs:
            kernel.set_context(s.target, s.brief)
            # Clear stale TASK_SPEC so understand_task re-runs with the new brief
            kernel.namespace.pop("TASK_SPEC", None)

        # Workspace awareness: map the whole tree (recursive, ignore-aware) so
        # the agent knows every file — not just the top-level data/brief it
        # auto-loaded — and can read or load the rest on demand.
        workspace_map: dict[str, Any] = {}
        try:
            from agent.api.agent_tools import scan_workspace
            workspace_map = scan_workspace(p)
            kernel.namespace["WORKSPACE_MAP"] = workspace_map
        except Exception:  # noqa: BLE001
            workspace_map = {}

        # Also load instruction documents that live in sub-folders, not only the
        # top level. The workspace scan already found them (ignore-aware); load a
        # bounded number of brief-type docs the top-level pass did not pick up.
        _already = {bf.resolve() for bf in brief_files}
        _sub_loaded = 0
        for _doc in workspace_map.get("by_category", {}).get("doc", []):
            if _sub_loaded >= 6:
                break
            _dp = Path(_doc)
            if _dp.suffix.lower() not in _BRIEF_EXTS or _dp.resolve() in _already:
                continue
            try:
                if _dp.suffix.lower() == ".pdf":
                    from agent.api.pdf_reader import extract as _pdf_extract
                    _text = _pdf_extract(_dp.read_bytes()).text
                else:
                    _text = _dp.read_text(errors="replace")
            except Exception:  # noqa: BLE001
                continue
            s.add_brief(_text, _dp.name)
            loaded_briefs.append(f"`{_dp.relative_to(p)}`")
            _sub_loaded += 1
        if _sub_loaded:
            kernel.set_context(s.target, s.brief)
            kernel.namespace.pop("TASK_SPEC", None)

        parts = [f"Working directory set to `{p}`."]
        if workspace_map.get("n_files"):
            counts = ", ".join(f"{c} {n}"
                               for c, n in workspace_map.get("counts", {}).items())
            parts.append(
                f"Mapped {workspace_map['n_files']} files across "
                f"{workspace_map['n_dirs']} sub-folders"
                + (f" ({counts})." if counts else ".")
                + " Ask me to 'explore the directory' for the full map, or to "
                  "read any specific file.")
        if loaded_data:
            parts.append("Data files loaded: " + ", ".join(loaded_data))
        if loaded_briefs:
            parts.append("Brief files loaded: " + ", ".join(loaded_briefs)
                         + " — say **'read the instructions'** to extract the task spec.")
        if not loaded_data and not loaded_briefs and not workspace_map.get("n_files"):
            parts.append(
                "No data or brief files found. "
                "Upload files or place them in this directory, then refresh."
            )
        reply = "\n".join(parts)
        s.add("assistant", reply)
        return {**s.to_public(), "loaded_data": loaded_data,
                "loaded_briefs": loaded_briefs,
                "workspace": {"n_files": workspace_map.get("n_files", 0),
                              "n_dirs": workspace_map.get("n_dirs", 0),
                              "counts": workspace_map.get("counts", {})}}

    @router.post("/sessions/{sid}/data")
    def register_data(sid: str, body: DataBody) -> dict[str, Any]:
        s = store.require(sid)
        p = Path(body.path).expanduser().resolve()
        if not p.exists():
            raise HTTPException(400, f"path not found: {p}")
        s.data_path = p
        if body.target:
            s.target = body.target
        kernel = _get_kernel(sid)
        load_msg = kernel.load_data(p)
        kernel.set_context(s.target, s.brief)
        s.add("user", f"registered data: {p}"
              + (f" (target={body.target})" if body.target else ""))
        s.add("assistant",
              f"Data loaded: `{p}`.\n{load_msg}\n"
              + (f"Target: `{body.target}`." if body.target
                 else "Ask me anything about the data, or say 'build notebook'."))
        return s.to_public()

    @router.post("/sessions/{sid}/upload")
    async def upload_data(
        sid: str, files: list[UploadFile] = File(...)  # noqa: B008
    ) -> dict[str, Any]:
        s = store.require(sid)
        sess_dir = _UPLOAD_DIR / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        kernel = _get_kernel(sid)
        summaries: list[str] = []
        for file in files:
            fname = file.filename or "upload.bin"
            out = sess_dir / fname
            with out.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            resolved = out.resolve()
            s.add_data_path(resolved)
            idx = s.data_paths.index(resolved)  # 0-based position
            if idx == 0:
                # Primary data file → load as `df`
                load_msg = kernel.load_data(resolved)
                kernel.set_context(s.target, s.brief)
                if schema_cache is not None:
                    try:
                        _df_obj = kernel.namespace.get("df")
                        if _df_obj is not None:
                            schema_cache.save(resolved, _df_obj)
                    except Exception:
                        pass
                summaries.insert(0, f"Primary: `{fname}` ({out.stat().st_size:,} bytes).\n{load_msg}")
            else:
                # Secondary files → load as df2, df3, … so tools can use them
                var_name = f"df{idx + 1}"
                _load_secondary(kernel, resolved, var_name)
                summaries.append(
                    f"Additional: `{fname}` ({out.stat().st_size:,} bytes) → loaded as `{var_name}`."
                )
        # Keep SECONDARY_DATA_PATHS in sync for legacy tool access
        if len(s.data_paths) > 1:
            kernel.execute(
                f"SECONDARY_DATA_PATHS = {[str(p) for p in s.data_paths[1:]]!r}"
            )
        combined = "\n".join(summaries)
        s.add("user", f"uploaded {len(files)} data file(s): {', '.join(f.filename or '?' for f in files)}")
        s.add("assistant", combined + "\nAsk me anything about the data, or say 'build notebook'.")
        return {**s.to_public(), "data_path": str(s.data_path),
                "load_summary": combined}

    @router.post("/sessions/{sid}/brief")
    async def upload_brief(
        sid: str, files: list[UploadFile] = File(...)  # noqa: B008
    ) -> dict[str, Any]:
        _ALLOWED_BRIEF_EXTS = {".pdf", ".md", ".txt", ".rst"}
        s = store.require(sid)
        kernel = _get_kernel(sid)
        combined_task_hint: str = ""
        combined_target: str | None = None
        all_data_dict_lines: list[str] = []
        total_chars = 0
        for file in files:
            fname = file.filename or "brief"
            ext = Path(fname).suffix.lower()
            if ext not in _ALLOWED_BRIEF_EXTS:
                raise HTTPException(
                    400,
                    f"Unsupported brief format `{ext}`. Accepted: {sorted(_ALLOWED_BRIEF_EXTS)}"
                )
            data = await file.read()
            if ext == ".pdf":
                summary = pdf_extract(data)
                text = summary.text
                if summary.task_hint and not combined_task_hint:
                    combined_task_hint = summary.task_hint
                if summary.mentioned_target and not combined_target:
                    combined_target = summary.mentioned_target
                all_data_dict_lines.extend(summary.data_dict_lines)
            else:
                text = data.decode("utf-8", errors="replace")
            s.add_brief(text, fname)
            total_chars += len(text)
        if combined_target and not s.target:
            s.target = combined_target
        kernel.set_context(s.target, s.brief)
        # Clear stale task spec — new brief means we must re-extract
        kernel.namespace.pop("TASK_SPEC", None)
        fnames = ", ".join(f.filename or "?" for f in files)
        s.add("user", f"uploaded brief(s): {fnames}")
        reply = f"Read {len(files)} brief file(s) ({total_chars:,} chars total): {fnames}."
        if combined_task_hint:
            reply += f" Task: **{combined_task_hint}**."
        if combined_target:
            reply += f" Suggested target: `{combined_target}`."
        s.add("assistant", reply)
        return {**s.to_public(),
                "task_hint": combined_task_hint,
                "mentioned_target": combined_target,
                "brief_chars": total_chars,
                "data_dict_lines": all_data_dict_lines[:25]}

    # -------- chat ------------------------------------------------------ #

    @router.post("/sessions/{sid}/message")
    def message(sid: str, body: MessageBody) -> dict[str, Any]:
        s = store.require(sid)
        kernel = _get_kernel(sid)
        text = body.text.strip()
        s.add("user", text)
        store.persist_message(s.session_id, "user", text)

        if not llm.is_available():
            reply = ("LLM (qwen3.8:27b-mlx via Ollama) is offline. "
                     "Start Ollama (`ollama serve`) and retry.")
            s.add("assistant", reply)
            return {"reply": reply, "session": s.to_public()}

        return _spawn_agent_job(s, kernel, text, body.multiagent,
                                body.allow_writes)

    # -------- background jobs ------------------------------------------- #

    def _spawn_agent_job(s: Session, kernel: SessionKernel,
                          text: str,
                          multiagent: bool | None = None,
                          allow_writes: bool | None = None) -> dict[str, Any]:
        """Route all user messages through the full agentic runner."""
        from agent.api import progress as _progress

        _inject_helpers(kernel, s)
        jid = uuid.uuid4().hex[:10]
        _progress.create(jid)           # open stream before job starts

        def _work() -> dict[str, Any]:
            try:
                result = agent_runner.run_reactive(
                    text, session=s, kernel=kernel,
                    llm_client=llm, job_id=jid, ltm=ltm,
                    multiagent=multiagent, allow_writes=allow_writes)
                reply = result.reply or "(no reply)"
                if not s.messages or s.messages[-1].text != reply:
                    s.add("assistant", reply)
                store.persist_message(s.session_id, "assistant", reply)
                store.persist_session(s)
                d = result_to_dict(result)
                d["notebook"] = str(s.notebook_path) if s.notebook_path else None
                return d
            finally:
                _progress.close(jid)    # signal SSE stream to close

        job = jobs.submit("agent", _work, job_id=jid)
        return {
            "job_id": job.job_id,
            "kind": job.kind,
            "poll": f"/chat/sessions/{s.session_id}/jobs/{job.job_id}",
            "stream": f"/chat/sessions/{s.session_id}/jobs/{job.job_id}/stream",
            "session": s.to_public(),
        }

    @router.get("/sessions/{sid}/jobs/{jid}")
    def job_status(sid: str, jid: str) -> dict[str, Any]:
        store.require(sid)
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, f"job {jid!r} not found")
        return job.to_public()

    @router.get("/sessions/{sid}/jobs/{jid}/stream")
    async def stream_job(sid: str, jid: str) -> StreamingResponse:
        """SSE endpoint — push progress events to the browser in real-time."""
        import asyncio
        import queue as _queue

        from agent.api import progress as _progress

        store.require(sid)

        async def _generate():
            import json as _json
            yield f"data: {_json.dumps({'type': 'connected', 'job_id': jid})}\n\n"
            deadline = 600          # max 10 min per stream
            t0 = __import__("time").time()
            while __import__("time").time() - t0 < deadline:
                q = _progress.get_queue(jid)
                if q is None:
                    # Queue already closed (job finished before client connected)
                    job = jobs.get(jid)
                    state = job.state if job else "unknown"
                    yield f"data: {_json.dumps({'type': 'done', 'state': state})}\n\n"
                    break
                try:
                    msg = q.get_nowait()
                    if msg is None:     # sentinel — job finished
                        job = jobs.get(jid)
                        state = job.state if job else "done"
                        yield f"data: {_json.dumps({'type': 'done', 'state': state})}\n\n"
                        break
                    yield f"data: {_json.dumps(msg)}\n\n"
                except _queue.Empty:
                    # Keepalive comment every 0.4s
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.4)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # -------- notebook download ----------------------------------------- #

    @router.get("/sessions/{sid}/notebook")
    def get_notebook(sid: str) -> FileResponse:
        s = store.require(sid)
        if s.notebook_path is None or not s.notebook_path.exists():
            raise HTTPException(
                404, "notebook not built yet — ask the agent to 'build notebook'")
        return FileResponse(
            s.notebook_path,
            media_type="application/x-ipynb+json",
            filename=s.notebook_path.name,
        )

    return router
