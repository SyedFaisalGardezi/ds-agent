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

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
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
log = logging.getLogger("dsagent.chat")

_DATA_EXTS  = {".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".feather"}
_BRIEF_EXTS = {".pdf", ".md", ".txt", ".rst"}


def _nearest_existing_dir(p: Path) -> str:
    """A sibling directory whose name matches `p` apart from case, surrounding
    spaces, or trailing punctuation, so a failed workdir can name the one the
    user meant. A path copied out of a sentence keeps the full stop that ended
    it, which is the mistake this catches. Empty when there is no near match.
    """
    def _key(name: str) -> str:
        return name.strip().rstrip(". ").lower()

    try:
        want = _key(p.name)
        if not want:
            return ""
        for sib in p.parent.iterdir():
            if sib.name != p.name and sib.is_dir() and _key(sib.name) == want:
                return str(sib)
    except OSError:
        pass
    return ""


def _load_secondary(kernel: SessionKernel, path: Path, var_name: str) -> str:
    """Load a secondary data file into the kernel as `var_name` (e.g. df2)."""
    from agent.api.data_loader import load
    result = load(path)
    if result.df is None or result.df.empty:
        return f"Could not load {path.name}: {result.error or 'No data rows found'}"
    kernel.invalidate_analysis()
    kernel.namespace[var_name] = result.df
    return f"Loaded {len(result.df)} rows as {var_name}."


class DataBody(BaseModel):
    path: str = Field(..., description="absolute path to CSV or parquet on the server")
    target: str | None = Field(None, description="target column (optional)")


class WorkDirBody(BaseModel):
    path: str = Field(..., description="absolute path to the session working directory")
    # Defaults to False on purpose. A working directory is one you point at, so
    # a path that does not exist is a typo, and creating it silently hands the
    # session an empty directory that looks like it worked.
    create: bool = Field(False, description="create the directory if it doesn't exist")


class PendingBody(BaseModel):
    # None means every staged change; a list selects specific ones by id.
    ids: list[str] | None = None


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
    active: set[str] = set()
    active_lock = threading.Lock()

    async def authorize(request: Request):
        from agent.api.auth import auth_enabled
        sid = request.path_params.get("sid")
        if not sid:
            yield
            return
        session = store.get(sid)
        if session is None:
            raise HTTPException(404, "Session not found")
        if auth_enabled() and session.owner_id != getattr(request.state, "user_email", None):
            raise HTTPException(404, "Session not found")
        jid = request.path_params.get("jid")
        if jid:
            job = jobs.get(jid)
            if job is None or job.session_id != sid:
                raise HTTPException(404, "Job not found")
        # Keep a mutation lease until the HTTP request completes; background
        # message processing explicitly retains it until the job finishes.
        mutating = request.method in {"POST", "DELETE", "PUT", "PATCH"}
        if mutating:
            with active_lock:
                if sid in active:
                    raise HTTPException(409, "This session is busy; wait for its current operation.")
                active.add(sid)
        try:
            yield
        finally:
            if mutating:
                store.persist_session(session)
            if mutating and not getattr(request.state, "job_started", False):
                with active_lock:
                    active.discard(sid)

    router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(authorize)])

    # -------- per-session kernel store ---------------------------------- #

    _kernels: dict[str, SessionKernel] = {}
    _k_lock = threading.Lock()

    def _get_kernel(sid: str) -> SessionKernel:
        with _k_lock:
            if sid not in _kernels:
                kernel = SessionKernel()
                session = store.require(sid)
                if session.data_path and session.data_path.is_file():
                    kernel.load_data(session.data_path)
                for i, path in enumerate(session.data_paths[1:], 2):
                    if path.is_file():
                        _load_secondary(kernel, path, f"df{i}")
                kernel.set_context(session.target, session.brief)
                kernel.namespace["WORK_DIR"] = session.work_dir
                _kernels[sid] = kernel
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
    def create_session(request: Request) -> dict[str, Any]:
        s = store.create(owner_id=getattr(request.state, "user_email", None))
        _get_kernel(s.session_id)  # warm up kernel eagerly
        s.add("system", "Session ready. Upload a CSV/parquet then ask me anything.")
        return s.to_public()

    @router.get("/sessions")
    def list_sessions(request: Request) -> dict[str, list[str]]:
        from agent.api.auth import auth_enabled
        owner = getattr(request.state, "user_email", None)
        return {"sessions": [sid for sid in store.list_ids()
                if not auth_enabled() or store.require(sid).owner_id == owner]}

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
        store.delete(sid)
        _drop_kernel(sid)
        return {"deleted": sid}

    # -------- data + brief registration --------------------------------- #

    @router.post("/sessions/{sid}/workdir")
    def set_workdir(sid: str, body: WorkDirBody) -> dict[str, Any]:
        s = store.require(sid)
        p = Path(body.path).expanduser().resolve()
        if not p.is_dir():
            if body.create:
                p.mkdir(parents=True, exist_ok=True)
            else:
                hint = _nearest_existing_dir(p)
                raise HTTPException(
                    400, f"not a directory: {p}"
                    + (f". Did you mean {hint}?" if hint else ""))
        s.work_dir = p
        kernel = _get_kernel(sid)
        kernel.namespace["WORK_DIR"] = p


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

        from agent.api.agent_tools import is_agent_generated_doc

        loaded_briefs: list[str] = []
        skipped_own: list[str] = []
        for bf in brief_files:
            text = bf.read_text(errors="replace") if bf.suffix.lower() != ".pdf" else ""
            if bf.suffix.lower() == ".pdf":
                from agent.api.pdf_reader import extract as _pdf_extract
                summary = _pdf_extract(bf.read_bytes())
                text = summary.text
            if is_agent_generated_doc(text):
                skipped_own.append(f"`{bf.name}`")
                continue
            s.add_brief(text, bf.name)
            loaded_briefs.append(f"`{bf.name}`")
        if loaded_briefs:
            kernel.set_context(s.target, s.brief)
            # Clear stale TASK_SPEC so understand_task re-runs with the new brief
            kernel.invalidate_analysis()
            kernel.set_context(s.target, s.brief)

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
            if is_agent_generated_doc(_text):
                skipped_own.append(f"`{_dp.relative_to(p)}`")
                continue
            s.add_brief(_text, _dp.name)
            loaded_briefs.append(f"`{_dp.relative_to(p)}`")
            _sub_loaded += 1
        if _sub_loaded:
            kernel.set_context(s.target, s.brief)
            kernel.invalidate_analysis()
            kernel.set_context(s.target, s.brief)

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
        if skipped_own:
            parts.append("Not used as instructions (plans the agent wrote "
                         "earlier): " + ", ".join(skipped_own) + ".")
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

    # -------- pending disk changes (nothing is written without approval) --- #

    @router.get("/sessions/{sid}/pending")
    def list_pending(sid: str) -> dict[str, Any]:
        store.require(sid)
        from agent.api.agent_tools import pending_changes
        return {"pending": pending_changes(_get_kernel(sid))}

    @router.post("/sessions/{sid}/pending/apply")
    def apply_changes(sid: str, body: PendingBody) -> dict[str, Any]:
        s = store.require(sid)
        from agent.api.agent_tools import apply_pending
        result = apply_pending(_get_kernel(sid), body.ids, session=s)
        if result["applied"]:
            s.add("assistant", "Applied: " + ", ".join(
                r["summary"] for r in result["applied"]))
        return result

    @router.post("/sessions/{sid}/pending/discard")
    def discard_changes(sid: str, body: PendingBody) -> dict[str, Any]:
        store.require(sid)
        from agent.api.agent_tools import discard_pending
        return discard_pending(_get_kernel(sid), body.ids)

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
            if Path(fname).name != fname or "\\" in fname:
                raise HTTPException(400, "Upload filename must not contain paths")
            if Path(fname).suffix.lower() in {".pkl", ".pickle"}:
                raise HTTPException(400, "Executable pickle files are not accepted")
            out = sess_dir / (uuid.uuid4().hex + Path(fname).suffix.lower())
            with out.open("wb") as fh:
                size = 0
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > int(os.environ.get("DSAGENT_UPLOAD_LIMIT_MB", "256")) * 1024 * 1024:
                        raise HTTPException(413, "Upload exceeds size limit")
                    fh.write(chunk)
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
        kernel.invalidate_analysis()
        kernel.set_context(s.target, s.brief)
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
    def message(sid: str, body: MessageBody, request: Request) -> dict[str, Any]:
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

        result = _spawn_agent_job(s, kernel, text, body.multiagent, body.allow_writes)
        request.state.job_started = True
        return result

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
            except Exception as exc:
                # Without this the run vanished. The finally below closes the
                # progress stream, so the UI printed "Done" and moved on, while
                # the conversation kept no record that anything had been asked:
                # the turn ended with the user's message and no answer at all.
                # Say what happened, in the stream and in the conversation, then
                # re-raise so the job is still marked failed.
                detail = f"{type(exc).__name__}: {exc}"
                log.exception("agent job %s failed", jid)
                _progress.emit(jid, f"Run failed: {detail[:300]}",
                               tool="error", level="error")
                reply = (f"The run stopped with an error and could not finish: "
                         f"{detail[:500]}\n\nAnything completed before this "
                         f"point is still in the session. Please try again, or "
                         f"narrow the request.")
                try:
                    s.add("assistant", reply)
                    store.persist_message(s.session_id, "assistant", reply)
                    store.persist_session(s)
                except Exception:  # noqa: BLE001
                    pass            # never let bookkeeping hide the real error
                raise
            finally:
                _progress.close(jid)    # signal SSE stream to close
                with active_lock:
                    active.discard(s.session_id)

        job = jobs.submit("agent", _work, job_id=jid, session_id=s.session_id)
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
            # Outlive the job itself (DSAGENT_JOB_TIMEOUT_S) plus time for its
            # last step: a 10-minute cap closed the stream mid-analysis, and
            # the page then showed an empty result for a run still working.
            from agent.api.jobs import DEFAULT_JOB_TIMEOUT_S
            deadline = DEFAULT_JOB_TIMEOUT_S + 600
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
