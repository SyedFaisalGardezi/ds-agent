"""OS-isolated custom analysis. Docker unavailable => fail closed, never host exec."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd

# Sub-directories of the scratch output mount with a fixed meaning. They are
# never offered for workspace publication: _display holds the plots shown in
# the chat, _state holds tables carried to the next analysis step.
_DISPLAY_DIR = "_display"
_STATE_DIR = "_state"
# Kernel names that are never carried back from a container, even when lower
# case: loaded source data and model objects have their own lifecycle and must
# not be replaced by generated code. UPPER_CASE names are control state and are
# excluded by rule, both here and inside the container.
_RESERVED_STATE = frozenset({"data", "df_eng", "best_model", "model"})
_STATE_BUDGET = 256 * 1024 * 1024
_MAX_FIGURES = 8
_MAX_FIGURE_BYTES = 5 * 1024 * 1024

_BOOTSTRAP = '''
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
# Printed tables are what the model reads; the default width hid most columns
# of a provider summary behind "...", so it could not see its own results.
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
WORK_DIR = Path("/work")
_manifest = json.loads(Path("/inputs/manifest.json").read_text())
_INPUTS = set()
_COLS = {}
for _p in Path("/inputs").glob("*.parquet"):
    _obj = pd.read_parquet(_p)
    if _manifest.get(_p.stem, {}).get("kind") == "series":
        _obj = _obj.iloc[:, 0]
    else:
        _COLS[_p.stem] = [str(_c) for _c in _obj.columns]
    globals()[_p.stem] = _obj
    _INPUTS.add(_p.stem)
globals().update(json.loads(Path("/inputs/context.json").read_text()))
exec(compile(Path("/inputs/code.py").read_text(), "<analysis>", "exec"))
# Loaded data may gain columns, as it would in a notebook, but nothing else:
# a step that also changed, dropped or filtered existing values is not kept.
_EXTENDS = set()
for _n, _before in _COLS.items():
    _now = globals().get(_n)
    if _manifest.get(_n, {}).get("mutable") or not isinstance(_now, pd.DataFrame):
        continue
    _added = [str(_c) for _c in _now.columns if str(_c) not in _before]
    if not _added:
        continue
    try:
        _orig = pd.read_parquet(f"/inputs/{_n}.parquet")
        _same = (len(_now) == len(_orig) and _now.index.equals(_orig.index)
                 and _now[list(_orig.columns)].equals(_orig))
    except Exception:
        _same = False
    if _same:
        _EXTENDS.add(_n)
    else:
        print(f"NOTE: {_n} was changed beyond adding columns ({', '.join(_added[:8])}), "
              f"so none of it is kept; {_n} is the loaded data. Keep derived data "
              f"under a new name, e.g. {_n}_x = {_n}.assign(...).")
_disp = WORK_DIR / "_display"
_disp.mkdir(exist_ok=True)
for _i in plt.get_fignums():
    plt.figure(_i).savefig(_disp / f"figure-{_i}.png", bbox_inches="tight", dpi=100)
# Carry DataFrames and Series forward so the next step can build on this one.
# Source data (non-mutable inputs), private and UPPER_CASE names are skipped.
_state = WORK_DIR / "_state"
_state.mkdir(exist_ok=True)
_kept, _used = {}, 0
for _name, _val in list(globals().items()):
    if _name.startswith("_") or not _name.isidentifier() or _name.isupper():
        continue
    if (_name in _INPUTS and not _manifest.get(_name, {}).get("mutable")
            and _name not in _EXTENDS):
        continue
    if isinstance(_val, pd.Series):
        _frame, _kind = _val.to_frame(name=str(_val.name) if _val.name is not None else "value"), "series"
    elif isinstance(_val, pd.DataFrame):
        _frame, _kind = _val, "frame"
    else:
        continue
    try:
        _frame = _frame.copy()
        _frame.columns = [str(_c) for _c in _frame.columns]
        _path = _state / f"{_name}.parquet"
        _frame.to_parquet(_path)
        _used += _path.stat().st_size
        if _used > ''' + str(_STATE_BUDGET) + ''':
            _path.unlink()
            break
        _kept[_name] = {"kind": _kind, "extends": _name in _EXTENDS}
    except Exception:
        pass
(_state / "manifest.json").write_text(json.dumps(_kept))
'''


def execute(code: str, *, session, kernel, dry_run: bool = False):
    """Run code in the sandbox. dry_run checks that code executes without
    carrying tables back or staging files, e.g. to verify an export."""
    from agent.api.agent_tools import ToolOutput, _stage_change, _ws_resolve_within, _ws_write_mode
    if not code.strip():
        return ToolOutput(False, "", error="No code provided.")
    docker = shutil.which("docker")
    if docker is None:
        return ToolOutput(False, "", error="Isolated execution requires Docker and the ds-agent-sandbox image. "
                          "No host Python code was executed. Use the built-in analysis tools meanwhile.")
    image = os.environ.get("DSAGENT_SANDBOX_IMAGE", "ds-agent-sandbox:local")
    # Probe with `image ls`, not `image inspect`: on Docker Desktop 29 with the
    # containerd image store, inspect intermittently answers "No such image"
    # for an image that `docker run` starts fine. A daemon waking from Resource
    # Saver can also fail the first call, so retry briefly before giving up.
    found = False
    for attempt in range(3):
        probe = subprocess.run([docker, "image", "ls", "-q", image], check=False,
                               capture_output=True, timeout=10)
        found = not probe.returncode and bool(probe.stdout.strip())
        if found:
            break
        if attempt < 2:
            time.sleep(3)
    if not found:
        hint = (f"Build the sandbox image: docker build -f docker/sandbox.Dockerfile -t {image} ."
                if not probe.returncode else
                "Docker is not running; start Docker Desktop.")
        return ToolOutput(False, "", error=f"Isolated execution is unavailable. {hint} "
                          "No host Python code was executed.")
    timeout = int(os.environ.get("DSAGENT_EXEC_TIMEOUT_S", "120"))
    name = "ds-agent-" + uuid.uuid4().hex
    carried = set(getattr(kernel, "sandbox_state", set()) or set())
    with tempfile.TemporaryDirectory(prefix="ds-sandbox-") as tmp:
        root = Path(tmp)
        inputs, outputs = root / "inputs", root / "outputs"
        inputs.mkdir(); outputs.mkdir()
        outputs.chmod(0o777)
        manifest: dict[str, dict] = {}
        unavailable: list[str] = []
        for key, value in list(kernel.namespace.items()):
            if not key.isidentifier() or key.startswith("_"):
                continue
            if dry_run and key in carried:
                continue  # a check must rebuild derived tables from source data
            if isinstance(value, pd.Series):
                frame, kind = value.to_frame(name=str(value.name) if value.name is not None else "value"), "series"
            elif isinstance(value, pd.DataFrame):
                frame, kind = value, "frame"
            else:
                continue
            if _export_frame(frame, inputs / f"{key}.parquet"):
                # Only tables an earlier sandbox step created may be updated by
                # a later one; loaded source data is never replaced.
                manifest[key] = {"kind": kind, "mutable": key in carried}
            else:
                unavailable.append(key)
        (inputs / "manifest.json").write_text(json.dumps(manifest))
        context = {key: kernel.namespace.get(key) for key in ("TARGET", "TASK_SPEC", "BRIEF", "LEAKAGE_COLS")}
        (inputs / "context.json").write_text(json.dumps(context, default=str))
        (inputs / "code.py").write_text(code)
        (inputs / "runner.py").write_text(_BOOTSTRAP)
        cmd = [docker, "run", "--rm", "--name", name, "--network=none", "--read-only",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
               "--memory=4g", "--cpus=2", "--ulimit=fsize=67108864:67108864", "--user=65534:65534",
               "--tmpfs=/tmp:rw,noexec,nosuid,size=256m", "--workdir=/work",
               "--env=MPLCONFIGDIR=/tmp/mpl", "--env=OPENBLAS_NUM_THREADS=1",
               "--mount", f"type=bind,src={inputs},dst=/inputs,readonly",
               "--mount", f"type=bind,src={outputs},dst=/work",
               image, "python", "/inputs/runner.py"]
        # Redirect output to temporary files rather than accumulating it in RAM.
        with (root / "stdout").open("w+b") as stdout, (root / "stderr").open("w+b") as stderr:
            try:
                proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=15, check=False)
                return ToolOutput(False, "", error="Analysis timed out; its container was terminated.")
            stdout.seek(0); stderr.seek(0)
            text = stdout.read(16000).decode(errors="replace")
            error = stderr.read(8000).decode(errors="replace")
        if proc.returncode:
            return ToolOutput(False, text, error=error or "Container execution failed.")
        if dry_run:
            return ToolOutput(True, text, artifacts={"dry_run": True})
        figures = _collect_figures(outputs)
        kept = _intake_state(outputs / _STATE_DIR, kernel, carried)
        proposed = []
        size = 0
        for path in outputs.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.relative_to(outputs).parts[0] in (_DISPLAY_DIR, _STATE_DIR):
                continue  # chat display and step state, never publication
            size += path.stat().st_size
            if size > 64 * 1024 * 1024:
                return ToolOutput(False, text, error="Generated artifacts exceed the 64 MB publication limit.")
            rel = str(path.relative_to(outputs))
            proposed.append(rel)
            if session.work_dir and _ws_write_mode(kernel) == "ask":
                destination = _ws_resolve_within(session.work_dir, rel)
                if destination is None:
                    continue
                rec = _stage_change(kernel, kind="file", rel=rel, path=destination,
                     summary=f"Publish sandbox output {rel}",
                     make_dirs=[str(destination.parent)] if not destination.parent.exists() else [])
                rec["binary"] = base64.b64encode(path.read_bytes()).decode()
        notes = []
        if kept:
            notes.append("Kept for later steps (reuse by name): " + ", ".join(kept) + ".")
        if unavailable:
            notes.append("Not available in this step (could not be serialised): "
                         + ", ".join(unavailable) + ".")
        if figures:
            notes.append(f"{len(figures)} plot(s) shown to the user.")
        if proposed:
            notes.append("Generated in scratch only: " + ", ".join(proposed)
                         + ". Workspace publication requires approval.")
        return ToolOutput(True, text + ("\n" + "\n".join(notes) if notes else ""),
                          figures=figures,
                          artifacts={"scratch_outputs": proposed, "published": False,
                                     "kept_state": kept})


def _export_frame(frame: pd.DataFrame, path: Path) -> bool:
    """Write a table for the container; retry with text columns if types mix."""
    try:
        frame.to_parquet(path)
        return True
    except Exception:
        pass
    try:
        safe = frame.copy()
        safe.columns = [str(c) for c in safe.columns]
        for col in safe.columns:
            if safe[col].dtype == object:
                safe[col] = safe[col].astype(str)
        safe.to_parquet(path)
        return True
    except Exception:
        return False


def _collect_figures(outputs: Path) -> list[str]:
    """Plots the step produced, base64-encoded for the chat, bounded in size.

    Open figures are captured into _display by the bootstrap. A plot the code
    saved with savefig and then closed exists only as a PNG file, so those are
    shown too (they also remain proposals the user can approve to keep).
    """
    display = outputs / _DISPLAY_DIR
    paths = sorted(display.glob("figure-*.png")) if display.is_dir() else []
    paths += sorted(p for p in outputs.rglob("*.png")
                    if p.relative_to(outputs).parts[0] not in (_DISPLAY_DIR, _STATE_DIR))
    figures: list[str] = []
    seen: set[bytes] = set()
    for path in paths:
        if len(figures) >= _MAX_FIGURES:
            break
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_FIGURE_BYTES:
            continue
        data = path.read_bytes()
        if data in seen:
            continue
        seen.add(data)
        figures.append(base64.b64encode(data).decode())
    return figures


def _added_columns(old, new) -> list[str] | None:
    """The columns `new` adds to `old`, or None unless every row and every
    existing value is unchanged. Checked here, not trusted from the container."""
    if not isinstance(old, pd.DataFrame) or not isinstance(new, pd.DataFrame):
        return None
    before = [str(c) for c in old.columns]
    if len(new) != len(old) or not set(before) <= {str(c) for c in new.columns}:
        return None
    try:
        left = new[before].reset_index(drop=True)
        right = old.copy()
        right.columns = before
        pd.testing.assert_frame_equal(left, right.reset_index(drop=True),
                                      check_dtype=False)
    except Exception:  # noqa: BLE001
        return None
    added = [str(c) for c in new.columns if str(c) not in before]
    return added or None


def _intake_state(state: Path, kernel, carried: set[str]) -> list[str]:
    """Load the tables a step produced back into the kernel, validated host-side.

    The container's own filtering is not trusted: every name is re-checked here.
    Only parquet is read (data-only with pyarrow >= 14.0.1), never pickle. A
    name is accepted only if it is a plain lower-case identifier outside the
    reserved set and does not overwrite a variable the sandbox did not create,
    so loaded data, models and control state cannot be replaced from inside it.
    """
    manifest_path = state / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return []
    try:
        entries = json.loads(manifest_path.read_text()[:1_000_000])
    except Exception:
        return []
    if not isinstance(entries, dict):
        return []
    kept: list[str] = []
    total = 0
    for name, meta in entries.items():
        if (not isinstance(name, str) or not name.isidentifier() or name.startswith("_")
                or name.isupper() or name in _RESERVED_STATE):
            continue
        extends = (isinstance(meta, dict) and meta.get("extends") is True
                   and name in kernel.namespace and name not in carried)
        if name in kernel.namespace and name not in carried and not extends:
            continue
        path = state / f"{name}.parquet"
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
        if total > _STATE_BUDGET:
            break
        try:
            value = pd.read_parquet(path)
        except Exception:
            continue
        if isinstance(meta, dict) and meta.get("kind") == "series" and value.shape[1] == 1:
            value = value.iloc[:, 0]
        if extends:
            added = _added_columns(kernel.namespace[name], value)
            if added is None:
                continue   # not a pure extension: the loaded data stays as it was
            kernel.namespace[name] = value
            kept.append(f"{name} (new column(s) kept: {', '.join(added[:8])})")
            continue
        kernel.namespace[name] = value
        carried.add(name)
        shape = "x".join(str(n) for n in value.shape)
        kept.append(f"{name} ({type(value).__name__} {shape})")
    if hasattr(kernel, "__dict__"):
        kernel.sandbox_state = carried
    return kept
