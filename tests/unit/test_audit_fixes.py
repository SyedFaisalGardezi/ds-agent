"""Behavioral regressions from the September audit; no live LLM/API required."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from agent.api.agent_tools import (
    ToolOutput,
    WriteFileTool,
    _is_unsuitable_target,
    apply_pending,
    discard_pending,
    pending_changes,
)
from agent.api.intent import TaskIntent, check_action, interpret
from agent.api.sessions import Session
from agent.pipeline.model_builder.supervised import dataset_hash, estimator, train


@pytest.mark.parametrize("query,decision,tools", [
    ("What is model accuracy?", "answer", []),
    ("Do not train a model", "answer", []),
    ("deploy the model", "act", ["deploy_model"]),
    ("Work out what explains these differences; leave every file untouched", "act", ["execute_code"]),
    ("Prepare a plan, but perform none of its steps", "answer", []),
])
def test_semantic_intent_drives_chat_without_keyword_routing(query, decision, tools, monkeypatch):
    from agent.api import agent_runner as runner
    intent = TaskIntent(decision=decision, objective=query, allowed_tools=tools)
    llm = Mock()
    llm._generate.return_value = intent.model_dump_json()
    llm.answer.return_value = SimpleNamespace(text="Explanation only")
    monkeypatch.setattr(runner, "_classify_tools", Mock(side_effect=AssertionError("Keyword routing called")))
    adaptive = Mock(return_value=("", [], []))
    monkeypatch.setattr(runner, "_run_guarded_adaptive", adaptive)
    kernel = SimpleNamespace(namespace={})
    out = runner.run_reactive(query, Session("s"), kernel, llm)
    assert bool(adaptive.call_count) == (decision == "act")
    assert out.steps == []
    assert kernel.task_intent.allowed_tools == tools


def test_full_request_and_earlier_constraints_reach_interpreter():
    llm = Mock()
    llm._generate.return_value = '{"decision":"answer","objective":"explain"}'
    long_request = "context " * 500 + "Do not train or save anything."
    interpret(long_request, context="brief", history=[{"role": "user", "content": "Only analyse 2023."}],
              tools=[], llm=llm)
    sent = llm._generate.call_args.args[0]
    assert long_request in sent and "Only analyse 2023." in sent


@pytest.mark.parametrize("response", ["bad json", '{"decision":"act","objective":"go"}',
    '{"decision":"act","objective":"go","allowed_tools":["invented"]}'])
def test_invalid_decision_fails_closed(response):
    llm = Mock(); llm._generate.return_value = response
    assert interpret("go", context="", history=[], tools=[], llm=llm).decision == "clarify"


def test_constraints_checked_before_concrete_action():
    intent = TaskIntent(decision="act", objective="inspect", constraints=["no training"], allowed_tools=["execute_code"])
    llm = Mock(); llm._generate.return_value = '{"allowed":false,"reason":"trains a model"}'
    ok, reason = check_action(intent, instruction="inspect without training", brief="", name="execute_code",
                              args={"code": "model.fit(X,y)"}, llm=llm)
    assert not ok and reason == "trains a model"
    assert not check_action(intent, instruction="", brief="", name="train_model", args={}, llm=llm)[0]
    llm._generate.side_effect = TimeoutError()
    assert not check_action(intent, instruction="", brief="", name="execute_code", args={}, llm=llm)[0]


def test_successful_irrelevant_tool_cannot_ground_invented_metric():
    from agent.api.agent_runner import StepResult, _grounded_reply
    step = StepResult(1, "read_file", {}, "", ToolOutput(True, "The README contains setup instructions."), 0)
    llm = Mock()
    llm._generate.return_value = '{"excerpts":[{"id":"E1","quote":"AUC is 0.99"}]}'
    result = _grounded_reply("AUC is 0.99", [step], llm)
    assert "0.99" not in result
    assert "README" in result


def test_grounding_preserves_supported_quote():
    from agent.api.agent_runner import StepResult, _grounded_reply
    step = StepResult(1, "quality_check", {}, "", ToolOutput(True, "Missing values: 12."), 0)
    llm = Mock(); llm._generate.return_value = '{"excerpts":[{"id":"E1","quote":"Missing values: 12."}]}'
    assert "Missing values: 12." in _grounded_reply("", [step], llm)
    assert "No computation" in _grounded_reply("made up", [], llm)


def test_empty_approval_does_not_apply_or_discard(tmp_path):
    session = Session("s", work_dir=tmp_path)
    kernel = SimpleNamespace(namespace={"FILE_WRITE_MODE": "ask"})
    WriteFileTool().execute({"path": "a.txt", "content": "new"}, session=session, kernel=kernel, llm_client=None)
    assert apply_pending(kernel, [], session=session)["applied"] == []
    assert discard_pending(kernel, [])["discarded"] == 0
    assert len(pending_changes(kernel)) == 1
    assert not (tmp_path / "a.txt").exists()
    assert len(apply_pending(kernel, None, session=session)["applied"]) == 1
    assert (tmp_path / "a.txt").read_text() == "new"


def test_stale_approval_does_not_overwrite_user_changes(tmp_path):
    session = Session("s", work_dir=tmp_path)
    kernel = SimpleNamespace(namespace={"FILE_WRITE_MODE": "ask"})
    path = tmp_path / "a.txt"; path.write_text("before")
    WriteFileTool().execute({"path": "a.txt", "content": "proposed", "overwrite": True}, session=session, kernel=kernel, llm_client=None)
    path.write_text("user edit")
    assert apply_pending(kernel, session=session)["failed"]
    assert path.read_text() == "user edit"
    assert len(pending_changes(kernel)) == 1


def test_continuous_regression_target_not_rejected():
    df = pd.DataFrame({"price": np.linspace(1.2, 100.8, 100)})
    assert not _is_unsuitable_target(df, "price")[0]


def test_data_reload_invalidates_derived_state(tmp_path):
    from agent.api.kernel import SessionKernel
    kernel = SessionKernel()
    kernel.namespace.update({"df_eng": pd.DataFrame({"old": [1]}), "best_model": object(),
                             "TASK_SPEC": {"old": True}, "ML_PLAN": {"old": True}})
    path = tmp_path / "data.csv"; path.write_text("new\n2\n3\n")
    kernel.load_data(path)
    assert kernel.namespace["df"].columns.tolist() == ["new"]
    for key in ("df_eng", "best_model", "TASK_SPEC", "ML_PLAN"):
        assert key not in kernel.namespace


def test_pickle_loader_rejects_without_deserializing(tmp_path, monkeypatch):
    from agent.api.data_loader import load
    path = tmp_path / "data.pkl"; path.write_bytes(b"not a pickle")
    monkeypatch.setattr(pd, "read_pickle", Mock(side_effect=AssertionError("unsafe deserialization")))
    assert "not accepted" in load(path).error


def test_sandbox_unavailable_does_not_execute_host_code(tmp_path, monkeypatch):
    from agent.api.sandbox import execute
    monkeypatch.setattr("agent.api.sandbox.shutil.which", lambda name: None)
    target = tmp_path / "must-not-exist"
    out = execute(f"open({str(target)!r}, 'w').write('bad')", session=Session("s"), kernel=SimpleNamespace(namespace={}))
    assert not out.success and not target.exists()


def test_training_pipeline_handles_unseen_categories_and_missing_labels():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=60), "kind": ["a", "b"] * 30,
                       "target": rng.normal(size=60)})
    df.loc[0, "target"] = np.nan
    result = train(df, target="target", task_type="regression", models=["ridge"], log_run=False)
    assert result["n_refit"] == 59 and result["dropped_missing_labels"] == 1
    assert not set(result["train_indices"]) & set(result["test_indices"])
    assert np.isfinite(result["model"].predict(pd.DataFrame({"x": [np.nan], "kind": ["unseen"]}))).all()
    assert result["feature_names"] == ["x", "kind"]


def test_holdout_labels_do_not_influence_cv():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    first = train(df, target="y", task_type="regression", models=["ridge"], log_run=False)
    df.loc[first["test_indices"], "y"] += 10000
    second = train(df, target="y", task_type="regression", models=["ridge"], log_run=False)
    assert first["cv_scores"] == second["cv_scores"]
    assert first["metrics"] != second["metrics"]


def test_group_and_temporal_splits_are_disjoint():
    df = pd.DataFrame({"x": np.arange(100), "entity": np.repeat(np.arange(20), 5),
                       "time": np.arange(100), "y": np.arange(100) * .5})
    grouped = train(df, target="y", task_type="regression", models=["ridge"], group="entity", log_run=False)
    assert not set(df.iloc[grouped["train_indices"]].entity) & set(df.iloc[grouped["test_indices"]].entity)
    temporal = train(df, target="y", task_type="regression", models=["ridge"], time_column="time", log_run=False)
    assert max(temporal["train_indices"]) < min(temporal["test_indices"])


def test_training_fails_when_all_candidates_fail():
    df = pd.DataFrame({"x": range(40), "y": range(40)})
    with pytest.raises(ValueError, match="No model completed"):
        train(df, target="y", task_type="regression", models=["not-a-model"], log_run=False)
    assert dataset_hash(df) != dataset_hash(df.assign(y=df.y + 1))
    assert estimator("rf", True, seed=1, balanced=True).class_weight == "balanced"


def test_auth_never_returns_code_on_delivery_failure(monkeypatch):
    from fastapi import HTTPException

    from agent.api import auth
    monkeypatch.setenv("DSAGENT_SMTP_HOST", "example.invalid")
    monkeypatch.setattr(auth, "_smtp_send", Mock(side_effect=OSError("offline")))
    with pytest.raises(HTTPException) as exc:
        auth.send_code("test@example.invalid", "123456")
    assert exc.value.status_code == 503
    monkeypatch.delenv("DSAGENT_AUTH_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        auth._secret()


def test_persistence_retains_owner_and_workdir_and_deletes(tmp_path):
    from agent.memory.session_store import PersistentSessionStore
    store = PersistentSessionStore(tmp_path / "memory.db")
    session = Session("s", owner_id="owner", work_dir=tmp_path)
    store.create_session("s"); store.save_session(session)
    restored = store.load_session("s")
    assert restored["owner_id"] == "owner" and restored["work_dir"] == str(tmp_path)
    store.delete_session("s")
    assert store.load_session("s") is None


def test_upload_traversal_and_session_ownership(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent.api import auth
    from agent.api.server import create_app
    monkeypatch.setenv("DSAGENT_AUTH_ENABLED", "1")
    monkeypatch.setenv("DSAGENT_AUTH_SECRET", "test-only-secret-" * 3)
    monkeypatch.setenv("DSAGENT_SMTP_HOST", "example.invalid")
    monkeypatch.setattr(auth, "USERS", {email: {"email": email} for email in ("a@example.invalid", "b@example.invalid")})
    with TestClient(create_app(leaderboard_db=tmp_path / "lb.db")) as client:
        a = {"Authorization": "Bearer " + auth.issue_token("a@example.invalid")}
        b = {"Authorization": "Bearer " + auth.issue_token("b@example.invalid")}
        sid = client.post("/chat/sessions", headers=a).json()["session_id"]
        assert client.get(f"/chat/sessions/{sid}", headers=b).status_code == 404
        assert sid not in client.get("/chat/sessions", headers=b).json()["sessions"]
        assert client.post(f"/chat/sessions/{sid}/pending/apply", headers=b, json={}).status_code == 404
        assert client.post(f"/chat/sessions/{sid}/upload", headers=a,
                           files={"files": ("../escape.csv", b"x\n1", "text/csv")}).status_code == 400
        assert client.post(f"/chat/sessions/{sid}/upload", headers=a,
                           files={"files": ("bad.pkl", b"not pickle", "application/octet-stream")}).status_code == 400


def test_session_mutations_are_serialized(tmp_path, monkeypatch):
    import threading

    from fastapi.testclient import TestClient

    from agent.api import agent_runner
    from agent.api.server import create_app
    entered, finish = threading.Event(), threading.Event()
    def work(*args, **kwargs):
        entered.set()
        finish.wait(5)
        return agent_runner.AgentRunResult(goal="test", reply="finished")
    monkeypatch.setenv("DSAGENT_AUTH_ENABLED", "0")
    app = create_app(leaderboard_db=tmp_path / "lb.db")
    monkeypatch.setattr(app.state.llm, "is_available", lambda: True)
    monkeypatch.setattr(agent_runner, "run_reactive", work)
    with TestClient(app) as client:
        sid = client.post("/chat/sessions").json()["session_id"]
        try:
            job = client.post(f"/chat/sessions/{sid}/message", json={"text": "compute"}).json()
            assert entered.wait(2)
            assert client.post(f"/chat/sessions/{sid}/message", json={"text": "again"}).status_code == 409
            assert client.post(f"/chat/sessions/{sid}/pending/apply", json={}).status_code == 409
            other = client.post("/chat/sessions").json()["session_id"]
            assert client.get(f"/chat/sessions/{other}/jobs/{job['job_id']}").status_code == 404
        finally:
            finish.set()


def test_sandbox_publishes_only_through_approval(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from agent.api import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")
    seen = []
    def run(cmd, **kwargs):
        seen.append(cmd)
        if "run" in cmd:
            assert "--network=none" in cmd and "--read-only" in cmd
            mount = next(a for a in cmd if "dst=/work" in a)
            path = Path(mount.split("src=", 1)[1].split(",dst=")[0])
            (path / "result.txt").write_text("computed")
            kwargs["stdout"].write(b"computed 3 rows")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"img\n")
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    kernel = SimpleNamespace(namespace={"df": pd.DataFrame({"x": [1, 2, 3]}), "FILE_WRITE_MODE": "ask"})
    session = Session("s", work_dir=tmp_path)
    out = sandbox.execute("print(len(df))", session=session, kernel=kernel)
    assert out.success and "3 rows" in out.text
    assert not (tmp_path / "result.txt").exists()
    assert len(pending_changes(kernel)) == 1
    assert apply_pending(kernel, session=session)["applied"]
    assert (tmp_path / "result.txt").read_text() == "computed"


def test_sandbox_timeout_terminates_container(monkeypatch):
    import subprocess

    from agent.api import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if "run" in cmd:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"img\n")
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    out = sandbox.execute("while True: pass", session=Session("s"), kernel=SimpleNamespace(namespace={}))
    assert not out.success and any("rm" in c for c in calls)


def test_dvc_cli_stages_preserve_raw_labels(tmp_path, monkeypatch):
    import yaml

    from agent.pipeline.cli import run_stage
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"x": range(20), "y": [None] + list(range(19))}).to_csv("source.csv", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"source_path": "source.csv", "target": "y",
                                     "task_type": "regression", "models": ["ridge"]}))
    run_stage("extract", str(config)); run_stage("transform", str(config))
    assert pd.read_parquet("outputs/datasets/clean.parquet").y.isna().sum() == 1
    with pytest.raises(ValueError):
        run_stage("unknown", str(config))


def test_deep_eda_templates_build_before_kernel_execution():
    from agent.api.agent_tools import DeepEdaTool
    from agent.api.kernel import CellOutput
    kernel = SimpleNamespace(execute=Mock(return_value=CellOutput(stdout="checked")), namespace={})
    DeepEdaTool._stage05_findings(kernel, "target", "regression")
    code = kernel.execute.call_args.args[0]
    compile(code, "<generated>", "exec")
    assert "_findings" in code


def test_exported_service_preserves_raw_record_predictions(tmp_path, monkeypatch):
    import importlib.util
    import sys

    import joblib
    from fastapi.testclient import TestClient

    from agent.pipeline.deploy_packager.fastapi_builder import FastAPIBuilder
    frame = pd.DataFrame({'x': range(30), 'category': ['a', 'b'] * 15,
                          'target': np.arange(30) * 1.5})
    result = train(frame, target='target', task_type='regression', models=['ridge'], log_run=False)
    model_path = tmp_path / 'model.joblib'
    joblib.dump(result['model'], model_path)
    FastAPIBuilder().build(str(model_path), str(tmp_path))
    monkeypatch.setenv('MODEL_URI', str(model_path))
    spec = importlib.util.spec_from_file_location('exported_service', tmp_path / 'app.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    records = [{'category': 'unseen', 'x': None}, {'category': 'a', 'x': 4}]
    expected = result['model'].predict(pd.DataFrame(records)).tolist()
    with TestClient(module.app) as client:
        response = client.post('/predict', json={'rows': records})
        assert response.status_code == 200
        assert response.json()['predictions'] == pytest.approx(expected)
        assert client.post('/predict', json={'rows': [{'x': 4}]}).status_code == 422
    assert 'COPY model.joblib' in (tmp_path / 'Dockerfile').read_text()


# ── sandbox step continuity and plots (review follow-up) ─────────────────────

def _fake_sandbox_run(monkeypatch, write_outputs):
    """Mock docker so a 'run' call lets the test write container outputs."""
    import subprocess
    from pathlib import Path

    from agent.api import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")

    def run(cmd, **kwargs):
        if "run" in cmd:
            mount = next(a for a in cmd if "dst=/work" in a)
            write_outputs(Path(mount.split("src=", 1)[1].split(",dst=")[0]))
            kwargs["stdout"].write(b"ok")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"img\n", stderr=b"")
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    return sandbox


def test_sandbox_returns_plots_inline_and_never_stages_them(tmp_path, monkeypatch):
    png = b"\x89PNG\r\n\x1a\nfake"

    def outputs(work):
        (work / "_display").mkdir()
        (work / "_display" / "figure-1.png").write_bytes(png)
    sandbox = _fake_sandbox_run(monkeypatch, outputs)
    kernel = SimpleNamespace(namespace={"df": pd.DataFrame({"x": [1]}), "FILE_WRITE_MODE": "ask"})
    out = sandbox.execute("plot", session=Session("s", work_dir=tmp_path), kernel=kernel)
    assert out.success and len(out.figures) == 1
    assert pending_changes(kernel) == []          # a chat plot is not a deliverable


def test_sandbox_carries_new_tables_forward_but_protects_existing(tmp_path, monkeypatch):
    def outputs(work):
        state = work / "_state"
        state.mkdir()
        pd.DataFrame({"a": [1, 2]}).to_parquet(state / "summary.parquet")
        pd.DataFrame({"a": [9]}).to_parquet(state / "df.parquet")          # source data
        pd.DataFrame({"a": [9]}).to_parquet(state / "TARGET.parquet")      # control state
        pd.DataFrame({"a": [9]}).to_parquet(state / "best_model.parquet")  # reserved
        (state / "manifest.json").write_text(json.dumps(
            {n: {"kind": "frame"} for n in ("summary", "df", "TARGET", "best_model")}))
    sandbox = _fake_sandbox_run(monkeypatch, outputs)
    df = pd.DataFrame({"x": [1, 2, 3]})
    kernel = SimpleNamespace(namespace={"df": df}, sandbox_state=set())
    out = sandbox.execute("x", session=Session("s"), kernel=kernel)
    assert out.success
    assert list(kernel.namespace["summary"]["a"]) == [1, 2]
    assert kernel.namespace["df"] is df                  # loaded data not replaced
    assert "TARGET" not in kernel.namespace and "best_model" not in kernel.namespace
    assert kernel.sandbox_state == {"summary"}
    assert "summary" in out.text                          # the model is told what it can reuse


def test_sandbox_state_is_cleared_when_inputs_change():
    from agent.api.kernel import SessionKernel
    k = SessionKernel()
    k.namespace["summary"] = pd.DataFrame({"a": [1]})
    k.sandbox_state = {"summary"}
    k.invalidate_analysis()
    assert "summary" not in k.namespace and k.sandbox_state == set()


def test_sandbox_reports_docker_not_running_without_raising(monkeypatch):
    import subprocess

    from agent.api import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")
    monkeypatch.setattr(sandbox.time, "sleep", lambda s: None)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stderr=b"Cannot connect to the Docker daemon. Is the docker daemon running?"))
    out = sandbox.execute("print(1)", session=Session("s"), kernel=SimpleNamespace(namespace={}))
    assert not out.success and "start Docker Desktop" in out.error


def test_sandbox_missing_image_is_not_reported_as_docker_down(monkeypatch):
    import subprocess

    from agent.api import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")
    monkeypatch.setattr(sandbox.time, "sleep", lambda s: None)
    # `image ls -q` succeeds with empty output when the image is absent.
    monkeypatch.setattr(sandbox.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=b"", stderr=b""))
    out = sandbox.execute("print(1)", session=Session("s"), kernel=SimpleNamespace(namespace={}))
    assert "Build the sandbox image" in out.error and "not running" not in out.error


def test_sandbox_probe_retries_while_docker_wakes(monkeypatch):
    import subprocess

    from agent.api import sandbox
    probes = iter([b"", b"446b5b6e293a\n"])   # nothing while the VM resumes, found on retry
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/fake/docker")
    monkeypatch.setattr(sandbox.time, "sleep", lambda s: None)

    def run(cmd, **kw):
        if cmd[1:3] == ["image", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=next(probes), stderr=b"")
        raise RuntimeError("reached docker run")
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="reached docker run"):
        sandbox.execute("print(1)", session=Session("s"), kernel=SimpleNamespace(namespace={}))


# ── evidence-checked interpretation and scoped permission checks ─────────────

def _step(ok, text="", error="", artifacts=None, tool="execute_code"):
    from agent.api.agent_runner import StepResult
    return StepResult(1, tool, {}, "", ToolOutput(ok, text, error=error, artifacts=artifacts or {}), 0)


def test_grounded_reply_keeps_supported_interpretation_and_drops_invented_figures():
    from agent.api.agent_runner import _grounded_reply
    ev = _step(True, "PRV901 7733 4.211353\nPRV902 765 0.875512")
    draft = ("PRV901 filed 7,733 claims, 4.2 SD above the mean, which suggests overservicing.\n"
             "The model reached an AUC of 0.97.")
    out = _grounded_reply(draft, [ev], None)
    assert "overservicing" in out and "7,733" in out     # interpretation survives
    assert "0.97" not in out and "Removed 1 statement" in out


def test_grounded_reply_summarises_failures_instead_of_showing_tracebacks():
    from agent.api.agent_runner import _grounded_reply
    out = _grounded_reply("Nothing numeric here, just a note.", [
        _step(True, "rows 10"),
        _step(False, error="Traceback (most recent call last):\n  File x\nTypeError: bad operand")], None)
    assert "Traceback" not in out and "TypeError: bad operand" in out


def test_grounded_reply_flags_staged_files_as_not_saved():
    from agent.api.agent_runner import _grounded_reply
    ev = _step(True, "Generated in scratch only: ranks.csv.", artifacts={"scratch_outputs": ["ranks.csv"]})
    out = _grounded_reply("I wrote the ranking to ranks.csv.", [ev], None)
    assert "NOT yet in your folder" in out and "ranks.csv" in out


def test_read_only_tools_skip_llm_check_but_keep_the_allowed_tools_boundary(monkeypatch):
    from agent.api import agent_runner as AR
    intent = TaskIntent(decision="act", objective="explore", allowed_tools=["execute_code"])
    kernel = SimpleNamespace(task_intent=intent, task_instruction="explore", namespace={})
    llm = Mock()
    llm._generate.side_effect = AssertionError("no LLM permission call for a read-only tool")
    tool = Mock(); tool.input_schema = {"type": "object"}
    tool.execute.return_value = ToolOutput(True, "done")
    monkeypatch.setitem(AR.TOOL_REGISTRY, "execute_code", tool)
    assert AR._execute_tool("execute_code", {}, session=Session("s"), kernel=kernel, llm_client=llm).success
    blocked = AR._execute_tool("train_model", {}, session=Session("s"), kernel=kernel, llm_client=llm)
    assert not blocked.success and "outside the interpreted request" in blocked.error


def test_side_effecting_tools_still_get_the_semantic_check(monkeypatch):
    from agent.api import agent_runner as AR
    assert AR._needs_semantic_check("train_model") and AR._needs_semantic_check("write_file")
    assert not AR._needs_semantic_check("execute_code")
    monkeypatch.setenv("DSAGENT_CHECK_ALL_ACTIONS", "1")
    assert AR._needs_semantic_check("execute_code")


def test_grounded_reply_drops_tables_left_without_rows():
    from agent.api.agent_runner import _grounded_reply
    ev = _step(True, "PRV008 0.071")
    draft = ("Summary line.\n| Provider | Score |\n|---|---|\n| PRV009 | 0.999 |\n\n"
             "| Provider | Risk |\n|---|---|\n| PRV008 | 0.071 |")
    out = _grounded_reply(draft, [ev], None)
    assert "| Provider | Score |" not in out      # header of an emptied table is gone
    assert "| PRV008 | 0.071 |" in out            # supported table kept intact


def test_sandbox_shows_plots_saved_to_files_and_still_proposes_them(tmp_path, monkeypatch):
    png = b"\x89PNG\r\n\x1a\nsaved-plot"

    def outputs(work):
        (work / "ratio_boxplot.png").write_bytes(png)   # savefig + close: no open figure
    sandbox = _fake_sandbox_run(monkeypatch, outputs)
    kernel = SimpleNamespace(namespace={"df": pd.DataFrame({"x": [1]}), "FILE_WRITE_MODE": "ask"})
    out = sandbox.execute("plot", session=Session("s", work_dir=tmp_path), kernel=kernel)
    assert len(out.figures) == 1                          # shown in the chat
    assert len(pending_changes(kernel)) == 1              # and still offered for saving


def test_agent_written_plans_are_recognised_and_marked():
    from agent.api.agent_tools import AGENT_PLAN_MARKER, is_agent_generated_doc
    legacy = ("# Analysis plan\n\n_First pass, written before EDA. The feature and model "
              "choices are assumptions to be tested._\n")
    assert is_agent_generated_doc(legacy)
    assert is_agent_generated_doc(AGENT_PLAN_MARKER + "\n# Analysis plan\n")
    assert not is_agent_generated_doc("Exercise: find provider billing anomalies.")


def _planning_kernel(plan_first, plan=None):
    intent = TaskIntent(decision="act", objective="EDA", allowed_tools=["work_plan"],
                        plan_first=plan_first)
    ns = {"df": pd.DataFrame({"x": [1, 2]})}
    if plan is not None:
        ns["WORK_PLAN"] = plan
    return SimpleNamespace(task_intent=intent, namespace=ns)


def test_plan_is_written_before_exploring_when_the_model_asks_for_one(monkeypatch):
    from agent.api import agent_runner as AR
    tool = Mock()

    def _plan(args, *, session, kernel, llm_client):
        kernel.namespace["WORK_PLAN"] = {"objective": "Rank providers",
                                         "eda_questions_task": ["q1", "q2"]}
        return ToolOutput(True, "Wrote the work plan")
    tool.execute.side_effect = _plan
    monkeypatch.setitem(AR.TOOL_REGISTRY, "work_plan", tool)
    events = []
    steps = AR._ensure_work_planned(Session("s"), _planning_kernel(True), None,
                                    lambda m, **k: events.append(m))
    assert [s.tool for s in steps] == ["work_plan"] and steps[0].output.success
    assert any("Plan written" in e and "2 EDA questions" in e for e in events)


@pytest.mark.parametrize("plan_first,plan", [(False, None), (True, {"objective": "x"})])
def test_no_plan_step_for_single_questions_or_when_a_plan_exists(plan_first, plan, monkeypatch):
    from agent.api import agent_runner as AR
    tool = Mock(); tool.execute.side_effect = AssertionError("should not plan")
    monkeypatch.setitem(AR.TOOL_REGISTRY, "work_plan", tool)
    assert AR._ensure_work_planned(Session("s"), _planning_kernel(plan_first, plan), None, None) == []


def test_failed_plan_is_reported_and_exploration_continues(monkeypatch):
    from agent.api import agent_runner as AR
    tool = Mock(); tool.execute.return_value = ToolOutput(False, "", error="model timed out")
    monkeypatch.setitem(AR.TOOL_REGISTRY, "work_plan", tool)
    events = []
    steps = AR._ensure_work_planned(Session("s"), _planning_kernel(True), None,
                                    lambda m, **k: events.append((m, k.get("level"))))
    assert not steps[0].output.success
    assert any("Could not write a plan" in m and lvl == "warn" for m, lvl in events)


def test_unlabelled_task_hides_target_only_tools_and_says_so(monkeypatch):
    from agent.api import agent_runner as AR
    kernel = SimpleNamespace(task_intent=None, remaining_tool_calls=5,
                             namespace={"TASK_SPEC": {"task_type": "anomaly_detection",
                                                      "target_column": ""}})
    llm = Mock()
    llm.chat_with_tools.return_value = {"content": "done", "tool_calls": []}
    AR._tool_call_loop("explore", Session("s"), kernel, llm)
    offered = {t["function"]["name"] for t in llm.chat_with_tools.call_args.args[1]}
    assert offered and not offered & AR._TARGET_ONLY_TOOLS
    assert "NO TARGET" in llm.chat_with_tools.call_args.kwargs["system"]


def test_target_tools_stay_available_when_a_target_exists():
    from agent.api import agent_runner as AR
    labelled = SimpleNamespace(namespace={"TASK_SPEC": {"target_column": "churn"}})
    unread = SimpleNamespace(namespace={})
    assert not AR._task_has_no_target(Session("s"), labelled)
    assert not AR._task_has_no_target(Session("s"), unread)   # brief not read yet


# ── EDA agenda: plan questions tracked open/answered, findings recorded ──────

def test_agenda_merges_revised_plans_without_losing_answers():
    from agent.api.agent_tools import add_lead, record_findings, sync_agenda
    ns = {}
    sync_agenda(ns, {"eda_questions_task": ["Which providers bill most?"],
                     "eda_questions_independent": ["Any duplicate claims?"]})
    assert [a["id"] for a in ns["EDA_AGENDA"]] == ["Q1", "Q2"]
    assert record_findings(ns, ["q1"], "PRV901 holds 47% of spend") == ["Q1"]
    sync_agenda(ns, {"eda_questions_task": ["Which providers bill most?", "Paid vs allowed?"]})
    assert [(a["id"], a["status"]) for a in ns["EDA_AGENDA"]] == [
        ("Q1", "answered"), ("Q2", "open"), ("Q3", "open")]
    assert add_lead(ns, "Why does PRV901 bill 6 lines per visit?") == "Q4"
    assert add_lead(ns, "Why does PRV901 bill 6 lines per visit?") is None   # no duplicates
    assert record_findings(ns, ["Q2"], "") == []                               # no empty findings


def test_leads_are_capped_so_the_plan_keeps_priority():
    from agent.api.agent_tools import add_lead
    ns = {}
    added = [add_lead(ns, f"lead {i}") for i in range(10)]
    assert sum(a is not None for a in added) == 6


def test_agenda_is_shown_to_the_model_with_findings():
    from agent.api.agent_runner import _data_context
    from agent.api.agent_tools import record_findings, sync_agenda
    ns = {"WORK_PLAN": {"version": 1}}
    sync_agenda(ns, {"eda_questions_task": ["Which providers bill most?", "Paid vs allowed?"]})
    record_findings(ns, ["Q1"], "PRV901 holds 47% of spend")
    ctx = _data_context(Session("s"), SimpleNamespace(namespace=ns))
    assert "EDA AGENDA (1 of 2 answered)" in ctx
    assert "[Q2] (brief) Paid vs allowed?" in ctx and "47% of spend" in ctx


def test_critic_reports_settled_items_and_new_leads():
    from agent.api.agent_runner import _critique_step
    llm = Mock()
    llm._generate.return_value = json.dumps({
        "verdict": "accept", "goal_met": False, "finding": "PRV901 bills 47% of spend",
        "answers": ["Q1"], "new_question": "Why is PRV901 so large?", "next_step": "x",
        "reason": "ok"})
    crit = _critique_step("explore", "execute_code", {}, "table", True, llm_client=llm,
                          repeats=1, agenda=[{"id": "Q1", "question": "Top billers?",
                                              "status": "open"}])
    assert crit.answers == ["Q1"] and crit.new_question.startswith("Why is PRV901")
    assert "[Q1] Top billers?" in llm._generate.call_args.args[0]


def test_loop_records_critic_findings_on_the_agenda(monkeypatch):
    from agent.api import agent_runner as AR
    from agent.api.agent_tools import sync_agenda
    ns = {"df": pd.DataFrame({"x": [1]})}
    sync_agenda(ns, {"eda_questions_task": ["Top billers?"]})
    kernel = SimpleNamespace(task_intent=None, remaining_tool_calls=5, namespace=ns)
    tool = Mock(); tool.input_schema = {"type": "object"}; tool.description = "run code"
    tool.execute.return_value = ToolOutput(True, "PRV901 47%")
    monkeypatch.setitem(AR.TOOL_REGISTRY, "execute_code", tool)
    monkeypatch.setattr(AR, "_CRITIC_ENABLED", True)
    monkeypatch.setattr(AR, "_critique_step", lambda *a, **k: AR.Critique(
        "accept", "", True, finding="PRV901 holds 47% of spend", answers=["Q1"],
        new_question="Is PRV901 billing many lines per visit?"))
    llm = Mock()
    llm.chat_with_tools.return_value = {"content": "", "tool_calls": [
        {"function": {"name": "execute_code", "arguments": {"code": "print(1)"}}}]}
    llm._generate.return_value = "summary"
    AR._tool_call_loop("explore", Session("s"), kernel, llm, max_steps=1)
    assert ns["EDA_AGENDA"][0]["status"] == "answered"
    assert ns["EDA_AGENDA"][1]["source"] == "lead"


def test_reply_carries_the_stored_plan_and_agenda_coverage():
    from agent.api.agent_runner import StepResult, _with_plan_and_agenda
    from agent.api.agent_tools import record_findings, sync_agenda
    ns = {"WORK_PLAN": {"version": 2, "objective": "Find risky providers"}}
    sync_agenda(ns, {"eda_questions_task": ["Top billers?", "Paid vs allowed?"]})
    record_findings(ns, ["Q1"], "PRV901")
    steps = [StepResult(1, "work_plan", {}, "", ToolOutput(True, "plan"), 0),
             StepResult(2, "execute_code", {}, "", ToolOutput(True, "PRV901"), 0)]
    out = _with_plan_and_agenda("Findings here.", steps, SimpleNamespace(namespace=ns))
    assert out.startswith("## Work plan (version 2)") and "Find risky providers" in out
    assert "## Results\n\nFindings here." in out
    assert "EDA agenda: 1 of 2 questions answered. Still open: [Q2] Paid vs allowed?" in out
    plain = _with_plan_and_agenda("Answer.", steps[1:], SimpleNamespace(namespace=ns))
    assert not plain.startswith("## Work plan")          # no plan written this turn


def test_new_data_clears_the_agenda(tmp_path):
    from agent.api.kernel import SessionKernel
    kernel = SessionKernel()
    kernel.namespace["EDA_AGENDA"] = [{"id": "Q1"}]
    path = tmp_path / "d.csv"; path.write_text("a\n1\n")
    kernel.load_data(path)
    assert "EDA_AGENDA" not in kernel.namespace


def test_planner_sees_the_brief_in_its_own_terms():
    from agent.api.agent_tools import _brief_context
    text = _brief_context({"brief": {"objective": "Find risky payments",
                                     "audiences": ["fraud investigator"],
                                     "data_notes": ["paid_amount: amount paid"]}})
    assert "Objective: Find risky payments" in text and "- fraud investigator" in text
    assert "paid_amount: amount paid" in text
    assert _brief_context({}) == "(not available)"


# ── Export the session's analysis as .ipynb / .py ─────────────────────────────

def _export_session(tmp_path):
    from agent.api.agent_tools import record_findings, sync_agenda
    (tmp_path / "claims.csv").write_text("provider_id,paid\nP1,1\n")
    session = Session("s", work_dir=tmp_path)
    session.data_paths = [tmp_path / "claims.csv"]
    ns = {"FILE_WRITE_MODE": "ask", "df": pd.DataFrame({"x": [1]}),
          "TASK_SPEC": {"brief": {"objective": "Find risky providers",
                                  "questions": ["Which providers present a risk?"]}}}
    sync_agenda(ns, {"eda_questions_task": ["Top billers?", "Paid vs allowed?"]})
    record_findings(ns, ["Q1"], "PRV901 holds 47% of spend")
    kernel = SimpleNamespace(namespace=ns, analysis_cells=[
        {"code": "prov = df.groupby('provider_id').paid.sum()", "output": "", "figures": 0,
         "finding": "PRV901 holds 47% of spend", "answers": ["Q1"]},
        {"code": "print(prov.head())", "output": "", "figures": 0, "finding": "", "answers": []}])
    return session, kernel


def test_successful_code_steps_are_recorded_for_export(monkeypatch):
    from agent.api import sandbox
    from agent.api.agent_tools import ExecuteCodeTool
    kernel = SimpleNamespace(namespace={})
    monkeypatch.setattr(sandbox, "execute", lambda code, **k: ToolOutput(True, "ok")
                        if "good" in code else ToolOutput(False, "", error="boom"))
    ExecuteCodeTool().execute({"code": "good()"}, session=Session("s"), kernel=kernel, llm_client=None)
    ExecuteCodeTool().execute({"code": "bad()"}, session=Session("s"), kernel=kernel, llm_client=None)
    assert [c["code"] for c in kernel.analysis_cells] == ["good()"]


def test_export_builds_a_notebook_from_the_sessions_analysis(tmp_path, monkeypatch):
    from agent.api import sandbox
    from agent.api.agent_tools import BuildNotebookTool
    seen = {}
    monkeypatch.setattr(sandbox, "execute", lambda code, **k: seen.update(code=code, **k)
                        or ToolOutput(True, ""))
    session, kernel = _export_session(tmp_path)
    out = BuildNotebookTool().execute({}, session=session, kernel=kernel, llm_client=None)
    assert out.success and "Re-run check passed" in out.text
    assert seen["dry_run"] is True and "groupby" in seen["code"]
    change = pending_changes(kernel)[0]
    assert change["rel"] == "analysis.ipynb" and not (tmp_path / "analysis.ipynb").exists()
    nb = json.loads(apply_pending(kernel, session=session) and
                    (tmp_path / "analysis.ipynb").read_text())
    text = [c["source"] for c in nb["cells"]]
    assert any("Which providers present a risk?" in t for t in text)
    assert any("df = pd.read_csv('claims.csv')" in t for t in text)      # relative data path
    assert any("Answers [Q1]: Top billers?" in t and "47% of spend" in t for t in text)
    assert any("Not yet answered" in t and "[Q2]" in t for t in text)
    assert [c["cell_type"] for c in nb["cells"]].count("code") == 3       # setup + 2 steps


def test_export_as_script_and_flags_a_failed_check(tmp_path, monkeypatch):
    from agent.api import sandbox
    from agent.api.agent_tools import BuildNotebookTool
    monkeypatch.setattr(sandbox, "execute", lambda code, **k: ToolOutput(
        False, "", error="Traceback\nNameError: name 'prov' is not defined"))
    session, kernel = _export_session(tmp_path)
    out = BuildNotebookTool().execute({"format": "py"}, session=session, kernel=kernel,
                                      llm_client=None)
    assert "Re-run check failed: NameError" in out.text
    apply_pending(kernel, session=session)
    script = (tmp_path / "analysis.py").read_text()
    assert script.startswith("# %% [markdown]\n# # Analysis")
    assert "# %%\nprov = df.groupby('provider_id').paid.sum()" in script
    assert "Check failed" in script
    compile(script, "analysis.py", "exec")                                 # valid Python


def test_export_without_analysis_says_what_to_do_first(tmp_path):
    from agent.api.agent_tools import BuildNotebookTool
    out = BuildNotebookTool().execute({}, session=Session("s", work_dir=tmp_path),
                                      kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert not out.success and "Run the analysis" in out.error


def test_export_check_rebuilds_derived_tables_from_source_only(tmp_path, monkeypatch):
    shipped = []

    def outputs(work):
        pass
    sandbox = _fake_sandbox_run(monkeypatch, outputs)
    real_export = sandbox._export_frame
    monkeypatch.setattr(sandbox, "_export_frame",
                        lambda frame, path: shipped.append(path.stem) or real_export(frame, path))
    kernel = SimpleNamespace(namespace={"df": pd.DataFrame({"x": [1]}),
                                        "prov": pd.DataFrame({"y": [2]})},
                             sandbox_state={"prov"})
    sandbox.execute("print(1)", session=Session("s"), kernel=kernel, dry_run=True)
    assert shipped == ["df"]


# ── write_sql: SQL from the loaded data, tested locally ───────────────────────

def _sql_session(tmp_path):
    session = Session("s", work_dir=tmp_path)
    session.data_paths = [tmp_path / "Medical Claims.csv"]
    df = pd.DataFrame({"provider_id": ["P1", "P1", "P2"], "paid_amount": [10.0, 30.0, 5.0]})
    kernel = SimpleNamespace(namespace={"df": df, "FILE_WRITE_MODE": "ask",
                                        "TASK_SPEC": {"brief": {"data_notes": [
                                            "paid_amount: amount paid (NZD)"]}}})
    return session, kernel


def test_write_sql_uses_real_names_and_tests_on_local_data(tmp_path):
    from agent.api.agent_tools import WriteSqlTool
    session, kernel = _sql_session(tmp_path)
    llm = Mock()
    llm._generate.return_value = ("```sql\nSELECT provider_id, SUM(paid_amount) AS paid\n"
                                  "FROM medical_claims GROUP BY provider_id ORDER BY paid DESC;\n```")
    out = WriteSqlTool().execute({"question": "total paid per provider", "save_to": "paid"},
                                 session=session, kernel=kernel, llm_client=llm)
    prompt = llm._generate.call_args.args[0]
    assert "medical_claims(provider_id VARCHAR, paid_amount DOUBLE)" in prompt
    assert "amount paid (NZD)" in prompt and "snowflake SQL" in prompt
    assert out.success and out.artifacts["tested"]
    assert "P1" in out.text and "40.0" in out.text                     # real result rows
    assert pending_changes(kernel)[0]["rel"] == "paid.sql"             # save awaits approval


def test_write_sql_reports_dialect_syntax_as_untested(tmp_path):
    from agent.api.agent_tools import WriteSqlTool
    session, kernel = _sql_session(tmp_path)
    llm = Mock()
    llm._generate.return_value = ("SELECT provider_id FROM medical_claims "
                                  "QUALIFY ROW_NUMBER() OVER (ORDER BY paid_amount) = 1")
    out = WriteSqlTool().execute({"question": "cheapest"}, session=session, kernel=kernel,
                                 llm_client=llm)
    assert out.success and not out.artifacts["tested"]
    assert "Not tested locally" in out.text and "passed" not in out.text


def test_write_sql_refuses_statements_that_change_data(tmp_path):
    from agent.api.agent_tools import WriteSqlTool
    session, kernel = _sql_session(tmp_path)
    llm = Mock(); llm._generate.return_value = "DELETE FROM medical_claims"
    out = WriteSqlTool().execute({"question": "x"}, session=session, kernel=kernel, llm_client=llm)
    assert not out.success and "modifies data" in out.error


def test_model_supplied_secrets_never_reach_a_connector(monkeypatch):
    from agent.api import agent_tools as AT
    seen = {}

    class Conn:
        def __init__(self, **cfg):
            seen.update(cfg)

        def connect(self):
            return SimpleNamespace(status="error", explanation="stop")
    import sys
    import types
    fake = types.ModuleType("fake_conn_mod"); fake.Conn = Conn
    monkeypatch.setitem(sys.modules, "fake_conn_mod", fake)
    monkeypatch.setitem(AT._CONNECTOR_CLASSES, "snowflake", ("fake_conn_mod", "Conn"))
    AT.SqlQueryTool().execute({"source": "snowflake", "sql": "select 1", "config": {
        "account": "acme", "password": "hunter2", "private_key_path": "/k"}},
        session=Session("s"), kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert seen == {"account": "acme"}


def test_revise_nudge_waits_until_the_agenda_is_worked_through():
    from agent.api.agent_runner import _data_context
    from agent.api.agent_tools import record_findings, sync_agenda
    ns = {"WORK_PLAN": {"version": 1, "revised_after_eda": False}}
    sync_agenda(ns, {"eda_questions_task": ["Top billers?", "Paid vs allowed?"]})
    kernel = SimpleNamespace(namespace=ns)
    assert "work_plan with revise=true" not in _data_context(Session("s"), kernel)
    record_findings(ns, ["Q1", "Q2"], "answered")
    assert "work_plan with revise=true" in _data_context(Session("s"), kernel)


@pytest.mark.parametrize("question,repeat", [
    ("Is provider_id consistently mapped to provider_name?", True),
    ("Are any service_date values in the future relative to today?", True),
    ("Which providers have the highest mean or median difference?", True),
    ("Which providers have the highest claim counts?", False),
    ("Which providers account for the largest share of total paid?", False),
    ("Is the mapping between provider_id and speciality one-to-one?", False),
])
def test_revised_plans_do_not_re_add_the_same_question(question, repeat):
    """Pairs from a real run, where a revision re-added every check reworded."""
    from agent.api.agent_tools import sync_agenda
    ns = {}
    sync_agenda(ns, {"eda_questions_task": [
        "Is there a consistent mapping between provider_id and provider_name across all rows?",
        "Which providers or specialities have the highest average difference between billed "
        "and allowed amounts?",
        "Are service_date values all in the past and not future dates?"]})
    sync_agenda(ns, {"eda_questions_task": [question]})
    assert (len(ns["EDA_AGENDA"]) == 3) == repeat


def test_plan_is_not_rewritten_twice_in_one_turn():
    from agent.api.agent_tools import WorkPlanTool
    ns = {"df": pd.DataFrame({"x": [1, 2]}),
          "WORK_PLAN": {"version": 1, "objective": "Find risky providers", "turn": 123.0}}
    kernel = SimpleNamespace(namespace=ns, turn_deadline=123.0)
    llm = Mock(); llm._generate.side_effect = AssertionError("no second plan this turn")
    out = WorkPlanTool().execute({}, session=Session("s"), kernel=kernel, llm_client=llm)
    assert out.success and out.artifacts["reused"] and "already written" in out.text
    kernel.turn_deadline = 456.0                     # a new request may re-plan
    llm._generate.side_effect = None
    llm._generate.return_value = json.dumps({"objective": "New objective"})
    WorkPlanTool().execute({}, session=Session("s"), kernel=kernel, llm_client=llm)
    assert ns["WORK_PLAN"]["version"] == 2 and ns["WORK_PLAN"]["turn"] == 456.0


def test_new_columns_on_loaded_data_are_kept_but_nothing_else(tmp_path):
    """Checked host-side: the container's claim that it only added columns is
    not trusted."""
    import json as _json

    from agent.api.sandbox import _added_columns, _intake_state
    old = pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    assert _added_columns(old, old.assign(r=[0.1, 0.2, 0.3])) == ["r"]
    assert _added_columns(old, old.assign(r=1).iloc[:2]) is None           # rows dropped
    assert _added_columns(old, old.assign(r=1, a=[9, 2, 3])) is None      # value changed
    assert _added_columns(old, old) is None                                # nothing added

    state = tmp_path / "_state"; state.mkdir()
    old.assign(r=1, a=[9, 2, 3]).to_parquet(state / "df.parquet")         # forged extension
    (state / "manifest.json").write_text(_json.dumps({"df": {"kind": "frame", "extends": True}}))
    kernel = SimpleNamespace(namespace={"df": old})
    assert _intake_state(state, kernel, set()) == []
    assert kernel.namespace["df"] is old

    old.assign(r=1).to_parquet(state / "df.parquet")                       # genuine extension
    kept = _intake_state(state, kernel, set())
    assert kept == ["df (new column(s) kept: r)"]
    assert list(kernel.namespace["df"].columns) == ["a", "s", "r"]
    assert "df" not in kernel.sandbox_state                                 # still the loaded data


def test_dead_backend_is_reported_not_disguised_as_an_unclear_request():
    llm = Mock()
    llm._generate.side_effect = RuntimeError("Server error '500 Internal Server Error'")
    intent = interpret("Explore the data", context="", history=[], tools=[], llm=llm)
    assert intent.decision == "clarify"
    assert "could not be reached" in intent.clarification and "500" in intent.clarification
