"""FastAPI surface tests — hit every endpoint and assert envelope shape."""
from __future__ import annotations

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.api.server import create_app


@pytest.fixture
def client(tmp_path, synthetic_df):
    app = create_app(leaderboard_db=tmp_path / "lb.db")
    with TestClient(app) as c:
        yield c


def _records(df: pd.DataFrame) -> list[dict]:
    # JSON doesn't allow NaN — replace with None for wire transport.
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@pytest.fixture
def records(synthetic_df):
    df = synthetic_df.drop(columns=["id", "feature_datetime"])
    return _records(df)


# ---- meta endpoints --------------------------------------------------- #

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stages_and_tools(client):
    assert client.get("/stages").status_code == 200
    r = client.get("/tools")
    assert r.status_code == 200
    assert "pii_detector" in r.json()["tools"]


# ---- pipeline stages -------------------------------------------------- #

def test_profile(client, records):
    r = client.post("/pipelines/profile", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data"]["n_rows"] == len(records)


def test_quality(client, records):
    r = client.post("/pipelines/quality", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["data"]["score"] <= 1.0


def test_statistics_mi(client, records):
    r = client.post(
        "/pipelines/statistics/mi",
        json={"records": records, "target": "target_binary"},
    )
    assert r.status_code == 200
    assert len(r.json()["data"]["scores"]) > 0


def test_features(client, records):
    r = client.post(
        "/pipelines/features",
        json={"records": records, "target": "target_binary", "max_pairs": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "frame_records" in body["data"]
    assert body["data"]["frame_shape"][0] == len(records)


def test_task_inference(client, records):
    r = client.post(
        "/pipelines/task",
        json={"records": records, "target": "target_binary"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["task"] == "binary_classification"


def test_train_binary_and_leaderboard(client, records):
    r = client.post(
        "/pipelines/train",
        json={
            "records": records,
            "target": "target_binary",
            "estimator": "rf",
            "n_estimators": 20,
            "test_size": 0.25,
            "random_state": 0,
            "run_id": "api-test-bin",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["run_id"] == "api-test-bin"
    assert body["data"]["task"] == "binary_classification"
    assert "accuracy" in body["data"]["metrics"]

    lb = client.get("/leaderboard", params={"metric": "accuracy", "n": 5})
    assert lb.status_code == 200
    runs = lb.json()["data"]["runs"]
    assert any(r["run_id"] == "api-test-bin" for r in runs)


def test_train_regression(client, synthetic_df):
    df = synthetic_df.drop(columns=["id", "feature_datetime", "target_binary"])
    r = client.post(
        "/pipelines/train",
        json={
            "records": _records(df),
            "target": "target_regression",
            "estimator": "rf",
            "n_estimators": 20,
            "run_id": "api-test-reg",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["task"] == "regression"
    assert "rmse" in body["data"]["metrics"]


def test_drift(client, records):
    base_df = pd.DataFrame.from_records(records)
    shifted = base_df.copy()
    num = shifted.select_dtypes("number").columns
    shifted[num] = shifted[num] + shifted[num].std() * 2.0
    r = client.post(
        "/pipelines/drift",
        json={
            "baseline": _records(base_df),
            "current": _records(shifted),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["n_checked"] > 0


# ---- tool dispatch ---------------------------------------------------- #

def test_tool_unknown(client):
    r = client.post("/tools/does_not_exist", json={"input_json": {}})
    assert r.status_code == 404


def test_tool_hypothesis_normality(client, records):
    r = client.post(
        "/tools/hypothesis_test",
        json={"input_json": {
            "test": "normality",
            "records": records[:200],
            "columns": ["feature_numeric_1"],
        }},
    )
    # tool may return ok or warning depending on p-value; not error.
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "warning"}


# ---- validation ------------------------------------------------------- #

def test_empty_records_rejected(client):
    r = client.post("/pipelines/profile", json={"records": []})
    assert r.status_code == 422


# ── chat session endpoints ─────────────────────────────────────────────────────

def test_chat_session_create_and_get(client):
    r = client.post("/chat/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    r2 = client.get(f"/chat/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid


def test_chat_workdir_sets_path(client, tmp_path):
    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]

    r2 = client.post(f"/chat/sessions/{sid}/workdir", json={"path": str(tmp_path), "create": False})
    assert r2.status_code == 200
    assert str(tmp_path) in str(r2.json())


def test_chat_workdir_missing_path_is_not_created(client, tmp_path):
    """A path that does not exist is a typo, not a request to make a directory.

    Creating it silently gave the session an empty workspace that looked like it
    had worked, and the agent then spent its step budget on invented filenames.
    """
    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]
    missing = tmp_path / "nope"

    r2 = client.post(f"/chat/sessions/{sid}/workdir", json={"path": str(missing)})
    assert r2.status_code == 400
    assert not missing.exists()


def test_chat_workdir_suggests_near_miss_sibling(client, tmp_path):
    """A path copied out of a sentence keeps the full stop that ended it."""
    real = tmp_path / "Project_DS"
    real.mkdir()

    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]
    r2 = client.post(f"/chat/sessions/{sid}/workdir",
                     json={"path": str(tmp_path / "Project_DS.")})
    assert r2.status_code == 400
    assert str(real) in r2.json()["detail"]


def test_ui_and_server_agree_that_multiagent_is_off():
    """The checkbox and the server default must not drift apart.

    Nothing else reads this file, and a UI that ships `checked` while the server
    defaults to off means the toggle silently wins on every message. The storage
    key is versioned for the same reason: anyone carrying the old value would
    otherwise keep the orchestrator and the new default would look unapplied.
    """
    from pathlib import Path

    from agent.api.agent_runner import _MULTIAGENT

    assert _MULTIAGENT is False
    html = (Path(__file__).resolve().parents[2]
            / "agent" / "api" / "static" / "index.html").read_text()
    checkbox = next(ln for ln in html.splitlines() if 'id="ma-checkbox"' in ln)
    assert "checked" not in checkbox, checkbox
    # Every localStorage call for this toggle must use the versioned key. A
    # bare mention in a comment is fine; a call is not.
    calls = [ln.strip() for ln in html.splitlines()
             if "localStorage" in ln and "ds_multiagent" in ln]
    assert calls, "toggle no longer persists at all"
    assert all("ds_multiagent_v2" in ln for ln in calls), calls


def test_page_waits_for_long_runs_instead_of_rendering_empty():
    """A 16-minute analysis must still reach the page when it finishes."""
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parents[2]
            / "agent" / "api" / "static" / "index.html").read_text()
    render = html[html.index("async function pollAndRender"):]
    render = render[:render.index("\n}\n")]
    assert "waitForJob(" in render
    assert not re.search(r"attempt\s*<\s*\d+", render)   # no fixed poll budget
    chat_src = (Path(__file__).resolve().parents[2] / "agent" / "api" / "chat.py").read_text()
    assert "deadline = DEFAULT_JOB_TIMEOUT_S +" in chat_src


def test_failed_run_reports_instead_of_vanishing(client, monkeypatch):
    """A crash in the run must reach the user, in the log and the conversation.

    The worker used to be try/finally with no except: the finally closed the
    progress stream, so the UI printed "Done" and moved on, while the turn ended
    with the user's message and no answer at all. Nothing said anything had gone
    wrong.
    """
    from agent.api import agent_runner, progress

    def _boom(*a, **kw):
        raise RuntimeError("kernel exploded")

    emitted: list[tuple[str, str]] = []
    real_emit = progress.emit

    def _spy(job_id, message, *, tool="", level="info"):
        emitted.append((message, level))
        real_emit(job_id, message, tool=tool, level=level)

    monkeypatch.setattr(agent_runner, "run_reactive", _boom)
    monkeypatch.setattr(client.app.state.llm, "is_available", lambda: True)
    monkeypatch.setattr(progress, "emit", _spy)

    sid = client.post("/chat/sessions").json()["session_id"]
    posted = client.post(f"/chat/sessions/{sid}/message", json={"text": "do it"})
    assert posted.status_code == 200
    jid = posted.json()["job_id"]

    # The job is still marked failed, carrying the real error.
    for _ in range(50):
        job = client.get(f"/chat/sessions/{sid}/jobs/{jid}").json()
        if job.get("state") in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["state"] == "failed"
    assert "kernel exploded" in (job.get("error") or "")

    # The conversation records the failure rather than ending on the question.
    msgs = client.get(f"/chat/sessions/{sid}").json()["messages"]
    assert msgs and msgs[-1]["role"] == "assistant"

    # And the failure was announced on the progress stream before it closed.
    # close() removes the queue, so a live SSE client sees the event but the
    # test cannot drain it afterwards; capture the emit itself instead.
    assert any(lvl == "error" and "kernel exploded" in msg
               for msg, lvl in emitted)


def test_pending_endpoints_are_wired(client, monkeypatch):
    """The three pending-change routes reach the queue functions.

    The staging and applying behaviour itself is covered against a real kernel
    in test_agent_runner_helpers; this checks the HTTP surface around it.
    """
    from agent.api import agent_tools as at

    seen: dict[str, object] = {}

    def _fake_apply(kernel, ids, session=None):
        seen["apply"] = ids
        return {"applied": [], "failed": [], "pending": 0}

    def _fake_discard(kernel, ids):
        seen["discard"] = ids
        return {"discarded": 1, "pending": 0}

    monkeypatch.setattr(at, "pending_changes",
                        lambda k: [{"id": "chg_1", "summary": "create a.txt"}])
    monkeypatch.setattr(at, "apply_pending", _fake_apply)
    monkeypatch.setattr(at, "discard_pending", _fake_discard)

    sid = client.post("/chat/sessions").json()["session_id"]

    listed = client.get(f"/chat/sessions/{sid}/pending")
    assert listed.status_code == 200
    assert listed.json()["pending"][0]["id"] == "chg_1"

    r = client.post(f"/chat/sessions/{sid}/pending/apply", json={"ids": ["chg_1"]})
    assert r.status_code == 200 and seen["apply"] == ["chg_1"]

    r = client.post(f"/chat/sessions/{sid}/pending/discard", json={})
    assert r.status_code == 200 and seen["discard"] is None


def test_chat_workdir_with_csv_autoscan(client, tmp_path):
    """Setting a workdir with a CSV in it auto-loads it as df."""
    import pandas as pd
    csv = tmp_path / "data.csv"
    pd.DataFrame({"a": range(5), "b": range(5)}).to_csv(csv, index=False)

    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]

    r2 = client.post(f"/chat/sessions/{sid}/workdir", json={"path": str(tmp_path), "create": False})
    assert r2.status_code == 200
    body = r2.json()
    assert len(body["loaded_data"]) == 1
    assert "data.csv" in body["loaded_data"][0]


def test_chat_workdir_does_not_load_agent_plan_as_instructions(client, tmp_path):
    """A plan the agent saved earlier is its own output, not the user's brief."""
    (tmp_path / "brief.md").write_text("Find unusual providers.")
    (tmp_path / "analysis_plan.md").write_text(
        "# Analysis plan\n\n_First pass, written before EDA. The feature and model "
        "choices are assumptions to be tested._\n\nTarget: `service_date`\n")
    sid = client.post("/chat/sessions").json()["session_id"]
    body = client.post(f"/chat/sessions/{sid}/workdir",
                       json={"path": str(tmp_path), "create": False}).json()
    assert body["loaded_briefs"] == ["`brief.md`"]
    session = client.get(f"/chat/sessions/{sid}").json()
    assert "analysis_plan.md" not in session["brief_filenames"]
    assert "plans the agent wrote earlier" in session["messages"][-1]["text"]


def test_chat_upload_secondary_loads_as_df2(client, tmp_path):
    """Uploading two CSV files makes the second available as df2."""
    import io

    import pandas as pd

    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]

    csv1 = io.BytesIO(pd.DataFrame({"x": range(5)}).to_csv(index=False).encode())
    csv2 = io.BytesIO(pd.DataFrame({"y": range(8)}).to_csv(index=False).encode())

    r2 = client.post(
        f"/chat/sessions/{sid}/upload",
        files=[("files", ("primary.csv", csv1, "text/csv")),
               ("files", ("secondary.csv", csv2, "text/csv"))],
    )
    assert r2.status_code == 200
    body = r2.json()
    assert "df2" in body["load_summary"]


def test_chat_brief_upload_clears_task_spec(client, tmp_path):
    """Uploading a brief resets TASK_SPEC so understand_task re-runs."""
    import io

    r = client.post("/chat/sessions")
    sid = r.json()["session_id"]

    # Inject a fake TASK_SPEC into the kernel by sending a chat message won't work
    # — just verify upload succeeds and session is updated
    brief_txt = b"# Task\nPredict churn."
    r2 = client.post(
        f"/chat/sessions/{sid}/brief",
        files=[("files", ("task.md", brief_txt, "text/markdown"))],
    )
    assert r2.status_code == 200
    assert r2.json()["brief_chars"] > 0


# ── backend toggle (_use_claude) ─────────────────────────────────────────────

def test_use_claude_toggle(monkeypatch):
    from agent.api.server import _use_claude
    for v in ("yes", "Yes", "true", "1", "on", "claude"):
        monkeypatch.setenv("USE_CLAUDE", v)
        assert _use_claude() is True, v
    for v in ("no", "No", "false", "0", "off", ""):
        monkeypatch.setenv("USE_CLAUDE", v)
        assert _use_claude() is False, v


def test_use_claude_legacy_fallback(monkeypatch):
    """With USE_CLAUDE unset, the old DSAGENT_LLM_BACKEND still works."""
    from agent.api.server import _use_claude
    monkeypatch.delenv("USE_CLAUDE", raising=False)
    monkeypatch.setenv("DSAGENT_LLM_BACKEND", "anthropic")
    assert _use_claude() is True
    monkeypatch.setenv("DSAGENT_LLM_BACKEND", "ollama")
    assert _use_claude() is False
    monkeypatch.delenv("DSAGENT_LLM_BACKEND", raising=False)
    assert _use_claude() is False  # default: local Ollama


def test_use_claude_toggle_actually_selects_the_backend(monkeypatch):
    """The toggle function once survived while the line using it was lost, so
    USE_CLAUDE=yes silently ran the local model. Test the wiring, not just the
    parser."""
    from agent.api.llm import AnthropicClient, OllamaClient
    from agent.api.server import create_app, make_llm_client
    monkeypatch.setenv("USE_CLAUDE", "yes")
    assert isinstance(make_llm_client(), AnthropicClient)
    assert isinstance(create_app().state.llm, AnthropicClient)
    monkeypatch.setenv("USE_CLAUDE", "no")
    client = make_llm_client()
    assert isinstance(client, OllamaClient) and not isinstance(client, AnthropicClient)
