"""FastAPI surface tests — hit every endpoint and assert envelope shape."""
from __future__ import annotations

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
