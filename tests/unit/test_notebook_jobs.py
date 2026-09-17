"""Tests for agent/api/notebook_builder.py and agent/api/jobs.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.api import notebook_builder
from agent.api.jobs import Job, JobRunner
from agent.api.notebook_builder import _code, _fmt_metrics, _guess_targets, _md, build

# ── notebook_builder helpers ──────────────────────────────────────────────────

def test_md_cell_structure():
    cell = _md("# Title\n\nBody")
    assert cell["cell_type"] == "markdown"
    assert isinstance(cell["source"], list)
    assert "# Title" in cell["source"][0]


def test_md_empty_text():
    cell = _md("")
    assert cell["cell_type"] == "markdown"
    assert cell["source"] == [""]


def test_code_cell_structure():
    cell = _code("print('hi')")
    assert cell["cell_type"] == "code"
    assert cell["execution_count"] is None
    assert cell["outputs"] == []
    assert isinstance(cell["source"], list)


def test_code_empty_src():
    cell = _code("")
    assert cell["source"] == [""]


def test_fmt_metrics_float_values():
    result = _fmt_metrics({"accuracy": 0.9523, "f1": 0.8801})
    assert "accuracy" in result
    assert "0.9523" in result
    assert "f1" in result


def test_fmt_metrics_non_float():
    result = _fmt_metrics({"model": "rf", "n_features": 10})
    assert "model" in result
    assert "rf" in result


def test_guess_targets_from_profile():
    artifacts = {
        "profile": {
            "profile": [
                {"column": "outcome", "n_unique": 2, "dtype": "int64"},
                {"column": "score", "n_unique": 500, "dtype": "float64"},
                {"column": "category", "n_unique": 5, "dtype": "object"},
            ]
        }
    }
    candidates = _guess_targets(artifacts)
    assert "outcome" in candidates
    assert "category" in candidates
    assert "score" not in candidates  # float, too many unique


def test_guess_targets_empty_artifacts():
    candidates = _guess_targets({})
    assert len(candidates) >= 1  # fallback message


def test_guess_targets_no_profile_key():
    candidates = _guess_targets({"train": {"metrics": {"accuracy": 0.9}}})
    assert isinstance(candidates, list)


# ── notebook build — EDA-only mode ───────────────────────────────────────────

def test_build_eda_only_creates_file(tmp_path):
    data_p = tmp_path / "data.csv"
    data_p.write_text("a,b\n1,2\n3,4\n")
    out = tmp_path / "notebook.ipynb"
    result = build(
        data_path=data_p,
        target=None,
        brief="",
        artifacts={},
        out_path=out,
    )
    assert result == out
    assert out.exists()
    nb = json.loads(out.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0


def test_build_eda_only_auto_target_cell_present(tmp_path):
    data_p = tmp_path / "data.csv"
    data_p.write_text("x,y\n1,2\n")
    out = tmp_path / "nb.ipynb"
    build(data_path=data_p, target=None, brief="", artifacts={}, out_path=out)
    nb = json.loads(out.read_text())
    sources = " ".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    assert "auto-target" in sources.lower() or "EDA" in sources


def test_build_with_target_full_pipeline(tmp_path):
    data_p = tmp_path / "data.parquet"
    import pandas as pd
    pd.DataFrame({"a": [1], "b": [2], "outcome": [1]}).to_parquet(data_p, index=False)
    out = tmp_path / "nb.ipynb"
    build(
        data_path=data_p,
        target="outcome",
        brief="Predict outcome.",
        artifacts={
            "task": {"task": "binary_classification"},
            "train": {"estimator": "rf", "n_train": 80, "n_test": 20,
                       "metrics": {"accuracy": 0.91, "roc_auc": 0.95}},
        },
        out_path=out,
    )
    nb = json.loads(out.read_text())
    sources = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "7. Model training" in sources
    assert "0.9100" in sources or "0.91" in sources


def test_build_with_brief(tmp_path):
    data_p = tmp_path / "d.csv"
    data_p.write_text("x\n1\n2\n")
    out = tmp_path / "nb.ipynb"
    build(
        data_path=data_p, target=None,
        brief="This is the project brief about hospital data.",
        artifacts={}, out_path=out,
    )
    nb = json.loads(out.read_text())
    sources = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "Project brief" in sources or "hospital data" in sources


def test_build_regression_task(tmp_path):
    data_p = tmp_path / "d.csv"
    data_p.write_text("x,price\n1,100\n2,200\n")
    out = tmp_path / "nb.ipynb"
    build(
        data_path=data_p, target="price",
        brief="",
        artifacts={"task": {"task": "regression"}},
        out_path=out,
    )
    nb = json.loads(out.read_text())
    sources = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "regression" in sources.lower()


def test_build_with_profile_artifact(tmp_path):
    data_p = tmp_path / "d.csv"
    data_p.write_text("x\n1\n")
    out = tmp_path / "nb.ipynb"
    build(
        data_path=data_p, target="x", brief="",
        artifacts={
            "profile": {"n_rows": 1000, "n_cols": 5,
                         "overall_null_rate": 0.02, "duplicate_rows": 3},
            "task": {"task": "binary_classification"},
        },
        out_path=out,
    )
    nb = json.loads(out.read_text())
    sources = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "1,000" in sources or "1000" in sources


def test_build_with_drift_artifact(tmp_path):
    data_p = tmp_path / "d.csv"
    data_p.write_text("a\n1\n2\n")
    out = tmp_path / "nb.ipynb"
    build(
        data_path=data_p, target="a", brief="",
        artifacts={
            "task": {"task": "binary_classification"},
            "drift": {"n_checked": 5, "features": {"col_a": {"drifted": True}}},
        },
        out_path=out,
    )
    nb = json.loads(out.read_text())
    sources = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "Drift" in sources


def test_build_creates_parent_dirs(tmp_path):
    data_p = tmp_path / "d.csv"
    data_p.write_text("x\n1\n")
    out = tmp_path / "deep" / "nested" / "nb.ipynb"
    build(data_path=data_p, target=None, brief="", artifacts={}, out_path=out)
    assert out.exists()


# ── jobs.py — Job dataclass ───────────────────────────────────────────────────

def test_job_to_public_pending():
    job = Job(job_id="abc", kind="build")
    pub = job.to_public()
    assert pub["job_id"] == "abc"
    assert pub["kind"] == "build"
    assert pub["state"] == "pending"
    assert pub["result"] is None
    assert pub["error"] is None
    assert pub["elapsed_s"] == 0.0


def test_job_to_public_done():
    job = Job(job_id="xyz", kind="train")
    job.state = "done"
    job.started_at = 1000.0
    job.finished_at = 1005.0
    job.result = {"accuracy": 0.9}
    pub = job.to_public()
    assert pub["state"] == "done"
    assert pub["result"] == {"accuracy": 0.9}
    assert pub["elapsed_s"] == pytest.approx(5.0, abs=0.1)


def test_job_to_public_failed():
    job = Job(job_id="err", kind="chat")
    job.state = "failed"
    job.error = "RuntimeError: something went wrong"
    pub = job.to_public()
    assert pub["state"] == "failed"
    assert pub["result"] is None
    assert "RuntimeError" in pub["error"]


# ── jobs.py — JobRunner ───────────────────────────────────────────────────────

def test_job_runner_submit_and_wait(tmp_path):
    runner = JobRunner(max_workers=1)
    job = runner.submit("test", lambda: 42)
    # Poll briefly for completion
    for _ in range(50):
        if job.state == "done":
            break
        time.sleep(0.05)
    assert job.state == "done"
    assert job.result == 42
    runner.shutdown()


def test_job_runner_failed_job(tmp_path):
    def boom():
        raise ValueError("intentional failure")

    runner = JobRunner(max_workers=1)
    job = runner.submit("fail", boom)
    for _ in range(100):
        if job.state == "failed" and job.error is not None:
            break
        time.sleep(0.05)
    assert job.state == "failed"
    assert job.error is not None
    assert "ValueError" in job.error
    runner.shutdown()


def test_job_runner_get_existing():
    runner = JobRunner()
    job = runner.submit("x", lambda: None, job_id="myjob")
    fetched = runner.get("myjob")
    assert fetched is job
    runner.shutdown()


def test_job_runner_get_missing():
    runner = JobRunner()
    assert runner.get("notfound") is None
    runner.shutdown()


def test_job_runner_custom_job_id():
    runner = JobRunner()
    job = runner.submit("kind", lambda: "result", job_id="custom123")
    assert job.job_id == "custom123"
    runner.shutdown()
