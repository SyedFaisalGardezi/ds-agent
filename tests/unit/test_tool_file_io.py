from __future__ import annotations

import json

import pytest

from agent.tools.file_io import FileIOTool


@pytest.fixture
def tool(tmp_path):
    return FileIOTool(root=tmp_path)


def _call(tool, **payload):
    return tool.run(json.dumps(payload))


def test_rejects_invalid_json(tool):
    r = tool.run("not-json")
    assert r.status == "error"
    assert r.error.startswith("JSONDecodeError")


def test_rejects_unknown_action(tool):
    r = _call(tool, action="nuke", path="x.txt")
    assert r.status == "error"
    assert "Unsupported action" in r.explanation


def test_rejects_missing_path(tool):
    r = _call(tool, action="read")
    assert r.status == "error"
    assert "path is required" in r.error


def test_rejects_path_escape(tool):
    r = _call(tool, action="read", path="../../etc/passwd")
    assert r.status == "error"
    assert r.error == "PathEscapeError"


def test_exists_false_then_true(tool, tmp_path):
    r = _call(tool, action="exists", path="new.txt")
    assert r.status == "ok" and r.data["exists"] is False
    (tmp_path / "new.txt").write_text("hi")
    r = _call(tool, action="exists", path="new.txt")
    assert r.status == "ok" and r.data["exists"] is True


def test_write_and_read_csv(tool):
    data = [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]
    r = _call(tool, action="write", path="data.csv", data=data)
    assert r.status == "ok"
    assert r.data["shape"] == [2, 2]

    r = _call(tool, action="read", path="data.csv")
    assert r.status == "ok"
    assert r.data["shape"] == [2, 2]
    assert r.data["records"][0]["a"] == 1
    assert "Shape: 2 × 2" in r.math_trace
    assert "Numeric summary" in r.math_trace


def test_write_and_read_parquet(tool):
    data = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}]
    r = _call(tool, action="write", path="data.parquet", data=data)
    assert r.status == "ok"

    r = _call(tool, action="read", path="data.parquet")
    assert r.status == "ok"
    assert r.data["shape"] == [3, 2]
    assert r.data["records"][-1]["y"] == "c"


def test_write_and_read_json(tool):
    payload = {"nested": {"items": [1, 2, 3]}}
    r = _call(tool, action="write", path="cfg.json", data=payload)
    assert r.status == "ok"
    r = _call(tool, action="read", path="cfg.json")
    assert r.status == "ok"
    assert r.data["content"] == payload


def test_write_and_read_jsonl(tool):
    recs = [{"i": 1}, {"i": 2}, {"i": 3}]
    r = _call(tool, action="write", path="events.jsonl", data=recs)
    assert r.status == "ok"
    r = _call(tool, action="read", path="events.jsonl")
    assert r.status == "ok"
    assert r.data["count"] == 3
    assert r.data["records"][1]["i"] == 2


def test_write_and_read_txt(tool):
    r = _call(tool, action="write", path="notes.txt", data="hello\nworld\n")
    assert r.status == "ok"
    r = _call(tool, action="read", path="notes.txt")
    assert r.status == "ok"
    assert r.data["text"] == "hello\nworld\n"


def test_read_missing_file(tool):
    r = _call(tool, action="read", path="does_not_exist.csv")
    assert r.status == "error"
    assert "Not a file" in r.explanation


def test_write_csv_rejects_wrong_type(tool):
    r = _call(tool, action="write", path="bad.csv", data="just a string")
    assert r.status == "error"
    assert "list[dict]" in r.explanation


def test_write_jsonl_rejects_non_list(tool):
    r = _call(tool, action="write", path="bad.jsonl", data={"not": "a list"})
    assert r.status == "error"


def test_write_txt_rejects_non_string(tool):
    r = _call(tool, action="write", path="bad.txt", data=[1, 2, 3])
    assert r.status == "error"


def test_write_creates_nested_dirs(tool, tmp_path):
    r = _call(tool, action="write", path="reports/q1/summary.txt", data="x")
    assert r.status == "ok"
    assert (tmp_path / "reports" / "q1" / "summary.txt").is_file()


def test_list_returns_entries(tool, tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.csv").write_text("2")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("3")

    r = _call(tool, action="list", path=".")
    assert r.status == "ok"
    names = {e["name"] for e in r.data["entries"]}
    assert {"a.txt", "b.csv", "sub"} <= names

    r = _call(tool, action="list", path=".", options={"recursive": True, "glob": "*.txt"})
    assert r.status == "ok"
    paths = {e["path"] for e in r.data["entries"]}
    assert "a.txt" in paths
    assert "sub/c.txt" in paths


def test_list_not_a_directory(tool, tmp_path):
    (tmp_path / "file.txt").write_text("x")
    r = _call(tool, action="list", path="file.txt")
    assert r.status == "error"
    assert "Not a directory" in r.explanation


def test_format_inferred_from_extension(tool):
    r = _call(tool, action="write", path="inferred.csv", data=[{"a": 1}])
    assert r.status == "ok"
    # No explicit format — should have been inferred as csv, so re-read works.
    r = _call(tool, action="read", path="inferred.csv")
    assert r.status == "ok"
    assert r.data["shape"] == [1, 1]


def test_explicit_format_overrides_extension(tool):
    r = _call(tool, action="write", path="data.bin", format="json", data={"k": "v"})
    assert r.status == "ok"
    r = _call(tool, action="read", path="data.bin", format="json")
    assert r.status == "ok"
    assert r.data["content"] == {"k": "v"}


def test_tabular_preview_truncates(tool, tmp_path, monkeypatch):
    """Writing >10k rows should mark the read-back preview as truncated."""
    import agent.tools.file_io as mod
    monkeypatch.setattr(mod, "_TABULAR_PREVIEW_ROWS", 5)
    data = [{"i": i} for i in range(10)]
    _call(tool, action="write", path="big.csv", data=data)
    r = _call(tool, action="read", path="big.csv")
    assert r.status == "ok"
    assert r.data["truncated"] is True
    assert len(r.data["records"]) == 5
    assert r.data["shape"] == [10, 1]


def test_size_cap_enforced(tool, tmp_path, monkeypatch):
    import agent.tools.file_io as mod
    monkeypatch.setattr(mod, "_MAX_READ_BYTES", 10)
    (tmp_path / "big.txt").write_text("x" * 100)
    r = _call(tool, action="read", path="big.txt")
    assert r.status == "error"
    assert r.error == "SizeLimitExceeded"


def test_register_adds_to_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("DS_AGENT_FILE_ROOT", str(tmp_path))
    # Reload module so the env var is picked up by the module-level default.
    import importlib

    import agent.tools.file_io as mod
    importlib.reload(mod)
    from agent.core.tool_registry import ToolRegistry
    tool = mod.register()
    assert ToolRegistry.get()["file_io"] is tool
