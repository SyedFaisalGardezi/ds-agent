"""Tests for agent/api/progress.py and agent/api/sessions.py."""
from __future__ import annotations

import threading
import time

import pytest

import agent.api.progress as progress
from agent.api.sessions import ChatMessage, Session, SessionStore

# ── progress.py ───────────────────────────────────────────────────────────────

def test_progress_create_returns_queue():
    q = progress.create("job1")
    assert q is not None
    assert q.empty()
    # Cleanup
    progress.close("job1")


def test_progress_emit_puts_message():
    progress.create("job2")
    progress.emit("job2", "hello", tool="eda", level="info")
    q = progress.get_queue("job2")
    assert q is not None
    item = q.get_nowait()
    assert item["message"] == "hello"
    assert item["tool"] == "eda"
    assert item["level"] == "info"
    assert item["type"] == "progress"
    assert "ts" in item
    progress.close("job2")


def test_progress_emit_unknown_job_is_noop():
    # Emitting to a non-existent job must not raise
    progress.emit("nonexistent_job_xyz", "ignored")


def test_progress_emit_full_queue_is_noop():
    q = progress.create("job3", maxsize=1)
    progress.emit("job3", "first")
    progress.emit("job3", "second")   # queue full — must not raise or block
    assert q.qsize() == 1
    q.get_nowait()  # drain so close() can put the sentinel without blocking
    progress.close("job3")


def test_progress_close_sends_sentinel_and_removes():
    progress.create("job4")
    progress.emit("job4", "msg")
    progress.close("job4")
    # Queue was removed — get_queue returns None
    assert progress.get_queue("job4") is None


def test_progress_get_queue_returns_none_for_missing():
    assert progress.get_queue("definitely_not_there") is None


def test_progress_close_missing_job_is_noop():
    progress.close("ghost_job_xyz")  # must not raise


def test_progress_default_level_and_tool():
    progress.create("job5")
    progress.emit("job5", "bare message")
    item = progress.get_queue("job5").get_nowait()
    assert item["tool"] == ""
    assert item["level"] == "info"
    progress.close("job5")


def test_progress_thread_safety():
    """Multiple threads emitting concurrently should not corrupt state."""
    progress.create("job_mt")
    errors = []

    def worker(n):
        try:
            for _ in range(10):
                progress.emit("job_mt", f"msg-{n}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    progress.close("job_mt")


# ── sessions.py — ChatMessage ─────────────────────────────────────────────────

def test_chat_message_fields():
    msg = ChatMessage(role="user", text="hello")
    assert msg.role == "user"
    assert msg.text == "hello"
    assert isinstance(msg.ts, float)
    assert msg.ts > 0


# ── sessions.py — Session ─────────────────────────────────────────────────────

def test_session_add_message():
    s = Session(session_id="abc")
    msg = s.add("user", "query text")
    assert isinstance(msg, ChatMessage)
    assert len(s.messages) == 1
    assert s.messages[0].role == "user"


def test_session_to_public_keys():
    s = Session(session_id="sid123")
    pub = s.to_public()
    assert pub["session_id"] == "sid123"
    assert pub["data_path"] is None
    assert pub["target"] is None
    assert pub["brief_chars"] == 0
    assert pub["notebook_ready"] is False
    assert pub["message_count"] == 0
    assert isinstance(pub["artifacts"], list)


def test_session_to_public_with_data(tmp_path):
    from pathlib import Path
    s = Session(session_id="s2")
    s.data_path = tmp_path / "data.csv"
    s.target = "outcome"
    s.brief = "some brief text"
    s.work_dir = tmp_path
    s.artifacts = {"profile": {}, "train": {}}
    pub = s.to_public()
    assert pub["target"] == "outcome"
    assert pub["brief_chars"] == len("some brief text")
    assert set(pub["artifacts"]) == {"profile", "train"}
    assert pub["work_dir"] == str(tmp_path)


# ── sessions.py — SessionStore ────────────────────────────────────────────────

def test_session_store_create_returns_session():
    store = SessionStore()
    s = store.create()
    assert isinstance(s, Session)
    assert len(s.session_id) == 12


def test_session_store_get_returns_session():
    store = SessionStore()
    s = store.create()
    fetched = store.get(s.session_id)
    assert fetched is s


def test_session_store_get_missing_returns_none():
    store = SessionStore()
    assert store.get("nonexistent") is None


def test_session_store_require_returns_session():
    store = SessionStore()
    s = store.create()
    assert store.require(s.session_id) is s


def test_session_store_require_raises_key_error():
    store = SessionStore()
    with pytest.raises(KeyError):
        store.require("missing_sid")


def test_session_store_list_ids():
    store = SessionStore()
    s1 = store.create()
    s2 = store.create()
    ids = store.list_ids()
    assert s1.session_id in ids
    assert s2.session_id in ids


def test_session_store_multiple_sessions_isolated():
    store = SessionStore()
    s1 = store.create()
    s2 = store.create()
    s1.target = "a"
    s2.target = "b"
    assert store.get(s1.session_id).target == "a"
    assert store.get(s2.session_id).target == "b"


# ── Session.add_data_path ─────────────────────────────────────────────────────

def test_add_data_path_sets_data_path(tmp_path):
    s = Session(session_id="x1")
    p = tmp_path / "data.csv"
    p.touch()
    s.add_data_path(p)
    assert s.data_path == p
    assert s.data_paths == [p]


def test_add_data_path_accumulates(tmp_path):
    s = Session(session_id="x2")
    p1 = tmp_path / "a.csv"; p1.touch()
    p2 = tmp_path / "b.csv"; p2.touch()
    s.add_data_path(p1)
    s.add_data_path(p2)
    assert s.data_paths == [p1, p2]
    # data_path always reflects the first file
    assert s.data_path == p1


def test_add_data_path_deduplicates(tmp_path):
    s = Session(session_id="x3")
    p = tmp_path / "data.csv"; p.touch()
    s.add_data_path(p)
    s.add_data_path(p)
    assert len(s.data_paths) == 1


def test_to_public_includes_data_paths(tmp_path):
    s = Session(session_id="x4")
    p1 = tmp_path / "a.csv"; p1.touch()
    p2 = tmp_path / "b.csv"; p2.touch()
    s.add_data_path(p1)
    s.add_data_path(p2)
    pub = s.to_public()
    assert len(pub["data_paths"]) == 2
    assert str(p1) in pub["data_paths"]
    assert str(p2) in pub["data_paths"]


# ── Session.add_brief ─────────────────────────────────────────────────────────

def test_add_brief_first_file():
    s = Session(session_id="b1")
    s.add_brief("task: predict churn", "brief.pdf")
    assert s.brief == "task: predict churn"
    assert s.brief_filename == "brief.pdf"
    assert s.brief_filenames == ["brief.pdf"]


def test_add_brief_second_file_concatenates():
    s = Session(session_id="b2")
    s.add_brief("first brief", "a.pdf")
    s.add_brief("second brief", "b.md")
    assert "first brief" in s.brief
    assert "second brief" in s.brief
    assert "b.md" in s.brief          # separator includes filename
    assert s.brief_filenames == ["a.pdf", "b.md"]
    assert s.brief_filename == "a.pdf"  # first stays as primary


def test_add_brief_keeps_first_filename_as_brief_filename():
    s = Session(session_id="b3")
    s.add_brief("text1", "instructions.pdf")
    s.add_brief("text2", "extra.txt")
    assert s.brief_filename == "instructions.pdf"


def test_to_public_includes_brief_filenames():
    s = Session(session_id="b4")
    s.add_brief("doc1", "f1.pdf")
    s.add_brief("doc2", "f2.md")
    pub = s.to_public()
    assert pub["brief_filenames"] == ["f1.pdf", "f2.md"]
    assert pub["brief_chars"] == len(s.brief)


# ── chat._load_secondary ──────────────────────────────────────────────────────

def test_load_secondary_csv(tmp_path):
    """_load_secondary loads a CSV into kernel as the given var_name."""
    import pandas as pd
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    csv = tmp_path / "secondary.csv"
    pd.DataFrame({"x": range(10), "y": range(10)}).to_csv(csv, index=False)

    kernel = SessionKernel()
    msg = _load_secondary(kernel, csv, "df2")

    assert kernel.namespace.get("df2") is not None
    assert len(kernel.namespace["df2"]) == 10
    assert "10" in msg


def test_load_secondary_unsupported_ext(tmp_path):
    """_load_secondary returns an error message for unsupported file types."""
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    fake = tmp_path / "file.json"
    fake.write_text("{}")
    kernel = SessionKernel()
    msg = _load_secondary(kernel, fake, "df2")
    assert "Unsupported" in msg or "unsupported" in msg.lower()


def test_load_secondary_missing_file(tmp_path):
    """_load_secondary returns an error message when the file doesn't exist."""
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    missing = tmp_path / "nonexistent.csv"
    kernel = SessionKernel()
    msg = _load_secondary(kernel, missing, "df2")
    # Should not raise — returns an error string
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_load_secondary_parquet(tmp_path):
    """_load_secondary loads a parquet file into kernel as var_name."""
    import pandas as pd
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    pq = tmp_path / "data.parquet"
    pd.DataFrame({"a": range(5)}).to_parquet(pq, index=False)

    kernel = SessionKernel()
    msg = _load_secondary(kernel, pq, "df3")
    assert kernel.namespace.get("df3") is not None
    assert len(kernel.namespace["df3"]) == 5


def test_load_secondary_tsv(tmp_path):
    """_load_secondary loads a TSV file into kernel as var_name."""
    import pandas as pd
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    tsv = tmp_path / "data.tsv"
    pd.DataFrame({"col": ["a", "b", "c"]}).to_csv(tsv, sep="\t", index=False)

    kernel = SessionKernel()
    msg = _load_secondary(kernel, tsv, "df4")
    assert kernel.namespace.get("df4") is not None
    assert len(kernel.namespace["df4"]) == 3


def test_load_secondary_excel(tmp_path):
    """_load_secondary loads an xlsx file into kernel as var_name."""
    pytest.importorskip("openpyxl")
    import pandas as pd
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel

    xls = tmp_path / "data.xlsx"
    pd.DataFrame({"val": [1, 2]}).to_excel(xls, index=False)

    kernel = SessionKernel()
    msg = _load_secondary(kernel, xls, "df5")
    assert kernel.namespace.get("df5") is not None
