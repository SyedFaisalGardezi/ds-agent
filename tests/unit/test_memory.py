"""Unit tests for agent/memory/ — all five memory layers.

All SQLite ops use tmp-file DBs.
All ChromaDB / HTTP / Ollama calls are mocked so tests run offline.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


def _tmp_csv() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    f.write("a,b,c\n1,2,3\n4,5,6\n")
    f.close()
    return Path(f.name)


def _fake_session(messages=None, isolation_mode=False, session_id="sess001"):
    msgs = messages or []
    s = SimpleNamespace(
        session_id=session_id,
        messages=msgs,
        isolation_mode=isolation_mode,
    )
    return s


def _make_msg(role, text):
    return SimpleNamespace(role=role, text=text)


# ── Layer 1: PersistentSessionStore ───────────────────────────────────────────

class TestPersistentSessionStore:
    def _make(self):
        from agent.memory.session_store import PersistentSessionStore
        return PersistentSessionStore(db_path=_tmp_db())

    def test_create_and_load_session(self):
        store = self._make()
        store.create_session("s1")
        row = store.load_session("s1")
        assert row is not None
        assert row["session_id"] == "s1"

    def test_load_missing_returns_none(self):
        store = self._make()
        assert store.load_session("nonexistent") is None

    def test_list_session_ids(self):
        store = self._make()
        store.create_session("a")
        store.create_session("b")
        ids = store.list_session_ids()
        assert "a" in ids
        assert "b" in ids

    def test_save_and_reload_session(self):
        store = self._make()
        store.create_session("s2")
        sess = SimpleNamespace(
            session_id="s2",
            target="churn",
            brief="predict churn",
            brief_filenames=["brief.pdf"],
            data_paths=[Path("/tmp/data.csv")],
            artifacts={"profile": {"status": "ok"}},
            notebook_path=None,
            isolation_mode=False,
        )
        store.save_session(sess)
        row = store.load_session("s2")
        assert row["target"] == "churn"
        assert json.loads(row["brief_filenames"]) == ["brief.pdf"]

    def test_save_and_load_messages(self):
        store = self._make()
        store.create_session("s3")
        store.save_message("s3", "user", "hello")
        store.save_message("s3", "assistant", "hi there")
        msgs = store.load_messages("s3")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hi there"

    def test_load_messages_order_oldest_first(self):
        store = self._make()
        store.create_session("s4")
        for i in range(5):
            store.save_message("s4", "user", f"msg{i}")
        msgs = store.load_messages("s4", n=5)
        assert [m["content"] for m in msgs] == [f"msg{i}" for i in range(5)]

    def test_load_messages_limit(self):
        store = self._make()
        store.create_session("s5")
        for i in range(10):
            store.save_message("s5", "user", f"m{i}")
        msgs = store.load_messages("s5", n=3)
        assert len(msgs) == 3

    def test_message_count(self):
        store = self._make()
        store.create_session("s6")
        store.save_message("s6", "user", "a")
        store.save_message("s6", "user", "b")
        assert store.message_count("s6") == 2

    def test_isolation_mode_roundtrip(self):
        store = self._make()
        store.create_session("s7")
        assert store.get_isolation_mode("s7") is False
        store.set_isolation_mode("s7", True)
        assert store.get_isolation_mode("s7") is True
        store.set_isolation_mode("s7", False)
        assert store.get_isolation_mode("s7") is False

    def test_delete_session(self):
        store = self._make()
        store.create_session("s8")
        store.save_message("s8", "user", "msg")
        store.delete_session("s8")
        assert store.load_session("s8") is None
        assert store.load_messages("s8") == []


# ── Layer 2: LongTermMemory ───────────────────────────────────────────────────

class TestLongTermMemory:
    """Mock httpx.post (embed) + chromadb.PersistentClient."""

    def _mock_col(self):
        col = MagicMock()
        col.count.return_value = 3
        col.query.return_value = {
            "documents": [["doc_a", "doc_b"]],
            "metadatas": [[{"session_id": "s1", "doc_type": "synthesis"},
                           {"session_id": "s2", "doc_type": "tool_output"}]],
            "distances": [[0.1, 0.3]],
        }
        return col

    def _mock_embed_response(self, vec=None):
        vec = vec or [0.1, 0.2, 0.3]
        resp = MagicMock()
        resp.json.return_value = {"embedding": vec}
        resp.raise_for_status.return_value = None
        return resp

    def _make(self):
        col = self._mock_col()
        client = MagicMock()
        client.get_or_create_collection.return_value = col

        with patch("chromadb.PersistentClient", return_value=client):
            from agent.memory.long_term import LongTermMemory
            mem = LongTermMemory.__new__(LongTermMemory)
            mem._client = client
            mem._col = col
        return mem

    def test_store_calls_add_when_embed_succeeds(self):
        mem = self._make()
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            mem.store("some content", session_id="s1", doc_type="synthesis")
        mem._col.add.assert_called_once()
        call_kwargs = mem._col.add.call_args
        assert "s1" in str(call_kwargs)

    def test_store_noop_when_embed_returns_none(self):
        mem = self._make()
        with patch("httpx.post", side_effect=Exception("offline")):
            mem.store("content", session_id="s1", doc_type="synthesis")
        mem._col.add.assert_not_called()

    def test_store_tool_output(self):
        mem = self._make()
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            mem.store_tool_output("s1", "eda_profile", "Shape: 100×5")
        mem._col.add.assert_called_once()
        content_arg = mem._col.add.call_args[1]["documents"][0]
        assert "eda_profile" in content_arg

    def test_store_model_result(self):
        mem = self._make()
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            mem.store_model_result("s1", "rf", "accuracy", 0.87, "titanic.csv")
        mem._col.add.assert_called_once()

    def test_retrieve_returns_list(self):
        mem = self._make()
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            results = mem.retrieve("churn prediction", n=5)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "content" in results[0]
        assert "distance" in results[0]

    def test_retrieve_returns_empty_when_embed_fails(self):
        mem = self._make()
        with patch("httpx.post", side_effect=Exception("offline")):
            results = mem.retrieve("query")
        assert results == []

    def test_retrieve_returns_empty_when_collection_empty(self):
        mem = self._make()
        mem._col.count.return_value = 0
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            results = mem.retrieve("query")
        assert results == []

    def test_retrieve_as_context_string_filters_by_distance(self):
        mem = self._make()
        # Override distances — one within threshold, one beyond
        mem._col.query.return_value = {
            "documents": [["close doc", "far doc"]],
            "metadatas": [[{"session_id": "s1", "doc_type": "synthesis"},
                           {"session_id": "s2", "doc_type": "synthesis"}]],
            "distances": [[0.1, 0.9]],
        }
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            ctx = mem.retrieve_as_context_string("query", distance_threshold=0.55)
        assert "close doc" in ctx
        assert "far doc" not in ctx

    def test_retrieve_as_context_string_empty_when_all_far(self):
        mem = self._make()
        mem._col.query.return_value = {
            "documents": [["far doc"]],
            "metadatas": [[{"session_id": "s1", "doc_type": "synthesis"}]],
            "distances": [[0.99]],
        }
        resp = self._mock_embed_response()
        with patch("httpx.post", return_value=resp):
            ctx = mem.retrieve_as_context_string("query", distance_threshold=0.55)
        assert ctx == ""

    def test_count(self):
        mem = self._make()
        assert mem.count() == 3


# ── Layer 3: SchemaCache ──────────────────────────────────────────────────────

class TestSchemaCache:
    def _make(self):
        from agent.memory.schema_cache import SchemaCache
        return SchemaCache(db_path=_tmp_db())

    def test_save_and_load_roundtrip(self):
        cache = self._make()
        csv = _tmp_csv()
        df = pd.read_csv(csv)
        cache.save(csv, df)
        loaded = cache.load(csv)
        assert loaded is not None
        assert loaded["shape"] == [2, 3]
        assert "a" in loaded["columns"]

    def test_load_miss_returns_none(self):
        cache = self._make()
        assert cache.load(Path("/nonexistent/file.csv")) is None

    def test_different_content_different_hash(self):
        cache = self._make()
        csv1 = _tmp_csv()
        df1 = pd.read_csv(csv1)
        cache.save(csv1, df1)

        # Write different content to a new file
        csv2 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        csv2.write("x,y\n10,20\n30,40\n50,60\n")
        csv2.close()
        csv2_path = Path(csv2.name)
        df2 = pd.read_csv(csv2_path)
        cache.save(csv2_path, df2)

        schema2 = cache.load(csv2_path)
        assert schema2 is not None
        assert "x" in schema2["columns"]

    def test_load_as_summary_format(self):
        cache = self._make()
        csv = _tmp_csv()
        df = pd.read_csv(csv)
        cache.save(csv, df)
        summary = cache.load_as_summary(csv)
        assert summary is not None
        assert "Shape:" in summary
        assert "2" in summary  # row count

    def test_load_as_summary_miss_returns_none(self):
        cache = self._make()
        assert cache.load_as_summary(Path("/no/file.csv")) is None

    def test_list_cached(self):
        cache = self._make()
        csv = _tmp_csv()
        df = pd.read_csv(csv)
        cache.save(csv, df)
        rows = cache.list_cached()
        assert len(rows) >= 1
        assert "filename" in rows[0]
        assert "rows" in rows[0]

    def test_overwrite_same_file(self):
        cache = self._make()
        csv = _tmp_csv()
        df = pd.read_csv(csv)
        cache.save(csv, df)
        cache.save(csv, df)  # second save — INSERT OR REPLACE
        assert len(cache.list_cached()) == 1


# ── Layer 4: build_context_messages ──────────────────────────────────────────

class TestBuildContextMessages:
    def _make_llm(self, summary="Summary text"):
        llm = MagicMock()
        llm._generate.return_value = summary
        return llm

    def test_empty_session_returns_empty(self):
        from agent.memory.session_memory import build_context_messages
        sess = _fake_session(messages=[])
        result = build_context_messages(sess, llm=None)
        assert result == []

    def test_recent_turns_verbatim(self):
        from agent.memory.session_memory import RECENT_TURNS, build_context_messages
        msgs = [_make_msg("user", f"q{i}") for i in range(3)]
        sess = _fake_session(messages=msgs)
        result = build_context_messages(sess, llm=None)
        # No older turns → no system block, just the 3 messages
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "q0"

    def test_older_turns_trigger_summary(self):
        from agent.memory.session_memory import RECENT_TURNS, build_context_messages
        # Create more messages than RECENT_TURNS to force summarisation
        msgs = [_make_msg("user" if i % 2 == 0 else "assistant", f"msg{i}")
                for i in range(RECENT_TURNS + 4)]
        sess = _fake_session(messages=msgs)
        llm = self._make_llm("This is a summary")
        result = build_context_messages(sess, llm=llm)
        # First block should be the summary system message
        assert result[0]["role"] == "system"
        assert "summary" in result[0]["content"].lower() or "Summary" in result[0]["content"]
        llm._generate.assert_called_once()

    def test_ltm_injection_when_available(self):
        from agent.memory.session_memory import build_context_messages
        msgs = [_make_msg("user", "what is churn?")]
        sess = _fake_session(messages=msgs)
        ltm = MagicMock()
        ltm.retrieve_as_context_string.return_value = "--- Relevant past context ---\nsome memory"
        result = build_context_messages(sess, llm=None, ltm=ltm, current_query="churn")
        roles = [m["role"] for m in result]
        assert "system" in roles

    def test_ltm_suppressed_in_isolation_mode(self):
        from agent.memory.session_memory import build_context_messages
        msgs = [_make_msg("user", "question")]
        sess = _fake_session(messages=msgs, isolation_mode=True)
        ltm = MagicMock()
        ltm.retrieve_as_context_string.return_value = "some memories"
        build_context_messages(sess, llm=None, ltm=ltm, current_query="question")
        ltm.retrieve_as_context_string.assert_not_called()

    def test_ltm_empty_string_not_injected(self):
        from agent.memory.session_memory import build_context_messages
        msgs = [_make_msg("user", "hi")]
        sess = _fake_session(messages=msgs)
        ltm = MagicMock()
        ltm.retrieve_as_context_string.return_value = ""
        result = build_context_messages(sess, llm=None, ltm=ltm, current_query="hi")
        # No system block — empty LTM result
        assert all(m["role"] != "system" for m in result)

    def test_user_content_trimmed(self):
        from agent.memory.session_memory import USER_CHAR_LIMIT, build_context_messages
        long_text = "x" * (USER_CHAR_LIMIT + 100)
        msgs = [_make_msg("user", long_text)]
        sess = _fake_session(messages=msgs)
        result = build_context_messages(sess, llm=None)
        assert len(result[0]["content"]) <= USER_CHAR_LIMIT + 1  # +1 for ellipsis char

    def test_system_messages_excluded_from_context(self):
        from agent.memory.session_memory import build_context_messages
        msgs = [
            _make_msg("system", "You are an agent"),
            _make_msg("user", "hello"),
            _make_msg("assistant", "hi"),
        ]
        sess = _fake_session(messages=msgs)
        result = build_context_messages(sess, llm=None)
        # system messages filtered out in input processing
        roles = [m["role"] for m in result]
        # Only user/assistant from the session messages
        assert roles.count("user") == 1
        assert roles.count("assistant") == 1


# ── Layer 5: isolation.py ─────────────────────────────────────────────────────

class TestDetectIsolationIntent:
    def test_fresh_start_enables(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("please give me a fresh start") == "enable"

    def test_clean_slate_enables(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("use a clean slate for this session") == "enable"

    def test_isolation_mode_keyword(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("enable isolation mode please") == "enable"

    def test_ignore_memory_enables(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("ignore my previous memory") == "enable"

    def test_disable_isolation(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("turn off isolation") == "disable"

    def test_restore_memory(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("restore memory context") == "disable"

    def test_unrelated_returns_none(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("train a random forest on my data") is None
        assert detect_isolation_intent("what is the accuracy?") is None

    def test_case_insensitive(self):
        from agent.memory.isolation import detect_isolation_intent
        assert detect_isolation_intent("FRESH START") == "enable"


class TestApplyIsolationIntent:
    def test_enable_sets_isolation_mode(self):
        from agent.memory.isolation import apply_isolation_intent
        sess = SimpleNamespace(session_id="s1", isolation_mode=False)
        reply = apply_isolation_intent("fresh start", sess, db_store=None)
        assert sess.isolation_mode is True
        assert reply is not None
        assert "isolation" in reply.lower()

    def test_disable_clears_isolation_mode(self):
        from agent.memory.isolation import apply_isolation_intent
        sess = SimpleNamespace(session_id="s1", isolation_mode=True)
        reply = apply_isolation_intent("turn off isolation", sess, db_store=None)
        assert sess.isolation_mode is False
        assert reply is not None

    def test_unrelated_returns_none(self):
        from agent.memory.isolation import apply_isolation_intent
        sess = SimpleNamespace(session_id="s1", isolation_mode=False)
        reply = apply_isolation_intent("train a model", sess, db_store=None)
        assert reply is None
        assert sess.isolation_mode is False

    def test_db_store_called_when_provided(self):
        from agent.memory.isolation import apply_isolation_intent
        sess = SimpleNamespace(session_id="s1", isolation_mode=False)
        db = MagicMock()
        apply_isolation_intent("fresh start", sess, db_store=db)
        db.set_isolation_mode.assert_called_once_with("s1", True)

    def test_db_store_failure_is_silent(self):
        from agent.memory.isolation import apply_isolation_intent
        sess = SimpleNamespace(session_id="s1", isolation_mode=False)
        db = MagicMock()
        db.set_isolation_mode.side_effect = Exception("DB down")
        reply = apply_isolation_intent("fresh start", sess, db_store=db)
        # Session still mutated despite DB failure
        assert sess.isolation_mode is True
        assert reply is not None
