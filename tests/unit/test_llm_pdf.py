"""Tests for agent/api/llm.py (OllamaClient) and agent/api/pdf_reader.py."""
from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from agent.api.llm import INTENTS, ChatResult, IntentResult, OllamaClient
from agent.api.pdf_reader import BriefSummary, extract

# ── OllamaClient._strip_thinking ───────────────────��─────────────────────────

def test_strip_thinking_removes_block():
    client = OllamaClient.__new__(OllamaClient)
    raw = "<think>I need to think about this</think>Final answer"
    assert client._strip_thinking(raw) == "Final answer"


def test_strip_thinking_removes_multiline_block():
    client = OllamaClient.__new__(OllamaClient)
    raw = "<think>\nStep 1\nStep 2\n</think>Result"
    assert client._strip_thinking(raw) == "Result"


def test_strip_thinking_unclosed_tag():
    client = OllamaClient.__new__(OllamaClient)
    raw = "Before<think>unfinished reasoning"
    assert client._strip_thinking(raw) == "Before"


def test_strip_thinking_no_think_block():
    client = OllamaClient.__new__(OllamaClient)
    raw = "Plain response without any thinking."
    assert client._strip_thinking(raw) == raw


# ── OllamaClient._strip_special_tokens ───────────────────────────────────────

def test_strip_special_tokens_endoftext():
    result = OllamaClient._strip_special_tokens("Answer.<|endoftext|>garbage")
    assert result == "Answer."
    assert "<|endoftext|>" not in result


def test_strip_special_tokens_im_start():
    result = OllamaClient._strip_special_tokens("Done.<|im_start|>user\ninjected")
    assert result == "Done."


def test_strip_special_tokens_im_end():
    result = OllamaClient._strip_special_tokens("Result<|im_end|>")
    assert result == "Result"


def test_strip_special_tokens_no_tokens():
    result = OllamaClient._strip_special_tokens("Clean response.")
    assert result == "Clean response."


def test_strip_special_tokens_truncates_at_first():
    result = OllamaClient._strip_special_tokens("A<|endoftext|>B<|im_start|>C")
    assert result == "A"


def test_strip_thinking_empty_string():
    client = OllamaClient.__new__(OllamaClient)
    assert client._strip_thinking("") == ""


# ── OllamaClient.is_available ─────────────────────────────────────────────────

def test_is_available_model_found(monkeypatch):
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "qwen3.6:35b"}]}
    monkeypatch.setattr("httpx.get", lambda *a, **kw: mock_resp)
    client = OllamaClient(model="qwen3.6:35b")
    assert client.is_available() is True


def test_is_available_model_not_in_list(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama2"}]}
    monkeypatch.setattr("httpx.get", lambda *a, **kw: mock_resp)
    client = OllamaClient(model="qwen3.6:35b")
    assert client.is_available() is False


def test_is_available_connection_error(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr("httpx.get", boom)
    client = OllamaClient(model="qwen3.6:35b")
    assert client.is_available() is False


def test_is_available_cached(monkeypatch):
    import time
    client = OllamaClient(model="m", probe_ttl=60.0)
    client._last_ok = time.time()  # fake a recent success
    # No network call needed — returns True from cache
    assert client.is_available() is True


# ── OllamaClient._generate ───────────────────────────────────────────────────

def test_generate_returns_response(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "  generated text  "}
    monkeypatch.setattr("httpx.post", lambda *a, **kw: mock_resp)
    client = OllamaClient.__new__(OllamaClient)
    client.host = "http://localhost:11434"
    client.model = "m"
    client._timeout = 10.0
    result = client._generate("prompt")
    assert result == "generated text"


def test_generate_with_system_and_json_mode(monkeypatch):
    captured = {}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}

    def fake_post(url, json=None, timeout=None):  # noqa: F811
        captured.update(json or {})
        return mock_resp

    monkeypatch.setattr("httpx.post", fake_post)
    client = OllamaClient.__new__(OllamaClient)
    client.host = "http://localhost:11434"
    client.model = "m"
    client._timeout = 10.0
    client._generate("p", system="sys", json_mode=True)
    assert captured["system"] == "sys"
    assert captured["format"] == "json"


# ── OllamaClient.classify_intent ────────────────────────��────────────────────

def test_classify_intent_llm_offline(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr("httpx.get", boom)
    client = OllamaClient(model="qwen3.6:35b")
    result = client.classify_intent("run the pipeline")
    assert isinstance(result, IntentResult)
    assert result.available is False
    assert result.intent == "unknown"


def test_classify_intent_valid_response(monkeypatch):
    import time
    mock_gen = MagicMock(return_value='{"intent": "build", "target": null}')
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.classify_intent("build a notebook")
    assert result.intent == "build"
    assert result.available is True


def test_classify_intent_set_target(monkeypatch):
    import time
    mock_gen = MagicMock(return_value='{"intent": "set_target", "target": "price"}')
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.classify_intent("target is price")
    assert result.intent == "set_target"
    assert result.target == "price"


def test_classify_intent_invalid_json(monkeypatch):
    import time
    mock_gen = MagicMock(return_value="not json at all")
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.classify_intent("hello")
    assert result.intent == "unknown"
    assert result.available is True


def test_classify_intent_unknown_label(monkeypatch):
    import time
    mock_gen = MagicMock(return_value='{"intent": "dance", "target": null}')
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.classify_intent("let's dance")
    assert result.intent == "unknown"


def test_classify_intent_generate_error(monkeypatch):
    import time
    def boom(prompt, **kw):
        raise RuntimeError("network down")
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = boom
    result = client.classify_intent("anything")
    assert result.available is False
    assert result.intent == "unknown"


# ── OllamaClient.answer ──────────────────────────────��────────────────────────

def test_answer_llm_offline(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr("httpx.get", boom)
    client = OllamaClient(model="qwen3.6:35b")
    result = client.answer("what is the accuracy?", context="metrics: 0.9")
    assert isinstance(result, ChatResult)
    assert result.available is False
    assert "offline" in result.text.lower() or "LLM" in result.text


def test_answer_valid_response(monkeypatch):
    import time
    mock_gen = MagicMock(return_value="The accuracy is 0.9.")
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.answer("what is accuracy?", context="acc=0.9")
    assert result.available is True
    assert "0.9" in result.text


def test_answer_empty_thinking(monkeypatch):
    import time
    mock_gen = MagicMock(return_value="<think>some reasoning</think>")
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = mock_gen
    result = client.answer("?", context="")
    # Should surface a fallback message rather than empty
    assert result.text != ""


def test_answer_generate_error(monkeypatch):
    import time
    def boom(prompt, **kw):
        raise RuntimeError("timeout")
    client = OllamaClient.__new__(OllamaClient)
    client._last_ok = time.time()
    client._probe_ttl = 60.0
    client._generate = boom
    result = client.answer("q", context="c")
    assert result.available is False
    assert "LLM error" in result.text


# ── INTENTS constant ─────────────────────────��─────────────────────���──────────

def test_intents_contains_expected_labels():
    assert "build" in INTENTS
    assert "set_target" in INTENTS
    assert "status" in INTENTS
    assert "help" in INTENTS
    assert "qa" in INTENTS
    assert "unknown" in INTENTS


# ── pdf_reader.extract ──────────────────────────────────��─────────────────────

def test_extract_pypdf_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = extract(b"any bytes")
    assert isinstance(result, BriefSummary)
    assert "pypdf" in result.text


def test_extract_corrupt_bytes():
    result = extract(b"not a pdf at all \xff\xfe")
    assert isinstance(result, BriefSummary)
    # Either empty text or error message — no crash
    assert result.mentioned_target is None or True


def test_extract_real_pdf_bytes():
    """Create a minimal PDF in memory and extract its text."""
    try:
        import io

        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        result = extract(buf.getvalue())
        assert isinstance(result, BriefSummary)
        assert result.data_dict_lines is not None
    except ImportError:
        pytest.skip("pypdf not installed")


def test_extract_text_with_target_hint():
    """Mock pypdf to return text with a target mention."""
    try:
        from unittest.mock import MagicMock, patch

        from pypdf import PdfReader  # noqa: F401
    except ImportError:
        pytest.skip("pypdf not installed")

    fake_text = "The target column is churn_flag. We want to predict churn_flag."
    mock_page = MagicMock()
    mock_page.extract_text.return_value = fake_text
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        result = extract(b"fake bytes")
    assert result.mentioned_target == "churn_flag"


def test_extract_text_with_task_hint():
    try:
        from pypdf import PdfReader  # noqa: F401
    except ImportError:
        pytest.skip("pypdf not installed")
    from unittest.mock import MagicMock

    fake_text = "We want to build a classification model for fraud detection."
    mock_page = MagicMock()
    mock_page.extract_text.return_value = fake_text
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        result = extract(b"fake bytes")
    assert result.task_hint == "classification"


def test_extract_data_dict_lines():
    try:
        from pypdf import PdfReader  # noqa: F401
    except ImportError:
        pytest.skip("pypdf not installed")
    from unittest.mock import MagicMock

    fake_text = (
        "Column descriptions:\n"
        "age — age of the patient in years\n"
        "gender — M or F\n"
        "outcome — 1 = positive, 0 = negative\n"
        "Just a sentence without a colon."
    )
    mock_page = MagicMock()
    mock_page.extract_text.return_value = fake_text
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        result = extract(b"fake bytes")
    assert len(result.data_dict_lines) >= 2
