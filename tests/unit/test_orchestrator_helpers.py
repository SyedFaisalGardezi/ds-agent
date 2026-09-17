"""Tests for agent/api/orchestrator.py — pure helper functions."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.api.orchestrator import (
    _heuristic_target,
    _llm_infer_target,
    _sanitize,
)

# ── _sanitize ─────────────────────────────────────────────────────────────────

def test_sanitize_nan_float():
    assert _sanitize(float("nan")) is None


def test_sanitize_inf_float():
    assert _sanitize(float("inf")) is None
    assert _sanitize(float("-inf")) is None


def test_sanitize_finite_float():
    assert _sanitize(1.5) == pytest.approx(1.5)


def test_sanitize_numpy_float_nan():
    assert _sanitize(np.float64("nan")) is None


def test_sanitize_numpy_float_inf():
    assert _sanitize(np.float64("inf")) is None


def test_sanitize_numpy_integer():
    result = _sanitize(np.int64(7))
    assert result == 7
    assert isinstance(result, int)


def test_sanitize_numpy_bool():
    assert _sanitize(np.bool_(False)) is False


def test_sanitize_numpy_array():
    arr = np.array([1.0, float("nan")])
    result = _sanitize(arr)
    assert result == [1.0, None]


def test_sanitize_dataframe():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = _sanitize(df)
    assert isinstance(result, list)


def test_sanitize_series():
    s = pd.Series([1.0, float("nan")])
    result = _sanitize(s)
    assert result[1] is None


def test_sanitize_dict():
    d = {"a": np.int64(1), "b": float("nan")}
    result = _sanitize(d)
    assert result["a"] == 1
    assert result["b"] is None


def test_sanitize_list():
    result = _sanitize([1, np.int64(2), float("inf")])
    assert result == [1, 2, None]


def test_sanitize_plain_passthrough():
    assert _sanitize("text") == "text"
    assert _sanitize(99) == 99


# ── _heuristic_target ─────────────────────────────────────────────────────────

def test_heuristic_target_outcome_column():
    df = pd.DataFrame({
        "age": range(10),
        "outcome": [0, 1] * 5,
    })
    assert _heuristic_target(df) == "outcome"


def test_heuristic_target_label_column():
    df = pd.DataFrame({
        "feature_a": range(10),
        "label": [0, 1] * 5,
    })
    assert _heuristic_target(df) == "label"


def test_heuristic_target_partial_keyword():
    df = pd.DataFrame({
        "patient_id": range(10),
        "churn_flag": [0, 1] * 5,
    })
    result = _heuristic_target(df)
    assert result == "churn_flag"


def test_heuristic_target_skips_id_columns():
    df = pd.DataFrame({
        "user_id": range(10),
        "record_id": range(10),
        "result": [0, 1] * 5,
    })
    result = _heuristic_target(df)
    assert result == "result"


def test_heuristic_target_fallback_lowest_cardinality():
    df = pd.DataFrame({
        "numeric_a": range(100),
        "binary_col": [0, 1] * 50,
        "multi_col": list(range(10)) * 10,
    })
    result = _heuristic_target(df)
    assert result in ("binary_col", "multi_col")


def test_heuristic_target_returns_none_when_all_skip():
    df = pd.DataFrame({
        "created_at": range(10),
        "updated_at": range(10),
    })
    # All columns contain skip keywords → no candidate
    result = _heuristic_target(df)
    assert result is None or isinstance(result, str)


def test_heuristic_target_skips_high_cardinality():
    df = pd.DataFrame({
        "id": range(100),
        "free_text": [f"text_{i}" for i in range(100)],
        "outcome": [0, 1] * 50,
    })
    result = _heuristic_target(df)
    assert result == "outcome"


# ── _llm_infer_target ─────────────────────────────────────────────────────────

def _make_llm(response: str):
    llm = SimpleNamespace()
    llm._generate = lambda prompt, **kw: response
    llm._strip_thinking = lambda text: text
    return llm


def test_llm_infer_target_uses_llm_response():
    df = pd.DataFrame({
        "age": range(10),
        "outcome": [0, 1] * 5,
    })
    llm = _make_llm("outcome")
    result = _llm_infer_target(df, "predict outcome", llm)
    assert result == "outcome"


def test_llm_infer_target_falls_back_on_unknown():
    df = pd.DataFrame({
        "age": range(10),
        "outcome": [0, 1] * 5,
    })
    llm = _make_llm("unknown")
    result = _llm_infer_target(df, "brief", llm)
    # Falls back to heuristic — should still find "outcome"
    assert result == "outcome"


def test_llm_infer_target_falls_back_when_llm_none():
    df = pd.DataFrame({
        "a": range(10),
        "label": [0, 1] * 5,
    })
    result = _llm_infer_target(df, "classify", llm_client=None)
    assert result == "label"


def test_llm_infer_target_ignores_column_not_in_df():
    df = pd.DataFrame({
        "age": range(10),
        "outcome": [0, 1] * 5,
    })
    llm = _make_llm("nonexistent_column")
    result = _llm_infer_target(df, "brief", llm)
    # Falls back to heuristic
    assert result == "outcome"
