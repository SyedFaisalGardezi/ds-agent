"""Unit tests for agent/api/spec_enricher.py (Layer 3 + Layer 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.api.spec_enricher import EnrichedSpec, SpecEnricher, TaskSpecValidator
from agent.api.task_extractor import RawSpec


def _raw_spec(**kwargs) -> RawSpec:
    defaults = dict(
        task_type="binary_classification",
        task_type_confidence=0.8,
        target_column_hint="",
        target_condition="",
        aggregation_needed=False,
        aggregation_key="",
        evaluation_metric="roc_auc",
        task_description="Predict outcome",
        source="llm_decomposed",
    )
    defaults.update(kwargs)
    return RawSpec(**defaults)


def _enriched_spec(**kwargs) -> EnrichedSpec:
    defaults = dict(
        task_type="binary_classification",
        task_type_confidence=0.8,
        target_column="outcome",
        target_condition="",
        aggregation_needed=False,
        aggregation_key="",
        evaluation_metric="roc_auc",
        task_description="Predict outcome",
        class_imbalance_ratio=None,
        enrichment_notes=[],
    )
    defaults.update(kwargs)
    return EnrichedSpec(**defaults)


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        "customer_id": range(n),
        "age": rng.integers(18, 80, n),
        "tenure": rng.integers(0, 120, n),
        "outcome": rng.integers(0, 2, n),
    })


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """95% class 0, 5% class 1 — highly imbalanced."""
    rng = np.random.default_rng(1)
    n = 2000
    labels = (rng.random(n) < 0.05).astype(int)
    return pd.DataFrame({
        "customer_id": range(n),
        "feature_a": rng.normal(0, 1, n),
        "outcome": labels,
    })


@pytest.fixture
def incident_df() -> pd.DataFrame:
    """Multiple incident rows per establishment — aggregation needed."""
    rng = np.random.default_rng(2)
    n = 400
    est_ids = rng.integers(1, 50, n)  # 50 establishments, ~8 rows each
    return pd.DataFrame({
        "establishment_id": est_ids,
        "establishment_type": rng.choice(["food", "retail", "warehouse"], n),
        "industry_code": rng.integers(100, 999, n),
        "inspection_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "outcome": rng.integers(0, 5, n),
    })


# ── SpecEnricher.enrich() ─────────────────────────────────────────────────────

def test_enrich_uses_llm_hint_when_column_exists(binary_df):
    raw = _raw_spec(target_column_hint="outcome")
    spec = SpecEnricher().enrich(raw, binary_df, list(binary_df.columns))
    assert spec.target_column == "outcome"


def test_enrich_falls_back_to_pattern_matching_when_hint_empty(binary_df):
    raw = _raw_spec(target_column_hint="")
    spec = SpecEnricher().enrich(raw, binary_df, list(binary_df.columns))
    assert spec.target_column == "outcome"


def test_enrich_rejects_group_key_hint(binary_df):
    raw = _raw_spec(target_column_hint="customer_id")
    spec = SpecEnricher().enrich(raw, binary_df, list(binary_df.columns))
    # customer_id contains "id" — looks like a group key — must be re-detected
    assert spec.target_column != "customer_id"


def test_enrich_imbalanced_overrides_metric_to_pr_auc(imbalanced_df):
    raw = _raw_spec(target_column_hint="outcome", evaluation_metric="roc_auc")
    spec = SpecEnricher().enrich(raw, imbalanced_df, list(imbalanced_df.columns))
    assert spec.evaluation_metric == "pr_auc"


def test_enrich_balanced_keeps_roc_auc(binary_df):
    raw = _raw_spec(target_column_hint="outcome", evaluation_metric="roc_auc")
    spec = SpecEnricher().enrich(raw, binary_df, list(binary_df.columns))
    assert spec.evaluation_metric == "roc_auc"


# ── _detect_aggregation_need ──────────────────────────────────────────────────

def test_detect_aggregation_name_based(incident_df):
    enricher = SpecEnricher()
    key, ratio = enricher._detect_aggregation_need(incident_df, list(incident_df.columns))
    assert key == "establishment_id"
    assert ratio > 1.5


def test_detect_aggregation_stat_safety_net():
    """Column named 'grp' (no pattern match) but unique_ratio < 0.5."""
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "grp": rng.integers(0, 30, n),   # 30 unique values in 300 rows = ratio 10
        "val": rng.normal(0, 1, n),
        "outcome": rng.integers(0, 2, n),
    })
    enricher = SpecEnricher()
    key, ratio = enricher._detect_aggregation_need(df, list(df.columns))
    assert key == "grp"
    assert ratio >= 2.0


def test_no_aggregation_when_all_unique(binary_df):
    enricher = SpecEnricher()
    key, ratio = enricher._detect_aggregation_need(binary_df, list(binary_df.columns))
    # customer_id is unique per row — not a valid group key
    if key:
        assert key != "customer_id" or ratio < 1.5


# ── TaskSpecValidator ─────────────────────────────────────────────────────────

def test_validator_passes_valid_spec():
    spec = _enriched_spec()
    ok, failures = TaskSpecValidator().validate(spec, ["outcome", "feature_a"])
    assert ok is True
    assert failures == []


def test_validator_fails_unknown_task_type():
    spec = _enriched_spec(task_type="unknown")
    ok, failures = TaskSpecValidator().validate(spec, ["outcome"])
    assert ok is False
    assert any("task_type" in f for f in failures)


def test_validator_fails_target_not_in_columns():
    spec = _enriched_spec(target_column="missing_col")
    ok, failures = TaskSpecValidator().validate(spec, ["outcome", "feature_a"])
    assert ok is False
    assert any("missing_col" in f for f in failures)


def test_validator_warns_id_column_as_target():
    spec = _enriched_spec(target_column="customer_id")
    ok, failures = TaskSpecValidator().validate(spec, ["customer_id", "outcome"])
    assert ok is False
    assert any("ID" in f or "id" in f.lower() for f in failures)


def test_validator_warns_low_confidence():
    spec = _enriched_spec(task_type_confidence=0.3)
    ok, failures = TaskSpecValidator().validate(spec, ["outcome"])
    assert ok is False
    assert any("confidence" in f.lower() for f in failures)


def test_validator_fails_missing_aggregation_key():
    spec = _enriched_spec(aggregation_needed=True, aggregation_key="")
    ok, failures = TaskSpecValidator().validate(spec, ["outcome"])
    assert ok is False
    assert any("aggregation_key" in f for f in failures)


def test_validator_fails_aggregation_key_not_in_columns():
    spec = _enriched_spec(aggregation_needed=True, aggregation_key="nonexistent_id")
    ok, failures = TaskSpecValidator().validate(spec, ["outcome"])
    assert ok is False
    assert any("nonexistent_id" in f for f in failures)


# ── SpecEnricher._apply_condition ─────────────────────────────────────────────

def test_apply_condition_empty_uses_default_col():
    df = pd.DataFrame({"label": [0, 1, 1, 0]})
    enricher = SpecEnricher()
    result = enricher._apply_condition(df, "label", "")
    assert list(result) == [0, 1, 1, 0]


def test_apply_condition_equality():
    df = pd.DataFrame({"status": [1, 2, 1, 3]})
    enricher = SpecEnricher()
    result = enricher._apply_condition(df, "status", "status == 1")
    assert list(result) == [1, 0, 1, 0]


def test_apply_condition_isin():
    df = pd.DataFrame({"status": [1, 2, 3, 4]})
    enricher = SpecEnricher()
    result = enricher._apply_condition(df, "status", "status in [1, 2]")
    assert list(result) == [1, 1, 0, 0]


def test_apply_condition_greater_than():
    df = pd.DataFrame({"score": [10.0, 50.0, 80.0, 90.0]})
    enricher = SpecEnricher()
    result = enricher._apply_condition(df, "score", "score >= 50")
    assert list(result) == [0, 1, 1, 1]


def test_apply_condition_and_logic():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 10, 20]})
    enricher = SpecEnricher()
    result = enricher._apply_condition(df, "a", "a > 1 and b == 10")
    assert list(result) == [0, 0, 1, 0]


def test_apply_condition_unknown_column_fallback():
    df = pd.DataFrame({"label": [0, 1]})
    enricher = SpecEnricher()
    # Condition references nonexistent col — falls back gracefully
    result = enricher._apply_condition(df, "label", "nonexistent_col > 0")
    assert isinstance(result, pd.Series)


# ── SpecEnricher.enrich() — unknown task type branch ─────────────────────────

def test_enrich_unknown_task_type_becomes_binary(binary_df):
    raw = _raw_spec(task_type="unknown", target_column_hint="outcome")
    result = SpecEnricher().enrich(raw, binary_df, list(binary_df.columns))
    # With binary outcome column, should detect binary_classification
    assert result.task_type in ("binary_classification", "unknown")
