"""Integration test: OSHA-like incident prediction scenario.

Replicates the exact failure mode from the OSHA run log:
  - Dataset has multiple incident rows per establishment
  - Brief says: aggregate to establishment level, target = serious injury
  - Without the fix: infer_target picks 'establishment_type' (group key column)
  - With the fix: pipeline correctly sets aggregation + correct target
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agent.api.doc_parser import DocumentParser
from agent.api.spec_enricher import SpecEnricher, TaskSpecValidator
from agent.api.task_extractor import RawSpec, TaskExtractor

OSHA_BRIEF = """\
Task Description
================
Predict whether an establishment will have a serious injury or illness
(outcome in [1, 2, 3]) in the next 12 months.

Each row in the dataset represents one inspection incident.
Aggregate incidents to the establishment level using establishment_id
before building the model.

Evaluation
==========
Use PR-AUC as the primary evaluation metric.
"""


@pytest.fixture
def osha_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 800
    est_ids = rng.integers(1, 100, n)   # 100 establishments, ~8 rows each
    return pd.DataFrame({
        "establishment_id": est_ids,
        "establishment_type": rng.choice(["food", "retail", "warehouse", "factory"], n),
        "industry_code": rng.integers(100, 999, n),
        "inspection_date": pd.date_range("2019-01-01", periods=n, freq="D"),
        "total_hours_worked": rng.integers(100, 10000, n),
        "outcome": rng.integers(0, 5, n),   # 0=no injury, 1-3=serious, 4=other
    })


def _mock_extractor_responses():
    """Returns a mock httpx.post that serves deterministic LLM answers."""
    call_count = [0]
    responses = [
        # Call 1: task type
        {"task_type": "binary_classification", "confidence": 0.92,
         "description": "Predict whether an establishment will have a serious injury"},
        # Call 2: target column
        {"target_column": "outcome", "target_condition": "outcome in [1, 2, 3]",
         "reasoning": "serious injury defined as outcome in 1,2,3"},
        # Call 3: aggregation
        {"aggregation_needed": True, "aggregation_key": "establishment_id",
         "reasoning": "each row is an incident; model should predict risk per establishment"},
        # Call 4: metric
        {"metric": "pr_auc", "reasoning": "explicitly requested in brief"},
    ]

    def mock_post(*args, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        m = MagicMock()
        m.json.return_value = {"response": json.dumps(responses[idx])}
        m.raise_for_status.return_value = None
        return m

    return mock_post


def test_osha_pipeline_produces_correct_spec(osha_df):
    columns = list(osha_df.columns)

    # Layer 1: parse brief
    doc = DocumentParser.parse_text(OSHA_BRIEF, file_type="text")

    # Layer 2: extract with mocked LLM (no Ollama needed)
    extractor = TaskExtractor("http://localhost:11434", "qwen3.6:35b", timeout=5.0)
    df_stats = {
        "n_rows": len(osha_df),
        "n_cols": len(columns),
        "columns": {
            col: {
                "dtype": str(osha_df[col].dtype),
                "n_unique": int(osha_df[col].nunique()),
                "null_rate": float(osha_df[col].isna().mean()),
                "value_counts": dict(osha_df[col].value_counts().head(6).to_dict())
                    if osha_df[col].nunique() <= 10 else {},
            }
            for col in columns
        },
    }
    with patch("httpx.post", side_effect=_mock_extractor_responses()):
        raw_spec = extractor.extract(doc, df_stats, columns)

    # Layer 3: enrich
    enriched = SpecEnricher().enrich(raw_spec, osha_df, columns)

    # Layer 4: validate
    is_valid, failures = TaskSpecValidator().validate(enriched, columns)

    # ── Assertions: the OSHA failure must not recur ───────────────────────
    assert enriched.target_column != "establishment_type", (
        "establishment_type is a group-key column, not a prediction target"
    )
    assert enriched.target_column == "outcome", (
        f"Expected target='outcome', got '{enriched.target_column}'"
    )
    assert enriched.aggregation_needed is True, (
        "Brief explicitly says aggregate to establishment level"
    )
    assert enriched.aggregation_key == "establishment_id", (
        f"Expected agg_key='establishment_id', got '{enriched.aggregation_key}'"
    )
    assert enriched.evaluation_metric == "pr_auc", (
        f"Brief explicitly requests PR-AUC, got '{enriched.evaluation_metric}'"
    )
    assert is_valid is True, f"Spec should be valid but has failures: {failures}"
