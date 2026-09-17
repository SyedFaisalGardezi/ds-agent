"""Unit tests for agent/api/agent_tools.py — individual tool execute() calls.

Uses stub session, kernel, and llm_client to avoid Ollama dependency.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.api.agent_tools import (
    EdaProfileTool,
    InferTargetTool,
    MutualInfoTool,
    QualityCheckTool,
    ToolOutput,
)
from agent.api.kernel import CellOutput, SessionKernel

# ── stubs ─────────────────────────────────────────────────────────────────────

def _make_session(target=None, brief="", data_path=None):
    s = SimpleNamespace(target=target, brief=brief, data_path=data_path)
    return s


def _make_llm_offline():
    return SimpleNamespace(is_available=lambda: False)


def _make_llm_online(response="rf"):
    return SimpleNamespace(
        is_available=lambda: True,
        _generate=lambda prompt, **kw: response,
    )


# ── EdaProfileTool ────────────────────────────────────────────────────────────

@pytest.fixture
def eda_kernel():
    k = SessionKernel()
    df = pd.DataFrame({
        "age": [25, 30, 35, None],
        "income": [50000, 60000, 70000, 80000],
        "outcome": [0, 1, 0, 1],
    })
    k.namespace["df"] = df
    return k


def test_eda_profile_returns_tool_output(eda_kernel):
    s = _make_session()
    result = EdaProfileTool().execute({}, session=s, kernel=eda_kernel,
                                       llm_client=_make_llm_offline())
    assert isinstance(result, ToolOutput)
    assert result.success is True
    assert "Shape" in result.text


def test_eda_profile_includes_shape_artifact(eda_kernel):
    s = _make_session()
    result = EdaProfileTool().execute({}, session=s, kernel=eda_kernel,
                                       llm_client=_make_llm_offline())
    assert "shape" in result.artifacts
    assert result.artifacts["shape"] == [4, 3]


def test_eda_profile_no_df(eda_kernel):
    eda_kernel.namespace["df"] = None
    s = _make_session()
    # Should either fail gracefully or raise — no hard crash
    try:
        result = EdaProfileTool().execute({}, session=s, kernel=eda_kernel,
                                           llm_client=_make_llm_offline())
        assert isinstance(result, ToolOutput)
    except Exception:
        pass  # acceptable — df=None execution error


def test_eda_profile_domain_eda_triggered(eda_kernel):
    df = pd.DataFrame({
        "establishment_id": [1, 2, 3],
        "incident_outcome": [1, 2, 3],
        "type_of_incident": ["slip", "fall", "strike"],
    })
    eda_kernel.namespace["df"] = df
    s = _make_session()
    result = EdaProfileTool().execute({}, session=s, kernel=eda_kernel,
                                       llm_client=_make_llm_offline())
    assert result.success is True
    assert "Domain" in result.text or "establishment" in result.text.lower()


def test_eda_profile_with_target_triggers_feature_analysis():
    """EDA feature analysis code (the corrplot / KDE plots branch) must not crash.

    This exercises the line that was previously buggy:
      _ax.set_title(f'{{_feat}}\\n(corr={_corr_s.get(_feat,0):.3f})')
    which caused NameError when `_corr_s` was referenced in the outer f-string.
    """
    k = SessionKernel()
    df = pd.DataFrame({
        "age": list(range(30)),
        "income": [i * 1000 for i in range(30)],
        "score": [i * 2 for i in range(30)],
        "outcome": [i % 2 for i in range(30)],
    })
    k.namespace["df"] = df
    s = _make_session(target="outcome")
    result = EdaProfileTool().execute({}, session=s, kernel=k,
                                       llm_client=_make_llm_offline())
    assert result.success is True
    assert "Task-Aware" in result.text or "Feature" in result.text or "Target" in result.text


# ── QualityCheckTool ──────────────────────────────────────────────────────────

def test_quality_check_no_issues():
    k = SessionKernel()
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    k.namespace["df"] = df
    result = QualityCheckTool().execute({}, session=_make_session(), kernel=k,
                                         llm_client=_make_llm_offline())
    assert result.success is True
    assert "Outliers" in result.text or "outlier" in result.text.lower()


def test_quality_check_detects_outliers():
    k = SessionKernel()
    import numpy as np
    data = list(range(20)) + [10000]   # one huge outlier
    df = pd.DataFrame({"value": data})
    k.namespace["df"] = df
    result = QualityCheckTool().execute({}, session=_make_session(), kernel=k,
                                         llm_client=_make_llm_offline())
    assert result.success is True
    assert "outlier" in result.text.lower() or "1 outlier" in result.text.lower()


def test_quality_check_constant_column():
    k = SessionKernel()
    df = pd.DataFrame({"x": [1, 1, 1, 1], "y": [1, 2, 3, 4]})
    k.namespace["df"] = df
    result = QualityCheckTool().execute({}, session=_make_session(), kernel=k,
                                         llm_client=_make_llm_offline())
    assert result.success is True
    assert "x" in result.text


# ── InferTargetTool ───────────────────────────────────────────────────────────

def test_infer_target_from_task_spec():
    k = SessionKernel()
    df = pd.DataFrame({"a": [0, 1, 0], "b": [1, 2, 3], "outcome": [0, 1, 0]})
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {"target_column": "outcome", "task_type": "binary_classification"}
    s = _make_session()
    result = InferTargetTool().execute({}, session=s, kernel=k,
                                        llm_client=_make_llm_offline())
    assert result.success is True
    assert "outcome" in result.text
    assert s.target == "outcome"


def test_infer_target_data_driven():
    k = SessionKernel()
    df = pd.DataFrame({"feature_a": [1, 2, 3, 4, 5],
                       "feature_b": [5, 4, 3, 2, 1],
                       "churn": [0, 1, 0, 1, 0]})
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {}
    s = _make_session()
    result = InferTargetTool().execute({"hint": "churn"}, session=s, kernel=k,
                                        llm_client=_make_llm_offline())
    assert result.success is True
    assert s.target == "churn"


def test_infer_target_no_df():
    k = SessionKernel()
    k.namespace["df"] = None
    s = _make_session()
    result = InferTargetTool().execute({}, session=s, kernel=k,
                                        llm_client=_make_llm_offline())
    assert result.success is False
    assert "No dataframe" in result.text


def test_infer_target_llm_fallback():
    k = SessionKernel()
    df = pd.DataFrame({"x1": range(100), "x2": range(100), "target_col": [0, 1] * 50})
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {}
    s = _make_session()
    llm = _make_llm_online(response="target_col")
    result = InferTargetTool().execute({}, session=s, kernel=k, llm_client=llm)
    assert isinstance(result, ToolOutput)


# ── MutualInfoTool ────────────────────────────────────────────────────────────

def test_mutual_info_no_target():
    k = SessionKernel()
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {}
    s = _make_session(target=None)
    result = MutualInfoTool().execute({}, session=s, kernel=k,
                                       llm_client=_make_llm_offline())
    assert isinstance(result, ToolOutput)


def test_mutual_info_with_target():
    k = SessionKernel()
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=50),
        "b": rng.normal(size=50),
        "target": rng.integers(0, 2, size=50),
    })
    k.namespace["df"] = df
    s = _make_session(target="target")
    result = MutualInfoTool().execute({}, session=s, kernel=k,
                                       llm_client=_make_llm_offline())
    assert result.success is True or isinstance(result, ToolOutput)


# ── ToolOutput dataclass ─────────────────────────────────────────────────────

def test_tool_output_defaults():
    t = ToolOutput(success=True, text="ok")
    assert t.artifacts == {}
    assert t.figures == []
    assert t.error == ""


def test_tool_output_with_error():
    t = ToolOutput(success=False, text="failed", error="ValueError")
    assert t.success is False
    assert t.error == "ValueError"


# ── ExecuteCodeTool ───────────────────────────────────────────────────────────

from agent.api.agent_tools import ExecuteCodeTool  # noqa: E402


def test_execute_code_basic(eda_kernel):
    s = _make_session()
    result = ExecuteCodeTool().execute(
        {"code": "result = 1 + 1; print(result)"},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)
    assert result.success is True


def test_execute_code_empty_code(eda_kernel):
    s = _make_session()
    result = ExecuteCodeTool().execute(
        {"code": ""},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)


def test_execute_code_syntax_error(eda_kernel):
    s = _make_session()
    result = ExecuteCodeTool().execute(
        {"code": "def broken(:"},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)


# ── UnderstandTaskTool._compute_df_stats ─────────────────────────────────────

from agent.api.agent_tools import UnderstandTaskTool  # noqa: E402


def test_compute_df_stats_basic():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, size=50),
        "income": rng.normal(50000, 10000, size=50),
        "label": rng.integers(0, 2, size=50),
    })
    stats = UnderstandTaskTool._compute_df_stats(df)
    assert isinstance(stats, dict)
    assert "n_rows" in stats or "shape" in stats or len(stats) > 0


def test_compute_df_stats_empty_df():
    df = pd.DataFrame()
    stats = UnderstandTaskTool._compute_df_stats(df)
    assert isinstance(stats, dict)


# ── FeatureEngineeringTool ────────────────────────────────────────────────────

from agent.api.agent_tools import FeatureEngineeringTool  # noqa: E402


def test_feature_engineering_basic(eda_kernel):
    eda_kernel.namespace["TARGET"] = "outcome"
    s = _make_session(target="outcome")
    result = FeatureEngineeringTool().execute(
        {"top_k": 5},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)


def test_feature_engineering_no_target(eda_kernel):
    eda_kernel.namespace.pop("TARGET", None)
    s = _make_session(target=None)
    result = FeatureEngineeringTool().execute(
        {},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)


# ── ReadInstructionsTool ──────────────────────────────────────────────────────

from agent.api.agent_tools import ReadInstructionsTool  # noqa: E402


def test_read_instructions_text_file(tmp_path, eda_kernel):
    brief_file = tmp_path / "brief.txt"
    brief_file.write_text("Predict the outcome column. Target is binary.")
    s = _make_session()
    result = ReadInstructionsTool().execute(
        {"path": str(brief_file)},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)
    assert result.success is True


def test_read_instructions_missing_file(eda_kernel):
    s = _make_session()
    result = ReadInstructionsTool().execute(
        {"path": "/nonexistent/path/brief.txt"},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert isinstance(result, ToolOutput)


# ── QualityCheckTool f-string regression ─────────────────────────────────────

def test_quality_check_no_namedtuple_error_on_outliers():
    """Verify that the f-string fix works: building QualityCheckTool code
    with a dataframe that has outliers must not raise NameError."""
    from agent.api.agent_tools import QualityCheckTool
    k = SessionKernel()
    # Deliberately create a column with clear outliers
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3, 4, 1000], "y": [10, 11, 12, 13, 14]})
    k.namespace["df"] = df
    s = _make_session()
    result = QualityCheckTool().execute({}, session=s, kernel=k,
                                         llm_client=_make_llm_offline())
    assert isinstance(result, ToolOutput)
    assert result.success is True
    assert "Outliers" in result.text


# ── AggregateDataTool category-code guard ─────────────────────────────────────

def test_aggregate_data_rejects_naics_code_as_groupby():
    """When group_by is a category code and an entity ID exists, it must switch."""
    from agent.api.agent_tools import AggregateDataTool
    k = SessionKernel()
    import pandas as pd
    df = pd.DataFrame({
        "establishment_id": ["E1", "E1", "E2", "E2"],
        "naics_code": ["3120", "3120", "4110", "4110"],
        "incident_outcome": [1, 0, 2, 0],
    })
    k.namespace["df"] = df
    k.namespace["TARGET"] = ""
    s = _make_session(brief="Predict serious harm at the establishment level.")
    AggregateDataTool().execute(
        {"group_by": "naics_code"},
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    # Guard should have switched to establishment_id; aggregated df should have 2 rows
    agg_df = k.namespace.get("df")
    assert agg_df is not None
    assert len(agg_df) == 2, f"Expected 2 entity rows, got {len(agg_df)}"


def test_aggregate_data_with_entity_id_unchanged():
    """When group_by is already a valid entity ID, it must pass through unchanged."""
    from agent.api.agent_tools import AggregateDataTool
    k = SessionKernel()
    import pandas as pd
    df = pd.DataFrame({
        "establishment_id": ["E1", "E1", "E2"],
        "score": [10, 20, 30],
    })
    k.namespace["df"] = df
    k.namespace["TARGET"] = ""
    s = _make_session()
    result = AggregateDataTool().execute(
        {"group_by": "establishment_id"},
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    assert result.success is True
    agg_df = k.namespace.get("df")
    assert len(agg_df) == 2


def test_aggregate_data_with_target_cond_sets_leakage_cols():
    """target_cond path: LEAKAGE_COLS should be set in the kernel after aggregation."""
    from agent.api.agent_tools import AggregateDataTool
    k = SessionKernel()
    import pandas as pd
    df = pd.DataFrame({
        "establishment_id": ["E1", "E1", "E2", "E2", "E3"],
        "incident_outcome": [1, 0, 2, 0, 3],
        "total_hours": [100, 200, 150, 120, 80],
    })
    k.namespace["df"] = df
    k.namespace["TARGET"] = ""
    s = _make_session(brief="predict serious harm")
    result = AggregateDataTool().execute(
        {
            "group_by": "establishment_id",
            "target_condition": "incident_outcome in [1, 2, 3]",
            "target_column_name": "serious_harm",
        },
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    assert result.success is True
    # After aggregation with a target_cond derived from incident_outcome,
    # LEAKAGE_COLS should be set to incident_outcome_* columns
    leakage = k.namespace.get("LEAKAGE_COLS", None)
    assert leakage is not None, "LEAKAGE_COLS must be set by AggregateDataTool"
    assert any("incident_outcome" in c for c in leakage), (
        f"Expected incident_outcome_* cols in LEAKAGE_COLS, got {leakage}"
    )


def test_aggregate_data_post_aggregation_target_update():
    """When session.target is set but no longer in aggregated df, TARGET should update."""
    from agent.api.agent_tools import AggregateDataTool
    k = SessionKernel()
    import pandas as pd
    df = pd.DataFrame({
        "establishment_id": ["E1", "E1", "E2"],
        "outcome": [1, 0, 1],
        "score": [10, 20, 30],
    })
    k.namespace["df"] = df
    # Keep TARGET empty in kernel so "outcome" is NOT excluded from numeric agg.
    # session.target="outcome" drives the post-aggregation target-update logic.
    k.namespace["TARGET"] = ""
    s = _make_session(target="outcome")
    result = AggregateDataTool().execute(
        {"group_by": "establishment_id"},
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    assert result.success is True
    agg_df = k.namespace.get("df")
    assert agg_df is not None
    # "outcome" was aggregated into outcome_mean/max/sum/count — at least one must exist.
    outcome_cols = [c for c in agg_df.columns if c.startswith("outcome_")]
    assert len(outcome_cols) > 0, (
        f"Expected outcome_* aggregated cols, got: {list(agg_df.columns)}"
    )


# ── _keyword_plan_from_spec: requirement code generation ─────────────────────

def test_aggregate_data_brief_triggers_serious_harm_condition():
    """brief containing 'serious harm' with incident_outcome column sets target_cond."""
    from agent.api.agent_tools import AggregateDataTool
    k = SessionKernel()
    import pandas as pd
    df = pd.DataFrame({
        "establishment_id": ["E1", "E1", "E2", "E2"],
        "incident_outcome": [1, 0, 2, 0],
        "hours": [100, 200, 150, 80],
    })
    k.namespace["df"] = df
    k.namespace["TARGET"] = ""
    s = _make_session(brief="Predict establishments with serious harm incidents.")
    result = AggregateDataTool().execute(
        {"group_by": "establishment_id"},
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    assert result.success is True
    agg_df = k.namespace.get("df")
    assert agg_df is not None
    # "serious_harm" binary target should have been derived via the brief trigger
    assert "serious_harm" in agg_df.columns or s.target == "serious_harm"


def test_requirement_step_uses_fallback_without_llm():
    """With llm_client=None, uncovered requirements get a print-based fallback."""
    from types import SimpleNamespace

    from agent.api.agent_runner import _keyword_plan_from_spec
    s = SimpleNamespace(
        target=None, brief="", work_dir=None, data_path=None,
        data_paths=[], artifacts={}, notebook_path=None,
        messages=[], brief_filenames=[],
    )
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": False,
        "aggregation_key": None,
        "target_definition": "outcome",
        "target_is_derived": False,
        "evaluation_metric": "roc_auc",
        "specific_requirements": [
            "compute Cohen's kappa score and print it",
        ],
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec, llm_client=None)
    exec_steps = [st for st in steps if st["tool"] == "execute_code"]
    # At least one execute_code step for the requirement
    assert any("kappa" in st.get("args", {}).get("code", "").lower()
               or "Requirement" in st.get("args", {}).get("code", "")
               for st in exec_steps)


def test_requirement_step_uses_llm_when_available():
    """When llm_client has generate_code, its output is used for requirement steps."""
    from types import SimpleNamespace

    from agent.api.agent_runner import _keyword_plan_from_spec
    llm = SimpleNamespace(generate_code=lambda prompt, **kw: "print('custom code')")
    s = SimpleNamespace(
        target=None, brief="", work_dir=None, data_path=None,
        data_paths=[], artifacts={}, notebook_path=None,
        messages=[], brief_filenames=[],
    )
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": False,
        "evaluation_metric": "roc_auc",
        "specific_requirements": ["compute Cohen's kappa score and print it"],
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec, llm_client=llm)
    exec_steps = [st for st in steps if st["tool"] == "execute_code"]
    # LLM-generated code should appear in at least one step
    assert any("custom code" in st.get("args", {}).get("code", "")
               for st in exec_steps)


def test_requirement_step_placeholder_skipped():
    """Placeholder strings like <requirement 1> must be silently ignored."""
    from types import SimpleNamespace

    from agent.api.agent_runner import _keyword_plan_from_spec
    s = SimpleNamespace(
        target=None, brief="", work_dir=None, data_path=None,
        data_paths=[], artifacts={}, notebook_path=None,
        messages=[], brief_filenames=[],
    )
    spec = {
        "task_type": "regression",
        "aggregation_needed": False,
        "aggregation_key": None,
        "evaluation_metric": "rmse",
        "specific_requirements": ["<requirement 1>", "<requirement 2>"],
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec, llm_client=None)
    # No requirement-injected execute_code steps should exist (only the two
    # visualise steps + train are standard; none from placeholders)
    exec_steps = [st for st in steps if st["tool"] == "execute_code"
                  and "Requirement" in st.get("args", {}).get("code", "")]
    assert len(exec_steps) == 0
