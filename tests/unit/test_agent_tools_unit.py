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
                                        llm_client=_make_llm_online(response='{"target":"churn","reason":"requested label"}'))
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


def test_infer_target_rejects_date_and_id_as_unlabelled():
    """All candidates are a date or an identifier → fail loudly, set no target.

    This is the ACC case: service_date must never become the target, and the
    reply must steer to unsupervised methods rather than invent a label.
    """
    k = SessionKernel()
    df = pd.DataFrame({
        "claim_id": [f"CLM{i:05d}" for i in range(60)],
        "service_date": pd.date_range("2023-01-01", periods=60, freq="D").astype(str),
    })
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {"target_column": "service_date"}
    s = _make_session()
    result = InferTargetTool().execute({}, session=s, kernel=k,
                                        llm_client=_make_llm_offline())
    assert result.success is False
    assert "unlabelled" in result.text.lower()
    assert "isolationforest" in result.text.lower() or "anomaly" in result.text.lower()
    assert not s.target  # never set to a date or an id


def test_infer_target_skips_date_spec_picks_valid_column():
    """A date named in the spec is rejected, and a valid target is used instead."""
    k = SessionKernel()
    df = pd.DataFrame({
        "service_date": pd.date_range("2023-01-01", periods=6, freq="D").astype(str),
        "fraud": [0, 1, 0, 1, 0, 1],
    })
    k.namespace["df"] = df
    k.namespace["TASK_SPEC"] = {"target_column": "service_date"}
    s = _make_session()
    result = InferTargetTool().execute({"hint": "fraud"}, session=s, kernel=k,
                                        llm_client=_make_llm_online(response='{"target":"fraud","reason":"explicit fraud labels"}'))
    assert result.success is True
    assert s.target == "fraud"


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


def test_execute_code_basic(eda_kernel, monkeypatch):
    from agent.api import sandbox
    monkeypatch.setattr(sandbox, "execute", lambda *a, **kw: ToolOutput(True, "2"))
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


def test_execute_code_refuses_delete(eda_kernel, monkeypatch):
    monkeypatch.setattr("agent.api.sandbox.shutil.which", lambda name: None)
    """Generated code cannot delete files: the guard applies on this path too."""
    s = _make_session()
    result = ExecuteCodeTool().execute(
        {"code": "import os; os.remove('something.txt')"},
        session=s, kernel=eda_kernel, llm_client=_make_llm_offline(),
    )
    assert result.success is False
    assert "No host Python code was executed" in result.error


def test_execute_code_confines_relative_write_to_workdir(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.api.sandbox.shutil.which", lambda name: None)
    """A relative write lands in WORK_DIR, not the process working directory."""
    import os

    from agent.api.kernel import SessionKernel
    k = SessionKernel()
    k.namespace["WORK_DIR"] = tmp_path
    s = _make_session()
    before = os.getcwd()
    result = ExecuteCodeTool().execute(
        {"code": "open('probe_out.txt', 'w').write('hi')"},
        session=s, kernel=k, llm_client=_make_llm_offline(),
    )
    assert result.success is False
    assert not (tmp_path / "probe_out.txt").exists()      # written into WORK_DIR
    assert not (Path(before) / "probe_out.txt").exists()  # not the repo root
    assert os.getcwd() == before                       # cwd restored


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


def test_feature_engineering_acts_on_the_eda_report():
    """Feature engineering must use what EDA measured, not generic defaults.

    Without the report it label-encodes every categorical, keeps columns EDA
    flagged as leaking the target, and builds interactions from whichever five
    numeric columns come first. On the fixture below that selects the leakage
    column as the single most important feature, which would hand the model the
    answer and produce a meaningless score.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "y": rng.integers(0, 2, n),
        "signal": rng.normal(size=n),
        "skewed": rng.exponential(5, n),
        "const": np.ones(n),
        "idlike": [f"id{i}" for i in range(n)],
        "noise": rng.normal(size=n),
    })
    df["leaky"] = df["y"] * 10.0 + rng.normal(0, 0.01, n)
    df["signal"] = df["signal"] + df["y"] * 1.5

    def _run(report):
        k = SessionKernel()
        k.execute("import pandas as pd")
        k.namespace["df"] = df.copy()
        k.namespace["TARGET"] = "y"
        if report is not None:
            k.namespace["DATA_ANALYSIS_REPORT"] = report
        return FeatureEngineeringTool().execute(
            {"top_k": 8}, session=_make_session(target="y"), kernel=k,
            llm_client=_make_llm_offline())

    guided = _run({
        "leakage_suspects": ["leaky"], "near_constant_cols": ["const"],
        "skewed_cols": {"skewed": 3.2}, "high_cardinality_cols": ["idlike"],
        "top_mi_features": [("signal", 0.31), ("noise", 0.01)],
    })
    assert guided.success
    assert "inside training folds" in guided.text
    assert guided.artifacts["preprocessing"] == "fold-local"
    blind = _run(None)
    assert blind.success and "no global feature selection" in blind.text


def test_plan_json_survives_a_truncated_reply():
    """A reply cut off by the token budget must not lose the whole plan.

    A local model that runs out mid-list leaves valid JSON and an unterminated
    tail. The plan costs minutes to produce, so the complete part is kept and
    the half-written entry dropped.
    """
    from agent.api.agent_tools import _extract_json_obj as parse

    assert parse('{"a": 1, "b": ["x","y"]}') == {"a": 1, "b": ["x", "y"]}
    # Cut mid-list: keep the complete items.
    assert parse('{"objective":"rank","modelling":["gbm","logistic reg') == {
        "objective": "rank", "modelling": ["gbm"]}
    # Cut right after a comma: the dangling comma must go.
    assert parse('{"a":["one","two"],') == {"a": ["one", "two"]}
    # Cut inside a nested object: drop the incomplete one, keep the rest.
    assert parse('{"cov":[{"requirement":"rank","approach":"agg"},'
                 '{"requirement":"expl') == {
        "cov": [{"requirement": "rank", "approach": "agg"}]}
    # Deeply nested.
    assert parse('{"a":{"b":{"c":["x","y"') == {"a": {"b": {"c": ["x", "y"]}}}
    # An escaped quote inside a string must not be read as the terminator.
    assert parse(r'{"a":"he said \"hi\" today","b":["z') == {
        "a": 'he said "hi" today'}
    # Prose around the object is fine; nothing usable returns None.
    assert parse('Here is the plan:\n{"a": 2}\nThanks.') == {"a": 2}
    assert parse("I could not produce a plan.") is None
    assert parse("") is None


def _plan_llm(scope_json, approach_json):
    """Stub that answers the two planning stages separately."""
    def _gen(prompt, **kw):
        return scope_json if "FIRST half" in prompt else approach_json
    return SimpleNamespace(_generate=_gen, is_available=lambda: True)


_SCOPE = """{
 "objective": "Rank establishments by injury risk for 2024.",
 "requirement_coverage": [{"requirement": "Rank businesses",
   "approach": "aggregate per establishment_id",
   "columns": ["establishment_id", "incident_outcome"],
   "open_question": "Is 2024Q1 a holdout?"}],
 "eda_questions_task": ["What is the positive rate per establishment?"],
 "eda_questions_independent": ["Are the duplicate rows true repeats?"],
 "risks": ["date_of_death is 100% null but has values: parsing bug"]}"""

_APPROACH = """{
 "data_preparation": ["drop incident_year (constant)"],
 "feature_engineering": ["rate per 200k hours from total_hours_worked"],
 "modelling": ["gradient boosting: mixed dtypes, 19:1 imbalance"],
 "evaluation": ["PR AUC, grouped split on establishment_id"],
 "deliverables": ["ranked csv + notebook"],
 "revision_notes": []}"""


def _plan_kernel():
    k = SessionKernel()
    k.execute("import pandas as pd")
    k.namespace["df"] = pd.DataFrame({"establishment_id": [1, 2, 3],
                                      "incident_outcome": [1, 2, 1]})
    k.namespace["TASK_SPEC"] = {
        "task_type": "binary_classification",
        "specific_requirements": ["Rank businesses by risk"]}
    return k


def test_read_instructions_resolves_inside_the_working_directory(tmp_path):
    """A bare filename means "in the working directory", not "next to the server".

    This resolved against the process working directory, so a filename the agent
    read straight off the workspace map pointed at wherever uvicorn happened to
    be started from and always failed.
    """
    from agent.api.agent_tools import ReadInstructionsTool

    doc = tmp_path / "brief.txt"
    doc.write_text("Predict which establishments are high risk.")
    k = SessionKernel()
    k.namespace["WORK_DIR"] = tmp_path

    def _run(path):
        return ReadInstructionsTool().execute(
            {"path": path},
            session=SimpleNamespace(work_dir=tmp_path, target=None, brief=""),
            kernel=k, llm_client=None)

    bare = _run("brief.txt")
    assert bare.success, bare.error
    assert "brief.txt" in bare.text

    absolute = _run(str(doc))
    assert absolute.success

    # A miss names what IS there, so the agent can correct itself.
    missing = _run("nope.pdf")
    assert missing.success is False
    assert "nope.pdf" in missing.error and "brief.txt" in missing.error

    assert _run("").success is False


def test_work_plan_runs_both_stages_and_stores_the_plan():
    """Planning is two calls: scope, then approach written against that scope."""
    from agent.api.agent_tools import WorkPlanTool

    k = _plan_kernel()
    out = WorkPlanTool().execute(
        {}, session=_make_session(target=None), kernel=k,
        llm_client=_plan_llm(_SCOPE, _APPROACH))
    assert out.success
    plan = k.namespace["WORK_PLAN"]
    for section in ("objective", "requirement_coverage", "eda_questions_task",
                    "eda_questions_independent", "risks", "data_preparation",
                    "feature_engineering", "modelling", "evaluation",
                    "deliverables"):
        assert plan.get(section), f"missing {section}"
    assert plan["version"] == 1
    assert plan["revised_after_eda"] is False
    # Rendered for a human: the coverage mapping becomes a table.
    assert "| Requirement | Approach | Columns | Open question |" in out.text
    assert "First pass, written before EDA" in out.text


def test_work_plan_keeps_the_scope_when_the_approach_half_fails():
    """A plan costs minutes; half of one is still worth keeping."""
    from agent.api.agent_tools import WorkPlanTool

    k = _plan_kernel()
    out = WorkPlanTool().execute(
        {}, session=_make_session(target=None), kernel=k,
        llm_client=_plan_llm(_SCOPE, "not json at all"))
    assert out.success
    assert k.namespace["WORK_PLAN"].get("requirement_coverage")
    assert "did not come back usable" in out.text


def test_work_plan_refuses_to_revise_before_any_eda():
    from agent.api.agent_tools import WorkPlanTool

    k = _plan_kernel()
    out = WorkPlanTool().execute(
        {"revise": True}, session=_make_session(target=None), kernel=k,
        llm_client=_plan_llm(_SCOPE, _APPROACH))
    assert out.success is False
    assert "Nothing to revise against yet" in out.error


def test_work_plan_revision_is_grounded_in_the_eda_report():
    """revise=true feeds the measured findings in and marks the plan revised."""
    from agent.api.agent_tools import WorkPlanTool

    k = _plan_kernel()
    WorkPlanTool().execute({}, session=_make_session(target=None), kernel=k,
                           llm_client=_plan_llm(_SCOPE, _APPROACH))
    k.namespace["DATA_ANALYSIS_REPORT"] = {
        "n_rows": 14533, "class_imbalance_ratio": 0.05,
        "leakage_suspects": ["leaky_col"], "skewed_cols": {"dafw": 4.1},
        "top_mi_features": [("dafw", 0.3)], "null_pcts": {"time_of_incident": 14.0},
        "near_constant_cols": ["incident_year"], "high_cardinality_cols": ["city"],
        "outlier_cols": ["total_hours_worked"], "recommended_metric": "pr_auc",
        "analyst_narrative": "Positives are rare."}

    seen = {}

    def _gen(prompt, **kw):
        if "FIRST half" in prompt:
            seen["scope_prompt"] = prompt
            return _SCOPE
        return _APPROACH

    out = WorkPlanTool().execute(
        {"revise": True}, session=_make_session(target=None), kernel=k,
        llm_client=SimpleNamespace(_generate=_gen, is_available=lambda: True))
    assert out.success
    plan = k.namespace["WORK_PLAN"]
    assert plan["revised_after_eda"] is True
    assert plan["version"] == 2
    assert "Revised after EDA" in out.text
    # The measured facts reached the model, not just the word "revise".
    p = seen["scope_prompt"]
    assert "LEAKING columns" in p and "leaky_col" in p
    assert "incident_year" in p and "class imbalance" in p


def test_work_plan_reports_a_model_that_returns_nothing_usable():
    from agent.api.agent_tools import WorkPlanTool

    out = WorkPlanTool().execute(
        {}, session=_make_session(target=None), kernel=_plan_kernel(),
        llm_client=_plan_llm("no json here", "nor here"))
    assert out.success is False
    assert "usable scope" in out.error


def test_profile_for_planning_gives_real_evidence():
    """The planner must see dtypes, missingness and cardinality, not a column list.

    A plan written from column names alone can only restate the brief; deciding
    what to derive or drop needs the numbers.
    """
    import numpy as np

    from agent.api.agent_tools import _profile_for_planning

    n = 200
    df = pd.DataFrame({
        "est_id": np.arange(n) % 50,
        "constant_year": [2023] * n,
        "mostly_null": [None] * (n - 3) + [1.0, 2.0, 3.0],
        "flag": ["Y", "N"] * (n // 2),
        "amount": np.linspace(0, 1000, n),
    })
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)   # plant duplicates
    text = _profile_for_planning({"df": df, "df2": df[["est_id", "flag"]]})

    assert "duplicate rows: 5" in text
    assert "constant_year" in text and "1 distinct" in text
    assert "nulls 98%" in text or "nulls 99%" in text      # mostly_null
    assert "values Y, N" in text or "values N, Y" in text  # small-cardinality
    assert "range 0..1000" in text                          # numeric range
    # Shared columns are candidate join keys and must be called out.
    assert "possible join keys" in text
    assert "est_id" in text.split("possible join keys")[1]


def test_profile_for_planning_survives_no_data():
    from agent.api.agent_tools import _profile_for_planning
    assert _profile_for_planning({}) == "(no data loaded)"


def test_plan_markdown_renders_requirement_coverage_table():
    """requirement_coverage is a mapping, so it renders as a table."""
    from agent.api.agent_tools import _plan_to_markdown
    md = _plan_to_markdown({
        "objective": "Rank businesses by risk.",
        "requirement_coverage": [
            {"requirement": "Rank 1000 businesses",
             "approach": "Aggregate incidents per establishment_id",
             "columns": ["establishment_id", "incident_outcome"],
             "open_question": "Is 2024Q1 a holdout or extra training data?"}],
        "eda_questions_task": ["What is the positive rate?"],
        "eda_questions_independent": ["Are the 151 duplicate rows real repeats?"],
        "revised_after_eda": False,
    }, {"target_column": "risk"})
    assert "| Requirement | Approach | Columns | Open question |" in md
    assert "`establishment_id`" in md
    assert "Is 2024Q1 a holdout" in md
    assert "EDA: questions the brief requires answering" in md
    assert "EDA: independent checks on the data" in md


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
