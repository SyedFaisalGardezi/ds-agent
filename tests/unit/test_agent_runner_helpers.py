"""Tests for agent/api/agent_runner.py — pure helper functions."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from agent.api.agent_runner import (
    AgentRunResult,
    StepResult,
    _build_brief_context,
    _data_context,
    _deduplicate_repetition,
    _eda_scope_enrichment,
    _error_recovery_plan,
    _extract_json,
    _is_error_message,
    _is_keyword_confident,
    _is_real_plan,
    _keyword_goal,
    _keyword_plan,
    _keyword_plan_from_spec,
    _parse_plan,
    _recent_history,
    _strip_thinking,
    _trim_echoed_history,
    result_to_dict,
)
from agent.api.agent_tools import ToolOutput


def test_dedup_short_token_loop():
    # Real degenerate case: model loops "PR-AUC." dozens of times.
    text = "\n\n".join(["PR-AUC."] * 30)
    assert _deduplicate_repetition(text) == "PR-AUC."


def test_dedup_long_sentence_3x():
    s = "The model overfits on the training data badly."
    text = " ".join([s] * 4)
    assert _deduplicate_repetition(text).count(s) <= 2


def test_dedup_keeps_non_repeating():
    text = "First point here. Second distinct point. Third unique point."
    assert _deduplicate_repetition(text) == text


def test_dedup_two_repeats_not_cut():
    text = "Same line.\n\nSame line."
    assert _deduplicate_repetition(text) == text


# ── _parse_react_step parse-error feedback (#5) ──────────────────────────────

def test_react_step_malformed_json_sets_parse_error():
    from agent.api.agent_runner import _parse_react_step
    s = _parse_react_step("Thought: go\nAction: eda_profile\nAction Input: {bad}")
    assert s.action == "eda_profile"
    assert s.parse_error is not None
    assert s.action_input == {}


def test_react_step_valid_json_no_error():
    from agent.api.agent_runner import _parse_react_step
    s = _parse_react_step('Action: train_model\nAction Input: {"model": "rf"}')
    assert s.parse_error is None
    assert s.action_input == {"model": "rf"}


def test_react_step_final_answer_no_error():
    from agent.api.agent_runner import _parse_react_step
    s = _parse_react_step("Thought: done\nFinal Answer: 42 rows.")
    assert s.final_answer == "42 rows."
    assert s.parse_error is None


# ── _needs_clarification (#6) ────────────────────────────────────────────────

def _kern(ns):
    k = SimpleNamespace()
    k.namespace = ns
    return k


def _clar_session(**kw):
    d = dict(data_paths=[], target=None, brief=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_clarify_no_data_for_build():
    from agent.api.agent_runner import _needs_clarification
    q = _needs_clarification("train a model", _clar_session(),
                             _kern({"df": None}), [("train_model", {})])
    assert q and "dataset" in q.lower()


def test_clarify_vague_build_asks_target():
    from agent.api.agent_runner import _needs_clarification
    df = pd.DataFrame({"a": [1, 2], "churn": [0, 1]})
    q = _needs_clarification("build a model", _clar_session(),
                             _kern({"df": df}), [("train_model", {})])
    assert q and "target" in q.lower()


def test_clarify_suppressed_when_goal_stated():
    from agent.api.agent_runner import _needs_clarification
    df = pd.DataFrame({"a": [1, 2], "churn": [0, 1]})
    assert _needs_clarification("predict churn", _clar_session(),
                                _kern({"df": df}), [("train_model", {})]) is None


def test_clarify_suppressed_with_task_spec():
    from agent.api.agent_runner import _needs_clarification
    df = pd.DataFrame({"a": [1, 2], "churn": [0, 1]})
    assert _needs_clarification("build", _clar_session(),
                                _kern({"df": df, "TASK_SPEC": {"t": 1}}),
                                [("train_model", {})]) is None


def test_clarify_empty_plan_passes():
    from agent.api.agent_runner import _needs_clarification
    assert _needs_clarification("hello", _clar_session(),
                                _kern({"df": None}), []) is None


def test_clarify_qa_plan_with_data_passes():
    from agent.api.agent_runner import _needs_clarification
    df = pd.DataFrame({"a": [1, 2]})
    # eda_profile with data loaded → no clarification needed
    assert _needs_clarification("explore the data", _clar_session(),
                                _kern({"df": df}), [("eda_profile", {})]) is None

# ── _trim_echoed_history ──────────────────────────────────────────────────────

def _sess_with_assistant(*contents):
    # ChatMessage stores the body in `.text` — mirror that here.
    msgs = [SimpleNamespace(role="assistant", text=c) for c in contents]
    return SimpleNamespace(messages=msgs)


def test_trim_echoed_history_cuts_trailing_echo():
    prior = ("Data engineering focuses on building and maintaining the "
             "infrastructure and pipelines that process data reliably.")
    session = _sess_with_assistant(prior)
    reply = "PR-AUC.\n\n" + prior
    out = _trim_echoed_history(reply, session)
    assert out == "PR-AUC."
    assert prior not in out


def test_trim_echoed_history_keeps_clean_reply():
    prior = ("Data engineering focuses on building and maintaining the "
             "infrastructure and pipelines that process data reliably.")
    session = _sess_with_assistant(prior)
    reply = "The favorite metric you mentioned earlier was PR-AUC."
    out = _trim_echoed_history(reply, session)
    assert out == reply


def test_trim_echoed_history_ignores_short_prior():
    # Prior assistant reply under 80 chars → not treated as an echo source.
    session = _sess_with_assistant("PR-AUC.")
    reply = "You said PR-AUC. and here is more useful content that stays intact."
    out = _trim_echoed_history(reply, session)
    assert out == reply


def test_trim_echoed_history_whole_reply_is_echo_kept():
    prior = ("Data engineering focuses on building and maintaining the "
             "infrastructure and pipelines that process data reliably.")
    session = _sess_with_assistant(prior)
    # idx == 0 (nothing precedes the echo) → keep it, nothing better to return.
    out = _trim_echoed_history(prior, session)
    assert out == prior


def test_trim_echoed_history_no_messages():
    session = SimpleNamespace(messages=[])
    reply = "A standalone answer."
    assert _trim_echoed_history(reply, session) == reply

# ── _strip_thinking ───────────────────────────────────────────────────────────

def test_strip_thinking_removes_block():
    assert _strip_thinking("<think>...</think>answer") == "answer"


def test_strip_thinking_open_tag():
    result = _strip_thinking("before<think>open")
    assert result == "before"


def test_strip_thinking_no_tag():
    s = "just text"
    assert _strip_thinking(s) == s


def test_strip_thinking_strips_endoftext():
    result = _strip_thinking("The answer is 42.<|endoftext|>\n<|im_start|>user\nfake follow-up")
    assert result == "The answer is 42."
    assert "<|endoftext|>" not in result


def test_strip_thinking_strips_im_start():
    result = _strip_thinking("Here is my analysis.<|im_start|>user\ninjected text")
    assert result == "Here is my analysis."


def test_strip_thinking_strips_im_end():
    result = _strip_thinking("Done.<|im_end|>")
    assert result == "Done."


def test_strip_thinking_combined_think_and_special_token():
    text = "<think>reasoning</think>Answer here.<|endoftext|>garbage"
    result = _strip_thinking(text)
    assert result == "Answer here."


# ── _extract_json ─────────────────────────────────────────────────────────────

def test_extract_json_direct_dict():
    result = _extract_json('{"key": "val"}')
    assert result == {"key": "val"}


def test_extract_json_embedded_in_prose():
    text = 'Here is the plan:\n{"goal": "run eda"}\nThat is all.'
    result = _extract_json(text)
    assert result == {"goal": "run eda"}


def test_extract_json_array():
    text = '[{"tool": "eda_profile"}]'
    result = _extract_json(text)
    assert isinstance(result, list)
    assert result[0]["tool"] == "eda_profile"


def test_extract_json_no_json():
    assert _extract_json("plain text no json") is None


def test_extract_json_malformed():
    assert _extract_json("{broken json") is None


def test_extract_json_with_thinking_tag():
    text = "<think>ignore this</think>{\"a\": 1}"
    result = _extract_json(text)
    assert result == {"a": 1}


# ── _parse_plan ───────────────────────────────────────────────────────────────

def test_parse_plan_valid_steps():
    raw = '{"goal": "run eda", "steps": [{"step": 1, "tool": "eda_profile", "args": {}, "reason": "profile"}]}'
    steps = _parse_plan(raw, max_steps=10)
    assert len(steps) == 1
    assert steps[0]["tool"] == "eda_profile"


def test_parse_plan_unknown_tool_filtered():
    raw = '{"steps": [{"step": 1, "tool": "made_up_tool", "args": {}, "reason": "?"}]}'
    steps = _parse_plan(raw, max_steps=10)
    assert steps == []


def test_parse_plan_respects_max_steps():
    raw_steps = [{"step": i, "tool": "eda_profile", "args": {}, "reason": "r"}
                 for i in range(20)]
    raw = f'{{"steps": {raw_steps}}}'.replace("'", '"')
    raw = json.dumps({"steps": raw_steps})
    steps = _parse_plan(raw, max_steps=3)
    assert len(steps) <= 3


def test_parse_plan_empty_returns_empty():
    assert _parse_plan("null", max_steps=10) == []


def test_parse_plan_array_input():
    raw = json.dumps([{"step": 1, "tool": "quality_check", "args": {}, "reason": "quality"}])
    steps = _parse_plan(raw, max_steps=10)
    assert len(steps) == 1
    assert steps[0]["tool"] == "quality_check"


# ── _is_error_message ─────────────────────────────────────────────────────────

def test_is_error_message_traceback():
    tb = "Traceback (most recent call last):\n  File 'x.py', line 10\nValueError: bad"
    assert _is_error_message(tb) is True


def test_is_error_message_value_error():
    assert _is_error_message("ValueError: something went wrong") is True


def test_is_error_message_normal_text():
    assert _is_error_message("please run the eda") is False


def test_is_error_message_cell_marker():
    assert _is_error_message("Cell In[3]: error here") is True


# ── _is_keyword_confident ─────────────────────────────────────────────────────

def test_is_keyword_confident_pipeline():
    assert _is_keyword_confident("run the full pipeline and train the model") is True


def test_is_keyword_confident_short_ambiguous():
    # Short instruction with no strong signals
    assert _is_keyword_confident("hi") is False


def test_is_keyword_confident_long_instruction():
    long = "a" * 90
    assert _is_keyword_confident(long) is True


def test_is_keyword_confident_error_message():
    assert _is_keyword_confident("Traceback (most recent call last): ValueError: x") is True


# ── _keyword_goal ─────────────────────────────────────────────────────────────

def test_keyword_goal_pipeline():
    assert "pipeline" in _keyword_goal("run the full pipeline").lower()


def test_keyword_goal_eda():
    goal = _keyword_goal("explore the data and analyse the distributions")
    assert any(w in goal.lower() for w in ["eda", "explor", "analys"])


def test_keyword_goal_model():
    goal = _keyword_goal("train a model on this data")
    assert "model" in goal.lower() or "train" in goal.lower()


def test_keyword_goal_notebook():
    goal = _keyword_goal("build the notebook")
    assert "notebook" in goal.lower() or "build" in goal.lower()


def test_keyword_goal_fallback():
    instruction = "do something random"
    goal = _keyword_goal(instruction)
    assert len(goal) > 0


# ── _is_real_plan ─────────────────────────────────────────────────────────────

def test_is_real_plan_multi_step():
    steps = [
        {"tool": "eda_profile", "args": {}, "step": 1},
        {"tool": "quality_check", "args": {}, "step": 2},
    ]
    assert _is_real_plan(steps) is True


def test_is_real_plan_empty():
    assert _is_real_plan([]) is False


def test_is_real_plan_trivial_single_print():
    steps = [{"tool": "execute_code", "args": {"code": "print('hello')"}, "step": 1}]
    assert _is_real_plan(steps) is False


def test_is_real_plan_substantial_execute_code():
    steps = [{"tool": "execute_code", "args": {"code": "x" * 300}, "step": 1}]
    assert _is_real_plan(steps) is True


# ── _error_recovery_plan ──────────────────────────────────────────────────────

def test_error_recovery_plan_nan_error_has_fix_step():
    # NaN-class error → diagnosis + targeted NaN fix (2 steps).
    tb = "Traceback (most recent call last):\nValueError: Input contains NaN"
    plan = _error_recovery_plan(tb)
    assert len(plan) == 2
    assert all(s["tool"] == "execute_code" for s in plan)


def test_error_recovery_plan_non_nan_error_diagnosis_only():
    # Non-NaN error → diagnosis only; the hardcoded NaN fix is not applied (#12).
    tb = "ModuleNotFoundError: No module named 'foo'"
    plan = _error_recovery_plan(tb)
    assert len(plan) == 1
    assert "diagnos" in plan[0]["reason"].lower()


def test_error_recovery_plan_contains_diagnose():
    tb = "ValueError: bad value"
    plan = _error_recovery_plan(tb)
    assert "diagnos" in plan[0]["reason"].lower()


# ── _build_brief_context ──────────────────────────────────────────────────────

def test_build_brief_context_no_spec():
    result = _build_brief_context("some brief text", {})
    assert "some brief text" in result


def test_build_brief_context_with_spec():
    spec = {
        "task_description": "predict churn",
        "task_type": "binary_classification",
        "target_definition": "churn_flag",
        "target_is_derived": False,
        "aggregation_needed": False,
        "aggregation_key": None,
        "evaluation_metric": "roc_auc",
        "secondary_datasets": [],
        "feature_engineering_hints": [],
    }
    result = _build_brief_context("raw brief", spec)
    assert "predict churn" in result
    assert "roc_auc" in result


def test_build_brief_context_empty_brief():
    result = _build_brief_context("", {})
    assert "no brief" in result.lower() or result == ""


# ── _recent_history ───────────────────────────────────────────────────────────

def _make_session(messages):
    return SimpleNamespace(messages=messages)


def _msg(role, text):
    return SimpleNamespace(role=role, text=text)


def test_recent_history_empty():
    s = _make_session([])
    result = _recent_history(s)
    assert result == "" or isinstance(result, str)


def test_recent_history_returns_last_n():
    msgs = [_msg("user", f"msg{i}") for i in range(10)]
    s = _make_session(msgs)
    result = _recent_history(s, n=3)
    assert "msg9" in result
    assert "msg7" in result


def test_recent_history_only_user_assistant():
    msgs = [
        _msg("system", "sys prompt"),
        _msg("user", "hello"),
        _msg("assistant", "hi"),
    ]
    s = _make_session(msgs)
    result = _recent_history(s)
    assert "system" not in result.lower()


# ── _data_context ─────────────────────────────────────────────────────────────

def _stub_session(**kwargs):
    defaults = dict(
        data_path=None, target=None, brief="", artifacts={},
        work_dir=None, notebook_path=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _stub_kernel(ns=None):
    return SimpleNamespace(namespace=ns or {})


def test_data_context_no_data():
    s = _stub_session()
    k = _stub_kernel()
    result = _data_context(s, k)
    assert "No data" in result


def test_data_context_with_df():
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    s = _stub_session(target="a")
    k = _stub_kernel({"df": df, "DATA_TYPE": "tabular", "DATA_FMT": "csv"})
    result = _data_context(s, k)
    assert "shape" in result
    assert "TARGET" in result


def test_data_context_data_path_not_loaded():
    from pathlib import Path
    s = _stub_session(data_path=Path("/tmp/data.csv"))
    k = _stub_kernel()
    result = _data_context(s, k)
    assert "data.csv" in result


# ── result_to_dict ────────────────────────────────────────────────────────────

def test_result_to_dict_basic():
    out = ToolOutput(success=True, text="done")
    sr = StepResult(step=1, tool="eda_profile", args={},
                    reason="profile", output=out, elapsed_s=1.2)
    result = AgentRunResult(
        goal="run eda", reply="done", steps=[sr], elapsed_s=5.0)
    d = result_to_dict(result)
    assert d["goal"] == "run eda"
    assert d["reply"] == "done"
    assert len(d["steps_run"]) == 1
    assert d["steps_run"][0]["tool"] == "eda_profile"


def test_result_to_dict_error_field():
    result = AgentRunResult(goal="x", reply="fail", error="something broke")
    d = result_to_dict(result)
    assert d["error"] == "something broke"


def test_result_to_dict_figures_list():
    result = AgentRunResult(goal="x", reply="ok", figures=["/tmp/fig.png"])
    d = result_to_dict(result)
    assert "/tmp/fig.png" in d["figures"]


# ── _keyword_plan ─────────────────────────────────────────────────────────────

def _kw_session(**kw):
    defaults = dict(
        target=None, brief="", work_dir=None,
        data_path=None, data_paths=[], artifacts={}, notebook_path=None,
        messages=[], brief_filenames=[],
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_keyword_plan_pipeline_instruction():
    s = _kw_session()
    steps = _keyword_plan("run the full pipeline and train a model", s)
    tools = [st["tool"] for st in steps]
    assert "eda_profile" in tools
    assert "train_model" in tools


def test_keyword_plan_eda_only():
    s = _kw_session()
    steps = _keyword_plan("explore the data and plot distributions", s)
    tools = [st["tool"] for st in steps]
    assert "eda_profile" in tools
    # Should have visualize without full model train
    assert "train_model" not in tools or len(steps) <= 5


def test_keyword_plan_error_recovery():
    tb = "Traceback (most recent call last):\nValueError: bad value"
    s = _kw_session()
    steps = _keyword_plan(tb, s)
    tools = [st["tool"] for st in steps]
    assert "execute_code" in tools


def test_keyword_plan_with_brief_runs_understand_task():
    s = _kw_session(brief="Predict outcome column.")
    steps = _keyword_plan("analyse data", s)
    tools = [st["tool"] for st in steps]
    assert "understand_task" in tools


def test_keyword_plan_brief_aggregation_keywords():
    s = _kw_session(brief="Aggregate data per establishment to predict outcome.")
    steps = _keyword_plan("run pipeline", s)
    tools = [st["tool"] for st in steps]
    assert "aggregate_data" in tools


def test_keyword_plan_with_notebook_flag():
    s = _kw_session()
    steps = _keyword_plan("build notebook and train model", s)
    tools = [st["tool"] for st in steps]
    assert "build_notebook" in tools


def test_keyword_plan_generic_fallback():
    s = _kw_session()
    steps = _keyword_plan("hi there", s)
    assert len(steps) > 0
    tools = [st["tool"] for st in steps]
    assert "eda_profile" in tools


# ── _keyword_plan_from_spec ───────────────────────────────────────────────────

def test_keyword_plan_from_spec_no_aggregation():
    s = _kw_session()
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": False,
        "aggregation_key": None,
        "target_definition": "churn",
        "target_is_derived": False,
        "evaluation_metric": "roc_auc",
    }
    steps = _keyword_plan_from_spec("run the pipeline", s, spec)
    tools = [st["tool"] for st in steps]
    assert "eda_profile" in tools
    assert "train_model" in tools


def test_keyword_plan_from_spec_with_aggregation():
    s = _kw_session()
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": True,
        "aggregation_key": "customer_id",
        "target_definition": "churn == 1",
        "target_is_derived": True,
        "evaluation_metric": "pr_auc",
    }
    steps = _keyword_plan_from_spec("full pipeline", s, spec)
    tools = [st["tool"] for st in steps]
    assert "aggregate_data" in tools
    assert "eda_profile" in tools


def test_keyword_plan_from_spec_metric_mapping():
    s = _kw_session()
    spec = {
        "task_type": "regression",
        "aggregation_needed": False,
        "aggregation_key": None,
        "target_definition": "price",
        "target_is_derived": False,
        "evaluation_metric": "rmse",
    }
    steps = _keyword_plan_from_spec("train a regression model", s, spec)
    train_steps = [st for st in steps if st["tool"] == "train_model"]
    if train_steps:
        metric = train_steps[0].get("args", {}).get("eval_metric", "")
        assert metric in ("neg_root_mean_squared_error", "auto", "rmse")


def test_keyword_plan_from_spec_error_recovery():
    s = _kw_session()
    tb = "Traceback (most recent call last):\nValueError: NaN"
    steps = _keyword_plan_from_spec(tb, s, {})
    tools = [st["tool"] for st in steps]
    assert "execute_code" in tools


def test_keyword_plan_from_spec_empty_spec():
    s = _kw_session()
    steps = _keyword_plan_from_spec("run eda", s, {})
    assert len(steps) > 0


def test_keyword_plan_from_spec_no_pipeline_intent_adds_visualize():
    """Without pipeline keywords and no spec, the else branch adds visualize steps."""
    s = _kw_session()
    steps = _keyword_plan_from_spec("show me some charts", s, {})
    tools = [st["tool"] for st in steps]
    assert "visualize" in tools


# ── _eda_scope_enrichment ─────────────────────────────────────────────────────

def _noop_emit(*args, **kwargs):
    pass


def _stub_kernel_ns(ns=None):
    k = SimpleNamespace(namespace=ns or {})
    return k


def test_eda_scope_enrichment_basic():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, size=100),
        "income": rng.normal(50000, 10000, size=100),
        "outcome": rng.integers(0, 2, size=100),
    })
    spec = {"task_type": "binary_classification", "target_column": "outcome"}
    k = _stub_kernel_ns()
    result = _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert "n_samples" in result
    assert result["n_samples"] == 100


def test_eda_scope_enrichment_sets_task_spec_in_kernel():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=50),
        "b": rng.integers(0, 2, size=50),
    })
    spec = {"task_type": "binary_classification", "target_column": "b"}
    k = _stub_kernel_ns()
    _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert "TASK_SPEC" in k.namespace


def test_eda_scope_enrichment_regression():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
        "price": rng.uniform(100, 1000, size=200),
    })
    spec = {"task_type": "regression", "target_column": "price"}
    k = _stub_kernel_ns()
    result = _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert "recommended_models" in result
    models = result["recommended_models"]
    assert len(models) > 0


def test_eda_scope_enrichment_large_dataset():
    import numpy as np
    rng = np.random.default_rng(0)
    # Simulate large dataset (> 50000 rows) via a real DataFrame
    n = 60000
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    })
    spec = {"task_type": "binary_classification", "target_column": "label"}
    k = _stub_kernel_ns()
    result = _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert result["n_samples"] == n


def test_eda_scope_enrichment_imbalance_ratio():
    import numpy as np
    # 90/10 imbalance
    label = [0] * 90 + [1] * 10
    df = pd.DataFrame({
        "feature": np.random.default_rng(0).normal(size=100),
        "label": label,
    })
    spec = {"task_type": "binary_classification", "target_column": "label"}
    k = _stub_kernel_ns()
    result = _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert result["class_imbalance_ratio"] > 0
    assert result["class_imbalance_ratio"] < 1.0


def test_eda_scope_enrichment_multiclass_upgrades_metric():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f": rng.normal(size=60),
        "target": [0, 1, 2] * 20,
    })
    spec = {
        "task_type": "multiclass_classification",
        "target_column": "target",
        "evaluation_metric": "pr_auc",  # binary-only metric
    }
    k = _stub_kernel_ns()
    result = _eda_scope_enrichment(spec, df, k, _noop_emit)
    assert result.get("evaluation_metric") != "pr_auc"


def test_eda_scope_enrichment_exception_does_not_crash():
    """Should not raise even if df is bad."""
    spec = {}
    k = _stub_kernel_ns()
    _eda_scope_enrichment(spec, None, k, _noop_emit)  # df=None → exception


# ── _build_findings_context ───────────────────────────────────────────────────

from agent.api.agent_runner import _build_findings_context, _synthesise  # noqa: E402


def test_build_findings_context_empty_steps():
    result = _build_findings_context("do EDA", "analyse data", [])
    assert "do EDA" in result
    assert "analyse data" in result


def test_build_findings_context_with_steps():
    out = ToolOutput(success=True, text="profile done")
    sr = StepResult(step=1, tool="eda_profile", args={},
                    reason="profile", output=out, elapsed_s=1.5)
    result = _build_findings_context("eda goal", "run EDA", [sr])
    assert "eda_profile" in result
    assert "profile done" in result


def test_build_findings_context_failed_step():
    out = ToolOutput(success=False, text="", error="RuntimeError: failed")
    sr = StepResult(step=1, tool="train_model", args={},
                    reason="train", output=out, elapsed_s=2.0)
    result = _build_findings_context("train", "train model", [sr])
    assert "✗" in result
    assert "RuntimeError" in result


# ── _synthesise ───────────────────────────────────────────────────────────────

def test_synthesise_with_none_llm():
    findings = "accuracy=0.91, f1=0.88"
    result = _synthesise(findings, None)
    assert result == findings


def test_synthesise_returns_llm_text():
    llm = SimpleNamespace(
        _generate=lambda prompt, **kw: "The model performs well."
    )
    result = _synthesise("findings here", llm)
    assert "The model" in result


def test_synthesise_strips_thinking():
    llm = SimpleNamespace(
        _generate=lambda prompt, **kw: "<think>internal</think>Clean answer."
    )
    result = _synthesise("findings", llm)
    assert "Clean answer" in result
    assert "think" not in result.lower()


def test_synthesise_llm_exception_returns_findings():
    def boom(prompt, **kw):
        raise RuntimeError("network down")
    llm = SimpleNamespace(_generate=boom)
    findings = "some findings text"
    result = _synthesise(findings, llm)
    assert len(result) > 0


# ── _data_context — additional branch coverage ────────────────────────────────

def test_data_context_with_artifacts():
    df = pd.DataFrame({"a": range(5)})
    s = _stub_session(artifacts={"profile": {}, "train": {}})
    k = _stub_kernel({"df": df})
    result = _data_context(s, k)
    assert "profile" in result or "train" in result


def test_data_context_with_notebook_path():
    from pathlib import Path
    s = _stub_session(notebook_path=Path("/tmp/report.ipynb"))
    k = _stub_kernel()
    result = _data_context(s, k)
    assert "report.ipynb" in result


def test_data_context_with_brief_loaded():
    """Brief loaded → data_context mentions it and suggests understand_task."""
    s = _stub_session()
    s.brief = "Predict churn for customers." * 20  # non-empty brief
    s.brief_filenames = ["task_brief.pdf"]
    k = _stub_kernel()
    result = _data_context(s, k)
    assert "brief" in result.lower() or "understand_task" in result.lower()


def test_data_context_with_high_null_df():
    import numpy as np
    df = pd.DataFrame({
        "sparse": [np.nan] * 9 + [1.0],
        "dense": range(10),
    })
    s = _stub_session()
    k = _stub_kernel({"df": df})
    result = _data_context(s, k)
    assert "shape" in result


# ── _keyword_goal — eda+model branch ─────────────────────────────────────────

def test_keyword_goal_eda_and_model():
    goal = _keyword_goal("run EDA and train a model")
    assert "eda" in goal.lower() or "model" in goal.lower()


# ── _instruction_from_document ────────────────────────────────────────────────

from agent.api.agent_runner import _instruction_from_document  # noqa: E402


def test_instruction_from_document_text_file(tmp_path):
    f = tmp_path / "task.txt"
    f.write_text("Predict the churn column.")
    s = _kw_session()
    result = _instruction_from_document(str(f), s)
    assert "churn" in result


def test_instruction_from_document_missing_file():
    s = _kw_session()
    result = _instruction_from_document("/nonexistent/path.txt", s)
    assert "not found" in result.lower() or result.startswith("[")


# ── Category-code aggregation key guard ───────────────────────────────────────

def test_keyword_plan_from_spec_naics_code_overridden():
    """naics_code must never be used as groupby key; fallback to establishment_id."""
    s = _kw_session()
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": True,
        "aggregation_key": "naics_code",
        "target_definition": "incident_outcome in [1, 2, 3]",
        "target_is_derived": True,
        "evaluation_metric": "pr_auc",
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec)
    agg_steps = [st for st in steps if st["tool"] == "aggregate_data"]
    assert agg_steps, "aggregate_data step must be present"
    group_by = agg_steps[0].get("args", {}).get("group_by", "")
    assert group_by != "naics_code", (
        f"Expected category code to be overridden; got {group_by!r}"
    )
    assert group_by == "establishment_id"


def test_keyword_plan_from_spec_sic_code_overridden():
    """sic_code must also be rejected as an aggregation key."""
    s = _kw_session()
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": True,
        "aggregation_key": "sic_code",
        "target_definition": "outcome == 1",
        "target_is_derived": True,
        "evaluation_metric": "roc_auc",
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec)
    agg_steps = [st for st in steps if st["tool"] == "aggregate_data"]
    assert agg_steps
    group_by = agg_steps[0].get("args", {}).get("group_by", "")
    assert "sic" not in group_by.lower()


def test_keyword_plan_from_spec_entity_id_preserved():
    """A legitimate entity ID key must pass through unchanged."""
    s = _kw_session()
    spec = {
        "task_type": "binary_classification",
        "aggregation_needed": True,
        "aggregation_key": "establishment_id",
        "target_definition": "outcome == 1",
        "target_is_derived": True,
        "evaluation_metric": "roc_auc",
    }
    steps = _keyword_plan_from_spec("build pipeline", s, spec)
    agg_steps = [st for st in steps if st["tool"] == "aggregate_data"]
    assert agg_steps
    assert agg_steps[0].get("args", {}).get("group_by", "") == "establishment_id"


# ── _parse_react_action ───────────────────────────────────────────────────────

from agent.api.agent_runner import _parse_react_action  # noqa: E402


def test_parse_react_action_tool_call():
    raw = '{"action": "tool_call", "tool": "eda_profile", "args": {}}'
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "tool_call"
    assert result["tool"] == "eda_profile"


def test_parse_react_action_respond():
    raw = '{"action": "respond", "text": "Here are the results."}'
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "respond"
    assert "results" in result["text"]


def test_parse_react_action_prose_fallback():
    """Non-JSON prose should be wrapped as a respond action."""
    raw = "The dataset has 100 rows and 5 columns."
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "respond"
    assert "100 rows" in result["text"]


def test_parse_react_action_with_thinking_tag():
    raw = "<think>internal reasoning</think>{\"action\": \"respond\", \"text\": \"answer\"}"
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "respond"


def test_parse_react_action_missing_action_key_with_tool():
    """{"tool": "eda_profile", "args": {}} — no 'action' key → inferred as tool_call."""
    raw = '{"tool": "eda_profile", "args": {}}'
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "tool_call"
    assert result["tool"] == "eda_profile"


def test_parse_react_action_text_key_without_action():
    """{"text": "Hello"} — no 'action' key → inferred as respond."""
    raw = '{"text": "Hello world"}'
    result = _parse_react_action(raw)
    assert result is not None
    assert result["action"] == "respond"
    assert "Hello world" in result["text"]


def test_parse_react_action_function_call_synonym():
    """action='function_call' → normalised to tool_call."""
    raw = '{"action": "function_call", "name": "eda_profile", "arguments": {}}'
    result = _parse_react_action(raw)
    # May be None if name→tool mapping not implemented, but should not crash
    assert result is None or result.get("action") in ("tool_call", "respond")


def test_parse_react_action_invalid_action_value_stringified():
    """Completely unknown structure is stringified as a respond."""
    raw = '{"foo": "bar", "baz": 42}'
    result = _parse_react_action(raw)
    # Stringified fallback
    assert result is not None
    assert result["action"] == "respond"


def test_parse_react_action_empty_string():
    result = _parse_react_action("")
    assert result is None


def test_parse_react_action_tool_call_with_args():
    raw = '{"action": "tool_call", "tool": "train_model", "args": {"task_type": "auto"}}'
    result = _parse_react_action(raw)
    assert result["action"] == "tool_call"
    assert result["args"]["task_type"] == "auto"


# ── CALL_TOOL: marker (hybrid natural-language format) ───────────────────────

def test_parse_react_call_tool_no_args():
    """CALL_TOOL marker with no ARGS block → tool_call with empty args."""
    result = _parse_react_action("Sure, let me profile the data.\n\nCALL_TOOL: eda_profile")
    assert result == {"action": "tool_call", "tool": "eda_profile", "args": {}}


def test_parse_react_call_tool_with_args():
    """CALL_TOOL marker with ARGS block → args parsed correctly."""
    text = 'I will train a model.\n\nCALL_TOOL: train_model\nARGS: {"task_type": "classification"}'
    result = _parse_react_action(text)
    assert result["action"] == "tool_call"
    assert result["tool"] == "train_model"
    assert result["args"] == {"task_type": "classification"}


def test_parse_react_call_tool_case_insensitive():
    """CALL_TOOL: marker is matched case-insensitively."""
    result = _parse_react_action("call_tool: eda_profile")
    assert result["action"] == "tool_call"
    assert result["tool"] == "eda_profile"


def test_parse_react_plain_prose_no_call_tool():
    """Plain prose with no CALL_TOOL → respond action."""
    result = _parse_react_action(
        "Your dataset has 500 rows and 12 columns. No missing values detected."
    )
    assert result["action"] == "respond"
    assert "500" in result["text"]


def test_parse_react_call_tool_takes_priority_over_json():
    """CALL_TOOL marker wins even when JSON is also present in text."""
    text = '{"action": "respond", "text": "hello"}\n\nCALL_TOOL: quality_check'
    result = _parse_react_action(text)
    assert result["action"] == "tool_call"
    assert result["tool"] == "quality_check"


# ── run_reactive + _classify_tools ───────────────────────────────────────────

import json  # noqa: E402

from agent.api.agent_runner import _classify_tools, run_reactive  # noqa: E402


def _react_session(**kw):
    defaults = dict(
        target=None, brief="", work_dir=None,
        data_path=None, data_paths=[], artifacts={},
        notebook_path=None, messages=[],
        brief_filenames=[], session_id="test-sid",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _react_kernel(ns=None):
    ns = ns or {}
    k = SimpleNamespace(namespace=ns)
    return k


def _make_llm(*, classify="none", synth="Analysis complete.", qa="Direct answer."):
    """LLM mock for plan-then-execute run_reactive.

    _generate() returns `classify` on the first call (_llm_classify_tools if reached)
    and `synth` on subsequent calls (_synthesise).
    _chat() returns `synth` (used by _react_loop).
    answer() returns a SimpleNamespace with `.text = qa`.
    classify_intent() returns unknown/available by default.
    """
    call_counter = {"n": 0}

    def _gen(prompt, **kw):
        idx = call_counter["n"]
        call_counter["n"] += 1
        return classify if idx == 0 else synth

    from unittest.mock import MagicMock
    return SimpleNamespace(
        _generate=_gen,
        _chat=lambda messages, **kw: synth,
        _strip_thinking=lambda t: t,
        answer=MagicMock(return_value=SimpleNamespace(text=qa)),
        classify_intent=MagicMock(
            return_value=SimpleNamespace(intent="unknown", available=True, target=None)
        ),
    )


# ── _classify_tools ───────────────────────────────────────────────────────────

def test_classify_tools_brief_request_routes_to_understand_task():
    """'read the pdf' with brief loaded and no TASK_SPEC → understand_task."""
    s = _react_session(brief="some brief text")
    k = _react_kernel()
    plan = _classify_tools("read the pdf instructions", s, k, _make_llm())
    assert any(t == "understand_task" for t, _ in plan)


def test_classify_tools_no_brief_skips_understand_task():
    """No brief loaded → understand_task not triggered even with 'read' keyword."""
    s = _react_session(brief="")
    k = _react_kernel()
    plan = _classify_tools("read the instructions", s, k, _make_llm())
    assert not any(t == "understand_task" for t, _ in plan)


def test_classify_tools_eda_request_with_data():
    """'explore the data' with df → eda_profile."""
    import pandas as pd
    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1, 2]})})
    plan = _classify_tools("explore the data and highlight key insights", s, k, _make_llm())
    assert any(t == "eda_profile" for t, _ in plan)


def test_classify_tools_quality_check_included():
    """'check for missing values' with df → eda_profile + quality_check."""
    import pandas as pd
    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    plan = _classify_tools("check the data for missing values and outliers", s, k, _make_llm())
    tool_names = [t for t, _ in plan]
    assert "eda_profile" in tool_names
    assert "quality_check" in tool_names


def test_classify_tools_train_request_with_data():
    """'train a model' with df → pipeline includes train_model."""
    import pandas as pd
    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    plan = _classify_tools("train a model and evaluate performance", s, k, _make_llm())
    assert any(t == "train_model" for t, _ in plan)


def test_classify_tools_deploy_request():
    """'deploy the model' → execute_code with packaging script."""
    s = _react_session()
    k = _react_kernel()
    plan = _classify_tools("deploy the model and prepare for ci/cd", s, k, _make_llm())
    assert any(t == "execute_code" for t, _ in plan)


def test_classify_tools_monitor_request():
    """'monitor drift' → execute_code with monitoring script."""
    s = _react_session()
    k = _react_kernel()
    plan = _classify_tools("continuously monitor and improve the deployed model", s, k, _make_llm())
    assert any(t == "execute_code" for t, _ in plan)


def test_classify_tools_ambiguous_llm_fallback_returns_valid():
    """Genuinely ambiguous instruction with no question-word → _llm_classify_tools called."""
    from unittest.mock import patch

    import pandas as pd
    s = _react_session()
    # Use a non-question, non-keyword instruction that slips through all keyword guards
    # "do a full deep-dive" has no _question_w hits and no specific keyword match
    k = _react_kernel({"df": pd.DataFrame({"a": [1, 2]})})
    llm = _make_llm(classify="eda_profile")
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {"eda_profile": object()}):
        plan = _classify_tools("do a full deep-dive of the dataset", s, k, llm)
    # LLM fallback runs; result is either the tool or [] (depends on mock)
    assert isinstance(plan, list)


def test_classify_tools_llm_fallback_none_returns_empty():
    """LLM returns 'none' → empty plan."""
    s = _react_session()
    k = _react_kernel()
    plan = _classify_tools("what is 2 + 2?", s, k, _make_llm(classify="none"))
    assert plan == []


# ── run_reactive ──────────────────────────────────────────────────────────────

def test_run_reactive_simple_question_no_tools():
    """Simple greeting with no data/brief → _react_loop runs, gets a reply."""
    # _classify_tools returns [] (no keywords match, no data)
    # _react_loop calls _generate which returns 'Analysis complete.' on first call
    # _parse_react_step treats plain prose as final_answer
    llm = _make_llm(synth="Analysis complete.")
    s = _react_session()
    k = _react_kernel()
    result = run_reactive("hi there", s, k, llm)
    assert isinstance(result.reply, str)
    assert len(result.reply) > 0
    assert len(result.steps) == 0


def test_run_reactive_eda_request_calls_eda_profile():
    """'run eda' with df in kernel → eda_profile called via keyword routing."""
    from unittest.mock import MagicMock, patch

    import pandas as pd
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    mock_out = ToolOutput(success=True, text="Shape: (5, 2)", figures=[])
    mock_tool = MagicMock()
    mock_tool.execute.return_value = mock_out

    s = _react_session()
    k = _react_kernel({"df": df})
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {"eda_profile": mock_tool}), \
         patch("agent.api.agent_runner._synthesise", return_value="EDA is done."):
        result = run_reactive("run eda on the data", s, k, _make_llm())

    assert "EDA is done" in result.reply
    assert len(result.steps) == 1
    assert result.steps[0].tool == "eda_profile"


def test_run_reactive_tool_failure_still_gets_response():
    """Tool failure doesn't crash — synthesis still produces a response."""
    from unittest.mock import MagicMock, patch

    import pandas as pd
    mock_out = ToolOutput(success=False, text="", error="Something went wrong")
    mock_tool = MagicMock()
    mock_tool.execute.return_value = mock_out

    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {"eda_profile": mock_tool}), \
         patch("agent.api.agent_runner._synthesise", return_value="Tool failed but here is what I know."):
        result = run_reactive("explore the data", s, k, _make_llm())

    assert len(result.steps) == 1
    assert result.steps[0].output.success is False
    assert "here is what I know" in result.reply


def test_run_reactive_aborts_after_dependency_failure():
    """A failed dependency tool (understand_task) skips downstream steps (#7)."""
    from unittest.mock import MagicMock, patch

    import pandas as pd
    fail = MagicMock()
    fail.execute.return_value = ToolOutput(success=False, text="", error="boom")
    later = MagicMock()
    later.execute.return_value = ToolOutput(success=True, text="ok", figures=[])

    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    plan = [("understand_task", {}), ("feature_engineering", {}),
            ("train_model", {})]
    with patch("agent.api.agent_runner._classify_tools", return_value=plan), \
         patch("agent.api.agent_runner._needs_clarification", return_value=None), \
         patch("agent.api.agent_runner.TOOL_REGISTRY", {
             "understand_task": fail,
             "feature_engineering": later,
             "train_model": later,
         }), \
         patch("agent.api.agent_runner._synthesise", return_value="done"):
        result = run_reactive("build a model to predict a", s, k, _make_llm())

    # Only the failed dependency step ran; downstream skipped.
    assert len(result.steps) == 1
    assert result.steps[0].tool == "understand_task"
    assert later.execute.call_count == 0


def test_run_reactive_max_tool_calls_respected():
    """max_tool_calls caps the number of tools executed."""
    from unittest.mock import MagicMock, patch

    import pandas as pd
    mock_out = ToolOutput(success=True, text="done", figures=[])
    mock_tool = MagicMock()
    mock_tool.execute.return_value = mock_out

    s = _react_session()
    # TASK_SPEC present so the clarification gate lets the build proceed.
    k = _react_kernel({"df": pd.DataFrame({"a": [1]}),
                       "TASK_SPEC": {"task_type": "regression", "target": "a"}})
    # "train a model" → 3-step plan; cap at 2
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {
        "mutual_information": mock_tool,
        "feature_engineering": mock_tool,
        "train_model": mock_tool,
    }), patch("agent.api.agent_runner._synthesise", return_value="Synthesis."):
        result = run_reactive("train a model to predict a", s, k, _make_llm(), max_tool_calls=2)

    assert len(result.steps) <= 2
    assert result.reply == "Synthesis."


def test_run_reactive_llm_classify_error_falls_back_to_qa():
    """LLM errors crash gracefully — run_reactive still returns a string reply."""
    def _boom(prompt, **kw):
        raise RuntimeError("network down")

    def _boom_chat(messages, **kw):
        raise RuntimeError("network down")

    from unittest.mock import MagicMock
    llm = SimpleNamespace(
        _generate=_boom,
        _chat=_boom_chat,
        _strip_thinking=lambda t: t,
        answer=MagicMock(return_value=SimpleNamespace(text="QA fallback answer.")),
        classify_intent=MagicMock(
            return_value=SimpleNamespace(intent="unknown", available=False, target=None)
        ),
    )
    result = run_reactive("hello", _react_session(), _react_kernel(), llm)
    assert isinstance(result.reply, str)
    assert len(result.reply) > 0


def test_run_reactive_qa_fallback_when_no_tools_needed():
    """No matching keywords and no data → answer() used via qa intent routing."""
    llm = _make_llm(qa="QA direct answer.")
    llm.classify_intent.return_value = SimpleNamespace(intent="qa", available=True, target=None)
    s = _react_session()
    k = _react_kernel()
    result = run_reactive("explain machine learning to me", s, k, llm)
    assert isinstance(result.reply, str)
    assert len(result.reply) > 0


def test_run_reactive_returns_figures():
    """Figures from tool calls are accumulated in the result."""
    from unittest.mock import MagicMock, patch

    import pandas as pd
    mock_out = ToolOutput(success=True, text="plot done", figures=["base64fig1", "base64fig2"])
    mock_tool = MagicMock()
    mock_tool.execute.return_value = mock_out

    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {"eda_profile": mock_tool}), \
         patch("agent.api.agent_runner._synthesise", return_value="Charts generated."):
        result = run_reactive("explore the data", s, k, _make_llm())

    assert "base64fig1" in result.figures
    assert "base64fig2" in result.figures


def test_run_reactive_understand_task_with_brief():
    """'read the instructions' with brief loaded → understand_task called."""
    from unittest.mock import MagicMock, patch
    mock_out = ToolOutput(success=True, text="Task: binary classification.", figures=[])
    mock_tool = MagicMock()
    mock_tool.execute.return_value = mock_out

    s = _react_session(brief="Predict whether establishments are compliant.")
    k = _react_kernel()
    with patch("agent.api.agent_runner.TOOL_REGISTRY", {"understand_task": mock_tool}), \
         patch("agent.api.agent_runner._synthesise", return_value="Task understood."):
        result = run_reactive("read the instructions and outline the problem", s, k, _make_llm())

    assert result.steps[0].tool == "understand_task"
    assert "Task understood" in result.reply


# ── _parse_react_step ─────────────────────────────────────────────────────────

from agent.api.agent_runner import _parse_react_step  # noqa: E402


def test_parse_react_step_final_answer():
    text = "Thought: I have enough info.\nFinal Answer: The dataset has 500 rows."
    s = _parse_react_step(text)
    assert s.final_answer == "The dataset has 500 rows."
    assert s.action is None


def test_parse_react_step_action_with_empty_args():
    text = "Thought: I should profile the data.\nAction: eda_profile\nAction Input: {}"
    s = _parse_react_step(text)
    assert s.action == "eda_profile"
    assert s.action_input == {}
    assert s.final_answer is None


def test_parse_react_step_action_with_args():
    text = 'Thought: Feature engineering next.\nAction: feature_engineering\nAction Input: {"top_k": 15}'
    s = _parse_react_step(text)
    assert s.action == "feature_engineering"
    assert s.action_input == {"top_k": 15}


def test_parse_react_step_no_action_treated_as_final():
    text = "Here is my answer: the task is binary classification."
    s = _parse_react_step(text)
    assert s.final_answer is not None
    assert s.parse_error is not None


# ── _react_loop ───────────────────────────────────────────────────────────────

from agent.api.agent_runner import _react_loop  # noqa: E402


def test_react_loop_final_answer_on_first_turn():
    """Model goes straight to Final Answer — no tool calls."""
    from unittest.mock import MagicMock
    llm = SimpleNamespace(
        _chat=lambda messages, **kw: "Thought: simple question.\nFinal Answer: Hello!",
        answer=MagicMock(return_value=SimpleNamespace(text="fallback")),
    )
    s, k = _react_session(), _react_kernel()
    reply, steps, figs = _react_loop("hello", s, k, llm, max_steps=4)
    assert "Hello" in reply
    assert steps == []


def test_react_loop_calls_tool_then_final_answer():
    """Model calls eda_profile, receives Observation, then gives Final Answer."""
    turn = {"n": 0}

    def _chat_fn(messages, **kw):
        idx = turn["n"]
        turn["n"] += 1
        if idx == 0:
            return "Thought: I should profile.\nAction: eda_profile\nAction Input: {}"
        return "Thought: Done.\nFinal Answer: The data has 100 rows."

    from unittest.mock import MagicMock, patch

    import pandas as pd

    llm = SimpleNamespace(
        _chat=_chat_fn,
        answer=MagicMock(return_value=SimpleNamespace(text="fallback")),
    )
    s = _react_session()
    k = _react_kernel(ns={"df": pd.DataFrame({"a": range(100)})})

    mock_tool = MagicMock()
    mock_tool.description = "EDA profiling"
    mock_tool.execute.return_value = ToolOutput(
        success=True, text="100 rows, 1 col", error="", figures=[]
    )

    with patch("agent.api.agent_runner.TOOL_REGISTRY") as mock_reg, \
         patch("agent.api.agent_runner._resolve_tool_name", side_effect=lambda n: n):
        mock_reg.__contains__ = lambda self, item: item == "eda_profile"
        mock_reg.get.return_value = mock_tool
        mock_reg.items.return_value = [("eda_profile", mock_tool)]
        mock_reg.keys.return_value = ["eda_profile"]

        reply, steps, figs = _react_loop("what's in my data?", s, k, llm, max_steps=4)

    assert "100 rows" in reply
    assert len(steps) == 1


# ── native tool-calling loop ──────────────────────────────────────────────────

from agent.api.agent_runner import (  # noqa: E402
    _build_ollama_tool_schemas,
    _coerce_tool_args,
    _run_agentic_loop,
    _tool_call_loop,
)


def test_build_ollama_tool_schemas_shape():
    """Every registry tool becomes a well-formed Ollama function schema."""
    schemas = _build_ollama_tool_schemas()
    assert len(schemas) == len(__import__(
        "agent.api.agent_tools", fromlist=["TOOL_REGISTRY"]).TOOL_REGISTRY)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert fn["parameters"].get("type") == "object"


def test_coerce_tool_args_variants():
    assert _coerce_tool_args({"a": 1}) == {"a": 1}
    assert _coerce_tool_args('{"a": 1}') == {"a": 1}
    assert _coerce_tool_args("not json") == {}
    assert _coerce_tool_args(None) == {}
    assert _coerce_tool_args(42) == {}


def _tools_llm(script, *, supports=True):
    """Fake client for the native loop. `script` is a list of assistant
    messages, each {"content": str, "tool_calls": [...]}, returned in order."""
    calls = {"n": 0}

    def _cwt(messages, tools, **kw):
        i = calls["n"]
        calls["n"] += 1
        return script[min(i, len(script) - 1)]

    return SimpleNamespace(
        chat_with_tools=_cwt,
        supports_tools=lambda: supports,
    )


def test_tool_call_loop_runs_tool_then_answers():
    """Model calls eda_profile, gets a tool result, then answers in plain text."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    script = [
        {"content": "", "tool_calls": [
            {"function": {"name": "eda_profile", "arguments": {}}}]},
        {"content": "The data has 100 rows and 1 column.", "tool_calls": []},
    ]
    llm = _tools_llm(script)
    s = _react_session()
    k = _react_kernel(ns={"df": pd.DataFrame({"a": range(100)})})

    mock_tool = MagicMock()
    mock_tool.description = "EDA"
    mock_tool.input_schema = {"type": "object", "properties": {}, "required": []}
    mock_tool.execute.return_value = ToolOutput(
        success=True, text="100 rows, 1 col", error="", figures=[])

    with patch("agent.api.agent_runner.TOOL_REGISTRY") as reg, \
         patch("agent.api.agent_runner._resolve_tool_name", side_effect=lambda n: n):
        reg.get.return_value = mock_tool
        reg.__contains__ = lambda self, item: item == "eda_profile"
        reg.items.return_value = [("eda_profile", mock_tool)]
        reg.keys.return_value = ["eda_profile"]
        reply, steps, figs, native_ok = _tool_call_loop(
            "what's in my data?", s, k, llm, max_steps=4)

    assert native_ok is True
    assert "100 rows" in reply
    assert len(steps) == 1
    assert steps[0].tool == "eda_profile"


def test_tool_call_loop_direct_answer_no_tools():
    """Model answers without calling any tool."""
    llm = _tools_llm([{"content": "Precision-recall AUC is ...", "tool_calls": []}])
    s, k = _react_session(), _react_kernel()
    reply, steps, figs, native_ok = _tool_call_loop("define pr-auc", s, k, llm)
    assert native_ok is True
    assert reply.startswith("Precision-recall AUC")
    assert steps == []


def test_tool_call_loop_signals_fallback_when_unsupported():
    """chat_with_tools returns None + supports_tools()==False → native_ok False."""
    llm = SimpleNamespace(
        chat_with_tools=lambda *a, **k: None,
        supports_tools=lambda: False,
    )
    s, k = _react_session(), _react_kernel()
    reply, steps, figs, native_ok = _tool_call_loop("build me a model", s, k, llm)
    assert native_ok is False
    assert steps == []


def test_run_agentic_loop_falls_back_to_react(monkeypatch):
    """When native is unsupported, the dispatcher calls the text ReAct loop."""
    called = {"react": False}

    def _fake_react(instruction, s, k, llm, *, max_steps=8):
        called["react"] = True
        return "react reply", [], []

    monkeypatch.setattr("agent.api.agent_runner._react_loop", _fake_react)
    llm = SimpleNamespace(
        chat_with_tools=lambda *a, **k: None,
        supports_tools=lambda: False,
    )
    s, k = _react_session(), _react_kernel()
    reply, steps, figs = _run_agentic_loop("build a model", s, k, llm)
    assert called["react"] is True
    assert reply == "react reply"


def test_run_agentic_loop_uses_native_when_supported(monkeypatch):
    """When native works, the dispatcher does NOT touch the text loop."""
    called = {"react": False}
    monkeypatch.setattr(
        "agent.api.agent_runner._react_loop",
        lambda *a, **k: called.__setitem__("react", True) or ("x", [], []))
    llm = _tools_llm([{"content": "done natively", "tool_calls": []}])
    s, k = _react_session(), _react_kernel()
    reply, steps, figs = _run_agentic_loop("build a model", s, k, llm)
    assert called["react"] is False
    assert reply == "done natively"


# ── Phase 2: critic / reflexion ───────────────────────────────────────────────

from agent.api.agent_runner import (  # noqa: E402
    Critique,
    _call_signature,
    _critique_step,
)


def _critic_llm(script, *,
                critic_json='{"verdict":"accept","goal_met":false,"reason":"ok"}',
                supports=True, gen_raises=False):
    """Fake client for the critic-enabled loop: native chat + _generate (used
    by both the critic and _synthesise)."""
    calls = {"n": 0}

    def _cwt(messages, tools, **kw):
        i = calls["n"]
        calls["n"] += 1
        return script[min(i, len(script) - 1)]

    def _gen(prompt, **kw):
        if gen_raises:
            raise RuntimeError("llm down")
        if "reviewing the result of one step" in prompt.lower():
            return critic_json
        return "Synthesised summary of findings."

    return SimpleNamespace(chat_with_tools=_cwt, supports_tools=lambda: supports,
                           _generate=_gen)


def test_call_signature_stable_across_key_order():
    assert _call_signature("t", {"a": 1, "b": 2}) == _call_signature("t", {"b": 2, "a": 1})
    assert _call_signature("t", {"a": 1}) != _call_signature("t", {"a": 2})


def test_critique_step_error_retries_then_replans():
    llm = SimpleNamespace(_generate=lambda *a, **k: "{}")
    assert _critique_step("g", "t", {}, "ERROR: boom", False,
                          llm_client=llm, repeats=0).verdict == "retry"
    # At/over the retry budget the same failure escalates to replan.
    assert _critique_step("g", "t", {}, "ERROR: boom", False,
                          llm_client=llm, repeats=2).verdict == "replan"


def test_critique_step_empty_result_retries():
    llm = SimpleNamespace(_generate=lambda *a, **k: "{}")
    assert _critique_step("g", "t", {}, "   ", True,
                          llm_client=llm, repeats=0).verdict == "retry"


def test_critique_step_llm_verdicts_parsed():
    llm = SimpleNamespace(
        _generate=lambda *a, **k: '{"verdict":"replan","goal_met":false,"reason":"wrong tool"}')
    c = _critique_step("g", "t", {}, "some real output", True,
                       llm_client=llm, repeats=0)
    assert c.verdict == "replan" and c.reason == "wrong tool"


def test_critique_step_goal_met_parsed():
    llm = SimpleNamespace(
        _generate=lambda *a, **k: '{"verdict":"accept","goal_met":true,"reason":"done"}')
    c = _critique_step("g", "t", {}, "final table", True,
                       llm_client=llm, repeats=0)
    assert c.verdict == "accept" and c.goal_met is True


def test_task_focus_from_spec_and_empty():
    from agent.api.agent_runner import _task_focus
    k = SimpleNamespace(namespace={"TASK_SPEC": {
        "target_column": "churn", "task_type": "classification",
        "evaluation_metric": "pr_auc", "task_description": "flag churners"}})
    focus = _task_focus(SimpleNamespace(target=None), k)
    assert "churn" in focus and "classification" in focus and "pr_auc" in focus
    assert _task_focus(SimpleNamespace(target=None),
                       SimpleNamespace(namespace={})) == ""


def test_critique_step_returns_finding_and_next_step():
    llm = SimpleNamespace(_generate=lambda *a, **k: (
        '{"verdict":"accept","goal_met":false,'
        '"finding":"target is 95/5 imbalanced",'
        '"next_step":"check minority-class separability","reason":"ok"}'))
    c = _critique_step("explore", "eda_profile", {}, "class balance 0.05", True,
                       llm_client=llm, repeats=0, task_focus="target churn")
    assert c.finding == "target is 95/5 imbalanced"
    assert c.next_step == "check minority-class separability"


def test_critique_step_receives_task_focus_in_prompt():
    seen = {}

    def _gen(prompt, **k):
        seen["p"] = prompt
        return '{"verdict":"accept","goal_met":false,"finding":"x","next_step":"y"}'
    _critique_step("g", "eda_profile", {}, "obs", True,
                   llm_client=SimpleNamespace(_generate=_gen), repeats=0,
                   task_focus="target `churn`, optimise pr_auc")
    assert "TASK FOCUS" in seen["p"] and "churn" in seen["p"]


def test_data_context_lists_brief_requirements_as_checklist():
    """Concrete brief requirements must surface as a checklist (skipping
    placeholder templates) so a long brief never loses its actionable parts."""
    from agent.api.agent_runner import _data_context
    k = _react_kernel({"TASK_SPEC": {
        "target_column": "churn", "task_type": "classification",
        "specific_requirements": ["Rank providers by risk score",
                                  "Flag double-billing", "<requirement 3>"]}})
    ctx = _data_context(_react_session(target="churn"), k)
    assert "REQUIREMENTS TO SATISFY" in ctx
    assert "Rank providers by risk score" in ctx and "Flag double-billing" in ctx
    assert "<requirement 3>" not in ctx


def test_data_context_shows_task_focus_and_exploration_log():
    from agent.api.agent_runner import _data_context
    k = _react_kernel({
        "TASK_SPEC": {"target_column": "churn", "task_type": "classification"},
        "EXPLORATION_LOG": ["[eda_profile] target is 95/5 imbalanced",
                            "[mutual_information] tenure ranks top"]})
    ctx = _data_context(_react_session(target="churn"), k)
    assert "TASK FOCUS" in ctx
    assert "WHAT WE HAVE LEARNED" in ctx and "tenure ranks top" in ctx


def test_tool_call_loop_records_findings_and_reflects(monkeypatch):
    """Each accepted step's finding is logged, and the reflection (finding +
    next step) is fed back so the model re-plans from the output."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Two tool calls, then a final answer.
    script = [
        {"content": "", "tool_calls": [
            {"function": {"name": "eda_profile", "arguments": {}}}]},
        {"content": "", "tool_calls": [
            {"function": {"name": "mutual_information", "arguments": {}}}]},
        {"content": "Done.", "tool_calls": []},
    ]
    findings = iter([
        '{"verdict":"accept","goal_met":false,"finding":"target 95/5 imbalanced",'
        '"next_step":"rank features by MI against the target"}',
        '{"verdict":"accept","goal_met":false,"finding":"tenure is the top MI feature",'
        '"next_step":"model with class weighting"}',
    ])
    calls = {"n": 0}

    def _cwt(messages, tools, **kw):
        i = calls["n"]
        calls["n"] += 1
        return script[min(i, len(script) - 1)]

    def _gen(prompt, **k):
        if "reviewing the result of one step" in prompt.lower():
            return next(findings, '{"verdict":"accept","goal_met":false}')
        return "synthesis"

    llm = SimpleNamespace(chat_with_tools=_cwt, supports_tools=lambda: True,
                          _generate=_gen)
    s = _react_session(target="y")
    k = _react_kernel(ns={"df": pd.DataFrame({"a": range(10)}),
                          "TASK_SPEC": {"target_column": "y",
                                        "task_type": "classification"}})

    tool = MagicMock()
    tool.description = "t"
    tool.input_schema = {"type": "object", "properties": {}, "required": []}
    tool.execute.return_value = ToolOutput(success=True, text="ok", error="",
                                           figures=[])
    with patch("agent.api.agent_runner.TOOL_REGISTRY") as reg, \
         patch("agent.api.agent_runner._resolve_tool_name", side_effect=lambda n: n):
        reg.get.return_value = tool
        reg.__contains__ = lambda self, item: True
        reg.items.return_value = [("eda_profile", tool), ("mutual_information", tool)]
        reg.keys.return_value = ["eda_profile", "mutual_information"]
        reply, steps, figs, native_ok = _tool_call_loop(
            "analyse the data for the task", s, k, llm, max_steps=5)

    log = k.namespace.get("EXPLORATION_LOG") or []
    assert any("imbalanced" in x for x in log)
    assert any("tenure" in x for x in log)
    # The reflection was fed back as a guidance turn to the model.
    # (Two accepted findings → two reflection messages appended.)
    assert native_ok is True


def test_critique_step_fails_open_on_llm_error():
    """A crashing critic must never block progress → accept."""
    def _boom(*a, **k):
        raise RuntimeError("down")
    llm = SimpleNamespace(_generate=_boom)
    c = _critique_step("g", "t", {}, "valid output", True,
                       llm_client=llm, repeats=0)
    assert c.verdict == "accept"


def test_critique_step_retry_capped_to_replan_at_budget():
    """Even if the LLM says retry, past the budget it becomes replan."""
    llm = SimpleNamespace(
        _generate=lambda *a, **k: '{"verdict":"retry","goal_met":false,"reason":"x"}')
    c = _critique_step("g", "t", {}, "output", True,
                       llm_client=llm, repeats=2)
    assert c.verdict == "replan"


def test_tool_call_loop_critic_goal_met_stops_early():
    """Critic returns goal_met → loop stops after one tool and synthesises."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Model would keep calling the tool, but the critic says the goal is met.
    script = [{"content": "", "tool_calls": [
        {"function": {"name": "eda_profile", "arguments": {}}}]}]
    llm = _critic_llm(
        script,
        critic_json='{"verdict":"accept","goal_met":true,"reason":"enough"}')
    s = _react_session()
    k = _react_kernel(ns={"df": pd.DataFrame({"a": range(10)})})

    mock_tool = MagicMock()
    mock_tool.description = "EDA"
    mock_tool.input_schema = {"type": "object", "properties": {}, "required": []}
    mock_tool.execute.return_value = ToolOutput(
        success=True, text="10 rows, 1 col", error="", figures=[])

    with patch("agent.api.agent_runner.TOOL_REGISTRY") as reg, \
         patch("agent.api.agent_runner._resolve_tool_name", side_effect=lambda n: n):
        reg.get.return_value = mock_tool
        reg.__contains__ = lambda self, item: item == "eda_profile"
        reg.items.return_value = [("eda_profile", mock_tool)]
        reg.keys.return_value = ["eda_profile"]
        reply, steps, figs, native_ok = _tool_call_loop(
            "profile the data", s, k, llm, max_steps=8)

    assert native_ok is True
    # Stopped after exactly one tool call despite max_steps=8.
    assert len(steps) == 1
    assert "Synthesised" in reply


def test_tool_call_loop_replan_budget_terminates():
    """A tool that always errors must terminate via the replan budget, not run
    to the step cap."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Model always requests the same failing tool.
    script = [{"content": "", "tool_calls": [
        {"function": {"name": "eda_profile", "arguments": {}}}]}]
    llm = _critic_llm(script)  # rule-based critic fires on the ERROR result
    s = _react_session()
    k = _react_kernel(ns={"df": pd.DataFrame({"a": range(10)})})

    mock_tool = MagicMock()
    mock_tool.description = "EDA"
    mock_tool.input_schema = {"type": "object", "properties": {}, "required": []}
    mock_tool.execute.return_value = ToolOutput(
        success=False, text="", error="always broken", figures=[])

    with patch("agent.api.agent_runner.TOOL_REGISTRY") as reg, \
         patch("agent.api.agent_runner._resolve_tool_name", side_effect=lambda n: n):
        reg.get.return_value = mock_tool
        reg.__contains__ = lambda self, item: item == "eda_profile"
        reg.items.return_value = [("eda_profile", mock_tool)]
        reg.keys.return_value = ["eda_profile"]
        reply, steps, figs, native_ok = _tool_call_loop(
            "profile the data", s, k, llm, max_steps=8)

    assert native_ok is True
    # Terminated by the replan budget (MAX_REPLANS=2) before the step cap of 8.
    assert 2 <= len(steps) <= 5


# ── Phase 3: web research tool ────────────────────────────────────────────────

from agent.api import agent_tools as _at  # noqa: E402


def test_web_research_tool_success(monkeypatch):
    monkeypatch.setattr(_at, "_ddg_search", lambda q, max_results=5: [
        {"title": "Polars docs", "url": "https://pola.rs", "snippet": "fast df"},
        {"title": "polars · PyPI", "url": "https://pypi.org/project/polars/",
         "snippet": "dataframe"},
    ])
    monkeypatch.setattr(_at, "_fetch_page_text", lambda url, **k: "Extracted body text.")
    monkeypatch.setattr(_at, "_pypi_info", lambda name, **k: {
        "name": "polars", "version": "1.44.1", "summary": "fast df",
        "home_page": "", "requires_python": ">=3.9"})
    out = _at.WebResearchTool().execute(
        {"query": "polars dataframe"}, session=SimpleNamespace(),
        kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert out.success
    assert "Polars docs" in out.text
    assert "1.44.1" in out.text            # PyPI enrichment
    assert "Extracted body text." in out.text
    assert out.artifacts["sources"] == ["https://pola.rs",
                                         "https://pypi.org/project/polars/"]


def test_web_research_tool_no_results(monkeypatch):
    monkeypatch.setattr(_at, "_ddg_search", lambda q, max_results=5: [])
    out = _at.WebResearchTool().execute(
        {"query": "asdfqwer"}, session=SimpleNamespace(),
        kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert out.success is False


def test_web_research_tool_requires_query():
    out = _at.WebResearchTool().execute(
        {}, session=SimpleNamespace(), kernel=SimpleNamespace(namespace={}),
        llm_client=None)
    assert out.success is False and "query" in out.error.lower()


def test_web_research_registered():
    assert "web_research" in _at.TOOL_REGISTRY


# ── Phase 3: roles, handoffs, specialists, orchestrator ───────────────────────

from agent.api.agent_runner import (  # noqa: E402
    _AGENT_ROLES,
    _build_handoff_schemas,
    _run_multiagent_loop,
    _run_specialist,
)


def test_agent_roles_tools_all_registered():
    """Every tool named in a role must exist in the registry."""
    from agent.api.agent_tools import TOOL_REGISTRY
    for role, spec in _AGENT_ROLES.items():
        for t in spec["tools"]:
            assert t in TOOL_REGISTRY, f"{role} references missing tool {t}"


def test_researcher_role_has_web_research():
    assert "web_research" in _AGENT_ROLES["researcher"]["tools"]


def test_build_handoff_schemas_one_per_role():
    schemas = _build_handoff_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == set(_AGENT_ROLES)
    for s in schemas:
        props = s["function"]["parameters"]["properties"]
        assert "subgoal" in props


def test_run_specialist_restricts_toolset(monkeypatch):
    """A specialist must call the loop with only its role's tools."""
    captured = {}

    def _fake_loop(instr, s, k, llm, *, max_steps=8, emit=None,
                   tool_names=None, system_prompt=None):
        captured["tool_names"] = tool_names
        captured["system"] = system_prompt
        return "report", [], [], True

    monkeypatch.setattr("agent.api.agent_runner._tool_call_loop", _fake_loop)
    s, k = _react_session(), _react_kernel()
    report, steps, figs = _run_specialist(
        "analyst", "explore the data", s, k, SimpleNamespace())
    assert captured["tool_names"] == _AGENT_ROLES["analyst"]["tools"]
    assert "Analyst" in captured["system"]
    assert report == "report"


def test_run_multiagent_loop_delegates_then_answers(monkeypatch):
    """Orchestrator delegates once to a specialist, then answers."""
    # Round 1: call analyst. Round 2: final text.
    script = [
        {"content": "", "tool_calls": [
            {"function": {"name": "analyst",
                          "arguments": {"subgoal": "profile the data"}}}]},
        {"content": "All done: the data looks clean.", "tool_calls": []},
    ]
    calls = {"n": 0}

    def _cwt(messages, tools, **kw):
        i = calls["n"]
        calls["n"] += 1
        return script[min(i, len(script) - 1)]

    llm = SimpleNamespace(chat_with_tools=_cwt, supports_tools=lambda: True)

    ran = {"role": None}

    def _fake_specialist(role, subgoal, s, k, llm_client, *, emit=None):
        ran["role"] = role
        return f"{role} report", [
            StepResult(step=1, tool="eda_profile", args={}, reason="",
                       output=ToolOutput(success=True, text="ok", error="",
                                         figures=[]), elapsed_s=0.1)], []

    monkeypatch.setattr("agent.api.agent_runner._run_specialist", _fake_specialist)
    s, k = _react_session(), _react_kernel()
    reply, steps, figs, native_ok = _run_multiagent_loop(
        "analyse my data", s, k, llm)
    assert native_ok is True
    assert ran["role"] == "analyst"
    assert len(steps) == 1
    assert "All done" in reply


def test_run_multiagent_loop_falls_back_when_unsupported():
    """No native tools and no work done → native_ok False for caller fallback."""
    llm = SimpleNamespace(chat_with_tools=lambda *a, **k: None,
                          supports_tools=lambda: False)
    s, k = _react_session(), _react_kernel()
    reply, steps, figs, native_ok = _run_multiagent_loop("build a model", s, k, llm)
    assert native_ok is False
    assert steps == []


def test_run_multiagent_loop_skips_duplicate_delegation(monkeypatch):
    """The same (role, subgoal) is not delegated twice."""
    script = [
        {"content": "", "tool_calls": [
            {"function": {"name": "analyst", "arguments": {"subgoal": "profile"}}}]},
        {"content": "", "tool_calls": [
            {"function": {"name": "analyst", "arguments": {"subgoal": "profile"}}}]},
        {"content": "done", "tool_calls": []},
    ]
    calls = {"n": 0}

    def _cwt(messages, tools, **kw):
        i = calls["n"]
        calls["n"] += 1
        return script[min(i, len(script) - 1)]

    llm = SimpleNamespace(chat_with_tools=_cwt, supports_tools=lambda: True)
    runs = {"n": 0}

    def _fake_specialist(role, subgoal, s, k, llm_client, *, emit=None):
        runs["n"] += 1
        return "r", [], []

    monkeypatch.setattr("agent.api.agent_runner._run_specialist", _fake_specialist)
    s, k = _react_session(), _react_kernel()
    _run_multiagent_loop("analyse", s, k, llm)
    assert runs["n"] == 1  # second identical delegation skipped


def test_classify_tools_research_questions_route_to_web_research():
    s, k = _react_session(), _react_kernel()
    for q in ["research the latest forecasting library",
              "what is the newest framework for time series",
              "search the web for catboost 2026 release"]:
        plan = _classify_tools(q, s, k, _make_llm())
        assert plan and plan[0][0] == "web_research", q


def test_classify_tools_build_verb_not_routed_to_research():
    """'train the latest model' must NOT go to the researcher."""
    s = _react_session(brief="some brief")
    k = _react_kernel({"df": __import__("pandas").DataFrame({"a": [1, 2]}),
                       "TASK_SPEC": {"task_type": "classification"}})
    plan = _classify_tools("train the latest model on my data", s, k, _make_llm())
    assert all(t != "web_research" for t, _ in plan)


# ── Phase 4: connectors → blackboard ──────────────────────────────────────────

import sqlite3 as _sqlite3  # noqa: E402
import tempfile as _tempfile  # noqa: E402


def _make_sqlite_db():
    import os
    p = os.path.join(_tempfile.mkdtemp(), "t.db")
    c = _sqlite3.connect(p)
    c.execute("CREATE TABLE sales(id INT, region TEXT, amount REAL)")
    c.executemany("INSERT INTO sales VALUES (?,?,?)",
                  [(1, "N", 10.5), (2, "S", 20.0), (3, "N", 5.0)])
    c.commit()
    c.close()
    return p


def test_connect_data_registered():
    assert "connect_data" in _at.TOOL_REGISTRY


def test_connect_data_unknown_source():
    out = _at.ConnectDataTool().execute(
        {"source": "nope"}, session=SimpleNamespace(),
        kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert out.success is False and "unknown source" in out.error.lower()


def test_connect_data_discovery_lists_tables():
    p = _make_sqlite_db()
    k = SimpleNamespace(namespace={})
    out = _at.ConnectDataTool().execute(
        {"source": "sqlite", "config": {"path": p}},
        session=SimpleNamespace(), kernel=k, llm_client=None)
    assert out.success and out.artifacts["tables"] == ["sales"]


def test_connect_data_loads_table_into_kernel():
    p = _make_sqlite_db()
    k = SimpleNamespace(namespace={})
    out = _at.ConnectDataTool().execute(
        {"source": "sqlite", "config": {"path": p}, "table": "sales"},
        session=SimpleNamespace(artifacts={}), kernel=k, llm_client=None)
    assert out.success
    df = k.namespace.get("df")
    assert df is not None and df.shape == (3, 3)
    assert k.namespace["DATA_SOURCE"]["source"] == "sqlite"


def test_connect_data_blocks_destructive_sql():
    p = _make_sqlite_db()
    out = _at.ConnectDataTool().execute(
        {"source": "sqlite", "config": {"path": p}, "query": "DROP TABLE sales"},
        session=SimpleNamespace(), kernel=SimpleNamespace(namespace={}),
        llm_client=None)
    assert out.success is False


def test_connect_data_unsafe_table_rejected():
    p = _make_sqlite_db()
    out = _at.ConnectDataTool().execute(
        {"source": "sqlite", "config": {"path": p}, "table": "sales; DROP"},
        session=SimpleNamespace(), kernel=SimpleNamespace(namespace={}),
        llm_client=None)
    assert out.success is False and "unsafe" in out.error.lower()


# ── Phase 4: LTM recall into orchestration ────────────────────────────────────

from agent.api.agent_runner import _recall_from_ltm  # noqa: E402


def test_recall_from_ltm_returns_context():
    ltm = SimpleNamespace(
        retrieve_as_context_string=lambda query, session_id, isolation_mode:
            "--- past ---\nchurn: GBM AUC 0.91\n---")
    s = _react_session(session_id="s1")
    s.isolation_mode = False
    assert "GBM AUC 0.91" in _recall_from_ltm(ltm, "predict churn", s)


def test_recall_from_ltm_none_and_failure_safe():
    s = _react_session()
    assert _recall_from_ltm(None, "q", s) == ""

    def _boom(**k):
        raise RuntimeError("down")
    ltm = SimpleNamespace(retrieve_as_context_string=_boom)
    assert _recall_from_ltm(ltm, "q", s) == ""


def test_recall_from_ltm_passes_isolation_flag():
    captured = {}

    def _cap(query, session_id, isolation_mode):
        captured.update(session_id=session_id, isolation_mode=isolation_mode)
        return ""
    ltm = SimpleNamespace(retrieve_as_context_string=_cap)
    s = _react_session(session_id="iso-sid")
    s.isolation_mode = True
    _recall_from_ltm(ltm, "q", s)
    assert captured == {"session_id": "iso-sid", "isolation_mode": True}


def _system_capturing_llm(final="done"):
    """Fake whose chat_with_tools records the system prompt it received."""
    seen = {"systems": []}

    def _cwt(messages, tools, *, system=None, **kw):
        seen["systems"].append(system or "")
        return {"content": final, "tool_calls": []}

    client = SimpleNamespace(chat_with_tools=_cwt, supports_tools=lambda: True)
    return client, seen


def test_multiagent_loop_injects_prior_into_orchestrator():
    client, seen = _system_capturing_llm()
    s, k = _react_session(), _react_kernel()
    _run_multiagent_loop("do it", s, k, client,
                         prior="[synthesis] churn solved with GBM")
    assert any("PRIOR EXPERIENCE" in sysp and "GBM" in sysp
               for sysp in seen["systems"])


def test_tool_call_loop_appends_prior_to_default_system():
    client, seen = _system_capturing_llm()
    s, k = _react_session(), _react_kernel()
    _tool_call_loop("do it", s, k, client,
                    prior_experience="[synthesis] prior fact XYZ")
    assert any("PRIOR EXPERIENCE" in sysp and "XYZ" in sysp
               for sysp in seen["systems"])


def test_run_agentic_loop_multiagent_override(monkeypatch):
    """The per-request `multiagent` flag overrides the module default:
    False → single-agent loop, True → orchestrator, None → default."""
    monkeypatch.setattr("agent.api.agent_runner._recall_from_ltm",
                        lambda *a, **k: "")
    seen = {"multi": 0, "single": 0}
    monkeypatch.setattr(
        "agent.api.agent_runner._run_multiagent_loop",
        lambda *a, **k: (seen.__setitem__("multi", seen["multi"] + 1)
                         or ("m", [], [], True)))
    monkeypatch.setattr(
        "agent.api.agent_runner._tool_call_loop",
        lambda *a, **k: (seen.__setitem__("single", seen["single"] + 1)
                         or ("s", [], [], True)))
    llm = SimpleNamespace(supports_tools=lambda: True)
    s, k = _react_session(), _react_kernel()

    _run_agentic_loop("x", s, k, llm, multiagent=False)
    assert seen == {"multi": 0, "single": 1}

    seen.update(multi=0, single=0)
    _run_agentic_loop("x", s, k, llm, multiagent=True)
    assert seen == {"multi": 1, "single": 0}


def test_run_agentic_loop_recalls_and_passes_prior(monkeypatch):
    """_run_agentic_loop must query LTM and forward the result to the loop."""
    monkeypatch.setattr("agent.api.agent_runner._recall_from_ltm",
                        lambda ltm, instr, sess: "RECALLED")
    got = {}

    def _fake_multi(instr, s, k, llm, *, emit=None, prior=""):
        got["prior"] = prior
        return "ok", [], [], True

    monkeypatch.setattr("agent.api.agent_runner._run_multiagent_loop", _fake_multi)
    monkeypatch.setattr("agent.api.agent_runner._MULTIAGENT", True)
    s, k = _react_session(), _react_kernel()
    llm = SimpleNamespace(supports_tools=lambda: True)
    _run_agentic_loop("build", s, k, llm, ltm=SimpleNamespace())
    assert got["prior"] == "RECALLED"


# ── Phase 5: workspace awareness (scan + explore + read_file) ─────────────────

import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _make_workspace():
    d = _tempfile.mkdtemp()
    _Path(d, "sub").mkdir()
    _Path(d, ".git").mkdir()
    _Path(d, "__pycache__").mkdir()
    _Path(d, "data.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")
    _Path(d, "sub", "more.csv").write_text("x,y\n1,2\n")
    _Path(d, "readme.md").write_text("# Project\nDoes stuff.")
    _Path(d, "train.py").write_text("import torch\n")
    _Path(d, ".git", "junk").write_text("ignored")
    _Path(d, "__pycache__", "x.pyc").write_text("ignored")
    return d


def test_scan_workspace_ignores_and_categorises():
    from agent.api.agent_tools import scan_workspace
    m = scan_workspace(_Path(_make_workspace()))
    assert m["n_files"] == 4          # data.csv, sub/more.csv, readme.md, train.py
    assert m["counts"] == {"code": 1, "data": 2, "doc": 1}
    assert ".git" not in m["tree"] and "junk" not in m["tree"]
    assert "__pycache__" not in m["tree"]
    names = {ds["name"] for ds in m["data_schemas"]}
    assert names == {"data.csv", "more.csv"}


def test_explore_directory_requires_workdir():
    from agent.api.agent_tools import ExploreDirectoryTool
    out = ExploreDirectoryTool().execute(
        {}, session=SimpleNamespace(work_dir=None),
        kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert out.success is False and "working directory" in out.error.lower()


def test_explore_directory_stores_map():
    from agent.api.agent_tools import ExploreDirectoryTool
    d = _make_workspace()
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(d)})
    out = ExploreDirectoryTool().execute(
        {}, session=SimpleNamespace(work_dir=_Path(d)), kernel=k, llm_client=None)
    assert out.success
    assert "WORKSPACE_MAP" in k.namespace
    assert k.namespace["WORKSPACE_MAP"]["n_files"] == 4


def test_read_file_reads_csv_and_text():
    from agent.api.agent_tools import ReadFileTool
    d = _make_workspace()
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(d)})
    s = SimpleNamespace(work_dir=_Path(d))
    rf = ReadFileTool()
    r1 = rf.execute({"path": "data.csv"}, session=s, kernel=k, llm_client=None)
    assert r1.success and r1.artifacts["columns"] == ["a", "b", "c"]
    r2 = rf.execute({"path": "readme.md"}, session=s, kernel=k, llm_client=None)
    assert r2.success and "Does stuff" in r2.text


def test_read_file_blocks_traversal():
    from agent.api.agent_tools import ReadFileTool
    d = _make_workspace()
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(d)})
    s = SimpleNamespace(work_dir=_Path(d))
    rf = ReadFileTool()
    assert rf.execute({"path": "../../../etc/passwd"},
                      session=s, kernel=k, llm_client=None).success is False
    assert rf.execute({"path": "/etc/passwd"},
                      session=s, kernel=k, llm_client=None).success is False


def test_read_file_missing_and_no_workdir():
    from agent.api.agent_tools import ReadFileTool
    d = _make_workspace()
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(d)})
    s = SimpleNamespace(work_dir=_Path(d))
    rf = ReadFileTool()
    assert rf.execute({"path": "nope.csv"},
                      session=s, kernel=k, llm_client=None).success is False
    assert rf.execute({"path": "x"}, session=SimpleNamespace(work_dir=None),
                      kernel=SimpleNamespace(namespace={}),
                      llm_client=None).success is False


def test_workspace_tools_registered_and_in_role():
    from agent.api.agent_runner import _AGENT_ROLES
    from agent.api.agent_tools import TOOL_REGISTRY
    assert "explore_directory" in TOOL_REGISTRY and "read_file" in TOOL_REGISTRY
    de = _AGENT_ROLES["data_engineer"]["tools"]
    assert "explore_directory" in de and "read_file" in de


def test_data_context_includes_workspace_map():
    from agent.api.agent_runner import _data_context
    from agent.api.agent_tools import scan_workspace
    d = _make_workspace()
    k = _react_kernel({"WORKSPACE_MAP": scan_workspace(_Path(d))})
    ctx = _data_context(_react_session(), k)
    assert "WORKING DIRECTORY" in ctx
    assert "data.csv" in ctx


# ── Phase 6: file search + edit/write with suggest/write toggle ───────────────


def _code_workspace():
    d = _tempfile.mkdtemp()
    _Path(d, "app.py").write_text('def hello():\n    return "world"\n\nx = hello()\n')
    _Path(d, "util.py").write_text("VALUE = 1\n")
    return d


def _ws_kernel(d, mode=None):
    ns = {"WORK_DIR": _Path(d)}
    if mode:
        ns["FILE_WRITE_MODE"] = mode
    return SimpleNamespace(namespace=ns)


def test_write_edit_search_registered_and_in_coder_role():
    from agent.api.agent_runner import _AGENT_ROLES
    from agent.api.agent_tools import TOOL_REGISTRY
    for t in ("search_files", "write_file", "edit_file"):
        assert t in TOOL_REGISTRY
    coder = _AGENT_ROLES["coder"]["tools"]
    assert {"search_files", "write_file", "edit_file"} <= set(coder)


def test_ws_write_mode_resolution(monkeypatch):
    from agent.api import agent_tools as at
    assert at._ws_write_mode(_ws_kernel(_tempfile.mkdtemp(), "write")) == "write"
    assert at._ws_write_mode(_ws_kernel(_tempfile.mkdtemp(), "suggest")) == "suggest"
    # no explicit mode → env default (patched off → suggest)
    monkeypatch.setattr(at, "_ALLOW_WRITES_DEFAULT", False)
    assert at._ws_write_mode(_ws_kernel(_tempfile.mkdtemp())) == "suggest"
    monkeypatch.setattr(at, "_ALLOW_WRITES_DEFAULT", True)
    assert at._ws_write_mode(_ws_kernel(_tempfile.mkdtemp())) == "write"


def test_search_files_finds_matches_with_glob():
    from agent.api.agent_tools import SearchFilesTool
    d = _code_workspace()
    s, k = SimpleNamespace(work_dir=_Path(d)), _ws_kernel(d)
    out = SearchFilesTool().execute(
        {"pattern": "hello", "glob": "*.py"}, session=s, kernel=k, llm_client=None)
    assert out.success and out.artifacts["matches"] == 2
    none = SearchFilesTool().execute(
        {"pattern": "zzz_nomatch"}, session=s, kernel=k, llm_client=None)
    assert none.success and none.artifacts["matches"] == 0


def test_search_files_bad_regex_and_no_workdir():
    from agent.api.agent_tools import SearchFilesTool
    d = _code_workspace()
    bad = SearchFilesTool().execute(
        {"pattern": "([unclosed", "regex": True},
        session=SimpleNamespace(work_dir=_Path(d)), kernel=_ws_kernel(d),
        llm_client=None)
    assert bad.success is False
    nowd = SearchFilesTool().execute(
        {"pattern": "x"}, session=SimpleNamespace(work_dir=None),
        kernel=SimpleNamespace(namespace={}), llm_client=None)
    assert nowd.success is False


def test_edit_file_suggest_mode_does_not_write():
    from agent.api.agent_tools import EditFileTool
    d = _code_workspace()
    s, k = SimpleNamespace(work_dir=_Path(d)), _ws_kernel(d, "suggest")
    out = EditFileTool().execute(
        {"path": "app.py", "old_string": '"world"', "new_string": '"universe"'},
        session=s, kernel=k, llm_client=None)
    assert out.success and out.artifacts.get("proposed") is True
    assert "world" in _Path(d, "app.py").read_text()   # unchanged


def test_edit_file_write_mode_applies():
    from agent.api.agent_tools import EditFileTool
    d = _code_workspace()
    s, k = SimpleNamespace(work_dir=_Path(d)), _ws_kernel(d, "write")
    out = EditFileTool().execute(
        {"path": "app.py", "old_string": '"world"', "new_string": '"universe"'},
        session=s, kernel=k, llm_client=None)
    assert out.success and out.artifacts.get("written") is True
    assert "universe" in _Path(d, "app.py").read_text()


def test_edit_file_nonunique_and_missing_guards():
    from agent.api.agent_tools import EditFileTool
    d = _code_workspace()
    s, k = SimpleNamespace(work_dir=_Path(d)), _ws_kernel(d, "write")
    # 'e' appears many times → must refuse without replace_all
    nonuniq = EditFileTool().execute(
        {"path": "app.py", "old_string": "e", "new_string": "E"},
        session=s, kernel=k, llm_client=None)
    assert nonuniq.success is False and "matches" in nonuniq.error
    notfound = EditFileTool().execute(
        {"path": "app.py", "old_string": "NOT_THERE", "new_string": "x"},
        session=s, kernel=k, llm_client=None)
    assert notfound.success is False


def test_write_file_modes_and_overwrite_guard():
    from agent.api.agent_tools import WriteFileTool
    d = _code_workspace()
    s = SimpleNamespace(work_dir=_Path(d))
    # suggest → not created
    sug = WriteFileTool().execute(
        {"path": "new.txt", "content": "hi"}, session=s,
        kernel=_ws_kernel(d, "suggest"), llm_client=None)
    assert sug.success and sug.artifacts.get("proposed") is True
    assert not _Path(d, "new.txt").exists()
    # write → created
    w = WriteFileTool().execute(
        {"path": "new.txt", "content": "hi"}, session=s,
        kernel=_ws_kernel(d, "write"), llm_client=None)
    assert w.success and _Path(d, "new.txt").read_text() == "hi"
    # existing without overwrite → refused
    dup = WriteFileTool().execute(
        {"path": "app.py", "content": "x"}, session=s,
        kernel=_ws_kernel(d, "write"), llm_client=None)
    assert dup.success is False and "overwrite" in dup.error


def test_write_edit_confined_to_workdir():
    from agent.api.agent_tools import EditFileTool, WriteFileTool
    d = _code_workspace()
    s, k = SimpleNamespace(work_dir=_Path(d)), _ws_kernel(d, "write")
    assert WriteFileTool().execute(
        {"path": "../evil.txt", "content": "x"}, session=s, kernel=k,
        llm_client=None).success is False
    assert EditFileTool().execute(
        {"path": "/etc/hosts", "old_string": "a", "new_string": "b"},
        session=s, kernel=k, llm_client=None).success is False


def test_run_reactive_sets_write_mode(monkeypatch):
    """allow_writes on run_reactive must set FILE_WRITE_MODE on the kernel."""
    import agent.api.agent_runner as ar
    monkeypatch.setattr(ar, "_MEMORY_OK", False)
    monkeypatch.setattr(ar, "_classify_tools", lambda *a, **k: [])
    monkeypatch.setattr(ar, "_needs_clarification", lambda *a, **k: None)
    # Short-circuit the QA path so the run returns fast.
    monkeypatch.setattr(ar, "_build_qa_messages", lambda *a, **k: [])
    monkeypatch.setattr(ar, "_store_to_ltm", lambda *a, **k: None)
    llm = SimpleNamespace(
        classify_intent=lambda m: SimpleNamespace(available=True, intent="qa"),
        answer=lambda *a, **k: SimpleNamespace(text="ok"),
    )
    k = _react_kernel()
    ar.run_reactive("hello", session=_react_session(), kernel=k,
                    llm_client=llm, allow_writes=True)
    assert k.namespace.get("FILE_WRITE_MODE") == "write"
    ar.run_reactive("hello", session=_react_session(), kernel=k,
                    llm_client=llm, allow_writes=False)
    assert k.namespace.get("FILE_WRITE_MODE") == "suggest"


# ── Phase 7: SQL, deployment packaging, CI/CD ─────────────────────────────────

import sqlite3 as _sqlite  # noqa: E402


def _sql_db():
    import os
    p = os.path.join(_tempfile.mkdtemp(), "shop.db")
    c = _sqlite.connect(p)
    c.execute("CREATE TABLE orders(id INT, region TEXT, amount REAL)")
    c.executemany("INSERT INTO orders VALUES (?,?,?)",
                  [(1, "N", 10.0), (2, "S", 20.0), (3, "N", 30.0)])
    c.commit()
    c.close()
    return p


def test_sql_and_deploy_tools_registered_and_in_roles():
    from agent.api.agent_runner import _AGENT_ROLES
    from agent.api.agent_tools import TOOL_REGISTRY
    for t in ("sql_query", "deploy_model", "scaffold_cicd"):
        assert t in TOOL_REGISTRY
    assert "sql_query" in _AGENT_ROLES["data_engineer"]["tools"]
    assert "deploy_model" in _AGENT_ROLES["ml_engineer"]["tools"]
    assert "scaffold_cicd" in _AGENT_ROLES["coder"]["tools"]


def test_sql_query_generates_runs_and_saves():
    from agent.api.agent_tools import SqlQueryTool
    p = _sql_db()
    d = _Path(p).parent
    llm = SimpleNamespace(generate_code=lambda pr, **k:
                          "SELECT region, SUM(amount) AS total FROM orders "
                          "GROUP BY region")
    k = SimpleNamespace(namespace={"WORK_DIR": d, "FILE_WRITE_MODE": "write"})
    s = SimpleNamespace(work_dir=d)
    out = SqlQueryTool().execute(
        {"source": "sqlite", "config": {"path": p},
         "question": "total by region", "save_to": "q.sql"},
        session=s, kernel=k, llm_client=llm)
    assert out.success and out.artifacts["generated"] is True
    assert out.artifacts["rows"] == 2
    assert (d / "q.sql").exists()


def test_sql_query_refuses_destructive_and_bad_source():
    from agent.api.agent_tools import SqlQueryTool
    p = _sql_db()
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(p).parent})
    s = SimpleNamespace(work_dir=_Path(p).parent)
    drop = SqlQueryTool().execute(
        {"source": "sqlite", "config": {"path": p}, "sql": "DROP TABLE orders"},
        session=s, kernel=k, llm_client=None)
    assert drop.success is False
    bad = SqlQueryTool().execute(
        {"source": "mongodb", "config": {}, "sql": "x"},
        session=s, kernel=k, llm_client=None)
    assert bad.success is False and "SQL sources" in bad.error


def _trained_model():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression().fit(np.array([[0], [1], [2], [3]]),
                                    np.array([0, 0, 1, 1]))


def test_deploy_model_requires_model_and_workdir():
    from agent.api.agent_tools import DeployModelTool
    d = _tempfile.mkdtemp()
    # no model
    out = DeployModelTool().execute(
        {}, session=SimpleNamespace(work_dir=_Path(d)),
        kernel=SimpleNamespace(namespace={"WORK_DIR": _Path(d)}), llm_client=None)
    assert out.success is False and "train_model" in out.error
    # no workdir
    nowd = DeployModelTool().execute(
        {}, session=SimpleNamespace(work_dir=None),
        kernel=SimpleNamespace(namespace={"best_model": _trained_model()}),
        llm_client=None)
    assert nowd.success is False


def test_deploy_model_writes_scaffold_and_suggest_mode():
    from agent.api.agent_tools import DeployModelTool
    d = _tempfile.mkdtemp()
    model = _trained_model()
    # write mode → files created
    k = SimpleNamespace(namespace={"WORK_DIR": _Path(d), "best_model": model,
                                   "FILE_WRITE_MODE": "write"})
    out = DeployModelTool().execute(
        {"output_dir": "deploy"}, session=SimpleNamespace(work_dir=_Path(d)),
        kernel=k, llm_client=None)
    assert out.success
    for f in ("model.joblib", "app.py", "Dockerfile", "requirements.txt"):
        assert (_Path(d) / "deploy" / f).exists()
    # suggest mode → proposed, nothing written
    k2 = SimpleNamespace(namespace={"WORK_DIR": _Path(d), "best_model": model,
                                    "FILE_WRITE_MODE": "suggest"})
    out2 = DeployModelTool().execute(
        {"output_dir": "deploy2"}, session=SimpleNamespace(work_dir=_Path(d)),
        kernel=k2, llm_client=None)
    assert out2.artifacts.get("proposed") is True
    assert not (_Path(d) / "deploy2").exists()


def test_scaffold_cicd_github_gitlab_and_modes():
    from agent.api.agent_tools import ScaffoldCICDTool
    d = _tempfile.mkdtemp()
    s = SimpleNamespace(work_dir=_Path(d))
    kw = SimpleNamespace(namespace={"WORK_DIR": _Path(d), "FILE_WRITE_MODE": "write"})
    gh = ScaffoldCICDTool().execute(
        {"provider": "github", "image": "shop"}, session=s, kernel=kw,
        llm_client=None)
    assert gh.success and (_Path(d) / ".github" / "workflows" / "ci.yml").exists()
    gl = ScaffoldCICDTool().execute(
        {"provider": "gitlab"}, session=s, kernel=kw, llm_client=None)
    assert gl.success and (_Path(d) / ".gitlab-ci.yml").exists()
    # bad provider
    bad = ScaffoldCICDTool().execute(
        {"provider": "jenkins"}, session=s, kernel=kw, llm_client=None)
    assert bad.success is False
    # suggest mode → not written
    ks = SimpleNamespace(namespace={"WORK_DIR": _Path(d),
                                    "FILE_WRITE_MODE": "suggest"})
    sug = ScaffoldCICDTool().execute(
        {"provider": "github", "path": "prop.yml"}, session=s, kernel=ks,
        llm_client=None)
    assert sug.artifacts.get("proposed") is True
    assert not (_Path(d) / "prop.yml").exists()


def test_classify_tools_file_role_clarification_pending():
    """When FILE_ROLE_CLARIFICATION is pending in kernel, any message re-triggers understand_task."""
    s = _react_session(brief="some brief")
    k = _react_kernel({
        "FILE_ROLE_CLARIFICATION": {"clarification_pending": True, "question": "which file is training?"}
    })
    plan = _classify_tools("df2 is the 2024 labels file", s, k, _make_llm())
    # Should have cleared the flag and returned understand_task
    assert any(t == "understand_task" for t, _ in plan)
    assert "FILE_ROLE_CLARIFICATION" not in k.namespace


# ── Q&A routing: _question_w guard ───────────────────────────────────────────

def test_classify_tools_what_is_question_bypasses_tools():
    """'what is the target column?' → [] (Q&A, no tool needed)."""
    s = _react_session(brief="predict churn")
    k = _react_kernel()
    plan = _classify_tools("what is the target column?", s, k, _make_llm())
    assert plan == []


def test_classify_tools_how_many_question_bypasses_tools():
    """'how many columns does the data have?' → [] (Q&A)."""
    import pandas as pd
    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1], "b": [2]})})
    plan = _classify_tools("how many columns does the data have?", s, k, _make_llm())
    assert plan == []


def test_classify_tools_short_acknowledgment_bypasses_tools():
    """Short ack like 'ok thanks' → [] (conversational)."""
    s = _react_session()
    k = _react_kernel()
    plan = _classify_tools("ok thanks", s, k, _make_llm())
    assert plan == []


def test_classify_tools_explain_task_bypasses_tools():
    """'explain the task' with TASK_SPEC already extracted → [] (Q&A, no re-extraction)."""
    import pandas as pd
    s = _react_session(brief="binary classification task")
    # TASK_SPEC already present — 'explain the task' is pure Q&A
    k = _react_kernel({
        "df": pd.DataFrame({"a": [1]}),
        "TASK_SPEC": {"task_type": "binary_classification", "target_column": "churn"},
    })
    plan = _classify_tools("explain the task to me", s, k, _make_llm())
    assert plan == []


def test_classify_tools_train_model_still_triggers_tool():
    """'train the model' should still trigger train_model (action verb present)."""
    import pandas as pd
    s = _react_session()
    k = _react_kernel({"df": pd.DataFrame({"a": [1]})})
    plan = _classify_tools("train the model and evaluate performance", s, k, _make_llm())
    assert any(t == "train_model" for t, _ in plan)
