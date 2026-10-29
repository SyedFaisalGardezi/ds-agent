"""Unit tests for agent/api/task_extractor.py (Layer 2)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.api.doc_parser import DocumentParser
from agent.api.task_extractor import RawSpec, TaskExtractor

EXTRACTOR = TaskExtractor(
    ollama_base_url="http://localhost:11434",
    model="qwen3.6:35b",
    timeout=10.0,
)

SIMPLE_BRIEF = "Predict churn for each customer. Use ROC-AUC."
COLUMNS = ["customer_id", "age", "tenure", "churn"]


def _make_doc(text: str):
    return DocumentParser.parse_text(text, file_type="text")


# ── _safe_json ────────────────────────────────────────────────────────────────

def test_safe_json_direct_parse():
    result = EXTRACTOR._safe_json('{"task_type": "binary_classification", "confidence": 0.9}')
    assert result["task_type"] == "binary_classification"
    assert result["confidence"] == 0.9


def test_safe_json_strips_markdown_fences():
    text = '```json\n{"metric": "pr_auc"}\n```'
    result = EXTRACTOR._safe_json(text)
    assert result.get("metric") == "pr_auc"


def test_safe_json_first_brace_block():
    text = 'Some preamble text {"target_column": "outcome", "target_condition": ""} extra'
    result = EXTRACTOR._safe_json(text)
    assert result.get("target_column") == "outcome"


def test_safe_json_key_by_key_fallback():
    text = '"task_type": "regression", "confidence": 0.8}'  # missing opening {
    result = EXTRACTOR._safe_json(text)
    assert result.get("task_type") == "regression"


def test_safe_json_empty_returns_empty_dict():
    assert EXTRACTOR._safe_json("") == {}


# ── think-block stripping ─────────────────────────────────────────────────────

def test_llm_call_strips_closed_think_block():
    raw_response = '<think>long reasoning here</think>{"metric": "roc_auc"}'
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": raw_response}}
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_resp):
        result = EXTRACTOR._llm_call("dummy prompt", max_tokens=50)

    assert "<think>" not in result
    assert "roc_auc" in result


def test_llm_call_strips_unclosed_think_tail():
    raw_response = '{"metric": "pr_auc"}<think>reasoning that never closes'
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": raw_response}}
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=mock_resp):
        result = EXTRACTOR._llm_call("dummy prompt", max_tokens=50)

    assert "<think>" not in result
    assert "pr_auc" in result


def test_llm_call_returns_empty_on_connection_error():
    import httpx
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        result = EXTRACTOR._llm_call("dummy prompt")
    assert result == ""


# ── extract() ────────────────────────────────────────────────────────────────

def _mock_post_factory(responses: list[dict]):
    """Returns a mock for httpx.post that cycles through response dicts."""
    call_count = [0]

    def mock_post(*args, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        m = MagicMock()
        m.json.return_value = {"message": {"content": json.dumps(responses[idx])}}
        m.raise_for_status.return_value = None
        return m

    return mock_post


def test_extract_happy_path():
    # extract() now makes 5 httpx.post calls (in order):
    #   0: _call_problem_scope (thinking pass)
    #   1: _call_task_type
    #   2: _call_target
    #   3: _call_aggregation
    #   4: _call_specific_requirements
    # _call_metric is skipped — _scan_explicit_metric matches "ROC-AUC" in the brief.
    responses = [
        {"analysis": "binary churn prediction per customer"},         # 0: scope
        {"task_type": "binary_classification", "confidence": 0.9, "description": "Predict churn"},  # 1
        {"target_column": "churn", "target_condition": "", "reasoning": "explicit"},                # 2
        {"aggregation_needed": False, "aggregation_key": "", "reasoning": "row level"},             # 3
        {"requirements": []},                                                                        # 4
    ]
    doc = _make_doc(SIMPLE_BRIEF)
    df_stats = {"n_rows": 1000, "n_cols": 4, "columns": {}}

    with patch("httpx.post", side_effect=_mock_post_factory(responses)):
        spec = EXTRACTOR.extract(doc, df_stats, COLUMNS)

    assert isinstance(spec, RawSpec)
    assert spec.task_type == "binary_classification"
    assert spec.target_column_hint == "churn"
    assert spec.evaluation_metric == "roc_auc"
    assert spec.source == "llm_decomposed"


def test_extract_all_calls_fail_returns_failed_spec():
    import httpx
    doc = _make_doc(SIMPLE_BRIEF)
    df_stats = {"n_rows": 100, "columns": {}}

    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        spec = EXTRACTOR.extract(doc, df_stats, COLUMNS)

    assert isinstance(spec, RawSpec)
    # source stays "llm_decomposed" but fields are defaults
    assert spec.task_type in ("unknown", "binary_classification", "regression",
                               "multiclass_classification", "")


# ── _parse_scope_analysis ─────────────────────────────────────────────────────

_WORKSAFE_SCOPE = """\
Some reasoning happened here.

--- SCOPE ANALYSIS ---
BUSINESS_GOAL: Help inspectors target high-risk establishments
ENTITY_UNIT: per establishment
TASK_TYPE: binary_classification
TASK_TYPE_REASON: "predict which businesses will have a serious harm event"
TARGET_COLUMN: DERIVE: incident_outcome in [1, 2, 3] → 1 else 0
AGGREGATION: needed
AGGREGATION_KEY: establishment_id
METRIC: pr_auc
SECONDARY_FILE: yes
SECONDARY_FILE_REASON: target comes from incidents_2024_q1.csv
SPECIFIC_STEPS: risk_ranking | output_predictions_csv | leakage_guard
--- END SCOPE ---
"""

_REGRESSION_SCOPE = """\
--- SCOPE ANALYSIS ---
BUSINESS_GOAL: Predict property sale prices
ENTITY_UNIT: per property
TASK_TYPE: regression
TASK_TYPE_REASON: sale_price is a continuous numeric target
TARGET_COLUMN: sale_price
AGGREGATION: not_needed
AGGREGATION_KEY: none
METRIC: rmse
SECONDARY_FILE: no
SECONDARY_FILE_REASON: n/a
SPECIFIC_STEPS: log_transform_target | feature_importance
--- END SCOPE ---
"""


def test_parse_scope_analysis_binary():
    priors = EXTRACTOR._parse_scope_analysis(_WORKSAFE_SCOPE)
    assert priors["TASK_TYPE"] == "binary_classification"
    assert priors["AGGREGATION"] == "needed"
    assert priors["AGGREGATION_KEY"] == "establishment_id"
    assert priors["METRIC"] == "pr_auc"
    assert priors["SECONDARY_FILE"] is True
    assert "DERIVE:" in priors["TARGET_COLUMN"]
    assert len(priors["SPECIFIC_STEPS"]) == 3


def test_parse_scope_analysis_regression():
    priors = EXTRACTOR._parse_scope_analysis(_REGRESSION_SCOPE)
    assert priors["TASK_TYPE"] == "regression"
    assert priors["AGGREGATION"] == "not_needed"
    assert "AGGREGATION_KEY" not in priors  # "none" value is skipped
    assert priors["METRIC"] == "rmse"
    assert priors["SECONDARY_FILE"] is False


def test_parse_scope_analysis_empty_returns_empty_dict():
    assert EXTRACTOR._parse_scope_analysis("") == {}
    assert EXTRACTOR._parse_scope_analysis("no structured block here") == {}


def test_parse_scope_task_type_prior_skips_llm():
    """When scope commits to binary, _call_task_type should return it without LLM."""
    with patch("httpx.post") as mock_post:
        task_type, conf, _ = EXTRACTOR._call_task_type(
            "some context", prior="binary_classification"
        )
    # prior used — httpx.post should NOT have been called
    mock_post.assert_not_called()
    assert task_type == "binary_classification"
    assert conf == 0.90


def test_parse_scope_aggregation_prior_skips_llm():
    """When scope commits to aggregation + key, _call_aggregation skips LLM."""
    with patch("httpx.post") as mock_post:
        needed, key = EXTRACTOR._call_aggregation(
            "some context",
            ["establishment_id", "naics_code", "state"],
            prior_needed="needed",
            prior_key="establishment_id",
        )
    mock_post.assert_not_called()
    assert needed is True
    assert key == "establishment_id"


def test_parse_scope_aggregation_prior_overrides_category_code():
    """If scope priors give a category code as key but entity ID exists, override it."""
    with patch("httpx.post") as mock_post:
        needed, key = EXTRACTOR._call_aggregation(
            "some context",
            ["establishment_id", "naics_code", "state"],
            prior_needed="needed",
            prior_key="naics_code",   # category code — should be overridden
        )
    mock_post.assert_not_called()
    assert needed is True
    assert key == "establishment_id"  # overridden to entity ID


def test_parse_scope_metric_prior_skips_llm():
    """When scope commits to a metric, _call_metric skips LLM."""
    with patch("httpx.post") as mock_post:
        metric = EXTRACTOR._call_metric(
            "no explicit metric phrase here",
            "binary_classification",
            prior="pr_auc",
        )
    mock_post.assert_not_called()
    assert metric == "pr_auc"


def test_parse_scope_secondary_file_prior():
    result = EXTRACTOR._requires_secondary_file("irrelevant text", prior=True)
    assert result is True
    result = EXTRACTOR._requires_secondary_file("irrelevant text", prior=False)
    assert result is False


# ── _call_file_roles ──────────────────────────────────────────────────────────

def _make_role_response(roles: dict, clarification=None) -> str:
    """Build a JSON string like the LLM would return."""
    data = {}
    for var, role in roles.items():
        data[var] = {"role": role, "confidence": 0.9, "reason": "test"}
    data["clarification_needed"] = clarification
    return json.dumps(data)


def test_call_file_roles_returns_map_when_confident():
    """When LLM returns high-confidence roles, file_role_map is populated."""
    doc = _make_doc("Derive the binary target from 2024 Q1 data.")
    df_stats = {
        "n_rows": 14533, "columns": {"incident_id": {}, "establishment_id": {}},
        "filename": "training_2023.csv",
        "secondary_data_files": {
            "df2": {
                "n_rows": 3800, "n_cols": 5,
                "columns": ["establishment_id", "outcome", "date"],
                "filename": "q1_2024.csv",
            }
        },
    }
    response_json = _make_role_response(
        {"df": "training_data", "df2": "target_derivation_source"}
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": response_json}}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_resp):
        result = EXTRACTOR._call_file_roles(doc, df_stats, user_hint="")

    assert result["df"]["role"] == "training_data"
    assert result["df2"]["role"] == "target_derivation_source"
    assert result["clarification_needed"] is None


def test_call_file_roles_returns_clarification_when_uncertain():
    """When LLM sets clarification_needed, it is returned verbatim."""
    doc = _make_doc("Build a model.")
    df_stats = {
        "n_rows": 5000, "columns": {"id": {}, "value": {}},
        "filename": "data_a.csv",
        "secondary_data_files": {
            "df2": {
                "n_rows": 3000, "n_cols": 3,
                "columns": ["id", "label"],
                "filename": "data_b.csv",
            }
        },
    }
    low_conf_data = {
        "df":  {"role": "unknown", "confidence": 0.4, "reason": "unclear"},
        "df2": {"role": "unknown", "confidence": 0.4, "reason": "unclear"},
        "clarification_needed": "Which file is training and which is labels?",
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": json.dumps(low_conf_data)}}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_resp):
        result = EXTRACTOR._call_file_roles(doc, df_stats, user_hint="")

    assert "clarification_needed" in result
    assert result["clarification_needed"] is not None
    assert len(result["clarification_needed"]) > 10


def test_extract_single_file_skips_role_call():
    """With only one file (no secondary_data_files), _call_file_roles is never called."""
    doc = _make_doc(SIMPLE_BRIEF)
    df_stats = {"n_rows": 100, "columns": {"customer_id": {}, "churn": {}}}

    task_type_json = json.dumps(
        {"task_type": "binary_classification", "confidence": 0.9, "description": "x"}
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": task_type_json}}
    mock_resp.raise_for_status = MagicMock()

    with patch.object(EXTRACTOR, "_call_file_roles") as mock_roles, \
         patch("httpx.post", return_value=mock_resp):
        spec = EXTRACTOR.extract(doc, df_stats, COLUMNS)

    mock_roles.assert_not_called()
    assert spec.file_role_map == {}


def test_extract_multi_file_with_user_hint():
    """user_hint is passed through to _call_file_roles."""
    doc = _make_doc("Derive target from 2024 data.")
    df_stats = {
        "n_rows": 500, "columns": {"id": {}},
        "secondary_data_files": {
            "df2": {"n_rows": 200, "n_cols": 2, "columns": ["id", "label"], "filename": "df2.csv"}
        },
    }
    with patch.object(EXTRACTOR, "_call_file_roles", return_value={
        "df": {"role": "training_data", "confidence": 0.9, "reason": ""},
        "df2": {"role": "target_derivation_source", "confidence": 0.9, "reason": ""},
        "clarification_needed": None,
    }) as mock_roles, \
         patch("httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "{}"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        EXTRACTOR.extract(doc, df_stats, ["id", "label"], user_hint="df2 is 2024 labels")

    call_kwargs = mock_roles.call_args
    assert call_kwargs is not None
    # user_hint is 3rd positional or keyword arg
    hint = call_kwargs.args[2] if len(call_kwargs.args) > 2 else call_kwargs.kwargs.get("user_hint", "")
    assert hint == "df2 is 2024 labels"


def test_parse_scope_none_answers_are_not_used_as_values():
    scope = """--- SCOPE ANALYSIS ---
TASK_TYPE: anomaly_detection
TARGET_COLUMN: none
AGGREGATION: none
AGGREGATION_KEY: none
METRIC: none
SPECIFIC_STEPS: none
--- END SCOPE ---"""
    priors = EXTRACTOR._parse_scope_analysis(scope)
    assert priors == {"TASK_TYPE": "anomaly_detection", "AGGREGATION": "not_needed"}


def test_risk_wording_does_not_force_binary_classification():
    """The scope's reading of the brief decides; 'high risk' is not a label."""
    with patch.object(EXTRACTOR, "_llm_call", side_effect=AssertionError("no call needed")):
        task_type, _, _ = EXTRACTOR._call_task_type(
            "Identify high risk payments; at least one provider may be at risk of fraud.",
            prior="anomaly_detection")
    assert task_type == "anomaly_detection"


def test_scope_prompt_has_no_example_deliverables_to_copy():
    prompt = EXTRACTOR._SCOPE_PROMPT
    assert "shap_values" not in prompt and "risk_ranking" not in prompt
    assert "do not leave blanks" not in prompt


def test_unlabelled_task_metric_and_declined_aggregation_are_not_guessed():
    with patch.object(EXTRACTOR, "_llm_call", side_effect=AssertionError("no guess")):
        assert EXTRACTOR._call_metric("find unusual providers", "anomaly_detection",
                                      prior="pr_auc") == ""
        assert EXTRACTOR._call_aggregation("ctx", ["claim_id", "provider_id"],
                                           prior_needed="not_needed") == (False, "")


# ── read_brief: the brief in its own terms, one call ──────────────────────────

_ACC_LIKE_REPLY = json.dumps({
    "objective": "Find payments to providers that suggest fraud.",
    "task_nature": "exploratory",
    "questions": ["Undertake an exploratory analysis and summarise key findings.",
                  "Identify populations of payments that present a risk.",
                  "How could findings identify future instances?",
                  "What recommendations would you make to each stakeholder?"],
    "deliverables": ["Summary of findings", "Code and working"],
    "audiences": ["data analyst", "fraud investigator", "senior decision maker"],
    "constraints": ["Disclose any AI use and the prompts used"],
    "data_notes": ["paid_amount: amount the insurer paid"],
    "task_type": "exploratory_analysis",
    "target_column": "",
    "aggregation_key": "Provider_ID",
    "evaluation_metric": "",
    "open_questions": [],
})


def _stats():
    return {"n_rows": 100, "columns": {"provider_id": {"dtype": "object", "n_unique": 20,
                                                        "null_rate": 0.0}}}


def test_read_brief_keeps_the_briefs_questions_and_audiences():
    from agent.api.task_extractor import read_brief
    llm = MagicMock()
    llm._generate.return_value = "```json\n" + _ACC_LIKE_REPLY + "\n```"
    spec = read_brief(llm, "brief text", _stats(), ["provider_id", "paid_amount"])
    assert spec.source == "brief_model" and spec.task_type == "exploratory_analysis"
    assert len(spec.brief["questions"]) == 4
    assert spec.brief["audiences"][1] == "fraud investigator"
    assert spec.specific_requirements[:4] == spec.brief["questions"]
    assert "Deliver: Summary of findings" in spec.specific_requirements
    assert spec.aggregation_key == "provider_id" and spec.aggregation_needed   # matched case-insensitively
    assert spec.target_column_hint == "" and spec.evaluation_metric == ""
    assert "brief text" in llm._generate.call_args.args[0]


@pytest.mark.parametrize("reply", ["not json", json.dumps({"objective": "x", "questions": []}),
                                   json.dumps({"questions": ["q"]})])
def test_read_brief_returns_none_when_unusable(reply):
    from agent.api.task_extractor import read_brief
    llm = MagicMock(); llm._generate.return_value = reply
    assert read_brief(llm, "brief", _stats(), ["provider_id"]) is None


def test_read_brief_drops_columns_that_do_not_exist():
    from agent.api.task_extractor import read_brief
    reply = json.loads(_ACC_LIKE_REPLY)
    reply.update(target_column="fraud_flag", aggregation_key="clinic", evaluation_metric="PR-AUC",
                 task_type="binary_classification")
    llm = MagicMock(); llm._generate.return_value = json.dumps(reply)
    spec = read_brief(llm, "brief", _stats(), ["provider_id"])
    assert spec.target_column_hint == "" and not spec.aggregation_needed
    assert spec.evaluation_metric == "pr_auc"


def test_understand_task_uses_the_brief_reading_and_keeps_questions():
    from types import SimpleNamespace

    import pandas as pd

    from agent.api.agent_tools import UnderstandTaskTool
    from agent.api.sessions import Session
    session = Session("s")
    session.brief = "Exercise brief"
    df = pd.DataFrame({"provider_id": ["P1", "P2"] * 10, "paid_amount": range(20)})
    kernel = SimpleNamespace(namespace={"df": df})
    llm = MagicMock()
    llm.is_available.return_value = True
    llm._generate.return_value = _ACC_LIKE_REPLY
    with patch("agent.api.task_extractor.TaskExtractor.extract",
               side_effect=AssertionError("fallback not needed")):
        out = UnderstandTaskTool().execute({}, session=session, kernel=kernel, llm_client=llm)
    spec = kernel.namespace["TASK_SPEC"]
    assert out.success and spec["task_type"] == "exploratory_analysis"
    assert spec["target_column"] == "" and spec["evaluation_metric"] == ""
    assert spec["brief"]["questions"][1].startswith("Identify populations")
    assert "Questions the brief asks" in out.text and "fraud investigator" in out.text
    assert spec["is_valid"], spec["validation_failures"]
