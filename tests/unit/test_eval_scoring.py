"""The evaluation's scoring and cases, checked without a model."""
from __future__ import annotations

from types import SimpleNamespace

from evals.cases import (
    CASES,
    Expected,
    Fact,
    Finding,
    IntentCase,
    _lines_per_visit,
    _price_ratio,
    _share,
    _synthetic_churn,
    _synthetic_claims,
)
from evals.scoring import (
    compare,
    fact_found,
    numbers,
    removed_statements,
    score_findings,
    score_intent,
    score_understanding,
    summary_markdown,
)


def test_numbers_ignore_identifiers():
    assert numbers("PRV020 holds 47% of spend, 6.01 lines per visit, 1,234.5 NZD") == [
        47.0, 6.01, 1234.5]


def test_facts_are_matched_in_the_sentence_that_names_the_entity():
    share = Fact("share", [0.47, 47.0])
    assert fact_found("PRV020 accounts for 47.2% of all payments.", "PRV020", share)
    assert fact_found("| PRV020 | 0.47 | high |", "PRV020", share)
    # The right number next to a different provider does not count.
    assert not fact_found("PRV020 is large. PRV014 accounts for 47%.", "PRV020", share)
    assert not fact_found("PRV020 accounts for 30% of payments.", "PRV020", share)


def test_ratio_facts_accept_the_usual_ways_of_saying_them():
    ratio = Fact("price", [1.79, 79.0, 0.79], rel_tol=0.06)
    for text in ("PRV014 charges 1.8x the peer median.",
                 "PRV014 bills 78% above its peers.",
                 "PRV014 prices are 0.8 above the median ratio."):
        assert fact_found(text, "PRV014", ratio), text


def _expected():
    return Expected(
        task_types={"exploratory_analysis", "anomaly_detection"}, target="",
        questions=[["main patterns"], ["billing looks unusual"]],
        audiences=["fraud investigation"],
        findings=[Finding("PRV103", "volume", [Fact("share", [0.38, 38.0])]),
                  Finding("PRV107", "price", [])])


def test_findings_recall_and_fact_accuracy():
    reply = "PRV103 took 38% of all spend. Nothing else stood out."
    scored = score_findings(reply, _expected())
    assert scored["recall"] == 0.5 and scored["fact_accuracy"] == 1.0
    assert [d["named"] for d in scored["detail"]] == [True, False]


def test_understanding_checks_type_target_questions_and_audiences():
    spec = {"task_type": "anomaly_detection", "target_column": "",
            "brief": {"questions": ["Explore the payments and summarise the main patterns.",
                                    "Which providers' billing looks unusual?"],
                      "audiences": ["Fraud investigation team"]}}
    scored = score_understanding(spec, _expected())
    assert scored["score"] == 1.0
    spec["target_column"] = "service_date"
    spec["brief"]["questions"] = spec["brief"]["questions"][:1]
    scored = score_understanding(spec, _expected())
    assert not scored["checks"]["target"] and not scored["checks"]["question_2"]
    assert score_understanding({}, _expected())["score"] == 0.0


def test_intent_scoring_enforces_required_and_forbidden_tools():
    case = IntentCase("Explore", {"act"}, required_tools={"execute_code"},
                      forbidden_tools={"train_model"}, plan_first=True)
    good = SimpleNamespace(decision="act", allowed_tools=["execute_code"], plan_first=True)
    bad = SimpleNamespace(decision="act", allowed_tools=["execute_code", "train_model"],
                          plan_first=False)
    assert score_intent(good, case)["pass"]
    failed = score_intent(bad, case)["checks"]
    assert not failed["forbidden_tools"] and not failed["plan_first"]


def test_removed_statement_count_is_read_from_the_reply_footer():
    assert removed_statements("...\nRemoved 3 statement(s) whose figures ...") == 3
    assert removed_statements("clean reply") == 0


def test_summary_and_compare_render():
    case = {"case": "synthetic_claims", "stage": "full",
            "understanding": {"score": 0.8, "checks": {"target": True}},
            "findings": {"recall": 0.5, "fact_accuracy": 0.5,
                         "detail": [{"entity": "PRV107", "named": False, "facts": {}}]},
            "process": {"minutes": 12.0, "steps_ok": 9, "steps_total": 11}}
    old = {"run_id": "a", "backend": "ollama", "model": "m", "stages": ["full"],
           "cases": [case], "intent": [{"query": "hi", "decision": "answer",
                                        "pass": True, "checks": {"decision": True}}]}
    new = {**old, "run_id": "b", "cases": [{**case, "findings": {**case["findings"],
                                                                 "recall": 1.0}}]}
    text = summary_markdown(old)
    assert "Intent: 1 of 1 passed" in text and "not named: PRV107" in text
    assert "recall 0.50 -> 1.00" in compare(old, new)


def test_synthetic_cases_are_deterministic_and_plant_their_signals():
    claims = _synthetic_claims()
    assert claims.equals(_synthetic_claims())
    paid = claims.groupby("provider_id").paid_amount.sum()
    assert _share(paid, "PRV103") == paid.max() / paid.sum()      # the largest biller
    assert _lines_per_visit(claims, "PRV103") > 4.5
    assert 1.5 < _price_ratio(claims, "PRV107") < 1.7
    assert abs(_price_ratio(claims, "PRV103") - 1.0) < 0.05        # volume, not price
    churn = _synthetic_churn()
    rates = churn.groupby("contract").churned.mean()
    assert rates["month-to-month"] > 2 * rates["two_year"]


def test_case_workspaces_load_like_the_web_ui(tmp_path):
    from evals.run_eval import _open_workspace
    case = next(c for c in CASES if c.name == "synthetic_churn")
    session, kernel, expected = _open_workspace(case, tmp_path / "ws")
    assert kernel.namespace["df"].shape == (3000, 7)
    assert "retention team" in session.brief and expected.target == "churned"
    assert session.work_dir == tmp_path / "ws"


def test_intent_cases_run_in_a_fresh_or_an_after_analysis_session(tmp_path):
    """Requests about earlier work are checked where that work exists, and
    where it does not, so asking for clarification is judged in context."""
    from agent.api.agent_runner import _data_context
    from evals.cases import INTENT_CASES
    from evals.run_eval import _intent_contexts
    contexts = _intent_contexts(tmp_path)
    assert {c.context for c in INTENT_CASES} <= set(contexts)
    session, kernel, history = contexts["after_analysis"]
    shown = _data_context(session, kernel)
    assert "EDA AGENDA (2 of 2 answered)" in shown and "pr_auc" in shown
    assert "trained_model" in shown and history[-1]["role"] == "assistant"
    fresh_session, fresh_kernel, fresh_history = contexts["fresh"]
    assert fresh_history == [] and "EDA AGENDA" not in _data_context(fresh_session, fresh_kernel)
    paired = {c.query for c in INTENT_CASES if c.context == "after_analysis"}
    assert paired and all(any(c.query == q and c.context == "fresh" for c in INTENT_CASES)
                          for q in paired)
