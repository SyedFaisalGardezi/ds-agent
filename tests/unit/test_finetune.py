"""Tracing, training tasks and dataset building, checked without a model."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.api import trace


class FakeLLM:
    model = "fake-model"

    def __init__(self, generate="{}", tool_reply=None):
        self._gen, self._tool_reply = generate, tool_reply

    def _generate(self, prompt, **kw):
        return self._gen(prompt) if callable(self._gen) else self._gen

    def _chat(self, messages, **kw):
        return "chat reply"

    def chat_with_tools(self, messages, tools, **kw):
        return self._tool_reply or {"content": "done", "tool_calls": []}

    def is_available(self):
        return True


def _records(folder: Path) -> list[dict]:
    return [json.loads(x) for x in (folder / "traces.jsonl").read_text().splitlines()]


# ── tracing ──────────────────────────────────────────────────────────────────

def test_tracing_records_each_call_with_its_job_and_turn(tmp_path):
    llm = trace.install(FakeLLM(), tmp_path)
    tools = [{"type": "function", "function": {"name": "execute_code"}}]
    trace.new_turn("sess", "explore the data")

    def _critique_step():
        return llm._generate("judge this")

    def _tool_call_loop():
        llm.chat_with_tools([{"role": "user", "content": "go", "images": ["b64"]}], tools)
        llm.chat_with_tools([{"role": "user", "content": "again"}], tools)
    _critique_step()
    _tool_call_loop()
    recs = _records(tmp_path)
    assert [r["purpose"] for r in recs] == ["critic", "agent_turn", "agent_turn"]
    assert len({r["turn_id"] for r in recs}) == 1 and [r["seq"] for r in recs] == [1, 2, 3]
    assert "images" not in recs[1]["input"]["messages"][0]          # images stripped
    assert len((tmp_path / "tools.jsonl").read_text().splitlines()) == 1   # schemas stored once


def test_tracing_identifies_jobs_in_the_real_agent_code(tmp_path):
    from agent.api.agent_tools import WorkPlanTool
    from agent.api.intent import interpret
    from agent.api.sessions import Session
    plan_json = json.dumps({"objective": "Find risky providers"})
    llm = trace.install(FakeLLM(generate=lambda p: plan_json if "planning" in p else
                                '{"decision":"answer","objective":"x"}'), tmp_path)
    trace.new_turn("s", "q")
    interpret("what is this?", context="", history=[], tools=[], llm=llm)
    kernel = SimpleNamespace(namespace={"df": pd.DataFrame({"x": [1, 2]})}, turn_deadline=1.0)
    WorkPlanTool().execute({}, session=Session("s"), kernel=kernel, llm_client=llm)
    assert [r["purpose"] for r in _records(tmp_path)] == ["intent", "plan_scope", "plan_approach"]


def test_tracing_is_off_without_a_folder_and_records_failures(tmp_path, monkeypatch):
    monkeypatch.delenv("DSAGENT_TRACE_DIR", raising=False)
    plain = FakeLLM()
    assert trace.install(plain) is plain and not getattr(plain, "_traced", False)

    def boom(prompt):
        raise TimeoutError("slow")
    llm = trace.install(FakeLLM(generate=boom), tmp_path)
    with pytest.raises(TimeoutError):
        llm._generate("x")
    assert _records(tmp_path)[0]["error"].startswith("TimeoutError")


# ── training tasks ───────────────────────────────────────────────────────────

def test_training_tasks_are_reproducible_and_disjoint_from_the_eval():
    from finetune.tasks import FAMILIES, make_task
    seen = set()
    for seed in range(1000, 1008):
        task, df, brief = make_task(seed)
        again, df2, _ = make_task(seed)
        assert df.equals(df2) and task.request == again.request
        assert task.expected.findings and all(isinstance(f.entity, str)
                                              for f in task.expected.findings)
        seen.add(task.family)
    assert seen == set(FAMILIES)
    from evals.cases import SEED
    for bad in (999, SEED):
        with pytest.raises(ValueError):
            make_task(bad)


def test_disjointness_guard_rejects_eval_material():
    from evals.cases import _CLAIMS_BRIEF
    from finetune.tasks import assert_disjoint_from_eval, make_task
    task, df, _ = make_task(1000)
    with pytest.raises(AssertionError):
        assert_disjoint_from_eval(task, df, _CLAIMS_BRIEF)
    with pytest.raises(AssertionError):
        assert_disjoint_from_eval(task, df.assign(provider_id="PRV101"), "brief")


# ── export ───────────────────────────────────────────────────────────────────

def _write_run(run: Path):
    (run / "trace").mkdir(parents=True)
    tools = [{"type": "function", "function": {"name": "execute_code", "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}}}}}]
    (run / "trace" / "tools.jsonl").write_text(json.dumps({"ref": "T1", "tools": tools}) + "\n")
    runs = [{"task_id": "claims_anomaly-1000", "turn_id": "good", "passed": True},
            {"task_id": "claims_anomaly-1004", "turn_id": "bad", "passed": False}]
    (run / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    call = {"content": "", "tool_calls": [{"function": {"name": "execute_code",
                                                        "arguments": {"code": "print(1)"}}}]}
    sys_msg, user = "You are an analyst.", {"role": "user", "content": "explore"}
    recs = [
        {"turn_id": "good", "seq": 1, "purpose": "intent", "method": "generate",
         "input": {"prompt": "interpret"}, "output": '{"decision":"act","objective":"x",'
                                                     '"allowed_tools":["execute_code"]}'},
        {"turn_id": "good", "seq": 2, "purpose": "critic", "method": "generate",
         "input": {"prompt": "judge"}, "output": "not json"},
        # step 1: a tool call that failed when it ran (see step 2's input)
        {"turn_id": "good", "seq": 3, "purpose": "agent_turn", "method": "chat_with_tools",
         "input": {"system": sys_msg, "messages": [user], "tools_ref": "T1"}, "output": call},
        # step 2: a tool call that succeeded (see step 3's input)
        {"turn_id": "good", "seq": 4, "purpose": "agent_turn", "method": "chat_with_tools",
         "input": {"system": sys_msg, "tools_ref": "T1", "messages": [
             user, call | {"role": "assistant"},
             {"role": "tool", "tool_name": "execute_code", "content": "ERROR: KeyError"}]},
         "output": call},
        # step 3: the final answer
        {"turn_id": "good", "seq": 5, "purpose": "agent_turn", "method": "chat_with_tools",
         "input": {"system": sys_msg, "tools_ref": "T1", "messages": [
             user, call | {"role": "assistant"},
             {"role": "tool", "tool_name": "execute_code", "content": "ERROR: KeyError"},
             call | {"role": "assistant"},
             {"role": "tool", "tool_name": "execute_code", "content": "PRV9 47%"}]},
         "output": {"content": "PRV9 holds 47% of spend.", "tool_calls": []}},
        {"turn_id": "bad", "seq": 1, "purpose": "intent", "method": "generate",
         "input": {"prompt": "interpret"}, "output": '{"decision":"answer","objective":"y"}'},
    ]
    (run / "trace" / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_export_keeps_only_checked_calls_from_passing_runs(tmp_path):
    from finetune.build_dataset import export
    _write_run(tmp_path / "run")
    out = export(SimpleNamespace(runs=[str(tmp_path / "run")], out=str(tmp_path / "ds"),
                                 purposes=""))
    card = json.loads((out / "dataset_card.json").read_text())
    examples = [json.loads(x) for f in ("train", "valid")
                for x in (out / f"{f}.jsonl").read_text().splitlines()]
    assert sorted(e["purpose"] for e in examples) == ["agent_turn", "agent_turn", "intent"]
    assert card["dropped"] == {"run did not pass": 1, "critic verdict missing": 1,
                               "the tool call failed when it ran": 1}
    final = next(e for e in examples if e["messages"][-1]["content"].startswith("PRV9"))
    roles = [m["role"] for m in final["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert final["messages"][3] == {"role": "tool", "content": "ERROR: KeyError",
                                    "name": "execute_code"}
    assert final["tools"][0]["function"]["name"] == "execute_code"
    assert {e["task_id"] for e in examples} == {"claims_anomaly-1000"}


def test_relabel_writes_teacher_examples_and_preference_pairs(tmp_path, monkeypatch):
    from finetune import build_dataset as B
    _write_run(tmp_path / "run")
    teacher = FakeLLM(generate='{"verdict":"accept","finding":"PRV9 47%"}')
    monkeypatch.setattr(B, "_client", lambda backend, confirm: teacher)
    out = B.relabel(SimpleNamespace(traces=str(tmp_path / "run" / "trace"),
                                    out=str(tmp_path / "rl"), purposes="critic", limit=0,
                                    confirm_spend=True))
    sft = [json.loads(x) for x in (out / "sft.jsonl").read_text().splitlines()]
    dpo = [json.loads(x) for x in (out / "dpo.jsonl").read_text().splitlines()]
    assert len(sft) == 1 and json.loads(sft[0]["messages"][-1]["content"])["verdict"] == "accept"
    assert dpo[0]["rejected"]["content"] == "not json"     # the student's invalid critic output


def test_teacher_runs_refuse_to_spend_without_confirmation(monkeypatch):
    from agent.api import llm as L
    from finetune.build_dataset import _client
    monkeypatch.setenv("USE_CLAUDE", "no")    # restored after the test; _client sets it
    monkeypatch.setattr(L.AnthropicClient, "is_available", lambda self: True)
    with pytest.raises(SystemExit, match="confirm-spend"):
        _client("claude", confirm=False)
