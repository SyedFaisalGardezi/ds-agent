"""Build an instruction-tuning / distillation dataset from the agent's own calls.

The student (the local model) is trained on exactly the inputs the agent sends
it, so the dataset is made from traces of real agent runs, not from prompts
written separately. Three commands, run from the repository root:

  collect   Run the agent with the teacher model on synthetic training tasks,
            with tracing on, and score every run with the evaluation scorers.

      .venv/bin/python -m finetune.build_dataset collect --backend claude \
          --seeds 1000-1019 --confirm-spend

  export    Keep the calls from runs that passed, check each call's output for
            its job (valid JSON, a known tool with valid arguments, a tool call
            that did not fail when it ran, and so on), and write chat-format
            JSONL split by task into train and validation files.

      .venv/bin/python -m finetune.build_dataset export \
          --runs finetune/data/runs/<id> --out finetune/data/dataset

  relabel   Take traces of the local model and ask the teacher for its output
            on each identical input. Writes training examples from the
            teacher, plus preference pairs where the local output failed its
            check and the teacher's passed (for DPO).

      .venv/bin/python -m finetune.build_dataset relabel \
          --traces finetune/data/runs/<local run>/trace --out finetune/data/relabel \
          --confirm-spend

Teacher runs call the Claude API and are billed per token; they refuse to start
without --confirm-spend. Check Anthropic's terms for your account before using
outputs to train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "finetune" / "data"

# Jobs worth training on, in the order they matter for analysis quality.
TRAIN_PURPOSES = ("agent_turn", "critic", "plan_scope", "plan_approach", "brief", "intent",
                  "sql", "action_check", "qa_answer", "synthesis")

PASS_RULES = {"understanding": 0.6, "recall": 0.5, "steps_ok_ratio": 0.7,
              "max_removed": 2}


# ── reading traces ───────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _tools_by_ref(trace_dir: Path) -> dict[str, list]:
    return {r["ref"]: r["tools"] for r in _read_jsonl(trace_dir / "tools.jsonl")}


def _tool_message(m: dict) -> dict:
    out = {"role": "tool", "content": m.get("content", "")}
    name = m.get("tool_name") or m.get("name")
    if name:
        out["name"] = name
    return out


def _assistant_message(output: Any) -> dict:
    if isinstance(output, dict):
        msg: dict = {"role": "assistant", "content": output.get("content") or ""}
        calls = []
        for c in output.get("tool_calls") or []:
            fn = c.get("function", {}) if isinstance(c, dict) else {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            calls.append({"type": "function",
                          "function": {"name": fn.get("name", ""), "arguments": args}})
        if calls:
            msg["tool_calls"] = calls
        return msg
    return {"role": "assistant", "content": str(output or "")}


def to_example(rec: dict, tools: dict[str, list]) -> dict:
    """A trace record as a chat-format training example."""
    inp = rec.get("input") or {}
    messages: list[dict] = []
    if inp.get("system"):
        messages.append({"role": "system", "content": inp["system"]})
    if rec["method"] == "generate":
        messages.append({"role": "user", "content": inp.get("prompt", "")})
    else:
        for m in inp.get("messages") or []:
            if m.get("role") == "tool":
                messages.append(_tool_message(m))
            elif m.get("role") == "assistant":
                messages.append(_assistant_message(m))
            else:
                messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append(_assistant_message(rec.get("output")))
    example = {"messages": messages, "purpose": rec["purpose"],
               "turn_id": rec.get("turn_id", ""), "source_model": rec.get("model", "")}
    if inp.get("tools_ref"):
        example["tools"] = tools.get(inp["tools_ref"], [])
    return example


# ── checking each output for its job ────────────────────────────────────────

def _json(text: Any) -> Any:
    from agent.api.intent import _json_text
    try:
        return json.loads(_json_text(str(text or "")))
    except Exception:  # noqa: BLE001
        return None


def _failed_tool_calls(rec: dict, later: list[dict]) -> bool:
    """True when the tool call this turn made came back as an error. The next
    agent turn's input holds the results of this turn's calls."""
    n_before = len((rec.get("input") or {}).get("messages") or [])
    for nxt in later:
        msgs = (nxt.get("input") or {}).get("messages") or []
        if len(msgs) <= n_before:
            continue
        for m in msgs[n_before + 1:]:
            if m.get("role") != "tool":
                break
            if str(m.get("content", "")).lstrip().startswith("ERROR"):
                return True
        return False
    return False


def check_output(rec: dict, later: list[dict] | None = None) -> str:
    """'' when the output is a usable target for its job, else the reason."""
    from jsonschema import validate

    from agent.api.agent_tools import _SQL_FORBIDDEN, TOOL_REGISTRY
    from agent.api.intent import TaskIntent
    purpose, out = rec.get("purpose"), rec.get("output")
    if rec.get("error"):
        return "call raised an error"
    if out in (None, "", {}):
        return "empty output"
    if purpose == "intent":
        try:
            TaskIntent.model_validate(_json(out) or {})
        except Exception:  # noqa: BLE001
            return "intent does not match the TaskIntent schema"
    elif purpose == "brief":
        data = _json(out)
        if not (isinstance(data, dict) and data.get("objective") and data.get("questions")):
            return "brief reading lacks objective or questions"
    elif purpose in ("plan_scope", "plan_approach"):
        if not isinstance(_json(out), dict):
            return "plan is not a JSON object"
    elif purpose == "critic":
        data = _json(out)
        if not (isinstance(data, dict) and data.get("verdict") in ("accept", "retry", "replan")):
            return "critic verdict missing"
    elif purpose == "action_check":
        data = _json(out)
        if not (isinstance(data, dict) and isinstance(data.get("allowed"), bool)):
            return "action check lacks a boolean 'allowed'"
    elif purpose == "sql":
        if _SQL_FORBIDDEN.search(str(out)):
            return "SQL modifies data"
    elif purpose == "agent_turn":
        msg = _assistant_message(out)
        for call in msg.get("tool_calls", []):
            name, args = call["function"]["name"], call["function"]["arguments"]
            tool = TOOL_REGISTRY.get(name)
            if tool is None:
                return f"unknown tool {name}"
            try:
                validate(args if isinstance(args, dict) else {},
                         getattr(tool, "input_schema", {"type": "object"}))
            except Exception:  # noqa: BLE001
                return f"invalid arguments for {name}"
        if msg.get("tool_calls") and later is not None and _failed_tool_calls(rec, later):
            return "the tool call failed when it ran"
        if not msg.get("tool_calls") and not msg["content"].strip():
            return "neither a tool call nor an answer"
    return ""


def run_passed(run: dict, rules: dict = PASS_RULES) -> tuple[bool, list[str]]:
    why = []
    if run.get("error"):
        why.append("run error")
    if (run.get("understanding") or 0) < rules["understanding"]:
        why.append("understanding below threshold")
    recall = run.get("recall")
    if recall is not None and recall < rules["recall"]:
        why.append("planted findings missed")
    steps, ok = run.get("steps_total") or 0, run.get("steps_ok") or 0
    if steps and ok / steps < rules["steps_ok_ratio"]:
        why.append("too many failed steps")
    if (run.get("removed_statements") or 0) > rules["max_removed"]:
        why.append("too many unsupported statements")
    return not why, why


def _split(turn_task: str, valid_share: float = 0.1) -> str:
    """Split by task, never by call, so no task appears in both files."""
    h = int(hashlib.sha256(turn_task.encode()).hexdigest()[:8], 16)
    return "valid" if (h % 1000) / 1000 < valid_share else "train"


# ── commands ─────────────────────────────────────────────────────────────────

def _client(backend: str | None, confirm: bool):
    if backend:
        os.environ["USE_CLAUDE"] = "yes" if backend == "claude" else "no"
    from agent.api.server import make_llm_client
    llm = make_llm_client()
    if type(llm).__name__ == "AnthropicClient" and not confirm:
        raise SystemExit("This uses the Claude API and is billed per token. "
                         "Re-run with --confirm-spend to proceed.")
    if not llm.is_available():
        raise SystemExit("The model backend is not reachable.")
    return llm


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def collect(args) -> Path:
    from agent.api import trace
    from agent.api.agent_runner import run_reactive
    from evals.cases import Case
    from evals.run_eval import _open_workspace
    from evals.scoring import removed_statements, score_findings, score_understanding
    from finetune.tasks import make_task, write_task

    out = Path(args.out or DATA / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    os.environ["DSAGENT_TRACE_DIR"] = str(out / "trace")
    llm = _client(args.backend, args.confirm_spend)
    for seed in _seeds(args.seeds):
        task, df, brief = make_task(seed, args.family)
        case = Case(task.task_id, task.family, task.request,
                    build=lambda ws, t=task, d=df, b=brief: (write_task(t, d, b, ws), t.expected)[1])
        session, kernel, expected = _open_workspace(case, out / "workspaces" / task.task_id)
        t0 = time.time()
        row: dict[str, Any] = {"task_id": task.task_id, "family": task.family, "seed": seed,
                               "request": task.request, "backend": type(llm).__name__}
        try:
            res = run_reactive(task.request, session, kernel, llm, ltm=None)
            turn = trace._TURN.get() or {}
            spec = kernel.namespace.get("TASK_SPEC")
            found = score_findings(res.reply, expected)
            row.update({
                "turn_id": turn.get("turn_id", ""),
                "understanding": score_understanding(spec if isinstance(spec, dict) else {},
                                                     expected)["score"],
                "recall": found["recall"], "fact_accuracy": found["fact_accuracy"],
                "steps_total": len(res.steps),
                "steps_ok": sum(s.output.success for s in res.steps),
                "removed_statements": removed_statements(res.reply)})
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        row["minutes"] = round((time.time() - t0) / 60, 2)
        row["passed"], row["why_not"] = run_passed(row)
        with (out / "runs.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        with (out / "tasks.jsonl").open("a") as fh:
            fh.write(json.dumps({"task_id": task.task_id, "request": task.request,
                                 "expected": asdict(task.expected)}, default=list) + "\n")
        print(f"{task.task_id}: {'PASS' if row['passed'] else 'fail'} "
              f"({row['minutes']} min) {', '.join(row['why_not'])}", flush=True)
    print(f"Runs: {out}")
    return out


def export(args) -> Path:
    out = Path(args.out or DATA / "dataset")
    out.mkdir(parents=True, exist_ok=True)
    purposes = set(args.purposes.split(",")) if args.purposes else set(TRAIN_PURPOSES)
    stats: Counter = Counter()
    dropped: Counter = Counter()
    seen: set[str] = set()
    files = {k: (out / f"{k}.jsonl").open("w") for k in ("train", "valid")}
    for run_dir in [Path(p) for p in args.runs]:
        runs = _read_jsonl(run_dir / "runs.jsonl")
        passing = {r["turn_id"]: r for r in runs if r.get("passed") and r.get("turn_id")}
        tools = _tools_by_ref(run_dir / "trace")
        by_turn: dict[str, list[dict]] = defaultdict(list)
        for rec in _read_jsonl(run_dir / "trace" / "traces.jsonl"):
            by_turn[rec.get("turn_id", "")].append(rec)
        for turn_id, recs in by_turn.items():
            if turn_id not in passing:
                dropped["run did not pass"] += len(recs)
                continue
            recs.sort(key=lambda r: r.get("seq", 0))
            agent_turns = [r for r in recs if r["purpose"] == "agent_turn"]
            for rec in recs:
                if rec["purpose"] not in purposes:
                    dropped[f"purpose {rec['purpose']} not exported"] += 1
                    continue
                later = agent_turns[agent_turns.index(rec) + 1:] if rec in agent_turns else None
                reason = check_output(rec, later)
                if reason:
                    dropped[reason] += 1
                    continue
                example = to_example(rec, tools)
                key = hashlib.sha256(json.dumps(example["messages"], sort_keys=True,
                                                default=str).encode()).hexdigest()
                if key in seen:
                    dropped["duplicate"] += 1
                    continue
                seen.add(key)
                example["task_id"] = passing[turn_id]["task_id"]
                split = _split(example["task_id"])
                files[split].write(json.dumps(example, ensure_ascii=False, default=str) + "\n")
                stats[f"{split}:{rec['purpose']}"] += 1
    for fh in files.values():
        fh.close()
    card = {"created": time.strftime("%Y-%m-%d %H:%M"), "runs": [str(p) for p in args.runs],
            "pass_rules": PASS_RULES, "purposes": sorted(purposes),
            "kept": dict(stats), "dropped": dict(dropped)}
    (out / "dataset_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1))
    return out


def relabel(args) -> Path:
    trace_dir = Path(args.traces)
    out = Path(args.out or DATA / "relabel")
    out.mkdir(parents=True, exist_ok=True)
    teacher = _client("claude", args.confirm_spend)
    tools = _tools_by_ref(trace_dir)
    purposes = set(args.purposes.split(",")) if args.purposes else set(TRAIN_PURPOSES)
    n = Counter()
    with (out / "sft.jsonl").open("a") as sft, (out / "dpo.jsonl").open("a") as dpo:
        for rec in _read_jsonl(trace_dir / "traces.jsonl"):
            if rec["purpose"] not in purposes or rec.get("error"):
                continue
            if args.limit and n["asked"] >= args.limit:
                break
            inp = rec["input"]
            n["asked"] += 1
            try:
                if rec["method"] == "generate":
                    output: Any = teacher._generate(inp["prompt"], system=inp.get("system"),
                                                    temperature=0.0,
                                                    max_tokens=inp.get("max_tokens") or 2000,
                                                    json_mode=inp.get("json_mode", False))
                elif rec["method"] == "chat_with_tools":
                    output = teacher.chat_with_tools(inp["messages"], tools[inp["tools_ref"]],
                                                     system=inp.get("system"), temperature=0.0,
                                                     max_tokens=inp.get("max_tokens") or 4000)
                else:
                    output = teacher._chat(inp["messages"], system=inp.get("system"),
                                           max_tokens=inp.get("max_tokens") or 2000)
            except Exception as exc:  # noqa: BLE001
                n[f"teacher error {type(exc).__name__}"] += 1
                continue
            taught = {**rec, "output": output, "model": getattr(teacher, "model", "")}
            if check_output(taught):
                n["teacher output failed its check"] += 1
                continue
            example = to_example(taught, tools)
            sft.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")
            n["sft"] += 1
            if check_output(rec):   # the student's own output failed: a preference pair
                student = to_example(rec, tools)
                dpo.write(json.dumps({"prompt": example["messages"][:-1],
                                      "chosen": example["messages"][-1],
                                      "rejected": student["messages"][-1],
                                      "tools": example.get("tools"),
                                      "purpose": rec["purpose"]},
                                     ensure_ascii=False, default=str) + "\n")
                n["dpo"] += 1
    print(dict(n))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--backend", choices=["ollama", "claude"], default=None)
    c.add_argument("--seeds", default="1000-1003")
    c.add_argument("--family", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--confirm-spend", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--purposes", default="")
    r = sub.add_parser("relabel")
    r.add_argument("--traces", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--purposes", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--confirm-spend", action="store_true")
    args = ap.parse_args(argv)
    {"collect": collect, "export": export, "relabel": relabel}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
