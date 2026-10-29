"""Evaluate the DS agent on fixed cases with known answers.

Run from the repository root:

    .venv/bin/python -m evals.run_eval --stage intent
    .venv/bin/python -m evals.run_eval --stage understand
    .venv/bin/python -m evals.run_eval --stage full --cases synthetic_claims
    .venv/bin/python -m evals.run_eval --stage all --backend claude
    .venv/bin/python -m evals.run_eval --compare evals/results/A/results.json \
                                                 evals/results/B/results.json

Stages, cheapest first:
  intent      each request in the intent suite through the interpreter only
              (about 15 to 80 s per query on the local model);
  understand  the case request and the brief reading only (about 2 min a case);
  full        the whole agent turn, as the web UI runs it (about 15 min a case).

The agent runs exactly as in the server (same client, same runner), with file
writes in "suggest" mode, no long-term memory, and an in-memory session, so an
evaluation never touches the user's session store or working folders. Each case
gets a fresh workspace under the run's output folder. Results are written to
evals/results/<run id>/results.json and summary.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.cases import AFTER_ANALYSIS, CASES, INTENT_CASES, Case  # noqa: E402
from evals.scoring import (  # noqa: E402
    compare,
    removed_statements,
    score_findings,
    score_intent,
    score_understanding,
    summary_markdown,
)

_DATA_EXTS = {".csv", ".parquet", ".xlsx", ".xls", ".tsv", ".feather"}
_BRIEF_EXTS = {".pdf", ".md", ".txt", ".rst"}


# ── plumbing ─────────────────────────────────────────────────────────────────

class CallMeter:
    """Counts and times every model call the agent makes."""

    def __init__(self, llm: Any) -> None:
        self.calls: list[tuple[str, float]] = []
        for name in ("_generate", "_chat", "chat_with_tools"):
            original = getattr(llm, name, None)
            if original is not None:
                setattr(llm, name, self._wrap(name, original))

    def _wrap(self, name, fn):
        def timed(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                self.calls.append((name, time.time() - t0))
        return timed

    def reset(self) -> None:
        self.calls.clear()

    def summary(self) -> dict[str, Any]:
        return {"llm_calls": len(self.calls),
                "llm_minutes": round(sum(s for _, s in self.calls) / 60, 2)}


def _hashes(ws: Path) -> dict[str, str]:
    return {str(p.relative_to(ws)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ws.rglob("*")) if p.is_file()}


def _open_workspace(case: Case, ws: Path):
    """Build the workspace and load it the way setting a working directory
    in the web UI does. Returns (session, kernel, expected)."""
    from agent.api.agent_tools import is_agent_generated_doc, scan_workspace
    from agent.api.chat import _load_secondary
    from agent.api.kernel import SessionKernel
    from agent.api.pdf_reader import extract
    from agent.api.sessions import Session

    ws.mkdir(parents=True, exist_ok=True)
    expected = case.build(ws)
    session = Session(f"eval-{ws.name}", work_dir=ws)
    kernel = SessionKernel()
    kernel.namespace["WORK_DIR"] = ws
    data = sorted(p for p in ws.iterdir() if p.suffix.lower() in _DATA_EXTS)
    for i, path in enumerate(data):
        session.add_data_path(path)
        if i == 0:
            kernel.load_data(path)
        else:
            _load_secondary(kernel, path, f"df{i + 1}")
    for path in sorted(p for p in ws.iterdir() if p.suffix.lower() in _BRIEF_EXTS):
        text = (extract(path.read_bytes()).text if path.suffix.lower() == ".pdf"
                else path.read_text(errors="replace"))
        if not is_agent_generated_doc(text):
            session.add_brief(text, path.name)
    kernel.set_context(session.target, session.brief)
    kernel.namespace["WORKSPACE_MAP"] = scan_workspace(ws)
    return session, kernel, expected


def _interpret(request: str, session, kernel, llm, history: list[dict] | None = None):
    from agent.api.agent_runner import _data_context
    from agent.api.agent_tools import get_tool_schemas
    from agent.api.intent import interpret
    context = _data_context(session, kernel) + "\nFULL BRIEF:\n" + (session.brief or "")
    tools = [{"name": t["name"], "description": t.get("description", "")}
             for t in get_tool_schemas()]
    return interpret(request, context=context, history=history or [], tools=tools, llm=llm)


def _intent_contexts(out: Path) -> dict[str, tuple]:
    """The two sessions the intent cases run in: (session, kernel, history)."""
    fresh_case = next(c for c in CASES if c.name == "synthetic_claims")
    fresh_s, fresh_k, _ = _open_workspace(fresh_case, out / "workspaces" / "intent_fresh")
    done_case = next(c for c in CASES if c.name == "synthetic_churn")
    done_s, done_k, _ = _open_workspace(done_case, out / "workspaces" / "intent_after")
    done_k.namespace.update(json.loads(json.dumps(AFTER_ANALYSIS["namespace"])))
    done_s.target = "churned"
    done_s.artifacts = dict(AFTER_ANALYSIS["artifacts"])
    return {"fresh": (fresh_s, fresh_k, []),
            "after_analysis": (done_s, done_k, AFTER_ANALYSIS["history"])}


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


# ── stages ───────────────────────────────────────────────────────────────────

def run_intent(llm, meter: CallMeter, out: Path, verbose: bool) -> list[dict]:
    contexts = _intent_contexts(out)
    rows = []
    for ic in INTENT_CASES:
        meter.reset()
        t0 = time.time()
        session, kernel, history = contexts[ic.context]
        intent = _interpret(ic.query, session, kernel, llm, history)
        row = {"query": ic.query, "context": ic.context, **score_intent(intent, ic),
               "plan_first": getattr(intent, "plan_first", None),
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        _log(verbose, f"  intent {'PASS' if row['pass'] else 'FAIL'} "
                      f"({row['seconds']}s) [{ic.context}] {ic.query[:60]} -> {row['decision']}")
    return rows


def run_case(case: Case, stage: str, rep: int, llm, meter: CallMeter,
             out: Path, verbose: bool) -> dict[str, Any]:
    from agent.api import progress
    from agent.api.agent_runner import run_reactive
    from agent.api.agent_tools import UnderstandTaskTool, _verify_export

    ws = out / "workspaces" / f"{case.name}_{stage}_{rep}"
    session, kernel, expected = _open_workspace(case, ws)
    before = _hashes(ws)
    meter.reset()
    t0 = time.time()
    record: dict[str, Any] = {"case": case.name, "stage": stage, "repeat": rep,
                              "expected": [asdict(f) for f in expected.findings]}
    progress.emit = (lambda jid, msg, tool="", level="info":
                     _log(verbose, f"    [{time.time() - t0:6.0f}s] {tool or '-'}: {msg[:150]}"))
    try:
        if stage == "understand":
            intent = _interpret(case.request, session, kernel, llm)
            record["intent"] = {"decision": intent.decision,
                                "plan_first": getattr(intent, "plan_first", None),
                                "allowed_tools": intent.allowed_tools}
            UnderstandTaskTool().execute({}, session=session, kernel=kernel,
                                         llm_client=llm)
            record["findings"] = {}
            steps: list = []
            reply = ""
        else:
            res = run_reactive(case.request, session, kernel, llm, ltm=None,
                               job_id="eval")
            reply, steps = res.reply, res.steps
            record["reply"] = reply
            record["findings"] = score_findings(reply, expected)
            cells = getattr(kernel, "analysis_cells", []) or []
            record["export_error"] = (_verify_export(cells, session=session, kernel=kernel)
                                      if cells else "no analysis cells")
    except Exception:  # noqa: BLE001
        record["error"] = traceback.format_exc(limit=3)[-600:]
        steps, reply = [], ""
    spec = kernel.namespace.get("TASK_SPEC")
    record["understanding"] = score_understanding(spec if isinstance(spec, dict) else {},
                                                  expected)
    agenda = kernel.namespace.get("EDA_AGENDA") or []
    ok = [s for s in steps if s.output.success]
    record["process"] = {
        "minutes": round((time.time() - t0) / 60, 2),
        **meter.summary(),
        "steps_total": len(steps),
        "steps_ok": len(ok),
        "tools": [f"{s.tool}{'' if s.output.success else '!'}" for s in steps],
        "sandbox_unavailable": sum("Isolated execution is unavailable" in (s.output.error or "")
                                   for s in steps),
        "removed_statements": removed_statements(reply),
        "agenda_answered": sum(a.get("status") == "answered" for a in agenda),
        "agenda_total": len(agenda),
        "export_reruns": (record.get("export_error") == "") if stage == "full" else None,
        "workspace_untouched": _hashes(ws) == before,
    }
    return record


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["intent", "understand", "full", "all"],
                    default="understand")
    ap.add_argument("--cases", default="", help="comma-separated case names (default all)")
    ap.add_argument("--backend", choices=["ollama", "claude"], default=None,
                    help="default: whatever USE_CLAUDE in .env selects")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "evals" / "results"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    args = ap.parse_args(argv)

    if args.compare:
        old, new = (json.loads(Path(p).read_text()) for p in args.compare)
        print(compare(old, new))
        return 0

    if args.backend:
        os.environ["USE_CLAUDE"] = "yes" if args.backend == "claude" else "no"
    from agent.api.server import make_llm_client  # also loads .env
    llm = make_llm_client()
    if not llm.is_available():
        print("The selected model backend is not reachable.", file=sys.stderr)
        return 2
    try:   # listing models can succeed while generation fails, so test one call
        llm._generate("Reply with the word ok.", temperature=0.0, max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        print(f"The model backend is up but cannot generate: {exc}", file=sys.stderr)
        return 2
    meter = CallMeter(llm)
    backend = "claude" if type(llm).__name__ == "AnthropicClient" else "ollama"
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{backend}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    stages = ["intent", "understand", "full"] if args.stage == "all" else [args.stage]
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in CASES if (not wanted or c.name in wanted)]
    for c in cases:
        if not c.available():
            print(f"skipping {c.name}: its data is not on this machine")
    cases = [c for c in cases if c.available()]

    results: dict[str, Any] = {"run_id": run_id, "backend": backend,
                               "model": getattr(llm, "model", "?"), "stages": stages,
                               "cases": [], "intent": None}
    started = time.time()
    if "intent" in stages:
        _log(verbose, "stage: intent")
        results["intent"] = run_intent(llm, meter, out, verbose)
    for stage in [s for s in stages if s != "intent"]:
        for case in cases:
            for rep in range(1, args.repeats + 1):
                _log(verbose, f"stage: {stage} | case: {case.name} | repeat {rep}")
                results["cases"].append(run_case(case, stage, rep, llm, meter, out, verbose))
                (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    results["minutes"] = round((time.time() - started) / 60, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    summary = summary_markdown(results)
    (out / "summary.md").write_text(summary)
    print(summary)
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
