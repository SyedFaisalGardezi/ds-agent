"""Scoring for the DS agent evaluation. Deterministic: no LLM judges.

Three questions are scored separately, because they fail for different reasons:

  understanding  did the agent read the brief correctly (task type, target,
                 the brief's questions and audiences)?
  findings       did the reply surface what a correct analysis must find, and
                 state the key numbers correctly?
  process        did the run work (steps that succeeded, statements removed
                 for lacking evidence, exported code that re-runs, no files
                 written without approval, time and model calls)?

Findings are checked on the reply text. An entity counts as found when it is
named in the reply; a fact counts as correct when a sentence or table row that
names the entity contains a number within tolerance of the true value. This is
a proxy: it cannot tell a finding stated as a risk from a passing mention.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any

from evals.cases import Expected, Fact, IntentCase

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text or "")]


def units(reply: str) -> list[str]:
    """Table rows and bullet lines whole; prose split into sentences."""
    out: list[str] = []
    for line in (reply or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("|", "-", "*")) or re.match(r"\d+\.\s", line):
            out.append(line)
        else:
            out.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
    return out


def mentions(text: str, entity: str) -> bool:
    low = text.lower()
    return any(form in low for form in {entity.lower(), entity.lower().replace("_", " ")})


def fact_found(reply: str, entity: str, fact: Fact) -> bool:
    for unit in units(reply):
        if not mentions(unit, entity):
            continue
        for x in numbers(unit):
            for v in fact.values:
                if abs(x - v) <= max(fact.abs_tol, fact.rel_tol * abs(v)):
                    return True
    return False


def score_findings(reply: str, expected: Expected) -> dict[str, Any]:
    rows = []
    for f in expected.findings:
        rows.append({
            "entity": f.entity, "why": f.why, "named": mentions(reply, f.entity),
            "facts": {fact.label: fact_found(reply, f.entity, fact) for fact in f.facts},
        })
    facts = [ok for r in rows for ok in r["facts"].values()]
    return {
        "recall": mean(r["named"] for r in rows) if rows else None,
        "fact_accuracy": mean(facts) if facts else None,
        "detail": rows,
    }


def score_understanding(spec: dict, expected: Expected) -> dict[str, Any]:
    spec = spec if isinstance(spec, dict) else {}
    brief = spec.get("brief") if isinstance(spec.get("brief"), dict) else {}
    captured = [str(q).lower() for q in (brief.get("questions") or [])]
    if not captured:   # older extraction path: requirements only
        captured = [str(q).lower() for q in (spec.get("specific_requirements") or [])]
    audiences = " | ".join(str(a).lower() for a in (brief.get("audiences") or []))
    checks = {
        "task_type": spec.get("task_type") in expected.task_types,
        # No spec at all must not pass as "correctly found no target".
        "target": bool(spec) and (spec.get("target_column") or "") == expected.target,
    }
    for i, phrases in enumerate(expected.questions, 1):
        checks[f"question_{i}"] = any(p.lower() in q for q in captured for p in phrases)
    for a in expected.audiences:
        checks[f"audience:{a}"] = a.lower() in audiences
    return {
        "score": mean(checks.values()),
        "checks": checks,
        "task_type": spec.get("task_type"),
        "target": spec.get("target_column") or "",
        "n_questions": len(captured),
    }


def score_intent(intent: Any, case: IntentCase) -> dict[str, Any]:
    allowed = set(getattr(intent, "allowed_tools", []) or [])
    checks = {
        "decision": getattr(intent, "decision", None) in case.decisions,
        "required_tools": case.required_tools <= allowed if
        getattr(intent, "decision", None) == "act" else True,
        "forbidden_tools": not (case.forbidden_tools & allowed),
    }
    if case.plan_first is not None:
        checks["plan_first"] = bool(getattr(intent, "plan_first", False)) == case.plan_first
    return {"pass": all(checks.values()), "checks": checks,
            "decision": getattr(intent, "decision", None),
            "allowed_tools": sorted(allowed)}


def removed_statements(reply: str) -> int:
    m = re.search(r"Removed (\d+) statement", reply or "")
    return int(m.group(1)) if m else 0


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def summary_markdown(results: dict[str, Any]) -> str:
    lines = [f"# Evaluation {results['run_id']}", "",
             f"Backend: {results['backend']} ({results['model']}). "
             f"Stages: {', '.join(results['stages'])}.", ""]
    intent = results.get("intent")
    if intent:
        passed = sum(r["pass"] for r in intent)
        lines += [f"## Intent: {passed} of {len(intent)} passed", "",
                  "| Query | Session | Decision | Pass | Failed checks |", "|---|---|---|---|---|"]
        for r in intent:
            failed = ", ".join(k for k, v in r["checks"].items() if not v) or "-"
            lines.append(f"| {r['query'][:60]} | {r.get('context', 'fresh')} | {r['decision']} | "
                         f"{_fmt(r['pass'])} | {failed} |")
        lines.append("")
    runs = results.get("cases") or []
    if runs:
        lines += ["## Cases", "",
                  "| Case | Stage | Understanding | Findings recall | Facts correct | "
                  "Steps ok | Removed stmts | Export re-runs | Workspace untouched | "
                  "Minutes | LLM calls |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            p = r.get("process", {})
            f = r.get("findings", {})
            lines.append(
                f"| {r['case']} | {r['stage']} | {_fmt(r['understanding'].get('score'))} | "
                f"{_fmt(f.get('recall'))} | {_fmt(f.get('fact_accuracy'))} | "
                f"{p.get('steps_ok', '-')}/{p.get('steps_total', '-')} | "
                f"{_fmt(p.get('removed_statements'))} | {_fmt(p.get('export_reruns'))} | "
                f"{_fmt(p.get('workspace_untouched'))} | {_fmt(p.get('minutes'))} | "
                f"{_fmt(p.get('llm_calls'))} |")
        lines.append("")
        for r in runs:
            missed = [k for k, v in r["understanding"].get("checks", {}).items() if not v]
            f = r.get("findings", {})
            not_named = [d["entity"] for d in f.get("detail", []) if not d["named"]]
            wrong = [f"{d['entity']}: {k}" for d in f.get("detail", [])
                     for k, ok in d["facts"].items() if not ok]
            if missed or not_named or wrong or r.get("error"):
                lines.append(f"- **{r['case']}**"
                             + (f" error: {r['error']};" if r.get("error") else "")
                             + (f" missed understanding: {', '.join(missed)};" if missed else "")
                             + (f" not named: {', '.join(not_named)};" if not_named else "")
                             + (f" facts not stated correctly: {', '.join(wrong)}" if wrong else ""))
    return "\n".join(lines) + "\n"


def compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Per-case change in the headline scores between two result files."""
    def index(res):
        return {(r["case"], r["stage"]): r for r in res.get("cases") or []}
    a, b = index(old), index(new)
    lines = [f"Compare {old['run_id']} -> {new['run_id']}", ""]
    for key in sorted(set(a) | set(b)):
        ra, rb = a.get(key), b.get(key)
        if not ra or not rb:
            lines.append(f"{key[0]} ({key[1]}): only in {'new' if rb else 'old'} run")
            continue

        def get(r, *path):
            for p in path:
                r = (r or {}).get(p)
            return r
        parts = []
        for label, path in (("understanding", ("understanding", "score")),
                            ("recall", ("findings", "recall")),
                            ("facts", ("findings", "fact_accuracy")),
                            ("minutes", ("process", "minutes"))):
            x, y = get(ra, *path), get(rb, *path)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                parts.append(f"{label} {x:.2f} -> {y:.2f}")
        lines.append(f"{key[0]} ({key[1]}): " + "; ".join(parts))
    oi, ni = old.get("intent") or [], new.get("intent") or []
    if oi and ni:
        lines.append(f"intent: {sum(r['pass'] for r in oi)}/{len(oi)} -> "
                     f"{sum(r['pass'] for r in ni)}/{len(ni)}")
    return "\n".join(lines) + "\n"
