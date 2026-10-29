"""Layer 2 — Decomposed LLM extraction (4 small focused calls).

Replaces the single large LLM call in UnderstandTaskTool with 4 small,
single-question calls. Each call asks exactly one yes/no or single-value
question so think-block truncation cannot silently corrupt the full spec.

Per INSTRUCTION_UNDERSTANDING_FIX.md Section 4.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

# Match the chat client's settings so this module does not undo them: holding
# the model in memory between calls, and a thinking budget for the one call
# that still thinks. See agent/api/llm.py for the reasoning behind both.
_KEEP_ALIVE = os.environ.get("DSAGENT_KEEP_ALIVE", "30m")
_SCOPE_TOKENS = int(os.environ.get("DSAGENT_SCOPE_TOKENS", "2500"))
_DISABLE_THINK = os.environ.get("DSAGENT_THINK", "1") != "1"


@dataclass
class RawSpec:
    task_type: str               # "binary_classification" | "multiclass_classification" |
                                 # "regression" | "timeseries_forecasting" | "clustering" |
                                 # "anomaly_detection" | "survival_analysis" | "unknown"
    task_type_confidence: float  # 0.0 – 1.0
    target_column_hint: str      # column name or "" if not found
    target_condition: str        # e.g. "outcome in [1,2,3]" or ""
    aggregation_needed: bool
    aggregation_key: str         # column name to group by, or ""
    evaluation_metric: str       # "pr_auc" | "roc_auc" | "rmse" | …
    task_description: str        # plain English, 1 sentence
    source: str                  # "llm_decomposed" | "heuristic" | "failed"
    specific_requirements: list = field(default_factory=list)  # explicit steps/features/models named in the brief
    requires_secondary_file: bool = False   # True when target must come from a 2nd file
    # File-role reasoning (populated when multiple files loaded)
    file_role_map: dict = field(default_factory=dict)
    # e.g. {"df": "training_data", "df2": "target_derivation_source"}
    # Roles: "training_data" | "target_derivation_source" | "test_data" | "supplementary" | "unknown"
    clarification_needed: str = ""
    # Non-empty → surface this question to user before proceeding with pipeline
    # The brief in its own terms (objective, questions, deliverables, audiences,
    # constraints, data notes, open questions). Filled by read_brief().
    brief: dict = field(default_factory=dict)


# ── Brief model: one call that reads the brief in its own terms ─────────────
#
# The decomposed extraction below reduces a brief to supervised-learning fields
# (target, metric, aggregation key). A brief that asks four analysis questions
# for three audiences does not fit that shape, and its questions were dropped:
# the requirements list came back empty. read_brief() keeps what the brief asks
# for, and still fills the spec fields the pipeline uses, in one LLM call made
# through the session's client, so it works on the Claude backend as well
# (TaskExtractor posts to Ollama's endpoint directly).

_BRIEF_TOKENS = int(os.environ.get("DSAGENT_BRIEF_TOKENS", "6000"))
_BRIEF_CHARS = int(os.environ.get("DSAGENT_BRIEF_CHARS", "12000"))

BRIEF_TASK_TYPES = (
    "exploratory_analysis", "anomaly_detection", "clustering",
    "binary_classification", "multiclass_classification", "regression",
    "timeseries_forecasting",
)
_BRIEF_METRICS = {"pr_auc", "roc_auc", "accuracy", "f1", "f1_macro", "rmse",
                  "mae", "r2", "mase", "average_precision"}

_BRIEF_PROMPT = """\
You are the lead data scientist reading a task brief before any work starts.
Describe the task in the brief's own terms. Do not plan the work, and do not
add anything the brief does not say.

BRIEF (may contain several documents, for example the task and a data dictionary):
{brief}

DATASET: {n_rows} rows
{columns}

Reply with a single JSON object and nothing else:
{{"objective": "one or two sentences: the decision or outcome the work supports",
  "task_nature": "exploratory | supervised | unsupervised | forecasting",
  "questions": ["every question or task the brief asks, one per item, in its own words and order"],
  "deliverables": ["each output the brief asks for, e.g. a summary, notebook, model, file"],
  "audiences": ["each audience the brief names for the results"],
  "constraints": ["each rule, deadline, tool restriction or disclosure the brief requires"],
  "data_notes": ["column: what it means, from the data dictionary or the brief"],
  "task_type": "one of {task_types}",
  "target_column": "exact column name holding a labelled outcome to predict, or empty",
  "target_condition": "a condition the brief itself states to define a label, or empty",
  "aggregation_key": "column to group rows by when the brief asks for results per entity, or empty",
  "evaluation_metric": "a metric the brief names, or empty",
  "open_questions": ["anything the brief leaves unclear that changes the work"]}}

Rules:
- Copy the brief's questions faithfully. Do not merge them, turn them into a
  different task, or add questions the brief does not ask.
- A supervised task_type needs a labelled outcome: a column in the data, or a
  condition the brief states. Words about risk, fraud or concern do not create
  a label. With no label, use exploratory_analysis when the brief asks you to
  explore and explain the data, anomaly_detection when it asks you to find
  unusual records or entities, and clustering when it asks for groups.
- Leave a field empty ("" or []) when the brief and the data do not determine it.
"""


def _column_summary(df_stats: dict, columns: list[str], limit: int = 40) -> str:
    """One line per column: dtype, distinct values, nulls, and small value sets."""
    stats = df_stats.get("columns", {}) if isinstance(df_stats, dict) else {}
    lines = []
    for col in columns[:limit]:
        s = stats.get(col, {})
        bits = [f"- {col}"]
        if s:
            bits.append(f"({s.get('dtype', '?')}, {s.get('n_unique', '?')} distinct, "
                        f"{100 * float(s.get('null_rate', 0)):.0f}% null)")
            vc = s.get("value_counts") or {}
            if 0 < len(vc) <= 8:
                bits.append("values: " + ", ".join(str(k)[:20] for k in list(vc)[:8]))
        lines.append(" ".join(bits))
    if len(columns) > limit:
        lines.append(f"- ... {len(columns) - limit} more columns")
    return "\n".join(lines)


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def read_brief(llm_client, brief_text: str, df_stats: dict,
               columns: list[str]) -> RawSpec | None:
    """Read the brief in its own terms with one LLM call.

    Returns None when the reply is unusable (no objective or no questions), so
    the caller can fall back to the decomposed extraction. Column-valued fields
    are kept only when they name a real column.
    """
    from agent.api.intent import _json_text

    prompt = _BRIEF_PROMPT.format(
        brief=(brief_text or "")[:_BRIEF_CHARS],
        n_rows=df_stats.get("n_rows", "unknown") if isinstance(df_stats, dict) else "unknown",
        columns=_column_summary(df_stats, columns),
        task_types=" | ".join(BRIEF_TASK_TYPES))
    try:
        raw = llm_client._generate(prompt, temperature=0.0,
                                   max_tokens=_BRIEF_TOKENS, json_mode=True)
        data = json.loads(_json_text(raw))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    objective = str(data.get("objective") or "").strip()
    questions = _str_list(data.get("questions"))
    if not objective or not questions:
        return None

    by_lower = {c.lower(): c for c in columns}

    def _column(key: str) -> str:
        return by_lower.get(str(data.get(key) or "").strip().strip("`").lower(), "")

    nature = str(data.get("task_nature") or "").strip().lower()
    task_type = str(data.get("task_type") or "").strip().lower()
    if task_type not in BRIEF_TASK_TYPES:
        task_type = {"exploratory": "exploratory_analysis",
                     "unsupervised": "anomaly_detection",
                     "forecasting": "timeseries_forecasting"}.get(nature, "unknown")
    metric = str(data.get("evaluation_metric") or "").strip().lower().replace("-", "_")
    agg_key = _column("aggregation_key")
    deliverables = _str_list(data.get("deliverables"))
    brief = {
        "objective": objective,
        "task_nature": nature,
        "questions": questions,
        "deliverables": deliverables,
        "audiences": _str_list(data.get("audiences")),
        "constraints": _str_list(data.get("constraints")),
        "data_notes": _str_list(data.get("data_notes")),
        "open_questions": _str_list(data.get("open_questions")),
    }
    return RawSpec(
        task_type=task_type,
        task_type_confidence=0.85 if task_type != "unknown" else 0.4,
        target_column_hint=_column("target_column"),
        target_condition=str(data.get("target_condition") or "").strip(),
        aggregation_needed=bool(agg_key),
        aggregation_key=agg_key,
        evaluation_metric=metric if metric in _BRIEF_METRICS else "",
        task_description=objective,
        source="brief_model",
        # The questions are what the work must answer; deliverables are what
        # it must hand over. Both become the checklist the agent works to.
        specific_requirements=questions + [f"Deliver: {d}" for d in deliverables],
        brief=brief,
    )


class TaskExtractor:
    """
    Two-phase extraction:
      Phase 0 — deep reasoning scope (qwen3.6:latest, thinking enabled).
                 Produces a structured SCOPE ANALYSIS block with committed values.
      Phase 1 — 5 small decomposed LLM calls that use the scope as fast-path priors.

    Falls back to RawSpec(source="failed") if all calls fail.
    Never raises — always returns a RawSpec.
    """

    THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _SCOPE_ANALYSIS_RE = re.compile(
        r"---\s*SCOPE ANALYSIS\s*---(.+?)---\s*END SCOPE\s*---",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, ollama_base_url: str, model: str,
                 timeout: float = 60.0,
                 scope_model: str | None = None) -> None:
        self.base_url = ollama_base_url.rstrip("/")
        self.model = model
        # scope_model is used for Phase 0 deep reasoning; defaults to model if not set.
        self.scope_model = scope_model or model
        self.timeout = timeout

    # ── public entry point ────────────────────────────────────────────

    def extract(
        self,
        doc: object,           # DocumentContent from doc_parser.py
        df_stats: dict,        # pre-computed from _compute_df_stats()
        column_names: list[str],
        user_hint: str = "",   # last user message — used when re-running after clarification
    ) -> RawSpec:
        """Two-phase extraction with optional file-role reasoning.

        Phase 0: deep reasoning scope using scope_model (thinking enabled).
                 Produces a structured SCOPE ANALYSIS block.
                 Parsed values are used as fast-path priors for Phase 1.
        Phase 0.5 (multi-file only): focused file-role assignment — maps each loaded
                 file (df, df2, …) to a role (training_data, target_derivation_source,
                 test_data, supplementary, unknown). If confidence is low, returns a
                 clarification question for the user instead of proceeding.
        Phase 1: 5 small decomposed extraction calls; each checks scope priors
                 before making its own LLM call.
        """
        context = self._build_context(doc, df_stats, column_names)

        # Phase 0 — deep reasoning
        scope = self._call_problem_scope(context)
        priors = self._parse_scope_analysis(scope)

        enriched = (
            context + "\n\n=== PROBLEM SCOPE ANALYSIS ===\n" + scope
            if scope else context
        )

        # Phase 0.5 — file-role reasoning (only when multiple files loaded)
        file_role_map: dict = {}
        if df_stats.get("secondary_data_files"):
            role_result = self._call_file_roles(doc, df_stats, user_hint)
            clarification = role_result.pop("clarification_needed", None)
            if clarification:
                # Return early — surface clarification to user before full extraction
                return RawSpec(
                    task_type="unknown",
                    task_type_confidence=0.0,
                    target_column_hint="",
                    target_condition="",
                    aggregation_needed=False,
                    aggregation_key="",
                    evaluation_metric="",
                    task_description="",
                    source="clarification_pending",
                    clarification_needed=clarification,
                )
            file_role_map = {
                var: info.get("role", "unknown")
                for var, info in role_result.items()
                if isinstance(info, dict)
            }

        # Phase 1 — decomposed extraction (priors short-circuit LLM calls where confident)
        task_type, confidence, description = self._call_task_type(
            enriched, prior=priors.get("TASK_TYPE")
        )
        target_hint, target_condition = self._call_target(
            enriched, column_names, prior=priors.get("TARGET_COLUMN")
        )
        needs_agg, agg_key = self._call_aggregation(
            enriched, column_names,
            prior_needed=priors.get("AGGREGATION"),
            prior_key=priors.get("AGGREGATION_KEY"),
        )
        metric = self._call_metric(
            enriched, task_type, prior=priors.get("METRIC")
        )
        requirements = self._call_specific_requirements(
            enriched, prior_steps=priors.get("SPECIFIC_STEPS", [])
        )
        # Derive requires_secondary_file from role map when available; else phrase-count
        if file_role_map:
            needs_second_file = any(
                role == "target_derivation_source"
                for role in file_role_map.values()
            )
        else:
            needs_second_file = self._requires_secondary_file(
                enriched, prior=priors.get("SECONDARY_FILE")
            )

        return RawSpec(
            task_type=task_type,
            task_type_confidence=confidence,
            target_column_hint=target_hint,
            target_condition=target_condition,
            aggregation_needed=needs_agg,
            aggregation_key=agg_key,
            evaluation_metric=metric,
            task_description=description,
            source="llm_decomposed",
            specific_requirements=requirements,
            requires_secondary_file=needs_second_file,
            file_role_map=file_role_map,
        )

    # ── shared context builder ────────────────────────────────────────

    def _build_context(
        self, doc: object, df_stats: dict, columns: list[str]
    ) -> str:
        """Compact shared context — all 4 calls use the same prefix."""
        task_section = next(
            (s.body for s in getattr(doc, "sections", [])
             if getattr(s, "section_type", "") == "task"),
            "",
        )
        eval_section = next(
            (s.body for s in getattr(doc, "sections", [])
             if getattr(s, "section_type", "") == "evaluation"),
            "",
        )
        if not task_section:
            task_section = getattr(doc, "raw_text", "")[:1500]

        parts = ["=== TASK INSTRUCTIONS ===", task_section[:2000]]
        if eval_section:
            parts += ["=== EVALUATION CRITERIA ===", eval_section[:500]]

        parts += [
            "=== DATASET STATISTICS ===",
            f"Columns ({len(columns)}): {', '.join(columns[:40])}",
            f"Row count: {df_stats.get('n_rows', 'unknown')}",
        ]

        # Outcome column value-count hints
        outcome_hints = []
        for col, stats in df_stats.get("columns", {}).items():
            if any(kw in col.lower() for kw in [
                "outcome", "result", "status", "label", "target",
                "class", "flag", "indicator", "churn", "default",
                "injury", "incident", "event", "fraud", "fail",
            ]):
                vc = stats.get("value_counts", {})
                if vc:
                    outcome_hints.append(
                        f"  {col}: {dict(list(vc.items())[:6])}"
                    )
        if outcome_hints:
            parts.append("=== POTENTIAL OUTCOME COLUMNS ===")
            parts.extend(outcome_hints)

        # Hints extracted by DocumentParser
        etm = getattr(doc, "explicit_target_mentions", [])
        egk = getattr(doc, "explicit_group_key_mentions", [])
        if etm:
            parts.append(f"Target keywords found near: {etm[:10]}")
        if egk:
            parts.append(f"Aggregation keywords found: {egk[:5]}")

        return "\n".join(parts)

    # ── phase 0: deep problem reasoning ──────────────────────────────

    _SCOPE_PROMPT = """\
Read the task brief and dataset information above very carefully.

Reason step-by-step about the problem, then commit your analysis using EXACTLY \
this format. Write none for any field the brief and the data do not determine; \
never invent a value to fill a field:

--- SCOPE ANALYSIS ---
BUSINESS_GOAL: <one sentence: what outcome the organisation needs>
ENTITY_UNIT: per <entity type, e.g. establishment / customer / patient / loan>
TASK_TYPE: <binary_classification | multiclass_classification | regression | timeseries_forecasting | anomaly_detection | clustering>
TASK_TYPE_REASON: <cite the EXACT phrase from the brief that determines this>
TARGET_COLUMN: <exact column name if already in data, OR "DERIVE: <condition the brief states>", OR none>
AGGREGATION: <needed | not_needed>
AGGREGATION_KEY: <column name to group by, or none>
METRIC: <pr_auc | roc_auc | f1 | rmse | mae | r2 | mase, or none — cite source or justify>
SECONDARY_FILE: <yes | no>
SECONDARY_FILE_REASON: <which file holds the target labels, or "n/a">
SPECIFIC_STEPS: <pipe-separated list of the deliverables the brief explicitly asks for, in its own words, or none>
--- END SCOPE ---

RULES:
- Supervised learning needs a labelled outcome. Choose a classification or regression
  type only when the data holds the outcome column, or the brief itself states the
  condition that defines it. Words about risk, fraud or concern do not create a label.
  With no labelled outcome, the task is anomaly_detection (finding unusual records or
  entities) or clustering, and TARGET_COLUMN is none.
- Aggregation is needed only when the brief asks for results per entity while the
  rows are finer grained. The key is then the entity's identifier, never a category
  code (naics_code, sic_code, zip_code).
- A binary flag derived from a FUTURE dataset = requires_secondary_file = yes
- If the brief names a specific metric (PR-AUC, ROC-AUC, RMSE), use that EXACTLY"""

    def _call_problem_scope(self, context: str) -> str:
        """Phase 0: deep chain-of-thought reasoning → structured SCOPE ANALYSIS block.

        Uses scope_model with thinking enabled (no /no_think) so the model
        can reason fully before the structured extraction calls begin.
        Returns the structured scope text; empty string on failure.
        """
        prompt = f"{context}\n\n{self._SCOPE_PROMPT}"
        raw = self._llm_call_think(prompt, max_tokens=_SCOPE_TOKENS)
        return raw or ""

    def _parse_scope_analysis(self, scope_text: str) -> dict:
        """Extract committed values from the structured SCOPE ANALYSIS block.

        Returns a dict with keys matching the SCOPE ANALYSIS fields.
        Empty dict if the block is not present or malformed.
        """
        if not scope_text:
            return {}
        m = self._SCOPE_ANALYSIS_RE.search(scope_text)
        block = m.group(1) if m else scope_text  # fall back to full text if markers absent

        priors: dict = {}
        for line in block.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if not val or val.startswith("<"):
                continue  # placeholder — skip
            if val.strip(" .\"'`").lower() in ("none", "n/a", "na", "null", "not applicable"):
                # An explicit "none" is an answer: nothing to use as a prior.
                if key == "AGGREGATION":
                    priors["AGGREGATION"] = "not_needed"
                continue

            if key == "TASK_TYPE":
                # Normalise to known values
                known = {
                    "binary": "binary_classification",
                    "multiclass": "multiclass_classification",
                    "regression": "regression",
                    "timeseries": "timeseries_forecasting",
                    "anomaly": "anomaly_detection",
                }
                vl = val.lower()
                for fragment, canonical in known.items():
                    if fragment in vl:
                        priors["TASK_TYPE"] = canonical
                        break
                else:
                    if vl in ("binary_classification", "multiclass_classification",
                               "regression", "timeseries_forecasting",
                               "anomaly_detection", "clustering"):
                        priors["TASK_TYPE"] = vl

            elif key == "TARGET_COLUMN":
                priors["TARGET_COLUMN"] = val

            elif key == "AGGREGATION":
                vl = val.lower()
                priors["AGGREGATION"] = (
                    "not_needed" if "not_needed" in vl or "not needed" in vl or vl == "not"
                    else "needed"
                )

            elif key == "AGGREGATION_KEY":
                if val.lower() not in ("none", "n/a", ""):
                    priors["AGGREGATION_KEY"] = val

            elif key == "METRIC":
                # Map free-form to canonical
                metric_map = {
                    "pr_auc": "pr_auc", "pr-auc": "pr_auc",
                    "precision-recall": "pr_auc", "average precision": "pr_auc",
                    "roc_auc": "roc_auc", "roc-auc": "roc_auc",
                    "f1": "f1", "rmse": "rmse", "mae": "mae",
                    "r2": "r2", "mase": "mase",
                }
                vl = val.lower()
                for fragment, canonical in metric_map.items():
                    if fragment in vl:
                        priors["METRIC"] = canonical
                        break

            elif key == "SECONDARY_FILE":
                priors["SECONDARY_FILE"] = val.lower().startswith("y")

            elif key == "SPECIFIC_STEPS":
                steps = [s.strip() for s in val.split("|") if s.strip()]
                if steps:
                    priors["SPECIFIC_STEPS"] = steps

        return priors

    def _llm_call_think(self, prompt: str, max_tokens: int = 700) -> str:
        """Ollama call with thinking enabled — uses scope_model, longer timeout.

        This is the one call in the agent that still thinks, because the scope
        analysis it produces seeds every later extraction. Thinking tokens are
        generated at the same rate as answer tokens, so the budget is the cost:
        on a local model at ~13 tok/s, 1500 tokens is about two minutes for this
        single call. 700 keeps the reasoning and halves the wait. Raise
        DSAGENT_SCOPE_TOKENS if scope analyses come back truncated.
        """
        try:
            r = httpx.post(
                # /api/chat, because a harmony-format reasoning model returns an
                # empty `response` on /api/generate and puts everything in its
                # thinking channel. See agent/api/llm.py._generate.
                f"{self.base_url}/api/chat",
                json={
                    "model": self.scope_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": _KEEP_ALIVE,
                    "options": {"num_predict": max_tokens, "temperature": 0.1},
                },
                timeout=self.timeout * 3,
            )
            r.raise_for_status()
            raw = ((r.json().get("message") or {}).get("content") or "").strip()
            # If thinking happened, keep only the post-think response
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            elif "<think>" in raw:
                raw = raw.split("<think>", 1)[0].strip()
            return raw
        except Exception:
            return ""

    # ── call 1: task type ─────────────────────────────────────────────

    def _call_task_type(
        self, context: str, prior: str | None = None
    ) -> tuple[str, float, str]:
        # Fast path: the scope analysis already committed to a task type. The
        # task type is the model's reading of the brief, not a phrase match:
        # a keyword list here turned "high risk" or "at least one" into binary
        # classification even when the data held no label to classify.
        if prior and prior in (
            "binary_classification", "multiclass_classification",
            "regression", "timeseries_forecasting", "anomaly_detection",
            "clustering",
        ):
            return (prior, 0.90, "")

        prompt = (
            f"{context}\n\n"
            "Based only on the information above, answer:\n"
            "IMPORTANT classification rules:\n"
            "1. If the instructions say 'binary', 'two-class', 'yes/no', '0 or 1', "
            "return binary_classification.\n"
            "2. If the instructions say 'predict WHICH [entities] will/are likely to "
            "[experience some event]' (e.g. 'predict which businesses will have a "
            "serious harm event') → this is ALWAYS binary_classification (each entity "
            "either has the event or it does not).\n"
            "3. If a binary flag is created from a future dataset to be the target, "
            "that is binary_classification regardless of how many outcome categories "
            "exist in the historical data.\n"
            "4. Only return multiclass_classification when there are 3+ explicitly "
            "named outcome classes that the model should distinguish between.\n"
            "Return ONLY a JSON object — no markdown, no explanation:\n"
            '{"task_type": "<one of: binary_classification, '
            "multiclass_classification, regression, timeseries_forecasting, "
            'anomaly_detection, clustering, survival_analysis, unknown>", '
            '"confidence": <float 0.0 to 1.0>, '
            '"description": "<one sentence: what should be predicted and for whom>"}'
        )
        raw = self._llm_call(prompt, max_tokens=150)
        p = self._safe_json(raw)
        return (
            p.get("task_type", "unknown"),
            float(p.get("confidence", 0.0)),
            p.get("description", ""),
        )

    # ── call 2: target column ─────────────────────────────────────────

    def _call_target(
        self, context: str, columns: list[str], prior: str | None = None
    ) -> tuple[str, str]:
        # Use scope prior if it matched a real column (not a DERIVE: placeholder)
        if prior and not prior.upper().startswith("DERIVE:"):
            col_lower = {c.lower(): c for c in columns}
            matched = col_lower.get(prior.lower())
            if matched:
                return (matched, "")
        # Prior is a DERIVE condition — extract condition and return empty col
        if prior and prior.upper().startswith("DERIVE:"):
            condition = prior[len("DERIVE:"):].strip()
            return ("", condition)

        col_list = ", ".join(f'"{c}"' for c in columns[:60])
        prompt = (
            f"{context}\n\n"
            f"Available column names: [{col_list}]\n\n"
            "Which column is the TARGET (what should be predicted)?\n"
            "If a condition is needed (e.g. 'outcome in [1,2,3] means serious harm'), "
            "state the condition too.\n"
            "Return ONLY a JSON object:\n"
            '{"target_column": "<exact column name from the list, or empty string>", '
            '"target_condition": "<condition string, or empty string>", '
            '"reasoning": "<one sentence why>"}'
        )
        raw = self._llm_call(prompt, max_tokens=180)
        p = self._safe_json(raw)
        return (p.get("target_column", ""), p.get("target_condition", ""))

    # ── call 3: aggregation need ──────────────────────────────────────

    # Entity-identifier column patterns that should be preferred as aggregation keys.
    # These are high-cardinality identifiers, not category/classification codes.
    _ENTITY_ID_KEYWORDS = [
        "establishment_id", "establishment_id", "business_id", "customer_id",
        "client_id", "patient_id", "account_id", "company_id", "employer_id",
        "entity_id", "firm_id", "store_id", "branch_id", "location_id",
        "_id",
    ]
    _CATEGORY_CODE_KEYWORDS = [
        "naics", "sic", "nace", "isco", "isic", "industry_code",
        "sector_code", "zip_code", "postal_code",
    ]

    def _preferred_entity_column(self, columns: list[str]) -> str:
        """Return the first column that looks like a business/entity identifier."""
        col_lower = {c: c.lower() for c in columns}
        for c, cl in col_lower.items():
            if any(kw in cl for kw in self._ENTITY_ID_KEYWORDS):
                return c
        return ""

    def _call_aggregation(
        self, context: str, columns: list[str],
        prior_needed: str | None = None,
        prior_key: str | None = None,
    ) -> tuple[bool, str]:
        # The scope already decided no aggregation: asking again with an
        # entity-column hint only invites a key the task does not need.
        if prior_needed == "not_needed":
            return (False, "")
        # Use scope priors if both aggregation decision and key are committed
        if prior_needed is not None and prior_key:
            needed = prior_needed == "needed"
            # Validate key exists in columns (case-insensitive)
            col_lower = {c.lower(): c for c in columns}
            matched_key = col_lower.get(prior_key.lower(), prior_key)
            # Guard: never accept a category code if an entity ID exists
            entity_hint = self._preferred_entity_column(columns)
            if entity_hint and any(
                kw in matched_key.lower() for kw in self._CATEGORY_CODE_KEYWORDS
            ):
                matched_key = entity_hint
            return (needed, matched_key if needed else "")

        entity_hint = self._preferred_entity_column(columns)
        entity_note = (
            f"\nHint: The column '{entity_hint}' looks like a business/entity identifier "
            f"and is likely the correct grouping key."
            if entity_hint else ""
        )
        col_list = ", ".join(f'"{c}"' for c in columns[:60])
        prompt = (
            f"{context}\n\n"
            f"Available column names: [{col_list}]\n\n"
            "Does the analysis require AGGREGATING rows before modelling?\n"
            "Example: 'each row is an incident but the model should predict "
            "risk per establishment' means aggregation IS needed.\n"
            "IMPORTANT: When choosing the aggregation key, PREFER business/entity "
            "identifier columns (e.g. establishment_id, customer_id) over "
            "industry/classification codes (e.g. naics_code, sic_code, zip_code). "
            "The grouping key should produce ONE ROW PER BUSINESS/ENTITY, not one "
            "row per category."
            f"{entity_note}\n"
            "Return ONLY a JSON object:\n"
            '{"aggregation_needed": <true or false>, '
            '"aggregation_key": "<column name to group by, or empty string>", '
            '"reasoning": "<one sentence why>"}'
        )
        raw = self._llm_call(prompt, max_tokens=150)
        p = self._safe_json(raw)
        agg_key = p.get("aggregation_key", "")
        # Hard override: if LLM picked a category code but an entity ID exists, use it
        if agg_key and entity_hint:
            agg_lower = agg_key.lower()
            if any(kw in agg_lower for kw in self._CATEGORY_CODE_KEYWORDS):
                agg_key = entity_hint
        return (bool(p.get("aggregation_needed", False)), agg_key)

    # ── call 4: evaluation metric ─────────────────────────────────────

    # Explicit metric phrases → canonical name (checked against raw context text)
    _EXPLICIT_METRIC_PHRASES: list[tuple[str, str]] = [
        ("pr-auc", "pr_auc"),
        ("pr auc", "pr_auc"),
        ("precision-recall auc", "pr_auc"),
        ("precision recall auc", "pr_auc"),
        ("average precision", "pr_auc"),
        ("roc-auc", "roc_auc"),
        ("roc auc", "roc_auc"),
        ("area under the roc", "roc_auc"),
        ("area under roc", "roc_auc"),
        ("rmse", "rmse"),
        ("root mean square", "rmse"),
        ("mean absolute error", "mae"),
        ("r-squared", "r2"),
        ("r2 score", "r2"),
        ("f1 score", "f1"),
        ("f1-score", "f1"),
    ]

    def _scan_explicit_metric(self, context: str) -> str | None:
        """Return canonical metric name if the context explicitly names one."""
        ctx_lower = context.lower()
        for phrase, canonical in self._EXPLICIT_METRIC_PHRASES:
            if phrase in ctx_lower:
                return canonical
        return None

    def _call_metric(self, context: str, task_type: str,
                     prior: str | None = None) -> str:
        # Fast path 1: instructions explicitly name a metric — highest trust
        explicit = self._scan_explicit_metric(context)
        if explicit is not None:
            return explicit
        # Unlabelled tasks have no ground truth to score against, so a
        # supervised metric such as pr_auc would be invented, not chosen.
        if task_type in ("anomaly_detection", "clustering"):
            return ""
        # Fast path 2: scope analysis committed to a metric
        if prior:
            return prior

        prompt = (
            f"{context}\n\n"
            f"Task type: {task_type}\n\n"
            "What evaluation metric should be used?\n"
            "FIRST: check if the task instructions explicitly name a metric "
            "(e.g. 'use PR-AUC', 'evaluate with ROC-AUC'). If yes, use THAT metric.\n"
            "If no metric is named: prefer pr_auc when positive cases are rare "
            "(minority class < 20%); otherwise use roc_auc for classification.\n"
            "Return ONLY a JSON object:\n"
            '{"metric": "<one of: pr_auc, roc_auc, accuracy, f1, rmse, mae, r2, mase>", '
            '"reasoning": "<one sentence why>"}'
        )
        raw = self._llm_call(prompt, max_tokens=100)
        p = self._safe_json(raw)
        return p.get("metric", "roc_auc")

    # ── secondary file detection ──────────────────────────────────────

    _SECONDARY_FILE_PHRASES = [
        "second file", "second dataset", "another file", "another dataset",
        "separate file", "separate dataset", "additional file",
        "test file", "test dataset", "future data", "q1", "q2", "q3", "q4",
        "2024", "next year", "following year", "future period",
        "create your target", "create the target", "derive the target",
        "target variable", "label from", "labels from",
    ]

    def _requires_secondary_file(self, context: str,
                                  prior: bool | None = None) -> bool:
        """Return True when the context strongly implies the target needs a 2nd file."""
        if prior is not None:
            return prior
        ctx_lower = context.lower()
        phrase_count = sum(1 for p in self._SECONDARY_FILE_PHRASES if p in ctx_lower)
        return phrase_count >= 2

    # ── phase 0.5: file-role reasoning ───────────────────────────────

    _FILE_ROLE_PROMPT = """\
You are determining the role of each loaded dataset for a data science task.

TASK BRIEF (excerpt):
{brief_excerpt}

LOADED FILES:
{file_descriptions}
{user_hint_section}
Assign exactly one role to each file:
  "training_data"             — primary dataset used to train the model
  "target_derivation_source"  — future-period or separate file from which the
                                 binary target label is derived
  "test_data"                 — held-out evaluation data (no labels yet)
  "supplementary"             — reference/lookup data not used directly for training
  "unknown"                   — cannot determine from available information

Reply with ONLY this JSON (one key per loaded file, plus clarification_needed):
{{
{file_json_keys}
  "clarification_needed": null
}}

If confidence < 0.65 for ANY file, set clarification_needed to a clear,
specific question that names each file with its row count and a brief
description of its columns, e.g.:
"I can see two files: df (15,000 rows, last year's incident data) and df2
(4,000 rows, similar columns). The brief mentions deriving a target from the
next quarter's data. Is df2 that file, or a different dataset?"

Otherwise set clarification_needed to null.
"""

    def _call_file_roles(
        self, doc: object, df_stats: dict, user_hint: str = ""
    ) -> dict:
        """Phase 0.5: LLM assigns a role to each loaded file.

        Returns a dict like::

            {
                "df":  {"role": "training_data",           "confidence": 0.9,  "reason": "..."},
                "df2": {"role": "target_derivation_source","confidence": 0.85, "reason": "..."},
                "clarification_needed": null
            }

        If confidence < 0.65 for any file, "clarification_needed" is a
        human-readable question string.  On any LLM/parse failure returns {}
        so the caller falls back to phrase-counting.
        """
        # ── 1. build brief excerpt ────────────────────────────────────
        brief_text = ""
        task_section = next(
            (s.body for s in getattr(doc, "sections", [])
             if getattr(s, "section_type", "") == "task"),
            "",
        )
        if task_section:
            brief_text = task_section[:800]
        else:
            brief_text = getattr(doc, "raw_text", "")[:800]

        # ── 2. build per-file descriptions ────────────────────────────
        primary_cols = list(df_stats.get("columns", {}).keys())[:20]
        primary_n_rows = df_stats.get("n_rows", "?")
        primary_filename = df_stats.get("filename", "primary file")
        primary_date_range = df_stats.get("date_range", "")

        def _file_desc(var: str, filename: str, n_rows, n_cols, cols: list,
                       date_range: str = "") -> str:
            col_preview = ", ".join(cols[:20])
            date_part = f"\n  Date range: {date_range}" if date_range else ""
            return (
                f"  {var} — \"{filename}\" | {n_rows:,} rows × {n_cols} cols\n"
                f"  Columns: {col_preview}{date_part}"
                if isinstance(n_rows, int) else
                f"  {var} — \"{filename}\" | {n_rows} rows × {n_cols} cols\n"
                f"  Columns: {col_preview}{date_part}"
            )

        n_primary_cols = len(df_stats.get("columns", {})) or len(primary_cols)
        descriptions = [
            _file_desc("df", primary_filename, primary_n_rows,
                       n_primary_cols, primary_cols, primary_date_range)
        ]
        file_vars = ["df"]

        sec_files: dict = df_stats.get("secondary_data_files", {})
        for var, info in sec_files.items():
            descriptions.append(
                _file_desc(
                    var,
                    info.get("filename", var),
                    info.get("n_rows", "?"),
                    info.get("n_cols", len(info.get("columns", []))),
                    info.get("columns", []),
                    info.get("date_range", ""),
                )
            )
            file_vars.append(var)

        # ── 3. build prompt ───────────────────────────────────────────
        file_json_keys = "\n".join(
            f'  "{v}": {{"role": "...", "confidence": 0.0, "reason": "..."}},'
            for v in file_vars
        )
        user_hint_section = (
            f"\nUSER CLARIFICATION: {user_hint}\n" if user_hint.strip() else ""
        )
        prompt = self._FILE_ROLE_PROMPT.format(
            brief_excerpt=brief_text,
            file_descriptions="\n".join(descriptions),
            user_hint_section=user_hint_section,
            file_json_keys=file_json_keys,
        )

        # ── 4. LLM call ───────────────────────────────────────────────
        raw = self._llm_call(prompt, max_tokens=600)
        if not raw:
            return {}

        # ── 5. parse response ─────────────────────────────────────────
        # _safe_json only handles known keys; parse manually for role map
        try:
            # Strip markdown fences + think blocks (already done by _llm_call)
            cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1:
                return {}
            data = json.loads(cleaned[start: end + 1])
        except (json.JSONDecodeError, ValueError):
            return {}

        # Validate: each file var should have a role dict
        result: dict = {}
        any_low_confidence = False
        for var in file_vars:
            entry = data.get(var)
            if not isinstance(entry, dict):
                return {}   # malformed — fall back
            role = entry.get("role", "unknown")
            confidence = float(entry.get("confidence", 0.0))
            reason = entry.get("reason", "")
            result[var] = {"role": role, "confidence": confidence, "reason": reason}
            if confidence < 0.65:
                any_low_confidence = True

        # Surface clarification_needed when LLM set it OR any confidence too low
        clar = data.get("clarification_needed")
        if isinstance(clar, str) and clar.strip():
            result["clarification_needed"] = clar.strip()
        elif any_low_confidence:
            # Build a fallback clarification question
            file_summaries = []
            for var in file_vars:
                info = result[var]
                if var == "df":
                    n = primary_n_rows
                    cols_hint = ", ".join(primary_cols[:5])
                else:
                    sec = sec_files.get(var, {})
                    n = sec.get("n_rows", "?")
                    cols_hint = ", ".join(sec.get("columns", [])[:5])
                file_summaries.append(f"{var} ({n} rows, columns: {cols_hint}…)")
            result["clarification_needed"] = (
                "I loaded multiple files but I'm not fully confident about their roles. "
                "Can you confirm which file is the training dataset and which (if any) "
                "is used to derive the target label?\n"
                + "\n".join(f"  • {s}" for s in file_summaries)
            )
        else:
            result["clarification_needed"] = None

        return result

    # ── call 5: specific requirements ────────────────────────────────

    def _call_specific_requirements(self, context: str,
                                     prior_steps: list | None = None) -> list:
        """Extract explicit task requirements that go beyond the standard pipeline."""
        # Use scope prior if it provided non-trivial steps
        if prior_steps:
            return prior_steps

        prompt = (
            f"{context}\n\n"
            "List any SPECIFIC requirements from the task instructions that must be performed.\n"
            "Include: specific columns/features to analyse, specific EDA analyses, "
            "specific models to compare, specific evaluation steps, output format requirements.\n"
            "Only include requirements explicitly stated — do not invent extras.\n"
            "Return ONLY a JSON object:\n"
            '{"requirements": ["<requirement 1>", "<requirement 2>"]}'
        )
        raw = self._llm_call(prompt, max_tokens=300)
        p = self._safe_json(raw)
        reqs = p.get("requirements", [])
        if isinstance(reqs, list):
            return [str(r) for r in reqs if r]
        return []

    # ── LLM call wrapper ──────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 700) -> str:
        """POST to Ollama, strip <think> blocks. Returns "" on any failure."""
        body = {
            "model": self.model,
            # /api/chat shape; see _llm_call_think for why.
            "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
            "stream": False,
            "format": "json",
            "keep_alive": _KEEP_ALIVE,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
        # Only suppress thinking when the configured model benefits from it.
        # A reasoning model given `think: false` returns its raw working instead
        # of an answer, which blanked every extraction call. See
        # agent/api/llm.py DISABLE_THINK.
        if _DISABLE_THINK:
            body["think"] = False
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body,
                           timeout=self.timeout)
            if r.status_code == 400:
                # Older Ollama rejects `think`. Retry without it rather than
                # returning "" and silently losing the extraction.
                r = httpx.post(
                    f"{self.base_url}/api/chat",
                    json={k: v for k, v in body.items() if k != "think"},
                    timeout=self.timeout)
            r.raise_for_status()
            raw = ((r.json().get("message") or {}).get("content") or "").strip()
            # Strip closed <think>...</think> blocks
            raw = self.THINK_PATTERN.sub("", raw).strip()
            # Strip unclosed <think> tail
            if "<think>" in raw:
                raw = raw.split("<think>", 1)[0].strip()
            return raw
        except Exception:
            return ""

    # ── robust JSON extraction ────────────────────────────────────────

    def _safe_json(self, text: str) -> dict:
        """4-attempt JSON extraction. Never raises, returns {} on total failure."""
        if not text:
            return {}
        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2. Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 3. First {...} block
        m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # 4. Key-by-key extraction
        result: dict = {}
        for key in [
            "task_type", "target_column", "target_condition",
            "aggregation_needed", "aggregation_key", "metric",
            "confidence", "reasoning", "description", "requirements",
        ]:
            pat = rf'"{key}"\s*:\s*(.+?)(?:,|\}})'
            km = re.search(pat, cleaned)
            if km:
                raw_val = km.group(1).strip().strip('"')
                if raw_val.lower() == "true":
                    result[key] = True
                elif raw_val.lower() == "false":
                    result[key] = False
                else:
                    try:
                        result[key] = float(raw_val)
                    except ValueError:
                        result[key] = raw_val
        return result
