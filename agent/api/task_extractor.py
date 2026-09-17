"""Layer 2 — Decomposed LLM extraction (4 small focused calls).

Replaces the single large LLM call in UnderstandTaskTool with 4 small,
single-question calls. Each call asks exactly one yes/no or single-value
question so think-block truncation cannot silently corrupt the full spec.

Per INSTRUCTION_UNDERSTANDING_FIX.md Section 4.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx


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

    BINARY_KEYWORDS = [
        # Explicit binary labels
        "binary classification", "binary outcome", "binary target",
        "two-class", "two class", "0 or 1", "0/1",
        "yes or no", "predict whether", "predict if",
        "dichotomous", "positive or negative", "presence or absence",
        "binary label", "binary response", "binary variable",
        # "predict which [entities] will/are likely to [event]" — always binary
        "predict which", "likely to have", "likely to experience",
        "at risk of", "will have a", "will experience",
        "serious harm", "serious injury", "serious incident",
        # Flag / indicator language
        "create a flag", "binary flag", "create a binary",
        "had at least one", "at least one",
        "indicating whether", "indicate whether",
        # Risk / occurrence framing
        "will occur", "will happen", "is likely to",
        "risk of harm", "risk of injury", "high risk",
    ]

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
this format (fill every field — do not leave blanks):

--- SCOPE ANALYSIS ---
BUSINESS_GOAL: <one sentence: what outcome the organisation needs>
ENTITY_UNIT: per <entity type, e.g. establishment / customer / patient / loan>
TASK_TYPE: <binary_classification | multiclass_classification | regression | timeseries_forecasting | anomaly_detection>
TASK_TYPE_REASON: <cite the EXACT phrase from the brief that determines this>
TARGET_COLUMN: <exact column name if already in data, OR "DERIVE: <condition>">
AGGREGATION: <needed | not_needed>
AGGREGATION_KEY: <column name to group by, e.g. establishment_id — NOT a category code like naics_code>
METRIC: <pr_auc | roc_auc | f1 | rmse | mae | r2 | mase — cite source or justify>
SECONDARY_FILE: <yes | no>
SECONDARY_FILE_REASON: <which file holds the target labels, or "n/a">
SPECIFIC_STEPS: <pipe-separated list of non-standard steps, e.g. risk_ranking | output_csv | shap_values>
--- END SCOPE ---

MANDATORY RULES you must follow:
- "predict which [entities] will [event]" = ALWAYS binary_classification (one prediction per entity)
- "at risk of", "likely to have", "serious harm event" = ALWAYS binary_classification
- If the raw data has one row per incident/transaction but prediction is per business/customer,
  aggregation IS needed and the key is the entity ID (e.g. establishment_id, customer_id)
- NEVER use a category code (naics_code, sic_code, zip_code) as aggregation key when an entity ID exists
- A binary flag derived from a FUTURE dataset = requires_secondary_file = yes
- If the brief names a specific metric (PR-AUC, ROC-AUC, RMSE), use that EXACTLY"""

    def _call_problem_scope(self, context: str) -> str:
        """Phase 0: deep chain-of-thought reasoning → structured SCOPE ANALYSIS block.

        Uses scope_model with thinking enabled (no /no_think) so the model
        can reason fully before the structured extraction calls begin.
        Returns the structured scope text; empty string on failure.
        """
        prompt = f"{context}\n\n{self._SCOPE_PROMPT}"
        raw = self._llm_call_think(prompt, max_tokens=1500)
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

    def _llm_call_think(self, prompt: str, max_tokens: int = 1500) -> str:
        """Ollama call with thinking enabled — uses scope_model, longer timeout."""
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.scope_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.1},
                },
                timeout=self.timeout * 3,
            )
            r.raise_for_status()
            raw = (r.json().get("response") or "").strip()
            # If thinking happened, keep only the post-think response
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            elif "<think>" in raw:
                raw = raw.split("<think>", 1)[0].strip()
            return raw
        except Exception:
            return ""

    # ── call 1: task type ─────────────────────────────────────────────

    def _keyword_binary_check(self, context: str) -> tuple[str, float] | None:
        """Return ("binary_classification", 0.92) if strong binary keywords found."""
        ctx_lower = context.lower()
        for kw in self.BINARY_KEYWORDS:
            if kw in ctx_lower:
                return ("binary_classification", 0.92)
        return None

    def _call_task_type(
        self, context: str, prior: str | None = None
    ) -> tuple[str, float, str]:
        # Fast path 1: explicit binary keywords in the document text
        keyword_result = self._keyword_binary_check(context)
        if keyword_result is not None:
            task_type, confidence = keyword_result
            return (task_type, confidence, "")

        # Fast path 2: scope analysis already committed to a task type
        if prior and prior in (
            "binary_classification", "multiclass_classification",
            "regression", "timeseries_forecasting", "anomaly_detection",
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
"I can see two files: df (14,533 rows, 2023 incident data) and df2 (3,800 rows,
similar columns). The brief mentions deriving a target from 2024 Q1 data.
Is df2 the 2024 Q1 file for target derivation, or a different dataset?"

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

    def _llm_call(self, prompt: str, max_tokens: int = 200) -> str:
        """POST to Ollama, strip <think> blocks. Returns "" on any failure."""
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"/no_think\n{prompt}",
                    "stream": False,
                    "format": "json",
                    "options": {"num_predict": max_tokens, "temperature": 0.0},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            raw = (r.json().get("response") or "").strip()
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
