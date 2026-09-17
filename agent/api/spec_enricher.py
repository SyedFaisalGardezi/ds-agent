"""Layer 3 — Data-Driven Spec Enrichment (zero LLM calls).

Fills every None / low-confidence field in a RawSpec using DataFrame
statistics alone. After this layer every required field must be non-empty.

Per INSTRUCTION_UNDERSTANDING_FIX.md Section 5 (v1.1 hardening).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from agent.api.task_extractor import RawSpec


@dataclass
class EnrichedSpec:
    task_type: str
    task_type_confidence: float
    target_column: str           # resolved column name — never empty after this layer
    target_condition: str        # condition string or ""
    aggregation_needed: bool
    aggregation_key: str
    evaluation_metric: str
    task_description: str
    class_imbalance_ratio: float | None   # minority/majority; None for regression
    enrichment_notes: list[str]


class SpecEnricher:
    """
    Pure Python, zero LLM. Deterministic.
    Every None field in RawSpec becomes a concrete value here.
    """

    OUTCOME_PATTERNS = [
        "outcome", "result", "status", "label", "target", "class",
        "flag", "indicator", "churn", "default", "injury", "incident_type",
        "severity", "risk", "fraud", "fail", "success", "event",
        "readmission", "recurrence", "response", "dependent",
        "mortality", "survival", "claim", "complaint",
    ]

    GROUP_KEY_PATTERNS = [
        "_id", "id_", "identifier", "code", "number", "no_",
        "establishment", "entity", "customer", "account", "patient",
        "employee", "company", "organisation", "organization",
        "business", "location", "site", "facility", "store",
        "branch", "unit", "group", "cohort",
    ]

    def enrich(
        self,
        raw_spec: RawSpec,
        df: pd.DataFrame | None,
        column_names: list[str],
    ) -> EnrichedSpec:
        notes: list[str] = []
        task_type = raw_spec.task_type
        target_col = raw_spec.target_column_hint
        target_cond = raw_spec.target_condition
        agg_needed = raw_spec.aggregation_needed
        agg_key = raw_spec.aggregation_key
        metric = raw_spec.evaluation_metric
        task_desc = raw_spec.task_description
        confidence = raw_spec.task_type_confidence
        imbalance_ratio: float | None = None

        # ── 1. Resolve target column ──────────────────────────────────
        if not target_col or target_col not in column_names:
            target_col, reason = self._infer_target_from_data(df, column_names)
            notes.append(f"Target auto-detected: {target_col} ({reason})")
        else:
            # Reject group-key columns supplied as target
            if self._looks_like_group_key(target_col):
                old = target_col
                target_col, reason = self._infer_target_from_data(
                    df, column_names, exclude=[old]
                )
                notes.append(
                    f"Supplied target '{old}' looks like a group key — "
                    f"re-detected: {target_col} ({reason})"
                )

        # ── 2. Detect aggregation need ────────────────────────────────
        if not agg_needed:
            detected_key, ratio = self._detect_aggregation_need(df, column_names)
            if detected_key and ratio > 1.5:
                agg_needed = True
                agg_key = detected_key
                notes.append(
                    f"Aggregation auto-detected: '{detected_key}' has avg "
                    f"{ratio:.1f} rows/group → aggregate first"
                )

        # ── 3. Class imbalance ────────────────────────────────────────
        # Skip raw-data class counting when the target is an aggregation-derived
        # column (e.g. incident_outcome_max, severity_mean).  The raw column
        # does not reflect post-aggregation cardinality, so counting its unique
        # values would incorrectly flag a binary task as multiclass.
        _AGG_SUFFIXES = ("_max", "_min", "_mean", "_sum", "_count",
                         "_std", "_median", "_mode", "_first", "_last")
        _target_is_derived = (
            agg_needed
            and target_col
            and any(target_col.lower().endswith(s) for s in _AGG_SUFFIXES)
        )
        if (task_type in ("binary_classification", "multiclass_classification",
                          "unknown")
                and df is not None
                and target_col
                and target_col in df.columns
                and not _target_is_derived):
            try:
                y = (self._apply_condition(df, target_col, target_cond)
                     if target_cond else df[target_col].dropna())
                vc = y.value_counts()
                if len(vc) == 2:
                    imbalance_ratio = float(vc.min() / vc.max())
                    if task_type == "unknown":
                        task_type = "binary_classification"
                        confidence = 0.85
                        notes.append(
                            f"Task type → binary_classification "
                            f"(imbalance={imbalance_ratio:.3f})"
                        )
                elif len(vc) > 2 and task_type == "unknown":
                    task_type = "multiclass_classification"
                    confidence = 0.75
                    notes.append(
                        f"Task type → multiclass_classification ({len(vc)} classes)"
                    )
            except Exception as exc:
                notes.append(f"Class balance check failed: {exc}")
        elif _target_is_derived:
            notes.append(
                f"Class balance check skipped: '{target_col}' is aggregation-derived "
                f"— raw cardinality does not reflect post-aggregation target distribution"
            )

        # ── 4. Metric override (sample-size-aware) ────────────────────
        # v1.1: PR-AUC is only preferred over ROC-AUC when the minority
        # sample is large enough for PR curves to be stable (Saito 2015).
        if (task_type == "binary_classification"
                and imbalance_ratio is not None
                and metric in ("roc_auc", "accuracy", "")):
            n_rows = len(df) if df is not None else 0
            n_minority = int(n_rows * imbalance_ratio / (1 + imbalance_ratio))

            if n_minority >= 100:
                threshold, rationale = 0.20, "standard threshold"
            elif n_minority >= 30:
                threshold, rationale = 0.10, f"moderate n_minority≈{n_minority}"
            else:
                threshold, rationale = 0.0, f"n_minority≈{n_minority} < 30 — PR-AUC unreliable"

            if threshold > 0 and imbalance_ratio < threshold:
                metric = "pr_auc"
                notes.append(
                    f"Metric → pr_auc (imbalance={imbalance_ratio:.3f}, "
                    f"n_minority≈{n_minority}, {rationale})"
                )
            else:
                notes.append(
                    f"Metric kept as {metric or 'roc_auc'}: "
                    f"imbalance={imbalance_ratio:.3f}, {rationale}"
                )
                metric = metric or "roc_auc"

        # ── 5. Final defaults ─────────────────────────────────────────
        if not metric:
            metric = "rmse" if task_type == "regression" else "roc_auc"
        if not task_desc or self._is_generic_description(task_desc):
            task_desc = (
                f"Predict {target_col} using "
                f"{task_type.replace('_', ' ')}"
                + (f" per {agg_key}" if agg_needed else "")
            )
            notes.append(f"Task description auto-generated: {task_desc}")

        return EnrichedSpec(
            task_type=task_type,
            task_type_confidence=confidence,
            target_column=target_col,
            target_condition=target_cond,
            aggregation_needed=agg_needed,
            aggregation_key=agg_key,
            evaluation_metric=metric,
            task_description=task_desc,
            class_imbalance_ratio=imbalance_ratio,
            enrichment_notes=notes,
        )

    # ── target inference ──────────────────────────────────────────────

    def _infer_target_from_data(
        self,
        df: pd.DataFrame | None,
        columns: list[str],
        exclude: list[str] | None = None,
    ) -> tuple[str, str]:
        """Score every column → return (best_column_name, reason_string)."""
        exclude = exclude or []
        best_col, best_score, best_reason = "", -999, "no match"

        for col in columns:
            if col in exclude:
                continue
            score = 0
            reasons: list[str] = []
            col_lower = col.lower()

            for pattern in self.OUTCOME_PATTERNS:
                if pattern in col_lower:
                    score += 3
                    reasons.append(f"name~'{pattern}'")
                    break

            for pattern in self.GROUP_KEY_PATTERNS:
                if pattern in col_lower:
                    score -= 5
                    break

            if df is not None and col in df.columns:
                series = df[col].dropna()
                n_unique = series.nunique()
                n_rows = len(series)

                if n_unique == 2:
                    score += 4
                    reasons.append("binary")
                elif 2 < n_unique <= 10 and pd.api.types.is_integer_dtype(series):
                    score += 2
                    reasons.append(f"low-card int ({n_unique})")
                elif n_unique > n_rows * 0.5:
                    score -= 3

                if pd.api.types.is_numeric_dtype(series):
                    vals = set(series.unique().tolist())
                    if vals and vals.issubset({1, 2, 3, 4, 5}):
                        score += 2
                        reasons.append("coded outcome (1-5)")

            if score > best_score:
                best_score = score
                best_col = col
                best_reason = "; ".join(reasons) if reasons else "highest score"

        return best_col, best_reason

    # ── aggregation detection ─────────────────────────────────────────

    def _detect_aggregation_need(
        self, df: pd.DataFrame | None, columns: list[str]
    ) -> tuple[str, float]:
        """
        Two-pass strategy (v1.1).
        Pass 1 — name-based (high precision).
        Pass 2 — statistical safety net (high recall).
        """
        if df is None:
            return "", 0.0
        n_rows = len(df)

        # Pass 1: name-based
        name_best, name_ratio = "", 0.0
        for col in columns:
            if not any(p in col.lower() for p in self.GROUP_KEY_PATTERNS):
                continue
            if col not in df.columns:
                continue
            n_groups = df[col].nunique()
            if n_groups < 5 or n_groups == 0:
                continue
            ratio = n_rows / n_groups
            if ratio > name_ratio:
                name_ratio = ratio
                name_best = col

        if name_best and name_ratio >= 1.5:
            return name_best, name_ratio

        # Pass 2: statistical safety net — catches arbitrary key names
        stat_best, stat_ratio = "", 0.0
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            n_groups = series.nunique()
            if n_groups < 10 or n_groups == 0:
                continue
            if pd.api.types.is_float_dtype(series):
                continue  # skip continuous features
            unique_ratio = n_groups / n_rows
            rows_per_group = n_rows / n_groups
            if unique_ratio < 0.5 and rows_per_group >= 2.0:
                if rows_per_group > stat_ratio:
                    stat_ratio = rows_per_group
                    stat_best = col

        if stat_best:
            return stat_best, stat_ratio

        return name_best, name_ratio

    # ── condition parser ──────────────────────────────────────────────

    def _apply_condition(
        self, df: pd.DataFrame, default_col: str, condition: str
    ) -> pd.Series:
        """
        Safely apply a target condition string → binary 0/1 Series.
        v1.1: AST-based whitelist parser for compound logic.
        Falls back to simple regex parser then raw column on failure.
        """
        if not condition or not condition.strip():
            return (df[default_col].astype(int)
                    if default_col in df.columns
                    else pd.Series(dtype=int))
        try:
            return self._parse_condition_safe(df, condition, default_col)
        except Exception:
            try:
                return self._apply_condition_simple(df, default_col, condition)
            except Exception:
                if default_col in df.columns:
                    return df[default_col].fillna(0).astype(int)
                return pd.Series([0] * len(df), index=df.index, dtype=int)

    def _parse_condition_safe(
        self, df: pd.DataFrame, condition: str, default_col: str
    ) -> pd.Series:
        import ast

        ALLOWED_NODES = (
            ast.Expression, ast.BoolOp, ast.And, ast.Or,
            ast.UnaryOp, ast.Not, ast.USub, ast.Invert,
            ast.Compare,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.In, ast.NotIn,
            ast.Name, ast.Constant, ast.Load,
            ast.List, ast.Tuple,
            ast.BinOp, ast.BitAnd, ast.BitOr,
        )

        cond_norm = re.sub(r"\bIN\b", "in", condition, flags=re.IGNORECASE)
        cond_norm = re.sub(r"\bAND\b", "and", cond_norm, flags=re.IGNORECASE)
        cond_norm = re.sub(r"\bOR\b", "or", cond_norm, flags=re.IGNORECASE)
        cond_norm = re.sub(r"\bNOT\b", "not", cond_norm, flags=re.IGNORECASE)

        tree = ast.parse(cond_norm, mode="eval")

        col_names_in_expr: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise ValueError(f"Disallowed AST node: {type(node).__name__}")
            if isinstance(node, ast.Name):
                if node.id not in df.columns and node.id not in {"True", "False", "None"}:
                    raise ValueError(f"Unknown column '{node.id}'")
                col_names_in_expr.add(node.id)

        class _BoolOpTransformer(ast.NodeTransformer):
            def visit_BoolOp(self, node: ast.BoolOp):
                self.generic_visit(node)
                op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
                result = node.values[0]
                for right in node.values[1:]:
                    result = ast.BinOp(left=result, op=op, right=right)
                return ast.copy_location(result, node)

            def visit_UnaryOp(self, node: ast.UnaryOp):
                self.generic_visit(node)
                if isinstance(node.op, ast.Not):
                    return ast.copy_location(
                        ast.UnaryOp(op=ast.Invert(), operand=node.operand), node
                    )
                return node

        tree = _BoolOpTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        namespace: dict = {"__builtins__": {}}
        for col in col_names_in_expr:
            namespace[col] = df[col]

        compiled = compile(tree, "<condition>", "eval")
        result = eval(compiled, namespace)  # noqa: S307 — whitelisted AST
        if isinstance(result, pd.Series):
            return result.fillna(False).astype(int)
        return pd.Series([int(bool(result))] * len(df), index=df.index, dtype=int)

    def _apply_condition_simple(
        self, df: pd.DataFrame, col: str, condition: str
    ) -> pd.Series:
        """v1.0 simple regex fallback."""
        m = re.search(r"in\s*\[([^\]]+)\]", condition)
        if m:
            vals: list = [v.strip().strip("'\"") for v in m.group(1).split(",")]
            try:
                vals = [int(v) for v in vals]
            except ValueError:
                pass
            return df[col].isin(vals).astype(int)
        m = re.search(r"([><=!]+)\s*([^\s]+)", condition)
        if m:
            op, val_str = m.group(1), m.group(2)
            val: float | str
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            ops = {
                "==": df[col] == val, "!=": df[col] != val,
                ">": df[col] > val, ">=": df[col] >= val,
                "<": df[col] < val, "<=": df[col] <= val,
            }
            if op in ops:
                return ops[op].astype(int)
        return df[col].astype(int) if col in df.columns else pd.Series(dtype=int)

    # ── helpers ───────────────────────────────────────────────────────

    def _looks_like_group_key(self, col_name: str) -> bool:
        col_lower = col_name.lower()
        return any(p in col_lower for p in self.GROUP_KEY_PATTERNS)

    def _is_generic_description(self, desc: str) -> bool:
        GENERIC = [
            "unknown task", "not specified", "cannot determine",
            "no task", "i cannot", "task not", "unclear",
            "no specific", "not provided", "not found",
        ]
        dl = desc.lower()
        return any(g in dl for g in GENERIC)


class TaskSpecValidator:
    """
    Layer 4 — validates an EnrichedSpec before plan building.
    Returns (is_valid, list[failure_messages]).
    Planning is blocked when is_valid=False.
    """

    VALID_TASK_TYPES = {
        "binary_classification", "multiclass_classification",
        "regression", "timeseries_forecasting", "anomaly_detection",
        "clustering", "survival_analysis",
    }
    VALID_METRICS = {
        "pr_auc", "roc_auc", "accuracy", "f1", "f1_macro",
        "rmse", "mae", "r2", "mase", "average_precision",
    }

    def validate(
        self,
        spec: EnrichedSpec,
        column_names: list[str],
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []

        if spec.task_type not in self.VALID_TASK_TYPES:
            failures.append(
                f"task_type='{spec.task_type}' unrecognised. "
                f"Valid: {sorted(self.VALID_TASK_TYPES)}"
            )

        if not spec.target_column:
            failures.append("target_column is empty — cannot build a supervised model.")
        elif spec.target_column not in column_names:
            failures.append(
                f"target_column='{spec.target_column}' not in dataset "
                f"(available: {column_names[:10]}…)"
            )

        # Warn if target looks like an ID column
        if spec.target_column and any(
            p in spec.target_column.lower()
            for p in ["_id", "id_", "number", "identifier", "code"]
        ):
            failures.append(
                f"target_column='{spec.target_column}' looks like an ID/key — "
                "verify this is the prediction target."
            )

        if spec.evaluation_metric not in self.VALID_METRICS:
            failures.append(
                f"evaluation_metric='{spec.evaluation_metric}' unknown. "
                "Defaulting to pr_auc/rmse."
            )

        if spec.aggregation_needed:
            if not spec.aggregation_key:
                failures.append(
                    "aggregation_needed=True but aggregation_key is empty."
                )
            elif spec.aggregation_key not in column_names:
                failures.append(
                    f"aggregation_key='{spec.aggregation_key}' not in dataset."
                )

        if spec.task_type_confidence < 0.5:
            failures.append(
                f"task_type confidence={spec.task_type_confidence:.2f} is low — "
                "upload clearer task instructions for better accuracy."
            )

        return len(failures) == 0, failures
