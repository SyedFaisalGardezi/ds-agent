"""Specialist Data Analysis Agent.

Runs a programmatic statistical scan of the dataset, then calls the LLM
to synthesise findings into a structured DATA_ANALYSIS_REPORT that the
ML Strategy Agent reads to plan model training.

The report is designed to be the "memory bridge" between data analysis
and ML planning — every recommendation in the ML_PLAN comes from evidence
in this report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── Report data structure ─────────────────────────────────────────────────────

@dataclass
class DataAnalysisReport:
    # ── Computed stats (no LLM needed) ───────────────────────────────────────
    n_rows: int = 0
    n_cols: int = 0
    target: str = ""
    task_type: str = "unknown"
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    datetime_cols: list = field(default_factory=list)
    # Target distribution: {value: count} for classification, {stat: val} for regression
    target_distribution: dict = field(default_factory=dict)
    class_imbalance_ratio: float | None = None  # minority/majority (binary/multiclass)
    recommended_metric: str = ""
    # Data quality issues
    null_pcts: dict = field(default_factory=dict)         # {col: pct} > 5%
    near_constant_cols: list = field(default_factory=list)
    high_cardinality_cols: list = field(default_factory=list)
    skewed_cols: dict = field(default_factory=dict)       # {col: skew_val} |skew| > 2
    outlier_cols: list = field(default_factory=list)      # >5% IQR outliers
    # Feature relevance
    top_mi_features: list = field(default_factory=list)   # [(col, mi_score)] top 15
    leakage_suspects: list = field(default_factory=list)  # high corr with target (|r|>0.9)
    # ── LLM-generated ────────────────────────────────────────────────────────
    analyst_narrative: str = ""
    ml_recommendations: dict = field(default_factory=dict)
    # ml_recommendations schema:
    # {
    #   "models": [{"name": str, "hyperparams": dict}, ...],  # 2-3 models
    #   "preprocessing": [str],
    #   "features_to_exclude": [str],
    #   "rationale": str,
    # }

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target": self.target,
            "task_type": self.task_type,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "datetime_cols": self.datetime_cols,
            "target_distribution": self.target_distribution,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "recommended_metric": self.recommended_metric,
            "null_pcts": self.null_pcts,
            "near_constant_cols": self.near_constant_cols,
            "high_cardinality_cols": self.high_cardinality_cols,
            "skewed_cols": self.skewed_cols,
            "outlier_cols": self.outlier_cols,
            "top_mi_features": self.top_mi_features,
            "leakage_suspects": self.leakage_suspects,
            "analyst_narrative": self.analyst_narrative,
            "ml_recommendations": self.ml_recommendations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataAnalysisReport:
        r = cls()
        for k, v in d.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r


# ── LLM prompts ───────────────────────────────────────────────────────────────

_ANALYST_SYSTEM = """\
/no_think
You are a senior data scientist performing a specialist data analysis review.
Your job: study the statistical summary and produce (1) a concise narrative
and (2) concrete ML recommendations with specific hyperparameter starting points.

Rules:
- Be concrete. Name specific models (lgbm, xgb, rf, lr, ridge, elasticnet).
- Give STARTING hyperparameter values, not search ranges.
- Recommendations must be justified by the stats (e.g. "imbalanced → use class_weight=balanced").
- 2-3 models maximum. Order by expected performance.
- If imbalance ratio < 0.3: use class_weight / scale_pos_weight, prefer lgbm/xgb.
- If n_rows > 50 000: prefer lgbm/xgb over sklearn RF (speed).
- If n_rows < 2 000: prefer lr/ridge + rf (avoid overfitting).
- If many nulls (>30% in ≥3 cols): include "impute_median" in preprocessing.
- If skewed numerics (>3 cols): include "log1p_skewed" in preprocessing.
- features_to_exclude: only list columns with obvious leakage (|corr with target| > 0.9)
  or near-zero variance — do NOT exclude useful features.

Output ONLY valid JSON matching this schema exactly:
{
  "analyst_narrative": "<200 word data quality + characteristics narrative>",
  "ml_recommendations": {
    "models": [
      {"name": "lgbm", "hyperparams": {"n_estimators": 400, "learning_rate": 0.05,
        "max_depth": 6, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8}},
      {"name": "rf", "hyperparams": {"n_estimators": 300, "max_depth": 12,
        "min_samples_leaf": 10, "class_weight": "balanced"}}
    ],
    "preprocessing": ["impute_median", "encode_ordinal_high_cardinality"],
    "features_to_exclude": [],
    "rationale": "2-sentence justification"
  }
}
"""

_ANALYST_USER_TMPL = """\
TASK: {task_type} — predict '{target}'

DATASET STATS:
  Rows: {n_rows:,} | Cols: {n_cols}
  Numeric: {numeric_cols}
  Categorical: {categorical_cols}

TARGET DISTRIBUTION:
{target_dist_str}
  Imbalance ratio: {imbalance_str}
  Recommended metric: {recommended_metric}

DATA QUALITY:
  Cols with >5% nulls: {null_cols}
  Skewed numeric cols (|skew|>2): {skewed_str}
  Near-constant cols: {near_constant}
  High-cardinality categoricals: {high_card}
  Outlier-rich cols (>5% IQR): {outlier_cols}

FEATURE RELEVANCE (top MI scores):
{mi_str}

LEAKAGE SUSPECTS (|corr with target|>0.9): {leakage}

Now produce the JSON report.
"""


# ── ML Plan data structure ─────────────────────────────────────────────────────

@dataclass
class MLPlan:
    models: list = field(default_factory=list)
    # [{"name": "lgbm", "hyperparams": {...}}, ...]
    preprocessing: list = field(default_factory=list)
    features_to_exclude: list = field(default_factory=list)
    eval_metric: str = ""
    task_type: str = ""
    rationale: str = ""
    source_report_summary: str = ""  # brief summary of data analysis used

    def to_dict(self) -> dict:
        return {
            "models": self.models,
            "preprocessing": self.preprocessing,
            "features_to_exclude": self.features_to_exclude,
            "eval_metric": self.eval_metric,
            "task_type": self.task_type,
            "rationale": self.rationale,
            "source_report_summary": self.source_report_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MLPlan:
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


# ── Core analyst class ────────────────────────────────────────────────────────

class DataAnalysisAgent:
    """Performs programmatic statistical analysis then calls LLM for synthesis.

    Usage::

        agent = DataAnalysisAgent(llm_client)
        report = agent.analyze(df, target="churn", task_type="binary_classification",
                               task_spec=kernel.namespace.get("TASK_SPEC"))
        kernel.namespace["DATA_ANALYSIS_REPORT"] = report.to_dict()
    """

    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    # ── public entry point ────────────────────────────────────────────────────

    def analyze(
        self,
        df: Any,                # pandas DataFrame
        target: str = "",
        task_type: str = "unknown",
        task_spec: dict | None = None,
        mi_scores: dict | None = None,    # from kernel["MI_SCORES"] if already computed
    ) -> DataAnalysisReport:
        """Run full analysis. Returns a DataAnalysisReport (never raises)."""
        try:
            import pandas as pd  # noqa: F401 (needed for df ops below)
        except ImportError:
            return DataAnalysisReport()

        report = DataAnalysisReport()
        report.n_rows = int(len(df))
        report.n_cols = int(df.shape[1])
        report.target = target
        report.task_type = task_type

        # ── col type classification ───────────────────────────────────────────
        try:
            report.numeric_cols = list(df.select_dtypes(include="number").columns)
            report.categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
            report.datetime_cols = list(df.select_dtypes(include=["datetime64"]).columns)
        except Exception:
            pass

        # ── target analysis ───────────────────────────────────────────────────
        if target and target in df.columns:
            try:
                y = df[target].dropna()
                if task_type in ("binary_classification", "multiclass_classification", "unknown") \
                        and (y.dtype == object or y.nunique() <= 30):
                    vc = y.value_counts()
                    report.target_distribution = {str(k): int(v) for k, v in vc.items()}
                    if len(vc) >= 2:
                        report.class_imbalance_ratio = float(vc.min() / vc.max())
                    # Metric
                    if task_spec:
                        report.recommended_metric = task_spec.get("evaluation_metric", "")
                    if not report.recommended_metric:
                        if report.class_imbalance_ratio and report.class_imbalance_ratio < 0.4:
                            report.recommended_metric = "average_precision (pr_auc) — imbalanced"
                        else:
                            report.recommended_metric = "roc_auc"
                else:
                    y_num = pd.to_numeric(y, errors="coerce").dropna()
                    report.target_distribution = {
                        "mean": round(float(y_num.mean()), 4),
                        "std":  round(float(y_num.std()), 4),
                        "min":  round(float(y_num.min()), 4),
                        "max":  round(float(y_num.max()), 4),
                    }
                    report.recommended_metric = task_spec.get("evaluation_metric", "rmse") \
                        if task_spec else "rmse"
            except Exception:
                pass

        # ── data quality ──────────────────────────────────────────────────────
        try:
            null_pct = (df.isna().mean() * 100).round(1)
            report.null_pcts = {
                col: float(pct)
                for col, pct in null_pct.items()
                if pct > 5.0
            }
        except Exception:
            pass

        try:
            num_df = df[report.numeric_cols] if report.numeric_cols else df.select_dtypes("number")
            for col in num_df.columns:
                try:
                    v = float(num_df[col].var())
                    if v < 1e-6:
                        report.near_constant_cols.append(col)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            for col in report.categorical_cols:
                n_unique = int(df[col].nunique())
                if n_unique > 50:
                    report.high_cardinality_cols.append(col)
        except Exception:
            pass

        try:
            num_df = df[report.numeric_cols] if report.numeric_cols else df.select_dtypes("number")
            for col in num_df.columns:
                try:
                    s = float(num_df[col].skew())
                    if abs(s) > 2.0:
                        report.skewed_cols[col] = round(s, 2)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            num_df = df[report.numeric_cols] if report.numeric_cols else df.select_dtypes("number")
            for col in num_df.columns:
                try:
                    q1 = float(num_df[col].quantile(0.25))
                    q3 = float(num_df[col].quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        outlier_rate = float(((num_df[col] < q1 - 1.5 * iqr) |
                                              (num_df[col] > q3 + 1.5 * iqr)).mean())
                        if outlier_rate > 0.05:
                            report.outlier_cols.append(col)
                except Exception:
                    pass
        except Exception:
            pass

        # ── feature relevance ─────────────────────────────────────────────────
        if mi_scores:
            # MI scores already computed by MutualInfoTool
            try:
                sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
                report.top_mi_features = [(c, round(float(v), 4)) for c, v in sorted_mi[:15]]
            except Exception:
                pass
        elif target and target in df.columns and report.n_rows <= 100_000:
            # Quick MI estimation (only for moderately sized datasets)
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                num_cols = [c for c in report.numeric_cols if c != target]
                if num_cols and report.n_rows >= 50:
                    X = df[num_cols].fillna(0).values
                    y = df[target].fillna(0)
                    if task_type in ("binary_classification", "multiclass_classification"):
                        mi = mutual_info_classif(X, y, random_state=42)
                    else:
                        mi = mutual_info_regression(X, y, random_state=42)
                    pairs = sorted(zip(num_cols, mi.tolist()), key=lambda x: x[1], reverse=True)
                    report.top_mi_features = [(c, round(float(v), 4)) for c, v in pairs[:15]]
            except Exception:
                pass

        # ── leakage suspects ──────────────────────────────────────────────────
        if target and target in df.columns:
            try:
                num_cols = [c for c in report.numeric_cols if c != target]
                if num_cols:
                    y = df[target].copy()
                    if y.dtype == object:
                        y = y.fillna("__null__")
                        y = pd.to_numeric(
                            pd.Categorical(y).codes, errors="coerce"
                        )
                    for col in num_cols:
                        try:
                            corr = abs(float(df[col].fillna(0).corr(y.fillna(0))))
                            if corr > 0.9:
                                report.leakage_suspects.append(col)
                        except Exception:
                            pass
            except Exception:
                pass

        # ── LLM synthesis ─────────────────────────────────────────────────────
        if self.llm is not None:
            try:
                self._call_llm_synthesis(report)
            except Exception:
                report.analyst_narrative = "(LLM synthesis unavailable)"

        # Fallback recommendations when LLM offline
        if not report.ml_recommendations:
            report.ml_recommendations = self._heuristic_recommendations(report)

        return report

    # ── LLM synthesis ─────────────────────────────────────────────────────────

    def _call_llm_synthesis(self, report: DataAnalysisReport) -> None:
        """Call LLM → populate analyst_narrative + ml_recommendations."""
        # Build user message
        target_dist_str = "\n".join(
            f"  {k}: {v}" for k, v in list(report.target_distribution.items())[:10]
        ) or "  (no target)"
        imbalance_str = (
            f"{report.class_imbalance_ratio:.3f}"
            if report.class_imbalance_ratio is not None else "n/a"
        )
        mi_str = "\n".join(
            f"  {col}: {score}" for col, score in report.top_mi_features[:10]
        ) or "  (not computed)"
        skewed_str = (
            ", ".join(f"{c}={v:+.1f}" for c, v in list(report.skewed_cols.items())[:8])
            or "none"
        )
        null_cols_str = (
            ", ".join(f"{c}={v:.0f}%" for c, v in list(report.null_pcts.items())[:10])
            or "none"
        )

        user_msg = _ANALYST_USER_TMPL.format(
            task_type=report.task_type,
            target=report.target or "(not set)",
            n_rows=report.n_rows,
            n_cols=report.n_cols,
            numeric_cols=report.numeric_cols[:20],
            categorical_cols=report.categorical_cols[:10],
            target_dist_str=target_dist_str,
            imbalance_str=imbalance_str,
            recommended_metric=report.recommended_metric,
            null_cols=null_cols_str,
            skewed_str=skewed_str,
            near_constant=report.near_constant_cols[:10] or "none",
            high_card=report.high_cardinality_cols[:10] or "none",
            outlier_cols=report.outlier_cols[:10] or "none",
            mi_str=mi_str,
            leakage=report.leakage_suspects[:10] or "none",
        )

        raw = self.llm._generate(
            user_msg,
            system=_ANALYST_SYSTEM,
            temperature=0.1,
            max_tokens=1200,
        )
        if not raw:
            return

        # Strip think blocks
        raw = self.llm._strip_thinking(raw)

        # Extract JSON
        parsed = _safe_json(raw)
        if not parsed:
            return

        report.analyst_narrative = parsed.get("analyst_narrative", "")
        rec = parsed.get("ml_recommendations", {})
        if isinstance(rec, dict):
            report.ml_recommendations = rec

    # ── heuristic fallback ────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_recommendations(report: DataAnalysisReport) -> dict:
        """Generate rule-based ML recommendations when LLM is offline."""
        is_cls = report.task_type in (
            "binary_classification", "multiclass_classification"
        )
        small = report.n_rows < 2_000
        large = report.n_rows > 50_000
        imbalanced = (
            report.class_imbalance_ratio is not None
            and report.class_imbalance_ratio < 0.4
        )

        if is_cls:
            if small:
                models = [
                    {"name": "lr", "hyperparams": {
                        "C": 1.0, "max_iter": 1000,
                        "class_weight": "balanced" if imbalanced else None
                    }},
                    {"name": "rf", "hyperparams": {
                        "n_estimators": 200, "max_depth": 8,
                        "class_weight": "balanced" if imbalanced else None
                    }},
                ]
            elif large:
                models = [
                    {"name": "lgbm", "hyperparams": {
                        "n_estimators": 500, "learning_rate": 0.05,
                        "max_depth": 6, "num_leaves": 63,
                        "is_unbalance": imbalanced,
                    }},
                    {"name": "xgb", "hyperparams": {
                        "n_estimators": 400, "learning_rate": 0.05,
                        "max_depth": 6,
                        "scale_pos_weight": 5 if imbalanced else 1,
                    }},
                ]
            else:
                models = [
                    {"name": "lgbm", "hyperparams": {
                        "n_estimators": 300, "learning_rate": 0.05,
                        "max_depth": 6, "num_leaves": 63,
                        "class_weight": "balanced" if imbalanced else None,
                    }},
                    {"name": "rf", "hyperparams": {
                        "n_estimators": 300, "max_depth": 10,
                        "class_weight": "balanced" if imbalanced else None,
                    }},
                    {"name": "lr", "hyperparams": {"C": 0.1, "max_iter": 500}},
                ]
        else:
            models = [
                {"name": "lgbm", "hyperparams": {
                    "n_estimators": 300, "learning_rate": 0.05,
                    "max_depth": 6, "num_leaves": 63,
                }},
                {"name": "ridge", "hyperparams": {"alpha": 1.0}},
            ]

        preprocessing = []
        if report.null_pcts:
            preprocessing.append("impute_median")
        if len(report.skewed_cols) > 2:
            preprocessing.append("log1p_skewed")
        if report.high_cardinality_cols:
            preprocessing.append("encode_high_cardinality_ordinal")
        if report.categorical_cols:
            preprocessing.append("encode_categoricals")

        return {
            "models": models,
            "preprocessing": preprocessing,
            "features_to_exclude": report.leakage_suspects,
            "rationale": (
                f"{'Imbalanced ' if imbalanced else ''}"
                f"{'small' if small else 'large' if large else 'medium'} dataset "
                f"({report.n_rows:,} rows). "
                f"Heuristic recommendations (LLM offline)."
            ),
        }


# ── ML Strategy Planner ────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
/no_think
You are an ML strategy expert. Given a data analysis report, produce a precise
ML_PLAN that the training tool will execute directly.

Output ONLY valid JSON:
{
  "models": [
    {"name": "lgbm",  "hyperparams": {"n_estimators": 400, "learning_rate": 0.05,
      "max_depth": 6, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
      "class_weight": "balanced"}},
    {"name": "rf",    "hyperparams": {"n_estimators": 300, "max_depth": 10,
      "min_samples_leaf": 5, "class_weight": "balanced"}}
  ],
  "preprocessing": ["impute_median", "log1p_skewed"],
  "features_to_exclude": ["col_a", "col_b"],
  "eval_metric": "average_precision",
  "rationale": "2-sentence justification"
}

Valid model names: lgbm, xgb, rf, extra_trees, gbm, lr, ridge, elasticnet, lasso
Valid eval_metric values: roc_auc, average_precision, f1_weighted, rmse, mae, r2

Rules:
- Use ONLY model names from the valid list above
- features_to_exclude should only include leakage columns (|corr|>0.9) or constants
- 2-3 models, ordered by expected performance
- Concrete hyperparameter values, not ranges
"""


class MLStrategyPlanner:
    """Reads DATA_ANALYSIS_REPORT from kernel → calls LLM → produces ML_PLAN.

    Usage::

        planner = MLStrategyPlanner(llm_client)
        ml_plan = planner.plan(report_dict, task_spec=kernel.namespace.get("TASK_SPEC"))
        kernel.namespace["ML_PLAN"] = ml_plan.to_dict()
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def plan(
        self,
        report: dict,
        task_spec: dict | None = None,
    ) -> MLPlan:
        """Produce an MLPlan from a DATA_ANALYSIS_REPORT dict."""
        ml_plan = MLPlan()
        ml_plan.task_type = report.get("task_type", "unknown")
        ml_plan.eval_metric = report.get("recommended_metric", "roc_auc")

        # If data analysis already includes ml_recommendations, use directly
        rec = report.get("ml_recommendations", {})
        if rec and rec.get("models"):
            ml_plan.models = rec["models"]
            ml_plan.preprocessing = rec.get("preprocessing", [])
            ml_plan.features_to_exclude = rec.get("features_to_exclude", [])
            ml_plan.rationale = rec.get("rationale", "")
            ml_plan.source_report_summary = self._summarise_report(report)
            # Optionally override eval_metric from task_spec
            if task_spec and task_spec.get("evaluation_metric"):
                ml_plan.eval_metric = task_spec["evaluation_metric"]
            return ml_plan

        # Fallback: call LLM for a fresh ML plan based on report
        if self.llm is not None:
            try:
                ml_plan = self._call_llm_plan(report, task_spec, ml_plan)
            except Exception:
                pass

        if not ml_plan.models:
            # Last resort: take heuristic from DataAnalysisAgent
            r = DataAnalysisReport.from_dict(report)
            rec = DataAnalysisAgent._heuristic_recommendations(r)
            ml_plan.models = rec["models"]
            ml_plan.preprocessing = rec["preprocessing"]
            ml_plan.features_to_exclude = rec["features_to_exclude"]
            ml_plan.rationale = rec["rationale"]

        ml_plan.source_report_summary = self._summarise_report(report)
        return ml_plan

    def _call_llm_plan(
        self, report: dict, task_spec: dict | None, ml_plan: MLPlan
    ) -> MLPlan:
        summary = self._summarise_report(report)
        metric_hint = ""
        if task_spec:
            metric_hint = f"\nRequired metric from TASK_SPEC: {task_spec.get('evaluation_metric', '')}"

        user_msg = f"DATA ANALYSIS REPORT SUMMARY:\n{summary}{metric_hint}\n\nProduce the ML_PLAN JSON."
        raw = self.llm._generate(
            user_msg,
            system=_PLANNER_SYSTEM,
            temperature=0.1,
            max_tokens=800,
        )
        if not raw:
            return ml_plan
        raw = self.llm._strip_thinking(raw)
        parsed = _safe_json(raw)
        if not parsed:
            return ml_plan

        ml_plan.models = parsed.get("models", [])
        ml_plan.preprocessing = parsed.get("preprocessing", [])
        ml_plan.features_to_exclude = parsed.get("features_to_exclude", [])
        ml_plan.eval_metric = parsed.get("eval_metric", ml_plan.eval_metric)
        ml_plan.rationale = parsed.get("rationale", "")
        return ml_plan

    @staticmethod
    def _summarise_report(report: dict) -> str:
        """Compact text representation of DATA_ANALYSIS_REPORT for LLM context."""
        lines = [
            f"Task: {report.get('task_type')} | Target: {report.get('target')}",
            f"Dataset: {report.get('n_rows', 0):,} rows × {report.get('n_cols', 0)} cols",
            f"Imbalance ratio: {report.get('class_imbalance_ratio', 'n/a')}",
            f"Recommended metric: {report.get('recommended_metric', 'n/a')}",
        ]
        if report.get("null_pcts"):
            top_nulls = list(report["null_pcts"].items())[:5]
            lines.append(f"Nulls: {top_nulls}")
        if report.get("skewed_cols"):
            lines.append(f"Skewed cols: {list(report['skewed_cols'].items())[:5]}")
        if report.get("high_cardinality_cols"):
            lines.append(f"High-cardinality: {report['high_cardinality_cols'][:5]}")
        if report.get("top_mi_features"):
            lines.append(f"Top features by MI: {report['top_mi_features'][:8]}")
        if report.get("leakage_suspects"):
            lines.append(f"Leakage suspects: {report['leakage_suspects'][:5]}")
        if report.get("analyst_narrative"):
            lines.append(f"Analyst note: {report['analyst_narrative'][:400]}")
        return "\n".join(lines)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _safe_json(text: str) -> dict | None:
    """Try multiple strategies to extract JSON from LLM output."""
    if not text:
        return None
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
    # 3. Find outermost {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
