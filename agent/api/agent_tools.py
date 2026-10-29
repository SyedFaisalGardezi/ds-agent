"""MCP-compatible tool registry for the ds-agent agentic runner.

Each tool exposes:
  - name          — unique identifier the planner LLM uses in JSON plans
  - description   — what it does (shown to planner)
  - input_schema  — JSON Schema for args (MCP-style)
  - execute()     — runs the actual logic; returns ToolOutput

The planner sees the tool list, picks which tools to call and in what order,
and the runner executes each one, feeding results back to the LLM.

Tools can be called from:
  - A structured plan (agent_runner.py)
  - Directly from a chat instruction ("run eda", "train a model")
  - Inside a code block the LLM writes (for execute_code)
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolOutput:
    success: bool
    text: str                          # human-readable / LLM context
    artifacts: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)   # base64 PNG
    error: str = ""
    math_trace: str = ""


# ── base class ────────────────────────────────────────────────────────────────

class BaseTool:
    name: str = ""
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    def execute(self, args: dict, *, session: Any, kernel: Any,
                llm_client: Any) -> ToolOutput:
        raise NotImplementedError


# ── tool implementations ──────────────────────────────────────────────────────

class EdaProfileTool(BaseTool):
    name = "eda_profile"
    description = (
        "Run exploratory data analysis: shape, dtypes, null rates, descriptive stats, "
        "duplicate check, and — when a target is set — task-aware feature-vs-target "
        "analysis (class separation, correlation ranking, univariate AUC, conditional "
        "distributions). Use as the FIRST step for any new dataset."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        # ── Section 1: structural profile (always runs) ─────────────────────
        base_code = textwrap.dedent("""
            print(f"Shape: {df.shape}")
            print(f"\\nDtypes:\\n{df.dtypes.to_string()}")
            _null_pct = (df.isna().mean()*100).round(1)
            _null_pct = _null_pct[_null_pct > 0]
            print(f"\\nNull % (non-zero only):\\n{_null_pct.to_string() if len(_null_pct) else '  none'}")
            print(f"\\nDescriptive stats (numeric):\\n{df.describe().T.to_string()}")
            _dups = df.duplicated().sum()
            print(f"\\nDuplicate rows: {_dups} ({_dups/len(df):.1%})")
            # Cardinality summary for categoricals
            _cats = df.select_dtypes('object')
            if len(_cats.columns):
                print("\\nCategorical cardinality:")
                for _c in _cats.columns:
                    print(f"  {_c}: {df[_c].nunique()} unique  "
                          f"(top: {df[_c].value_counts().index[0]!r} "
                          f"= {df[_c].value_counts().iloc[0]/len(df):.1%})")
        """).strip()
        out = kernel.execute(base_code)
        all_text = out.as_text(4000)
        all_figs = out.figures[:]

        # ── Section 2: task-aware feature-vs-target analysis ────────────────
        target = session.target or kernel.namespace.get("TARGET")
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        task_type = task_spec.get("task_type", "")

        if target and kernel.namespace.get("df") is not None:
            df_check = kernel.namespace["df"]
            if target not in df_check.columns:
                target = None  # stale target, skip

        if target:
            is_cls = ("classif" in task_type or "binary" in task_type
                      or df_check[target].nunique() < 20)

            feature_analysis_code = textwrap.dedent(f"""
                import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                from sklearn.preprocessing import LabelEncoder
                from sklearn.metrics import roc_auc_score

                _tgt = {repr(target)}
                _is_cls = {repr(is_cls)}
                _df = df.copy()

                # ── encode target if categorical ─────────────────────────────
                _y_raw = _df[_tgt]
                if _y_raw.dtype == object:
                    _le_t = LabelEncoder()
                    _y = pd.Series(_le_t.fit_transform(_y_raw.astype(str)), name=_tgt)
                else:
                    _y = _y_raw.fillna(_y_raw.median())

                _n_cls = _y.nunique()
                print("\\n=== Task-Aware Feature Analysis ===")
                print(f"Target: {{_tgt}}  |  type: {repr(task_type) or 'auto'}  "
                      f"|  classes: {{_n_cls}}  |  n_rows: {{len(_df)}}")

                # ── Target distribution ──────────────────────────────────────
                print(f"\\nTarget distribution:")
                print(_y_raw.value_counts().head(15).to_string())
                if _is_cls:
                    _vc = _y.value_counts()
                    _imb = _vc.min()/_vc.max()
                    print(f"  Imbalance ratio (min/max): {{_imb:.3f}}"
                          + (" ⚠ HIGHLY IMBALANCED" if _imb < 0.2 else ""))

                # ── Numeric feature correlations with target ─────────────────
                _num_cols = [c for c in _df.select_dtypes('number').columns if c != _tgt]
                _corr_with_tgt = {{}}
                for _c in _num_cols:
                    try:
                        _corr_with_tgt[_c] = _df[_c].fillna(0).corr(_y)
                    except Exception:
                        pass
                _corr_s = pd.Series(_corr_with_tgt).dropna().sort_values(key=abs, ascending=False)
                print(f"\\nTop numeric features by |correlation| with target:")
                print(_corr_s.head(20).round(4).to_string())

                # ── Univariate AUC per feature (classification only) ─────────
                if _is_cls and _n_cls == 2:
                    _aucs = {{}}
                    for _c in _num_cols[:40]:
                        try:
                            _vals = _df[_c].fillna(0).values
                            _auc = roc_auc_score(_y, _vals)
                            _aucs[_c] = max(_auc, 1 - _auc)  # flip if <0.5
                        except Exception:
                            pass
                    _auc_s = pd.Series(_aucs).sort_values(ascending=False)
                    print(f"\\nTop features by univariate ROC-AUC:")
                    print(_auc_s.head(15).round(4).to_string())
                    # Store for downstream use
                    EDA_UNIVARIATE_AUC = _auc_s

                # ── Categorical feature association with target ───────────────
                _cat_cols = [c for c in _df.select_dtypes('object').columns if c != _tgt
                             and _df[c].nunique() <= 50]
                if _cat_cols:
                    print(f"\\nCategorical feature association with target (Cramér's V):")
                    from scipy.stats import chi2_contingency
                    _cramers = {{}}
                    for _c in _cat_cols:
                        try:
                            _ct = pd.crosstab(_df[_c], _y)
                            _chi2 = chi2_contingency(_ct, correction=False)[0]
                            _n = _ct.sum().sum()
                            _k = min(_ct.shape) - 1
                            _cramers[_c] = np.sqrt(_chi2 / (_n * _k)) if _k > 0 else 0
                        except Exception:
                            pass
                    _cv_s = pd.Series(_cramers).sort_values(ascending=False)
                    print(_cv_s.round(4).to_string())

                # ── Missing value pattern by target class ────────────────────
                if _df.isna().any().any() and _is_cls:
                    print(f"\\nMissing-value rates by target class:")
                    _miss_by_cls = _df.groupby(_tgt).apply(lambda g: g.isna().mean())
                    _diff = _miss_by_cls.max() - _miss_by_cls.min()
                    _biased = _diff[_diff > 0.05].sort_values(ascending=False)
                    if len(_biased):
                        print("  Columns with >5% missing-rate difference across classes:")
                        print(_biased.head(10).round(3).to_string())
                    else:
                        print("  No strong missing-value bias detected across classes.")

                # ── Plot 1: Correlation bar (top-20 features vs target) ───────
                _plot_corr = _corr_s.head(20)
                fig, ax = plt.subplots(figsize=(9, max(4, len(_plot_corr)*0.38)))
                colors = ['#d62728' if v < 0 else '#1f77b4' for v in _plot_corr.values]
                ax.barh(_plot_corr.index[::-1], _plot_corr.values[::-1], color=colors[::-1])
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel('Pearson correlation with target')
                ax.set_title(f'Feature → Target Correlation  (target: {{_tgt}})')
                plt.tight_layout(); plt.show()

                # ── Plot 2: Univariate AUC ranking (binary cls only) ─────────
                if _is_cls and _n_cls == 2 and 'EDA_UNIVARIATE_AUC' in dir():
                    _auc_plot = EDA_UNIVARIATE_AUC.head(15)
                    fig2, ax2 = plt.subplots(figsize=(9, max(4, len(_auc_plot)*0.38)))
                    ax2.barh(_auc_plot.index[::-1], _auc_plot.values[::-1], color='#2ca02c')
                    ax2.axvline(0.5, color='red', linestyle='--', linewidth=0.8,
                                label='random (0.5)')
                    ax2.set_xlim(0.45, 1.0)
                    ax2.set_xlabel('Univariate ROC-AUC (higher = more separating)')
                    ax2.set_title(f'Feature Separability  (target: {{_tgt}})')
                    ax2.legend(fontsize=8)
                    plt.tight_layout(); plt.show()

                # ── Plot 3: KDE distributions by class (top-6 numeric feats) ─
                if _is_cls and _n_cls <= 6:
                    _top_feats = _corr_s.head(6).index.tolist()
                    _classes = sorted(_y.unique())
                    _palette = plt.cm.Set1.colors
                    _nf = len(_top_feats)
                    _ncols = min(3, _nf); _nrows = (_nf + _ncols - 1) // _ncols
                    fig3, axes3 = plt.subplots(_nrows, _ncols,
                                               figsize=(_ncols*5, _nrows*3.5))
                    if _nf == 1: axes3 = [[axes3]]
                    elif _nrows == 1: axes3 = [axes3]
                    for _i, _feat in enumerate(_top_feats):
                        _ax = axes3[_i//_ncols][_i%_ncols]
                        for _j, _cls in enumerate(_classes):
                            _vals = _df.loc[_y == _cls, _feat].dropna()
                            if len(_vals) > 1:
                                _vals.plot.kde(ax=_ax,
                                               label=f'class {{_cls}} (n={{len(_vals)}})',
                                               color=_palette[_j % len(_palette)],
                                               linewidth=1.8)
                        _ax.set_title(f'{{_feat}}\\n(corr={{_corr_s.get(_feat,0):.3f}})',
                                      fontsize=9)
                        _ax.legend(fontsize=7); _ax.set_xlabel('')
                    for _i in range(_nf, _nrows*_ncols):
                        axes3[_i//_ncols][_i%_ncols].set_visible(False)
                    plt.suptitle(f'Feature Distributions by Target Class  ({{_tgt}})',
                                 fontsize=12, y=1.01)
                    plt.tight_layout(); plt.show()

                # ── Plot 4: Categorical distributions by class (stacked bar) ──
                if _is_cls and _cat_cols and 'EDA_UNIVARIATE_AUC' not in dir():
                    # for multiclass — show top Cramér's V categoricals
                    pass
                if _is_cls and _cat_cols:
                    _top_cats = list(_cv_s.head(3).index) if '_cv_s' in dir() else _cat_cols[:3]
                    for _cc in _top_cats:
                        _ct = pd.crosstab(_df[_cc], _y, normalize='index')
                        _ct.plot(kind='bar', stacked=True, figsize=(10, 3),
                                 colormap='Set2', edgecolor='white', linewidth=0.4)
                        plt.title(f'{{_cc}} → class distribution  '
                                  f'(Cramér V={{_cramers.get(_cc,0):.3f}})')
                        plt.ylabel('proportion'); plt.xlabel(_cc)
                        plt.legend(title=_tgt, bbox_to_anchor=(1.01,1), loc='upper left',
                                   fontsize=8)
                        plt.tight_layout(); plt.show()
            """).strip()

            feat_out = kernel.execute(feature_analysis_code)
            all_text += "\n\n" + feat_out.as_text(5000)
            all_figs += feat_out.figures

        artifacts = {
            "shape": list(kernel.namespace["df"].shape)
            if kernel.namespace.get("df") is not None else [],
        }
        return ToolOutput(
            success=out.success,
            text=all_text,
            artifacts=artifacts,
            figures=all_figs,
            error=out.error,
        )


class QualityCheckTool(BaseTool):
    name = "quality_check"
    description = (
        "Assess data quality: outliers (IQR), high-cardinality columns, constant "
        "columns, skewed distributions, multicollinearity (VIF), and potential "
        "target-leakage detection when a target is set."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        target = session.target or kernel.namespace.get("TARGET") or ""
        code = textwrap.dedent(f"""
            import numpy as np
            import warnings
            warnings.filterwarnings('ignore')

            _tgt = {repr(target)}
            num = df.select_dtypes(include='number')

            # ── Outliers (IQR) ───────────────────────────────────────────────
            issues = []
            for col in num.columns:
                q1, q3 = num[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                n_out = ((num[col] < q1 - 1.5*iqr) | (num[col] > q3 + 1.5*iqr)).sum()
                if n_out > 0:
                    issues.append(f"  {{col}}: {{n_out}} outliers ({{n_out/len(df):.1%}})")
            print("=== Outliers ===")
            print("\\n".join(issues) if issues else "  None detected")

            # ── Constants & near-constants ───────────────────────────────────
            const = [c for c in df.columns if df[c].nunique() <= 1]
            near_const = [c for c in df.columns
                          if 1 < df[c].nunique() <= max(2, len(df)*0.001)]
            print(f"\\nConstant columns: {{const}}")
            if near_const:
                print(f"Near-constant columns (≤0.1% unique): {{near_const}}")

            # ── High-cardinality categoricals ────────────────────────────────
            hi_card = [(c, df[c].nunique()) for c in df.select_dtypes('object').columns
                       if df[c].nunique() > 50]
            print(f"\\nHigh-cardinality categoricals:")
            for _c, _n in sorted(hi_card, key=lambda x: -x[1]):
                print(f"  {{_c}}: {{_n}} unique")
            if not hi_card:
                print("  None")

            # ── Skewness ─────────────────────────────────────────────────────
            skewed = num.skew().abs().sort_values(ascending=False).head(8)
            print(f"\\nMost skewed (|skew|):\\n{{skewed.round(2).to_string()}}")

            # ── Multicollinearity — VIF ──────────────────────────────────────
            _vif_cols = [c for c in num.columns if c != _tgt and num[c].std() > 1e-9]
            print(f"\\n=== Multicollinearity (VIF) — top correlated pairs ===")
            _corr_mat = num[_vif_cols].fillna(0).corr().abs()
            _pairs = []
            for _i, _ci in enumerate(_vif_cols):
                for _j, _cj in enumerate(_vif_cols):
                    if _j <= _i: continue
                    _r = _corr_mat.loc[_ci, _cj]
                    if _r > 0.85:
                        _pairs.append((_ci, _cj, _r))
            _pairs.sort(key=lambda x: -x[2])
            if _pairs:
                print("  Highly correlated pairs (|r|>0.85):")
                for _ci, _cj, _r in _pairs[:10]:
                    print(f"    {{_ci}} ↔ {{_cj}}: r={{_r:.3f}}")
            else:
                print("  No highly correlated feature pairs (|r|>0.85) detected")

            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                _Xvif = num[_vif_cols].fillna(0)
                _vif_vals = [variance_inflation_factor(_Xvif.values, i)
                             for i in range(len(_vif_cols))]
                _vif_s = pd.Series(_vif_vals, index=_vif_cols).sort_values(ascending=False)
                _high_vif = _vif_s[_vif_s > 10]
                print(f"\\n  VIF > 10 (severe multicollinearity):")
                print(_high_vif.round(1).to_string() if len(_high_vif) else "  None")
            except Exception as _e:
                print(f"  VIF skipped (statsmodels unavailable: {{_e}})")

            # ── Target-leakage detection ─────────────────────────────────────
            if _tgt and _tgt in df.columns:
                print(f"\\n=== Potential Target Leakage (target: {{_tgt}}) ===")
                _y = df[_tgt].fillna(0)
                _leak_feats = []
                for _c in [c for c in num.columns if c != _tgt]:
                    try:
                        _r = abs(df[_c].fillna(0).corr(_y))
                        if _r > 0.95:
                            _leak_feats.append((_c, _r))
                    except Exception:
                        pass
                if _leak_feats:
                    print("  ⚠ HIGH CORRELATION features (|r|>0.95) — possible leakage:")
                    for _c, _r in sorted(_leak_feats, key=lambda x: -x[1]):
                        print(f"    {{_c}}: r={{_r:.4f}}")
                else:
                    print("  No obvious numeric leakage detected (|r|≤0.95 for all features)")
        """).strip()
        out = kernel.execute(code)
        return ToolOutput(
            success=out.success, text=out.as_text(5000),
            figures=out.figures, error=out.error,
        )


def _is_unsuitable_target(df, col) -> tuple[bool, str]:
    """Reject columns that cannot be a supervised prediction target.

    A date/time column, an identifier (a distinct value in nearly every row),
    or a constant is not something to predict. Selecting one produces the
    service_date-as-target failure: a label derived from the date is then
    "predicted" from date features, which leaks and scores a perfect AUC, and
    the datetime column crashes the model. Returns (unsuitable, reason).
    """
    import warnings

    import pandas as _pd
    s = df[col]
    n = len(s)
    nun = int(s.nunique(dropna=True))
    if nun <= 1:
        return True, "it is constant (one unique value)"
    if _pd.api.types.is_datetime64_any_dtype(s):
        return True, "it is a date/time column"
    # An identifier has many distinct values, close to one per row.
    if not _pd.api.types.is_numeric_dtype(s) and nun > 50 and n > 0 and nun / n > 0.5:
        return True, f"it looks like an identifier ({nun} unique values in {n} rows)"
    # A date stored as text (e.g. service_date as an object column).
    if s.dtype == object:
        sample = s.dropna().astype(str).head(200)
        if len(sample) >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = _pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return True, "its values are dates"
    return False, ""


_NO_TARGET_MSG = (
    "No suitable supervised target column was found (rejected: {detail}). The "
    "dataset looks UNLABELLED: the candidate columns are dates or identifiers, "
    "which cannot be a prediction target. Do NOT invent a label and train a "
    "classifier on it — that only leaks and scores a meaningless perfect AUC. "
    "If the task is fraud or anomaly detection, use unsupervised methods "
    "(IsolationForest / DBSCAN) and rule-based rate indicators per provider "
    "(e.g. paid-to-billed ratio, claims per client, price vs peer median)."
)


class InferTargetTool(BaseTool):
    name = "infer_target"
    description = (
        "Automatically determine the best target column for ML using column stats, "
        "the project brief, and (optionally) a keyword hint. Sets TARGET in the kernel."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "hint": {"type": "string",
                     "description": "Optional keyword hint, e.g. 'churn', 'price', 'outcome'"}
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        df = kernel.namespace.get("df")
        if df is None:
            return ToolOutput(success=False, text="No dataframe loaded.",
                              error="df is None")

        rejected: list[str] = []

        def _accept(target: str, reason: str) -> ToolOutput | None:
            """Set the target only if it survives the suitability guard."""
            if not target or target not in df.columns:
                return None
            bad, why = _is_unsuitable_target(df, target)
            if bad:
                rejected.append(f"`{target}` ({why})")
                return None
            session.target = target
            kernel.namespace["TARGET"] = target
            return ToolOutput(
                success=True,
                text=f"Target column selected: `{target}` ({reason})\n"
                     f"Unique: {df[target].nunique()} | Dtype: {df[target].dtype} | "
                     f"Sample: {df[target].value_counts().head(5).to_dict()}",
                artifacts={"target": target},
            )

        # Priority 1: TASK_SPEC already set by understand_task
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        if task_spec and task_spec.get("target_column"):
            out = _accept(task_spec["target_column"], "from task spec")
            if out:
                return out

        # Infer from task meaning and provenance, never column position or a word list.
        if llm_client and llm_client.is_available():
            prompt = (
                "Identify a supervised target ONLY if supported by the user's task. "
                "A continuous numeric outcome may have one unique value per row. "
                "Unlabelled exploration/anomaly detection has no supervised target. "
                "Do not invent labels. Return JSON {\"target\": null or exact column, "
                "\"reason\": \"evidence from task\"}.\n" + json.dumps({
                    "brief": session.brief, "hint": args.get("hint", ""),
                    "task": task_spec, "columns": {str(c): str(t) for c, t in df.dtypes.items()},
                    "request": getattr(kernel, "task_instruction", "")}, default=str))
            try:
                data = json.loads(llm_client._generate(prompt, temperature=0.0,
                                  max_tokens=500, json_mode=True))
                out = _accept(data.get("target"), str(data.get("reason", "")))
                if out:
                    return out
            except Exception:
                pass

        # Nothing suitable. Every candidate was a date or an identifier, which
        # means this is very likely an unlabelled task. Fail loudly and steer
        # the agent to unsupervised methods rather than let it invent a target.
        detail = "; ".join(rejected) if rejected else "no candidate columns"
        return ToolOutput(
            success=False,
            text=_NO_TARGET_MSG.format(detail=detail),
            error="no valid target; task appears unlabelled",
        )


class MutualInfoTool(BaseTool):
    name = "mutual_information"
    description = (
        "Compute mutual information scores between ALL features (numeric + categorical) "
        "and the target. Reports raw MI, entropy-normalised MI (IGR), and — for binary "
        "classification — permutation importance. Stores MI_SCORES in kernel for "
        "downstream use. Requires target to be set."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    @staticmethod
    def _ensure_target(session, kernel, llm_client) -> str | None:
        """Return current target, auto-inferring it if not yet set."""
        target = session.target or kernel.namespace.get("TARGET")
        if target:
            return target
        try:
            out = InferTargetTool().execute({}, session=session,
                                            kernel=kernel, llm_client=llm_client)
            if out.success:
                return session.target or kernel.namespace.get("TARGET")
        except Exception:
            pass
        return None

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        target = self._ensure_target(session, kernel, llm_client)
        if not target:
            return ToolOutput(success=False, text="Target not set. Run infer_target first.",
                              error="no target")
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        task_type = task_spec.get("task_type", "")
        code = textwrap.dedent(f"""
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            from sklearn.preprocessing import LabelEncoder
            from sklearn.inspection import permutation_importance
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            import numpy as np

            _tgt = {repr(target)}
            _task_type = {repr(task_type)}
            if _tgt not in df.columns:
                raise KeyError(
                    f"Target column '{{_tgt}}' not found in df. "
                    f"Available columns: {{list(df.columns[:15])}}"
                )
            _df_mi = df.copy()

            # ── Encode categorical features ──────────────────────────────────
            _le_map = {{}}
            for _c in _df_mi.select_dtypes('object').columns:
                if _c == _tgt: continue
                _le_map[_c] = LabelEncoder()
                _df_mi[_c] = _le_map[_c].fit_transform(_df_mi[_c].astype(str))

            _X = _df_mi.drop(columns=[_tgt]).select_dtypes(include='number').fillna(0)
            _y_raw = _df_mi[_tgt]

            # ── Encode target if categorical ─────────────────────────────────
            _is_cls = (_y_raw.dtype == object or _y_raw.nunique() < 20
                       or 'classif' in _task_type or 'binary' in _task_type)
            if _y_raw.dtype == object:
                _le_t = LabelEncoder()
                _y = pd.Series(_le_t.fit_transform(_y_raw.astype(str)), name=_tgt)
            else:
                _y = _y_raw.fillna(_y_raw.median())

            # ── Mutual Information ───────────────────────────────────────────
            if _is_cls:
                _mi = mutual_info_classif(_X, _y, random_state=0, n_neighbors=5)
            else:
                _mi = mutual_info_regression(_X, _y, random_state=0, n_neighbors=5)
            _mi_s = pd.Series(_mi, index=_X.columns).sort_values(ascending=False)

            # ── Information Gain Ratio (normalise by feature entropy) ────────
            _igr = {{}}
            for _c in _X.columns:
                _p = _X[_c].value_counts(normalize=True)
                _h_feat = float(-(_p * np.log(_p)).sum())
                _igr[_c] = _mi_s[_c] / _h_feat if _h_feat > 1e-9 else 0.0
            _igr_s = pd.Series(_igr).sort_values(ascending=False)

            # Store in kernel for downstream consumption
            MI_SCORES = _mi_s
            IGR_SCORES = _igr_s

            print("=== Mutual Information Ranking ===")
            print(f"Features evaluated: {{len(_X.columns)}} "
                  f"({{len([c for c in _le_map])}} categoricals encoded)")
            print(f"\\nTop features by raw MI:")
            print(_mi_s.head(20).round(4).to_string())
            print(f"\\nTop features by Information Gain Ratio (entropy-normalised):")
            print(_igr_s.head(20).round(4).to_string())

            # ── Permutation importance (quick RF baseline) ───────────────────
            _n = len(_X)
            _perm_s = None
            if _n <= 50_000:
                _rf_mi = (RandomForestClassifier if _is_cls else RandomForestRegressor)(
                    n_estimators=50, max_depth=6, random_state=0, n_jobs=-1)
                _rf_mi.fit(_X, _y)
                _perm = permutation_importance(_rf_mi, _X, _y, n_repeats=5,
                                               random_state=0, n_jobs=-1)
                _perm_s = pd.Series(_perm.importances_mean, index=_X.columns
                                    ).sort_values(ascending=False)
                print(f"\\nTop features by Permutation Importance (RF baseline):")
                print(_perm_s.head(20).round(4).to_string())
                PERM_SCORES = _perm_s

            # ── Plot: side-by-side MI and IGR ────────────────────────────────
            _top_n = 20
            _mi_plot = _mi_s.head(_top_n)
            _igr_plot = _igr_s.reindex(_mi_plot.index)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(5, _top_n*0.38)))
            ax1.barh(_mi_plot.index[::-1], _mi_plot.values[::-1], color='steelblue')
            ax1.set_xlabel('Mutual Information'); ax1.set_title('Raw MI (bits)')
            ax2.barh(_igr_plot.index[::-1], _igr_plot.values[::-1], color='#ff7f0e')
            ax2.set_xlabel('Information Gain Ratio')
            ax2.set_title('IGR — entropy-normalised MI\\n(controls for high-cardinality bias)')
            plt.suptitle(f'Feature Relevance  (target: {{_tgt}})', fontsize=13)
            plt.tight_layout(); plt.show()

            # ── Plot: permutation importance if computed ──────────────────────
            if _perm_s is not None:
                _pp = _perm_s.head(_top_n)
                fig2, ax3 = plt.subplots(figsize=(9, max(4, len(_pp)*0.38)))
                ax3.barh(_pp.index[::-1], _pp.values[::-1], color='#2ca02c')
                ax3.axvline(0, color='black', linewidth=0.8)
                ax3.set_xlabel('Mean decrease in score (permutation)')
                ax3.set_title(f'Permutation Importance — RF  (target: {{_tgt}})')
                plt.tight_layout(); plt.show()
        """).strip()
        out = kernel.execute(code)
        return ToolOutput(success=out.success, text=out.as_text(5000),
                          figures=out.figures, error=out.error)


class FeatureEngineeringTool(BaseTool):
    name = "feature_engineering"
    description = (
        "Generate interaction features, encode categoricals, handle nulls, "
        "and select the top-K most important features using SHAP or permutation importance."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "top_k": {"type": "integer", "default": 30,
                      "description": "Number of features to keep after selection"}
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        if kernel.namespace.get("df") is None:
            return ToolOutput(False, "", error="Load data first.")
        kernel.namespace.pop("df_eng", None)
        return ToolOutput(True, "Feature preprocessing will be fitted inside training folds. "
                          "Raw predictors and labels are preserved; no global feature selection was fitted.",
                          artifacts={"preprocessing": "fold-local"})


class TrainModelTool(BaseTool):
    name = "train_model"
    description = (
        "Train and compare ML models selected dynamically from problem type, data size, and "
        "class distribution (RF, ExtraTrees, GBM, LightGBM, XGBoost, LR/Ridge/ElasticNet). "
        "Performs 5-fold CV, reports the most appropriate eval metric, plots importances."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["classification", "regression", "auto"],
                "default": "auto",
            },
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Model keys to compare: rf, extra_trees, gbm, lgbm, xgb, lr "
                    "(classification) or rf, extra_trees, gbm, lgbm, xgb, ridge, "
                    "elasticnet, lasso (regression). If omitted, uses recommended_models "
                    "from TASK_SPEC or a data-size default."
                ),
            },
            "eval_metric": {
                "type": "string",
                "description": "Override evaluation metric (auto by default)",
                "default": "auto",
            },
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        from agent.pipeline.model_builder.supervised import train
        kernel.namespace.pop("best_model", None)
        kernel.namespace.pop("TRAINING_RESULT", None)
        spec = kernel.namespace.get("TASK_SPEC") or {}
        target = session.target or kernel.namespace.get("TARGET") or spec.get("target_column")
        task = args.get("task_type", "auto")
        if task == "auto":
            task = spec.get("task_type")
        if not target or task not in {"classification", "binary_classification",
                                      "multiclass_classification", "regression"}:
            return ToolOutput(False, "", error="Identify a target and supervised task type first; "
                              "an unlabelled task must not be converted into supervised training.")
        models = args.get("models") or spec.get("recommended_models") or ["lgbm", "rf", "lr"]
        try:
            result = train(kernel.namespace["df"], target=target, task_type=task,
                           models=models, leakage=kernel.namespace.get("LEAKAGE_COLS", []),
                           metric=args.get("eval_metric") or spec.get("evaluation_metric"),
                           group=spec.get("group_column"), time_column=spec.get("time_column"))
        except Exception as exc:
            return ToolOutput(False, "", error=f"Training failed: {exc}")
        kernel.namespace["best_model"] = result["model"]
        summary = {k: v for k, v in result.items() if k != "model"}
        kernel.namespace["TRAINING_RESULT"] = summary
        return ToolOutput(True, json.dumps(summary, indent=2), artifacts=summary,
                          math_trace=result["math_trace"])


class VisualizeTool(BaseTool):
    name = "visualize"
    description = (
        "Generate visualizations: distributions, correlations, pairplots, "
        "confusion matrix, ROC curve, residual plots, time-series, image grids, "
        "audio waveforms / spectrograms. Task-aware plots: feature_target (KDE/violin "
        "by class), class_separation (univariate AUC ranking), correlation_target "
        "(feature-target correlation bar)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "plot_type": {
                "type": "string",
                "enum": ["distributions", "correlation", "target_dist",
                         "feature_target", "class_separation", "correlation_target",
                         "pairplot", "confusion_matrix", "roc_curve",
                         "residuals", "waveform", "spectrogram", "image_grid"],
                "description": (
                    "Type of plot. Task-aware: 'feature_target' = KDE/violin by class, "
                    "'class_separation' = univariate AUC per feature, "
                    "'correlation_target' = feature-vs-target correlation bar."
                ),
            },
            "columns": {
                "type": "array", "items": {"type": "string"},
                "description": "Columns to include (optional subset)",
            },
            "top_n": {
                "type": "integer",
                "description": "Max features to show (default 12 for feature_target, 20 for others)",
            },
        },
        "required": ["plot_type"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        plot_type = args.get("plot_type", "distributions")
        columns = args.get("columns", [])
        top_n = args.get("top_n", 0)
        cols_repr = repr(columns)
        target = session.target or kernel.namespace.get("TARGET") or ""

        code_map = {
            "distributions": textwrap.dedent(f"""
                _cols = {cols_repr} or df.select_dtypes('number').columns[:12].tolist()
                fig, axes = plt.subplots(len(_cols)//4+1, 4, figsize=(16, 4*(len(_cols)//4+1)))
                for ax, col in zip(axes.flat, _cols):
                    df[col].dropna().hist(ax=ax, bins=30, color='steelblue', edgecolor='white')
                    ax.set_title(col, fontsize=9); ax.set_xlabel('')
                for ax in axes.flat[len(_cols):]: ax.set_visible(False)
                plt.suptitle('Feature Distributions', fontsize=13, y=1.01)
                plt.tight_layout(); plt.show()
            """),
            "correlation": textwrap.dedent("""
                import seaborn as _sns
                _num = df.select_dtypes('number').fillna(0)
                _corr = _num.corr(method='spearman')
                _n_c = len(_corr)
                try:
                    _cg = _sns.clustermap(
                        _corr, method='ward', metric='euclidean',
                        cmap='coolwarm', center=0, vmin=-1, vmax=1,
                        annot=_n_c <= 15, fmt='.1f',
                        linewidths=0.2 if _n_c <= 15 else 0,
                        figsize=(max(8, _n_c*0.55), max(7, _n_c*0.55)),
                    )
                    _cg.ax_heatmap.set_title('Clustered Spearman Correlation', pad=12)
                    plt.show()
                except Exception:
                    plt.figure(figsize=(max(8, _n_c//2), max(6, _n_c//2)))
                    _sns.heatmap(_corr, annot=_n_c<=15, fmt='.2f', cmap='coolwarm',
                                center=0, square=True, linewidths=0.3)
                    plt.title('Spearman Correlation Matrix'); plt.tight_layout(); plt.show()
            """),
            "target_dist": textwrap.dedent(f"""
                _t = {repr(target)} or (TARGET if 'TARGET' in dir() else '')
                if _t and _t in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if df[_t].dtype == object or df[_t].nunique() < 20:
                        df[_t].value_counts().sort_index().plot(kind='bar', ax=ax,
                            color='coral', edgecolor='white')
                        for p in ax.patches:
                            ax.annotate(f'{{p.get_height():,.0f}}',
                                        (p.get_x()+p.get_width()/2, p.get_height()),
                                        ha='center', va='bottom', fontsize=8)
                    else:
                        df[_t].hist(ax=ax, bins=40, color='coral', edgecolor='white')
                    ax.set_title(f'Target distribution: {{_t}}')
                    plt.tight_layout(); plt.show()
                else:
                    print("Target not set — set TARGET first")
            """),
            # ── task-aware: KDE / violin by class ──────────────────────────
            "feature_target": textwrap.dedent(f"""
                from sklearn.preprocessing import LabelEncoder
                import numpy as np
                _tgt = {repr(target)} or (TARGET if 'TARGET' in dir() else '')
                _top_n = {top_n or 12}
                _user_cols = {cols_repr}
                if not _tgt or _tgt not in df.columns:
                    print("Target not set or not in df — cannot plot feature_target")
                else:
                    _y = df[_tgt]
                    _classes = sorted(_y.dropna().unique())
                    _n_cls = len(_classes)
                    _palette = plt.cm.Set1.colors if _n_cls <= 9 else plt.cm.tab20.colors

                    # Pick features to plot
                    if _user_cols:
                        _num_feats = [c for c in _user_cols if c != _tgt
                                      and c in df.select_dtypes('number').columns]
                        _cat_feats = [c for c in _user_cols if c != _tgt
                                      and df[c].dtype == object]
                    else:
                        # Use MI_SCORES if available, else correlation
                        if 'MI_SCORES' in dir():
                            _rank = MI_SCORES.drop(index=[_tgt], errors='ignore')
                            _num_feats = [c for c in _rank.index
                                         if c in df.select_dtypes('number').columns][:_top_n]
                            _cat_feats = [c for c in _rank.index
                                         if df[c].dtype == object][:4]
                        else:
                            _num_cols_all = [c for c in df.select_dtypes('number').columns
                                             if c != _tgt]
                            _corrs = {{c: abs(df[c].fillna(0).corr(_y))
                                      for c in _num_cols_all}}
                            _num_feats = sorted(_corrs, key=_corrs.get, reverse=True)[:_top_n]
                            _cat_feats = [c for c in df.select_dtypes('object').columns
                                         if c != _tgt and df[c].nunique() <= 30][:4]

                    # ── Numeric: KDE by class ────────────────────────────────
                    if _num_feats:
                        _nf = len(_num_feats)
                        _ncols = min(3, _nf)
                        _nrows = (_nf + _ncols - 1) // _ncols
                        fig, axes = plt.subplots(_nrows, _ncols,
                                                 figsize=(_ncols*5, _nrows*3.5))
                        _axs = axes.flat if hasattr(axes, 'flat') else [axes]
                        for _i, _feat in enumerate(_num_feats):
                            _ax = list(_axs)[_i]
                            for _j, _cls in enumerate(_classes):
                                _vals = df.loc[_y == _cls, _feat].dropna()
                                if len(_vals) > 1:
                                    try:
                                        _vals.plot.kde(ax=_ax,
                                            label=f'{{_cls}} (n={{len(_vals)}})',
                                            color=_palette[_j % len(_palette)],
                                            linewidth=2)
                                    except Exception:
                                        _vals.hist(ax=_ax, bins=20, alpha=0.5,
                                            label=f'{{_cls}}',
                                            color=_palette[_j % len(_palette)])
                            _ax.set_title(_feat, fontsize=9)
                            _ax.legend(fontsize=7); _ax.set_xlabel('')
                        for _ax in list(_axs)[_nf:]: _ax.set_visible(False)
                        plt.suptitle(f'Feature Distributions by Class  (target: {{_tgt}})',
                                     fontsize=12, y=1.01)
                        plt.tight_layout(); plt.show()

                    # ── Categorical: stacked bar by class ────────────────────
                    for _cc in _cat_feats:
                        try:
                            _ct = pd.crosstab(df[_cc], _y, normalize='index')
                            _ct.plot(kind='bar', stacked=True, figsize=(10, 3),
                                     colormap='Set2', edgecolor='white', linewidth=0.4)
                            plt.title(f'{{_cc}} → class mix  (target: {{_tgt}})')
                            plt.ylabel('proportion'); plt.xlabel(_cc)
                            plt.legend(title=_tgt, bbox_to_anchor=(1.01,1),
                                       loc='upper left', fontsize=8)
                            plt.tight_layout(); plt.show()
                        except Exception as _e:
                            print(f"  Skipped {{_cc}}: {{_e}}")
            """),
            # ── task-aware: univariate AUC per feature ──────────────────────
            "class_separation": textwrap.dedent(f"""
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import LabelEncoder
                import numpy as np
                _tgt = {repr(target)} or (TARGET if 'TARGET' in dir() else '')
                _top_n = {top_n or 20}
                if not _tgt or _tgt not in df.columns:
                    print("Target not set — cannot compute class separation")
                else:
                    _y_raw = df[_tgt]
                    if _y_raw.dtype == object:
                        _le = LabelEncoder()
                        _y = pd.Series(_le.fit_transform(_y_raw.astype(str)))
                    else:
                        _y = _y_raw.fillna(_y_raw.median())
                    _n_cls = _y.nunique()
                    _num_cols = [c for c in df.select_dtypes('number').columns if c != _tgt]
                    _aucs = {{}}
                    for _c in _num_cols:
                        try:
                            _v = df[_c].fillna(df[_c].median()).values
                            if _n_cls == 2:
                                _a = roc_auc_score(_y, _v)
                                _aucs[_c] = max(_a, 1 - _a)
                            else:
                                # OvR mean AUC for multiclass
                                from sklearn.preprocessing import label_binarize
                                _yb = label_binarize(_y, classes=sorted(_y.unique()))
                                _auc_vals = []
                                for _k in range(_yb.shape[1]):
                                    try:
                                        _auc_vals.append(roc_auc_score(_yb[:,_k], _v))
                                    except Exception:
                                        pass
                                _aucs[_c] = np.mean(_auc_vals) if _auc_vals else 0.5
                        except Exception:
                            pass
                    _auc_s = pd.Series(_aucs).sort_values(ascending=False)
                    print(f"Univariate class-separation AUC (target: {{_tgt}}):")
                    print(_auc_s.head(30).round(4).to_string())

                    _plot = _auc_s.head(_top_n)
                    fig, ax = plt.subplots(figsize=(9, max(4, len(_plot)*0.38)))
                    _colors = ['#d62728' if v < 0.55 else '#2ca02c' if v > 0.75
                               else '#ff7f0e' for v in _plot.values]
                    ax.barh(_plot.index[::-1], _plot.values[::-1], color=_colors[::-1])
                    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8,
                               label='random baseline')
                    ax.set_xlabel('Univariate ROC-AUC  (0.5=random, 1.0=perfect)')
                    ax.set_title(f'Feature Separability Ranking  (target: {{_tgt}})')
                    ax.legend(fontsize=8)
                    plt.tight_layout(); plt.show()
            """),
            # ── task-aware: feature-target correlation bar ──────────────────
            "correlation_target": textwrap.dedent(f"""
                import numpy as np
                from sklearn.preprocessing import LabelEncoder
                _tgt = {repr(target)} or (TARGET if 'TARGET' in dir() else '')
                _top_n = {top_n or 25}
                if not _tgt or _tgt not in df.columns:
                    print("Target not set — cannot plot correlation_target")
                else:
                    _y_raw = df[_tgt]
                    if _y_raw.dtype == object:
                        _le = LabelEncoder()
                        _y = pd.Series(_le.fit_transform(_y_raw.astype(str)), name=_tgt)
                    else:
                        _y = _y_raw.fillna(_y_raw.median())
                    _num_cols = [c for c in df.select_dtypes('number').columns if c != _tgt]
                    _corrs = {{c: df[c].fillna(0).corr(_y) for c in _num_cols}}
                    _corr_s = pd.Series(_corrs).dropna().sort_values(key=abs, ascending=False)
                    print(f"Feature→target Pearson correlations (top {{_top_n}}):")
                    print(_corr_s.head(_top_n).round(4).to_string())

                    _plot = _corr_s.head(_top_n)
                    fig, ax = plt.subplots(figsize=(9, max(4, len(_plot)*0.38)))
                    _colors = ['#d62728' if v < 0 else '#1f77b4' for v in _plot.values]
                    ax.barh(_plot.index[::-1], _plot.values[::-1], color=_colors[::-1])
                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.set_xlabel('Pearson r with target')
                    ax.set_title(f'Feature → Target Correlation  (target: {{_tgt}})')
                    for _i, (_name, _val) in enumerate(zip(_plot.index[::-1], _plot.values[::-1])):
                        ax.text(_val + (0.005 if _val >= 0 else -0.005), _i,
                                f'{{_val:.3f}}', va='center',
                                ha='left' if _val >= 0 else 'right', fontsize=7)
                    plt.tight_layout(); plt.show()
            """),
            "waveform": textwrap.dedent("""
                if 'audio' in dir() and audio is not None:
                    import librosa.display
                    plt.figure(figsize=(12, 3))
                    librosa.display.waveshow(audio if audio.ndim==1 else audio[0],
                                             sr=sample_rate)
                    plt.title('Waveform'); plt.tight_layout(); plt.show()
                else:
                    print("No audio loaded")
            """),
            "spectrogram": textwrap.dedent("""
                if 'audio' in dir() and audio is not None:
                    import librosa, librosa.display
                    _a = audio if audio.ndim==1 else audio[0]
                    _D = librosa.amplitude_to_db(np.abs(librosa.stft(_a)), ref=np.max)
                    plt.figure(figsize=(12, 4))
                    librosa.display.specshow(_D, sr=sample_rate, x_axis='time', y_axis='hz')
                    plt.colorbar(format='%+2.0f dB'); plt.title('Spectrogram')
                    plt.tight_layout(); plt.show()
                else:
                    print("No audio loaded")
            """),
            "image_grid": textwrap.dedent("""
                if 'image' in dir() and image is not None:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(image); plt.axis('off'); plt.title('Image')
                    plt.tight_layout(); plt.show()
                else:
                    print("No image loaded")
            """),
        }
        code = code_map.get(plot_type, f"print('Unknown plot type: {plot_type}')")
        out = kernel.execute(code.strip())
        return ToolOutput(success=out.success, text=out.as_text(2000),
                          figures=out.figures, error=out.error)


class ExecuteCodeTool(BaseTool):
    name = "execute_code"
    description = (
        "Run Python in an isolated sandbox for custom analysis, transformation, "
        "or visualization not covered by other tools. Loaded data is available "
        "by name (e.g. df). Each call runs fresh: pandas DataFrames and Series "
        "you create are kept and can be reused by name in later calls. New "
        "columns you add to the loaded data (df, df2, ...) are kept too, but "
        "any other change to it (dropping, filtering or overwriting values) is "
        "discarded; keep such versions under a new name (e.g. claims = "
        "df[df.paid_amount > 0]). Other "
        "objects (models, dicts, numbers) are not, so recompute or print what "
        "you need. Available: pandas, numpy, scipy, scikit-learn (e.g. "
        "IsolationForest, DBSCAN), matplotlib, seaborn; shap, xgboost, "
        "lightgbm and statsmodels are NOT installed. Print every figure you "
        "will cite. matplotlib plots are shown to the user automatically, but "
        "you cannot see them: only printed text comes back to you, so state "
        "findings from printed numbers, never from how a chart looks. To "
        "propose saving a file, write it under WORK_DIR; it is only published "
        "to the user's folder after they approve it. No network access."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        from agent.api.sandbox import execute
        code = args.get("code", "")
        try:
            out = execute(code, session=session, kernel=kernel)
        except Exception as exc:
            return ToolOutput(False, "", error=f"Isolated execution failed: {exc}")
        if out.success:
            # Record the step so the session's analysis can be exported as a
            # notebook or script later (build_notebook).
            cells = getattr(kernel, "analysis_cells", None)
            if not isinstance(cells, list):
                cells = []
                try:
                    kernel.analysis_cells = cells
                except Exception:  # noqa: BLE001
                    return out
            cells.append({"code": code, "output": (out.text or "")[:3000],
                          "figures": len(out.figures or []),
                          "finding": "", "answers": []})
        return out


class UnderstandTaskTool(BaseTool):
    name = "understand_task"
    description = (
        "Parse the project brief / PDF to extract the full DS task specification: "
        "what to predict, how the target is defined, evaluation metric, whether "
        "aggregation is needed, and what features to engineer. "
        "Run this FIRST when a brief is loaded. "
        "Uses the 4-layer pipeline: DocumentParser → TaskExtractor → "
        "SpecEnricher → TaskSpecValidator."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        from agent.api.doc_parser import DocumentParser
        from agent.api.spec_enricher import SpecEnricher, TaskSpecValidator
        from agent.api.task_extractor import TaskExtractor

        brief = session.brief or kernel.namespace.get("BRIEF", "")
        df = kernel.namespace.get("df")

        if not brief.strip():
            return ToolOutput(
                success=False,
                text="No brief loaded. Upload a PDF task brief first.",
                error="no brief",
            )

        # ── Collect secondary DataFrames (df2, df3, …) ───────────────
        secondary_dfs: list[Any] = []
        secondary_col_names: list[list[str]] = []
        for i in range(2, 10):
            sdf = kernel.namespace.get(f"df{i}")
            if sdf is not None:
                secondary_dfs.append(sdf)
                try:
                    secondary_col_names.append(list(sdf.columns))
                except Exception:
                    secondary_col_names.append([])

        # ── Layer 1: parse document ───────────────────────────────────
        parser = DocumentParser()
        # If the brief was loaded from a file, re-parse from the original
        # path for best structural extraction; otherwise wrap raw text.
        if session.brief_filename:
            # Try known upload location first
            upload_root = __import__("pathlib").Path("outputs/uploads") / session.session_id
            candidate = upload_root / session.brief_filename
            if candidate.exists():
                doc = parser.parse(candidate)
            else:
                doc = parser.parse_text(brief)
        else:
            doc = parser.parse_text(brief)

        # ── Layer 2: decomposed LLM extraction ───────────────────────
        df_stats = self._compute_df_stats(df)
        # Append secondary file info to df_stats so the extractor knows about them
        if secondary_dfs:
            sec_info: dict = {}
            for i, sdf in enumerate(secondary_dfs, start=2):
                try:
                    var = f"df{i}"
                    # Derive filename from session upload paths (index i-2 = second file = index 1)
                    path_idx = i - 1  # data_paths[0] = df, data_paths[1] = df2, etc.
                    filename = (
                        session.data_paths[path_idx].name
                        if path_idx < len(session.data_paths)
                        else var
                    )
                    sec_info[var] = {
                        "n_rows": int(sdf.shape[0]),
                        "n_cols": int(sdf.shape[1]),
                        "columns": list(sdf.columns[:30]),
                        "filename": filename,
                    }
                except Exception:
                    pass
            if sec_info:
                df_stats["secondary_data_files"] = sec_info
        kernel.namespace["df_stats"] = df_stats
        # Combine all column names across primary and secondary files
        col_names = list(df.columns) if df is not None else []
        for scols in secondary_col_names:
            for c in scols:
                if c not in col_names:
                    col_names.append(c)

        # Stash primary filename in df_stats so _call_file_roles can use it
        if session.data_path:
            df_stats["filename"] = session.data_path.name

        # Grab last user message as hint for file-role re-runs after clarification
        user_hint = next(
            (m.text for m in reversed(session.messages) if m.role == "user"), ""
        )

        raw_spec = None
        if llm_client and llm_client.is_available():
            from agent.api.llm import SCOPE_MODEL  # noqa: PLC0415
            from agent.api.task_extractor import read_brief  # noqa: PLC0415
            extractor = TaskExtractor(
                ollama_base_url=llm_client.host,
                model=llm_client.model,
                timeout=llm_client._timeout,
                scope_model=SCOPE_MODEL,
            )
            # One call that keeps the brief's own questions and deliverables.
            # The decomposed extraction is the fallback when that reply is
            # unusable.
            raw_spec = read_brief(llm_client, brief, df_stats, col_names)
            if raw_spec is not None and df_stats.get("secondary_data_files"):
                roles = extractor._call_file_roles(doc, df_stats, user_hint)
                question = roles.pop("clarification_needed", None)
                if question:
                    raw_spec.clarification_needed = question
                else:
                    raw_spec.file_role_map = {
                        var: info.get("role", "unknown")
                        for var, info in roles.items() if isinstance(info, dict)}
            if raw_spec is None:
                raw_spec = extractor.extract(doc, df_stats, col_names, user_hint=user_hint)

        # ── Clarification early return ────────────────────────────────
        # If file-role reasoning couldn't assign roles confidently, surface
        # the clarification question and wait for a user reply before continuing.
        from agent.api.task_extractor import RawSpec  # noqa: PLC0415
        if raw_spec is not None and getattr(raw_spec, "clarification_needed", False):
            kernel.namespace["FILE_ROLE_CLARIFICATION"] = {
                "clarification_pending": True,
                "question": raw_spec.clarification_needed,
            }
            return ToolOutput(
                success=True,
                text=raw_spec.clarification_needed,
                artifacts={"clarification_pending": True},
            )

        # ── Layer 3: data-driven enrichment ──────────────────────────
        if raw_spec is None:
            raw_spec = RawSpec(
                task_type="unknown", task_type_confidence=0.0,
                target_column_hint="", target_condition="",
                aggregation_needed=False, aggregation_key="",
                evaluation_metric="roc_auc", task_description="",
                source="failed",
            )
        enricher = SpecEnricher()
        enriched = enricher.enrich(raw_spec, df, col_names)

        # ── Layer 4: validation ───────────────────────────────────────
        validator = TaskSpecValidator()
        is_valid, failures = validator.validate(enriched, col_names)

        # ── Build TASK_SPEC dict (backward-compatible key names) ──────
        spec: dict = {
            # New fields
            "task_description":       enriched.task_description,
            "task_type":              enriched.task_type,
            "target_column":          enriched.target_column,
            "target_condition":       enriched.target_condition,
            "aggregation_needed":     enriched.aggregation_needed,
            "aggregation_key":        enriched.aggregation_key,
            "evaluation_metric":      enriched.evaluation_metric,
            "class_imbalance_ratio":  enriched.class_imbalance_ratio,
            "confidence":             enriched.task_type_confidence,
            "enrichment_notes":       enriched.enrichment_notes,
            "is_valid":               is_valid,
            "validation_failures":    failures,
            # Specific requirements extracted verbatim from the brief
            "specific_requirements":  getattr(raw_spec, "specific_requirements", []),
            # The brief in its own terms: objective, questions, deliverables,
            # audiences, constraints, data notes, open questions
            "brief":                  getattr(raw_spec, "brief", {}) or {},
            # True when instructions indicate the target variable needs a second file
            "requires_secondary_file": getattr(raw_spec, "requires_secondary_file", False),
            # File-role map (populated when multiple files loaded)
            "file_role_map": getattr(raw_spec, "file_role_map", {}),
            # Kernel var name of the file holding target labels (if any)
            "target_source_var": next(
                (var for var, role in getattr(raw_spec, "file_role_map", {}).items()
                 if role == "target_derivation_source"),
                None,
            ),
            # Backward-compatible aliases used by downstream tools
            "target_definition": (
                enriched.target_condition
                if enriched.target_condition
                else enriched.target_column
            ),
            "target_is_derived": bool(enriched.target_condition),
        }

        kernel.namespace["TASK_SPEC"] = spec
        if enriched.aggregation_key:
            kernel.namespace["AGGREGATION_KEY"] = enriched.aggregation_key

        # Set TARGET if the column exists in the dataframe already
        if enriched.target_column and df is not None and enriched.target_column in df.columns:
            session.target = enriched.target_column
            kernel.namespace["TARGET"] = enriched.target_column

        # ── Build summary ─────────────────────────────────────────────
        conf = enriched.task_type_confidence
        conf_emoji = "✓" if conf >= 0.7 else ("⚠" if conf >= 0.5 else "✗")
        lines = [
            "=== Task Specification (4-layer pipeline) ===",
            f"Task      : {enriched.task_description}",
            f"Type      : {enriched.task_type}  |  Metric: {enriched.evaluation_metric}",
            f"Target    : {enriched.target_column}"
            + (f" (condition: {enriched.target_condition})" if enriched.target_condition else ""),
            f"Derived?  : {bool(enriched.target_condition)}",
            f"Aggregate : {enriched.aggregation_needed}"
            + (f" (by \'{enriched.aggregation_key}\')" if enriched.aggregation_needed else ""),
            f"Confidence: {conf:.0%} {conf_emoji}",
        ]
        if enriched.class_imbalance_ratio is not None:
            lines.append(f"Imbalance : {enriched.class_imbalance_ratio:.3f}")
        if enriched.enrichment_notes:
            lines.append("Notes     : " + "; ".join(enriched.enrichment_notes))
        brief_model = getattr(raw_spec, "brief", {}) or {}
        reqs = getattr(raw_spec, "specific_requirements", [])
        if brief_model:
            for label, key in (("Questions the brief asks", "questions"),
                               ("Deliverables", "deliverables"),
                               ("Audiences", "audiences"),
                               ("Constraints", "constraints"),
                               ("Open questions", "open_questions")):
                items = brief_model.get(key) or []
                if items:
                    lines.append(f"{label}:")
                    lines.extend(f"  {i}. {x}" for i, x in enumerate(items[:12], 1))
        elif reqs:
            lines.append("Requirements:")
            lines.extend(f"  • {r}" for r in reqs[:10])
        # File roles section
        file_role_map = getattr(raw_spec, "file_role_map", {})
        if file_role_map:
            lines.append("\n✓ File roles identified:")
            for var, role in file_role_map.items():
                df_obj = kernel.namespace.get(var)
                shape_info = (
                    f" ({df_obj.shape[0]:,} rows × {df_obj.shape[1]} cols)"
                    if df_obj is not None else ""
                )
                lines.append(f"  {var}{shape_info} → {role}")
        elif getattr(raw_spec, "requires_secondary_file", False):
            # Fall back to old phrase-count-based warning when no role map (LLM offline etc.)
            _df2 = kernel.namespace.get("df2")
            if _df2 is None:
                lines.append(
                    "\n⚠ SECOND FILE NEEDED: The target variable must be derived from "
                    "a separate data file mentioned in the brief (e.g. a future-period "
                    "dataset). Please upload that file before running the pipeline, "
                    "otherwise the agent will fall back to the training data's outcome "
                    "column which will cause data leakage."
                )
            else:
                lines.append(
                    f"\n✓ Secondary file already loaded as df2 "
                    f"({_df2.shape[0]:,} rows × {_df2.shape[1]} cols) — "
                    "target derivation will use this dataset."
                )
        if not is_valid:
            lines.append("⚠ Validation issues:")
            lines.extend(f"  - {f}" for f in failures)
        if enriched.target_condition:
            lines.append(
                f"\n→ Target must be DERIVED via aggregate_data "
                f"(group by \'{enriched.aggregation_key or '?'}\', "
                f"condition: {enriched.target_condition})"
            )
        elif enriched.target_column and df is not None and enriched.target_column in df.columns:
            lines.append(f"\n✓ TARGET set to existing column: `{enriched.target_column}`")

        return ToolOutput(
            success=True,
            text="\n".join(lines),
            artifacts={
                "task_spec": spec,
                "confidence": conf,
                "brief_type": raw_spec.source,
            },
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_df_stats(df) -> dict:
        """Pre-compute column statistics for TaskExtractor and SpecEnricher."""
        if df is None or (hasattr(df, "empty") and df.empty):
            return {"n_rows": 0, "n_cols": 0, "columns": {}}
        stats: dict = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": {},
        }
        for col in df.columns:
            series = df[col].dropna()
            col_stats: dict = {
                "dtype": str(df[col].dtype),
                "n_unique": int(series.nunique()),
                "null_rate": float(df[col].isnull().mean()),
            }
            if series.nunique() <= 20:
                col_stats["value_counts"] = series.value_counts().to_dict()
            if 1 < series.nunique() < len(df) * 0.5:
                col_stats["rows_per_group"] = len(df) / series.nunique()
            stats["columns"][col] = col_stats
        return stats

class AggregateDataTool(BaseTool):
    name = "aggregate_data"
    description = (
        "Aggregate incident/transaction-level data to entity-level "
        "(e.g. one row per establishment, customer, patient). "
        "Creates count, sum, mean, rate, and indicator features grouped by a key column. "
        "Optionally loads a secondary dataset and derives a binary target from it. "
        "Replaces df in the kernel with the aggregated entity-level dataframe."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "group_by": {
                "type": "string",
                "description": "Column to group by (e.g. 'establishment_id')",
            },
            "target_source_path": {
                "type": "string",
                "description": (
                    "Absolute path to a secondary CSV/parquet to derive the target from. "
                    "Leave empty to skip target creation."
                ),
            },
            "target_condition": {
                "type": "string",
                "description": (
                    "Python expression applied to the secondary dataset to flag positive cases. "
                    "e.g. \"incident_outcome.isin([1, 2, 3])\"  "
                    "The result is grouped by group_by column → any True → target = 1."
                ),
            },
            "target_column_name": {
                "type": "string",
                "default": "target",
                "description": "Name for the derived binary target column.",
            },
        },
        "required": ["group_by"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        group_by = args.get("group_by", "")
        target_path = args.get("target_source_path", "")
        target_cond = args.get("target_condition", "")
        target_col_name = args.get("target_column_name", "target")

        # ── Auto-fill from TASK_SPEC when planner didn't pass full args ─────
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        if task_spec:
            if not group_by and task_spec.get("aggregation_key"):
                group_by = task_spec["aggregation_key"]
            # Pull target definition when target must be derived
            if (not target_cond and task_spec.get("target_is_derived")
                    and task_spec.get("target_definition")):
                raw_def = task_spec["target_definition"]
                # Only use it if it looks like a usable pandas expression
                if any(op in raw_def for op in ["isin", "==", ">=", "<=", ">", "<", "in ["]):
                    target_cond = raw_def
            # Use task_spec metric hint to name target column clearly
            if task_spec.get("target_column") and target_col_name == "target":
                tc = task_spec["target_column"]
                if tc and tc != "null":
                    target_col_name = tc

        # ── If brief hints at serious harm / fatality, build default condition ─
        if not target_cond:
            brief_lo = (session.brief or "").lower()
            df_check = kernel.namespace.get("df")
            if (any(w in brief_lo for w in ["serious harm", "fatality", "fatal", "died"])
                    and df_check is not None
                    and "incident_outcome" in df_check.columns):
                target_cond = "incident_outcome in [1, 2, 3]"
                target_col_name = "serious_harm"

        # Auto-binarize: when task is binary_classification but target has >2 raw classes
        if not target_cond:
            ts_type = task_spec.get("task_type", "")
            orig_tgt = (task_spec.get("target_column", "")
                        or session.target
                        or kernel.namespace.get("TARGET", ""))
            df_check = kernel.namespace.get("df")
            if (df_check is not None
                    and "binary" in ts_type
                    and orig_tgt
                    and orig_tgt in df_check.columns
                    and df_check[orig_tgt].nunique() > 2):
                majority_class = int(df_check[orig_tgt].value_counts().index[0])
                target_cond = f"{orig_tgt} != {majority_class}"
                target_col_name = f"{orig_tgt}_binary"
                print(f"[auto-binarize] binary_classification detected, majority={majority_class}, "
                      f"condition: {target_cond}")

        # Guard: never aggregate by an industry/classification code
        _AGG_CAT_KWS = ("naics", "sic", "nace", "isco", "isic",
                         "industry_code", "sector_code", "zip_code", "postal_code")
        if group_by and any(kw in group_by.lower() for kw in _AGG_CAT_KWS):
            df_check = kernel.namespace.get("df")
            _override = ""
            if df_check is not None:
                for _eid in ["establishment_id", "employer_id", "entity_id",
                              "company_id", "customer_id"]:
                    if _eid in df_check.columns:
                        _override = _eid
                        break
            group_by = _override or "establishment_id"

        if not group_by:
            # Last resort: look for obvious key column in df
            df_check = kernel.namespace.get("df")
            if df_check is not None:
                for candidate in ["establishment_id", "employer_id", "entity_id",
                                   "company_id", "customer_id"]:
                    if candidate in df_check.columns:
                        group_by = candidate
                        break
        if not group_by:
            return ToolOutput(success=False, text="group_by column is required.",
                              error="missing group_by")

        agg_code = textwrap.dedent(f"""
            import numpy as np, pandas as pd
            _key = {repr(group_by)}
            # Save original incident-level df BEFORE aggregation (needed for target creation)
            df_orig = df.copy()
            print(f"Aggregating {{len(df)}} rows → {{df[_key].nunique()}} groups by '{{_key}}'")

            # ── numeric aggregations ─────────────────────────────────────────
            _orig_target = TARGET if 'TARGET' in dir() else ''
            _excluded = {{_key}} | ({{_orig_target}} if _orig_target else set())
            _num = [c for c in df.select_dtypes('number').columns if c not in _excluded]
            _agg_funcs = {{'mean': 'mean', 'sum': 'sum', 'max': 'max', 'count': 'count'}}
            _parts = []
            for _fn_name, _fn in _agg_funcs.items():
                _part = df.groupby(_key)[_num].agg(_fn)
                _part.columns = [f'{{c}}_{{_fn_name}}' for c in _part.columns]
                _parts.append(_part)
            _agg_df = pd.concat(_parts, axis=1).reset_index()

            # ── total row count per group ────────────────────────────────────
            _agg_df['n_incidents'] = df.groupby(_key).size().values

            # ── incident rate (per 100 hours or employees if available) ──────
            if 'total_hours_worked_mean' in _agg_df.columns:
                _agg_df['incident_rate_per_100k_hrs'] = (
                    _agg_df['n_incidents'] / (_agg_df['total_hours_worked_mean'] / 100_000 + 1e-9)
                ).round(4)

            # ── categorical mode features (pandas 2.x safe — no lambda in agg) ─
            _cat = [c for c in df.select_dtypes('object').columns
                    if c != _key and df[c].nunique() <= 100][:6]
            for _col in _cat:
                # nunique per group (always safe)
                _agg_df[f'{{_col}}_nunique'] = (
                    df.groupby(_key)[_col].nunique().values
                )
                # mode per group — use apply on GroupBy object, then map back
                try:
                    _grp = df.groupby(_key, sort=False)[_col]
                    _mode_s = _grp.apply(
                        lambda s: s.dropna().mode().iloc[0]
                        if s.dropna().size > 0 else None
                    )
                    # Ensure index aligns with _agg_df's key column
                    _mode_s.index.name = _key
                    _agg_df[f'{{_col}}_mode'] = (
                        _agg_df[_key].map(_mode_s).values
                    )
                except Exception as _mode_err:
                    # Final fallback: just use nunique, skip mode
                    print(f"  mode fallback for {{_col}}: {{_mode_err}}")

            df = _agg_df.copy()
            print(f"Aggregated dataframe shape: {{df.shape}}")
            print(f"Columns (first 20): {{list(df.columns[:20])}}")
        """).strip()

        out = kernel.execute(agg_code)
        if not out.success:
            return ToolOutput(success=False, text=out.as_text(2000),
                              error=out.error)

        # ── optional: load secondary dataset and derive binary target ────────
        if target_path or target_cond:
            load_target_code = ""
            if target_path:
                p = Path(target_path).expanduser().resolve()
                suffix = p.suffix.lower()
                reader = "pd.read_parquet" if suffix == ".parquet" else "pd.read_csv"
                load_target_code += textwrap.dedent(f"""
                    _df_sec = {reader}(r{repr(str(p))})
                    print(f"Secondary dataset loaded: {{_df_sec.shape}}")
                    print(f"Columns: {{_df_sec.columns.tolist()[:20]}}")
                """).strip() + "\n"
            else:
                load_target_code += "_df_sec = df_orig if 'df_orig' in dir() else df\n"

            if target_cond:
                load_target_code += textwrap.dedent(f"""
                    _key = {repr(group_by)}
                    _tgt_col = {repr(target_col_name)}
                    _cond_str = {repr(target_cond)}
                    # Evaluate condition on incident-level dataset (df_orig)
                    # Several eval strategies tried in sequence for robustness
                    _pos_mask = None
                    # Strategy 1: pandas eval (works for simple comparisons)
                    try:
                        _pos_mask = _df_sec.eval(_cond_str)
                    except Exception:
                        pass
                    # Strategy 2: rewrite "col in [a,b,c]" → col.isin([a,b,c])
                    if _pos_mask is None:
                        import re as _re
                        _expr2 = _re.sub(
                            r'(\\w+)\\s+in\\s+(\\[[\\d,\\s]+\\])',
                            r'_df_sec["\\1"].isin(\\2)',
                            _cond_str,
                        )
                        try:
                            _pos_mask = eval(_expr2, {{'_df_sec': _df_sec, 'pd': pd}})
                        except Exception:
                            pass
                    # Strategy 3: direct Python eval with df columns in scope
                    if _pos_mask is None:
                        try:
                            _ns = {{'df': _df_sec, '_df_sec': _df_sec, 'pd': pd}}
                            _ns.update({{c: _df_sec[c] for c in _df_sec.columns}})
                            _pos_mask = eval(_cond_str, _ns)
                        except Exception as _e3:
                            print(f"Target condition eval failed: {{_e3}}. Falling back to all-zero target.")
                            _pos_mask = pd.Series([False] * len(_df_sec), index=_df_sec.index)
                    _pos_ids = _df_sec.loc[_pos_mask, _key].unique()
                    df[_tgt_col] = df[_key].isin(_pos_ids).astype(int)
                    _vc = df[_tgt_col].value_counts()
                    print(f"Target '{{_tgt_col}}' distribution: {{_vc.to_dict()}}")
                    print(f"Positive rate: {{_vc.get(1,0)/len(df):.1%}}")
                    TARGET = _tgt_col
                """).strip()
                kernel.execute(f"TARGET = {repr(target_col_name)}")
                session.target = target_col_name

            out2 = kernel.execute(load_target_code)

            # Record which aggregated columns derive from the target source
            # so TrainModelTool can drop them as leakage.
            if target_cond:
                import re as _re_lk
                _m = _re_lk.match(r'\s*(\w+)\s+(?:in\b|==|!=|>=|<=|>|<)', target_cond.strip())
                if _m:
                    _leak_src = _m.group(1)
                    kernel.execute(
                        f"LEAKAGE_COLS = [c for c in df.columns "
                        f"if c.startswith({repr(_leak_src + '_')})]"
                    )

            combined_text = out.as_text(2000) + "\n" + out2.as_text(2000)
            return ToolOutput(
                success=out2.success,
                text=combined_text,
                figures=out.figures + out2.figures,
                artifacts={"aggregated": True, "target_column": target_col_name},
                error=out2.error,
            )

        # ── Post-aggregation: update TARGET to point at an aggregated column ─
        # When no target condition was applied, the original target column
        # (e.g. "outcome") no longer exists — it's now "outcome_max" etc.
        # Pick the best available aggregated version and update TARGET.
        old_target = session.target or kernel.namespace.get("TARGET", "")
        if old_target:
            df_agg = kernel.namespace.get("df")
            if df_agg is not None and old_target not in df_agg.columns:
                # Prefer _max (captures worst incident), then _mean, _sum, _count
                for suffix in ("_max", "_mean", "_sum", "_count"):
                    candidate = f"{old_target}{suffix}"
                    if candidate in df_agg.columns:
                        kernel.execute(f"TARGET = {repr(candidate)}")
                        session.target = candidate
                        task_spec = kernel.namespace.get("TASK_SPEC", {})
                        if task_spec:
                            task_spec["target_column"] = candidate
                        out_extra = f"\nTARGET updated: '{old_target}' → '{candidate}' (post-aggregation)"
                        return ToolOutput(
                            success=True,
                            text=out.as_text(3000) + out_extra,
                            figures=out.figures,
                            artifacts={"aggregated": True, "target_column": candidate},
                        )

        return ToolOutput(
            success=True,
            text=out.as_text(3000),
            figures=out.figures,
            artifacts={"aggregated": True},
            error=out.error,
        )


class BuildNotebookTool(BaseTool):
    name = "build_notebook"
    description = (
        "Export the analysis done in this session as a runnable Jupyter notebook "
        "(format='ipynb') or Python script (format='py'): the brief's questions, "
        "the plan, every successful execute_code step in order with the finding "
        "it produced, and a summary of the EDA agenda. The code is re-run in the "
        "sandbox from the source data first to check it executes. The file is a "
        "proposal that the user approves before it is written."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "format": {"type": "string", "enum": ["ipynb", "py"],
                       "description": "ipynb for a notebook, py for a script "
                                      "(default ipynb)"},
            "path": {"type": "string",
                     "description": "File name in the working directory, "
                                    "e.g. analysis.ipynb"},
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(False, "", error="Set a working directory first; "
                              "the file is proposed there.")
        cells = getattr(kernel, "analysis_cells", None) or []
        if not cells:
            return ToolOutput(False, "", error=(
                "No analysis has run in this session yet, so there is nothing to "
                "export. Run the analysis (execute_code) first."))
        path = str(args.get("path") or "").strip()
        fmt = str(args.get("format") or "").strip().lower().lstrip(".")
        if not fmt:
            fmt = "py" if path.endswith(".py") else "ipynb"
        if fmt not in ("ipynb", "py"):
            return ToolOutput(False, "", error="format must be 'ipynb' or 'py'.")
        path = path or f"analysis.{fmt}"
        if not path.endswith(f".{fmt}"):
            path = f"{Path(path).stem}.{fmt}"

        sections = _export_sections(session, kernel, cells, root)
        check = _verify_export(cells, session=session, kernel=kernel)
        if check:
            sections.insert(1, ("md", "> **Check failed:** re-running these cells in "
                                      f"the sandbox stopped with: `{check}`. "
                                      "Review before relying on this file."))
        content = _as_notebook(sections) if fmt == "ipynb" else _as_script(sections)
        out = WriteFileTool().execute({"path": path, "content": content, "overwrite": True},
                                      session=session, kernel=kernel, llm_client=llm_client)
        status = (f"Re-run check failed: {check}" if check else
                  "Re-run check passed: every cell executed from the source data.")
        return ToolOutput(out.success,
                          f"Exported {len(cells)} analysis step(s) as {path}. {status}\n"
                          + (out.text or ""),
                          error=out.error,
                          artifacts={"export": path, "verified": not check})


def _data_loader_lines(session, root: Path) -> list[str]:
    """Code that loads the session's data files under the names the analysis
    used (df, df2, ...), with paths relative to the working directory."""
    readers = {".csv": "pd.read_csv", ".parquet": "pd.read_parquet",
               ".xlsx": "pd.read_excel", ".xls": "pd.read_excel",
               ".json": "pd.read_json", ".tsv": "pd.read_csv"}
    lines = []
    for i, p in enumerate(getattr(session, "data_paths", []) or [], start=1):
        var = "df" if i == 1 else f"df{i}"
        p = Path(p)
        try:
            shown = str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            shown = str(p)
        reader = readers.get(p.suffix.lower())
        extra = ', sep="\\t"' if p.suffix.lower() == ".tsv" else ""
        lines.append(f"{var} = {reader}({shown!r}{extra})" if reader
                     else f"# {var}: load {shown} here")
    return lines or ["# df = pd.read_csv('your_data.csv')"]


def _export_sections(session, kernel, cells: list[dict], root: Path) -> list[tuple[str, str]]:
    """Markdown and code sections for an export, in reading order."""
    ns = kernel.namespace
    raw_spec = ns.get("TASK_SPEC")
    spec: dict = raw_spec if isinstance(raw_spec, dict) else {}
    raw_brief = spec.get("brief")
    brief: dict = raw_brief if isinstance(raw_brief, dict) else {}
    raw_agenda = ns.get("EDA_AGENDA")
    agenda: list = raw_agenda if isinstance(raw_agenda, list) else []
    by_id = {a.get("id"): a for a in agenda}
    title = brief.get("objective") or spec.get("task_description") or "Data analysis"
    out: list[tuple[str, str]] = [("md", f"# Analysis\n\n{title}")]
    if brief.get("questions"):
        out.append(("md", "## Questions from the brief\n\n" + "\n".join(
            f"{i}. {q}" for i, q in enumerate(brief["questions"], 1))))
    plan = ns.get("WORK_PLAN")
    if isinstance(plan, dict):
        md = _plan_to_markdown(plan, spec)
        out.append(("md", "## Plan\n\n" + (md.split("\n", 1)[1].lstrip()
                                          if md.startswith("# ") else md)))
    out.append(("md", "## Setup"))
    out.append(("code", "\n".join([
        "import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt",
        "import seaborn as sns", "from pathlib import Path", "",
        "pd.set_option('display.max_columns', 50)",
        "WORK_DIR = Path('.')", *_data_loader_lines(session, root)])))
    for i, cell in enumerate(cells, 1):
        head = [f"## Step {i}"]
        for qid in cell.get("answers") or []:
            if qid in by_id:
                head.append(f"Answers [{qid}]: {by_id[qid].get('question', '')}")
        if cell.get("finding"):
            head.append(f"Finding: {cell['finding']}")
        out.append(("md", "\n\n".join(head)))
        out.append(("code", str(cell.get("code", "")).strip()))
    if agenda:
        answered = [a for a in agenda if a.get("status") == "answered"]
        still = [a for a in agenda if a.get("status") != "answered"]
        lines = ["## Summary of findings", ""]
        lines += [f"- [{a['id']}] {a['question']}: {a.get('finding', '')}" for a in answered]
        if still:
            lines += ["", "Not yet answered:"] + [f"- [{a['id']}] {a['question']}" for a in still]
        out.append(("md", "\n".join(lines)))
    return out


def _verify_export(cells: list[dict], *, session, kernel) -> str:
    """Re-run the exported cells in the sandbox from the source data only.
    Returns "" when they all execute, otherwise the error headline."""
    from agent.api.sandbox import execute as sandbox_execute
    code = "\n\n".join(["import seaborn as sns"]
                       + [str(c.get("code", "")) for c in cells])
    try:
        res = sandbox_execute(code, session=session, kernel=kernel, dry_run=True)
    except Exception as exc:  # noqa: BLE001
        return f"could not run the check: {exc}"
    if res.success:
        return ""
    lines = [ln for ln in (res.error or "").strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "unknown error")[:300]


def _as_notebook(sections: list[tuple[str, str]]) -> str:
    cells: list[dict] = []
    for kind, text in sections:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": text})
    return json.dumps({"nbformat": 4, "nbformat_minor": 5,
                       "metadata": {"kernelspec": {"name": "python3", "language": "python",
                                                   "display_name": "Python 3"}},
                       "cells": cells}, indent=1)


def _as_script(sections: list[tuple[str, str]]) -> str:
    """Percent-format script: runs as plain Python, opens as cells in VS Code
    and Jupytext."""
    parts = []
    for kind, text in sections:
        if kind == "md":
            parts.append("# %% [markdown]\n" + "\n".join(
                f"# {ln}" if ln else "#" for ln in text.splitlines()))
        else:
            parts.append("# %%\n" + text)
    return "\n\n".join(parts) + "\n"


class ReadInstructionsTool(BaseTool):
    name = "read_instructions"
    description = (
        "Read and parse a document (PDF, text file) to extract analysis instructions, "
        "data dictionary, or task specification. Updates BRIEF in the kernel."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Path to the document. A bare filename is "
                                    "resolved inside the working directory."},
        },
        "required": ["path"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolOutput(success=False, text="",
                              error="read_instructions needs a 'path'.")
        # Resolve inside the working directory first. Every other file tool does
        # this via _ws_resolve_within; this one resolved against the process
        # working directory instead, so a bare filename pointed at wherever the
        # server happened to be started from and always failed.
        root = _ws_root(session, kernel)
        p = None
        if root is not None:
            cand = _ws_resolve_within(root, rel)
            if cand is not None and cand.is_file():
                p = cand
        if p is None:
            cand = Path(rel).expanduser().resolve()
            if cand.is_file():
                p = cand
        if p is None:
            hint = ""
            if root is not None:
                docs = sorted(f.name for f in root.iterdir()
                              if f.is_file()
                              and f.suffix.lower() in {".pdf", ".md", ".txt", ".rst"})
                if docs:
                    hint = (" Documents in the working directory: "
                            + ", ".join(repr(d) for d in docs[:8]) + ".")
            return ToolOutput(success=False, text="",
                              error=f"File not found: {rel!r}.{hint}")
        try:
            if p.suffix.lower() == ".pdf":
                from agent.api.pdf_reader import extract
                data = p.read_bytes()
                summary = extract(data)
                text = summary.text
                if summary.mentioned_target and not session.target:
                    session.target = summary.mentioned_target
                    kernel.namespace["TARGET"] = summary.mentioned_target
            else:
                text = p.read_text(errors="replace")
            session.brief += f"\n\n[Instructions from {p.name}]:\n{text}"
            kernel.namespace["BRIEF"] = session.brief
            return ToolOutput(
                success=True,
                text=f"Read {len(text):,} chars from `{p.name}`.\nPreview:\n{text[:500]}…",
                artifacts={"brief_source": str(p)},
            )
        except Exception as exc:
            return ToolOutput(success=False, text="", error=str(exc))


# ── web research (live internet, no API key) ──────────────────────────────────

_RESEARCH_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _ddg_search(query: str, max_results: int = 5,
                timeout: float = 12.0) -> list[dict]:
    """DuckDuckGo HTML endpoint search — no API key. Returns a list of
    {title, url, snippet}. Empty on any failure."""
    import httpx
    from lxml import html as lh

    try:
        r = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query}, headers={"User-Agent": _RESEARCH_UA},
            timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        return []

    doc = lh.fromstring(r.text)
    anchors = doc.xpath('//a[contains(@class,"result__a")]')
    snippets = doc.xpath('//a[contains(@class,"result__snippet")]')
    out: list[dict] = []
    for i, a in enumerate(anchors[:max_results]):
        href = a.get("href") or ""
        # DDG sometimes wraps the real URL in a redirect param uddg=.
        if "uddg=" in href:
            from urllib.parse import parse_qs, unquote, urlparse
            qs = parse_qs(urlparse(href).query)
            if qs.get("uddg"):
                href = unquote(qs["uddg"][0])
        snip = (snippets[i].text_content().strip()
                if i < len(snippets) else "")
        out.append({
            "title": a.text_content().strip(),
            "url": href,
            "snippet": snip,
        })
    return out


def _fetch_page_text(url: str, max_chars: int = 2500,
                     timeout: float = 12.0) -> str:
    """Fetch a page and extract its main visible text. Empty on failure."""
    import re

    import httpx
    from lxml import html as lh

    try:
        r = httpx.get(url, headers={"User-Agent": _RESEARCH_UA},
                      timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return ""
    except Exception:  # noqa: BLE001
        return ""

    try:
        doc = lh.fromstring(r.text)
        for bad in doc.xpath(
                "//script|//style|//noscript|//nav|//footer|//header|//aside"):
            bad.getparent().remove(bad)
        text = doc.text_content()
        text = re.sub(r"\n\s*\n+", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        return text[:max_chars]
    except Exception:  # noqa: BLE001
        return ""


def _pypi_info(name: str, timeout: float = 8.0) -> dict | None:
    """Look up a package on PyPI (official JSON API). None if not found."""
    import httpx
    try:
        r = httpx.get(f"https://pypi.org/pypi/{name}/json", timeout=timeout)
        if r.status_code != 200:
            return None
        info = r.json().get("info", {})
        return {
            "name": info.get("name", name),
            "version": info.get("version", ""),
            "summary": (info.get("summary") or "")[:200],
            "home_page": info.get("home_page") or info.get("project_url") or "",
            "requires_python": info.get("requires_python") or "",
        }
    except Exception:  # noqa: BLE001
        return None


class WebResearchTool(BaseTool):
    name = "web_research"
    description = (
        "Research a topic, method, or library on the LIVE web when the answer "
        "may be newer than your training data: latest library versions, new "
        "packages, recent best practices, current APIs, 2025/2026 developments. "
        "Runs a DuckDuckGo search, optionally reads the top result pages, and "
        "for Python packages adds the current PyPI version and summary. Returns "
        "titles, snippets, source URLs, and extracted text — always cite the URLs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "What to research, e.g. "
                      "'latest Python library for time-series forecasting 2026'"},
            "max_results": {"type": "integer",
                            "description": "Search results to return (default 5)"},
            "fetch_pages": {"type": "boolean",
                            "description": "Read the top result pages for detail "
                            "(default true)"},
        },
        "required": ["query"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        query = (args.get("query") or args.get("q") or "").strip()
        if not query:
            return ToolOutput(success=False, text="",
                              error="web_research needs a 'query'.")
        max_results = int(args.get("max_results", 5) or 5)
        fetch_pages = args.get("fetch_pages", True)

        results = _ddg_search(query, max_results=max_results)
        if not results:
            return ToolOutput(
                success=False, text="",
                error="Web search returned no results (search endpoint may be "
                      "rate-limited or offline). Try rephrasing the query.")

        lines = [f"Web research for: {query}", ""]
        for i, res in enumerate(results, 1):
            lines.append(f"[{i}] {res['title']}")
            lines.append(f"    {res['url']}")
            if res["snippet"]:
                lines.append(f"    {res['snippet'][:220]}")

        # PyPI enrichment: any pypi.org/project/<slug> result → official facts.
        seen_pkgs: set[str] = set()
        pypi_lines: list[str] = []
        for res in results:
            if "pypi.org/project/" in res["url"]:
                slug = res["url"].split("pypi.org/project/", 1)[1].strip("/")
                slug = slug.split("/")[0]
                if slug and slug.lower() not in seen_pkgs:
                    seen_pkgs.add(slug.lower())
                    info = _pypi_info(slug)
                    if info:
                        pypi_lines.append(
                            f"  {info['name']} {info['version']} — "
                            f"{info['summary']} "
                            f"(requires-python {info['requires_python'] or 'any'})")
        # If the query is one or two words, also try a direct PyPI lookup.
        if not pypi_lines and len(query.split()) <= 2:
            info = _pypi_info(query.split()[0].lower())
            if info:
                pypi_lines.append(
                    f"  {info['name']} {info['version']} — {info['summary']}")
        if pypi_lines:
            lines += ["", "PyPI (official, current):", *pypi_lines]

        # Read the top pages for substance.
        if fetch_pages:
            lines += ["", "Extracted from top sources:"]
            fetched = 0
            for res in results:
                if fetched >= 2:
                    break
                text = _fetch_page_text(res["url"])
                if text:
                    fetched += 1
                    lines.append(f"\n— {res['url']}\n{text}")

        return ToolOutput(
            success=True,
            text="\n".join(lines),
            artifacts={"sources": [r["url"] for r in results],
                       "query": query},
        )


# ── external data sources (connectors → session blackboard) ───────────────────

_CONNECTOR_CLASSES: dict[str, tuple[str, str]] = {
    "sqlite":     ("agent.connectors.sqlite_conn", "SQLiteConnector"),
    "flatfile":   ("agent.connectors.flatfile_conn", "FlatFileConnector"),
    "csv":        ("agent.connectors.flatfile_conn", "FlatFileConnector"),
    "postgres":   ("agent.connectors.postgres_conn", "PostgresConnector"),
    "postgresql": ("agent.connectors.postgres_conn", "PostgresConnector"),
    "mysql":      ("agent.connectors.mysql_conn", "MySQLConnector"),
    "bigquery":   ("agent.connectors.bigquery_conn", "BigQueryConnector"),
    "snowflake":  ("agent.connectors.snowflake_conn", "SnowflakeConnector"),
    "mongodb":    ("agent.connectors.mongodb_conn", "MongoDBConnector"),
    "redis":      ("agent.connectors.redis_conn", "RedisConnector"),
    "kafka":      ("agent.connectors.kafka_conn", "KafkaConnector"),
    "azure_sql":  ("agent.connectors.azure_sql_conn", "AzureSQLConnector"),
    "rest":       ("agent.connectors.rest_conn", "RESTConnector"),
}

_SQL_SOURCES = {"sqlite", "postgres", "postgresql", "mysql", "bigquery",
                "snowflake", "azure_sql"}
_SAFE_TABLE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _connector_records_to_df(data):
    """Best-effort conversion of a connector ToolResult.data payload into a
    DataFrame. Returns None when the shape isn't tabular."""
    import pandas as pd
    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return pd.DataFrame(data["records"])
        if isinstance(data.get("rows"), list) and data.get("columns"):
            return pd.DataFrame(data["rows"], columns=data["columns"])
        if isinstance(data.get("data"), list):
            return pd.DataFrame(data["data"])
    if isinstance(data, list):
        return pd.DataFrame(data)
    return None


class ConnectDataTool(BaseTool):
    name = "connect_data"
    description = (
        "Connect to an EXTERNAL data source and load a query result or table "
        "into the session as `df`. Sources: sqlite, flatfile/csv, postgres, "
        "mysql, bigquery, snowflake, mongodb, redis, kafka, azure_sql, rest. "
        "Provide `source`, a `config` dict of non-secret connection settings "
        "(e.g. path, host, database, user; passwords come from environment "
        "variables), and either a `query` or a `table`. "
        "With neither, lists the available tables for discovery. Read-only."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string",
                       "description": "Connector name, e.g. 'sqlite' or 'postgres'"},
            "config": {"type": "object",
                       "description": "Connection settings passed to the "
                       "connector, e.g. {\"path\": \"data.db\"} or "
                       "{\"host\": ..., \"database\": ..., \"user\": ...}"},
            "query": {"type": "string",
                      "description": "Query to run (SQL for databases; the "
                      "connector's own query form otherwise)"},
            "table": {"type": "string",
                      "description": "Table to load in full (SQL sources only)"},
        },
        "required": ["source"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        import importlib

        source = (args.get("source") or "").lower().strip()
        cfg = {k: v for k, v in (args.get("config") or {}).items()
               if not _SECRET_KEY.search(str(k))}   # secrets come from env vars
        query = (args.get("query") or "").strip()
        table = (args.get("table") or "").strip()

        if source not in _CONNECTOR_CLASSES:
            return ToolOutput(
                success=False, text="",
                error=f"Unknown source '{source}'. Choose from: "
                      f"{', '.join(sorted(_CONNECTOR_CLASSES))}.")
        if not isinstance(cfg, dict):
            return ToolOutput(success=False, text="",
                              error="`config` must be an object of settings.")

        mod_name, cls_name = _CONNECTOR_CLASSES[source]
        try:
            conn = getattr(importlib.import_module(mod_name), cls_name)(**cfg)
        except TypeError as exc:
            return ToolOutput(success=False, text="",
                              error=f"Bad config for {source}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(success=False, text="",
                              error=f"Could not create {source} connector: {exc}")

        res = conn.connect()
        if getattr(res, "status", "error") != "ok":
            return ToolOutput(
                success=False, text="",
                error=f"Connect failed: {getattr(res, 'explanation', '')} "
                      f"{getattr(res, 'error', '')}".strip())

        try:
            # Discovery mode: no query and no table → list tables.
            if not query and not table:
                listing = conn.list_schemas()
                names = listing.data if getattr(listing, "status", "") == "ok" else []
                return ToolOutput(
                    success=True,
                    text=f"Connected to {source}. Available tables/objects "
                         f"({len(names)}):\n" + "\n".join(f"  - {n}" for n in names)
                         + "\n\nCall connect_data again with a `table` or `query` "
                           "to load data.",
                    artifacts={"source": source, "tables": names})

            # Build the query. `table` on a SQL source → SELECT *.
            run_q = query
            if not run_q and table:
                if source not in _SQL_SOURCES:
                    return ToolOutput(
                        success=False, text="",
                        error=f"`table` is only supported for SQL sources; for "
                              f"{source} pass a `query` instead.")
                if not _SAFE_TABLE.match(table):
                    return ToolOutput(success=False, text="",
                                      error=f"Unsafe table name: {table!r}.")
                run_q = f"SELECT * FROM {table}"

            qres = conn.query(run_q)
            if getattr(qres, "status", "error") != "ok":
                return ToolOutput(
                    success=False, text="",
                    error=f"Query failed: {getattr(qres, 'explanation', '')} "
                          f"{getattr(qres, 'error', '')}".strip())

            df = _connector_records_to_df(getattr(qres, "data", None))
            if df is None or df.empty:
                return ToolOutput(
                    success=True,
                    text=f"Query ran on {source} but returned no tabular rows. "
                         f"Connector said: {getattr(qres, 'explanation', '')}",
                    artifacts={"source": source})

            # Wire into the blackboard: this is the frame every other tool uses.
            kernel.namespace["df"] = df
            kernel.namespace["DATA_SOURCE"] = {"source": source, "query": run_q}
            try:
                session.artifacts = getattr(session, "artifacts", {}) or {}
                session.artifacts["data_source"] = source
            except Exception:  # noqa: BLE001
                pass

            preview = df.head(5).to_string()
            return ToolOutput(
                success=True,
                text=f"Loaded {len(df):,} rows × {df.shape[1]} columns from "
                     f"{source} into `df`.\nColumns: {list(df.columns)}\n\n"
                     f"Preview:\n{preview}",
                artifacts={"source": source, "shape": list(df.shape),
                           "columns": list(df.columns)})
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


# ── workspace awareness (Claude-Code-style: map the folder, read on demand) ───

_WS_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", "dist", "build", ".cache",
    ".DS_Store", "site-packages", ".tox", ".eggs",
}
_WS_CATEGORY: dict[str, str] = {}
for _cat, _exts in {
    "data": (".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls",
             ".jsonl", ".orc", ".arrow", ".dta", ".sav"),
    "doc": (".md", ".txt", ".rst", ".pdf", ".docx", ".doc"),
    "code": (".py", ".r", ".sql", ".js", ".ts", ".java", ".cpp", ".c",
             ".go", ".rs", ".sh", ".scala", ".jl"),
    "notebook": (".ipynb",),
    "config": (".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".env"),
    "image": (".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".tiff", ".webp"),
}.items():
    for _e in _exts:
        _WS_CATEGORY[_e] = _cat

_WS_DATA_READ_EXTS = {".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls"}


def _ws_categorise(path: Path) -> str:
    return _WS_CATEGORY.get(path.suffix.lower(), "other")


def _ws_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{int(f)}{u}" if u == "B" else f"{f:.1f}{u}"
        f /= 1024
    return f"{n}B"


def _ws_data_columns(path: Path) -> list[str]:
    """Cheap header/schema read for a tabular file. [] on any failure."""
    import pandas as pd
    try:
        ext = path.suffix.lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return list(pd.read_csv(path, sep=sep, nrows=0).columns)
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return [str(n) for n in pq.ParquetFile(path).schema.names]
        if ext == ".feather":
            return list(pd.read_feather(path).columns)
        if ext in (".xlsx", ".xls"):
            return list(pd.read_excel(path, nrows=0).columns)
    except Exception:  # noqa: BLE001
        return []
    return []


def scan_workspace(root: Path, *, max_files: int = 600, max_depth: int = 6,
                   profile_data: bool = True) -> dict:
    """Build a compact, ignore-aware map of a directory tree.

    Returns structure (a `tree` string), files grouped by category, light
    schemas for a few tabular files, and a `summary` block ready for prompt
    injection. Content is NOT read here (that is `read_file`'s job) — this stays
    cheap so it can run on every workdir change.
    """
    root = Path(root).expanduser().resolve()
    by_cat: dict[str, list[str]] = {}
    data_schemas: list[dict] = []
    tree_lines: list[str] = []
    n_files = n_dirs = 0
    truncated = False

    def _walk(d: Path, depth: int, prefix: str) -> None:
        nonlocal n_files, n_dirs, truncated
        if depth > max_depth or n_files >= max_files:
            truncated = truncated or n_files >= max_files
            return
        try:
            entries = sorted(d.iterdir(),
                             key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir()
                and e.name not in _WS_IGNORE_DIRS
                and not e.name.startswith(".")]
        files = [e for e in entries if e.is_file()
                 and e.name not in _WS_IGNORE_DIRS]
        for e in dirs:
            n_dirs += 1
            if len(tree_lines) < 80:
                tree_lines.append(f"{prefix}{e.name}/")
            _walk(e, depth + 1, prefix + "  ")
        for e in files:
            if n_files >= max_files:
                truncated = True
                return
            n_files += 1
            cat = _ws_categorise(e)
            by_cat.setdefault(cat, []).append(str(e))
            try:
                size = e.stat().st_size
            except OSError:
                size = 0
            if len(tree_lines) < 80:
                tree_lines.append(f"{prefix}{e.name}  ({_ws_size(size)})")
            if (profile_data and cat == "data"
                    and e.suffix.lower() in _WS_DATA_READ_EXTS
                    and len(data_schemas) < 8):
                cols = _ws_data_columns(e)
                if cols:
                    data_schemas.append({
                        "path": str(e), "name": e.name,
                        "n_cols": len(cols), "columns": cols[:25]})

    _walk(root, 0, "")

    counts = {c: len(v) for c, v in sorted(by_cat.items())}
    summary_lines = [
        f"WORKSPACE: {root}",
        f"{n_files} files, {n_dirs} sub-folders"
        + (" (truncated)" if truncated else "")
        + " — by type: "
        + ", ".join(f"{c} {n}" for c, n in counts.items()) if counts else "",
        "",
        "Structure:",
        *tree_lines,
    ]
    if data_schemas:
        summary_lines += ["", "Data files (columns):"]
        for ds in data_schemas:
            summary_lines.append(
                f"  {ds['name']}: {ds['n_cols']} cols — {ds['columns']}")
    summary_lines += [
        "",
        "Use `read_file` to read any file, `explore_directory` to refresh this "
        "map, `connect_data`/register to load a table into df.",
    ]

    return {
        "root": str(root),
        "n_files": n_files, "n_dirs": n_dirs, "truncated": truncated,
        "counts": counts,
        "tree": "\n".join(tree_lines),
        "by_category": by_cat,
        "data_schemas": data_schemas,
        "summary": "\n".join(x for x in summary_lines if x is not None),
    }


def _ws_root(session, kernel) -> Path | None:
    wd = kernel.namespace.get("WORK_DIR") or getattr(session, "work_dir", None)
    if wd:
        try:
            return Path(wd).expanduser().resolve()
        except Exception:  # noqa: BLE001
            return None
    return None


def _ws_rel(root: Path, p: Path) -> str:
    """`p` as a workspace-relative path for messages, absolute if it is outside."""
    try:
        return str(p.relative_to(root)) or "."
    except ValueError:
        return str(p)


def _ws_resolve_within(root: Path, rel: str) -> Path | None:
    """Resolve `rel` against `root` and confine it to the tree. None if it
    escapes (path traversal guard)."""
    try:
        root = root.expanduser().resolve()
        cand = Path(rel).expanduser()
        target = (cand if cand.is_absolute() else root / cand).resolve()
        if target == root or root in target.parents:
            return target
    except Exception:  # noqa: BLE001
        return None
    return None


class ExploreDirectoryTool(BaseTool):
    name = "explore_directory"
    description = (
        "Map the working directory: a recursive, ignore-aware tree of every "
        "file and sub-folder (skips .git/.venv/node_modules/etc.), grouped by "
        "type, with the column names of tabular files. Run this first to learn "
        "what is in the workspace before reading or loading anything. Stores the "
        "map so later turns stay aware of the folder."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subpath": {"type": "string",
                        "description": "Optional sub-folder within the working "
                        "directory to map instead of the whole tree"},
        },
        "required": [],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(
                success=False, text="",
                error="No working directory set. Set one first (the 'Set "
                      "working dir' control), then explore it.")
        target = root
        sub = (args.get("subpath") or "").strip()
        if sub:
            resolved = _ws_resolve_within(root, sub)
            if resolved is None or not resolved.is_dir():
                return ToolOutput(success=False, text="",
                                  error=f"Not a sub-folder of the workspace: {sub!r}")
            target = resolved
        result = scan_workspace(target)
        kernel.namespace["WORKSPACE_MAP"] = result
        return ToolOutput(success=True, text=result["summary"],
                          artifacts={"n_files": result["n_files"],
                                     "counts": result["counts"]})


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read one file from inside the working directory. Type-aware: tabular "
        "files return shape + columns + a sample; notebooks return their "
        "markdown and code-cell headers; PDFs return extracted text; text/code/"
        "config files return their contents. Confined to the working directory "
        "and size-bounded. Use after explore_directory to inspect a specific file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "File path, relative to the working "
                     "directory (or absolute inside it)"},
            "max_chars": {"type": "integer",
                          "description": "Max characters to return (default 8000)"},
        },
        "required": ["path"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set.")
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolOutput(success=False, text="",
                              error="read_file needs a 'path'.")
        target = _ws_resolve_within(root, rel)
        if target is None:
            return ToolOutput(
                success=False, text="",
                error=f"Path escapes the working directory: {rel!r}")
        if not target.is_file():
            return ToolOutput(success=False, text="",
                              error=f"Not a file: {rel!r}")

        max_chars = int(args.get("max_chars", 8000) or 8000)
        ext = target.suffix.lower()
        cat = _ws_categorise(target)
        try:
            size = target.stat().st_size

            if cat == "data" and ext in _WS_DATA_READ_EXTS:
                import pandas as pd
                if ext in (".csv", ".tsv"):
                    sep = "\t" if ext == ".tsv" else ","
                    df = pd.read_csv(target, sep=sep, nrows=200)
                elif ext == ".parquet":
                    df = pd.read_parquet(target)
                elif ext == ".feather":
                    df = pd.read_feather(target)
                else:
                    df = pd.read_excel(target, nrows=200)
                text = (f"{target.name}: {df.shape[0]}+ rows × {df.shape[1]} cols\n"
                        f"Columns: {list(df.columns)}\n\n"
                        f"Sample:\n{df.head(10).to_string()}\n\n"
                        f"(Use connect_data / register to load it into df.)")
                return ToolOutput(success=True, text=text[:max_chars],
                                  artifacts={"kind": "data",
                                             "columns": list(df.columns)})

            if ext == ".ipynb":
                import json as _json
                nb = _json.loads(target.read_text(errors="replace"))
                cells = nb.get("cells", [])
                out = [f"{target.name}: {len(cells)} cells"]
                for i, c in enumerate(cells[:40]):
                    src = "".join(c.get("source", []))[:200]
                    out.append(f"[{i} {c.get('cell_type')}] {src}")
                return ToolOutput(success=True, text="\n".join(out)[:max_chars],
                                  artifacts={"kind": "notebook",
                                             "n_cells": len(cells)})

            if ext == ".pdf":
                from agent.api.pdf_reader import extract as _pdf_extract
                summary = _pdf_extract(target.read_bytes())
                return ToolOutput(success=True, text=(summary.text or "")[:max_chars],
                                  artifacts={"kind": "pdf"})

            if cat == "image":
                return ToolOutput(
                    success=True,
                    text=f"{target.name}: image ({_ws_size(size)}). Binary — not "
                         "shown as text. Use visualize/execute_code to display it.",
                    artifacts={"kind": "image", "size": size})

            # text / code / config / other → read as text, bounded.
            raw = target.read_text(errors="replace")
            note = "" if len(raw) <= max_chars else \
                f"\n\n[truncated — {len(raw)} chars total, showing {max_chars}]"
            return ToolOutput(success=True, text=raw[:max_chars] + note,
                              artifacts={"kind": cat, "size": size})
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(success=False, text="",
                              error=f"Could not read {rel!r}: {exc}")


# ── file search + editing (with a suggest / write mode toggle) ────────────────

# Default write mode when the request/session does not set one. Off (suggest)
# is the safe default: edits are proposed as diffs, nothing touches disk until
# the user enables writes (the web-UI toggle, or DSAGENT_ALLOW_WRITES=1).
_ALLOW_WRITES_DEFAULT = os.environ.get("DSAGENT_ALLOW_WRITES", "0") != "0"

# The three write modes:
#   suggest  propose a diff and stop. Nothing is recorded, nothing is written.
#   ask      stage the change and wait for the user to approve it. This is what
#            the web-UI WRITES toggle turns on, so enabling writes means "you
#            may propose changes to disk", not "you may change my files".
#   write    apply immediately. For automation and tests only; a person driving
#            the agent through the UI never selects it.
_WRITE_MODES = ("suggest", "ask", "write")


def _ws_write_mode(kernel) -> str:
    """One of _WRITE_MODES. Per-request FILE_WRITE_MODE in the kernel wins;
    otherwise the DSAGENT_ALLOW_WRITES default applies. The default for enabled
    writes is 'ask', never 'write', so a stray env var cannot silently grant
    unattended write access."""
    m = kernel.namespace.get("FILE_WRITE_MODE") if kernel is not None else None
    if m in _WRITE_MODES:
        return m
    return "ask" if _ALLOW_WRITES_DEFAULT else "suggest"


# ── pending disk changes ─────────────────────────────────────────────────────
#
# A tool that wants to change the workspace stages the change here and reports
# that it is waiting. The user approves or rejects each one, and only then does
# it land. Creating a directory is part of the record, so "this also creates
# outputs/" is something the user reads before it happens rather than discovers
# afterwards.

def _pending_list(kernel) -> list[dict]:
    """The session's queue of staged changes, created on first use."""
    if kernel is None:
        return []
    q = getattr(kernel, "pending_changes", None)
    if not isinstance(q, list):
        q = []
        kernel.pending_changes = q
    return q


def _file_hash(path: Path) -> str | None:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _stage_change(kernel, *, kind: str, rel: str, path: Path, summary: str,
                  diff: str = "", content: str | None = None,
                  make_dirs: list[str] | None = None,
                  extra: dict | None = None) -> dict:
    """Record a disk change for approval and return it. Touches nothing."""
    import secrets
    import time
    q = _pending_list(kernel)
    rec = {
        "id": f"chg_{secrets.token_hex(4)}",
        "kind": kind,
        "rel": rel,
        "path": str(path),
        "summary": summary,
        "diff": diff[:8000],
        "content": content,
        "base_hash": _file_hash(path),
        "root": str(path.parent.resolve()),
        "make_dirs": [str(d) for d in (make_dirs or [])],
        "extra": extra or {},
        "staged_at": time.time(),
    }
    q.append(rec)
    return rec


def _hold_for_approval(kernel, *, kind: str, rel: str, path: Path, verb: str,
                       content: str | None = None, diff: str = "",
                       make_dirs: list[str] | None = None,
                       extra: dict | None = None) -> ToolOutput | None:
    """Stage the change and return the tool's reply, or None when the caller
    should go ahead and write (mode 'write'). Callers handle 'suggest' first."""
    if _ws_write_mode(kernel) != "ask":
        return None
    dirs = [str(d) for d in (make_dirs or [])]
    summary = verb + (f", creating {len(dirs)} new folder"
                      f"{'' if len(dirs) == 1 else 's'}" if dirs else "")
    rec = _stage_change(kernel, kind=kind, rel=rel, path=path, summary=summary,
                        diff=diff, content=content, make_dirs=dirs, extra=extra)
    body = (f"\n\n{diff[:3000]}" if diff else "")
    return ToolOutput(
        success=True,
        text=f"AWAITING YOUR APPROVAL: {summary}. Nothing has been written. "
             f"Approve or reject it in the Pending changes panel "
             f"(id {rec['id']}).{body}",
        artifacts={"pending": rec["id"], "path": str(path), "op": verb})


def pending_changes(kernel) -> list[dict]:
    """What is waiting for approval, without the full file contents."""
    return [{k: v for k, v in rec.items() if k not in {"content", "binary"}}
            for rec in _pending_list(kernel)]


def _apply_one(rec: dict, *, session, kernel) -> None:
    """Perform one approved change. Raises on failure."""
    root = _ws_root(session, kernel)
    path = Path(rec["path"])
    if root is None or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Approved destination escapes the workspace")
    if _file_hash(path) != rec.get("base_hash"):
        raise ValueError("File changed since the proposed diff; regenerate the proposal")
    for d in rec.get("make_dirs") or []:
        Path(d).mkdir(parents=True, exist_ok=True)
    kind = rec.get("kind")
    if kind == "file":
        import base64
        import tempfile
        data = (base64.b64decode(rec["binary"]) if "binary" in rec
                else (rec.get("content") or "").encode())
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
        return
    if kind == "tool":
        # Re-run the tool that staged this, with writes applied. The kernel is
        # the same one, so the model and data it needs are still there.
        name = (rec.get("extra") or {}).get("tool", "")
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            raise ValueError(f"unknown tool in staged change: {name!r}")
        saved = kernel.namespace.get("FILE_WRITE_MODE")
        kernel.namespace["FILE_WRITE_MODE"] = "write"
        try:
            out = tool.execute((rec.get("extra") or {}).get("args") or {},
                               session=session, kernel=kernel, llm_client=None)
        finally:
            kernel.namespace["FILE_WRITE_MODE"] = saved
        if not out.success:
            raise RuntimeError(out.error or f"{name} failed")
        return
    raise ValueError(f"unknown change kind: {kind!r}")


def apply_pending(kernel, ids: list[str] | None = None, *,
                  session=None) -> dict[str, Any]:
    """Apply approved changes. `ids` selects a subset; None applies all."""
    wanted = set(ids) if ids is not None else None
    applied: list[dict] = []
    failed: list[dict] = []
    kept: list[dict] = []
    for rec in list(_pending_list(kernel)):
        if wanted is not None and rec["id"] not in wanted:
            kept.append(rec)
            continue
        try:
            _apply_one(rec, session=session, kernel=kernel)
            applied.append({"id": rec["id"], "rel": rec["rel"],
                            "summary": rec["summary"]})
        except Exception as exc:  # noqa: BLE001
            kept.append(rec)
            failed.append({"id": rec["id"], "rel": rec["rel"],
                           "error": str(exc)})
    kernel.pending_changes = kept
    return {"applied": applied, "failed": failed, "pending": len(kept)}


def discard_pending(kernel, ids: list[str] | None = None) -> dict[str, Any]:
    """Drop staged changes without touching disk."""
    wanted = set(ids) if ids is not None else None
    kept = [r for r in _pending_list(kernel)
            if wanted is not None and r["id"] not in wanted]
    dropped = len(_pending_list(kernel)) - len(kept)
    kernel.pending_changes = kept
    return {"discarded": dropped, "pending": len(kept)}


def _unified_diff(old: str, new: str, path: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"))


def _ws_iter_files(root: Path, max_files: int = 4000):
    """Yield files under root, ignore-aware (same skips as scan_workspace)."""
    stack = [root]
    count = 0
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name in _WS_IGNORE_DIRS or e.name.startswith("."):
                    continue
                stack.append(e)
            elif e.is_file() and e.name not in _WS_IGNORE_DIRS:
                count += 1
                if count > max_files:
                    return
                yield e


def _is_probably_text(path: Path, size_limit: int = 2_000_000) -> bool:
    try:
        if path.stat().st_size > size_limit:
            return False
        with open(path, "rb") as f:
            return b"\x00" not in f.read(2048)
    except OSError:
        return False


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = (
        "Search the working directory for a text pattern across files, like "
        "grep. Returns file:line: matched line. Ignore-aware (skips "
        ".git/.venv/node_modules/etc. and binaries). Use to find where "
        "something is defined or used before reading or editing it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regex to find"},
            "regex": {"type": "boolean",
                      "description": "Treat pattern as a regex (default true)"},
            "case_sensitive": {"type": "boolean",
                               "description": "Case-sensitive (default false)"},
            "glob": {"type": "string",
                     "description": "Only search files whose name matches this "
                     "glob, e.g. '*.py'"},
            "max_results": {"type": "integer",
                            "description": "Max matches to return (default 50)"},
        },
        "required": ["pattern"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        import fnmatch
        import re as _re

        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set.")
        pattern = args.get("pattern") or ""
        if not pattern:
            return ToolOutput(success=False, text="",
                              error="search_files needs a 'pattern'.")
        use_regex = args.get("regex", True)
        flags = 0 if args.get("case_sensitive", False) else _re.IGNORECASE
        try:
            rx = _re.compile(pattern if use_regex else _re.escape(pattern), flags)
        except _re.error as exc:
            return ToolOutput(success=False, text="",
                              error=f"Bad regex: {exc}")
        glob = args.get("glob") or ""
        max_results = int(args.get("max_results", 50) or 50)

        hits: list[str] = []
        files_scanned = 0
        for f in _ws_iter_files(root):
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            if not _is_probably_text(f):
                continue
            files_scanned += 1
            try:
                rel = f.relative_to(root)
            except ValueError:
                rel = f
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                            if len(hits) >= max_results:
                                break
            except OSError:
                continue
            if len(hits) >= max_results:
                break

        if not hits:
            return ToolOutput(
                success=True,
                text=f"No matches for {pattern!r} in {files_scanned} files.",
                artifacts={"matches": 0})
        head = f"{len(hits)} match(es) for {pattern!r}"
        head += " (capped)" if len(hits) >= max_results else ""
        return ToolOutput(success=True, text=head + ":\n" + "\n".join(hits),
                          artifacts={"matches": len(hits)})


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Create or overwrite a file in the working directory. When file writes "
        "are ENABLED it writes to disk; when DISABLED (suggest mode) it returns "
        "the diff without changing anything. Pass overwrite=true to replace an "
        "existing file. Confined to the working directory."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "File path, relative to the working directory"},
            "content": {"type": "string", "description": "Full file contents"},
            "overwrite": {"type": "boolean",
                          "description": "Replace the file if it already exists "
                          "(default false)"},
            "create_dirs": {"type": "boolean",
                            "description": "Create missing parent folders "
                            "(default false). Set this ONLY when the user asked "
                            "for a new folder. Otherwise a missing folder means "
                            "the path is wrong."},
        },
        "required": ["path", "content"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set.")
        rel = (args.get("path") or "").strip()
        if not rel:
            return ToolOutput(success=False, text="",
                              error="write_file needs a 'path'.")
        content = args.get("content")
        if content is None:
            return ToolOutput(success=False, text="",
                              error="write_file needs 'content'.")
        target = _ws_resolve_within(root, rel)
        if target is None:
            return ToolOutput(success=False, text="",
                              error=f"Path escapes the working directory: {rel!r}")
        exists = target.is_file()
        if exists and not args.get("overwrite", False):
            return ToolOutput(
                success=False, text="",
                error=f"{rel} already exists — pass overwrite=true to replace it "
                      "(or use edit_file for a targeted change).")

        # A missing parent folder is almost always a wrong path, not a request
        # for a new folder. Creating it silently is how the agent ends up
        # building a directory tree nobody asked for.
        make_dirs: list[str] = []
        if not target.parent.is_dir():
            if not args.get("create_dirs", False):
                return ToolOutput(
                    success=False, text="",
                    error=f"Folder does not exist: {_ws_rel(root, target.parent)}"
                          f". Check the path. Only if the user asked for a new "
                          "folder, call again with create_dirs=true.")
            make_dirs = [str(target.parent)]

        old = target.read_text(errors="replace") if exists else ""
        diff = _unified_diff(old, content, rel)
        mode = _ws_write_mode(kernel)
        verb = "overwrite" if exists else "create"
        if mode == "suggest":
            body = diff or content[:2000]
            return ToolOutput(
                success=True,
                text=f"PROPOSED {verb} of {rel} (writes are OFF — nothing "
                     f"written). Enable writes to apply.\n\n{body[:4000]}",
                artifacts={"proposed": True, "path": str(target), "op": verb})
        held = _hold_for_approval(kernel, kind="file", rel=rel, path=target,
                                  verb=f"{verb} {rel}", content=content,
                                  diff=diff, make_dirs=make_dirs)
        if held is not None:
            return held
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except OSError as exc:
            return ToolOutput(success=False, text="",
                              error=f"Write failed: {exc}")
        return ToolOutput(
            success=True,
            text=f"Wrote {len(content):,} chars to {rel} ({verb}d).",
            artifacts={"written": True, "path": str(target), "op": verb})


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Make a targeted edit to a file by exact string replacement. `old_string` "
        "must appear EXACTLY once (include enough surrounding context to be "
        "unique) unless replace_all=true. When file writes are ENABLED the edit "
        "is applied; when DISABLED (suggest mode) the diff is returned without "
        "changing anything. Read the file first if unsure. Confined to the "
        "working directory."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "File to edit, relative to the working directory"},
            "old_string": {"type": "string",
                           "description": "Exact text to replace (with context)"},
            "new_string": {"type": "string",
                           "description": "Replacement text"},
            "replace_all": {"type": "boolean",
                            "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set.")
        rel = (args.get("path") or "").strip()
        target = _ws_resolve_within(root, rel) if rel else None
        if target is None:
            return ToolOutput(success=False, text="",
                              error=f"Path escapes the working directory: {rel!r}")
        if not target.is_file():
            return ToolOutput(success=False, text="", error=f"Not a file: {rel!r}")
        old_string = args.get("old_string")
        new_string = args.get("new_string", "")
        if not old_string:
            return ToolOutput(success=False, text="",
                              error="edit_file needs a non-empty 'old_string'.")
        if old_string == new_string:
            return ToolOutput(success=False, text="",
                              error="old_string and new_string are identical.")

        try:
            content = target.read_text(errors="replace")
        except OSError as exc:
            return ToolOutput(success=False, text="", error=f"Read failed: {exc}")

        occurrences = content.count(old_string)
        replace_all = bool(args.get("replace_all", False))
        if occurrences == 0:
            return ToolOutput(
                success=False, text="",
                error=f"old_string not found in {rel}. Read the file and copy the "
                      "exact text (whitespace included).")
        if occurrences > 1 and not replace_all:
            return ToolOutput(
                success=False, text="",
                error=f"old_string matches {occurrences} places in {rel}. Add "
                      "more surrounding context to make it unique, or set "
                      "replace_all=true.")

        new_content = (content.replace(old_string, new_string)
                       if replace_all
                       else content.replace(old_string, new_string, 1))
        diff = _unified_diff(content, new_content, rel)
        n = occurrences if replace_all else 1
        mode = _ws_write_mode(kernel)
        if mode == "suggest":
            return ToolOutput(
                success=True,
                text=f"PROPOSED edit to {rel} ({n} replacement(s); writes are "
                     f"OFF — nothing written). Enable writes to apply.\n\n"
                     f"{diff[:4000]}",
                artifacts={"proposed": True, "path": str(target),
                           "replacements": n})
        held = _hold_for_approval(
            kernel, kind="file", rel=rel, path=target,
            verb=f"edit {rel} ({n} replacement(s))",
            content=new_content, diff=diff)
        if held is not None:
            return held
        try:
            target.write_text(new_content)
        except OSError as exc:
            return ToolOutput(success=False, text="",
                              error=f"Write failed: {exc}")
        return ToolOutput(
            success=True,
            text=f"Edited {rel} ({n} replacement(s)).\n\n{diff[:2000]}",
            artifacts={"written": True, "path": str(target), "replacements": n})


# ── SQL authoring + execution (databases and warehouses, incl. Snowflake) ─────

_SQL_GEN_PROMPT = (
    "You are an expert SQL author. Write a single {dialect} SQL SELECT query that "
    "answers the question. Use only the tables and columns in the schema. Return "
    "the SQL only, no prose, no code fences.\n\n"
    "SCHEMA:\n{schema}\n\nQUESTION: {question}\n\nSQL:")

_SECRET_KEY = __import__("re").compile(
    r"pass|secret|token|private_key|credential|api_key", __import__("re").IGNORECASE)

_SQL_FORBIDDEN = __import__("re").compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|GRANT|"
    r"REVOKE)\b", __import__("re").IGNORECASE)


def _connector_schema_context(conn, max_tables: int = 12) -> str:
    """Compact 'table(col, col, ...)' listing from a connected source, for
    grounding SQL generation. Best-effort."""
    try:
        listing = conn.list_schemas()
        tables = listing.data if getattr(listing, "status", "") == "ok" else []
    except Exception:  # noqa: BLE001
        tables = []
    lines: list[str] = []
    for t in list(tables)[:max_tables]:
        try:
            desc = conn.describe_table(t)
            cols = [c.get("name", "?") for c in desc.data.get("columns", [])] \
                if getattr(desc, "status", "") == "ok" else []
            lines.append(f"{t}({', '.join(cols[:40])})" if cols else str(t))
        except Exception:  # noqa: BLE001
            lines.append(str(t))
    return "\n".join(lines) if lines else "(schema unavailable)"


class SqlQueryTool(BaseTool):
    name = "sql_query"
    description = (
        "Author and run SQL against a connected database or warehouse "
        "(Snowflake, Postgres, BigQuery, MySQL, SQLite, Azure SQL). Give a "
        "natural-language `question` to generate SQL from the live schema, or "
        "an explicit `sql` to run. Read-only: destructive statements are "
        "refused. Optionally `save_to` a .sql file. Returns the SQL and the "
        "result rows."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string",
                       "description": "Connector: snowflake, postgres, bigquery, "
                       "mysql, sqlite, azure_sql"},
            "config": {"type": "object",
                       "description": "Non-secret connection settings (account, "
                       "warehouse, database, schema, path). Passwords and keys "
                       "come from environment variables such as "
                       "SNOWFLAKE_PASSWORD, never from here."},
            "question": {"type": "string",
                         "description": "Natural-language question to turn into SQL"},
            "sql": {"type": "string",
                    "description": "Explicit SQL to run instead of generating it"},
            "save_to": {"type": "string",
                        "description": "Optional .sql path in the working "
                        "directory to save the query to"},
            "max_rows": {"type": "integer",
                         "description": "Max result rows to show (default 50)"},
        },
        "required": ["source"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        import importlib

        source = (args.get("source") or "").lower().strip()
        if source not in _SQL_SOURCES:
            return ToolOutput(
                success=False, text="",
                error=f"sql_query supports SQL sources only: "
                      f"{', '.join(sorted(_SQL_SOURCES))}.")
        # Secrets never travel through model-written arguments: they would sit
        # in the prompt, the transcript and the logs. Connectors read them from
        # environment variables (SNOWFLAKE_PASSWORD, POSTGRES_PASSWORD, ...).
        cfg = {k: v for k, v in (args.get("config") or {}).items()
               if not _SECRET_KEY.search(str(k))}
        question = (args.get("question") or "").strip()
        sql = (args.get("sql") or "").strip()
        if not question and not sql:
            return ToolOutput(success=False, text="",
                              error="Provide a 'question' or an explicit 'sql'.")

        mod_name, cls_name = _CONNECTOR_CLASSES[source]
        try:
            conn = getattr(importlib.import_module(mod_name), cls_name)(**cfg)
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(success=False, text="",
                              error=f"Could not create {source} connector: {exc}")
        res = conn.connect()
        if getattr(res, "status", "error") != "ok":
            return ToolOutput(success=False, text="",
                              error=f"Connect failed: {getattr(res, 'explanation', '')}")

        try:
            generated = False
            if not sql:
                schema = _connector_schema_context(conn)
                dialect = "Snowflake" if source == "snowflake" else source
                prompt = _SQL_GEN_PROMPT.format(
                    dialect=dialect, schema=schema, question=question)
                sql = (llm_client.generate_code(prompt) if llm_client else "").strip()
                sql = sql.strip("`").removeprefix("sql").strip().rstrip(";")
                generated = True
                if not sql:
                    return ToolOutput(success=False, text="",
                                      error="SQL generation returned nothing "
                                            "(LLM offline?). Pass explicit 'sql'.")
            if _SQL_FORBIDDEN.search(sql):
                return ToolOutput(
                    success=False, text=f"SQL:\n{sql}",
                    error="Refusing to run: the SQL contains a destructive "
                          "statement. This tool is read-only.")

            saved_note = ""
            save_to = (args.get("save_to") or "").strip()
            if save_to:
                saved_note = self._save_sql(save_to, sql, session, kernel)

            qres = conn.query(sql)
            if getattr(qres, "status", "error") != "ok":
                return ToolOutput(
                    success=(save_to != ""), text=f"SQL:\n{sql}\n{saved_note}",
                    error=f"Query failed: {getattr(qres, 'explanation', '')} "
                          f"{getattr(qres, 'error', '')}".strip())

            df = _connector_records_to_df(getattr(qres, "data", None))
            max_rows = int(args.get("max_rows", 50) or 50)
            if df is not None and not df.empty:
                body = (f"{len(df)} row(s) × {df.shape[1]} col(s):\n"
                        f"{df.head(max_rows).to_string()}")
            else:
                body = getattr(qres, "explanation", "Query ran; no rows.")
            head = ("Generated SQL" if generated else "SQL") + f":\n{sql}\n"
            return ToolOutput(
                success=True, text=head + saved_note + "\n" + body,
                artifacts={"sql": sql, "generated": generated,
                           "rows": int(df.shape[0]) if df is not None else 0})
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _save_sql(save_to: str, sql: str, session, kernel) -> str:
        root = _ws_root(session, kernel)
        if root is None:
            return "\n(Not saved: no working directory set.)"
        target = _ws_resolve_within(root, save_to)
        if target is None:
            return f"\n(Not saved: path escapes the working directory: {save_to!r}.)"
        if _ws_write_mode(kernel) == "suggest":
            return (f"\n(Would save the SQL to {save_to} — writes are OFF, "
                    "nothing written.)")
        if not target.parent.is_dir():
            return (f"\n(Not saved: folder does not exist: "
                    f"{_ws_rel(root, target.parent)}.)")
        held = _hold_for_approval(kernel, kind="file", rel=save_to, path=target,
                                  verb=f"save SQL to {save_to}",
                                  content=sql + "\n")
        if held is not None:
            return (f"\n(Awaiting your approval to save the SQL to {save_to}; "
                    "nothing written.)")
        try:
            target.write_text(sql + "\n")
            return f"\n(Saved SQL to {save_to}.)"
        except OSError as exc:
            return f"\n(Save failed: {exc}.)"


# ── SQL authoring from the loaded data (no warehouse connection needed) ───────

_SQL_DIALECTS = ("snowflake", "postgres", "bigquery", "mysql", "sqlite", "tsql")

_SQL_WRITE_PROMPT = """\
Write one {dialect} SQL query that answers the request. Use only the tables and
columns below, with their exact names. Prefer clear CTEs over nested
subqueries, and comment any step that is not obvious with a -- line.
Return the SQL only, no prose and no code fences.

TABLES:
{schema}

WHAT THE COLUMNS MEAN:
{notes}

REQUEST: {question}
"""


def _sql_type(dtype) -> str:
    kind = getattr(dtype, "kind", "O")
    return {"i": "INTEGER", "u": "INTEGER", "f": "DOUBLE", "b": "BOOLEAN",
            "M": "TIMESTAMP"}.get(kind, "VARCHAR")


def _sql_tables(session, kernel, primary: str = "") -> dict[str, Any]:
    """The session's frames under table names: the data file's stem, so the SQL
    reads like it targets the real table (df -> medical_claims_dataset)."""
    import re as _re
    ns = kernel.namespace
    paths = list(getattr(session, "data_paths", []) or [])
    tables: dict[str, Any] = {}
    for i in range(1, 10):
        var = "df" if i == 1 else f"df{i}"
        frame = ns.get(var)
        if frame is None or not hasattr(frame, "columns"):
            continue
        if i == 1 and primary:
            name = primary
        elif i - 1 < len(paths):
            name = Path(paths[i - 1]).stem
        else:
            name = var
        name = _re.sub(r"\W+", "_", name).strip("_").lower() or var
        tables[name] = frame
    return tables


class WriteSqlTool(BaseTool):
    name = "write_sql"
    description = (
        "Write a SQL query (Snowflake by default; also postgres, bigquery, mysql, "
        "sqlite, tsql) against the data loaded in this session, without a "
        "warehouse connection. The loaded files become tables named after the "
        "file. The query is test-run on the local data with SQLite and the "
        "result rows are shown; syntax SQLite does not support is reported as "
        "untested, never as passed. Optionally save_to a .sql file (the user "
        "approves the write). To run SQL on a live database, use sql_query."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "What the query should compute"},
            "dialect": {"type": "string", "enum": list(_SQL_DIALECTS),
                        "description": "Target SQL dialect (default snowflake)"},
            "table": {"type": "string",
                      "description": "Table name to use for the main dataset, "
                                     "e.g. the warehouse table's real name"},
            "save_to": {"type": "string",
                        "description": "Optional .sql file in the working directory"},
        },
        "required": ["question"],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        question = str(args.get("question") or "").strip()
        dialect = str(args.get("dialect") or "snowflake").lower()
        if dialect not in _SQL_DIALECTS:
            return ToolOutput(False, "", error=f"dialect must be one of {_SQL_DIALECTS}.")
        tables = _sql_tables(session, kernel, str(args.get("table") or ""))
        if not tables:
            return ToolOutput(False, "", error="No data is loaded, so there is no "
                              "schema to write SQL against.")
        if llm_client is None:
            return ToolOutput(False, "", error="No language model is available.")
        schema = "\n".join(
            f"{name}(" + ", ".join(f"{c} {_sql_type(frame[c].dtype)}"
                                   for c in list(frame.columns)[:60]) + ")"
            for name, frame in tables.items())
        spec = kernel.namespace.get("TASK_SPEC")
        brief = spec.get("brief", {}) if isinstance(spec, dict) else {}
        notes = "\n".join(f"- {n}" for n in (brief.get("data_notes") or [])[:40]) or "(none)"
        try:
            raw = llm_client._generate(
                _SQL_WRITE_PROMPT.format(dialect=dialect, schema=schema, notes=notes,
                                         question=question),
                temperature=0.0, max_tokens=4000)
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(False, "", error=f"SQL generation failed: {exc}")
        sql = _strip_sql(raw)
        if not sql:
            return ToolOutput(False, "", error="The model returned no SQL.")
        if _SQL_FORBIDDEN.search(sql):
            return ToolOutput(False, f"SQL:\n{sql}", error="Refusing: the SQL modifies "
                              "data. Only read queries are written by this tool.")
        tested, result = _test_sql_locally(sql, tables)
        lines = [f"{dialect.capitalize()} SQL:", sql, ""]
        if tested:
            lines.append(f"Test run on the local data (SQLite): {result}")
        else:
            lines.append(f"Not tested locally: SQLite could not run it ({result}). "
                         f"{dialect.capitalize()}-specific syntax cannot be checked "
                         "without the warehouse.")
        save_to = str(args.get("save_to") or "").strip()
        if save_to:
            if not save_to.endswith(".sql"):
                save_to += ".sql"
            lines.append(SqlQueryTool._save_sql(save_to, sql, session, kernel).strip())
        return ToolOutput(True, "\n".join(lines),
                          artifacts={"sql": sql, "dialect": dialect, "tested": tested})


def _strip_sql(raw: str) -> str:
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", raw or "", flags=_re.DOTALL).strip()
    fence = _re.search(r"```(?:sql)?\s*(.*?)```", text, flags=_re.DOTALL | _re.IGNORECASE)
    if fence:
        text = fence.group(1)
    return text.strip().rstrip(";").strip()


def _test_sql_locally(sql: str, tables: dict[str, Any],
                      max_rows: int = 20) -> tuple[bool, str]:
    """Run the SQL against the frames in an in-memory SQLite database.
    Returns (ran, summary of rows or the error)."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        for name, frame in tables.items():
            frame.head(500_000).to_sql(name, conn, index=False)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows)
        if not cols:
            return True, "ran, no rows returned."
        import pandas as _pd
        shown = _pd.DataFrame(rows, columns=cols).to_string(index=False)
        return True, f"first {len(rows)} row(s):\n{shown}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc).splitlines()[0][:200]
    finally:
        conn.close()


# ── model deployment packaging + CI/CD (generate artifacts, do not push) ──────

class DeployModelTool(BaseTool):
    name = "deploy_model"
    description = (
        "Package the trained model as a deployable service: serialise it and "
        "scaffold a FastAPI prediction API, a Dockerfile, and requirements in "
        "the working directory. It does NOT build or push images or deploy to a "
        "live system — that runs from your CI/CD with your own credentials. Run "
        "after a model has been trained (looks for `best_model` in the session)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "output_dir": {"type": "string",
                           "description": "Folder to scaffold into (default 'deploy')"},
            "model_var": {"type": "string",
                          "description": "Kernel variable holding the model "
                          "(default 'best_model')"},
        },
        "required": [],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set — set one to scaffold "
                                    "the deployment into it.")
        model_var = (args.get("model_var") or "best_model").strip()
        model = kernel.namespace.get(model_var) or kernel.namespace.get("best_model")
        if model is None:
            return ToolOutput(success=False, text="",
                              error="No trained model in the session. Run "
                                    "train_model first.")
        out_rel = (args.get("output_dir") or "deploy").strip()
        out = _ws_resolve_within(root, out_rel)
        if out is None:
            return ToolOutput(success=False, text="",
                              error=f"Path escapes the working directory: {out_rel!r}")

        files = ["model.joblib", "app.py", "Dockerfile", "requirements.txt"]
        run_hint = (f"\n\nTo run locally:\n  cd {out_rel} && pip install -r "
                    "requirements.txt && uvicorn app:app --port 8080\n"
                    "To containerise: docker build -t my-model "
                    f"{out_rel}\n(Build/push/deploy run from your CI/CD or shell — "
                    "this tool does not execute them.)")
        if _ws_write_mode(kernel) == "suggest":
            return ToolOutput(
                success=True,
                text=f"PROPOSED deployment scaffold in {out_rel}/ (writes are OFF "
                     f"— nothing written): {', '.join(files)}."
                     f"\nEnable writes to generate them." + run_hint,
                artifacts={"proposed": True, "output_dir": str(out)})

        try:
            import base64
            import tempfile

            import joblib

            from agent.pipeline.deploy_packager.fastapi_builder import FastAPIBuilder
            with tempfile.TemporaryDirectory(prefix="ds-deploy-") as tmp:
                scratch = Path(tmp)
                joblib.dump(model, scratch / "model.joblib")
                FastAPIBuilder().build(model_uri="model.joblib", output_dir=tmp)
                for item in scratch.iterdir():
                    destination = out / item.name
                    rec = _stage_change(kernel, kind="file", rel=str(destination.relative_to(root)),
                        path=destination, summary=f"Publish deployment file {item.name}",
                        make_dirs=[str(out)] if not out.exists() else [])
                    rec["binary"] = base64.b64encode(item.read_bytes()).decode()
            return ToolOutput(True, "Deployment files staged as immutable snapshots. "
                              "Approve publication in Pending changes." + run_hint,
                              artifacts={"staged": True, "files": files})
        except Exception as exc:
            return ToolOutput(False, "", error=f"Deployment packaging failed: {exc}")


_GITHUB_CI = """\
name: ci-cd
on:
  push:
    branches: [main]
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff pytest
      - run: ruff check .
      - run: pytest -q
  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    # Deploy only from main; wire REGISTRY/credentials as repo secrets first.
    if: github.ref == 'refs/heads/main'
    environment: production        # add a required reviewer for manual approval
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t {image}:${{{{ github.sha }}}} {deploy_dir}
      # - name: Log in to registry
      #   run: echo "${{{{ secrets.REGISTRY_TOKEN }}}}" | docker login -u ${{{{ secrets.REGISTRY_USER }}}} --password-stdin {registry}
      # - name: Push image
      #   run: docker push {image}:${{{{ github.sha }}}}
      # - name: Deploy
      #   run: echo "add your kubectl/helm/cloud deploy step here"
"""

_GITLAB_CI = """\
stages: [test, build, deploy]
test:
  stage: test
  image: python:3.11
  script:
    - pip install ruff pytest
    - ruff check .
    - pytest -q
build:
  stage: build
  image: docker:latest
  services: [docker:dind]
  script:
    - docker build -t {image}:$CI_COMMIT_SHA {deploy_dir}
  only: [main]
deploy:
  stage: deploy
  when: manual        # manual approval before a real deploy
  script:
    - echo "wire your registry login, docker push, and cloud deploy here"
  only: [main]
"""


class ScaffoldCICDTool(BaseTool):
    name = "scaffold_cicd"
    description = (
        "Generate a CI/CD workflow that lints, tests, builds the Docker image "
        "from the deploy folder, and has a gated deploy step. Supports GitHub "
        "Actions and GitLab CI. Written into the working directory; wire your "
        "registry and cluster credentials as CI secrets to enable the real "
        "deploy. The push/deploy steps are commented or manual by default."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "provider": {"type": "string",
                         "description": "'github' (default) or 'gitlab'"},
            "path": {"type": "string",
                     "description": "Where to write the workflow (defaults per "
                     "provider)"},
            "image": {"type": "string",
                      "description": "Docker image name (default 'my-model')"},
            "deploy_dir": {"type": "string",
                           "description": "Folder with the Dockerfile (default "
                           "'deploy')"},
            "registry": {"type": "string",
                         "description": "Registry host for the push step comment"},
        },
        "required": [],
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        root = _ws_root(session, kernel)
        if root is None:
            return ToolOutput(success=False, text="",
                              error="No working directory set.")
        provider = (args.get("provider") or "github").lower().strip()
        image = (args.get("image") or "my-model").strip()
        deploy_dir = (args.get("deploy_dir") or "deploy").strip()
        registry = (args.get("registry") or "registry.example.com").strip()
        if provider not in ("github", "gitlab"):
            return ToolOutput(success=False, text="",
                              error="provider must be 'github' or 'gitlab'.")
        if provider == "github":
            content = _GITHUB_CI.format(image=image, deploy_dir=deploy_dir,
                                        registry=registry)
            default_path = ".github/workflows/ci.yml"
        else:
            content = _GITLAB_CI.format(image=image, deploy_dir=deploy_dir)
            default_path = ".gitlab-ci.yml"
        rel = (args.get("path") or default_path).strip()
        target = _ws_resolve_within(root, rel)
        if target is None:
            return ToolOutput(success=False, text="",
                              error=f"Path escapes the working directory: {rel!r}")

        if _ws_write_mode(kernel) == "suggest":
            return ToolOutput(
                success=True,
                text=f"PROPOSED {provider} CI/CD at {rel} (writes are OFF — "
                     f"nothing written). Enable writes to create it.\n\n{content}",
                artifacts={"proposed": True, "path": str(target)})
        held = _hold_for_approval(
            kernel, kind="file", rel=rel, path=target,
            verb=f"create {provider} CI/CD at {rel}", content=content,
            diff=_unified_diff("", content, rel),
            make_dirs=[] if target.parent.is_dir() else [str(target.parent)])
        if held is not None:
            return held
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except OSError as exc:
            return ToolOutput(success=False, text="",
                              error=f"Write failed: {exc}")
        return ToolOutput(
            success=True,
            text=f"Wrote {provider} CI/CD workflow to {rel}. The push and deploy "
                 "steps are commented/manual — add your registry and cluster "
                 "secrets to enable them.",
            artifacts={"written": True, "path": str(target)})


class DeepEdaTool(BaseTool):
    """Six-stage deep EDA: quality radar → univariate panorama → bivariate
    type-dispatched analysis → clustered association matrix → multivariate
    structure (PCA + parallel coords) → missingness map → LLM narrative.
    Stores EDA_CTX dict in kernel for downstream use."""

    name = "deep_eda"
    description = (
        "Run a comprehensive six-stage EDA that produces rich visualisations and "
        "an LLM-generated narrative discussion. Covers: quality radar, outlier map, "
        "univariate panorama (box plots + normality), bivariate type-dispatched "
        "analysis (num×num / cat×num / cat×cat with correct stats), clustered "
        "correlation & Cramér's V heatmaps, PCA + parallel coordinates, missingness "
        "matrix, and an LLM synthesis paragraph. Works without a target: the "
        "target-vs-feature stage and the narrative then are skipped."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": (
                    "'quick' skips PCA/t-SNE and bivariate grid; "
                    "'standard' (default) runs all stages except t-SNE; "
                    "'deep' adds t-SNE and full pairplot."
                ),
            },
            "sample_rows": {
                "type": "integer",
                "description": "Max rows to use for heavy plots (default 5000)",
            },
        },
    }

    # ── stage helpers — each returns (text_summary, list[base64_fig]) ────────

    @staticmethod
    def _stage05_findings(kernel, target: str, task_type: str) -> tuple[str, list]:
        """Deterministic 12-check heuristic scan. Emits EDA_CTX['findings'] as a
        severity-ranked list that Stage 6 narrative uses as primary input."""
        code = textwrap.dedent(f"""
            import numpy as np
            from scipy import stats as _scipy_stats

            _tgt = {repr(target)}
            _task_type = {repr(task_type)}
            _findings = []  # list of dicts: severity/check/detail/action

            def _find(severity, check, detail, action=""):
                _findings.append({{'severity': severity, 'check': check,
                                   'detail': detail, 'action': action}})

            _num = df.select_dtypes('number')
            _cats = df.select_dtypes('object')
            _n = len(df)

            # ── Check 1: Placeholder / sentinel value detection ──────────────
            _SENTINELS = [-999, -9999, 9999, 99999, -1, 999, -998, -1.0]
            for _c in _num.columns:
                for _sv in _SENTINELS:
                    _cnt = (df[_c] == _sv).sum()
                    if _cnt >= 3 and _cnt / _n < 0.15:
                        _find('warning', 'placeholder_value',
                              f"Column '{{_c}}': {{_cnt}} rows = {{_sv}} "
                              f"({{_cnt/_n:.1%}}) — likely encoded null",
                              f"Replace {{_sv}} in '{{_c}}' with NaN before modelling")

            # ── Check 2: Zero-inflation ──────────────────────────────────────
            for _c in _num.columns:
                _zero_rate = (df[_c] == 0).mean()
                if _zero_rate > 0.5 and df[_c].std() > 0:
                    _find('info', 'zero_inflation',
                          f"'{{_c}}': {{_zero_rate:.1%}} zeros",
                          f"Consider log1p({{_c}}) or binary indicator for zero-ness")

            # ── Check 3: Extreme skew → log transform recommendation ─────────
            for _c in _num.columns:
                if df[_c].min() >= 0:
                    _sk = abs(float(df[_c].skew()))
                    if _sk > 3:
                        _find('info', 'extreme_skew',
                              f"'{{_c}}': |skew|={{_sk:.2f}}",
                              f"Apply log1p('{{_c}}') for linear/distance-based models")

            # ── Check 4: Near-constant columns ───────────────────────────────
            for _c in df.columns:
                _top_freq = df[_c].value_counts(normalize=True).iloc[0]
                if _top_freq > 0.99:
                    _find('warning', 'near_constant',
                          f"'{{_c}}': dominant value covers {{_top_freq:.1%}} of rows",
                          f"Drop '{{_c}}' — provides near-zero information")

            # ── Check 5: Bimodality coefficient per numeric column ───────────
            for _c in _num.columns[:30]:
                _col = df[_c].dropna()
                if len(_col) < 20: continue
                _sk = float(_col.skew())
                _ku = float(_col.kurtosis())
                _n_c = len(_col)
                # Bimodality coefficient BC = (skew²+1)/(kurtosis + 3(n-1)²/((n-2)(n-3)))
                _denom = _ku + 3 * (_n_c-1)**2 / max((_n_c-2)*(_n_c-3), 1)
                _bc = (_sk**2 + 1) / _denom if _denom != 0 else 0
                if _bc > 0.555:
                    _find('info', 'bimodal',
                          f"'{{_c}}': bimodality coefficient BC={{_bc:.3f}}>0.555 — "
                          f"possible hidden subpopulation",
                          f"Investigate '{{_c}}' for hidden grouping; "
                          f"consider creating a binary split feature")

            # ── Check 6: Structural break (sudden mean shift) ────────────────
            for _c in _num.columns[:15]:
                _col = df[_c].fillna(df[_c].median()).values
                _mid = len(_col) // 2
                if _mid < 10: continue
                _m1, _m2 = np.mean(_col[:_mid]), np.mean(_col[_mid:])
                _std_pool = np.std(_col) + 1e-9
                if abs(_m1 - _m2) / _std_pool > 1.5:
                    _find('warning', 'structural_break',
                          f"'{{_c}}': mean shifts from {{_m1:.3g}} (first half) "
                          f"to {{_m2:.3g}} (second half) — z={{abs(_m1-_m2)/_std_pool:.2f}}",
                          f"Check data ordering; consider row-index as a feature "
                          f"or split dataset at the break point")

            # ── Check 7: Duplicate-value columns (correlation > 0.99) ────────
            _num_filled = _num.fillna(0)
            _corr_m = _num_filled.corr().abs()
            for _i, _ci in enumerate(_num_filled.columns):
                for _j, _cj in enumerate(_num_filled.columns):
                    if _j <= _i: continue
                    if _corr_m.loc[_ci, _cj] > 0.99:
                        _find('warning', 'redundant_pair',
                              f"'{{_ci}}' ↔ '{{_cj}}': |r|={{_corr_m.loc[_ci, _cj]:.4f}} "
                              f"— near-duplicate columns",
                              f"Drop one of '{{_ci}}' / '{{_cj}}' to reduce multicollinearity")

            # ── Check 8: Class imbalance severity ────────────────────────────
            if _tgt and _tgt in df.columns and ('classif' in _task_type or 'binary' in _task_type):
                _vc_tgt = df[_tgt].value_counts()
                _imb = _vc_tgt.min() / _vc_tgt.max()
                if _imb < 0.1:
                    _find('critical', 'severe_class_imbalance',
                          f"Target '{{_tgt}}': imbalance ratio={{_imb:.3f}} "
                          f"({{_vc_tgt.min()}} minority vs {{_vc_tgt.max()}} majority)",
                          "Use PR-AUC metric; apply SMOTE or class_weight='balanced'; "
                          "consider cost-sensitive learning")
                elif _imb < 0.3:
                    _find('warning', 'class_imbalance',
                          f"Target '{{_tgt}}': imbalance ratio={{_imb:.3f}}",
                          "Prefer PR-AUC over accuracy; use class_weight='balanced'")

            # ── Check 9: Target leakage (single-feature AUC > 0.85) ─────────
            if _tgt and _tgt in df.columns and ('classif' in _task_type or 'binary' in _task_type):
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import LabelEncoder
                _y_raw = df[_tgt]
                if _y_raw.dtype == object:
                    _y_leak = pd.Series(LabelEncoder().fit_transform(_y_raw.astype(str)))
                else:
                    _y_leak = _y_raw.fillna(_y_raw.median())
                if _y_leak.nunique() == 2:
                    for _c in [c for c in _num.columns if c != _tgt]:
                        try:
                            _v = df[_c].fillna(df[_c].median()).values
                            _a = roc_auc_score(_y_leak, _v)
                            _a = max(_a, 1 - _a)
                            if _a > 0.85:
                                _find('critical', 'leakage_candidate',
                                      f"'{{_c}}': single-feature AUC={{_a:.4f}} > 0.85 — "
                                      f"suspiciously predictive",
                                      f"Verify '{{_c}}' is causally available at prediction "
                                      f"time and not derived from the target or post-event data")
                        except Exception:
                            pass

            # ── Check 10: Missing > 30% (high-risk for imputation) ───────────
            _high_miss = (df.isna().mean() > 0.30)
            for _c in _high_miss[_high_miss].index:
                _find('warning', 'high_missingness',
                      f"'{{_c}}': {{df[_c].isna().mean():.1%}} missing",
                      f"Consider dropping '{{_c}}' or using iterative imputation; "
                      f"create missingness indicator feature")

            # ── Check 11: Highly skewed target (regression) ──────────────────
            if _tgt and _tgt in df.columns and 'regress' in _task_type:
                _tgt_col = df[_tgt].dropna()
                if _tgt_col.min() >= 0:
                    _tgt_sk = abs(float(_tgt_col.skew()))
                    if _tgt_sk > 2:
                        _find('warning', 'skewed_target',
                              f"Target '{{_tgt}}': |skew|={{_tgt_sk:.2f}}",
                              "Apply log1p transform to target; use RMSE on log scale "
                              "(RMSLE); or switch to quantile regression")

            # ── Check 12: Single category dominates > 95% ────────────────────
            for _c in _cats.columns:
                _top = df[_c].value_counts(normalize=True).iloc[0]
                if _top > 0.95:
                    _find('warning', 'dominant_category',
                          f"'{{_c}}': top category = {{_top:.1%}} of rows",
                          f"Drop '{{_c}}' — near-zero variance categorical")

            # Sort: critical first, then warning, then info
            _sev_order = {{'critical': 0, 'warning': 1, 'info': 2}}
            _findings.sort(key=lambda x: _sev_order.get(x['severity'], 9))
            EDA_CTX['findings'] = _findings

            # Print severity-ranked summary
            print("=== Findings — Severity Ranked ===")
            for _f in _findings:
                _icon = {{'critical': '🚨', 'warning': '⚠', 'info': 'ℹ'}}.get(
                    _f['severity'], '?')
                print(f"  [{{_f['severity'].upper()}}] {{_icon}} {{_f['check']}}")
                print(f"    Detail : {{_f['detail']}}")
                print(f"    Action : {{_f['action']}}")
                print()
            if not _findings:
                print("  No significant findings detected.")
        """).strip()
        out = kernel.execute(code)
        return out.as_text(5000), out.figures

    @staticmethod
    def _stage0_quality_radar(kernel) -> tuple[str, list]:
        """Quality radar + Isolation Forest outlier map."""
        code = textwrap.dedent("""
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyArrowPatch

            # ── 5-dimension quality scores ───────────────────────────────────
            _n = len(df)
            _qd = {}
            _qd['Completeness'] = 1 - df.isna().mean().mean()
            _num = df.select_dtypes('number')
            _qd['Validity'] = float(np.isfinite(_num.values).mean()) if len(_num.columns) else 1.0
            _qd['Uniqueness'] = 1 - df.duplicated().mean()
            _const = sum(df[c].nunique() <= 1 for c in df.columns)
            _qd['Consistency'] = 1 - _const / max(len(df.columns), 1)
            _skew_mean = min(_num.skew().abs().mean() / 5, 1.0) if len(_num.columns) else 0
            _qd['Distribution'] = round(1 - _skew_mean, 3)

            print("=== Data Quality Scores ===")
            for _k, _v in _qd.items():
                _bar = '█' * int(_v*20) + '░' * (20 - int(_v*20))
                print(f"  {_k:<14} {_bar} {_v:.2f}")
            _overall = sum(_qd.values()) / len(_qd)
            print(f"  Overall Q-score: {_overall:.3f}")
            EDA_CTX['quality'] = {**_qd, 'overall': _overall}

            # ── Radar chart ──────────────────────────────────────────────────
            _cats = list(_qd.keys())
            _vals = list(_qd.values())
            _N = len(_cats)
            _angles = [n / float(_N) * 2 * np.pi for n in range(_N)]
            _angles += _angles[:1]
            _vals_plot = _vals + _vals[:1]
            fig0, ax0 = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            ax0.set_theta_offset(np.pi / 2); ax0.set_theta_direction(-1)
            ax0.set_xticks(_angles[:-1])
            ax0.set_xticklabels(_cats, size=10)
            ax0.plot(_angles, _vals_plot, 'o-', linewidth=2, color='#1f77b4')
            ax0.fill(_angles, _vals_plot, alpha=0.25, color='#1f77b4')
            ax0.set_ylim(0, 1)
            ax0.set_title(f'Data Quality Radar  (Q={_overall:.2f})', pad=20, fontsize=12)
            plt.tight_layout(); plt.show()

            # ── Isolation Forest + Mahalanobis outlier detection ────────────
            from sklearn.ensemble import IsolationForest
            from sklearn.covariance import MinCovDet
            from scipy.stats import chi2 as _chi2_dist
            _Xif = _num.fillna(_num.median())
            if len(_Xif.columns) >= 2:
                # Isolation Forest
                _if = IsolationForest(contamination=0.05, random_state=0, n_jobs=-1)
                _outlier_flag_if = _if.fit_predict(_Xif) == -1
                _n_if = _outlier_flag_if.sum()
                print(f"\\nIsolation Forest: {_n_if} multivariate outliers "
                      f"({_n_if/_n:.1%} of rows)")

                # Mahalanobis with MinCovDet (robust covariance)
                _outlier_flag_mah = np.zeros(_n, dtype=bool)
                try:
                    _ncols_mah = min(len(_Xif.columns), 20)
                    _Xmah = _Xif.iloc[:, :_ncols_mah].values
                    _mcd = MinCovDet(random_state=0, support_fraction=0.75).fit(_Xmah)
                    _D2 = _mcd.mahalanobis(_Xmah)
                    _thresh_mah = _chi2_dist.ppf(0.99, df=_ncols_mah)
                    _outlier_flag_mah = _D2 > _thresh_mah
                    _n_mah = _outlier_flag_mah.sum()
                    print(f"Mahalanobis (MinCovDet): {_n_mah} outliers "
                          f"({_n_mah/_n:.1%}) — threshold χ²(p={_ncols_mah}, 0.99)"
                          f"={_thresh_mah:.2f}")
                    EDA_CTX['mahalanobis_n_outliers'] = int(_n_mah)
                except Exception as _mah_err:
                    print(f"Mahalanobis skipped: {_mah_err}")

                # Union of both methods
                _outlier_flag = _outlier_flag_if | _outlier_flag_mah
                _n_out = _outlier_flag.sum()
                print(f"Combined outlier flag (IF ∪ Mahalanobis): {_n_out} rows "
                      f"({_n_out/_n:.1%})")
                EDA_CTX['n_outliers'] = int(_n_out)
                EDA_CTX['outlier_flag'] = _outlier_flag

                # Parallel coordinates of top-6 features coloured by outlier flag
                from sklearn.preprocessing import MinMaxScaler
                _top6 = _Xif.columns[:6].tolist()
                _Xscaled = pd.DataFrame(
                    MinMaxScaler().fit_transform(_Xif[_top6]),
                    columns=_top6
                )
                _Xscaled['_is_outlier'] = _outlier_flag
                fig1, ax1 = plt.subplots(figsize=(12, 4))
                for _, _row in _Xscaled[~_Xscaled['_is_outlier']].sample(
                        min(300, (~_outlier_flag).sum()), random_state=0).iterrows():
                    ax1.plot(range(len(_top6)), _row[_top6], color='steelblue',
                             alpha=0.15, linewidth=0.8)
                for _, _row in _Xscaled[_Xscaled['_is_outlier']].iterrows():
                    ax1.plot(range(len(_top6)), _row[_top6], color='#d62728',
                             alpha=0.7, linewidth=1.2)
                ax1.set_xticks(range(len(_top6))); ax1.set_xticklabels(_top6, rotation=20)
                ax1.set_title(f'Parallel Coordinates — Outliers (red, n={_n_out}) vs Inliers (blue)')
                from matplotlib.lines import Line2D
                ax1.legend(handles=[
                    Line2D([0],[0], color='#d62728', label=f'Outlier (n={_n_out})'),
                    Line2D([0],[0], color='steelblue', alpha=0.5, label='Inlier'),
                ], loc='upper right', fontsize=8)
                plt.tight_layout(); plt.show()
            else:
                EDA_CTX['n_outliers'] = 0
        """).strip()
        out = kernel.execute(code)
        return out.as_text(2000), out.figures

    @staticmethod
    def _stage1_univariate(kernel, sample_rows: int) -> tuple[str, list]:
        """Combined box plots, normality stats, categorical bars."""
        code = textwrap.dedent(f"""
            import numpy as np
            from scipy import stats as _scipy_stats
            import matplotlib.pyplot as plt

            _sr = {sample_rows}
            _df_s = df.sample(min(_sr, len(df)), random_state=0) if len(df) > _sr else df

            _num = _df_s.select_dtypes('number')
            _cats = _df_s.select_dtypes('object')

            # ── Normality stats table ────────────────────────────────────────
            print("=== Univariate Normality Stats ===")
            _norm_rows = []
            for _c in _num.columns[:30]:
                _col = _num[_c].dropna()
                _sk = float(_col.skew())
                _ku = float(_col.kurtosis())
                _sw_p = None
                if 5 <= len(_col) <= 5000:
                    try:
                        _, _sw_p = _scipy_stats.shapiro(_col.sample(min(len(_col), 5000),
                                                                     random_state=0))
                    except Exception:
                        pass
                _norm_rows.append({{
                    'column': _c, 'skew': round(_sk, 2), 'kurtosis': round(_ku, 2),
                    'shapiro_p': round(_sw_p, 4) if _sw_p is not None else 'N/A',
                    'normal': 'yes' if (_sw_p is not None and _sw_p > 0.05) else 'no'
                }})
            _norm_df = pd.DataFrame(_norm_rows)
            print(_norm_df.to_string(index=False))
            EDA_CTX['normality'] = _norm_rows

            # ── Bimodality coefficient table ─────────────────────────────────
            _bimo = []
            for _c in _num.columns[:30]:
                _col = _num[_c].dropna()
                if len(_col) < 20: continue
                _sk2 = float(_col.skew()) ** 2
                _ku = float(_col.kurtosis())
                _n_c = len(_col)
                _denom = _ku + 3 * (_n_c-1)**2 / max((_n_c-2)*(_n_c-3), 1)
                _bc = (_sk2 + 1) / _denom if _denom != 0 else 0
                _bimo.append({'column': _c, 'BC': round(_bc, 3),
                              'bimodal': 'yes' if _bc > 0.555 else 'no'})
            _bimo_df = pd.DataFrame(_bimo)
            if len(_bimo_df):
                print("\\nBimodality Coefficient (BC > 0.555 = bimodal):")
                print(_bimo_df[_bimo_df['bimodal']=='yes'].to_string(index=False)
                      if (_bimo_df['bimodal']=='yes').any()
                      else "  No bimodal columns detected")

            # ── Structured transformation recommendations ─────────────────────
            _rec_transforms = {
                'log1p': [],
                'winsorize': [],
                'target_encode': [],
                'drop_constant': [],
                'drop_high_corr': [],
            }
            for _c in _num.columns:
                if df[_c].min() >= 0 and abs(float(df[_c].skew())) > 2:
                    _rec_transforms['log1p'].append(_c)
                _q1, _q3 = df[_c].quantile(0.25), df[_c].quantile(0.75)
                _iqr = _q3 - _q1
                _out_rate = ((df[_c] < _q1 - 3*_iqr) | (df[_c] > _q3 + 3*_iqr)).mean()
                if _out_rate > 0.05:
                    _rec_transforms['winsorize'].append(_c)
            for _c in df.select_dtypes('object').columns:
                if df[_c].nunique() > 20:
                    _rec_transforms['target_encode'].append(_c)
                if df[_c].nunique() <= 1:
                    _rec_transforms['drop_constant'].append(_c)
            EDA_CTX['recommended_transforms'] = _rec_transforms
            print("\\nRecommended Transforms:")
            for _op, _cols in _rec_transforms.items():
                if _cols:
                    print(f"  {{_op}}: {{_cols[:5]}}"
                          + (f" (+{{len(_cols)-5}} more)" if len(_cols) > 5 else ""))

            # ── Combined box plot for all numerics ──────────────────────────
            if len(_num.columns) > 0:
                _cols_bp = _num.columns[:20].tolist()
                _ncols_bp = min(4, len(_cols_bp))
                _nrows_bp = (len(_cols_bp) + _ncols_bp - 1) // _ncols_bp
                fig2, axes2 = plt.subplots(_nrows_bp, _ncols_bp,
                                            figsize=(_ncols_bp*4, _nrows_bp*3))
                _axs2 = axes2.flat if hasattr(axes2, 'flat') else [axes2]
                for _ax, _c in zip(_axs2, _cols_bp):
                    _ax.boxplot(_num[_c].dropna(), patch_artist=True,
                                boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                                medianprops=dict(color='#d62728', linewidth=2),
                                whiskerprops=dict(color='#1f77b4'),
                                flierprops=dict(marker='o', color='gray',
                                                markersize=3, alpha=0.5))
                    _ax.set_title(_c, fontsize=8)
                for _ax in list(_axs2)[len(_cols_bp):]: _ax.set_visible(False)
                plt.suptitle('Numeric Feature Box Plots', fontsize=12, y=1.01)
                plt.tight_layout(); plt.show()

            # ── Categorical horizontal bar charts ────────────────────────────
            if len(_cats.columns) > 0:
                _cat_cols_show = [c for c in _cats.columns if _df_s[c].nunique() <= 50][:6]
                if _cat_cols_show:
                    _nc2 = min(2, len(_cat_cols_show))
                    _nr2 = (len(_cat_cols_show) + _nc2 - 1) // _nc2
                    fig3, axes3 = plt.subplots(_nr2, _nc2,
                                               figsize=(_nc2*7, _nr2*3.5))
                    _axs3 = axes3.flat if hasattr(axes3, 'flat') else [axes3]
                    for _ax, _cc in zip(_axs3, _cat_cols_show):
                        _vc = _df_s[_cc].value_counts().head(10)
                        _vc.plot(kind='barh', ax=_ax, color='#ff7f0e', edgecolor='white')
                        _ax.set_title(f'{{_cc}} (top-10)', fontsize=9)
                        _ax.invert_yaxis()
                    for _ax in list(_axs3)[len(_cat_cols_show):]: _ax.set_visible(False)
                    plt.suptitle('Categorical Feature Distributions', fontsize=12, y=1.01)
                    plt.tight_layout(); plt.show()
        """).strip()
        out = kernel.execute(code)
        return out.as_text(3000), out.figures

    @staticmethod
    def _stage2_bivariate(kernel, target: str, sample_rows: int) -> tuple[str, list]:
        """Task-type-dispatched bivariate analysis.

        Branches:
          timeseries_forecasting → ADF+KPSS 4-cell matrix, STL, ACF/PACF
          anomaly_detection      → GMM-BIC, reconstruction error baseline
          classification/binary  → violin+Kruskal-Wallis, Chi-Square, leakage AUC→findings
          regression             → KDE jointplots + Spearman ρ
        """
        code = textwrap.dedent(f"""
            import numpy as np
            from scipy import stats as _scipy_stats
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import LabelEncoder

            _tgt = {repr(target)}
            _sr = {sample_rows}
            _task_spec = TASK_SPEC if 'TASK_SPEC' in dir() else {{}}
            _task_type = _task_spec.get('task_type', 'auto')
            _df_b = df.sample(min(_sr, len(df)), random_state=0) if len(df) > _sr else df
            _findings_ref = EDA_CTX.get('findings', []) if 'EDA_CTX' in dir() else []

            print(f"=== Bivariate Analysis — task_type={{_task_type}} ===")

            # ════════════════════════════════════════════════════════════════
            # BRANCH A: TIME-SERIES FORECASTING
            # ════════════════════════════════════════════════════════════════
            if 'timeseries' in _task_type or 'time_series' in _task_type or 'forecast' in _task_type:
                print("  Branch: timeseries_forecasting")
                if _tgt not in df.columns:
                    print(f"  Target '{{_tgt}}' not in df — skipping")
                else:
                    _ts = df[_tgt].dropna()

                    # ── ADF + KPSS with 4-cell decision matrix ────────────────
                    from statsmodels.tsa.stattools import adfuller, kpss
                    try:
                        _adf_stat, _adf_p, *_ = adfuller(_ts, autolag='AIC')
                        _kpss_stat, _kpss_p, *_ = kpss(_ts, regression='c', nlags='auto')
                        _adf_rej = _adf_p < 0.05
                        _kpss_rej = _kpss_p < 0.05
                        _decision_map = {{
                            (True, False):  'STATIONARY — ADF rejects unit root, KPSS non-reject',
                            (False, True):  'NON-STATIONARY — ADF non-reject, KPSS rejects stationarity',
                            (True, True):   'TREND-STATIONARY — both reject; remove trend',
                            (False, False): 'DIFFERENCE-STATIONARY — neither rejects; consider I(1)',
                        }}
                        _decision = _decision_map[(_adf_rej, _kpss_rej)]
                        print(f"  ADF:  stat={{_adf_stat:.4f}}, p={{_adf_p:.4f}} "
                              f"({{'reject' if _adf_rej else 'non-reject'}} H0)")
                        print(f"  KPSS: stat={{_kpss_stat:.4f}}, p≈{{_kpss_p:.4f}} "
                              f"({{'reject' if _kpss_rej else 'non-reject'}} H0)")
                        print(f"  Decision: {{_decision}}")
                        EDA_CTX['stationarity'] = {{'adf_p': _adf_p, 'kpss_p': _kpss_p,
                                                    'decision': _decision}}
                        if not _adf_rej:
                            _findings_ref.append({{'severity': 'warning', 'check': 'non_stationary',
                                'detail': f"Target '{{_tgt}}' is non-stationary (ADF p={{_adf_p:.3f}})",
                                'action': 'Apply differencing or log-diff before modelling'}})
                    except Exception as _e:
                        print(f"  ADF/KPSS failed: {{_e}}")

                    # ── STL decomposition ─────────────────────────────────────
                    try:
                        from statsmodels.tsa.seasonal import STL
                        _period = max(2, min(12, len(_ts)//4))
                        _stl = STL(_ts.values, period=_period, robust=True).fit()
                        _trend_var = np.var(_stl.trend)
                        _seas_var  = np.var(_stl.seasonal)
                        _resid_var = np.var(_stl.resid)
                        _total_var = _trend_var + _seas_var + _resid_var + 1e-12
                        print(f"  STL: trend={{_trend_var/_total_var:.1%}}, "
                              f"seasonal={{_seas_var/_total_var:.1%}}, "
                              f"residual={{_resid_var/_total_var:.1%}}")
                        EDA_CTX['stl'] = {{'trend_pct': _trend_var/_total_var,
                                           'seasonal_pct': _seas_var/_total_var,
                                           'residual_pct': _resid_var/_total_var}}
                        fig_stl, axes_stl = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                        for _ax, _comp, _lbl in zip(axes_stl,
                            [_ts.values, _stl.trend, _stl.seasonal, _stl.resid],
                            ['Observed', 'Trend', 'Seasonal', 'Residual']):
                            _ax.plot(_comp, linewidth=0.9)
                            _ax.set_ylabel(_lbl, fontsize=9)
                        plt.suptitle(f'STL Decomposition  ({{_tgt}}, period={{_period}})')
                        plt.tight_layout(); plt.show()
                    except Exception as _e:
                        print(f"  STL skipped: {{_e}}")

                    # ── ACF / PACF ────────────────────────────────────────────
                    try:
                        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                        fig_acf, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 4))
                        plot_acf(_ts, lags=min(40, len(_ts)//3), ax=ax_acf, alpha=0.05)
                        plot_pacf(_ts, lags=min(40, len(_ts)//3), ax=ax_pacf,
                                  alpha=0.05, method='ywm')
                        ax_acf.set_title('ACF')
                        ax_pacf.set_title('PACF')
                        plt.suptitle(f'Autocorrelation — {{_tgt}}')
                        plt.tight_layout(); plt.show()
                    except Exception as _e:
                        print(f"  ACF/PACF skipped: {{_e}}")

            # ════════════════════════════════════════════════════════════════
            # BRANCH B: ANOMALY DETECTION
            # ════════════════════════════════════════════════════════════════
            elif 'anomaly' in _task_type or 'outlier' in _task_type:
                print("  Branch: anomaly_detection")
                _num_b = _df_b.select_dtypes('number').fillna(0)
                if len(_num_b.columns) >= 2:
                    # ── GMM-BIC model selection ───────────────────────────────
                    from sklearn.mixture import GaussianMixture
                    from sklearn.preprocessing import StandardScaler
                    _Xgmm = StandardScaler().fit_transform(_num_b.iloc[:, :10])
                    _bics = []
                    for _k in range(1, min(8, len(_df_b)//10 + 1)):
                        try:
                            _g = GaussianMixture(n_components=_k, random_state=0,
                                                 covariance_type='full').fit(_Xgmm)
                            _bics.append((_k, _g.bic(_Xgmm)))
                        except Exception:
                            pass
                    if _bics:
                        _best_k = min(_bics, key=lambda x: x[1])[0]
                        print(f"  GMM-BIC: optimal k={{_best_k}} components")
                        EDA_CTX['gmm_k'] = _best_k
                        _ks = [b[0] for b in _bics]; _bv = [b[1] for b in _bics]
                        fig_bic, ax_bic = plt.subplots(figsize=(7, 3.5))
                        ax_bic.plot(_ks, _bv, 'o-', color='steelblue')
                        ax_bic.axvline(_best_k, color='#d62728', linestyle='--',
                                       label=f'Best k={{_best_k}}')
                        ax_bic.set_xlabel('# GMM components')
                        ax_bic.set_ylabel('BIC')
                        ax_bic.set_title('GMM-BIC model selection')
                        ax_bic.legend(fontsize=8)
                        plt.tight_layout(); plt.show()

                    # ── Autoencoder reconstruction error baseline ─────────────
                    try:
                        from sklearn.neural_network import MLPRegressor
                        _ae = MLPRegressor(hidden_layer_sizes=(min(16, _Xgmm.shape[1]//2),),
                                           max_iter=200, random_state=0)
                        _ae.fit(_Xgmm, _Xgmm)
                        _recon = _ae.predict(_Xgmm)
                        _recon_err = np.mean((_Xgmm - _recon)**2, axis=1)
                        _thresh_ae = np.percentile(_recon_err, 95)
                        _n_anom = (_recon_err > _thresh_ae).sum()
                        print(f"  Autoencoder reconstruction error: "
                              f"{{_n_anom}} samples above 95th-percentile threshold "
                              f"(thresh={{_thresh_ae:.4f}})")
                        EDA_CTX['autoencoder_n_anomalies'] = int(_n_anom)
                        fig_ae, ax_ae = plt.subplots(figsize=(9, 3.5))
                        ax_ae.hist(_recon_err, bins=50, color='steelblue', edgecolor='white')
                        ax_ae.axvline(_thresh_ae, color='#d62728', linestyle='--',
                                      label=f'95th pct ({{_thresh_ae:.4f}})')
                        ax_ae.set_xlabel('Reconstruction error (MSE per sample)')
                        ax_ae.set_title('Autoencoder Anomaly Score Distribution')
                        ax_ae.legend(fontsize=8)
                        plt.tight_layout(); plt.show()
                    except Exception as _ae_e:
                        print(f"  Autoencoder baseline skipped: {{_ae_e}}")

            # ════════════════════════════════════════════════════════════════
            # BRANCH C: CLASSIFICATION / BINARY (default)
            # ════════════════════════════════════════════════════════════════
            else:
                _y_raw = _df_b[_tgt]
                _is_cls = _y_raw.dtype == object or _y_raw.nunique() < 20
                if _y_raw.dtype == object:
                    _le_b = LabelEncoder()
                    _y = pd.Series(_le_b.fit_transform(_y_raw.astype(str)), name=_tgt)
                else:
                    _y = _y_raw.fillna(_y_raw.median())
                _n_cls = _y.nunique()

                _num_feats = [c for c in _df_b.select_dtypes('number').columns if c != _tgt]

                # ── NUM × TARGET: violin + Kruskal-Wallis ────────────────────
                if _num_feats and _is_cls and _n_cls <= 10:
                    _corr_rank = {{c: abs(_df_b[c].fillna(0).corr(_y)) for c in _num_feats}}
                    _top_num = sorted(_corr_rank, key=_corr_rank.get, reverse=True)[:8]
                    _classes = sorted(_y.unique())
                    _nc_v = min(4, len(_top_num))
                    _nr_v = (len(_top_num) + _nc_v - 1) // _nc_v
                    fig_v, axes_v = plt.subplots(_nr_v, _nc_v,
                                                  figsize=(_nc_v*4.5, _nr_v*3.5))
                    _axs_v = axes_v.flat if hasattr(axes_v, 'flat') else [axes_v]
                    _palette = plt.cm.Set2.colors
                    print("  Numeric × Target — Kruskal-Wallis test:")
                    for _ax, _feat in zip(_axs_v, _top_num):
                        _groups = [_df_b.loc[_y == _cls, _feat].dropna().values
                                   for _cls in _classes]
                        _groups = [g for g in _groups if len(g) > 1]
                        _stat, _p = (None, None)
                        if len(_groups) >= 2:
                            try:
                                _stat, _p = _scipy_stats.kruskal(*_groups)
                            except Exception:
                                pass
                        _p_str = f'KW p={{_p:.3f}}' if _p is not None else ''
                        print(f"    {{_feat}}: {{_p_str}}")
                        _vp_data = [_df_b.loc[_y == _cls, _feat].dropna().values
                                    for _cls in _classes]
                        try:
                            _vp = _ax.violinplot(_vp_data, positions=range(len(_classes)),
                                                 showmedians=True, showextrema=False)
                            for _i, _pc in enumerate(_vp['bodies']):
                                _pc.set_facecolor(_palette[_i % len(_palette)])
                                _pc.set_alpha(0.8)
                        except Exception:
                            for _j, _cls in enumerate(_classes):
                                _vals = _df_b.loc[_y == _cls, _feat].dropna().values
                                _ax.boxplot([_vals], positions=[_j], patch_artist=True,
                                            boxprops=dict(facecolor=_palette[_j % len(_palette)],
                                                          alpha=0.7))
                        _ax.set_xticks(range(len(_classes)))
                        _ax.set_xticklabels([str(c) for c in _classes], fontsize=7)
                        _ax.set_title(f'{{_feat}}\\n{{_p_str}}', fontsize=8)
                    for _ax in list(_axs_v)[len(_top_num):]: _ax.set_visible(False)
                    plt.suptitle(f'Numeric × Target  ({{_tgt}})', fontsize=12, y=1.01)
                    plt.tight_layout(); plt.show()

                    # ── Leakage AUC check → write critical findings ───────────
                    if _n_cls == 2:
                        from sklearn.metrics import roc_auc_score
                        print("\\n  Single-feature leakage AUC check:")
                        for _c in _num_feats[:40]:
                            try:
                                _v = _df_b[_c].fillna(_df_b[_c].median()).values
                                _a = roc_auc_score(_y, _v)
                                _a = max(_a, 1 - _a)
                                if _a > 0.85:
                                    print(f"    🚨 LEAKAGE CANDIDATE: '{{_c}}' AUC={{_a:.4f}} > 0.85")
                                    _findings_ref.append({{
                                        'severity': 'critical',
                                        'check': 'leakage_candidate',
                                        'detail': f"'{{_c}}': single-feature AUC={{_a:.4f}} > 0.85",
                                        'action': (f"Verify '{{_c}}' is not derived from the target "
                                                   f"or unavailable at prediction time")
                                    }})
                            except Exception:
                                pass
                        # Update EDA_CTX findings
                        _sev_order = {{'critical': 0, 'warning': 1, 'info': 2}}
                        _findings_ref.sort(key=lambda x: _sev_order.get(x['severity'], 9))
                        EDA_CTX['findings'] = _findings_ref

                # ── NUM × NUM: jointplot for regression ──────────────────────
                elif _num_feats and not _is_cls:
                    import itertools
                    _pairs = list(itertools.combinations(_num_feats[:8], 2))
                    _corr_pairs = [(a, b, abs(_df_b[a].fillna(0).corr(_df_b[b].fillna(0))))
                                   for a, b in _pairs]
                    _corr_pairs.sort(key=lambda x: -x[2])
                    print("  Numeric × Numeric — Spearman ρ for top correlated pairs:")
                    for _a, _b, _r in _corr_pairs[:5]:
                        _sp_r, _sp_p = _scipy_stats.spearmanr(
                            _df_b[_a].fillna(0), _df_b[_b].fillna(0))
                        print(f"    {{_a}} ↔ {{_b}}: ρ={{_sp_r:.3f}}, p={{_sp_p:.4f}}")
                    for _a, _b, _r in _corr_pairs[:3]:
                        try:
                            import seaborn as _sns
                            _sns.jointplot(data=_df_b.fillna(0), x=_a, y=_b,
                                           kind='kde', height=5, ratio=3)
                            plt.suptitle(f'Joint Distribution: {{_a}} vs {{_b}}', y=1.01)
                            plt.tight_layout(); plt.show()
                        except Exception as _e:
                            print(f"  jointplot skipped ({{_e}})")

                # ── CAT × TARGET: Chi-Square + stacked bar ───────────────────
                _cat_feats = [c for c in _df_b.select_dtypes('object').columns
                              if c != _tgt and _df_b[c].nunique() <= 30]
                if _cat_feats and _is_cls:
                    from scipy.stats import chi2_contingency
                    print("\\n  Categorical × Target — Chi-Square tests:")
                    _chi_results = []
                    for _cc in _cat_feats:
                        try:
                            _ct = pd.crosstab(_df_b[_cc], _y)
                            _chi2, _p_chi, _, _ = chi2_contingency(_ct, correction=False)
                            _n_chi = _ct.sum().sum()
                            _k_chi = min(_ct.shape) - 1
                            _cv = np.sqrt(_chi2 / (_n_chi * _k_chi)) if _k_chi > 0 else 0
                            _chi_results.append((_cc, _chi2, _p_chi, _cv))
                            print(f"    {{_cc}}: χ²={{_chi2:.2f}}, p={{_p_chi:.4f}}, "
                                  f"Cramér V={{_cv:.3f}}")
                        except Exception:
                            pass
                    EDA_CTX['chi2_results'] = _chi_results
                    _chi_results.sort(key=lambda x: -x[3])
                    for _cc, _chi2, _p_chi, _cv in _chi_results[:3]:
                        try:
                            _ct_norm = pd.crosstab(_df_b[_cc], _y, normalize='index')
                            _ct_norm.plot(kind='bar', stacked=True, figsize=(10, 3),
                                          colormap='Set2', edgecolor='white', linewidth=0.4)
                            plt.title(f'{{_cc}} × class mix  '
                                      f'(χ²={{_chi2:.1f}}, p={{_p_chi:.4f}}, V={{_cv:.3f}})')
                            plt.ylabel('proportion'); plt.xlabel(_cc)
                            plt.legend(title=_tgt, bbox_to_anchor=(1.01,1), loc='upper left',
                                       fontsize=8)
                            plt.tight_layout(); plt.show()
                        except Exception:
                            pass
        """).strip()
        out = kernel.execute(code)
        return out.as_text(5000), out.figures

    @staticmethod
    def _stage3_association_matrix(kernel) -> tuple[str, list]:
        """Clustered Spearman heatmap + Cramér's V heatmap."""
        code = textwrap.dedent("""
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as _sns
            from scipy.stats import chi2_contingency

            _num = df.select_dtypes('number').fillna(0)
            _cats = df.select_dtypes('object')

            print("=== Global Association Matrix ===")

            # ── Clustered Spearman correlation heatmap ───────────────────────
            if len(_num.columns) >= 2:
                _spear = _num.corr(method='spearman')
                _n_c = len(_spear)
                _annot = _n_c <= 15
                try:
                    _cg = _sns.clustermap(
                        _spear, method='ward', metric='euclidean',
                        cmap='coolwarm', center=0, vmin=-1, vmax=1,
                        annot=_annot, fmt='.1f', linewidths=0.2 if _annot else 0,
                        figsize=(max(8, _n_c*0.5), max(7, _n_c*0.5)),
                    )
                    _cg.ax_heatmap.set_title('Clustered Spearman Correlation', pad=12)
                    plt.tight_layout(); plt.show()
                    # Flag high-correlation pairs
                    _pairs_high = []
                    for _i, _ci in enumerate(_spear.columns):
                        for _j, _cj in enumerate(_spear.columns):
                            if _j <= _i: continue
                            if abs(_spear.loc[_ci, _cj]) > 0.8:
                                _pairs_high.append((_ci, _cj, _spear.loc[_ci, _cj]))
                    if _pairs_high:
                        print("  High Spearman |ρ|>0.8 pairs:")
                        for _ci, _cj, _r in sorted(_pairs_high, key=lambda x: -abs(x[2])):
                            print(f"    {_ci} ↔ {_cj}: ρ={_r:.3f}")
                except Exception as _e:
                    print(f"  clustermap failed ({_e}), using regular heatmap")
                    fig_h, ax_h = plt.subplots(figsize=(max(8, _n_c//2), max(6, _n_c//2)))
                    _sns.heatmap(_spear, annot=_annot, fmt='.2f', cmap='coolwarm',
                                 center=0, square=True, ax=ax_h)
                    ax_h.set_title('Spearman Correlation Matrix')
                    plt.tight_layout(); plt.show()

            # ── Cramér's V heatmap (categorical-categorical) ─────────────────
            _cat_cols_cv = [c for c in _cats.columns if _cats[c].nunique() <= 30][:12]
            if len(_cat_cols_cv) >= 2:
                _cv_mat = pd.DataFrame(np.eye(len(_cat_cols_cv)),
                                       index=_cat_cols_cv, columns=_cat_cols_cv)
                for _ci in _cat_cols_cv:
                    for _cj in _cat_cols_cv:
                        if _ci == _cj: continue
                        try:
                            _ct = pd.crosstab(df[_ci], df[_cj])
                            _chi2 = chi2_contingency(_ct, correction=False)[0]
                            _n = _ct.sum().sum()
                            _k = min(_ct.shape) - 1
                            _cv_mat.loc[_ci, _cj] = min(np.sqrt(_chi2/(_n*_k)), 1) if _k > 0 else 0
                        except Exception:
                            pass
                fig_cv, ax_cv = plt.subplots(figsize=(max(6, len(_cat_cols_cv)*0.7),
                                                       max(5, len(_cat_cols_cv)*0.7)))
                _sns.heatmap(_cv_mat.astype(float), annot=True, fmt='.2f',
                             cmap='YlOrRd', vmin=0, vmax=1,
                             square=True, linewidths=0.3, ax=ax_cv)
                ax_cv.set_title("Cramér's V Matrix (Categorical Association)")
                plt.tight_layout(); plt.show()
                print("  Cramér's V matrix computed for categorical features.")
        """).strip()
        out = kernel.execute(code)
        return out.as_text(2000), out.figures

    @staticmethod
    def _stage4_multivariate(kernel, target: str, depth: str,
                             sample_rows: int) -> tuple[str, list]:
        """PCA scatter + scree + loadings. t-SNE if depth='deep'. Parallel coords."""
        code = textwrap.dedent(f"""
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            _tgt = {repr(target)}
            _depth = {repr(depth)}
            _sr = {sample_rows}
            _df_pca = df.sample(min(_sr, len(df)), random_state=0) if len(df) > _sr else df

            _num_cols = [c for c in _df_pca.select_dtypes('number').columns if c != _tgt]
            if len(_num_cols) < 2:
                print("Not enough numeric features for PCA")
            else:
                _Xs = pd.DataFrame(
                    StandardScaler().fit_transform(_df_pca[_num_cols].fillna(0)),
                    columns=_num_cols
                )
                _pca = PCA(random_state=0)
                _pca.fit(_Xs)
                _comps = _pca.transform(_Xs)
                _var_exp = _pca.explained_variance_ratio_

                print("=== PCA ===")
                print(f"  PC1 explains {{_var_exp[0]:.1%}}, "
                      f"PC1+PC2 explains {{_var_exp[:2].sum():.1%}} of variance")
                EDA_CTX['pca_var_exp'] = _var_exp[:5].tolist()

                # ── Scree plot ───────────────────────────────────────────────
                _n_comp = min(15, len(_var_exp))
                fig_scree, ax_scr = plt.subplots(figsize=(8, 3.5))
                ax_scr.bar(range(1, _n_comp+1), _var_exp[:_n_comp]*100,
                           color='steelblue', alpha=0.8)
                ax_scr.plot(range(1, _n_comp+1),
                            np.cumsum(_var_exp[:_n_comp])*100,
                            'o-', color='#d62728', label='Cumulative %')
                ax_scr.axhline(80, color='gray', linestyle='--', linewidth=0.8,
                               label='80% threshold')
                ax_scr.set_xlabel('Principal Component')
                ax_scr.set_ylabel('Explained Variance (%)')
                ax_scr.set_title('PCA Scree Plot')
                ax_scr.legend(fontsize=8)
                plt.tight_layout(); plt.show()

                # ── PC1 vs PC2 scatter coloured by target ────────────────────
                if _tgt and _tgt in _df_pca.columns:
                    _y_pca = _df_pca[_tgt].values
                    _classes_pca = None
                    _is_cls_pca = _df_pca[_tgt].nunique() < 20
                    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
                    if _is_cls_pca:
                        _classes_pca = sorted(_df_pca[_tgt].unique())
                        _palette_pca = plt.cm.Set1.colors
                        for _j, _cls in enumerate(_classes_pca):
                            _mask_pca = _y_pca == _cls
                            ax_pca.scatter(_comps[_mask_pca, 0], _comps[_mask_pca, 1],
                                           c=[_palette_pca[_j % len(_palette_pca)]],
                                           label=str(_cls), alpha=0.5, s=20, edgecolors='none')
                        ax_pca.legend(title=_tgt, fontsize=8, markerscale=1.5)
                    else:
                        _sc = ax_pca.scatter(_comps[:, 0], _comps[:, 1],
                                             c=_y_pca, cmap='viridis', alpha=0.5, s=20)
                        plt.colorbar(_sc, ax=ax_pca, label=_tgt)
                    ax_pca.set_xlabel(f'PC1 ({{_var_exp[0]:.1%}})')
                    ax_pca.set_ylabel(f'PC2 ({{_var_exp[1]:.1%}})')
                    ax_pca.set_title(f'PCA — PC1 vs PC2  (target: {{_tgt}})')
                    plt.tight_layout(); plt.show()

                # ── PC1 loadings ─────────────────────────────────────────────
                _load1 = pd.Series(_pca.components_[0], index=_num_cols
                                   ).sort_values(key=abs, ascending=False).head(15)
                fig_load, ax_load = plt.subplots(figsize=(9, max(4, len(_load1)*0.38)))
                _lc = ['#d62728' if v < 0 else '#1f77b4' for v in _load1.values]
                ax_load.barh(_load1.index[::-1], _load1.values[::-1], color=_lc[::-1])
                ax_load.axvline(0, color='black', linewidth=0.8)
                ax_load.set_xlabel('Loading on PC1')
                ax_load.set_title(f'PC1 Feature Loadings  ({{_var_exp[0]:.1%}} variance)')
                plt.tight_layout(); plt.show()

                # ── t-SNE (deep only, small datasets) ───────────────────────
                if _depth == 'deep' and len(_df_pca) <= 5000:
                    try:
                        from sklearn.manifold import TSNE
                        _tsne_emb = TSNE(n_components=2, random_state=0,
                                         perplexity=min(30, len(_Xs)//4)
                                         ).fit_transform(_Xs.values)
                        fig_t, ax_t = plt.subplots(figsize=(8, 6))
                        if _is_cls_pca:
                            for _j, _cls in enumerate(_classes_pca or []):
                                _m = _y_pca == _cls
                                ax_t.scatter(_tsne_emb[_m, 0], _tsne_emb[_m, 1],
                                             c=[_palette_pca[_j % len(_palette_pca)]],
                                             label=str(_cls), alpha=0.6, s=15,
                                             edgecolors='none')
                            ax_t.legend(title=_tgt, fontsize=8)
                        else:
                            _sc_t = ax_t.scatter(_tsne_emb[:, 0], _tsne_emb[:, 1],
                                                 c=_y_pca, cmap='viridis', alpha=0.6, s=15)
                            plt.colorbar(_sc_t, ax=ax_t, label=_tgt)
                        ax_t.set_title(f't-SNE 2D Embedding  (target: {{_tgt}})')
                        plt.tight_layout(); plt.show()
                        print("  t-SNE embedding computed.")
                    except Exception as _te:
                        print(f"  t-SNE skipped: {{_te}}")

                # ── Parallel coordinates (top-6 features coloured by target) ─
                if _tgt and _tgt in _df_pca.columns:
                    from sklearn.preprocessing import MinMaxScaler
                    _top6_pc = _load1.index[:6].tolist()
                    _Xpc = pd.DataFrame(
                        MinMaxScaler().fit_transform(_df_pca[_top6_pc].fillna(0)),
                        columns=_top6_pc
                    )
                    _Xpc['_target'] = _df_pca[_tgt].values
                    _is_cls_pc = _df_pca[_tgt].nunique() < 20
                    fig_pc, ax_pc = plt.subplots(figsize=(12, 4))
                    if _is_cls_pc:
                        _cls_list = sorted(_Xpc['_target'].unique())
                        _pal_pc = plt.cm.Set1.colors
                        for _j, _cls in enumerate(_cls_list):
                            _sub = _Xpc[_Xpc['_target'] == _cls].sample(
                                min(150, (_Xpc['_target'] == _cls).sum()), random_state=0)
                            for _, _r in _sub.iterrows():
                                ax_pc.plot(range(len(_top6_pc)), _r[_top6_pc].values,
                                           color=_pal_pc[_j % len(_pal_pc)],
                                           alpha=0.2, linewidth=0.8)
                        from matplotlib.lines import Line2D
                        ax_pc.legend(handles=[
                            Line2D([0],[0], color=_pal_pc[_j % len(_pal_pc)],
                                   label=str(_cls))
                            for _j, _cls in enumerate(_cls_list)
                        ], loc='upper right', fontsize=8, title=_tgt)
                    else:
                        _cmap_pc = plt.cm.viridis
                        _norm_pc = plt.Normalize(_Xpc['_target'].min(),
                                                 _Xpc['_target'].max())
                        for _, _r in _Xpc.sample(min(300, len(_Xpc)), random_state=0).iterrows():
                            ax_pc.plot(range(len(_top6_pc)), _r[_top6_pc].values,
                                       color=_cmap_pc(_norm_pc(_r['_target'])),
                                       alpha=0.3, linewidth=0.6)
                    ax_pc.set_xticks(range(len(_top6_pc)))
                    ax_pc.set_xticklabels(_top6_pc, rotation=20, fontsize=9)
                    ax_pc.set_title(f'Parallel Coordinates — Top PC1 features  ({{_tgt}})')
                    plt.tight_layout(); plt.show()
        """).strip()
        out = kernel.execute(code)
        return out.as_text(2000), out.figures

    @staticmethod
    def _stage5_missingness(kernel) -> tuple[str, list]:
        """Missingness heatmap and pairwise missingness correlations."""
        code = textwrap.dedent("""
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as _sns

            _miss = df.isna()
            _miss_rate = _miss.mean()
            _miss_cols = _miss_rate[_miss_rate > 0].sort_values(ascending=False)

            print("=== Missingness Analysis ===")
            if len(_miss_cols) == 0:
                print("  No missing values detected.")
            else:
                print(f"  {len(_miss_cols)} columns with missing values:")
                print(_miss_cols.round(4).to_string())

                # ── Visual missingness matrix (like msno.matrix) ─────────────
                _cols_miss = _miss_cols.index[:20].tolist()
                _sample_idx = np.linspace(0, len(df)-1, min(500, len(df)), dtype=int)
                _miss_sample = _miss.iloc[_sample_idx][_cols_miss]
                fig_m, ax_m = plt.subplots(figsize=(max(8, len(_cols_miss)*0.7), 5))
                _sns.heatmap(_miss_sample.T, cmap=['#f7f7f7', '#d62728'],
                             cbar=False, ax=ax_m, xticklabels=False,
                             linewidths=0, yticklabels=_cols_miss)
                ax_m.set_xlabel('Rows (sampled)')
                ax_m.set_title('Missingness Map  (red = missing)')
                plt.tight_layout(); plt.show()

                # ── Pairwise missingness correlation ─────────────────────────
                if len(_cols_miss) >= 2:
                    _miss_corr = _miss[_cols_miss].corr()
                    _pairs_miss = []
                    for _i, _ci in enumerate(_cols_miss):
                        for _j, _cj in enumerate(_cols_miss):
                            if _j <= _i: continue
                            _r = _miss_corr.loc[_ci, _cj]
                            if abs(_r) > 0.3:
                                _pairs_miss.append((_ci, _cj, _r))
                    if _pairs_miss:
                        print("  Correlated missingness patterns (|r|>0.3):")
                        for _ci, _cj, _r in sorted(_pairs_miss, key=lambda x: -abs(x[2])):
                            print(f"    {_ci} & {_cj}: r={_r:.3f}")
                        _pmdf = pd.DataFrame(_pairs_miss, columns=['feat1','feat2','r'])
                        fig_mc, ax_mc = plt.subplots(figsize=(8, max(3, len(_pmdf)*0.4)))
                        ax_mc.barh(range(len(_pmdf)),
                                   _pmdf['r'].values,
                                   color=['#d62728' if v < 0 else '#1f77b4'
                                          for v in _pmdf['r'].values])
                        ax_mc.set_yticks(range(len(_pmdf)))
                        ax_mc.set_yticklabels(
                            [f"{r['feat1']} ↔ {r['feat2']}" for _, r in _pmdf.iterrows()],
                            fontsize=8)
                        ax_mc.axvline(0, color='black', linewidth=0.8)
                        ax_mc.set_xlabel('Missingness correlation r')
                        ax_mc.set_title('Pairwise Missingness Correlations')
                        plt.tight_layout(); plt.show()
                    else:
                        print("  No strong pairwise missingness correlations found.")
        """).strip()
        out = kernel.execute(code)
        return out.as_text(2000), out.figures

    def _stage6_narrative(self, kernel, target: str,
                          llm_client, all_text: str) -> str:
        """LLM-generated EDA narrative.

        Primary input: EDA_CTX['findings'] (severity-ranked, deterministic).
        Secondary input: raw EDA stats (truncated). This ordering ensures the
        LLM discusses the most important discovered issues rather than re-deriving
        patterns from noisy raw numbers.
        """
        ctx = kernel.namespace.get("EDA_CTX", {})
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        task_type = task_spec.get("task_type", "unknown")
        metric = task_spec.get("evaluation_metric", "auto")

        # ── Build findings block (primary signal) ──────────────────────────
        findings: list = ctx.get("findings", [])
        if findings:
            findings_txt = "\n".join(
                f"  [{f['severity'].upper()}] {f['check']}: {f['detail']}"
                f"\n    → Action: {f['action']}"
                for f in findings
            )
        else:
            findings_txt = "  None detected."

        quality = ctx.get("quality", {})
        q_score = quality.get("overall", "N/A")
        q_dims = ", ".join(f"{k}={v:.2f}" for k, v in quality.items()
                           if k != "overall") if quality else ""
        n_outliers = ctx.get("n_outliers", "N/A")
        rec_tf = ctx.get("recommended_transforms", {})
        rec_txt = "; ".join(f"{op}: {cols[:3]}" for op, cols in rec_tf.items()
                            if cols) if rec_tf else "none"
        pca_var = ctx.get("pca_var_exp", [])
        pca_txt = (f"PC1+PC2 explain {sum(pca_var[:2]):.1%}" if len(pca_var) >= 2
                   else "PCA not run")
        stl = ctx.get("stl", {})
        stl_txt = (f"STL: trend={stl.get('trend_pct',0):.1%}, "
                   f"seasonal={stl.get('seasonal_pct',0):.1%}, "
                   f"residual={stl.get('residual_pct',0):.1%}")  if stl else ""

        prompt = (
            f"You are a senior data scientist writing an EDA findings report.\n\n"
            f"TASK: {task_type} | TARGET: '{target}' | METRIC: {metric}\n"
            f"DATA QUALITY: overall={q_score} ({q_dims})\n"
            f"OUTLIERS: {n_outliers} (Isolation Forest ∪ Mahalanobis)\n"
            f"RECOMMENDED TRANSFORMS: {rec_txt}\n"
            f"MULTIVARIATE STRUCTURE: {pca_txt}\n"
            + (f"TIME-SERIES: {stl_txt}\n" if stl_txt else "")
            + f"\n"
            f"=== SEVERITY-RANKED FINDINGS (primary input) ===\n"
            f"{findings_txt}\n\n"
            f"=== RAW EDA STATS (secondary context) ===\n"
            f"{all_text[:2500]}\n\n"
            f"Write exactly 3 paragraphs:\n"
            f"1. STRONGEST PREDICTORS — name the top features, direction of effect, "
            f"   and any surprising relationships. Use LaTeX (e.g. $\\rho$) where natural.\n"
            f"2. RISKS — address every CRITICAL finding first, then WARNING findings. "
            f"   Be specific: name the column, the statistic, and why it matters.\n"
            f"3. RECOMMENDATIONS — concrete feature engineering steps (log1p which columns, "
            f"   target-encode which columns, drop which columns) and model selection advice "
            f"   (tree vs linear, which metric, any regularisation). No vague suggestions.\n"
            f"Be direct. No introductory sentences like 'In this analysis...'"
        )
        try:
            raw = llm_client._generate(
                prompt, temperature=0.2, max_tokens=800,
            )
            return llm_client._strip_thinking(raw)
        except Exception as exc:
            return f"[LLM narrative unavailable: {exc}]"

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        depth = args.get("depth", "standard")
        sample_rows = args.get("sample_rows", 5000)
        target = session.target or kernel.namespace.get("TARGET") or ""
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        task_type = task_spec.get("task_type", "")

        # Initialise EDA_CTX dict in kernel
        kernel.execute("EDA_CTX = {}")

        all_text: list[str] = []
        all_figs: list[str] = []

        def _run(label: str, fn, *fn_args):
            try:
                t, f = fn(*fn_args)
                all_text.append(f"--- {label} ---\n{t}")
                all_figs.extend(f)
            except Exception as exc:
                all_text.append(f"--- {label} --- [SKIPPED: {exc}]")

        _run("Stage 0: Quality Radar + Outlier Map",
             self._stage0_quality_radar, kernel)
        _run("Stage 0.5: Findings Collector",
             self._stage05_findings, kernel, target, task_type)
        _run("Stage 1: Univariate Panorama",
             self._stage1_univariate, kernel, sample_rows)
        if target:
            _run("Stage 2: Bivariate Analysis",
                 self._stage2_bivariate, kernel, target, sample_rows)
        _run("Stage 3: Association Matrix",
             self._stage3_association_matrix, kernel)
        if depth != "quick":
            _run("Stage 4: Multivariate Structure (PCA)",
                 self._stage4_multivariate, kernel, target, depth, sample_rows)
        _run("Stage 5: Missingness Map",
             self._stage5_missingness, kernel)

        full_text = "\n\n".join(all_text)

        # Stage 6: LLM narrative
        narrative = ""
        if target:
            narrative = self._stage6_narrative(
                kernel, target, llm_client, full_text
            )
            full_text += f"\n\n--- Stage 6: LLM Narrative ---\n{narrative}"
            kernel.execute(f"EDA_CTX['narrative'] = {repr(narrative)}")

        return ToolOutput(
            success=True,
            text=full_text[:8000],
            figures=all_figs,
            artifacts={"deep_eda": True, "n_stages": 6, "narrative_generated": bool(narrative)},
        )


# ── Data Analysis Agent tool ─────────────────────────────────────────────────

class DataAnalysisAgentTool(BaseTool):
    """Specialist data analysis agent.

    Runs a programmatic statistical scan (nulls, skew, cardinality, outliers,
    MI feature ranking, leakage detection) then calls the LLM to synthesise
    findings into a structured DATA_ANALYSIS_REPORT with concrete ML
    recommendations. Stores the report in kernel["DATA_ANALYSIS_REPORT"].
    The MLStrategyAgentTool (or directly train_model) reads this report to
    configure model selection and hyperparameters.
    """

    name = "data_analysis_agent"
    description = (
        "Specialist data analysis agent. Runs statistical profiling + LLM synthesis "
        "to produce a DATA_ANALYSIS_REPORT with ML recommendations (model types, "
        "hyperparameter hints, preprocessing steps, features to exclude). "
        "Run this BEFORE ml_strategy_agent or train_model for intelligent model selection."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "depth": {
                "type": "string",
                "enum": ["quick", "full"],
                "description": "'quick' skips MI computation; 'full' (default) computes MI scores.",
            }
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        from agent.api.data_analysis_agent import DataAnalysisAgent

        df = kernel.namespace.get("df")
        if df is None:
            return ToolOutput(success=False, text="No data loaded.",
                              error="no df in kernel")

        target = (session.target or kernel.namespace.get("TARGET") or "")
        task_spec = kernel.namespace.get("TASK_SPEC", {})
        task_type = task_spec.get("task_type", "unknown")
        depth = args.get("depth", "full")

        # Pass any existing MI scores so we don't recompute
        mi_scores: dict | None = None
        if depth == "full":
            raw_mi = kernel.namespace.get("MI_SCORES")
            if isinstance(raw_mi, dict):
                mi_scores = raw_mi

        agent = DataAnalysisAgent(llm_client=llm_client)
        report = agent.analyze(
            df,
            target=target,
            task_type=task_type,
            task_spec=task_spec,
            mi_scores=mi_scores,
        )

        report_dict = report.to_dict()
        kernel.namespace["DATA_ANALYSIS_REPORT"] = report_dict

        # Build human-readable summary
        lines = [
            "=== Data Analysis Agent Report ===",
            f"Dataset : {report.n_rows:,} rows × {report.n_cols} cols",
            f"Target  : {report.target or '(not set)'} | Task: {report.task_type}",
        ]
        if report.class_imbalance_ratio is not None:
            lines.append(
                f"Imbalance ratio: {report.class_imbalance_ratio:.3f} | "
                f"Metric: {report.recommended_metric}"
            )
        if report.null_pcts:
            top_nulls = list(report.null_pcts.items())[:5]
            lines.append(f"Cols with nulls (>5%): {top_nulls}")
        if report.skewed_cols:
            top_skew = list(report.skewed_cols.items())[:5]
            lines.append(f"Skewed cols: {top_skew}")
        if report.high_cardinality_cols:
            lines.append(f"High-cardinality: {report.high_cardinality_cols[:5]}")
        if report.near_constant_cols:
            lines.append(f"Near-constant (drop): {report.near_constant_cols[:5]}")
        if report.outlier_cols:
            lines.append(f"Outlier-rich cols: {report.outlier_cols[:8]}")
        if report.top_mi_features:
            lines.append(
                "Top features by MI: "
                + ", ".join(f"{c}={v:.3f}" for c, v in report.top_mi_features[:8])
            )
        if report.leakage_suspects:
            lines.append(f"⚠ Leakage suspects (|corr|>0.9): {report.leakage_suspects}")

        if report.analyst_narrative:
            lines.append(f"\nAnalyst narrative:\n{report.analyst_narrative}")

        rec = report.ml_recommendations
        if rec:
            lines.append("\n✓ ML Recommendations (written to DATA_ANALYSIS_REPORT):")
            for m in rec.get("models", []):
                if isinstance(m, dict):
                    lines.append(f"  Model: {m['name']} | params: {m.get('hyperparams', {})}")
            if rec.get("preprocessing"):
                lines.append(f"  Preprocessing: {rec['preprocessing']}")
            if rec.get("features_to_exclude"):
                lines.append(f"  Exclude: {rec['features_to_exclude']}")
            if rec.get("rationale"):
                lines.append(f"  Rationale: {rec['rationale']}")

        lines.append(
            "\n→ Run 'ml_strategy_agent' to confirm the ML plan, "
            "or 'train_model' directly — it will read DATA_ANALYSIS_REPORT automatically."
        )

        return ToolOutput(
            success=True,
            text="\n".join(lines),
            artifacts={
                "data_analysis_report": True,
                "n_rows": report.n_rows,
                "n_cols": report.n_cols,
                "target": report.target,
                "n_top_features": len(report.top_mi_features),
                "recommendations_generated": bool(rec),
            },
        )


# ── ML Strategy Agent tool ────────────────────────────────────────────────────

class MLStrategyAgentTool(BaseTool):
    """ML strategy agent — reads DATA_ANALYSIS_REPORT → produces ML_PLAN.

    Reads the structured report from DataAnalysisAgentTool, calls the LLM
    to refine and confirm the model strategy, then writes ML_PLAN to the
    kernel. train_model reads ML_PLAN to use the confirmed model list and
    hyperparameters instead of its internal defaults.
    """

    name = "ml_strategy_agent"
    description = (
        "ML strategy agent. Reads DATA_ANALYSIS_REPORT produced by data_analysis_agent, "
        "calls LLM to confirm/refine model selection strategy, writes ML_PLAN to kernel. "
        "train_model automatically reads ML_PLAN for model types and hyperparameter hints. "
        "Run after data_analysis_agent and before train_model."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        from agent.api.data_analysis_agent import MLStrategyPlanner

        report_dict: dict = kernel.namespace.get("DATA_ANALYSIS_REPORT", {})
        if not report_dict:
            return ToolOutput(
                success=False,
                text="No DATA_ANALYSIS_REPORT found. Run data_analysis_agent first.",
                error="missing DATA_ANALYSIS_REPORT",
            )

        task_spec = kernel.namespace.get("TASK_SPEC", {})
        planner = MLStrategyPlanner(llm_client=llm_client)
        ml_plan = planner.plan(report_dict, task_spec=task_spec)

        plan_dict = ml_plan.to_dict()
        kernel.namespace["ML_PLAN"] = plan_dict

        lines = [
            "=== ML Strategy Agent — Plan ===",
            f"Task type : {ml_plan.task_type}",
            f"Metric    : {ml_plan.eval_metric}",
        ]
        if ml_plan.models:
            lines.append("\nModels to train (in priority order):")
            for i, m in enumerate(ml_plan.models, 1):
                if isinstance(m, dict):
                    lines.append(f"  {i}. {m['name']} — {m.get('hyperparams', {})}")
        if ml_plan.preprocessing:
            lines.append(f"\nPreprocessing: {ml_plan.preprocessing}")
        if ml_plan.features_to_exclude:
            lines.append(f"Exclude features: {ml_plan.features_to_exclude}")
        if ml_plan.rationale:
            lines.append(f"\nRationale: {ml_plan.rationale}")
        if ml_plan.source_report_summary:
            lines.append(f"\nBased on:\n{ml_plan.source_report_summary[:400]}")

        lines.append(
            "\n✓ ML_PLAN written to kernel. "
            "Run 'train_model' — it will use this plan automatically."
        )

        return ToolOutput(
            success=True,
            text="\n".join(lines),
            artifacts={
                "ml_plan": True,
                "models": [m["name"] for m in ml_plan.models if isinstance(m, dict)],
                "eval_metric": ml_plan.eval_metric,
            },
        )


# ── registry ──────────────────────────────────────────────────────────────────

# Output budget per planning stage. A reasoning model spends most of this
# thinking: measured with gpt-oss:20b on the WorkForce brief, the scope stage
# produced 10,147 characters of thinking before emitting the plan, and at 1500
# and 3000 tokens it never reached the JSON at all. At 8000 it completed in
# 284 s. Override with DSAGENT_PLAN_TOKENS.
_PLAN_STAGE_TOKENS = int(os.environ.get("DSAGENT_PLAN_TOKENS", "8000"))

_PLAN_SECTIONS = ("objective", "requirement_coverage", "data_preparation",
                  "eda_questions_task", "eda_questions_independent", "risks",
                  "feature_engineering", "modelling", "evaluation",
                  "deliverables")

_PLAN_COMMON = """\
THE BRIEF IN ITS OWN TERMS:
{brief_context}

WHAT THE BRIEF ASKS FOR:
{requirements}

TASK SPEC EXTRACTED FROM THE BRIEF:
{task_spec}

THE ACTUAL DATA (dtypes, missingness, cardinality, value ranges):
{data_summary}

{evidence}"""

# Stage one: what is being asked, against what the data actually holds.
_PLAN_SCOPE_PROMPT = """\
/no_think
You are the lead data scientist. This is the FIRST half of planning: work out
what is really being asked and whether this data can answer it.

{mode_instruction}

Reading the brief is step one, not the plan. Do not restate the brief. Compare
what it asks for against the columns below and say what follows.

""" + _PLAN_COMMON + """

Think it through in this order:
1. For each thing the brief asks for, which columns actually serve it?
2. What does the brief ask for that the data does NOT contain, so has to be
   derived, joined or approximated? Name the derivation.
3. What could invalidate the whole approach (leakage, a target that is mostly
   missing, too few positives, a join that does not hold, a column that is
   constant or malformed)? Those become questions EDA must settle first.

Reply with a single JSON object and nothing else:
{{"objective": "one or two sentences naming the decision this work supports",
  "requirement_coverage": [
     {{"requirement": "the ask, in a few words",
       "approach": "what you will do about it",
       "columns": ["column names that serve it"],
       "open_question": "what must be checked before this is safe, or empty"}}],
  "eda_questions_task": ["a question that must be answered to deliver the \
brief, phrased so an analysis can settle it", ...],
  "eda_questions_independent": ["a question about the data in its own right \
that the brief does not ask but a careful analyst would check anyway", ...],
  "risks": ["what could make this analysis wrong, and how it will be checked"]}}

Rules:
- Every entry names a real column, a real number, or a real derivation. An
  entry that would read the same for any dataset is wrong: delete it.
- requirement_coverage must cover every requirement listed above. If the data
  cannot support one, say so in open_question rather than inventing a step.
- eda_questions_independent must NOT be restatements of the brief. They are the
  checks you would run on this data whatever the task was: distributions,
  missingness structure, duplicates, time coverage, relationships between
  columns, anything that looks wrong or malformed above.
- eda_questions_task holds 4 to 8 ANALYTICAL questions whose answers deliver
  the brief (patterns, comparisons between entities, concentration, trends),
  most important first. Data-quality checks belong in
  eda_questions_independent, at most 5 of them. Never list the same question
  twice in different words.
"""

# Stage two: how the work gets done, grounded in stage one.
_PLAN_APPROACH_PROMPT = """\
/no_think
You are the lead data scientist. This is the SECOND half of planning. The scope
below is already settled; decide how the work gets done.

{mode_instruction}

""" + _PLAN_COMMON + """

SCOPE ALREADY AGREED (plan to serve this):
{scope}

Reply with a single JSON object and nothing else:
{{"data_preparation": ["concrete step naming real columns", ...],
  "feature_engineering": ["concrete step, naming columns", ...],
  "modelling": ["candidate model and why it suits THIS data", ...],
  "evaluation": ["metric and validation scheme, with the reason", ...],
  "deliverables": ["what is handed over, matching what the brief asked for"],
  "revision_notes": ["what changed from the previous plan and which finding \
caused it", ...]}}

Rules:
- Every choice is justified by a number, a dtype or a finding you can point to
  above. No generic checklists.
- Address the open_question entries and risks from the scope: either a step
  resolves one, or say which EDA answer it waits on.
- modelling lists only what the brief's questions need. With no labelled
  target, name unsupervised or rule-based methods, never a supervised model.
  When the brief asks for no model at all, return [].
- deliverables match what the brief asks for and who it is for (audiences).
- revision_notes is empty on a first plan.
"""


class WorkPlanTool(BaseTool):
    name = "work_plan"
    description = (
        "Write or revise the plan for the whole piece of work, and keep it on "
        "the session as WORK_PLAN. Call it early to plan from the brief and the "
        "data schema, then call it AGAIN with revise=true after EDA "
        "(eda_profile / deep_eda / data_analysis_agent) so the feature "
        "engineering and model choices are rewritten from what the data "
        "actually showed rather than from assumptions. Pass save_to to also "
        "write the plan as a markdown file in the working directory."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "revise": {"type": "boolean",
                       "description": "True to revise the existing plan using "
                                      "EDA findings (the post-EDA checkpoint)"},
            "save_to": {"type": "string",
                        "description": "Optional markdown filename, e.g. "
                                       "analysis_plan.md"},
            "notes": {"type": "string",
                      "description": "Anything the user asked for that the "
                                     "brief does not cover"},
        },
    }

    def execute(self, args, *, session, kernel, llm_client) -> ToolOutput:
        ns = kernel.namespace
        spec: dict = ns.get("TASK_SPEC") or {}
        revise = bool(args.get("revise"))
        existing = ns.get("WORK_PLAN") if isinstance(ns.get("WORK_PLAN"), dict) else None

        report: dict = ns.get("DATA_ANALYSIS_REPORT") or {}
        log = ns.get("EXPLORATION_LOG")
        log = log if isinstance(log, list) else []
        turn = getattr(kernel, "turn_deadline", None)
        if (existing and not revise and not args.get("notes")
                and turn is not None and existing.get("turn") == turn):
            # Already planned this turn: rewriting from the same inputs costs
            # minutes of model time and changes nothing.
            return ToolOutput(
                success=True,
                text=(f"The work plan (version {existing.get('version', 1)}) was "
                      "already written this turn; it is unchanged. Work through "
                      "the EDA agenda, then call work_plan with revise=true.\n\n"
                      + _plan_to_markdown(existing, spec))[:6000],
                artifacts={"work_plan": True, "version": existing.get("version", 1),
                           "reused": True})
        if revise and not eda_has_run(kernel):
            return ToolOutput(
                success=False, text="",
                error="Nothing to revise against yet. Explore the data first "
                      "(execute_code, eda_profile, deep_eda or "
                      "data_analysis_agent), then call work_plan with "
                      "revise=true.")

        reqs = [r for r in (spec.get("specific_requirements") or [])
                if isinstance(r, str) and r.strip()]
        req_text = "\n".join(f"- {r[:300]}" for r in reqs[:15]) or "(not stated)"
        if args.get("notes"):
            req_text += f"\n- (from the user) {str(args['notes'])[:300]}"

        # Bounded: this goes in a prompt alongside the brief and the
        # EDA evidence, and num_ctx is small on a local model.
        data_profile = _profile_for_planning(ns)[:6000]

        evidence = ""
        if revise:
            evidence = "WHAT EDA ACTUALLY FOUND (plan against these, not assumptions):\n"
            evidence += _summarise_report(report)
            if log:
                evidence += "\nRunning findings:\n" + "\n".join(
                    f"- {x}" for x in log[-10:])
            asked = [str(a.get("question", "")) for a in (ns.get("EDA_AGENDA") or [])
                     if isinstance(a, dict)]
            if asked:
                evidence += ("\n\nEDA QUESTIONS ALREADY ON THE AGENDA (do not repeat "
                             "them; add only new questions the findings raise):\n"
                             + "\n".join(f"- {q[:160]}" for q in asked[:24]))
            if existing:
                evidence += ("\n\nTHE PLAN YOU ARE REVISING:\n"
                             + json.dumps({k: existing.get(k)
                                           for k in ("feature_engineering",
                                                     "modelling", "evaluation")},
                                          indent=1)[:1500])
            mode = ("Revise the plan below using the EDA findings. Change the "
                    "feature_engineering, modelling and evaluation sections "
                    "wherever a finding contradicts the original assumption, "
                    "and record each change in revision_notes.")
        else:
            mode = ("Write the first plan. It will be revised after EDA, so "
                    "state the assumptions the later evidence should test.")

        common = dict(
            brief_context=_brief_context(spec),
            requirements=req_text,
            task_spec=json.dumps({k: spec.get(k) for k in (
                "task_type", "target_column", "evaluation_metric",
                "aggregation_needed", "task_description")}, indent=1)[:800],
            data_summary=data_profile,
            evidence=evidence)

        def _ask(prompt: str):
            return _extract_json_obj(llm_client._generate(
                prompt, temperature=0.1, max_tokens=_PLAN_STAGE_TOKENS,
                json_mode=True))

        # Two stages on purpose. One call carrying every section runs past the
        # request timeout on a large local model, and a reply cut off mid-JSON
        # loses the lot. Split, the scope survives even if the approach fails,
        # and the approach is written against a settled scope rather than
        # against the brief alone.
        try:
            scope = _ask(_PLAN_SCOPE_PROMPT.format(
                mode_instruction=mode, **common))
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(success=False, text="",
                              error=f"Plan (scope) failed: {exc}")
        if not isinstance(scope, dict) or not scope.get("objective"):
            return ToolOutput(success=False, text="",
                              error="The model did not return a usable scope. "
                                    "Try again, or narrow the request.")

        plan: dict = dict(scope)
        partial = ""
        try:
            approach = _ask(_PLAN_APPROACH_PROMPT.format(
                mode_instruction=mode,
                scope=json.dumps({k: scope.get(k) for k in (
                    "objective", "requirement_coverage", "risks")},
                    indent=1)[:2000],
                **common))
        except Exception as exc:  # noqa: BLE001
            approach, partial = None, f" (the approach half failed: {exc})"
        if isinstance(approach, dict):
            plan.update({k: v for k, v in approach.items() if v})
        elif not partial:
            partial = (" (the approach half did not come back usable; the "
                       "scope below stands and can be extended by calling "
                       "work_plan again)")

        plan["revised_after_eda"] = revise or bool(
            existing and existing.get("revised_after_eda"))
        plan["version"] = int((existing or {}).get("version", 0)) + 1
        plan["turn"] = turn
        ns["WORK_PLAN"] = plan
        agenda = sync_agenda(ns, plan)
        n_open = sum(a.get("status") == "open" for a in agenda)

        md = _plan_to_markdown(plan, spec)
        text = (f"{'Revised' if revise else 'Wrote'} the work plan "
                f"(version {plan['version']}"
                + (", grounded in EDA findings" if revise else
                   ", assumptions to be tested by EDA") + ")."
                + partial
                + f" EDA agenda: {len(agenda)} questions, {n_open} open."
                + "\n\n" + md)

        save_to = (args.get("save_to") or "").strip()
        if save_to:
            saved = _save_plan_file(save_to, md, session=session, kernel=kernel)
            text += f"\n\n{saved}"
        return ToolOutput(success=True, text=text[:6000],
                          artifacts={"work_plan": True,
                                     "version": plan["version"],
                                     "revised_after_eda": plan["revised_after_eda"]})


def _brief_context(spec: dict) -> str:
    """The brief's objective, audiences, constraints and data notes, for the
    planner. Empty sections are left out; bounded for a small context."""
    brief = spec.get("brief") if isinstance(spec.get("brief"), dict) else {}
    if not brief:
        return "(not available)"
    parts = []
    if brief.get("objective"):
        parts.append(f"Objective: {brief['objective']}")
    if brief.get("task_nature"):
        parts.append(f"Nature of the task: {brief['task_nature']}")
    for label, key, n in (("Audiences", "audiences", 6), ("Constraints", "constraints", 8),
                          ("Data dictionary", "data_notes", 30),
                          ("Left unclear by the brief", "open_questions", 6)):
        items = [str(x)[:200] for x in (brief.get(key) or [])[:n]]
        if items:
            parts.append(f"{label}:\n" + "\n".join(f"- {x}" for x in items))
    return "\n".join(parts)[:3500]


def _extract_json_obj(raw: str):
    """First JSON object in the model's reply, or None.

    Falls back to repairing a reply that was cut off by the token budget: a
    local model that runs out mid-list leaves valid JSON followed by an
    unterminated tail, and discarding the whole plan over the last item wastes
    the minutes it took to produce. Closes any open string and brackets and
    reparses.
    """
    import re as _re
    txt = (raw or "").strip()
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        pass
    m = _re.search(r"\{.*\}", txt, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:  # noqa: BLE001
            pass
    start = txt.find("{")
    if start == -1:
        return None
    return _repair_truncated_json(txt[start:])


def _repair_truncated_json(txt: str):
    """Close whatever the model left open, then parse. None if still broken.

    Walks once recording, for every position, whether it sits inside a string
    and what brackets are open there. Then tries cut points from the end
    backwards, stopping at each place a complete value ends, trimming any
    trailing comma and closing the brackets that were open AT THAT POINT. The
    stack has to match the cut, which is what a single pass over the whole
    string gets wrong when the tail is discarded.
    """
    stacks: list[tuple[bool, tuple]] = []      # per index: (in_string, stack)
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in txt:
        stacks.append((in_str, tuple(stack)))
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    for i in range(len(txt) - 1, -1, -1):
        # A complete value ends at }, ] or the closing quote of a string.
        if txt[i] not in "}]\"":
            continue
        if txt[i] == '"' and stacks[i][0] is False:
            continue                     # an opening quote, not a closing one
        body = txt[:i + 1].rstrip()
        while body.endswith(","):
            body = body[:-1].rstrip()
        # Brackets open immediately AFTER this character.
        depth = list(stacks[i][1])
        if txt[i] in "}]" and depth:
            depth.pop()
        body += "".join(reversed(depth))
        try:
            parsed = json.loads(body)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _profile_for_planning(ns: dict, max_cols: int = 40) -> str:
    """Real evidence about the loaded data, for the planner to reason over.

    A plan written from a column list can only restate the brief. To decide
    what to derive, what to join on and what the data cannot answer, the
    planner needs dtypes, how much is missing, how many distinct values a
    column holds, and what the values look like. This is cheap: it reads the
    frames already in memory and never runs a model.
    """
    out: list[str] = []
    frames = [(n, ns.get(n)) for n in ["df"] + [f"df{i}" for i in range(2, 8)]]
    frames = [(n, f) for n, f in frames if f is not None]
    if not frames:
        return "(no data loaded)"
    for name, df in frames:
        if df is None:
            continue
        try:
            out.append(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
            n_dup = int(df.duplicated().sum())
            if n_dup:
                out.append(f"  duplicate rows: {n_dup:,}")
            cols = list(df.columns)[:max_cols]
            for c in cols:
                s = df[c]
                null_pct = float(s.isna().mean() * 100)
                nun = int(s.nunique(dropna=True))
                bits = [f"  - {c} ({s.dtype})", f"nulls {null_pct:.0f}%",
                        f"{nun:,} distinct"]
                if nun and nun <= 8:
                    vals = [str(v)[:24] for v in s.dropna().unique()[:8]]
                    bits.append("values " + ", ".join(vals))
                elif s.dtype.kind in "if":
                    try:
                        bits.append(f"range {s.min():.4g}..{s.max():.4g}, "
                                    f"median {s.median():.4g}")
                    except Exception:  # noqa: BLE001
                        pass
                elif s.dtype.kind in "OU":
                    vals = [str(v)[:24] for v in s.dropna().unique()[:3]]
                    if vals:
                        bits.append("e.g. " + ", ".join(vals))
                out.append(" | ".join(bits))
            if len(df.columns) > max_cols:
                out.append(f"  … +{len(df.columns) - max_cols} more columns")
        except Exception:  # noqa: BLE001
            out.append(f"{name}: (could not profile)")
    # Shared columns matter: they are the join keys a plan has to name.
    if len(frames) > 1:
        try:
            first = frames[0][1]
            if first is None:
                return "\n".join(out)
            shared = set(first.columns)
            for _, f in frames[1:]:
                if f is not None:
                    shared &= set(f.columns)
            if shared:
                out.append(f"\nColumns present in every file (possible join "
                           f"keys): {sorted(shared)[:20]}")
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(out)


def _summarise_report(report: dict) -> str:
    """The EDA facts a plan should react to, as short lines."""
    if not report:
        return "(no structured analysis report; rely on the running findings)"
    out: list[str] = []
    for key, label in (
        ("n_rows", "rows"), ("n_cols", "columns"), ("target", "target"),
        ("task_type", "task type"), ("recommended_metric", "suggested metric"),
        ("class_imbalance_ratio", "class imbalance (minority/majority)"),
    ):
        if report.get(key) not in (None, "", {}, []):
            out.append(f"- {label}: {report[key]}")
    for key, label in (
        ("leakage_suspects", "LEAKING columns (must be dropped)"),
        ("near_constant_cols", "near-constant columns"),
        ("high_cardinality_cols", "high-cardinality categoricals"),
        ("outlier_cols", "columns with heavy outliers"),
    ):
        vals = report.get(key) or []
        if vals:
            out.append(f"- {label}: {list(vals)[:10]}")
    skew = report.get("skewed_cols") or {}
    if skew:
        out.append(f"- skewed columns: {list(skew)[:10]}")
    nulls = report.get("null_pcts") or {}
    if nulls:
        out.append(f"- columns with notable nulls: "
                   f"{list(nulls.items())[:8]}")
    mi = report.get("top_mi_features") or []
    if mi:
        out.append(f"- most informative features: {mi[:10]}")
    if report.get("analyst_narrative"):
        out.append(f"- analyst notes: {str(report['analyst_narrative'])[:400]}")
    return "\n".join(out) or "(report present but empty)"


def _plan_to_markdown(plan: dict, spec: dict) -> str:
    """Render the plan for a human, and for the .md file."""
    titles = {
        "objective": "Objective",
        "requirement_coverage": "How each requirement is covered",
        "data_preparation": "Data preparation",
        "eda_questions_task": "EDA: questions the brief requires answering",
        "eda_questions_independent": "EDA: independent checks on the data",
        "risks": "What could make this wrong",
        "feature_engineering": "Feature engineering",
        "modelling": "Modelling",
        "evaluation": "Evaluation",
        "deliverables": "Deliverables",
    }
    lines = ["# Analysis plan", ""]
    if plan.get("revised_after_eda"):
        lines += ["_Revised after EDA: the feature and model choices below "
                  "follow what the data showed._", ""]
    else:
        lines += ["_First pass, written before EDA. The feature and model "
                  "choices are assumptions to be tested._", ""]
    if spec.get("target_column"):
        lines += [f"Target: `{spec['target_column']}`  ",
                  f"Metric: `{spec.get('evaluation_metric') or 'to be decided'}`", ""]
    for key in _PLAN_SECTIONS:
        val = plan.get(key)
        if not val:
            continue
        lines.append(f"## {titles.get(key, key)}")
        if isinstance(val, str):
            lines.append(val)
        elif key == "requirement_coverage":
            # A table, because the point of this section is the mapping: which
            # ask, handled how, using which columns, and what is still unknown.
            lines += ["| Requirement | Approach | Columns | Open question |",
                      "| --- | --- | --- | --- |"]
            for item in val:
                if not isinstance(item, dict):
                    lines.append(f"| {item} |  |  |  |")
                    continue
                cols = item.get("columns") or []
                cols = ", ".join(f"`{c}`" for c in cols) if isinstance(cols, list) \
                    else str(cols)
                lines.append(
                    f"| {item.get('requirement', '')} "
                    f"| {item.get('approach', '')} "
                    f"| {cols} "
                    f"| {item.get('open_question', '') or '-'} |")
        else:
            lines += [f"{i}. {v}" for i, v in enumerate(val, 1)]
        lines.append("")
    notes = plan.get("revision_notes") or []
    if notes:
        lines += ["## What changed after EDA, and why", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    return "\n".join(lines)


# First line of every plan file the agent writes. Documents in the working
# directory are loaded as the user's instructions, and a saved plan is the
# agent's own output: read back as a brief, an old plan (e.g. one that named a
# wrong target) would override what the user actually asked.
AGENT_PLAN_MARKER = "<!-- ds-agent generated plan: not user instructions -->"
# Plans saved before the marker existed start with one of these lines.
_LEGACY_PLAN_LINES = (
    "_First pass, written before EDA. The feature and model choices are "
    "assumptions to be tested._",
    "_Revised after EDA: the feature and model choices below follow what "
    "the data showed._",
)


def is_agent_generated_doc(text: str) -> bool:
    """True when a document is a plan this agent wrote, not user instructions."""
    head = (text or "")[:600]
    return AGENT_PLAN_MARKER in head or any(line in head for line in _LEGACY_PLAN_LINES)


def _save_plan_file(save_to: str, md: str, *, session, kernel) -> str:
    """Write the plan through the same approval path as any other file."""
    md = f"{AGENT_PLAN_MARKER}\n{md}"
    root = _ws_root(session, kernel)
    if root is None:
        return "(Not saved: no working directory set.)"
    if not save_to.lower().endswith(".md"):
        save_to += ".md"
    target = _ws_resolve_within(root, save_to)
    if target is None:
        return f"(Not saved: path escapes the working directory: {save_to!r}.)"
    if not target.parent.is_dir():
        return f"(Not saved: folder does not exist: {_ws_rel(root, target.parent)}.)"
    mode = _ws_write_mode(kernel)
    if mode == "suggest":
        return f"(Would save the plan to {save_to}; writes are OFF.)"
    old = target.read_text(errors="replace") if target.is_file() else ""
    held = _hold_for_approval(
        kernel, kind="file", rel=save_to, path=target,
        verb=f"{'update' if old else 'create'} {save_to}",
        content=md, diff=_unified_diff(old, md, save_to))
    if held is not None:
        return (f"(Awaiting your approval to save the plan to {save_to}; "
                "nothing written.)")
    try:
        target.write_text(md)
    except OSError as exc:
        return f"(Save failed: {exc}.)"
    return f"(Saved the plan to {save_to}.)"


_ALL_TOOLS: list[BaseTool] = [
    WorkPlanTool(),
    UnderstandTaskTool(),
    AggregateDataTool(),
    EdaProfileTool(),
    QualityCheckTool(),
    DeepEdaTool(),
    DataAnalysisAgentTool(),
    MLStrategyAgentTool(),
    InferTargetTool(),
    MutualInfoTool(),
    FeatureEngineeringTool(),
    TrainModelTool(),
    VisualizeTool(),
    ExecuteCodeTool(),
    BuildNotebookTool(),
    ReadInstructionsTool(),
    WebResearchTool(),
    ConnectDataTool(),
    ExploreDirectoryTool(),
    ReadFileTool(),
    SearchFilesTool(),
    WriteFileTool(),
    EditFileTool(),
    SqlQueryTool(),
    WriteSqlTool(),
    DeployModelTool(),
    ScaffoldCICDTool(),
]

TOOL_REGISTRY: dict[str, BaseTool] = {t.name: t for t in _ALL_TOOLS}


# Tools whose output is evidence about the data, as opposed to evidence about
# the task or the filesystem. Only these count as "EDA has happened".
# execute_code is included because the model designs most of its EDA there
# (group comparisons, ratios, concentration); TOOLS_RUN records successes only.
_EDA_TOOLS = frozenset({
    "eda_profile", "deep_eda", "quality_check", "mutual_information",
    "data_analysis_agent", "execute_code",
})


# ── EDA agenda: the plan's questions, tracked open or answered ──────────────
#
# The plan lists the questions EDA must settle. Shown as plain text they were
# never checked off, so the loop could not tell what was left and wandered into
# unrelated work. The agenda keeps each question with an id, a status and the
# finding that settled it; the critic updates it after each step, and a result
# that raises a new question adds it as a lead.

_AGENDA_MAX = 18
_LEADS_MAX = 6


def _agenda(ns: dict) -> list[dict]:
    agenda = ns.get("EDA_AGENDA")
    return agenda if isinstance(agenda, list) else []


_STOPWORDS = frozenset(
    "a across an and any all are as at be between by do does each for from has "
    "have how in is it of on or per that the there this to what which with "
    "within".split())


def _words(text: str) -> frozenset:
    """Content words, lightly stemmed so "mapped" matches "mapping". Column
    names (with underscores) are kept whole: they are what a question is
    about."""
    import re as _re
    out = set()
    for w in _re.findall(r"[a-z0-9_]+", text.lower()):
        if w in _STOPWORDS:
            continue
        out.add(w if "_" in w or len(w) <= 4 else w[:4])
    return frozenset(out)


def _is_repeat(text: str, agenda: list[dict]) -> bool:
    """True when a question restates one already on the agenda, judged by the
    share of the shorter question's content words the two have in common. A
    revised plan tends to re-add "Is provider_id consistently mapped to
    provider_name?" for "Is there a consistent mapping between provider_id and
    provider_name?"."""
    words = _words(text)
    cols = {w for w in words if "_" in w}
    for a in agenda:
        other = _words(str(a.get("question", "")))
        other_cols = {w for w in other if "_" in w}
        if cols and other_cols and cols != other_cols:
            continue          # about different columns: a different question
        small = min(len(words), len(other))
        if small >= 3 and len(words & other) / small >= 0.6:
            return True
    return False


def sync_agenda(ns: dict, plan: dict) -> list[dict]:
    """Add the plan's EDA questions to the agenda, keeping what is answered."""
    agenda = _agenda(ns)
    for source, key in (("brief", "eda_questions_task"),
                        ("check", "eda_questions_independent")):
        for q in plan.get(key) or []:
            text = str(q).strip() if isinstance(q, str) else ""
            if not text or len(agenda) >= _AGENDA_MAX or _is_repeat(text, agenda):
                continue
            agenda.append({"id": f"Q{len(agenda) + 1}", "question": text[:300],
                           "source": source, "status": "open", "finding": ""})
    ns["EDA_AGENDA"] = agenda
    return agenda


def record_findings(ns: dict, ids: list, finding: str) -> list[str]:
    """Mark agenda items answered with the finding that settled them."""
    finding = (finding or "").strip()
    if not finding:
        return []
    wanted = {str(i).strip().upper() for i in ids or []}
    done = []
    for item in _agenda(ns):
        if item.get("id", "").upper() in wanted:
            item["status"] = "answered"
            item["finding"] = finding[:400]
            done.append(item["id"])
    return done


def add_lead(ns: dict, question: str) -> str | None:
    """Add a follow-up question a result raised; bounded so leads cannot crowd
    out the plan's own questions."""
    text = (question or "").strip()
    agenda = _agenda(ns)
    if (not text or len(agenda) >= _AGENDA_MAX
            or sum(a.get("source") == "lead" for a in agenda) >= _LEADS_MAX
            or _is_repeat(text, agenda)):
        return None
    item = {"id": f"Q{len(agenda) + 1}", "question": text[:300],
            "source": "lead", "status": "open", "finding": ""}
    agenda.append(item)
    ns["EDA_AGENDA"] = agenda
    return item["id"]


def eda_has_run(kernel) -> bool:
    """True when an EDA tool has actually produced evidence about the data.

    Deliberately not `EXPLORATION_LOG` truthiness, which is what this used to
    be. The critic writes that log after ANY tool it judges, and it only skips
    the retrieval tools, so `understand_task` alone was enough to make a plan
    look stale: the gate fired before a single row had been profiled, and
    work_plan(revise=true) then "revised against EDA findings" that contained
    no EDA. TOOLS_RUN is the critic-independent record, so this also works with
    DSAGENT_CRITIC=0, where EXPLORATION_LOG is never written at all.
    """
    ns = getattr(kernel, "namespace", {}) if kernel is not None else {}
    if ns.get("DATA_ANALYSIS_REPORT") or ns.get("MI_SCORES"):
        return True
    run = ns.get("TOOLS_RUN")
    if isinstance(run, list) and any(t in _EDA_TOOLS for t in run):
        return True
    log = ns.get("EXPLORATION_LOG")
    if isinstance(log, list):
        tags = tuple(f"[{t}]" for t in _EDA_TOOLS)
        return any(isinstance(x, str) and x.startswith(tags) for x in log)
    return False


def work_plan_is_stale(kernel) -> bool:
    """True when EDA has produced findings the plan has not been revised for.

    A plan written before any data was profiled is a guess about feature
    engineering and model choice; this is what forces it to be revisited.
    """
    ns = getattr(kernel, "namespace", {}) if kernel is not None else {}
    plan = ns.get("WORK_PLAN")
    if not isinstance(plan, dict):
        return False          # no plan at all is a separate problem
    if not eda_has_run(kernel):
        return False          # nothing to revise against yet
    return not plan.get("revised_after_eda", False)


def get_tool_schemas() -> list[dict]:
    """Return MCP-style tool definitions for the planner prompt."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
        }
        for t in _ALL_TOOLS
    ]
