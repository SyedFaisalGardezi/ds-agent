"""Run the full ds-agent pipeline on a registered dataset.

Calls the pipeline stages **in-process** (not via HTTP) to collect
every artifact a notebook report needs, then hands them to
`notebook_builder.build`. Designed to be invoked from the chat router
when a user asks to "build the notebook".
"""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.api import notebook_builder
from agent.api.sessions import Session
from agent.monitoring.drift_detector import DriftDetector
from agent.pipeline.eda.profiler import DataProfiler
from agent.pipeline.eda.quality_scorer import QualityScorer
from agent.pipeline.eda.statistics import StatisticsAnalyser
from agent.pipeline.evaluator.leaderboard import Leaderboard
from agent.pipeline.feature_engineering.generator import FeatureGenerator
from agent.pipeline.feature_engineering.selector import FeatureSelector
from agent.pipeline.model_builder.task_detector import TaskDetector
from agent.pipeline.model_builder.trainer import ModelTrainer

_NOTEBOOK_DIR = Path("outputs/notebooks")


def _sanitize(obj: Any) -> Any:
    """Numpy-safe, NaN-safe JSON coercion."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.head(50).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _load(data_path: Path, sample: int | None = None) -> pd.DataFrame:
    """Load any supported format into a DataFrame via the universal data loader."""
    from agent.api.data_loader import load as _dl_load
    result = _dl_load(data_path)
    if result.error and result.df is None:
        raise ValueError(result.error)
    df = result.df
    if df is None:
        raise ValueError(
            f"Could not produce a tabular DataFrame from {data_path} "
            f"(type={result.data_type}, fmt={result.format_name}). "
            f"The code agent can still work with this file — just ask."
        )
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)
    return df


_TARGET_PROMPT = """\
/no_think
You are a senior data scientist. Pick the single best target column for a machine \
learning task from the dataset below.

Dataset columns and statistics:
{col_summary}

Project brief / instructions:
{brief}

Selection rules (in priority order):
1. If the brief EXPLICITLY names a target column, use exactly that name.
2. Prefer outcome/result/status/flag columns — things that describe what happened \
   or the end state of an event (e.g. incident_outcome, churn, default, survived).
3. Prefer low cardinality (2–30 unique values) for classification, or a continuous \
   numeric column for regression.
4. Avoid: ID columns, free-text columns, date/timestamp columns, columns whose \
   name contains 'id', 'date', 'time', 'month', 'year', 'created', 'updated'.
5. If multiple columns look equally valid, pick the one most central to the \
   business question described in the brief.

You MUST pick one column from the list. Do NOT say "unknown" or refuse. \
Reply with ONLY the column name — no explanation, no punctuation, nothing else.
"""

# Keywords that strongly suggest a target column (checked in order).
_TARGET_KEYWORDS = [
    "outcome", "result", "target", "label", "class", "status", "flag",
    "response", "dependent", "death", "survived", "churn", "fraud", "default",
    "failure", "success", "risk", "score", "grade", "category", "incident_outcome",
    "y", "output", "indicator", "event",
]


def _heuristic_target(df: pd.DataFrame) -> str | None:
    """Keyword-based fallback — used when the LLM cannot decide."""
    cols_lower = {c.lower(): c for c in df.columns}
    skip = {"id", "date", "time", "month", "year", "created", "updated", "index"}

    def _ok(col: str) -> bool:
        lo = col.lower()
        if any(s in lo for s in skip):
            return False
        n = df[col].nunique()
        return 2 <= n <= 50

    # 1. Exact keyword hit
    for kw in _TARGET_KEYWORDS:
        if kw in cols_lower and _ok(cols_lower[kw]):
            return cols_lower[kw]
    # 2. Partial keyword hit
    for kw in _TARGET_KEYWORDS:
        for lo, orig in cols_lower.items():
            if kw in lo and _ok(orig):
                return orig
    # 3. Lowest-cardinality column that looks like a label
    candidates = [c for c in df.columns if _ok(c)]
    if candidates:
        return min(candidates, key=lambda c: df[c].nunique())
    return None


def _llm_infer_target(df: pd.DataFrame, brief: str, llm_client: Any) -> str | None:
    """Use LLM to pick a target column; fall back to heuristics if it hedges."""
    col_lines = []
    for col in df.columns:
        s = df[col]
        sample_vals = s.dropna().unique()[:5].tolist()
        col_lines.append(
            f"  {col}: dtype={s.dtype}, n_unique={s.nunique()}, "
            f"null%={s.isna().mean():.1%}, sample={sample_vals}"
        )
    col_summary = "\n".join(col_lines)
    brief_snip = (brief[:3000] if brief else
                  "(no brief — infer from column names and statistics)")
    prompt = _TARGET_PROMPT.format(col_summary=col_summary, brief=brief_snip)

    if llm_client is not None:
        # Try twice — first with temperature 0, then 0.1 if the first reply is bad.
        for temp in (0.0, 0.1):
            try:
                raw = llm_client._generate(prompt, temperature=temp, max_tokens=32)
                raw = llm_client._strip_thinking(raw).strip().strip("`\"' \n")
                # Take first token in case the model added punctuation
                candidate = raw.split()[0] if raw.split() else ""
                candidate = candidate.strip(".,;:'\"")
                if candidate and candidate.lower() != "unknown" and candidate in df.columns:
                    return candidate
            except Exception:  # noqa: BLE001
                break

    # Heuristic fallback — never leave the user in EDA-only mode.
    return _heuristic_target(df)


def run_full_pipeline(
    session: Session,
    *,
    leaderboard: Leaderboard,
    sample: int | None = 20_000,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Run every stage; write a notebook; mutate `session.artifacts`.

    Returns a summary dict suitable for assistant chat reply.
    """
    if session.data_path is None:
        return {"error": "no data file registered. POST /chat/sessions/{sid}/data first."}
    target = session.target  # may be None → EDA-only mode

    run_id = f"chat-{session.session_id}-{uuid.uuid4().hex[:6]}"
    session.run_id = run_id
    artifacts: dict[str, Any] = {}
    started = time.time()

    try:
        df = _load(session.data_path, sample=sample)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed to read data: {exc}"}

    if target is not None and target not in df.columns:
        return {"error": f"target {target!r} not in columns {list(df.columns)[:20]}..."}

    # Drop types the downstream stages can't handle cleanly.
    df = df.select_dtypes(exclude=["datetime64[ns]", "timedelta64[ns]"])

    # 1. profile
    prof = DataProfiler().run(df, run_id=run_id)
    if prof.status == "ok":
        artifacts["profile"] = _sanitize(prof.data)

    # 2. quality
    qual = QualityScorer().score(df)
    if qual.status in {"ok", "warning"}:
        artifacts["quality"] = _sanitize(qual.data)

    # 2b. LLM-driven target inference (if target not set by user).
    #     Runs after profiling so the LLM sees real column stats.
    if target is None and llm_client is not None:
        inferred = _llm_infer_target(df, session.brief, llm_client)
        if inferred:
            target = inferred
            session.target = inferred
            artifacts["target_inference"] = {
                "method": "llm",
                "inferred_target": inferred,
                "note": (f"Target `{inferred}` was automatically selected by the LLM "
                         "based on the data profile and project brief."),
            }

    eda_only = target is None
    if eda_only:
        artifacts["mode"] = "EDA-only (LLM could not infer target — MI, feature engineering, training skipped)"

    augmented = df.copy()

    if not eda_only:
        # 3. mutual information
        try:
            mi = StatisticsAnalyser().mutual_information(df, target=target)
            if mi.status == "ok":
                artifacts["mi"] = _sanitize(mi.data)
        except Exception as exc:  # noqa: BLE001
            artifacts["mi_error"] = str(exc)

        # 4. feature engineering
        try:
            gen = FeatureGenerator(max_pairs=5, max_onehot_cardinality=8)
            feat = gen.generate(df, target=target)
            if feat.status == "ok":
                augmented = feat.data["frame"]
                artifacts["features"] = {
                    "frame_shape": list(augmented.shape),
                    "new_features": [c for c in augmented.columns
                                     if c not in df.columns][:30],
                }
        except Exception as exc:  # noqa: BLE001
            artifacts["features_error"] = str(exc)

        # 4b. feature selection — keep top 40 features.
        if target in augmented.columns:
            try:
                sel_result = FeatureSelector().shap_selection(
                    augmented, target=target, top_k=40)
                if sel_result.status == "ok":
                    keep = [c for c in sel_result.data.get("selected", [])
                            if c in augmented.columns]
                    if keep:
                        augmented = augmented[keep + [target]]
                        artifacts["selected_features"] = {
                            "method": sel_result.data.get("method", "?"),
                            "n_selected": len(keep),
                            "selected": keep[:30],
                        }
            except Exception as exc:  # noqa: BLE001
                artifacts["selection_error"] = str(exc)

        # 5. task detection
        task = TaskDetector().infer(augmented, target=target)
        if task.status == "ok":
            artifacts["task"] = _sanitize(task.data)
            task_type = task.data.get("task", "binary_classification")
        else:
            task_type = "binary_classification"

        # 6. train
        try:
            trainer = ModelTrainer(
                task_type=task_type,
                config={"target": target, "estimator": "rf",
                        "n_estimators": 200, "test_size": 0.25, "random_state": 0},
            )
            trained = trainer.run(augmented, run_id=run_id, work_dir=session.work_dir)
            if trained.status == "ok":
                metrics = trained.data.get("metrics", {}) or {}
                artifacts["train"] = _sanitize({
                    "run_id": run_id,
                    "task": task_type,
                    "metrics": metrics,
                    "feature_names": trained.data.get("feature_names", [])[:100],
                    "n_train": trained.data.get("n_train"),
                    "n_test": trained.data.get("n_test"),
                    "model_path": trained.data.get("model_path"),
                })
                leaderboard.log_run(
                    run_id=run_id, model_type="rf",
                    hyperparams={"n_estimators": 200, "test_size": 0.25,
                                 "random_state": 0},
                    metrics=metrics, train_time_s=time.time() - started,
                    inference_ms=0.0, model_mb=0.0, shap_runtime_s=0.0,
                )
        except Exception as exc:  # noqa: BLE001
            artifacts["train_error"] = str(exc)

        # 7. drift: first half vs second half
        try:
            feat_only = augmented.drop(columns=[target], errors="ignore")
            half = len(feat_only) // 2
            if half > 10:
                drift = DriftDetector().check_all_features(
                    feat_only.iloc[:half], feat_only.iloc[half:])
                if drift.status == "ok":
                    artifacts["drift"] = _sanitize(drift.data)
        except Exception as exc:  # noqa: BLE001
            artifacts["drift_error"] = str(exc)

    # 8. notebook — write into session.work_dir when set, else default.
    import json as _json
    out_dir = session.work_dir if session.work_dir else _NOTEBOOK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_id}.ipynb"
    notebook_builder.build(
        data_path=session.data_path,
        target=target or "",
        brief=session.brief,
        artifacts=artifacts,
        out_path=out,
    )
    # Also dump artifacts alongside the notebook for downstream tooling.
    artifacts_path = out_dir / f"{run_id}_artifacts.json"
    try:
        artifacts_path.write_text(_json.dumps(artifacts, indent=2, default=str))
    except Exception:  # noqa: BLE001
        artifacts_path = None

    session.notebook_path = out
    session.artifacts.update(artifacts)

    mode = "EDA-only" if eda_only else "full pipeline"
    train_info = artifacts.get("train", {})
    inferred = artifacts.get("target_inference", {})
    target_note = (
        f" Target `{inferred['inferred_target']}` was automatically selected from EDA + brief."
        if inferred else ""
    )
    if not eda_only and train_info:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in train_info.get("metrics", {}).items())
        pipeline_summary = (
            f"Task: **{train_info.get('task')}** | "
            f"Best model: **{train_info.get('estimator', 'RandomForest')}** | "
            f"Metrics: {metrics_str} | "
            f"n_train={train_info.get('n_train')}, n_test={train_info.get('n_test')}"
        )
    else:
        pipeline_summary = (
            "Profile and quality analysis done."
            if eda_only else "Pipeline complete."
        )
    return {
        "run_id": run_id,
        "mode": mode,
        "task": train_info.get("task") if not eda_only else "eda_only",
        "target": target,
        "metrics": train_info.get("metrics", {}),
        "model_path": train_info.get("model_path"),
        "notebook": str(out),
        "work_dir": str(out_dir),
        "artifacts_file": str(artifacts_path) if artifacts_path else None,
        "elapsed_s": round(time.time() - started, 2),
        "artifacts": sorted(artifacts.keys()),
        "reply": (
            f"Pipeline complete ({mode}).{target_note}\n\n"
            + pipeline_summary
            + f"\n\nNotebook saved to `{out}`."
        ),
    }
