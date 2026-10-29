"""Shared split-first training; preprocessing travels with the exported model."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def dataset_hash(df: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(json.dumps([(str(c), str(t)) for c, t in df.dtypes.items()]).encode())
    return digest.hexdigest()


def estimator(name: str, classification: bool, *, seed: int, balanced: bool) -> Any:
    cw = "balanced" if balanced else None
    catalog = {
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed,
                                      class_weight=cw) if classification else
              RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, n_jobs=1, random_state=seed,
                                            class_weight=cw) if classification else
                       ExtraTreesRegressor(n_estimators=100, n_jobs=1, random_state=seed),
        "lr": LogisticRegression(max_iter=2000, class_weight=cw) if classification else Ridge(),
        "linear": LogisticRegression(max_iter=2000, class_weight=cw) if classification else Ridge(),
        "ridge": Ridge(), "elasticnet": ElasticNet(max_iter=5000), "lasso": Lasso(max_iter=5000),
    }
    if name == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        return (LGBMClassifier(class_weight=cw, verbosity=-1, n_jobs=1, random_state=seed)
                if classification else LGBMRegressor(verbosity=-1, n_jobs=1, random_state=seed))
    if name == "xgb":
        # Class weights are supplied as sample weights during fit/CV below.
        from xgboost import XGBClassifier, XGBRegressor
        return (XGBClassifier(n_jobs=1, random_state=seed) if classification else
                XGBRegressor(n_jobs=1, random_state=seed))
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        return (GradientBoostingClassifier(random_state=seed) if classification else
                GradientBoostingRegressor(random_state=seed))
    if name not in catalog or (classification and name in {"ridge", "elasticnet", "lasso"}):
        raise ValueError(f"Unsupported model for this task: {name}")
    return catalog[name]


def train(df: pd.DataFrame, *, target: str, task_type: str,
          models: list[str], leakage: list[str] | None = None,
          metric: str | None = None, group: str | None = None,
          time_column: str | None = None, test_size: float = 0.2,
          seed: int = 42, log_run: bool = True) -> dict[str, Any]:
    if target not in df or df[target].nunique() < 2:
        raise ValueError("A nonconstant, explicitly identified target is required.")
    classification = task_type in {"classification", "binary_classification", "multiclass_classification"}
    if not classification and task_type != "regression":
        raise ValueError("Specify classification or regression; targets are not guessed from uniqueness.")
    frame = df.loc[df[target].notna()].copy()
    if len(frame) < 12:
        raise ValueError("At least 12 labelled rows are required for selection and held-out evaluation.")
    if group and time_column:
        raise ValueError("Combined group/temporal evaluation needs an explicit split design.")
    if time_column:
        frame = frame.sort_values(time_column, kind="stable")
    excluded = {target, *(leakage or []), *([group] if group else []),
                *([time_column] if time_column else [])}
    X = frame.drop(columns=[c for c in excluded if c in frame]).copy()
    if X.empty or any(pd.api.types.is_datetime64_any_dtype(t) for t in X.dtypes):
        raise ValueError("Provide usable predictors; explicitly transform date features before modelling.")
    # Normalize categoricals without learning categories from held-out rows.
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    for c in categorical:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    y = frame[target]
    indices = np.arange(len(frame))
    if time_column:
        boundary = int(len(frame) * (1 - test_size))
        tr, te = indices[:boundary], indices[boundary:]
        if frame.iloc[tr][time_column].max() >= frame.iloc[te][time_column].min():
            raise ValueError("Temporal boundary contains overlapping timestamps; choose a distinct cutoff.")
        cv: Any = TimeSeriesSplit(n_splits=3)
    elif group:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                         random_state=seed).split(X, y, frame[group]))
        cv = GroupKFold(n_splits=min(5, frame.iloc[tr][group].nunique()))
    else:
        tr, te = train_test_split(indices, test_size=test_size, random_state=seed,
                                  stratify=y if classification else None)
        splits = min(5, int(y.iloc[tr].value_counts().min())) if classification else 5
        if splits < 2:
            raise ValueError("Insufficient examples per class for validation.")
        cv = StratifiedKFold(splits, shuffle=True, random_state=seed) if classification else KFold(5, shuffle=True, random_state=seed)
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(keep_empty_features=True)),
                               ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                   ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    counts = y.iloc[tr].value_counts()
    balanced = classification and counts.min() / counts.max() < 0.4
    scoring = metric if metric not in {None, "auto", "?"} else (
        "f1_weighted" if classification else "neg_root_mean_squared_error")
    if scoring == "pr_auc":
        scoring = "average_precision"
    candidates, failures, scores = {}, {}, {}
    for name in dict.fromkeys(models):
        try:
            est = estimator(name, classification, seed=seed, balanced=balanced)
            if balanced and name in {"xgb", "gbm"}:
                raise ValueError("This estimator requires fold-local sample weighting; use rf/lgbm/lr.")
            if classification and name == "xgb" and sorted(y.unique().tolist()) != list(range(y.nunique())):
                raise ValueError("XGBoost requires explicit numeric label mapping; use rf/lgbm/lr.")
            pipe = Pipeline([("preprocess", clone(pre)), ("model", est)])
            values = cross_val_score(pipe, X.iloc[tr], y.iloc[tr], cv=cv,
                groups=frame.iloc[tr][group] if group else None,
                scoring=scoring, n_jobs=1, error_score="raise")
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite validation scores")
            scores[name] = values.tolist()
            candidates[name] = pipe
        except Exception as exc:
            failures[name] = str(exc)
    if not candidates:
        raise ValueError(f"No model completed validation: {failures}")
    best = max(scores, key=lambda n: float(np.mean(scores[n])))
    evaluation_model = clone(candidates[best]).fit(X.iloc[tr], y.iloc[tr])
    pred = evaluation_model.predict(X.iloc[te])
    metrics = ({"accuracy": float(accuracy_score(y.iloc[te], pred)),
                "f1_weighted": float(f1_score(y.iloc[te], pred, average="weighted"))}
               if classification else {"rmse": float(np.sqrt(mean_squared_error(y.iloc[te], pred))),
                                        "r2": float(r2_score(y.iloc[te], pred))})
    if classification:
        metrics["f1_macro"] = float(f1_score(y.iloc[te], pred, average="macro"))
        if y.nunique() == 2 and y.iloc[te].nunique() == 2:
            labels = evaluation_model.classes_
            metrics["roc_auc"] = float(roc_auc_score(y.iloc[te] == labels[1],
                                                       evaluation_model.predict_proba(X.iloc[te])[:, 1]))
    if not all(np.isfinite(list(metrics.values()))):
        raise ValueError("Nonfinite hold-out metrics")
    # A distinct estimator: evaluation artifacts are not overwritten by refitting.
    model = clone(candidates[best]).fit(X, y)
    result = {"model": model, "best_name": best, "metrics": metrics,
              "cv_scores": scores, "failed_models": failures, "scoring": scoring,
              "feature_names": X.columns.tolist(), "task": task_type,
              "target": target, "dataset_hash": dataset_hash(df), "seed": seed,
              "train_indices": tr.tolist(), "test_indices": te.tolist(),
              "n_train": len(tr), "n_test": len(te), "n_refit": len(X),
              "dropped_missing_labels": len(df) - len(frame),
              "split": "time" if time_column else "group" if group else "random",
              "math_trace": r"$\hat\theta=\arg\min_\theta\sum_{i\in train}\ell(y_i,f_\theta(x_i))$. "
                            r"Preprocessing is fitted within each fold; $test\cap train=\varnothing$.",
              "model_path": None}
    if log_run:
        result["model_path"] = persist_run(df, result)
    return result


def persist_run(df: pd.DataFrame, result: dict[str, Any]) -> str:
    """Managed internal run store; never publish directly to a user's workspace."""
    import mlflow

    from agent.pipeline.etl.provenance import track_run
    root = Path(os.environ.get("DSAGENT_STATE_DIR", str(Path.home() / ".ds-agent")))
    folder = root / "runs" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    model_path = folder / "model.joblib"
    df.to_parquet(folder / "dataset.parquet", index=False)
    joblib.dump(result["model"], model_path)
    manifest = {k: v for k, v in result.items() if k != "model"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # A separate tracking URI avoids changing process-global MLflow configuration.
    client = mlflow.MlflowClient(tracking_uri=(root / "mlruns").as_uri())
    exp = client.get_experiment_by_name("ds-agent")
    exp_id = exp.experiment_id if exp else client.create_experiment("ds-agent")
    run = client.create_run(exp_id)
    try:
        for key, value in result["metrics"].items():
            client.log_metric(run.info.run_id, key, value)
        client.log_param(run.info.run_id, "dataset_hash", result["dataset_hash"])
        client.log_param(run.info.run_id, "model", result["best_name"])
        client.log_artifacts(run.info.run_id, str(folder))
        track_run(folder)
        client.set_terminated(run.info.run_id)
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    return str(model_path)
