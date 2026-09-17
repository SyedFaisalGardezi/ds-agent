"""ModelTrainer — sklearn-backed training stage.

Config keys (all optional):
  target: str                 — required unless task == 'clustering'
  estimator: str              — 'rf' | 'gbm' | 'linear' | 'kmeans' | 'iforest'
  test_size: float            — default 0.2
  random_state: int           — default 0
  n_estimators: int           — default 200
  n_clusters: int             — default 4  (clustering)
  contamination: float        — default 0.1 (anomaly)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineStage

_CLASSIF = {"binary_classification", "multiclass_classification"}
_MODEL_DIR = __import__("pathlib").Path("outputs/models")


class ModelTrainer(PipelineStage):
    """Train a model appropriate to ``task_type`` and report metrics."""

    name = "train"

    def __init__(self, task_type: str, config: dict[str, Any] | None = None) -> None:
        self._task_type = task_type
        self._config = config or {}

    # ---- helpers ----------------------------------------------------- #

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        target = self._config.get("target")
        if not target:
            raise KeyError("config.target is required for supervised tasks")
        if target not in df.columns:
            raise KeyError(f"target {target!r} not in frame")
        X = df.drop(columns=[target]).select_dtypes("number").fillna(0.0)
        y = df[target]
        return X, y

    def _pick_supervised(self, is_classif: bool):
        name = self._config.get("estimator", "rf")
        n_est = int(self._config.get("n_estimators", 200))
        rs = int(self._config.get("random_state", 0))
        if name == "rf":
            return (RandomForestClassifier if is_classif else RandomForestRegressor)(
                n_estimators=n_est, random_state=rs, n_jobs=1,
            )
        if name == "gbm":
            return (GradientBoostingClassifier if is_classif else GradientBoostingRegressor)(
                n_estimators=n_est, random_state=rs,
            )
        if name == "linear":
            return (LogisticRegression(max_iter=500, random_state=rs)
                    if is_classif else Ridge(random_state=rs))
        raise ValueError(f"unknown estimator {name!r}")

    # ---- run --------------------------------------------------------- #

    def run(self, df: pd.DataFrame, run_id: str,
            work_dir: __import__("pathlib").Path | None = None) -> ToolResult:
        try:
            if self._task_type in _CLASSIF or self._task_type == "regression":
                return self._supervised(df, run_id=run_id, work_dir=work_dir)
            if self._task_type == "clustering":
                return self._clustering(df, run_id=run_id, work_dir=work_dir)
            if self._task_type == "anomaly_detection":
                return self._anomaly(df, run_id=run_id, work_dir=work_dir)
            return ToolResult(status="error", data=None,
                              explanation=f"Unsupported task {self._task_type!r}.",
                              math_trace="", error="UnsupportedTask")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"Training failed: {exc}",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")

    def _save_model(self, model: object, run_id: str,
                    work_dir: __import__("pathlib").Path | None) -> str | None:

        import joblib
        out_dir = (work_dir / "models") if work_dir else _MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}_{type(model).__name__}.pkl"
        try:
            joblib.dump(model, path)
            return str(path)
        except Exception:
            return None

    def _supervised(self, df: pd.DataFrame, run_id: str = "run",
                    work_dir: __import__("pathlib").Path | None = None) -> ToolResult:
        X, y = self._split_xy(df)
        is_classif = self._task_type in _CLASSIF
        y_fit = pd.factorize(y)[0] if is_classif and y.dtype == object else y
        if is_classif and not np.issubdtype(np.asarray(y_fit).dtype, np.number):
            y_fit = pd.factorize(y_fit)[0]
        test_size = float(self._config.get("test_size", 0.2))
        rs = int(self._config.get("random_state", 0))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.to_numpy(), np.asarray(y_fit),
            test_size=test_size, random_state=rs,
            stratify=np.asarray(y_fit) if is_classif else None,
        )
        model = self._pick_supervised(is_classif)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        metrics: dict[str, float] = {}
        if is_classif:
            metrics["accuracy"] = float(accuracy_score(y_te, pred))
            metrics["f1_macro"] = float(f1_score(y_te, pred, average="macro"))
            if hasattr(model, "predict_proba") and len(np.unique(y_tr)) == 2:
                try:
                    proba = model.predict_proba(X_te)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
                except Exception:
                    pass
        else:
            metrics["mae"] = float(mean_absolute_error(y_te, pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_te, pred)))
            metrics["r2"] = float(r2_score(y_te, pred))
        model_path = self._save_model(model, run_id, work_dir)
        return ToolResult(
            status="ok",
            data={"model": model, "metrics": metrics,
                  "feature_names": list(X.columns),
                  "task": self._task_type, "n_train": int(len(y_tr)),
                  "n_test": int(len(y_te)),
                  "model_path": model_path},
            explanation=(
                f"Trained {type(model).__name__} ({self._task_type}); "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                + (f" | saved: {model_path}" if model_path else "")
            ),
            math_trace=(
                "θ̂ = argmin_θ Σ ℓ(y_i, f_θ(x_i)); "
                "metrics evaluated on held-out split."
            ),
        )

    def _clustering(self, df: pd.DataFrame, run_id: str = "run",
                    work_dir: __import__("pathlib").Path | None = None) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if X.shape[0] < 2 or X.shape[1] == 0:
            return ToolResult(status="error", data=None,
                              explanation="need ≥2 rows and ≥1 numeric feature.",
                              math_trace="", error="ValueError")
        k = int(self._config.get("n_clusters", 4))
        k = max(2, min(k, X.shape[0] - 1))
        model = KMeans(n_clusters=k, n_init=10,
                       random_state=int(self._config.get("random_state", 0)))
        labels = model.fit_predict(X.to_numpy())
        sil = float(silhouette_score(X.to_numpy(), labels)) if len(set(labels)) > 1 else 0.0
        model_path = self._save_model(model, run_id, work_dir)
        return ToolResult(
            status="ok",
            data={"model": model, "labels": labels.tolist(),
                  "metrics": {"silhouette": sil, "inertia": float(model.inertia_)},
                  "k": k, "task": "clustering", "model_path": model_path},
            explanation=f"KMeans(k={k}); silhouette={sil:.4f}.",
            math_trace="argmin_C Σ_k Σ_{x∈C_k} ‖x − μ_k‖²; s(i) = (b−a)/max(a,b).",
        )

    def _anomaly(self, df: pd.DataFrame, run_id: str = "run",
                 work_dir: __import__("pathlib").Path | None = None) -> ToolResult:
        X = df.select_dtypes("number").fillna(0.0)
        if X.shape[0] < 2:
            return ToolResult(status="error", data=None,
                              explanation="need ≥2 rows.",
                              math_trace="", error="ValueError")
        contam = float(self._config.get("contamination", 0.1))
        model = IsolationForest(
            contamination=contam,
            n_estimators=int(self._config.get("n_estimators", 200)),
            random_state=int(self._config.get("random_state", 0)), n_jobs=1,
        )
        preds = model.fit_predict(X.to_numpy())  # 1=normal, −1=outlier
        scores = (-model.score_samples(X.to_numpy())).tolist()
        n_out = int((preds == -1).sum())
        model_path = self._save_model(model, run_id, work_dir)
        return ToolResult(
            status="ok",
            data={"model": model,
                  "labels": (preds == -1).astype(int).tolist(),
                  "scores": scores,
                  "metrics": {"n_outliers": n_out, "outlier_rate": n_out / len(preds)},
                  "contamination": contam, "task": "anomaly_detection",
                  "model_path": model_path},
            explanation=f"IsolationForest flagged {n_out} / {len(preds)} outlier(s).",
            math_trace="score = 2^(−E[h(x)]/c(n)); path-length anomaly detector.",
        )
