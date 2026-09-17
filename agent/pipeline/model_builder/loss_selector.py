"""LossSelector — map task type (and imbalance ratio) to a loss + rationale."""
from __future__ import annotations

from agent.core.types import ToolResult
from agent.pipeline.model_builder.task_detector import TaskType


class LossSelector:
    """Principled loss-function selection given task and class imbalance."""

    def select(self, task: TaskType, class_imbalance_ratio: float = 1.0) -> ToolResult:
        if class_imbalance_ratio <= 0:
            return ToolResult(status="error", data=None,
                              explanation="class_imbalance_ratio must be > 0.",
                              math_trace="", error="ValueError")

        rationale: str
        math_trace: str
        if task == "binary_classification":
            if class_imbalance_ratio > 5:
                loss = "focal"
                rationale = (f"Imbalance {class_imbalance_ratio:.1f}:1 ≥ 5:1 → "
                             "focal down-weights easy negatives.")
                math_trace = "FL(p) = −(1 − p_t)^γ · log p_t, γ=2."
            elif class_imbalance_ratio > 2:
                loss = "weighted_bce"
                rationale = f"Imbalance {class_imbalance_ratio:.1f}:1 > 2:1 → class-weighted BCE."
                math_trace = "BCE_w = −[w₁·y·log p + w₀·(1−y)·log(1−p)], w_k ∝ 1/n_k."
            else:
                loss = "bce"
                rationale = "Near-balanced classes → vanilla binary cross-entropy."
                math_trace = "BCE = −[y·log p + (1−y)·log(1−p)]."
        elif task == "multiclass_classification":
            if class_imbalance_ratio > 2:
                loss = "weighted_cross_entropy"
                rationale = f"Imbalance {class_imbalance_ratio:.1f}:1 → class-weighted CE."
            else:
                loss = "cross_entropy"
                rationale = "Balanced classes → standard cross-entropy."
            math_trace = "CE = −Σ_k y_k · log p_k; w_k ∝ 1/n_k if weighted."
        elif task == "regression":
            loss = "huber"
            rationale = "Huber is robust to outliers while smooth near zero."
            math_trace = ("Huber_δ(r) = ½r² if |r|≤δ else δ(|r|−½δ); "
                          "quadratic centre + linear tails.")
        elif task == "timeseries_forecasting":
            loss = "quantile"
            rationale = "Quantile (pinball) loss produces calibrated forecast intervals."
            math_trace = "ρ_τ(r) = max(τ·r, (τ−1)·r)."
        elif task == "anomaly_detection":
            loss = "reconstruction_mse"
            rationale = "Anomalies flagged by reconstruction error magnitude."
            math_trace = "MSE = ‖x − x̂‖² / d."
        elif task == "clustering":
            loss = "silhouette"
            rationale = "Silhouette optimises intra-vs-inter cluster separation."
            math_trace = "s(i) = (b(i) − a(i)) / max(a(i), b(i))."
        elif task == "survival_analysis":
            loss = "cox_partial_likelihood"
            rationale = "Cox partial-likelihood handles right-censored events."
            math_trace = "L = Π_i exp(βᵀx_i) / Σ_{j∈R(t_i)} exp(βᵀx_j)."
        elif task == "causal_inference":
            loss = "doubly_robust"
            rationale = "DR combines outcome regression with propensity weighting."
            math_trace = "ψ = m̂₁(X) − m̂₀(X) + [A(Y − m̂₁(X))/ê(X)] − [(1−A)(Y − m̂₀(X))/(1−ê(X))]."
        else:
            return ToolResult(status="error", data=None,
                              explanation=f"Unknown task {task!r}.",
                              math_trace="", error="UnknownTask")

        return ToolResult(
            status="ok",
            data={"loss": loss, "task": task,
                  "class_imbalance_ratio": float(class_imbalance_ratio),
                  "rationale": rationale},
            explanation=f"Selected loss {loss!r}: {rationale}",
            math_trace=math_trace,
        )
