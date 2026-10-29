"""HPOptimiser — thin Optuna wrapper returning a ToolResult.

Search-space format::

    {"lr": ("loguniform", 1e-5, 1e-1),
     "depth": ("int", 2, 10),
     "estimator": ("categorical", ["rf", "gbm"]),
     "alpha": ("uniform", 0.0, 1.0)}
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from agent.core.types import ToolResult

_ALLOWED = {"uniform", "loguniform", "int", "categorical"}


class HPOptimiser:
    """Bayesian HPO over a search space, minimising or maximising an objective."""

    def __init__(self, direction: Literal["minimize", "maximize"] = "minimize",
                 seed: int = 0) -> None:
        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        self._direction = direction
        self._seed = int(seed)

    @staticmethod
    def _suggest(trial, name: str, spec: tuple) -> Any:
        kind = spec[0]
        if kind not in _ALLOWED:
            raise ValueError(f"unknown distribution {kind!r}")
        if kind == "uniform":
            return trial.suggest_float(name, spec[1], spec[2])
        if kind == "loguniform":
            return trial.suggest_float(name, spec[1], spec[2], log=True)
        if kind == "int":
            return trial.suggest_int(name, spec[1], spec[2])
        # categorical
        return trial.suggest_categorical(name, list(spec[1]))

    def optimise(
        self,
        objective: Callable[[dict], float],
        search_space: dict[str, tuple],
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> ToolResult:
        if not search_space:
            return ToolResult(status="error", data=None,
                              explanation="search_space must be non-empty.",
                              math_trace="", error="ValueError")
        if n_trials < 1:
            return ToolResult(status="error", data=None,
                              explanation="n_trials must be ≥ 1.",
                              math_trace="", error="ValueError")
        try:
            import optuna
        except ImportError as exc:
            return ToolResult(status="error", data=None,
                              explanation="optuna not installed.",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")

        # Validate specs up-front so bad specs error cleanly.
        for name, spec in search_space.items():
            if not isinstance(spec, tuple) or len(spec) < 2 or spec[0] not in _ALLOWED:
                return ToolResult(status="error", data=None,
                                  explanation=f"bad search spec for {name!r}: {spec!r}",
                                  math_trace="", error="ValueError")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study = optuna.create_study(direction=self._direction, sampler=sampler)

        def _wrapped(trial):
            params = {n: self._suggest(trial, n, s) for n, s in search_space.items()}
            return float(objective(params))

        study.optimize(_wrapped, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False, n_jobs=1)
        best = study.best_trial
        history = [{"number": t.number, "value": t.value, "params": t.params}
                   for t in study.trials]
        return ToolResult(
            status="ok",
            data={"best_params": dict(best.params),
                  "best_value": float(best.value) if best.value is not None else float("nan"),
                  "n_trials": len(study.trials),
                  "direction": self._direction,
                  "history": history},
            explanation=(
                f"Optuna {self._direction}d objective over {len(study.trials)} "
                f"trial(s); best={best.value:.6g}."
            ),
            math_trace=(
                "x* = arg{min|max}_x f(x); "
                "TPE samples p(x|y<y*) / p(x|y≥y*) (Bergstra 2011)."
            ),
        )
