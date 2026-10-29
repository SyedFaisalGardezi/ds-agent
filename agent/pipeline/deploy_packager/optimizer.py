"""ModelOptimizer — ONNX export, dynamic quantisation, pruning, benchmarking.

All heavy-weight deps (torch, onnx, onnxruntime) are imported lazily. If a
dep is missing we degrade to a clean error ToolResult rather than crashing.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from agent.core.types import ToolResult


class ModelOptimizer:
    # ---- ONNX --------------------------------------------------------- #

    def onnx_export(self, model, sample_input, output_path: str) -> ToolResult:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation="torch not installed",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        if not hasattr(model, "forward"):
            return ToolResult(status="error", data=None,
                              explanation="model must be a torch.nn.Module",
                              math_trace="", error="TypeError")
        if not isinstance(sample_input, torch.Tensor):
            sample_input = torch.tensor(np.asarray(sample_input), dtype=torch.float32)
        model.eval()
        try:
            torch.onnx.export(
                model, sample_input, str(out),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"onnx export failed: {exc}",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        size = out.stat().st_size
        return ToolResult(
            status="ok",
            data={"output_path": str(out), "size_bytes": int(size)},
            explanation=f"Exported ONNX to {out} ({size} bytes).",
            math_trace="torch → ONNX graph (opset 17).",
        )

    # ---- quantise ----------------------------------------------------- #

    def dynamic_quantize(self, model_path: str) -> ToolResult:
        src = Path(model_path)
        if not src.exists():
            return ToolResult(status="error", data=None,
                              explanation=f"missing {src}",
                              math_trace="", error="FileNotFoundError")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation="onnxruntime.quantization not installed",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        dst = src.with_name(src.stem + ".int8" + src.suffix)
        try:
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"quantize failed: {exc}",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        return ToolResult(
            status="ok",
            data={"src": str(src), "dst": str(dst),
                  "src_bytes": int(src.stat().st_size),
                  "dst_bytes": int(dst.stat().st_size)},
            explanation=f"Dynamic int8 quantisation: {src.name} → {dst.name}.",
            math_trace="weights float32 → int8 (symmetric per-tensor).",
        )

    # ---- prune ------------------------------------------------------- #

    def prune(self, model, sparsity: float = 0.2) -> ToolResult:
        if not 0 < sparsity < 1:
            return ToolResult(status="error", data=None,
                              explanation="sparsity must be in (0, 1)",
                              math_trace="", error="ValueError")
        try:
            import torch.nn as nn
            import torch.nn.utils.prune as prune
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation="torch not installed",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(model, nn.Module):
            return ToolResult(status="error", data=None,
                              explanation="model must be torch.nn.Module",
                              math_trace="", error="TypeError")
        pruned_layers = 0
        total_zeroed = 0
        total_params = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=float(sparsity))
                prune.remove(module, "weight")
                w = module.weight.detach().cpu().numpy()
                total_zeroed += int((w == 0).sum())
                total_params += w.size
                pruned_layers += 1
        empirical = total_zeroed / total_params if total_params else 0.0
        return ToolResult(
            status="ok",
            data={"sparsity": sparsity, "layers_pruned": pruned_layers,
                  "empirical_sparsity": float(empirical),
                  "params_zeroed": total_zeroed, "total_params": total_params},
            explanation=(
                f"Pruned {pruned_layers} layer(s); "
                f"zeroed {total_zeroed}/{total_params} "
                f"({empirical:.2%})."
            ),
            math_trace="|w| smallest α-fraction per layer set to 0 (L1 unstructured).",
        )

    # ---- benchmark --------------------------------------------------- #

    def benchmark(self, model, optimised_model, n_samples: int = 1000,
                  input_dim: int = 8) -> ToolResult:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, input_dim)).astype(np.float32)

        def _time(callable_):
            t0 = time.perf_counter()
            _ = callable_(X)
            return time.perf_counter() - t0

        try:
            t_base = _time(self._predict(model))
            t_opt = _time(self._predict(optimised_model))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", data=None,
                              explanation=f"benchmark failed: {exc}",
                              math_trace="", error=f"{type(exc).__name__}: {exc}")
        speedup = (t_base / t_opt) if t_opt > 0 else float("inf")
        return ToolResult(
            status="ok",
            data={"baseline_s": t_base, "optimised_s": t_opt,
                  "speedup": float(speedup), "n_samples": int(n_samples)},
            explanation=(
                f"Baseline {t_base * 1000:.2f} ms, optimised "
                f"{t_opt * 1000:.2f} ms → {speedup:.2f}× speed-up."
            ),
            math_trace="speed-up = t_baseline / t_optimised.",
        )

    @staticmethod
    def _predict(model):
        if hasattr(model, "predict"):
            return lambda X: model.predict(X)
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                model.eval()
                def _call(X):
                    with torch.no_grad():
                        return model(torch.from_numpy(X)).cpu().numpy()
                return _call
        except Exception:
            pass
        raise TypeError("model must support .predict or be a torch.nn.Module")
