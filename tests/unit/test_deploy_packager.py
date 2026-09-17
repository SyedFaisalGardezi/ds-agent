"""Unit tests for deploy_packager."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent.pipeline.deploy_packager.docker_builder import DockerBuilder
from agent.pipeline.deploy_packager.fastapi_builder import FastAPIBuilder
from agent.pipeline.deploy_packager.hpc_dispatcher import HPCDispatcher
from agent.pipeline.deploy_packager.optimizer import ModelOptimizer

# --------------------------------------------------------------------------- #
# DockerBuilder
# --------------------------------------------------------------------------- #


class _FakeRun:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        return SimpleNamespace(stdout="built", returncode=0)


def test_docker_builder_build_issues_expected_cmd():
    runner = _FakeRun()
    db = DockerBuilder(runner=runner, which=lambda _: "/usr/bin/docker",
                       context=".", dockerfile="Dockerfile")
    r = db.build(tag="ds:latest", target="runtime",
                 build_args={"ENV": "ci"})
    assert r.status == "ok"
    cmd = runner.calls[0]
    assert cmd[:5] == ["docker", "build", "-t", "ds:latest", "--target"]
    assert "--build-arg" in cmd and "ENV=ci" in cmd


def test_docker_builder_missing_binary():
    db = DockerBuilder(runner=_FakeRun(), which=lambda _: None)
    r = db.build(tag="x")
    assert r.status == "error"
    assert r.error == "DockerNotInstalled"


def test_docker_builder_empty_tag():
    r = DockerBuilder(runner=_FakeRun(),
                      which=lambda _: "/usr/bin/docker").build(tag="")
    assert r.status == "error"


def test_docker_builder_push_issues_tag_and_push():
    runner = _FakeRun()
    db = DockerBuilder(runner=runner, which=lambda _: "/usr/bin/docker")
    r = db.push(tag="ds:latest", registry="registry.io/team")
    assert r.status == "ok"
    assert runner.calls[0][0:2] == ["docker", "tag"]
    assert runner.calls[1][0:2] == ["docker", "push"]
    assert r.data["pushed"] == "registry.io/team/ds:latest"


def test_docker_builder_handles_nonzero_rc():
    def runner(args, **kw):
        raise subprocess.CalledProcessError(returncode=2, cmd=args, stderr="boom")
    db = DockerBuilder(runner=runner, which=lambda _: "/usr/bin/docker")
    r = db.build(tag="ds:x")
    assert r.status == "error"
    assert r.data["returncode"] == 2


# --------------------------------------------------------------------------- #
# FastAPIBuilder
# --------------------------------------------------------------------------- #


def test_fastapi_builder_writes_three_files(tmp_path):
    r = FastAPIBuilder().build(model_uri=str(tmp_path / "model.joblib"),
                               output_dir=str(tmp_path / "deploy"))
    assert r.status == "ok"
    out = Path(r.data["output_dir"])
    assert (out / "app.py").is_file()
    assert (out / "Dockerfile").is_file()
    assert (out / "requirements.txt").is_file()
    # Template interpolation sanity
    assert "MODEL_URI" in (out / "app.py").read_text()


def test_fastapi_builder_requires_model_uri():
    r = FastAPIBuilder().build(model_uri="", output_dir="x")
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# HPCDispatcher
# --------------------------------------------------------------------------- #


def test_hpc_dispatcher_renders_slurm_script(tmp_path):
    runner_calls = []

    def runner(args, **kw):
        runner_calls.append(args)
        return SimpleNamespace(stdout="Submitted batch job 12345\n",
                               returncode=0)

    dispatcher = HPCDispatcher(node="gpu01", scheduler="slurm",
                               script_dir=str(tmp_path),
                               runner=runner,
                               which=lambda _: "/usr/bin/sbatch")
    r = dispatcher.submit("etl", {"cpus": 8, "gpus": 2, "mem": "16G"})
    assert r.status == "ok"
    assert r.data["job_id"] == "12345"
    script = Path(r.data["script_path"]).read_text()
    assert "#SBATCH --job-name=ds-etl" in script
    assert "--cpus-per-task=8" in script
    assert "--gres=gpu:2" in script
    assert "--nodelist=gpu01" in script


def test_hpc_dispatcher_renders_pbs_script(tmp_path):
    def runner(args, **kw):
        return SimpleNamespace(stdout="987.pbs-master\n", returncode=0)

    d = HPCDispatcher(node="n1", scheduler="pbs", script_dir=str(tmp_path),
                      runner=runner, which=lambda _: "/usr/bin/qsub")
    r = d.submit("train", {"cpus": 4})
    assert r.status == "ok"
    assert r.data["job_id"] == "987.pbs-master"
    script = Path(r.data["script_path"]).read_text()
    assert "#PBS -N ds-train" in script
    assert "ncpus=4" in script


def test_hpc_dispatcher_missing_scheduler():
    d = HPCDispatcher(node="n1", scheduler="slurm",
                      script_dir="/tmp",
                      runner=lambda *a, **kw: None,
                      which=lambda _: None)
    r = d.submit("etl", {})
    assert r.status == "error"
    assert r.error == "SchedulerNotInstalled"


def test_hpc_dispatcher_invalid_scheduler_raises():
    with pytest.raises(ValueError):
        HPCDispatcher(node="n1", scheduler="lsf")


def test_hpc_dispatcher_tail_logs_missing_file(tmp_path):
    d = HPCDispatcher(node="n1", script_dir=str(tmp_path),
                      runner=lambda *a, **k: None,
                      which=lambda _: "/usr/bin/sbatch")
    r = d.tail_logs("nope")
    assert r.status == "warning"


def test_hpc_dispatcher_tail_logs_reads_last_n(tmp_path):
    d = HPCDispatcher(node="n1", script_dir=str(tmp_path),
                      runner=lambda *a, **k: None,
                      which=lambda _: "/usr/bin/sbatch")
    (Path(tmp_path) / "42.out").write_text("\n".join(str(i) for i in range(100)))
    r = d.tail_logs("42", n=5)
    assert r.data["lines"] == ["95", "96", "97", "98", "99"]


def test_hpc_dispatcher_status_pending():
    d = HPCDispatcher(node="n1", script_dir="/tmp",
                      runner=lambda *a, **k: None,
                      which=lambda _: "/usr/bin/sbatch")
    s = d.status("42")
    assert s.state == "pending"
    assert s.job_id == "42"


# --------------------------------------------------------------------------- #
# ModelOptimizer
# --------------------------------------------------------------------------- #


def test_optimizer_prune_sparsity_range():
    r = ModelOptimizer().prune(model=object(), sparsity=1.5)
    assert r.status == "error"
    r2 = ModelOptimizer().prune(model=object(), sparsity=0.0)
    assert r2.status == "error"


def test_optimizer_prune_rejects_non_module():
    try:
        import torch.nn as nn  # noqa: F401
    except Exception:
        pytest.skip("torch not installed")
    r = ModelOptimizer().prune(model=object(), sparsity=0.2)
    assert r.status == "error"


def test_optimizer_prune_zeroes_linear_weights():
    try:
        import torch.nn as nn
    except Exception:
        pytest.skip("torch not installed")
    model = nn.Sequential(nn.Linear(10, 6), nn.ReLU(), nn.Linear(6, 2))
    r = ModelOptimizer().prune(model, sparsity=0.5)
    assert r.status == "ok"
    assert r.data["layers_pruned"] == 2
    assert r.data["params_zeroed"] > 0
    # Empirical sparsity should be close to requested 0.5
    assert 0.3 < r.data["empirical_sparsity"] <= 1.0


def test_optimizer_benchmark_with_predict_duck_type():
    class M:
        def predict(self, X):
            return np.zeros(len(X))
    r = ModelOptimizer().benchmark(M(), M(), n_samples=100, input_dim=4)
    assert r.status == "ok"
    assert r.data["n_samples"] == 100
    assert r.data["speedup"] > 0


def test_optimizer_onnx_export_rejects_non_module(tmp_path):
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not installed")
    r = ModelOptimizer().onnx_export(
        model=object(), sample_input=np.zeros((1, 4)),
        output_path=str(tmp_path / "m.onnx"))
    assert r.status == "error"


def test_optimizer_dynamic_quantize_missing_src(tmp_path):
    r = ModelOptimizer().dynamic_quantize(str(tmp_path / "nope.onnx"))
    assert r.status == "error"
    assert r.error == "FileNotFoundError"


# --------------------------------------------------------------------------- #
# ModelOptimizer — additional gap coverage
# --------------------------------------------------------------------------- #


def test_optimizer_dynamic_quantize_success(tmp_path, monkeypatch):
    """Mock onnxruntime to test success path of dynamic_quantize."""
    src = tmp_path / "model.onnx"
    src.write_bytes(b"fake onnx content")

    class FakeQuantType:
        QInt8 = "QInt8"

    def fake_quantize(src_path, dst_path, weight_type):
        # Create the destination file to simulate success
        import shutil
        shutil.copy(src_path, dst_path)

    import sys
    fake_quant_mod = type(sys)("onnxruntime.quantization")
    fake_quant_mod.QuantType = FakeQuantType
    fake_quant_mod.quantize_dynamic = fake_quantize
    monkeypatch.setitem(sys.modules, "onnxruntime", type(sys)("onnxruntime"))
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", fake_quant_mod)

    r = ModelOptimizer().dynamic_quantize(str(src))
    assert r.status == "ok"
    assert "int8" in r.data["dst"]
    assert r.data["src_bytes"] > 0


def test_optimizer_onnx_export_success(tmp_path, monkeypatch):
    """Mock torch.onnx.export to test success path."""
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not installed")

    model = nn.Sequential(nn.Linear(4, 2))
    out_path = tmp_path / "model.onnx"

    real_export_called = []

    def fake_export(model, inp, out_str, **kwargs):
        # Write a fake ONNX file so stat().st_size works
        import pathlib
        pathlib.Path(out_str).write_bytes(b"fake onnx")
        real_export_called.append(True)

    import torch
    monkeypatch.setattr(torch.onnx, "export", fake_export)

    import numpy as np
    sample = np.zeros((1, 4), dtype=np.float32)
    r = ModelOptimizer().onnx_export(model, sample, str(out_path))
    assert r.status == "ok"
    assert r.data["size_bytes"] > 0
    assert len(real_export_called) == 1


def test_optimizer_predict_torch_module():
    """_predict() should return a callable for a torch.nn.Module."""
    try:
        import numpy as np
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not installed")

    model = nn.Sequential(nn.Linear(4, 2))
    fn = ModelOptimizer._predict(model)
    assert callable(fn)
    X = np.zeros((3, 4), dtype=np.float32)
    out = fn(X)
    assert out is not None


def test_optimizer_predict_unknown_type():
    """_predict() raises TypeError for models that neither predict nor have forward."""
    import pytest
    with pytest.raises(TypeError):
        ModelOptimizer._predict(object())()


def test_optimizer_benchmark_exception_path():
    class Bad:
        def predict(self, X):
            raise RuntimeError("intentional error")

    r = ModelOptimizer().benchmark(Bad(), Bad(), n_samples=10, input_dim=4)
    assert r.status == "error"
    assert "RuntimeError" in r.error
