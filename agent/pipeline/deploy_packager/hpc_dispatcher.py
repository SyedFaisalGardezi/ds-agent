"""HPCDispatcher — submit pipeline stages to SLURM or PBS.

Keeps the real scheduler at arms' length via an injectable ``runner`` so
tests never touch a cluster. Script generation is deterministic and records
the exact script for provenance.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from agent.core.types import JobStatus, ToolResult


class HPCDispatcher:
    _JOB_ID_SLURM = re.compile(r"Submitted batch job (\d+)")
    _JOB_ID_PBS = re.compile(r"^(\S+)")

    def __init__(self, node: str, scheduler: str = "slurm",
                 script_dir: str = "outputs/hpc", runner=None, which=None) -> None:
        if scheduler not in {"slurm", "pbs"}:
            raise ValueError("scheduler must be 'slurm' or 'pbs'")
        self._node = node
        self._scheduler = scheduler
        self._dir = Path(script_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._runner = runner or subprocess.run
        self._which = which or shutil.which

    def _render(self, stage: str, config: dict) -> str:
        cmd = config.get("cmd", f"python -m agent.pipeline.{stage}")
        job_name = config.get("job_name", f"ds-{stage}")
        time_limit = config.get("time", "01:00:00")
        mem = config.get("mem", "8G")
        cpus = int(config.get("cpus", 4))
        gpus = int(config.get("gpus", 0))
        log = f"{self._dir}/{job_name}.%j.log"
        if self._scheduler == "slurm":
            directives = [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --time={time_limit}",
                f"#SBATCH --mem={mem}",
                f"#SBATCH --cpus-per-task={cpus}",
                f"#SBATCH --output={log}",
                f"#SBATCH --nodelist={self._node}",
            ]
            if gpus > 0:
                directives.append(f"#SBATCH --gres=gpu:{gpus}")
        else:  # pbs
            directives = [
                "#!/bin/bash",
                f"#PBS -N {job_name}",
                f"#PBS -l walltime={time_limit}",
                f"#PBS -l mem={mem}",
                f"#PBS -l select=1:ncpus={cpus}" +
                (f":ngpus={gpus}" if gpus > 0 else ""),
                f"#PBS -o {log}",
                f"#PBS -l nodes={self._node}",
            ]
        return "\n".join(directives + ["", cmd, ""])

    def submit(self, stage: str, config: dict) -> ToolResult:
        submit_bin = "sbatch" if self._scheduler == "slurm" else "qsub"
        if self._which(submit_bin) is None:
            return ToolResult(status="error", data=None,
                              explanation=f"{submit_bin} not on PATH",
                              math_trace="", error="SchedulerNotInstalled")
        script = self._render(stage, config)
        script_path = self._dir / f"{stage}.sh"
        script_path.write_text(script)
        try:
            res = self._runner([submit_bin, str(script_path)],
                               check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            return ToolResult(status="error", data=None,
                              explanation=f"{submit_bin} failed rc={exc.returncode}",
                              math_trace="", error=f"CalledProcessError: {exc}")
        out = getattr(res, "stdout", "").strip()
        pattern = self._JOB_ID_SLURM if self._scheduler == "slurm" else self._JOB_ID_PBS
        match = pattern.search(out)
        job_id = match.group(1) if match else out or "unknown"
        return ToolResult(
            status="ok",
            data={"job_id": job_id, "scheduler": self._scheduler,
                  "node": self._node, "stage": stage,
                  "script_path": str(script_path)},
            explanation=f"Submitted {stage} as job {job_id} on {self._node}.",
            math_trace=f"render({stage}) → {submit_bin} → id={job_id}.",
        )

    def status(self, job_id: str) -> JobStatus:
        stdout_path = str(self._dir / f"{job_id}.out")
        stderr_path = str(self._dir / f"{job_id}.err")
        # In-process we can't query a real scheduler; return a pending skeleton.
        return JobStatus(
            job_id=job_id, node=self._node, scheduler=self._scheduler,  # type: ignore[arg-type]
            state="pending",
            stdout_path=stdout_path, stderr_path=stderr_path,
            exit_code=None,
        )

    def tail_logs(self, job_id: str, n: int = 200) -> ToolResult:
        path = self._dir / f"{job_id}.out"
        if not path.exists():
            return ToolResult(status="warning", data={"lines": []},
                              explanation=f"no log at {path}",
                              math_trace="|lines|=0.")
        lines = path.read_text().splitlines()[-n:]
        return ToolResult(
            status="ok",
            data={"lines": lines, "path": str(path)},
            explanation=f"Tailed last {len(lines)} line(s) of {path}.",
            math_trace=f"tail -n {n} {path}.",
        )
