"""Executable DVC stages; no learned transformations before the training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from agent.pipeline.model_builder.supervised import train


def run_stage(stage: str, config_path: str = "configs/pipeline.yaml") -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    raw = Path("outputs/datasets/raw.parquet")
    clean = Path("outputs/datasets/clean.parquet")
    if stage == "extract":
        frame = pd.read_csv(config["source_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(raw, index=False)
    elif stage == "transform":
        pd.read_parquet(raw).dropna(how="all").to_parquet(clean, index=False)
    elif stage == "train":
        import joblib
        result = train(pd.read_parquet(clean), target=config["target"],
                       task_type=config["task_type"], models=config["models"])
        models, reports = Path("outputs/models"), Path("outputs/reports")
        models.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], models / "pipeline.joblib")
        (reports / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    else:
        raise ValueError(f"Unknown pipeline stage: {stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "transform", "train"])
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run_stage(args.stage, args.config)
