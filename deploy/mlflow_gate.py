from __future__ import annotations

import os
import subprocess
import sys

import mlflow

PRIMARY_METRIC = os.environ.get("PRIMARY_METRIC", "pr_auc")
MARGIN = float(os.environ.get("METRIC_GATE_MARGIN", "0.01"))
CHALLENGER_SHA = os.environ.get("CHALLENGER_SHA", "")
MODEL_NAME = "ds-agent-champion"


def main() -> None:
    client = mlflow.MlflowClient()

    try:
        champion = client.get_registered_model(MODEL_NAME)
        champion_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        champion_run = client.get_run(champion_version.run_id)
        champion_metric = champion_run.data.metrics[PRIMARY_METRIC]
    except Exception:
        print("No champion found — promoting challenger as first production model.")
        _promote_challenger(client)
        return

    challenger_runs = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT_NAME", "ds-agent")).experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not challenger_runs:
        print("No challenger run found. Exiting.")
        sys.exit(1)

    challenger_run = challenger_runs[0]
    challenger_metric = challenger_run.data.metrics.get(PRIMARY_METRIC, 0.0)

    print(f"Champion {PRIMARY_METRIC}: {champion_metric:.4f}")
    print(f"Challenger {PRIMARY_METRIC}: {challenger_metric:.4f}")

    if challenger_metric >= champion_metric + MARGIN:
        print("Challenger passes metric gate — promoting.")
        _promote_challenger(client, challenger_run.info.run_id)
    else:
        print("Challenger fails metric gate — keeping champion.")
        sys.exit(1)


def _promote_challenger(client: mlflow.MlflowClient, run_id: str | None = None) -> None:
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        client.create_registered_model(MODEL_NAME)
        version = client.create_model_version(MODEL_NAME, model_uri, run_id)
        client.transition_model_version_stage(MODEL_NAME, version.version, "Production")


if __name__ == "__main__":
    main()
