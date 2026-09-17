"""Drive the ds-agent API on a CSV file.

Usage:
    python run_on_csv.py <path/to/data.csv> <target_column> [--estimator rf]

Runs:  profile → quality → mutual-info → features → task → train → drift
Prints a concise per-stage report and the leaderboard row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

API = "http://127.0.0.1:9090"


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe row dicts — NaN → None."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _post(path: str, body: dict) -> dict:
    r = requests.post(f"{API}{path}", json=body, timeout=300)
    if r.status_code >= 400:
        print(f"  ✗ {path} → {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    return r.json()


def _get(path: str, **params) -> dict:
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    return r.json()


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="path to CSV")
    ap.add_argument("target", help="target column name")
    ap.add_argument("--estimator", default="rf",
                    choices=["rf", "gbm", "linear"])
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N rows (0 = all)")
    ap.add_argument("--run-id", default="csv-run")
    args = ap.parse_args()

    # --- load ----------------------------------------------------------
    df = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=0).reset_index(drop=True)
    if args.target not in df.columns:
        print(f"target {args.target!r} not in CSV columns: {list(df.columns)}")
        sys.exit(2)

    # Keep only numeric / categorical / bool — drop things the API can't JSON.
    df = df.select_dtypes(exclude=["datetime64[ns]", "timedelta64[ns]"])
    banner(f"Loaded {args.csv.name}: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.dtypes.to_string())

    recs = _records(df)

    # --- 1. profile ----------------------------------------------------
    banner("1. Profile")
    r = _post("/pipelines/profile", {"records": recs})
    d = r["data"]
    print(f"n_rows={d['n_rows']}  n_cols={d['n_cols']}  "
          f"null_rate={d['overall_null_rate']:.3%}  dups={d['duplicate_rows']}")

    # --- 2. quality ----------------------------------------------------
    banner("2. Quality score")
    r = _post("/pipelines/quality", {"records": recs})
    d = r["data"]
    print(f"score={d['score']:.3f}  status={r['status']}")
    for k, v in d["dimensions"].items():
        print(f"  {k:14s} {v:.3f}")

    # --- 3. mutual information ----------------------------------------
    banner("3. Mutual information vs target")
    r = _post("/pipelines/statistics/mi",
              {"records": recs, "target": args.target})
    for item in r["data"]["scores"][:10]:
        print(f"  {item['feature']:25s}  {item['mi']:.4f}")

    # --- 4. feature generation ----------------------------------------
    banner("4. Feature generation")
    r = _post("/pipelines/features",
              {"records": recs, "target": args.target,
               "max_pairs": 5, "max_onehot_cardinality": 8})
    d = r["data"]
    print(f"augmented shape: {d.get('frame_shape')}")
    print(f"new features: {d.get('new_features', [])[:10]}")

    # --- 5. task detection --------------------------------------------
    banner("5. Task detection")
    r = _post("/pipelines/task",
              {"records": recs, "target": args.target})
    print(f"task = {r['data']['task']}   reasons: {r['data']['reasons']}")

    # --- 6. train ------------------------------------------------------
    banner("6. Train")
    r = _post("/pipelines/train", {
        "records": recs,
        "target": args.target,
        "estimator": args.estimator,
        "n_estimators": args.n_estimators,
        "test_size": 0.25,
        "random_state": 0,
        "run_id": args.run_id,
    })
    d = r["data"]
    print(f"run_id={d['run_id']}  task={d['task']}")
    print(f"n_train={d['n_train']}  n_test={d['n_test']}")
    print("metrics:")
    for k, v in d["metrics"].items():
        print(f"  {k:20s} {v}")

    # --- 7. drift (train split vs held-out) ---------------------------
    banner("7. Drift (first half vs second half)")
    half = len(recs) // 2
    r = _post("/pipelines/drift",
              {"baseline": recs[:half], "current": recs[half:]})
    d = r["data"]
    print(f"features checked: {d.get('n_checked')}")
    print(f"drifting: {d.get('drifting', [])[:10]}")

    # --- 8. leaderboard -----------------------------------------------
    banner("8. Leaderboard")
    metric = ("accuracy" if r["status"] != "error"
              and "accuracy" in _post("/pipelines/task",
                                      {"records": recs,
                                       "target": args.target})["data"].get(
                  "reasons", [""])[0].lower() else "accuracy")
    r = _get("/leaderboard", metric="accuracy", n=5)
    for run in r["data"].get("runs", [])[:5]:
        metrics = json.loads(run["metrics"])
        print(f"  {run['run_id']:20s} {run['model_type']:6s} "
              f"{metrics}")

    print("\n✓ pipeline complete")


if __name__ == "__main__":
    main()
