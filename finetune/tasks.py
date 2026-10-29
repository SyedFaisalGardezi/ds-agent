"""Synthetic training tasks for distillation: workspaces with planted answers.

Each task is a data file, a brief, a request, and the findings a correct
analysis must surface (computed from the generated data). The teacher's runs on
these tasks are scored with the evaluation's scorers, and only runs that pass
become training data, so the dataset contains analyses that found what was
planted, not merely analyses that sounded right.

Four families vary the kind of work: anomaly hunting without labels, a labelled
classification task, a regression task, and a time series with a level shift.
Planted patterns, entity ids, sizes, brief wording and requests are drawn from a
seeded generator, so task `seed` is reproducible.

Contamination guard: training tasks never reuse the evaluation's data, ids or
brief text. Provider ids here are drawn from other prefixes than the eval's
PRV1xx, seeds start at 1,000 (the eval uses one fixed seed), and the real ACC
data is never used. `assert_disjoint_from_eval` checks this for every task.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evals.cases import (
    _CHURN_BRIEF,
    _CLAIMS_BRIEF,
    SEED,
    Expected,
    Fact,
    Finding,
    _lines_per_visit,
    _price_ratio,
    _ratio_fact,
    _share,
    _share_fact,
)

FAMILIES = ("claims_anomaly", "classification", "regression", "timeseries")


@dataclass
class TrainTask:
    task_id: str
    family: str
    seed: int
    request: str
    data_file: str
    brief_file: str
    expected: Expected


# ── shared pieces ─────────────────────────────────────────────────────────────

_PLAN_REQUESTS = [
    "Read the brief, plan the work and complete the EDA.",
    "Look at the documents and data in the working directory, plan the whole "
    "work from the instructions and complete the exploratory analysis.",
    "Plan the analysis based on the instructions, then explore the data.",
    "Go through the brief and the dataset, and do a thorough exploratory analysis.",
    "Explore the data and tell me what stands out, with the numbers.",
]


def _pick(rng: np.random.Generator, items: list):
    return items[int(rng.integers(len(items)))]


def _brief(title: str, context: str, asks: list[str], audience: str,
           dictionary: dict[str, str] | None) -> str:
    lines = [f"# {title}", "", context, "", "Please:"]
    lines += [f"{i}. {a}" for i, a in enumerate(asks, 1)]
    lines += ["", f"The results are for {audience}."]
    if dictionary:
        lines += ["", "## Data dictionary"] + [f"- {k}: {v}" for k, v in dictionary.items()]
    return "\n".join(lines) + "\n"


# ── family 1: claims with planted billing anomalies (no labels) ──────────────

def _claims_task(seed: int, rng: np.random.Generator) -> tuple[pd.DataFrame, str, Expected]:
    prefix = _pick(rng, ["DR", "CL", "PV", "HP", "MED"])
    n_prov = int(rng.integers(8, 17))
    providers = [f"{prefix}{int(n):03d}"
                 for n in rng.choice(np.arange(200, 999), n_prov, replace=False)]
    patterns = list(rng.choice(["overservicing", "price", "few_clients"],
                               size=int(rng.integers(1, 3)), replace=False))
    planted = {str(k): str(v) for k, v in zip(
        patterns, rng.choice(providers, len(patterns), replace=False), strict=True)}
    services = {f"S{n:02d}": float(rng.uniform(40, 130)) for n in range(int(rng.integers(4, 9)))}
    codes = list(services)
    dates = pd.date_range("2023-01-01", periods=int(rng.integers(300, 730)), freq="D")
    rows = []
    for prov in providers:
        over = planted.get("overservicing") == prov
        few = planted.get("few_clients") == prov
        price_mult = float(rng.uniform(1.45, 1.9)) if planted.get("price") == prov else 1.0
        clients = [f"K{n}" for n in rng.integers(10000, 60000, size=5 if few else 150)]
        n_visits = int(rng.integers(220, 380)) * (2 if over or few else 1)
        for _ in range(n_visits):
            client, date, code = (_pick(rng, clients), _pick(rng, list(dates)), _pick(rng, codes))
            lines = int(rng.integers(4, 8)) if over else int(rng.choice([1, 1, 1, 2]))
            for _ in range(lines):
                billed = round(services[code] * price_mult * float(rng.normal(1, 0.04)), 2)
                allowed = round(billed * float(rng.uniform(0.85, 0.96)), 2)
                rows.append((prov, client, date.date().isoformat(), code, billed, allowed,
                             round(allowed * float(rng.uniform(0.95, 1.0)), 2)))
    df = pd.DataFrame(rows, columns=["provider_id", "client_id", "service_date", "service_code",
                                     "billed_amount", "allowed_amount", "paid_amount"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "claim_id", [f"C{seed}-{n}" for n in range(len(df))])

    paid = df.groupby("provider_id").paid_amount.sum()
    findings = []
    if "overservicing" in planted:
        p = planted["overservicing"]
        findings.append(Finding(p, "many service lines per visit", [
            _share_fact("share of total paid", _share(paid, p)),
            Fact("service lines per visit", [_lines_per_visit(df, p)], rel_tol=0.08)]))
    if "price" in planted:
        p = planted["price"]
        findings.append(Finding(p, "prices above peers",
                                [_ratio_fact("price vs peer median", _price_ratio(df, p))]))
    if "few_clients" in planted:
        p = planted["few_clients"]
        findings.append(Finding(p, "high volume concentrated on very few clients",
                                [Fact("distinct clients",
                                      [float(df[df.provider_id == p].client_id.nunique())])]))
    asks = _pick(rng, [
        ["Explore the claims and summarise the main patterns.",
         "Identify providers whose billing looks unusual and say what in the data shows it.",
         "Suggest how to monitor for this going forward."],
        ["Profile the payments and describe what you find.",
         "Point out any billing behaviour that deserves an investigator's attention.",
         "Recommend rules or checks we could run every month."],
        ["Give an overview of who bills what.",
         "Find any providers that stand out from their peers, with evidence.",
         "Explain how you would rank providers for review."],
    ])
    brief = _brief(_pick(rng, ["Billing review", "Provider audit request", "Payment integrity check"]),
                   "We pay providers for services they bill. We have no confirmed fraud cases, "
                   "but we want to know where to look first.",
                   asks, _pick(rng, ["the audit team", "our investigators", "the payments manager"]),
                   {"provider_id": "provider who billed", "client_id": "patient",
                    "service_date": "date of service", "service_code": "type of service",
                    "billed_amount": "amount charged", "allowed_amount": "schedule amount",
                    "paid_amount": "amount paid"} if rng.random() < 0.7 else None)
    exp = Expected(task_types={"exploratory_analysis", "anomaly_detection", "clustering"},
                   target="", questions=[[a[:30].lower()] for a in asks], audiences=[],
                   findings=findings)
    return df, brief, exp


# ── family 2: labelled classification with planted drivers ───────────────────

def _classification_task(seed: int, rng: np.random.Generator):
    n = int(rng.integers(1500, 4000))
    label = _pick(rng, ["churned", "cancelled", "defaulted", "left_service"])
    feats = {
        "tenure_months": rng.integers(1, 73, n),
        "plan_type": rng.choice(["basic", "standard", "premium"], n),
        "monthly_spend": np.round(rng.normal(60, 20, n).clip(5, 200), 2),
        "complaints": rng.poisson(1.0, n),
        "age": rng.integers(18, 80, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
    }
    df = pd.DataFrame(feats)
    drivers = [str(d) for d in rng.choice(["tenure_months", "complaints", "plan_type"], 2,
                                           replace=False)]
    logit = np.full(n, -1.6)
    if "tenure_months" in drivers:
        logit -= 0.04 * df.tenure_months
    if "complaints" in drivers:
        logit += 0.6 * df.complaints
    if "plan_type" in drivers:
        logit += 1.4 * (df.plan_type == "basic")
    df[label] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.insert(0, "customer_id", [f"U{seed}-{i}" for i in range(n)])
    brief = _brief("Who is likely to leave?",
                   f"The column `{label}` records which customers left last period (1 = yes). "
                   "We want to find who is at risk before they go.",
                   ["Explore the data and describe who leaves.",
                    "Identify the main drivers.",
                    "Recommend how a model should be evaluated for this use."],
                   _pick(rng, ["the retention manager", "the customer team"]), None)
    exp = Expected(task_types={"binary_classification"}, target=label,
                   questions=[["describe who"], ["main drivers"], ["evaluated"]], audiences=[],
                   findings=[Finding(label, "base rate",
                                     [_share_fact("overall rate", float(df[label].mean()))])]
                   + [Finding(d, "planted driver") for d in drivers])
    return df, brief, exp


# ── family 3: regression with planted drivers ────────────────────────────────

def _regression_task(seed: int, rng: np.random.Generator):
    n = int(rng.integers(1200, 3000))
    df = pd.DataFrame({
        "distance_km": np.round(rng.uniform(0.5, 25, n), 2),
        "items": rng.integers(1, 15, n),
        "hour": rng.integers(7, 23, n),
        "courier_experience_months": rng.integers(0, 60, n),
        "weather": rng.choice(["clear", "rain", "wind"], n, p=[0.6, 0.3, 0.1]),
    })
    df["delivery_minutes"] = np.round(
        12 + 2.1 * df.distance_km + 0.8 * df["items"] + 9 * (df.weather == "rain")
        - 0.05 * df.courier_experience_months + rng.normal(0, 4, n), 1)
    df.insert(0, "order_id", [f"O{seed}-{i}" for i in range(n)])
    brief = _brief("Delivery times",
                   "Customers complain about slow deliveries. `delivery_minutes` is the time "
                   "from order to door.",
                   ["Explore what drives delivery time.", "Quantify the biggest effects.",
                    "Suggest what we could change."], "the operations lead", None)
    exp = Expected(task_types={"regression"}, target="delivery_minutes",
                   questions=[["drives delivery"], ["biggest effects"], ["could change"]],
                   audiences=[],
                   findings=[Finding("distance_km", "strongest driver"),
                             Finding("rain", "rain adds about 9 minutes")])
    return df, brief, exp


# ── family 4: time series with a planted level shift ─────────────────────────

def _timeseries_task(seed: int, rng: np.random.Generator):
    stores = [f"ST{n:02d}" for n in range(int(rng.integers(5, 11)))]
    shifted = _pick(rng, stores)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    cut = int(rng.integers(120, 260))
    drop = float(rng.uniform(0.25, 0.45))
    rows = []
    for s in stores:
        base = float(rng.uniform(800, 2500))
        for i, d in enumerate(days):
            level = base * (1 - drop if s == shifted and i >= cut else 1.0)
            weekly = 1.25 if d.dayofweek >= 5 else 1.0
            rows.append((d.date().isoformat(), s, round(level * weekly * float(rng.normal(1, 0.06)), 2)))
    df = pd.DataFrame(rows, columns=["date", "store_id", "sales"])
    brief = _brief("Store sales", "Overall sales were lower in the second half of the year.",
                   ["Explore the daily sales by store.", "Find what explains the decline.",
                    "Say how we could detect a change like this sooner."],
                   "the regional manager", None)
    exp = Expected(task_types={"timeseries_forecasting", "exploratory_analysis",
                               "anomaly_detection", "regression"},
                   target="", questions=[["daily sales"], ["explains the decline"],
                                         ["detect a change"]], audiences=[],
                   findings=[Finding(shifted, "level drop after a date",
                                     [_share_fact("size of the drop", drop)]),
                             Finding("weekend", "weekend sales about 25% higher")])
    return df, brief, exp


_BUILDERS = {"claims_anomaly": _claims_task, "classification": _classification_task,
             "regression": _regression_task, "timeseries": _timeseries_task}


def make_task(seed: int, family: str | None = None) -> tuple[TrainTask, pd.DataFrame, str]:
    """The task for `seed`: same seed, same data, brief and request."""
    if seed < 1000 or seed == SEED:
        raise ValueError("training seeds start at 1000 and never equal the eval seed")
    rng = np.random.default_rng(seed)
    family = family or FAMILIES[seed % len(FAMILIES)]
    df, brief, exp = _BUILDERS[family](seed, rng)
    request = _pick(rng, _PLAN_REQUESTS)
    task = TrainTask(f"{family}-{seed}", family, seed, request,
                     f"{family}_{seed}.csv", "brief.md", exp)
    assert_disjoint_from_eval(task, df, brief)
    return task, df, brief


def write_task(task: TrainTask, df: pd.DataFrame, brief: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / task.data_file, index=False)
    (folder / task.brief_file).write_text(brief)
    return folder


def assert_disjoint_from_eval(task: TrainTask, df: pd.DataFrame, brief: str) -> None:
    """Fail loudly if a training task could leak into the evaluation."""
    eval_text = {_CLAIMS_BRIEF.strip(), _CHURN_BRIEF.strip()}
    if brief.strip() in eval_text:
        raise AssertionError(f"{task.task_id}: brief copies an evaluation brief")
    if "provider_id" in df.columns and df.provider_id.astype(str).str.match(r"PRV\d").any():
        raise AssertionError(f"{task.task_id}: uses the evaluation's provider ids")
