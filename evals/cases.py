"""Evaluation cases for the DS agent: workspaces with a known right answer.

Each case is a working directory (data plus a brief), the request a user would
type, and what a correct run must show:

  * understanding: the task type, whether there is a target, and the brief's
    questions and audiences, checked against what understand_task extracted;
  * findings: the entities a good analysis must surface (e.g. a provider whose
    billing is unusual), each with numeric facts that are computed from the
    data by this module, never typed in by hand.

The synthetic cases plant their signals with a fixed seed, so the right answer
is known by construction and the data is identical on every run. The ACC case
uses the real exercise data when it is present on this machine and is skipped
otherwise.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ACC_DIR = Path("/Users/ishtiasyed/My_Data/Job_application/ACC_DS_interview/my_agent_work")
SEED = 20260930


@dataclass
class Fact:
    """A number a correct analysis should state about an entity.

    `values` lists the accepted forms (e.g. a share as 0.47 or 47 percent).
    A number in the reply matches when it is within max(abs_tol,
    rel_tol * |value|) of any accepted form.
    """
    label: str
    values: list[float]
    rel_tol: float = 0.05
    abs_tol: float = 0.0


@dataclass
class Finding:
    """Something a good analysis must surface. `entity` is the text that has
    to appear in the reply (an id such as PRV020, or a column name)."""
    entity: str
    why: str
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Expected:
    task_types: set[str]            # accepted task types
    target: str                     # "" when the data has no labelled outcome
    questions: list[list[str]]      # per brief question: phrases, one must match
    audiences: list[str]            # phrases, each must appear in an audience
    findings: list[Finding]


@dataclass
class Case:
    name: str
    description: str
    request: str
    build: Callable[[Path], Expected]   # writes the workspace, returns truth
    available: Callable[[], bool] = lambda: True


# ── shared helpers ────────────────────────────────────────────────────────────

def _share(values: pd.Series, key: str) -> float:
    return float(values.get(key, 0.0) / values.sum())


def _lines_per_visit(df: pd.DataFrame, provider: str) -> float:
    visits = df.groupby(["provider_id", "client_id", "service_date"]).size()
    return float(visits.loc[provider].mean())


def _price_ratio(df: pd.DataFrame, provider: str) -> float:
    """Mean billed price per service for `provider`, over the median of the
    other providers' mean price for the same service, averaged over services."""
    unit = df.groupby(["provider_id", "service_code"]).billed_amount.mean().unstack()
    peers = unit.drop(index=provider).median()
    return float((unit.loc[provider] / peers).dropna().mean())


def _share_fact(label: str, share: float) -> Fact:
    return Fact(label, [share, share * 100], rel_tol=0.05)


def _ratio_fact(label: str, ratio: float) -> Fact:
    # "1.8x", "80% above", or "0.8 above" are all ways to state the same ratio.
    return Fact(label, [ratio, (ratio - 1) * 100, ratio - 1], rel_tol=0.06)


# ── case 1: the ACC fraud exercise (real data, no labels) ─────────────────────

def _build_acc(ws: Path) -> Expected:
    for f in ["medical_claims_dataset.csv", "Candidate Exercise - Data Scientist.pdf",
              "data_dictionary.pdf"]:
        (ws / f).write_bytes((ACC_DIR / f).read_bytes())
    df = pd.read_csv(ws / "medical_claims_dataset.csv")
    paid = df.groupby("provider_id").paid_amount.sum()
    prv014 = df[df.provider_id == "PRV014"]
    return Expected(
        task_types={"exploratory_analysis", "anomaly_detection", "clustering"},
        target="",
        questions=[["exploratory analysis", "key findings"],
                   ["populations of payments", "present a risk"],
                   ["future instances", "long-term solution", "long term solution"],
                   ["recommendations", "stakeholders"]],
        audiences=["data analyst", "fraud investigator", "senior decision maker"],
        findings=[
            Finding("PRV020", "far more claims and service lines per visit than any peer",
                    [_share_fact("share of total paid", _share(paid, "PRV020")),
                     Fact("service lines per visit", [_lines_per_visit(df, "PRV020")],
                          rel_tol=0.06)]),
            Finding("PRV014", "prices well above peers, concentrated on a handful of clients",
                    [_ratio_fact("price vs peer median", _price_ratio(df, "PRV014")),
                     Fact("distinct clients", [float(prv014.client_id.nunique())])]),
        ])


# ── case 2: synthetic provider claims with planted billing anomalies ─────────

_CLAIMS_BRIEF = """\
# Provider payments review

We pay healthcare providers for the services they bill us for. We are worried
that some providers are billing for more than they deliver, but we have no list
of confirmed fraud cases.

Please:
1. Explore the payments data and summarise the main patterns.
2. Identify any providers whose billing looks unusual, and explain which
   features of the data show it.
3. Suggest how we could keep monitoring for this behaviour in future.

The results are for our fraud investigation team.

## Data dictionary
- claim_id: one row per billed service line
- provider_id: the provider who billed the service
- client_id: the patient who received it
- service_date: date the service was delivered
- service_code: the type of service
- billed_amount: amount the provider charged (NZD)
- allowed_amount: amount our schedule allows for that service (NZD)
- paid_amount: amount we actually paid (NZD)
"""


def _synthetic_claims(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    providers = [f"PRV{n}" for n in range(101, 113)]
    services = {"PHY01": 60.0, "PHY02": 85.0, "MAS01": 55.0, "ACU01": 70.0,
                "CHI01": 65.0, "OST01": 90.0}
    codes = list(services)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for prov in providers:
        # PRV103 over-services: many visits, five lines of one service each.
        n_visits = 520 if prov == "PRV103" else int(rng.integers(260, 360))
        clients = [f"C{n}" for n in rng.integers(1000, 4000, size=120)]
        for _ in range(n_visits):
            client = clients[int(rng.integers(len(clients)))]
            date = dates[int(rng.integers(len(dates)))]
            code = codes[int(rng.integers(len(codes)))]
            lines = 5 if prov == "PRV103" else int(rng.choice([1, 1, 1, 1, 2]))
            for _ in range(lines):
                base = services[code] * (1.6 if prov == "PRV107" else 1.0)
                billed = round(base * float(rng.normal(1.0, 0.04)), 2)
                allowed = round(billed * float(rng.uniform(0.86, 0.95)), 2)
                paid = round(allowed * float(rng.uniform(0.95, 1.0)), 2)
                rows.append((prov, client, date.date().isoformat(), code,
                             billed, allowed, paid))
    df = pd.DataFrame(rows, columns=["provider_id", "client_id", "service_date",
                                     "service_code", "billed_amount",
                                     "allowed_amount", "paid_amount"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "claim_id", [f"CLM{n:06d}" for n in range(len(df))])
    return df


def _build_claims(ws: Path) -> Expected:
    df = _synthetic_claims()
    df.to_csv(ws / "provider_claims.csv", index=False)
    (ws / "brief.md").write_text(_CLAIMS_BRIEF)
    paid = df.groupby("provider_id").paid_amount.sum()
    return Expected(
        task_types={"exploratory_analysis", "anomaly_detection", "clustering"},
        target="",
        questions=[["main patterns", "explore the payments"],
                   ["billing looks unusual", "providers whose billing"],
                   ["monitoring", "in future"]],
        audiences=["fraud investigation"],
        findings=[
            Finding("PRV103", "over-servicing: five lines of one service per visit",
                    [_share_fact("share of total paid", _share(paid, "PRV103")),
                     Fact("service lines per visit", [_lines_per_visit(df, "PRV103")],
                          rel_tol=0.06)]),
            Finding("PRV107", "billed prices about 1.6 times its peers",
                    [_ratio_fact("price vs peer median", _price_ratio(df, "PRV107"))]),
        ])


# ── case 3: synthetic churn data with a real label ───────────────────────────

_CHURN_BRIEF = """\
# Customer churn

We want to predict which customers will cancel in the next quarter so the
retention team can contact them first. The column `churned` records who left
last quarter (1 = left).

Please:
1. Explore the data and describe who churns.
2. Identify the main drivers of churn.
3. Recommend how a model should be evaluated, given that the retention team can
   only call the top 10% of customers each quarter.

The audience is the retention team lead.
"""


def _synthetic_churn(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = 3000
    tenure = rng.integers(1, 73, size=n)
    contract = rng.choice(["month-to-month", "one_year", "two_year"], size=n,
                          p=[0.5, 0.3, 0.2])
    calls = rng.poisson(1.5, size=n)
    charges = np.round(rng.normal(70, 18, size=n).clip(20, 140), 2)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    logit = (-1.8 + 1.7 * (contract == "month-to-month") - 0.035 * tenure
             + 0.35 * calls)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": [f"CU{n:05d}" for n in range(n)], "tenure_months": tenure,
        "contract": contract, "monthly_charges": charges, "support_calls": calls,
        "region": region, "churned": churned})


def _build_churn(ws: Path) -> Expected:
    df = _synthetic_churn()
    df.to_csv(ws / "customers.csv", index=False)
    (ws / "brief.md").write_text(_CHURN_BRIEF)
    m2m = float(df[df.contract == "month-to-month"].churned.mean())
    return Expected(
        task_types={"binary_classification"},
        target="churned",
        questions=[["who churns", "describe who"],
                   ["drivers of churn", "main drivers"],
                   ["evaluated", "top 10%"]],
        audiences=["retention team"],
        findings=[
            Finding("churn", "overall churn rate",
                    [_share_fact("overall churn rate", float(df.churned.mean()))]),
            Finding("month-to-month", "month-to-month contracts churn most",
                    [_share_fact("month-to-month churn rate", m2m)]),
            Finding("tenure", "short tenure raises churn", []),
            Finding("support_calls", "more support calls raise churn", []),
        ])


CASES: list[Case] = [
    Case("acc_claims", "ACC fraud exercise, real data, no labels",
         "Look into documents and data at working directory and based on "
         "instructions plan the whole work and complete EDA",
         _build_acc, available=lambda: ACC_DIR.is_dir()),
    Case("synthetic_claims", "Planted over-servicing and over-pricing, no labels",
         "Read the brief, plan the work and complete the EDA.", _build_claims),
    Case("synthetic_churn", "Labelled churn data with planted drivers",
         "Read the brief, plan the work and complete the EDA.", _build_churn),
]


# ── intent suite: does the model read requests correctly? ────────────────────

@dataclass
class IntentCase:
    query: str
    decisions: set[str]                          # accepted decisions
    required_tools: set[str] = field(default_factory=set)
    forbidden_tools: set[str] = field(default_factory=set)
    plan_first: bool | None = None               # None = not checked
    # "fresh": a session with data and a brief, nothing done yet.
    # "after_analysis": the same kind of session after an EDA and a trained
    # model (see AFTER_ANALYSIS), where requests about earlier work make sense.
    context: str = "fresh"


INTENT_CASES: list[IntentCase] = [
    IntentCase("hi", {"answer"}),
    IntentCase("What does the brief ask me to deliver?", {"answer"}),
    IntentCase("What is model accuracy?", {"answer", "clarify"}),
    IntentCase("Do not train a model. Explain how you would assess it.", {"answer"},
               forbidden_tools={"train_model"}),
    IntentCase("Plan the whole work based on the instructions and the data.", {"act"},
               required_tools={"work_plan"}, plan_first=True),
    IntentCase("Explore the data in depth and find which providers look anomalous.",
               {"act"}, forbidden_tools={"train_model"}, plan_first=True),
    IntentCase("Show me the number of missing values per column.", {"act"},
               forbidden_tools={"train_model", "write_file"}),
    IntentCase("Write the Snowflake SQL for total paid per provider.", {"act"},
               required_tools={"write_sql"}),
    # Requests about earlier work are checked in both situations: with nothing
    # done yet, asking (or saying so) is right; after the work, acting is.
    IntentCase("Export the analysis as a Python script.", {"clarify", "answer"}),
    IntentCase("Export the analysis as a Python script.", {"act"},
               required_tools={"build_notebook"}, context="after_analysis"),
    IntentCase("Train a model to predict which claims are fraudulent.",
               {"act", "clarify", "answer"}, forbidden_tools={"train_model"}),
    IntentCase("Deploy the already trained model without retraining.",
               {"clarify", "answer"}, forbidden_tools={"train_model"}),
    IntentCase("Deploy the already trained model without retraining.", {"act"},
               required_tools={"deploy_model"}, forbidden_tools={"train_model"},
               context="after_analysis"),
    IntentCase("Summarise what you found so far, without running anything new.",
               {"answer", "clarify"}),
    IntentCase("Summarise what you found so far, without running anything new.",
               {"answer"}, context="after_analysis"),
]

# The state of a churn session after an EDA and a trained model, for the
# "after_analysis" intent cases. The findings match the synthetic churn data
# (20.5% overall, 31.7% on month-to-month contracts); the model and its score
# are a stand-in, since only the interpreter is under test.
AFTER_ANALYSIS = {
    "history": [
        {"role": "user", "content": "Read the brief, plan the work, complete the EDA "
                                    "and train a model to predict churn."},
        {"role": "assistant", "content": "Done. 20.5% of customers churned, and "
                                         "month-to-month contracts churn most (31.7%). Short "
                                         "tenure and more support calls also raise churn. A "
                                         "gradient boosting model reached a PR-AUC of 0.61 on "
                                         "the hold-out set."},
    ],
    "namespace": {
        "TASK_SPEC": {"task_type": "binary_classification", "target_column": "churned",
                      "evaluation_metric": "pr_auc",
                      "task_description": "Predict which customers will churn"},
        "TARGET": "churned",
        "WORK_PLAN": {"version": 2, "objective": "Predict churn and explain its drivers",
                      "revised_after_eda": True},
        "EDA_AGENDA": [
            {"id": "Q1", "question": "Who churns?", "source": "brief", "status": "answered",
             "finding": "20.5% churned overall; month-to-month contracts 31.7%"},
            {"id": "Q2", "question": "What drives churn?", "source": "brief",
             "status": "answered",
             "finding": "contract type, short tenure and support calls"},
        ],
        "LAST_METRICS": {"best_model": "gradient_boosting", "pr_auc": 0.61},
    },
    "artifacts": {"trained_model": "gradient_boosting"},
}
