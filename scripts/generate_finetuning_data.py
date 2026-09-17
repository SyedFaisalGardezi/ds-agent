"""Generate instruction fine-tuning dataset for the understand_task / problem-scoping step.

Two dataset modes are produced (controlled by --mode):

  scope   (default) — teaches qwen3.6:latest to output the structured
                       SCOPE ANALYSIS block consumed by _parse_scope_analysis().
                       Input:  task_brief + dataset_stats
                       Output: reasoning + structured --- SCOPE ANALYSIS --- block

  extract — teaches the model to emit a flat JSON RawSpec (legacy / LLM extraction calls).
             Input:  brief + dataset_stats
             Output: JSON object with RawSpec fields

Usage:
    python scripts/generate_finetuning_data.py \\
        --out data/finetuning/understand_task_scope.jsonl \\
        --mode scope \\
        --n_augment 3

    python scripts/generate_finetuning_data.py \\
        --out data/finetuning/understand_task_extract.jsonl \\
        --mode extract \\
        --n_augment 3
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

random.seed(42)

# ── System prompts ─────────────────────────────────────────────────────────────

SCOPE_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert data science problem analyst.

Given a task brief and dataset information, reason carefully and then commit \
your analysis using EXACTLY this format:

--- SCOPE ANALYSIS ---
BUSINESS_GOAL: <one sentence: what the organisation needs>
ENTITY_UNIT: per <entity type, e.g. establishment / customer / patient / loan>
TASK_TYPE: <binary_classification | multiclass_classification | regression | timeseries_forecasting | anomaly_detection>
TASK_TYPE_REASON: <cite the EXACT phrase from the brief that determines this>
TARGET_COLUMN: <exact column name if already in data, OR "DERIVE: <condition>">
AGGREGATION: <needed | not_needed>
AGGREGATION_KEY: <column name to group by — MUST be entity ID, never category code>
METRIC: <pr_auc | roc_auc | f1 | rmse | mae | r2 | mase>
SECONDARY_FILE: <yes | no>
SECONDARY_FILE_REASON: <which file holds the target labels, or "n/a">
SPECIFIC_STEPS: <pipe-separated non-standard steps, e.g. risk_ranking | output_csv>
--- END SCOPE ---

MANDATORY RULES:
- "predict which [entities] will [event]" = ALWAYS binary_classification
- "at risk of", "likely to have", "serious harm", "will experience" = binary_classification
- Raw incident/transaction rows → aggregated per entity = aggregation IS needed
- aggregation_key MUST be an entity identifier (e.g. establishment_id, customer_id)
  NEVER use category codes (naics_code, sic_code, zip_code) as aggregation key
- Target from a FUTURE dataset = SECONDARY_FILE yes
- Always cite the explicit metric if named in the brief
""")

EXTRACT_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert data science task specification analyst.

Given a task brief and dataset information, produce a complete JSON task
specification by reasoning step-by-step:

1. BUSINESS CONTEXT — what is the org trying to achieve?
2. PREDICTION GOAL — what exactly is predicted, per which entity?
3. TASK TYPE — binary (yes/no per entity), multiclass, or regression?
   RULE: "predict which [entities] will [experience event]" = binary_classification.
4. DATA STRUCTURE — is raw data at incident/transaction level needing aggregation?
5. TARGET DEFINITION — derived from a second file? What condition?
6. EVALUATION — metric explicitly stated or best choice for the problem.
7. CONSTRAINTS & DELIVERABLES — leakage rules, output files, required steps.

Output ONLY a JSON object with these keys:
  task_type, task_type_confidence, target_column, target_condition,
  aggregation_needed, aggregation_key, evaluation_metric,
  task_description, specific_requirements, requires_secondary_file,
  reasoning (brief chain-of-thought, 3-5 sentences)
""")


# ── Example builder ─────────────────────────────────────────────────────────────

def _build_user_msg(
    brief: str,
    columns: list[str],
    n_rows: int,
    outcome_hints: dict[str, dict],
) -> str:
    col_str = ", ".join(columns[:50])
    hint_str = ""
    if outcome_hints:
        lines = [f"  {c}: {v}" for c, v in outcome_hints.items()]
        hint_str = "\n=== POTENTIAL OUTCOME COLUMNS ===\n" + "\n".join(lines)
    return (
        f"=== TASK INSTRUCTIONS ===\n{brief.strip()}\n\n"
        f"=== DATASET STATISTICS ===\n"
        f"Columns ({len(columns)}): {col_str}\n"
        f"Row count: {n_rows}"
        f"{hint_str}"
    )


def _scope_example(
    brief: str,
    columns: list[str],
    n_rows: int,
    outcome_hints: dict[str, dict],
    scope: str,           # the full assistant response including reasoning + SCOPE ANALYSIS block
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_msg(brief, columns, n_rows, outcome_hints)},
            {"role": "assistant", "content": scope.strip()},
        ]
    }


def _extract_example(
    brief: str,
    columns: list[str],
    n_rows: int,
    outcome_hints: dict[str, dict],
    spec: dict,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_msg(brief, columns, n_rows, outcome_hints)},
            {"role": "assistant", "content": json.dumps(spec, indent=2)},
        ]
    }


# ── Helper to build a scope block ─────────────────────────────────────────────

def _scope_block(
    business_goal: str,
    entity_unit: str,
    task_type: str,
    task_type_reason: str,
    target_column: str,
    aggregation: str,
    aggregation_key: str,
    metric: str,
    secondary_file: str,
    secondary_file_reason: str,
    specific_steps: str,
    reasoning: str = "",
) -> str:
    parts = []
    if reasoning:
        parts.append(reasoning.strip())
        parts.append("")
    parts.append("--- SCOPE ANALYSIS ---")
    parts.append(f"BUSINESS_GOAL: {business_goal}")
    parts.append(f"ENTITY_UNIT: {entity_unit}")
    parts.append(f"TASK_TYPE: {task_type}")
    parts.append(f"TASK_TYPE_REASON: {task_type_reason}")
    parts.append(f"TARGET_COLUMN: {target_column}")
    parts.append(f"AGGREGATION: {aggregation}")
    parts.append(f"AGGREGATION_KEY: {aggregation_key}")
    parts.append(f"METRIC: {metric}")
    parts.append(f"SECONDARY_FILE: {secondary_file}")
    parts.append(f"SECONDARY_FILE_REASON: {secondary_file_reason}")
    parts.append(f"SPECIFIC_STEPS: {specific_steps}")
    parts.append("--- END SCOPE ---")
    return "\n".join(parts)


# ── Seed examples ─────────────────────────────────────────────────────────────

# WorkSafe brief and columns (reused for both scope and extract seeds)
_WS_BRIEF = """\
You are a data scientist at WorkSafe. Your team's mission is to help inspectors
target businesses that pose the greatest risk of harm to workers.
You have a dataset of all individual workplace incidents reported by 1,000 US
businesses during 2023. Build a model to predict which of these businesses
are likely to have a serious harm event in Q1 2024.

Files:
  incidents_2023.csv  — training data (individual incident records, 2023)
  incidents_2024_q1.csv — use this to create the binary target variable

Step 1: Aggregate incidents_2023.csv to one row per establishment_id.
Step 2: Risk ranking using severity hierarchy: Death > Days away from work >
        Job transfer or restriction > Other (least severe). Show top-10.
Step 3: From incidents_2024_q1.csv, create a binary flag
        had_serious_harm_in_2024 = 1 if any incident_outcome in [1,2,3].
Step 4: Build a binary classification model predicting had_serious_harm_in_2024.
Step 5: Evaluate using PR-AUC (primary) and ROC-AUC (secondary).
Deliverable: predictions_q1_2024.csv with columns: establishment_id,
p_serious_q1, risk_score_2023, rank_2023.
Leakage rule: use only data up to 31-Dec-2023 as features."""

_WS_COLS = [
    "establishment_id", "city", "state", "naics_code", "naics_year",
    "industry_description", "establishment_type", "size",
    "annual_average_employees", "total_hours_worked", "soc_code",
    "soc_description", "soc_reviewed", "soc_probability",
    "date_of_incident", "incident_outcome", "dafw_num_away",
    "djtr_num_tr", "type_of_incident", "time_started_work",
    "time_of_incident", "time_unknown", "date_of_death",
    "incident_month", "incident_year",
]

_WS_HINTS = {"incident_outcome": {1: 4, 2: 7420, 3: 1823, 4: 5286}}

SCOPE_SEEDS: list[dict] = [

    # ── 1. WorkSafe — binary, secondary file, entity agg, PR-AUC ─────────────
    _scope_example(
        brief=_WS_BRIEF,
        columns=_WS_COLS,
        n_rows=14533,
        outcome_hints=_WS_HINTS,
        scope=_scope_block(
            business_goal="Help inspectors target establishments most likely to have serious worker harm events in Q1 2024.",
            entity_unit="per establishment",
            task_type="binary_classification",
            task_type_reason='"predict which of these businesses are likely to have a serious harm event"',
            target_column="DERIVE: had_serious_harm_in_2024 = 1 if incident_outcome in [1,2,3] else 0, from incidents_2024_q1.csv",
            aggregation="needed",
            aggregation_key="establishment_id",
            metric="pr_auc",
            secondary_file="yes",
            secondary_file_reason="incidents_2024_q1.csv holds the future-period labels used to derive the target",
            specific_steps="risk_ranking_severity_hierarchy | output_predictions_q1_2024_csv | leakage_guard_2023_only",
            reasoning=(
                "The brief says 'predict which businesses are likely to have a serious harm event' — "
                "this is a binary yes/no per establishment, not a multiclass problem. "
                "The raw data is at the incident level (one row per incident) so aggregation "
                "to establishment_id is mandatory. The target must be derived from a future file "
                "(incidents_2024_q1.csv) making this a secondary-file task. "
                "PR-AUC is explicitly named as the primary metric."
            ),
        ),
    ),

    # ── 2. Customer churn — binary, per customer, single file ────────────────
    _scope_example(
        brief="""\
Predict which customers are likely to churn in the next 30 days.
The dataset contains one row per customer interaction. Aggregate to one row
per customer_id using their last 90 days of activity.
Target: churned (1 = churned within 30 days of observation window end, 0 = retained).
The churned column is already present in the dataset.
Evaluate with ROC-AUC. Use a logistic regression baseline and a gradient-boosted
tree model. Report feature importances.""",
        columns=[
            "customer_id", "interaction_date", "channel", "product_category",
            "amount_spent", "session_duration", "pages_visited",
            "support_calls", "churned", "tenure_days", "subscription_tier",
            "last_login_days_ago", "nps_score", "region",
        ],
        n_rows=250000,
        outcome_hints={"churned": {0: 212000, 1: 38000}},
        scope=_scope_block(
            business_goal="Identify customers likely to churn in the next 30 days so the retention team can intervene.",
            entity_unit="per customer",
            task_type="binary_classification",
            task_type_reason='"predict which customers are likely to churn" — binary yes/no per entity',
            target_column="churned",
            aggregation="needed",
            aggregation_key="customer_id",
            metric="roc_auc",
            secondary_file="no",
            secondary_file_reason="n/a — churned column already present in the dataset",
            specific_steps="aggregate_last_90_days | logistic_regression_baseline | feature_importances",
            reasoning=(
                "'Predict which customers are likely to churn' = binary per customer. "
                "Data is interaction-level so aggregation by customer_id to 90-day summary rows is needed. "
                "Target 'churned' already exists. ROC-AUC explicit."
            ),
        ),
    ),

    # ── 3. Hospital readmission — binary, imbalanced, PR-AUC ─────────────────
    _scope_example(
        brief="""\
Use historical inpatient records to build a 30-day readmission risk model.
Each row is a single hospital visit. Aggregate visits per patient_id to create
one feature row per patient.
A patient is at risk if they were readmitted within 30 days of discharge.
The column readmitted_30d (1/0) in the data marks this outcome.
Because readmissions are rare (~8%), please use Average Precision (AP) as
your evaluation metric. Use SHAP values to explain the top predictors.""",
        columns=[
            "patient_id", "admission_date", "discharge_date", "diagnosis_code",
            "procedure_code", "length_of_stay", "age", "gender",
            "primary_payer", "department", "num_medications", "num_lab_tests",
            "readmitted_30d", "hospital_id", "attending_physician_id",
        ],
        n_rows=85000,
        outcome_hints={"readmitted_30d": {0: 78200, 1: 6800}},
        scope=_scope_block(
            business_goal="Predict 30-day hospital readmission risk per patient to enable targeted post-discharge follow-up.",
            entity_unit="per patient",
            task_type="binary_classification",
            task_type_reason='"A patient is at risk if readmitted within 30 days" — binary per patient',
            target_column="readmitted_30d",
            aggregation="needed",
            aggregation_key="patient_id",
            metric="pr_auc",
            secondary_file="no",
            secondary_file_reason="n/a — readmitted_30d column already in data",
            specific_steps="aggregate_visits_per_patient | shap_explanation",
            reasoning=(
                "Binary readmission per patient (~8% positive = imbalanced). "
                "Average Precision (pr_auc) explicitly required due to imbalance. "
                "Aggregation by patient_id from visit-level rows is mandatory."
            ),
        ),
    ),

    # ── 4. Loan default — binary, no aggregation, leakage rule ───────────────
    _scope_example(
        brief="""\
Build a model to predict whether a loan will default within 12 months.
The dataset has one row per loan. No aggregation is needed.
Default is defined as payment_status == 'default' or 'charged_off'.
Do not use any columns collected after the loan origination date
(e.g. collection_status, settlement_amount) to avoid data leakage.
Use ROC-AUC for evaluation. Compare logistic regression, random forest,
and XGBoost. Export a CSV with loan_id and predicted default probability.""",
        columns=[
            "loan_id", "borrower_id", "loan_amount", "interest_rate",
            "term_months", "credit_score", "annual_income", "dti_ratio",
            "employment_length", "home_ownership", "loan_purpose",
            "origination_date", "payment_status", "collection_status",
            "settlement_amount", "installment", "grade", "sub_grade",
        ],
        n_rows=500000,
        outcome_hints={"payment_status": {"current": 420000, "default": 55000, "charged_off": 25000}},
        scope=_scope_block(
            business_goal="Predict binary loan default within 12 months to inform credit underwriting decisions.",
            entity_unit="per loan",
            task_type="binary_classification",
            task_type_reason='"predict whether a loan will default" — binary yes/no per loan',
            target_column="DERIVE: payment_status in ['default', 'charged_off'] → 1 else 0",
            aggregation="not_needed",
            aggregation_key="none",
            metric="roc_auc",
            secondary_file="no",
            secondary_file_reason="n/a — target derivable from payment_status column in same file",
            specific_steps="leakage_guard_post_origination | compare_lr_rf_xgb | export_loan_id_proba_csv",
            reasoning=(
                "One row per loan, no aggregation. Target is binary: default/charged_off=1, current=0. "
                "ROC-AUC explicit. Leakage constraint: exclude collection_status and settlement_amount."
            ),
        ),
    ),

    # ── 5. Equipment failure — binary, sensor aggregation ────────────────────
    _scope_example(
        brief="""\
Predict which machines in a manufacturing plant are likely to fail in the
next 7 days. Sensor readings are logged every 5 minutes (one row per reading).
Aggregate the last 24h of readings per machine_id to create features.
The failure_label column (0/1) is populated from the maintenance log:
1 if the machine failed within 7 days of the reading window end.
Failures are rare (<3%). Use F1 score on the positive class for evaluation.
Provide a ranked list of machines most at risk.""",
        columns=[
            "machine_id", "timestamp", "temperature", "vibration", "pressure",
            "rpm", "voltage", "current", "oil_level", "noise_db",
            "failure_label", "maintenance_cycle_days", "machine_type",
            "plant_id", "shift",
        ],
        n_rows=2000000,
        outcome_hints={"failure_label": {0: 1940000, 1: 60000}},
        scope=_scope_block(
            business_goal="Identify machines at risk of failure in the next 7 days for proactive maintenance scheduling.",
            entity_unit="per machine",
            task_type="binary_classification",
            task_type_reason='"predict which machines are likely to fail" — binary per entity',
            target_column="failure_label",
            aggregation="needed",
            aggregation_key="machine_id",
            metric="f1",
            secondary_file="no",
            secondary_file_reason="n/a — failure_label already in dataset from maintenance log",
            specific_steps="aggregate_last_24h_per_machine | ranked_risk_list",
            reasoning=(
                "'Predict which machines are likely to fail' = binary. "
                "F1 on positive class explicitly required (rare failures, <3%). "
                "Sensor stream must be aggregated per machine_id over last 24h."
            ),
        ),
    ),

    # ── 6. Insurance fraud — binary, precision-recall critical ───────────────
    _scope_example(
        brief="""\
Build a fraud detection model on auto insurance claims.
Each row is a claim. No aggregation needed.
A claim is fraudulent if the fraud_reported column is 'Y'.
The business requires high precision to avoid false accusations.
Use precision-recall AUC as the primary metric.
Flag the top 1% highest-risk claims for manual review.""",
        columns=[
            "claim_id", "policy_id", "claim_date", "incident_type",
            "incident_severity", "authorities_contacted", "number_of_vehicles",
            "bodily_injuries", "witnesses", "police_report_available",
            "total_claim_amount", "vehicle_claim", "property_claim",
            "injury_claim", "fraud_reported", "insured_age", "insured_education",
            "insured_occupation", "auto_year",
        ],
        n_rows=15000,
        outcome_hints={"fraud_reported": {"N": 11800, "Y": 3200}},
        scope=_scope_block(
            business_goal="Detect fraudulent auto insurance claims to flag them for manual review while minimising false accusations.",
            entity_unit="per claim",
            task_type="binary_classification",
            task_type_reason='"fraudulent if fraud_reported == Y" — binary yes/no per claim',
            target_column="DERIVE: fraud_reported == 'Y' → 1 else 0",
            aggregation="not_needed",
            aggregation_key="none",
            metric="pr_auc",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="flag_top_1pct_highest_risk",
            reasoning=(
                "Binary fraud label per claim. Precision-recall AUC explicit. "
                "No aggregation (one row per claim). High precision priority."
            ),
        ),
    ),

    # ── 7. House price — regression ──────────────────────────────────────────
    _scope_example(
        brief="""\
Predict the sale price of residential properties.
Each row is a property transaction. No aggregation required.
Use RMSE as the primary evaluation metric and R² as secondary.
The sale_price column is the target. Apply log-transformation to the target.
Feature importance analysis is required.""",
        columns=[
            "property_id", "sale_date", "sale_price", "bedrooms", "bathrooms",
            "sqft_living", "sqft_lot", "floors", "waterfront", "view",
            "condition", "grade", "sqft_above", "sqft_basement",
            "yr_built", "yr_renovated", "zipcode", "lat", "long",
            "sqft_living15", "sqft_lot15",
        ],
        n_rows=21613,
        outcome_hints={},
        scope=_scope_block(
            business_goal="Predict residential property sale prices to support valuation and market analysis.",
            entity_unit="per property",
            task_type="regression",
            task_type_reason='"predict the sale price" — continuous numeric target',
            target_column="sale_price",
            aggregation="not_needed",
            aggregation_key="none",
            metric="rmse",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="log_transform_target | r2_secondary_metric | feature_importance",
            reasoning=(
                "Continuous numeric target (sale_price) = regression. "
                "RMSE primary, R² secondary, both explicit. Log-transform of target required."
            ),
        ),
    ),

    # ── 8. Product defect — multiclass ───────────────────────────────────────
    _scope_example(
        brief="""\
Classify manufacturing defects into one of four categories:
  0 = No defect, 1 = Surface scratch, 2 = Dimensional error, 3 = Material flaw.
Each row is a single product inspection. No aggregation is needed.
Use macro F1-score for evaluation because classes are balanced.
Provide a confusion matrix and per-class precision/recall.""",
        columns=[
            "product_id", "inspection_date", "line_id", "operator_id",
            "weight_g", "length_mm", "width_mm", "height_mm",
            "surface_roughness", "hardness", "tensile_strength",
            "colour_deviation", "defect_category", "shift", "batch_id",
        ],
        n_rows=120000,
        outcome_hints={"defect_category": {0: 60000, 1: 20000, 2: 22000, 3: 18000}},
        scope=_scope_block(
            business_goal="Classify which type of manufacturing defect each product has to guide quality-control rework decisions.",
            entity_unit="per product inspection",
            task_type="multiclass_classification",
            task_type_reason='"one of four categories: 0=No defect, 1=Surface scratch, 2=Dimensional error, 3=Material flaw"',
            target_column="defect_category",
            aggregation="not_needed",
            aggregation_key="none",
            metric="f1",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="confusion_matrix | per_class_precision_recall | macro_f1",
            reasoning=(
                "Four named defect categories → multiclass (not binary). "
                "Macro F1 explicit. One row per inspection, no aggregation."
            ),
        ),
    ),

    # ── 9. Employee attrition — binary ───────────────────────────────────────
    _scope_example(
        brief="""\
Predict which employees are at risk of voluntarily leaving the company
within the next 6 months. The dataset has one row per employee.
The attrition column (Yes/No) is the target.
Use ROC-AUC for evaluation. Use SHAP to identify the top 5 drivers.
Present results to HR leadership in plain language.""",
        columns=[
            "employee_id", "age", "department", "education", "environment_satisfaction",
            "gender", "job_involvement", "job_level", "job_role", "job_satisfaction",
            "marital_status", "monthly_income", "monthly_rate", "num_companies_worked",
            "overtime", "percent_salary_hike", "performance_rating",
            "relationship_satisfaction", "stock_option_level",
            "total_working_years", "training_times_last_year",
            "work_life_balance", "years_at_company", "years_in_current_role",
            "years_since_last_promotion", "years_with_curr_manager", "attrition",
        ],
        n_rows=1470,
        outcome_hints={"attrition": {"No": 1233, "Yes": 237}},
        scope=_scope_block(
            business_goal="Identify employees most at risk of voluntary attrition to enable proactive HR retention actions.",
            entity_unit="per employee",
            task_type="binary_classification",
            task_type_reason='"predict which employees are at risk of voluntarily leaving" — binary per employee',
            target_column="DERIVE: attrition == 'Yes' → 1 else 0",
            aggregation="not_needed",
            aggregation_key="none",
            metric="roc_auc",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="shap_top5_drivers | plain_language_hr_summary",
            reasoning=(
                "'At risk of voluntarily leaving' = binary per entity. "
                "ROC-AUC explicit. One row per employee, no aggregation. "
                "SHAP explanation required."
            ),
        ),
    ),

    # ── 10. Weekly sales forecast — timeseries ────────────────────────────────
    _scope_example(
        brief="""\
Forecast weekly unit sales for each store-product combination for the next 4 weeks.
Data is at the weekly level with one row per (store_id, product_id, week).
Use MASE (Mean Absolute Scaled Error) as the evaluation metric.
Generate predictions for all store-product pairs present in the training data.""",
        columns=[
            "store_id", "product_id", "week_start_date", "units_sold",
            "price", "promo_active", "shelf_position", "competitor_price",
            "store_size", "region", "product_category", "brand",
        ],
        n_rows=180000,
        outcome_hints={},
        scope=_scope_block(
            business_goal="Forecast 4-week-ahead unit sales per store-product pair for inventory planning.",
            entity_unit="per (store, product) pair",
            task_type="timeseries_forecasting",
            task_type_reason='"Forecast weekly unit sales... for the next 4 weeks" — time-ordered future prediction',
            target_column="units_sold",
            aggregation="not_needed",
            aggregation_key="none",
            metric="mase",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="4_week_ahead_forecast | all_store_product_pairs",
            reasoning=(
                "Forecasting future values of units_sold over time = timeseries_forecasting. "
                "MASE explicit. One row per (store, product, week); no further aggregation needed."
            ),
        ),
    ),

    # ── 11. Network intrusion — binary, high recall ───────────────────────────
    _scope_example(
        brief="""\
Build an intrusion detection model on network connection logs.
Each row is a single network connection. Predict whether a connection
is malicious (label = 1) or benign (label = 0).
Because the cost of missing an attack is very high, optimise for recall.
Use F1-score (macro) for the primary metric. Attacks are ~5% of connections.""",
        columns=[
            "connection_id", "duration", "protocol_type", "service", "flag",
            "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
            "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "is_host_login", "is_guest_login",
            "count", "srv_count", "serror_rate", "label",
        ],
        n_rows=4900000,
        outcome_hints={"label": {0: 4655000, 1: 245000}},
        scope=_scope_block(
            business_goal="Detect malicious network connections to trigger security alerts with minimal missed attacks.",
            entity_unit="per connection",
            task_type="binary_classification",
            task_type_reason='"malicious (label=1) or benign (label=0)" — explicit binary labels',
            target_column="label",
            aggregation="not_needed",
            aggregation_key="none",
            metric="f1",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="optimise_for_recall | macro_f1",
            reasoning=(
                "Binary malicious/benign per connection. F1 macro explicit. "
                "No aggregation. ~5% positive — imbalanced but F1 explicitly requested."
            ),
        ),
    ),

    # ── 12. Patient disease severity — ordinal multiclass ─────────────────────
    _scope_example(
        brief="""\
Classify patients into disease severity levels: mild, moderate, severe, critical.
Each row represents a patient assessment (one per patient).
Use weighted F1-score because classes are imbalanced.
Provide a calibration curve for the probability outputs.""",
        columns=[
            "patient_id", "assessment_date", "age", "bmi", "blood_pressure_sys",
            "blood_pressure_dia", "heart_rate", "spo2", "temperature",
            "creatinine", "wbc_count", "platelet_count", "alt", "ast",
            "severity_label", "comorbidities_count", "icu_admission",
        ],
        n_rows=35000,
        outcome_hints={"severity_label": {"mild": 14000, "moderate": 11000,
                                           "severe": 7000, "critical": 3000}},
        scope=_scope_block(
            business_goal="Classify disease severity to triage patients into appropriate care levels.",
            entity_unit="per patient",
            task_type="multiclass_classification",
            task_type_reason='"four categories: mild, moderate, severe, critical" — 4 named classes',
            target_column="severity_label",
            aggregation="not_needed",
            aggregation_key="none",
            metric="f1",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="weighted_f1 | calibration_curve",
            reasoning=(
                "Four named severity levels = multiclass (not binary). "
                "Weighted F1 explicit due to imbalance. Calibration curve required."
            ),
        ),
    ),

    # ── 13. Telecom churn — binary, secondary file ────────────────────────────
    _scope_example(
        brief="""\
Predict which telecom subscribers will churn in the next month.
Training data: usage_history.csv — one row per subscriber per month.
Target labels: churn_labels.csv — contains subscriber_id and churned (0/1)
for the prediction month. This file must not be used as a training feature.
Aggregate usage_history.csv to one row per subscriber_id.
Use PR-AUC. Apply SMOTE to handle class imbalance.""",
        columns=[
            "subscriber_id", "month", "plan_type", "monthly_charge",
            "data_usage_gb", "call_minutes", "sms_count", "roaming_charges",
            "customer_service_calls", "contract_type", "payment_method",
            "tenure_months", "region", "device_type",
        ],
        n_rows=2400000,
        outcome_hints={},
        scope=_scope_block(
            business_goal="Predict which subscribers will churn next month to enable targeted retention campaigns.",
            entity_unit="per subscriber",
            task_type="binary_classification",
            task_type_reason='"predict which telecom subscribers will churn" — binary per subscriber',
            target_column="DERIVE: churned == 1 from churn_labels.csv",
            aggregation="needed",
            aggregation_key="subscriber_id",
            metric="pr_auc",
            secondary_file="yes",
            secondary_file_reason="churn_labels.csv holds the target labels for the prediction month",
            specific_steps="aggregate_usage_per_subscriber | join_churn_labels | smote_imbalance",
            reasoning=(
                "'Predict which subscribers will churn' = binary per entity. "
                "Target comes from separate churn_labels.csv. Aggregation by subscriber_id required. "
                "PR-AUC explicit. SMOTE for imbalance."
            ),
        ),
    ),

    # ── 14. Energy consumption — regression, building aggregation ─────────────
    _scope_example(
        brief="""\
Predict next month's energy consumption (kWh) for each building.
Raw data contains hourly meter readings — one row per (building_id, hour).
Aggregate to monthly features per building_id.
Use MAE as the primary metric. Also report MAPE.
Identify the top 3 energy-intensive buildings.""",
        columns=[
            "building_id", "timestamp", "kwh_reading", "temperature_outside",
            "occupancy_rate", "hvac_status", "lighting_status",
            "building_type", "floor_area_sqm", "age_years",
        ],
        n_rows=8760000,
        outcome_hints={},
        scope=_scope_block(
            business_goal="Forecast monthly energy consumption per building to support efficiency planning and budget allocation.",
            entity_unit="per building",
            task_type="regression",
            task_type_reason='"predict next month\'s energy consumption (kWh)" — continuous numeric forecast per entity',
            target_column="DERIVE: monthly_sum of kwh_reading per building_id",
            aggregation="needed",
            aggregation_key="building_id",
            metric="mae",
            secondary_file="no",
            secondary_file_reason="n/a — target aggregated from same file",
            specific_steps="aggregate_hourly_to_monthly | mape_secondary | top3_energy_intensive",
            reasoning=(
                "Predicting a continuous quantity (kWh) = regression. "
                "MAE primary, MAPE secondary, both explicit. "
                "Hourly data must be aggregated monthly per building_id."
            ),
        ),
    ),

    # ── 15. Anomaly detection — unsupervised ──────────────────────────────────
    _scope_example(
        brief="""\
Identify anomalous transactions in a financial system. No labelled fraud data
is available. Use unsupervised anomaly detection.
Each row is a transaction. Flag the top 0.5% most anomalous transactions.
Use reconstruction error from an autoencoder as the anomaly score.
Report precision at k (k=100) as the evaluation metric.""",
        columns=[
            "transaction_id", "account_id", "timestamp", "amount",
            "merchant_category", "merchant_country", "transaction_type",
            "card_present", "hour_of_day", "day_of_week",
            "rolling_7d_avg_amount", "rolling_7d_count", "distance_from_home_km",
        ],
        n_rows=10000000,
        outcome_hints={},
        scope=_scope_block(
            business_goal="Surface the most anomalous financial transactions for manual investigation without labelled fraud data.",
            entity_unit="per transaction",
            task_type="anomaly_detection",
            task_type_reason='"No labelled fraud data is available. Use unsupervised anomaly detection."',
            target_column="none",
            aggregation="not_needed",
            aggregation_key="none",
            metric="pr_auc",
            secondary_file="no",
            secondary_file_reason="n/a",
            specific_steps="autoencoder_reconstruction_error | flag_top_0_5pct | precision_at_k_100",
            reasoning=(
                "No labels available = anomaly_detection (unsupervised). "
                "Autoencoder explicitly required. Per-transaction, no aggregation."
            ),
        ),
    ),
]

# ── Extract-format seeds (same 15 briefs, JSON output) ─────────────────────────

EXTRACT_SEEDS: list[dict] = [
    _extract_example(
        brief=_WS_BRIEF,
        columns=_WS_COLS,
        n_rows=14533,
        outcome_hints=_WS_HINTS,
        spec={
            "task_type": "binary_classification",
            "task_type_confidence": 0.97,
            "target_column": "had_serious_harm_in_2024",
            "target_condition": "incident_outcome in [1, 2, 3] → 1, else 0",
            "aggregation_needed": True,
            "aggregation_key": "establishment_id",
            "evaluation_metric": "pr_auc",
            "task_description": (
                "Predict whether each establishment will have a serious harm event "
                "in Q1 2024, using aggregated 2023 incident features."
            ),
            "specific_requirements": [
                "Aggregate incidents_2023.csv to one row per establishment_id",
                "Severity-weighted risk score: Death=4, DAFW=3, Job transfer=2, Other=1",
                "Top-10 risk establishments by risk score",
                "Derive target from incidents_2024_q1.csv",
                "Output predictions_q1_2024.csv: establishment_id, p_serious_q1, risk_score_2023, rank_2023",
            ],
            "requires_secondary_file": True,
            "reasoning": (
                "The task asks 'predict which businesses will have a serious harm event' — binary per entity. "
                "The target comes from a separate future-period file (incidents_2024_q1.csv). "
                "Aggregation is needed because raw data is incident-level. PR-AUC is explicitly required."
            ),
        },
    ),
]


# ── Augmentation ─────────────────────────────────────────────────────────────

def _augment(seed: dict, n: int = 3) -> list[dict]:
    """Lightweight paraphrase augmentation — rephrases the brief opening."""
    results = []
    original_user = seed["messages"][1]["content"]
    brief_line = original_user.split("\n")[1][:80] if "\n" in original_user else original_user[:80]
    prefixes = [
        "Objective: ", "Background: ", "Task summary: ",
        "Problem statement: ", "Project brief: ",
    ]
    for i in range(min(n, len(prefixes))):
        new_user = original_user.replace(brief_line, prefixes[i] + brief_line.lstrip(), 1)
        aug = {
            "messages": [
                seed["messages"][0],
                {"role": "user", "content": new_user},
                seed["messages"][2],
            ]
        }
        results.append(aug)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/finetuning/understand_task_scope.jsonl")
    parser.add_argument(
        "--mode",
        choices=["scope", "extract", "both"],
        default="scope",
        help=(
            "scope   → structured SCOPE ANALYSIS format (for qwen3.6:latest)\n"
            "extract → flat JSON RawSpec format (for extraction validation)\n"
            "both    → write both files (--out used as prefix)"
        ),
    )
    parser.add_argument("--n_augment", type=int, default=2,
                        help="Augmented copies per seed (0 = no augmentation)")
    args = parser.parse_args()

    def _write(seeds: list[dict], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        examples = list(seeds)
        if args.n_augment > 0:
            for seed in seeds:
                examples.extend(_augment(seed, args.n_augment))
        random.shuffle(examples)
        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} examples → {out_path}")
        print(f"  Seeds: {len(seeds)} | Augmented: {len(examples) - len(seeds)}")

        types: dict[str, int] = {}
        for ex in examples:
            content = ex["messages"][2]["content"]
            if "TASK_TYPE:" in content:
                for line in content.splitlines():
                    if line.strip().startswith("TASK_TYPE:"):
                        t = line.split(":", 1)[1].strip()
                        types[t] = types.get(t, 0) + 1
                        break
            else:
                try:
                    spec = json.loads(content)
                    t = spec.get("task_type", "unknown")
                    types[t] = types.get(t, 0) + 1
                except Exception:
                    pass
        print("\nTask type distribution:")
        for t, cnt in sorted(types.items()):
            print(f"  {t:35s}: {cnt}")

    out_path = Path(args.out)

    if args.mode == "scope":
        _write(SCOPE_SEEDS, out_path)
    elif args.mode == "extract":
        _write(EXTRACT_SEEDS, out_path)
    else:  # both
        stem = out_path.stem
        suffix = out_path.suffix
        parent = out_path.parent
        _write(SCOPE_SEEDS, parent / f"{stem}_scope{suffix}")
        _write(EXTRACT_SEEDS, parent / f"{stem}_extract{suffix}")


if __name__ == "__main__":
    main()
