# Fraud Risk API

End-to-end machine learning project demonstrating how to build, validate, deploy, and monitor a fraud risk scoring service.

## Business Problem

Financial institutions must decide whether to authorize a payment in near real time. This project will train a model that estimates the probability that a transaction is fraudulent **using only information available immediately before transaction authorization**. Model scores will inform risk decisions; they will not replace explicit business rules or human review.

## Dataset

The project uses the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic mobile-money fraud dataset. The raw CSV is **not committed** to this repository. Download it manually and place it under `data/raw/` (see `data/README.md`).

## Prediction Moment

**Prediction moment: immediately before transaction authorization.**

Predictors may only use information that would actually be available at that moment. Post-transaction balances, outcomes, and existing fraud-rule flags are treated as leakage or out of scope unless the feature contract documents a strong, explicit exception.

See `docs/feature_contract.md` for the full V1 modeling contract.

## V1 Modeling Scope

**V1 modeling scope: `TRANSFER` and `CASH_OUT` transactions only.**

The PaySim audit found all labeled fraud in those two types (`PAYMENT`, `CASH_IN`, and `DEBIT` had zero positive fraud examples in this dataset). Restricting V1 to `TRANSFER` and `CASH_OUT` avoids an artificially easy classification problem driven by types with no fraud labels.

This is a **property of the PaySim simulation and a project modeling-scope decision**. It does **not** mean those other types can never be fraudulent in real financial systems.

## Baseline Modeling Approach

Task 2 builds a deliberately conservative leakage-safe baseline (Model A):

- **Predictors:** `type`, `amount`, `oldbalanceOrg` only
- **Target:** `isFraud`
- **Temporal key:** `step` for chronological train / validation / test splits (~70% / 15% / 15% of rows via cumulative step counts; not used as a predictor)
- **Model:** scikit-learn Logistic Regression with one-hot encoding, numeric standardization, and balanced class weights
- **Evaluation:** PR-AUC / Average Precision, ROC-AUC, precision, recall, F1; provisional threshold chosen on validation only

IDs, post-transaction balances, destination pre-balance, and `isFlaggedFraud` are excluded.

## Feature Engineering and Model Comparison

Task 3 asks whether false positives can be reduced while holding approximately **80% fraud recall**. It compares three configurations under the same temporal protocol and validation-only operating policy (highest precision subject to recall ≥ 80%):

| Model | Description |
| --- | --- |
| **A** | Frozen Task 2 Logistic Regression baseline (`type`, `amount`, `oldbalanceOrg`) |
| **B** | Logistic Regression on baseline features plus small leakage-safe engineered features (`log_amount`, `log_origin_balance`, `amount_to_balance_ratio`, `origin_balance_zero`, `amount_exceeds_balance`) |
| **C** | XGBoost on the same engineered feature set, with `scale_pos_weight` from the **train** split only |

Primary operational comparison: **false positives at approximately 80% recall** (not accuracy or ROC-AUC maximization). XGBoost scores are used for ranking/thresholding and are **not** treated as calibrated probabilities.

### PaySim limitation (Task 3A)

PaySim is **synthetic** and implements a specific account-takeover scenario in which fraudsters often drain the victim’s balance. Engineered features such as `amount_to_balance_ratio` and `amount_exceeds_balance` can align strongly with that label-generation mechanism. Very strong Task 3 scores are therefore treated as a **simulator artifact / synthetic shortcut** risk—not as temporal leakage, and not as proof the same features will generalize to real-world fraud. See `notebooks/03a_simulator_artifact_audit.ipynb` for the pattern audit and feature ablation. This is an example of why diagnostics and domain understanding matter before treating a high leaderboard score as a deployment decision.

## Project Principles

- Prevent target leakage
- Use temporal validation
- Start with simple baselines
- Evaluate metrics appropriate for fraud detection (e.g. precision/recall, PR-AUC)
- Separate model scores from business decisions
- Keep training and deployment reproducible

## Repository Structure

```text
fraud-risk-api/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   ├── README.md
│   └── raw/                 # place PaySim CSV here (not in Git)
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_logistic_baseline.ipynb
│   ├── 03_feature_engineering_xgboost.ipynb
│   └── 03a_simulator_artifact_audit.ipynb
├── src/
│   └── fraud_risk/
│       ├── __init__.py
│       ├── data.py
│       ├── dataset.py
│       ├── diagnostics.py
│       ├── features.py
│       └── modeling.py
├── tests/
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_diagnostics.py
│   └── test_features.py
└── docs/
    └── feature_contract.md
```

## Local Setup

Reference development version: **Python 3.13** (`requires-python >= 3.13`).

```bash
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Notebooks

1. Place the PaySim CSV in `data/raw/` (exactly one `.csv` file).
2. Register the project kernel if needed: `python -m ipykernel install --user --name=fraud-risk-api`
3. Open and run:
   - `notebooks/01_data_audit.ipynb` — exploratory audit
   - `notebooks/02_logistic_baseline.ipynb` — leakage-safe logistic baseline
   - `notebooks/03_feature_engineering_xgboost.ipynb` — engineered features + XGBoost comparison
   - `notebooks/03a_simulator_artifact_audit.ipynb` — PaySim drain-pattern audit + feature ablation

Notebooks import reusable helpers from `fraud_risk`; they do not reimplement core data or modeling logic.

## Tests

```bash
pytest
ruff check .
```

Tests use synthetic CSVs / DataFrames and do not require the Kaggle dataset.

## Current Status

**Task 1 — Data audit and feature contract.** Complete.

**Task 2 — Modeling dataset and Logistic Regression baseline.** Complete. Model A remains the fixed baseline.

**Task 3 — Leakage-safe feature engineering and XGBoost comparison.** Complete. Engineered Logistic Regression (Model B) and XGBoost (Model C) are compared against Model A on false positives at approximately 80% recall.

**Task 3A — Simulator artifact and feature ablation audit.** Complete. Quantifies PaySim account-drain alignment and ablates explicit balance-drain features before treating near-perfect scores as deployment-ready.

No FastAPI service, Docker image, CI/CD pipeline, AWS deployment, probability calibration, or production monitoring exists yet.
