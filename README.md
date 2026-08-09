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
│   └── 01_data_audit.ipynb
├── src/
│   └── fraud_risk/
│       ├── __init__.py
│       └── data.py
├── tests/
│   └── test_data.py
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

## Running the Data Audit Notebook

1. Place the PaySim CSV in `data/raw/` (exactly one `.csv` file).
2. Register the project kernel if needed: `python -m ipykernel install --user --name=fraud-risk-api`
3. Open and run `notebooks/01_data_audit.ipynb`.

The notebook imports `load_raw_data` from `fraud_risk.data`; it does not reimplement CSV discovery.

## Tests

```bash
pytest
ruff check .
```

Tests use synthetic CSVs and do not require the Kaggle dataset.

## Current Status

**Task 1 — Data audit and feature contract.**

No trained model, FastAPI service, Docker image, CI/CD pipeline, or AWS deployment exists yet.
