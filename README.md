# Fraud Risk API

End-to-end machine learning project demonstrating how to build, validate, deploy, and monitor a fraud risk scoring service.

## Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Prediction Moment](#prediction-moment)
- [Modeling Scope](#modeling-scope)
- [Leakage-Safe Logistic Baseline](#leakage-safe-logistic-baseline)
- [Feature Engineering and Model Comparison](#feature-engineering-and-model-comparison)
- [Project Principles](#project-principles)
- [Repository Structure](#repository-structure)
- [Local Setup](#local-setup)
- [Running Notebooks](#running-notebooks)
- [Frozen Model Artifact](#frozen-model-artifact)
- [Local FastAPI Service](#local-fastapi-service)
- [Docker](#docker)
- [Continuous Integration](#continuous-integration)
- [Tests](#tests)
- [Current Status](#current-status)

## Business Problem

Financial institutions must decide whether to authorize a payment in near real time. This project will train a model that estimates the probability that a transaction is fraudulent **using only information available immediately before transaction authorization**. Model scores will inform risk decisions; they will not replace explicit business rules or human review.

## Dataset

The project uses the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic mobile-money fraud dataset. The raw CSV is **not committed** to this repository. Download it manually and place it under `data/raw/` (see `data/README.md`).

### How to download the data?

Automatic Kaggle auth/download is **not** built into this repo. As an optional local shortcut, you can use the [Kaggle CLI](https://www.kaggle.com/docs/api) after configuring credentials (`kaggle.json` in `~/.kaggle/`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`):

```bash
# one-time: put kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME / KAGGLE_KEY)
pip install kaggle
kaggle datasets download -d ealaxi/paysim1 -p data/raw/ --unzip
```

Keep exactly one `.csv` under `data/raw/` afterward.

## Prediction Moment

**Prediction moment: immediately before transaction authorization.**

Predictors may only use information that would actually be available at that moment. Post-transaction balances, outcomes, and existing fraud-rule flags are treated as leakage or out of scope unless the feature contract documents a strong, explicit exception.

See `docs/feature_contract.md` for the full V1 modeling contract.

## Modeling Scope

**Modeling scope: `TRANSFER` and `CASH_OUT` transactions only.**

The PaySim audit found all labeled fraud in those two types (`PAYMENT`, `CASH_IN`, and `DEBIT` had zero positive fraud examples in this dataset). Restricting to `TRANSFER` and `CASH_OUT` avoids an artificially easy classification problem driven by types with no fraud labels.

This is a **property of the PaySim simulation and a project modeling-scope decision**. It does **not** mean those other types can never be fraudulent in real financial systems.

## Leakage-Safe Logistic Baseline

**Goal:** Establish a simple, trustworthy reference model before adding engineered features or stronger algorithms.

Builds the first fraud classifier using only information available at authorization time, trains and evaluates it with chronological splits, and freezes that setup as **Model A** for later comparison.

**Model A (the baseline):**
- **Algorithm:** scikit-learn Logistic Regression (one-hot encoding, numeric standardization, balanced class weights)
- **Predictors:** `type`, `amount`, `oldbalanceOrg` only
- **Target:** `isFraud`
- **Splits:** chronological train / validation / test by `step` (~70% / 15% / 15% of rows via cumulative step counts; `step` is not a predictor)
- **Metrics:** PR-AUC / Average Precision, ROC-AUC, precision, recall, F1; provisional threshold chosen on validation only

**Excluded (leakage / out of scope):** IDs, post-transaction balances, destination pre-balance, and `isFlaggedFraud`.

Model A is intentionally minimal. Later tasks keep it frozen and measure whether richer features or XGBoost reduce false positives without losing roughly 80% fraud recall.

## Feature Engineering and Model Comparison

Task 3 asks whether false positives can be reduced while holding approximately **80% fraud recall**. It compares three configurations under the same temporal protocol and validation-only operating policy (highest precision subject to recall ≥ 80%):

| Model | Description |
| --- | --- |
| **A** | Task 2 reference: Logistic Regression on `type`, `amount`, `oldbalanceOrg` only (frozen; no new features) |
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
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── README.md
│   └── raw/                 # place PaySim CSV here (not in Git)
├── artifacts/
│   └── README.md            # generated model binaries are local-only
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_logistic_baseline.ipynb
│   ├── 03_feature_engineering_xgboost.ipynb
│   ├── 03a_simulator_artifact_audit.ipynb
│   └── 04_probability_calibration.ipynb
├── scripts/
│   ├── create_ci_smoke_artifact.py
│   └── ci_http_smoke.py
├── src/
│   └── fraud_risk/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       ├── calibration.py
│       ├── data.py
│       ├── dataset.py
│       ├── diagnostics.py
│       ├── features.py
│       ├── inference.py
│       ├── modeling.py
│       └── train_final.py
├── tests/
│   ├── test_api.py
│   ├── test_calibration.py
│   ├── test_ci_smoke_artifact.py
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_diagnostics.py
│   ├── test_docker.py
│   ├── test_features.py
│   └── test_inference.py
└── docs/
    ├── feature_contract.md
    └── inference_contract.md
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
   - `notebooks/04_probability_calibration.ipynb` — Platt calibration + inference contract

Notebooks import reusable helpers from `fraud_risk`; they do not reimplement core data or modeling logic.

## Frozen Model Artifact

`xgb-transformed-v1` is the frozen portfolio deployment model (XGB Transformed + Platt calibration + threshold `0.044`). Generated binary artifacts are **not** committed to Git; rebuild them locally.

### Generate the local artifact

Requires the PaySim CSV under `data/raw/`:

```bash
python -m fraud_risk.train_final
```

Writes `artifacts/xgb-transformed-v1/{model.joblib,calibrator.joblib,metadata.json}`.

### Run a local prediction

```bash
python -m fraud_risk.inference \
  --artifact-dir artifacts/xgb-transformed-v1 \
  --transaction-type TRANSFER \
  --amount 8500 \
  --origin-balance 9000
```

Output is compact JSON matching `docs/inference_contract.md` (`fraud_probability`, `decision`, `threshold`, `model_version`). Scoring does not load the PaySim dataset.

## Local FastAPI Service

### What this API does

The FastAPI service scores a single payment **immediately before authorization** and returns a calibrated fraud probability plus an illustrative review decision. Callers send only business fields (`transaction_type`, `amount`, `origin_balance`); the service owns feature engineering, Platt calibration, and the frozen threshold. It does **not** authorize payments by itself—scores inform risk review alongside rules and human oversight.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check; reports whether the model artifact is loaded |
| `GET /model/info` | Model version, frozen threshold, allowed types, prediction moment |
| `POST /predict` | Score one `TRANSFER` or `CASH_OUT` transaction |
| `GET /docs` | Interactive OpenAPI / Swagger UI |

**Request** (`POST /predict`):

```json
{"transaction_type": "TRANSFER", "amount": 8500.0, "origin_balance": 9000.0}
```

**Response:**

```json
{
  "fraud_probability": 0.35,
  "decision": "review",
  "threshold": 0.044,
  "model_version": "xgb-transformed-v1"
}
```

`decision` is `review` when `fraud_probability >= threshold`, otherwise `pass`. Only `TRANSFER` and `CASH_OUT` are in scope; other types are rejected. Full contract: [`docs/inference_contract.md`](docs/inference_contract.md).

The API is a thin serving layer over `FraudPredictor`. It does not reimplement feature engineering, calibration, or threshold logic.

### Generate the artifact if missing

```bash
python -m fraud_risk.train_final
```

### Start the API

```bash
FRAUD_MODEL_DIR=artifacts/xgb-transformed-v1 uvicorn fraud_risk.api.main:app --reload
```

If `FRAUD_MODEL_DIR` is unset, the app defaults to `artifacts/xgb-transformed-v1`.

Interactive OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example prediction

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"transaction_type":"TRANSFER","amount":8500.0,"origin_balance":9000.0}'
```

### Manual integration smoke test

With the real local artifact (not required by the automated test suite):

1. `python -m fraud_risk.train_final` (if needed)
2. Start uvicorn as above
3. `GET /health` and `GET /model/info`
4. `POST /predict` for one `TRANSFER` and one `CASH_OUT`
5. Confirm `/docs` loads

## Docker

Package the frozen FastAPI service into a self-contained Linux image. The image includes the Python runtime, project dependencies, and a copy of the local `xgb-transformed-v1` artifact. PaySim, notebooks, and host virtualenvs are not required at container runtime.

The frozen artifact is **copied into the image during a local Docker build** but remains **excluded from Git**. Generate it on the host before building.

### 1. Prerequisite artifact

```bash
python -m fraud_risk.train_final
```

Confirm `artifacts/xgb-transformed-v1/{model.joblib,calibrator.joblib,metadata.json}` exist.

### 2. Build

```bash
docker build -t fraud-risk-api:local .
```

### 3. Run

```bash
docker run --rm --name fraud-risk-api -p 8000:8000 fraud-risk-api:local
```

No volume mounts are needed: the model is already inside the image.

### 4. Health check

`GET http://127.0.0.1:8000/health` should return `{"status":"ok","model_loaded":true}`. Docker also runs a built-in `HEALTHCHECK` against the same endpoint.

### 5. Swagger

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Stop

```bash
docker stop fraud-risk-api
```

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on pull requests, pushes to `main`, and manual workflow dispatch.

1. **python-quality** — `ruff check .` then `pytest` on Python 3.13.
2. **docker-smoke** — after quality passes, builds the Docker image with a synthetic `ci-smoke-v1` artifact, waits until Docker reports the container `healthy`, then runs `scripts/ci_http_smoke.py` against `/health`, `/model/info`, and one `/predict` (bounded retries for transient transport errors only).

CI does **not** use PaySim or the real portfolio binary. It generates a synthetic compatible package via `scripts/create_ci_smoke_artifact.py` (`model_version = ci-smoke-v1`) only to exercise the serving path. CI never evaluates model performance.

## Tests

```bash
pytest
ruff check .
```

Tests use synthetic CSVs / DataFrames (and a fake predictor for the API) and do not require the Kaggle dataset or the real XGBoost artifact.

## Current Status (Tasks):

**Task 1 — Data audit and feature contract.** Complete.

**Task 2 — Modeling dataset and Logistic Regression baseline.** Complete. Produced **Model A** (leakage-safe Logistic Regression on `type`, `amount`, `oldbalanceOrg`), which stays frozen as the comparison baseline for Task 3+.

**Task 3 — Leakage-safe feature engineering and XGBoost comparison.** Complete. Engineered Logistic Regression (Model B) and XGBoost (Model C) are compared against Model A on false positives at approximately 80% recall.

**Task 3A — Simulator artifact and feature ablation audit.** Complete. Quantifies PaySim account-drain alignment and ablates explicit balance-drain features before treating near-perfect scores as deployment-ready.

**Task 4 — Probability calibration and inference contract.** Complete. **XGB Transformed** is the selected portfolio deployment model; scores are Platt-calibrated before serving; the public inference API contract is defined in `docs/inference_contract.md`.

**Task 5 — Frozen model artifact and local inference pipeline.** Complete. Reproducible artifact generation (`python -m fraud_risk.train_final`) and local scoring (`FraudPredictor` / `python -m fraud_risk.inference`) are available for `xgb-transformed-v1`. Binary artifacts stay local (not in Git).

**Task 6 — FastAPI local model service.** Complete. Minimal local API (`/health`, `/model/info`, `/predict`) loads `FraudPredictor` once at startup and reuses it across requests.

**Task 7 — Dockerize the Fraud API.** Complete. A two-stage `Dockerfile` packages the runtime dependencies and frozen `xgb-transformed-v1` artifact into a non-root Linux image that serves the existing FastAPI app on port 8000.

**Task 8 — GitHub Actions Continuous Integration.** Complete. CI runs lint/tests, then a Docker serving smoke test using a synthetic `ci-smoke-v1` artifact: wait for Docker health, then HTTP contract checks via `scripts/ci_http_smoke.py` (PaySim and the real model binary stay out of Git and out of CI).

No AWS deployment, authentication, or production monitoring exists yet.
