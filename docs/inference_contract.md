# Inference Contract (V1)

This document defines the public request/response interface that the future FastAPI fraud-risk scoring service must implement. **API implementation is out of scope for Task 4**; this is the contract only.

## Deployment model

| Item | Value |
| --- | --- |
| Selected model | **XGB Transformed** |
| Model version | `xgb-transformed-v1` |
| Scores served | **Calibrated** probabilities (Platt / sigmoid scaling) |
| Operating threshold | Frozen from **validation** calibrated probabilities only |

### Features used internally

`type`, `amount`, `oldbalanceOrg`, `log_amount`, `log_origin_balance`, `origin_balance_zero`

### Explicitly excluded (PaySim drain-artifact encodings)

`amount_to_balance_ratio`, `amount_exceeds_balance`

Clients never send engineered features. Feature engineering is owned by the model service.

## Request

The public request contains **business / source variables only**.

### Example

```json
{
  "transaction_type": "TRANSFER",
  "amount": 8500.0,
  "origin_balance": 9000.0
}
```

### Fields

| Field | Type | Rules |
| --- | --- | --- |
| `transaction_type` | string | Must be one of `TRANSFER`, `CASH_OUT` |
| `amount` | number | Finite, `>= 0` |
| `origin_balance` | number | Finite, `>= 0` |

### Request validation

- Reject requests whose `transaction_type` is not `TRANSFER` or `CASH_OUT`. Other PaySim types (`PAYMENT`, `CASH_IN`, `DEBIT`, …) are **outside V1 modeling scope** and must be rejected rather than silently scored.
- Reject non-finite values (`NaN`, `±Inf`) for `amount` or `origin_balance`.
- Reject negative `amount` or `origin_balance`.

## Internal feature mapping

Public fields map to internal PaySim-style column names, then to engineered predictors:

| Public field | Internal column |
| --- | --- |
| `transaction_type` | `type` |
| `amount` | `amount` |
| `origin_balance` | `oldbalanceOrg` |

Internal engineering (service-owned):

- `log_amount` = `log1p(amount)`
- `log_origin_balance` = `log1p(oldbalanceOrg)`
- `origin_balance_zero` = `1` if `oldbalanceOrg == 0`, else `0`

The client must **not** send these engineered fields.

## Response

### Example

```json
{
  "fraud_probability": 0.0327,
  "decision": "review",
  "threshold": 0.0185,
  "model_version": "xgb-transformed-v1"
}
```

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fraud_probability` | float in `[0, 1]` | **Calibrated** model probability of fraud |
| `decision` | string | `review` when `fraud_probability >= threshold`; otherwise `pass` |
| `threshold` | float | Frozen validation-selected operating threshold |
| `model_version` | string | Stable identifier, e.g. `xgb-transformed-v1` |

### Decision policy (illustrative)

`decision` is an **illustrative portfolio review policy** for this project: flag transactions whose calibrated probability meets or exceeds the frozen threshold for manual / secondary review.

It must **not** be described as a production financial authorization policy. Real authorization systems combine model scores with hard rules, limits, customer context, and human oversight.

## Temporal and calibration discipline (service expectations)

- The underlying XGBoost model is fit on an early chronological **model-fit** subset of training data.
- Platt scaling is fit on a later chronological **calibration** subset of training data only.
- The operating threshold is selected on **validation** calibrated probabilities (highest precision subject to recall ≥ 80% when feasible) and then frozen.
- The **test** period remains an untouched future holdout for one-shot evaluation; it must not be used to fit the model, calibrator, or threshold.

## Out of scope for this document

FastAPI / Pydantic implementation, model persistence, Docker, CI/CD, AWS, SHAP, monitoring, and hyperparameter search belong to later tasks.
