# Feature Contract (V1)

## Prediction moment

**Prediction moment: immediately before transaction authorization.**

A predictor may only use information that would actually be available at that moment. Anything that becomes known only after the transaction posts, after balances update, or as a result of a downstream fraud rule is **not** a valid V1 input unless a strong, documented exception applies.

This document distinguishes:

| Layer | Meaning |
| --- | --- |
| Dataset contents | What PaySim provides as columns in the synthetic CSV |
| Real-world availability | What we would realistically expect to know at authorization time |
| V1 decision | Whether we will use the column as a model input in the first modeling iteration |

## V1 modeling scope

**V1 modeling scope: `TRANSFER` and `CASH_OUT` transactions only.**

Audit snapshot (full PaySim CSV):

| Metric | Value |
| --- | --- |
| Total rows | 6,362,620 |
| Fraud cases | 8,213 |
| Fraud rate | ~0.1291% |
| Fraud in `TRANSFER` | 4,097 |
| Fraud in `CASH_OUT` | 4,116 |
| Fraud in `PAYMENT` / `CASH_IN` / `DEBIT` | 0 |

In this dataset, labeled fraud appears only in `TRANSFER` and `CASH_OUT`. Including types with zero positive fraud examples would make the classification problem artificially easy (the model could lean on type alone). V1 therefore **filters to those two types** before feature selection and training (dataset construction is deferred to a later task).

This reflects **PaySim’s simulation design and an explicit project scope choice**. It does **not** claim that `PAYMENT`, `CASH_IN`, or `DEBIT` can never be fraudulent in real financial systems.

## PaySim source columns

| Source feature | Interpretation | Available at prediction time? | V1 decision | Rationale |
| --- | --- | --- | --- | --- |
| `step` | Discrete simulation time unit (1 step ≈ 1 hour in PaySim). Useful as a temporal axis for ordering and validation. | **Needs review** — a wall-clock / session timestamp would usually be available; the exact PaySim `step` encoding is synthetic. | Candidate (temporal key / feature with care) | Prefer using time for **ordering and temporal splits**, not as a naive numeric leak of future hours. Derive safe calendar-like features later if justified. |
| `type` | Transaction type (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`). | Yes — known when the payment request is submitted. | Candidate (within V1 scope) | Available pre-authorization. V1 rows are restricted to `TRANSFER` and `CASH_OUT`; type may still distinguish those two. |
| `amount` | Transaction amount requested. | Yes — part of the authorization request. | Candidate | Core risk signal; available at prediction moment. |
| `nameOrig` | Originating account / customer identifier. | Yes — known for the initiating party. | Exclude (direct ID) | High-cardinality identifier. Do not use raw IDs in V1; optional later via aggregated history features computed only from past data. |
| `oldbalanceOrg` | Originator balance **before** the transaction. | **Needs review** — often available from ledger/core banking at auth time, but PaySim’s exact balance semantics are simulated. | Needs review | Likely usable if confirmed as pre-transaction ledger balance; validate against product constraints before promoting to Candidate. |
| `newbalanceOrig` | Originator balance **after** the transaction. | No — post-transaction / counterfactual balance. | Exclude | Post-transaction information; using it as a predictor would be target leakage relative to the prediction moment. |
| `nameDest` | Destination account / merchant / customer identifier. | Yes — destination is usually known at authorization. | Exclude (direct ID) | High-cardinality identifier. Same policy as `nameOrig` for V1. |
| `oldbalanceDest` | Destination balance **before** the transaction. | **Needs review** — often **not** available to the authorizing system for external destinations (especially merchants / other banks). | Needs review / likely Exclude | Even if present in PaySim, real-world availability is doubtful for many destination types. Do not treat as safe by default. |
| `newbalanceDest` | Destination balance **after** the transaction. | No — post-transaction balance. | Exclude | Clear post-transaction leakage relative to authorization-time scoring. |
| `isFraud` | Ground-truth fraud label (synthetic). | No — outcome / label. | Target | Supervision only; never a feature. |
| `isFlaggedFraud` | Existing PaySim business-rule flag (large TRANSFER blocked in the simulator). | Ambiguous / effectively a rule output — **not** a clean pre-model feature. | Exclude | Existing fraud-rule output. Using it would mix rule systems with the model and can leak rule logic; V1 learns from raw request fields instead. |

## Summary of V1 intent

**Row scope:** `TRANSFER` and `CASH_OUT` only.

**Likely inputs to explore first:** `type`, `amount`, and carefully justified pre-transaction originator context (if `oldbalanceOrg` passes review).

**Explicit non-inputs for V1:** `newbalanceOrig`, `newbalanceDest`, raw `nameOrig` / `nameDest`, `isFlaggedFraud`, and the label `isFraud`.

**Open questions / needs review:**

1. Is `oldbalanceOrg` reliably available in the real product flow at authorization, or only in the simulator?
2. Is `oldbalanceDest` ever available for the destination types we care about, or is it a PaySim-only convenience field?
3. How should `step` map to production timestamps and temporal validation windows?
4. Should destination *type* (e.g. merchant vs customer prefix in PaySim) be engineered as a coarse categorical without using the raw ID?

## Out of scope for this document

Feature engineering, encoding, scaling, train/test splitting, modeling-dataset materialization, and modeling are deferred to later tasks. This contract locks the prediction moment, the V1 transaction-type scope, and the initial column-level decisions.
