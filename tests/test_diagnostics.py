"""Tests for simulator-artifact diagnostics and ablation feature sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_risk.diagnostics import (
    AMOUNT_EQUALS_BALANCE_ATOL,
    amount_equals_origin_balance,
    drain_pattern_summary,
)
from fraud_risk.features import (
    BALANCE_DRAIN_FEATURE_NAMES,
    ENGINEERED_MODEL_FEATURES,
    LOGREG_TRANSFORMED_FEATURES,
    XGB_CORE_FEATURES,
    XGB_FULL_FEATURES,
    XGB_TRANSFORMED_FEATURES,
    prepare_model_frame,
    validate_safe_feature_names,
)
from fraud_risk.modeling import build_logistic_feature_pipeline, build_xgboost_pipeline


def test_amount_equals_balance_uses_isclose_tolerance() -> None:
    amount = np.array([100.0, 100.0, 100.0 + AMOUNT_EQUALS_BALANCE_ATOL / 2, 101.0])
    balance = np.array([100.0, 99.0, 100.0, 100.0])
    mask = amount_equals_origin_balance(amount, balance)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_drain_pattern_summary_separates_fraud_and_legit() -> None:
    frame = pd.DataFrame(
        {
            "amount": [100.0, 50.0, 200.0, 10.0],
            "oldbalanceOrg": [100.0, 0.0, 50.0, 10.0],
            "isFraud": [1, 1, 0, 0],
        }
    )
    summary = drain_pattern_summary(frame, split_name="train")
    assert set(summary["label"]) == {"fraud", "legit"}
    fraud = summary.loc[summary["label"] == "fraud"].iloc[0]
    legit = summary.loc[summary["label"] == "legit"].iloc[0]
    assert fraud["count"] == 2
    assert legit["count"] == 2
    assert fraud["pct_amount_eq_balance"] == pytest.approx(0.5)
    assert fraud["pct_origin_balance_zero"] == pytest.approx(0.5)
    assert legit["pct_amount_exceeds_balance"] == pytest.approx(0.5)


def test_ablation_feature_sets_exclude_or_include_drain_features() -> None:
    assert XGB_CORE_FEATURES == ("type", "amount", "oldbalanceOrg")
    assert "amount_to_balance_ratio" not in XGB_TRANSFORMED_FEATURES
    assert "amount_exceeds_balance" not in XGB_TRANSFORMED_FEATURES
    assert "origin_balance_zero" in XGB_TRANSFORMED_FEATURES
    assert set(BALANCE_DRAIN_FEATURE_NAMES).issubset(set(XGB_FULL_FEATURES))
    assert XGB_FULL_FEATURES == ENGINEERED_MODEL_FEATURES
    assert "amount_to_balance_ratio" not in LOGREG_TRANSFORMED_FEATURES
    assert "amount_exceeds_balance" not in LOGREG_TRANSFORMED_FEATURES


def test_prepare_model_frame_respects_feature_subset() -> None:
    frame = pd.DataFrame(
        {
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [100.0, 50.0],
            "oldbalanceOrg": [100.0, 0.0],
            "isFraud": [1, 0],
        }
    )
    core = prepare_model_frame(frame, XGB_CORE_FEATURES)
    assert list(core.columns) == list(XGB_CORE_FEATURES)
    transformed = prepare_model_frame(frame, XGB_TRANSFORMED_FEATURES)
    assert list(transformed.columns) == list(XGB_TRANSFORMED_FEATURES)
    assert "amount_to_balance_ratio" not in transformed.columns


def test_validate_safe_feature_names_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown or disallowed"):
        validate_safe_feature_names(("type", "newbalanceOrig"))


def test_xgboost_and_logreg_accept_ablation_subsets() -> None:
    frame = pd.DataFrame(
        {
            "type": ["TRANSFER", "CASH_OUT"] * 10,
            "amount": [float(i + 1) for i in range(20)],
            "oldbalanceOrg": [float(i) for i in range(20)],
            "isFraud": [1 if i % 5 == 0 else 0 for i in range(20)],
        }
    )
    y = frame["isFraud"]
    X = prepare_model_frame(frame, XGB_TRANSFORMED_FEATURES)
    xgb = build_xgboost_pipeline(
        scale_pos_weight=float((y == 0).sum()) / float((y == 1).sum()),
        feature_names=XGB_TRANSFORMED_FEATURES,
    )
    xgb.fit(X, y)
    assert xgb.predict_proba(X).shape == (len(X), 2)

    logreg = build_logistic_feature_pipeline(LOGREG_TRANSFORMED_FEATURES)
    logreg.fit(X, y)
    assert logreg.predict_proba(X).shape == (len(X), 2)
