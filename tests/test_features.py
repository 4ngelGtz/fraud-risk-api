"""Tests for leakage-safe engineered features and Model B/C helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_risk.dataset import (
    BASELINE_FEATURES,
    EXCLUDED_BASELINE_COLUMNS,
    TARGET_COLUMN,
    chronological_train_val_test_split,
    select_baseline_frame,
)
from fraud_risk.features import (
    ENGINEERED_FEATURE_NAMES,
    ENGINEERED_MODEL_FEATURES,
    ZERO_BALANCE_RATIO_DENOMINATOR,
    add_engineered_features,
    amount_to_balance_ratio,
    prepare_engineered_model_frame,
    select_model_feature_frame,
)
from fraud_risk.modeling import (
    DEFAULT_RECALL_FLOOR,
    build_logistic_baseline_pipeline,
    build_logistic_engineered_pipeline,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
    select_operating_threshold,
    threshold_analysis,
)


def _synthetic_baseline_rows() -> pd.DataFrame:
    """Tiny frame covering zero balance, amount>balance, and normal cases."""
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
            "amount": [100.0, 50.0, 0.0, 200.0, 75.0, 10.0],
            "oldbalanceOrg": [0.0, 100.0, 0.0, 50.0, 75.0, 10.0],
            "isFraud": [1, 0, 0, 1, 0, 0],
            # Leakage columns present in a raw-like frame but must never be used.
            "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "nameDest": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "newbalanceOrig": [0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            "newbalanceDest": [100.0, 150.0, 0.0, 250.0, 150.0, 20.0],
            "oldbalanceDest": [0.0, 100.0, 0.0, 50.0, 75.0, 10.0],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
        }
    )


def test_log_features_match_log1p() -> None:
    frame = _synthetic_baseline_rows()
    enriched = add_engineered_features(frame)
    np.testing.assert_allclose(
        enriched["log_amount"].to_numpy(),
        np.log1p(frame["amount"].to_numpy(dtype=float)),
    )
    np.testing.assert_allclose(
        enriched["log_origin_balance"].to_numpy(),
        np.log1p(frame["oldbalanceOrg"].to_numpy(dtype=float)),
    )


def test_amount_exceeds_balance_correct() -> None:
    frame = _synthetic_baseline_rows()
    enriched = add_engineered_features(frame)
    expected = (frame["amount"] > frame["oldbalanceOrg"]).astype(np.int8)
    pd.testing.assert_series_equal(
        enriched["amount_exceeds_balance"],
        expected,
        check_names=False,
    )
    # Explicit known cases: amount 200 > balance 50; amount 75 == balance 75 → False.
    assert enriched.loc[3, "amount_exceeds_balance"] == 1
    assert enriched.loc[4, "amount_exceeds_balance"] == 0


def test_zero_origin_balance_ratio_uses_unit_denominator() -> None:
    amount = np.array([100.0, 50.0, 0.0])
    balance = np.array([0.0, 100.0, 0.0])
    ratios = amount_to_balance_ratio(amount, balance)
    np.testing.assert_allclose(
        ratios,
        [100.0 / ZERO_BALANCE_RATIO_DENOMINATOR, 50.0 / 100.0, 0.0],
    )
    assert np.isfinite(ratios).all()


def test_engineered_features_have_no_nan_or_infinity() -> None:
    frame = _synthetic_baseline_rows()
    enriched = add_engineered_features(frame)
    for name in ENGINEERED_FEATURE_NAMES:
        values = enriched[name].to_numpy(dtype=float)
        assert np.isfinite(values).all(), name
        assert not np.isnan(values).any(), name


def test_origin_balance_zero_flag() -> None:
    enriched = add_engineered_features(_synthetic_baseline_rows())
    assert enriched.loc[0, "origin_balance_zero"] == 1
    assert enriched.loc[1, "origin_balance_zero"] == 0


def test_excluded_columns_not_in_model_feature_lists() -> None:
    for col in EXCLUDED_BASELINE_COLUMNS:
        assert col not in BASELINE_FEATURES
        assert col not in ENGINEERED_FEATURE_NAMES
        assert col not in ENGINEERED_MODEL_FEATURES

    features = prepare_engineered_model_frame(_synthetic_baseline_rows())
    assert list(features.columns) == list(ENGINEERED_MODEL_FEATURES)
    for col in EXCLUDED_BASELINE_COLUMNS:
        assert col not in features.columns


def test_select_model_feature_frame_rejects_leakage_columns() -> None:
    frame = add_engineered_features(_synthetic_baseline_rows())
    with pytest.raises(ValueError, match="Excluded / leakage"):
        select_model_feature_frame(
            frame,
            feature_names=("type", "amount", "newbalanceOrig"),
        )


def test_scale_pos_weight_uses_training_labels_only() -> None:
    y_train = pd.Series([0, 0, 0, 0, 1, 1])
    y_val = pd.Series([1, 1, 1, 1, 1, 0])  # must not affect the weight
    weight = compute_scale_pos_weight(y_train)
    assert weight == pytest.approx(4.0 / 2.0)
    # Sanity: if someone wrongly used validation labels the weight would differ.
    assert compute_scale_pos_weight(y_val) != weight


def test_xgboost_pipeline_receives_train_only_scale_pos_weight() -> None:
    """Build Model C with weight from train labels; fit on synthetic engineered data."""
    rows: list[dict[str, object]] = []
    for step in range(1, 12):
        rows.append(
            {
                "step": step,
                "type": "TRANSFER" if step % 2 else "CASH_OUT",
                "amount": float(10 * step),
                "oldbalanceOrg": float(5 * step) if step % 3 else 0.0,
                "isFraud": 1 if step in (3, 6, 9) else 0,
            }
        )
    frame = select_baseline_frame(pd.DataFrame(rows))
    split = chronological_train_val_test_split(frame)
    X_train = prepare_engineered_model_frame(split.train)
    y_train = split.train[TARGET_COLUMN]
    X_val = prepare_engineered_model_frame(split.validation)

    weight = compute_scale_pos_weight(y_train)
    assert weight == pytest.approx(
        float((y_train == 0).sum()) / float((y_train == 1).sum())
    )

    model = build_xgboost_pipeline(scale_pos_weight=weight)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]
    assert scores.shape == (len(X_val),)
    assert np.isfinite(scores).all()
    assert model.named_steps["classifier"].get_params()["scale_pos_weight"] == weight


def test_operating_threshold_satisfies_recall_floor_when_feasible() -> None:
    # Construct scores where a threshold clearly achieves recall >= 0.80.
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.95, 0.90, 0.85, 0.80, 0.20, 0.70, 0.10, 0.05, 0.02, 0.01])
    threshold, reason, table = select_operating_threshold(
        y_true, y_score, recall_floor=DEFAULT_RECALL_FLOOR
    )
    selected = table.loc[np.isclose(table["threshold"], threshold)].iloc[0]
    assert selected["recall"] >= DEFAULT_RECALL_FLOOR
    assert "recall >=" in reason


def test_operating_threshold_falls_back_when_recall_infeasible() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Positives all score 0, so no threshold in (0, 1) can recover recall >= 80%.
    y_score = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.7, 0.6])
    _threshold, reason, _table = select_operating_threshold(
        y_true, y_score, recall_floor=0.80
    )
    assert "infeasible" in reason.lower() or "No threshold reached recall" in reason


def test_engineered_logistic_pipeline_fits_synthetic_data() -> None:
    rows: list[dict[str, object]] = []
    for step in range(1, 12):
        rows.append(
            {
                "step": step,
                "type": "TRANSFER",
                "amount": float(100 * step),
                "oldbalanceOrg": float(50 * step) if step % 2 else 0.0,
                "isFraud": 1 if step % 4 == 0 else 0,
            }
        )
        rows.append(
            {
                "step": step,
                "type": "CASH_OUT",
                "amount": float(80 * step),
                "oldbalanceOrg": float(40 * step),
                "isFraud": 0,
            }
        )
    frame = pd.DataFrame(rows)
    split = chronological_train_val_test_split(frame)
    X_train = prepare_engineered_model_frame(split.train)
    y_train = split.train[TARGET_COLUMN]
    pipeline = build_logistic_engineered_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(prepare_engineered_model_frame(split.validation))
    assert proba.shape[1] == 2


def test_baseline_pipeline_unchanged_feature_contract() -> None:
    """Model A still rejects unapproved features and accepts baseline only."""
    pipeline = build_logistic_baseline_pipeline()
    assert pipeline is not None
    with pytest.raises(ValueError, match="must match BASELINE_FEATURES"):
        build_logistic_baseline_pipeline(
            categorical_features=("type",),
            numeric_features=("amount", "log_amount"),
        )


def test_threshold_analysis_includes_operational_columns() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.6])
    table = threshold_analysis(y_true, y_score, thresholds=(0.5,))
    assert "false_positives" in table.columns
    assert "alert_rate" in table.columns
