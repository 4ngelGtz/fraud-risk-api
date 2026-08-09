"""Tests for leakage-aware modeling dataset construction and temporal splits."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_risk.dataset import (
    BASELINE_FEATURES,
    EXCLUDED_BASELINE_COLUMNS,
    MODELING_TYPES,
    TARGET_COLUMN,
    TEMPORAL_COLUMN,
    chronological_train_val_test_split,
    restrict_to_modeling_scope,
    select_baseline_frame,
)
from fraud_risk.modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_logistic_baseline_pipeline,
)


def _modeling_frame() -> pd.DataFrame:
    """Small synthetic PaySim-like frame spanning multiple steps and types."""
    rows: list[dict[str, object]] = []
    # Steps 1–10 with TRANSFER / CASH_OUT (in scope) plus PAYMENT (out of scope).
    for step in range(1, 11):
        rows.append(
            {
                "step": step,
                "type": "TRANSFER",
                "amount": 100.0 * step,
                "nameOrig": f"C{step}",
                "oldbalanceOrg": 1000.0 * step,
                "newbalanceOrig": 900.0 * step,
                "nameDest": f"D{step}",
                "oldbalanceDest": 50.0,
                "newbalanceDest": 150.0,
                "isFraud": 1 if step % 4 == 0 else 0,
                "isFlaggedFraud": 0,
            }
        )
        rows.append(
            {
                "step": step,
                "type": "CASH_OUT",
                "amount": 80.0 * step,
                "nameOrig": f"C{step}b",
                "oldbalanceOrg": 800.0 * step,
                "newbalanceOrig": 720.0 * step,
                "nameDest": f"D{step}b",
                "oldbalanceDest": 40.0,
                "newbalanceDest": 120.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            }
        )
        rows.append(
            {
                "step": step,
                "type": "PAYMENT",
                "amount": 10.0 * step,
                "nameOrig": f"C{step}p",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 90.0,
                "nameDest": f"M{step}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 10.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(rows)


def test_modeling_scope_only_transfer_and_cash_out() -> None:
    scoped = restrict_to_modeling_scope(_modeling_frame())
    assert set(scoped["type"].unique()) == set(MODELING_TYPES)
    assert "PAYMENT" not in set(scoped["type"])
    assert "CASH_IN" not in set(scoped["type"])
    assert "DEBIT" not in set(scoped["type"])


def test_baseline_feature_list_matches_contract() -> None:
    assert BASELINE_FEATURES == ("type", "amount", "oldbalanceOrg")
    assert CATEGORICAL_FEATURES == ("type",)
    assert NUMERIC_FEATURES == ("amount", "oldbalanceOrg")
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(BASELINE_FEATURES)


def test_post_transaction_variables_excluded_from_baseline() -> None:
    frame = select_baseline_frame(restrict_to_modeling_scope(_modeling_frame()))
    for col in EXCLUDED_BASELINE_COLUMNS:
        assert col not in frame.columns
    # Temporal key is retained for splitting only; not a baseline predictor.
    assert TEMPORAL_COLUMN in frame.columns
    assert TEMPORAL_COLUMN not in BASELINE_FEATURES
    assert list(BASELINE_FEATURES) + [TARGET_COLUMN] == [
        c for c in frame.columns if c != TEMPORAL_COLUMN
    ]


def test_restrict_and_select_do_not_mutate_input() -> None:
    original = _modeling_frame()
    snapshot = original.copy(deep=True)
    scoped = restrict_to_modeling_scope(original)
    _ = select_baseline_frame(scoped)
    pd.testing.assert_frame_equal(original, snapshot)


def test_temporal_splits_do_not_overlap() -> None:
    baseline = select_baseline_frame(restrict_to_modeling_scope(_modeling_frame()))
    result = chronological_train_val_test_split(baseline)

    train_idx = set(result.train.index)
    val_idx = set(result.validation.index)
    test_idx = set(result.test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx | val_idx | test_idx == set(baseline.index)


def test_temporal_splits_are_strictly_chronological() -> None:
    baseline = select_baseline_frame(restrict_to_modeling_scope(_modeling_frame()))
    result = chronological_train_val_test_split(baseline)

    assert result.train["step"].max() < result.validation["step"].min()
    assert result.validation["step"].max() < result.test["step"].min()
    assert result.boundaries.train_max_step < result.boundaries.validation_min_step
    assert result.boundaries.validation_max_step < result.boundaries.test_min_step


def test_one_step_cannot_appear_in_multiple_splits() -> None:
    baseline = select_baseline_frame(restrict_to_modeling_scope(_modeling_frame()))
    result = chronological_train_val_test_split(baseline)

    train_steps = set(result.train["step"])
    val_steps = set(result.validation["step"])
    test_steps = set(result.test["step"])

    assert train_steps.isdisjoint(val_steps)
    assert train_steps.isdisjoint(test_steps)
    assert val_steps.isdisjoint(test_steps)

    # All rows for a given step stay together in exactly one split.
    split_by_name = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    for step, group in baseline.groupby("step"):
        membership = [
            name
            for name, steps in (
                ("train", train_steps),
                ("validation", val_steps),
                ("test", test_steps),
            )
            if step in steps
        ]
        assert len(membership) == 1
        split_frame = split_by_name[membership[0]]
        assert set(group.index).issubset(set(split_frame.index))


def _uneven_step_frame() -> pd.DataFrame:
    """Synthetic frame with highly uneven observations per step (still chronological)."""
    # Step sizes chosen so unique-step splitting (~70% of 10 steps) would differ
    # sharply from cumulative-row splitting (~70% of rows).
    step_sizes = {
        1: 40,
        2: 35,
        3: 30,
        4: 25,
        5: 20,
        6: 15,
        7: 100,  # large late train / early val mass
        8: 50,
        9: 45,
        10: 40,
    }
    rows: list[dict[str, object]] = []
    for step, n_rows in step_sizes.items():
        for i in range(n_rows):
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
                    "amount": float(10 * step + i),
                    "oldbalanceOrg": float(100 * step + i),
                    "isFraud": 1 if i == 0 and step % 3 == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_row_proportions_approximate_70_15_15_with_uneven_steps() -> None:
    frame = _uneven_step_frame()
    result = chronological_train_val_test_split(frame)
    total = len(frame)
    train_share = len(result.train) / total
    val_share = len(result.validation) / total
    test_share = len(result.test) / total

    assert 0.60 <= train_share <= 0.80
    assert 0.08 <= val_share <= 0.25
    assert 0.08 <= test_share <= 0.25
    assert abs(train_share - 0.70) < abs(train_share - 0.50)


def test_highly_uneven_step_sizes_keep_steps_intact_and_chronological() -> None:
    # One enormous early step, tiny middle steps, one large late step.
    rows: list[dict[str, object]] = []
    for step, n_rows in ((1, 700), (2, 5), (3, 5), (4, 5), (5, 150), (6, 135)):
        for i in range(n_rows):
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER",
                    "amount": float(i + 1),
                    "oldbalanceOrg": float(i + 10),
                    "isFraud": 0,
                }
            )
    frame = pd.DataFrame(rows)
    result = chronological_train_val_test_split(frame)

    assert result.train["step"].max() < result.validation["step"].min()
    assert result.validation["step"].max() < result.test["step"].min()

    for step, group in frame.groupby("step"):
        in_train = set(group.index).issubset(set(result.train.index))
        in_val = set(group.index).issubset(set(result.validation.index))
        in_test = set(group.index).issubset(set(result.test.index))
        assert sum([in_train, in_val, in_test]) == 1

    total = len(frame)
    assert 0.55 <= len(result.train) / total <= 0.85
    assert len(result.validation) > 0
    assert len(result.test) > 0


def test_pipeline_rejects_unapproved_features() -> None:
    with pytest.raises(ValueError, match="must match BASELINE_FEATURES"):
        build_logistic_baseline_pipeline(
            categorical_features=("type",),
            numeric_features=("amount", "oldbalanceDest"),
        )


def test_logistic_pipeline_fits_synthetic_data() -> None:
    baseline = select_baseline_frame(restrict_to_modeling_scope(_modeling_frame()))
    result = chronological_train_val_test_split(baseline)
    X_train = result.train.loc[:, list(BASELINE_FEATURES)]
    y_train = result.train[TARGET_COLUMN]

    pipeline = build_logistic_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(result.validation.loc[:, list(BASELINE_FEATURES)])
    assert proba.shape == (len(result.validation), 2)
