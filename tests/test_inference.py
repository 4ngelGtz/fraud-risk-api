"""Tests for frozen artifact persistence and local inference."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_risk.calibration import (
    DEPLOYMENT_MODEL_VERSION,
    FROZEN_OPERATING_THRESHOLD,
    PREDICTION_MOMENT,
    PlattCalibrator,
    fit_platt_calibrator,
    predict_raw_margin,
)
from fraud_risk.dataset import (
    BASELINE_FEATURES,
    MODELING_TYPES,
    TARGET_COLUMN,
    chronological_model_fit_calibration_split,
    chronological_train_val_test_split,
    select_baseline_frame,
)
from fraud_risk.features import (
    BALANCE_DRAIN_FEATURE_NAMES,
    XGB_TRANSFORMED_FEATURES,
    prepare_model_frame,
)
from fraud_risk.inference import (
    ArtifactLoadError,
    FraudPredictor,
    InferenceInputError,
    decide,
    prediction_response,
    public_row_to_source_frame,
    validate_inference_input,
)
from fraud_risk.modeling import build_xgboost_pipeline, compute_scale_pos_weight
from fraud_risk.train_final import (
    CALIBRATOR_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    TrainedArtifact,
    build_metadata,
    save_artifact,
    train_xgb_transformed_v1,
)


def _synthetic_paysim_like(n_steps: int = 24, rows_per_step: int = 16) -> pd.DataFrame:
    """Small chronological frame with both classes for fit + calibration."""
    rows: list[dict[str, object]] = []
    for step in range(1, n_steps + 1):
        for i in range(rows_per_step):
            is_fraud = 1 if (i == 0 and step % 3 == 0) else 0
            amount = float(50 * step + i + (5000 if is_fraud else 0))
            balance = float(100 * step + i + (amount if is_fraud else 200))
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
                    "amount": amount,
                    "nameOrig": f"C{step}_{i}",
                    "oldbalanceOrg": balance,
                    "newbalanceOrig": max(balance - amount, 0.0),
                    "nameDest": f"D{step}_{i}",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": amount,
                    "isFraud": is_fraud,
                    "isFlaggedFraud": 0,
                }
            )
    return pd.DataFrame(rows)


def _tiny_trained_artifact() -> TrainedArtifact:
    """Fit a tiny XGB Transformed + Platt package on synthetic data."""
    frame = select_baseline_frame(_synthetic_paysim_like())
    # Ensure positives in early (model-fit) and late (calibration) windows.
    frame.loc[frame["step"].isin([3, 6, 9, 12, 15, 18, 21, 24]), "isFraud"] = 1
    split = chronological_train_val_test_split(frame)
    cal_split = chronological_model_fit_calibration_split(split.train)

    y_fit = cal_split.model_fit[TARGET_COLUMN]
    y_cal = cal_split.calibration[TARGET_COLUMN]
    X_fit = prepare_model_frame(cal_split.model_fit, XGB_TRANSFORMED_FEATURES)
    X_cal = prepare_model_frame(cal_split.calibration, XGB_TRANSFORMED_FEATURES)

    weight = compute_scale_pos_weight(y_fit)
    model = build_xgboost_pipeline(
        scale_pos_weight=weight,
        feature_names=XGB_TRANSFORMED_FEATURES,
    )
    model.fit(X_fit, y_fit)
    calibrator = fit_platt_calibrator(predict_raw_margin(model, X_cal), y_cal)

    metadata = build_metadata(
        model_version=DEPLOYMENT_MODEL_VERSION,
        threshold=FROZEN_OPERATING_THRESHOLD,
        temporal=split,
        cal_split=cal_split,
    )
    # Override step ranges to the known Task 4 frozen periods for metadata tests.
    metadata["model_fit_step_range"] = [1, 279]
    metadata["calibration_step_range"] = [280, 322]
    metadata["validation_step_range"] = [323, 376]
    metadata["test_step_range"] = [377, 743]
    return TrainedArtifact(
        model=model,
        calibrator=calibrator,
        metadata=metadata,
        scale_pos_weight=float(weight),
    )


def test_valid_inference_input_accepted() -> None:
    validate_inference_input(
        transaction_type="TRANSFER",
        amount=8500.0,
        origin_balance=9000.0,
    )
    validate_inference_input(
        transaction_type="CASH_OUT",
        amount=0.0,
        origin_balance=0.0,
    )


@pytest.mark.parametrize("tx_type", ["PAYMENT", "CASH_IN", "DEBIT", "transfer", ""])
def test_unsupported_transaction_types_rejected(tx_type: str) -> None:
    with pytest.raises(InferenceInputError, match="Unsupported transaction_type"):
        validate_inference_input(
            transaction_type=tx_type,
            amount=100.0,
            origin_balance=100.0,
        )


def test_negative_amount_rejected() -> None:
    with pytest.raises(InferenceInputError, match="amount"):
        validate_inference_input(
            transaction_type="TRANSFER",
            amount=-1.0,
            origin_balance=100.0,
        )


def test_negative_origin_balance_rejected() -> None:
    with pytest.raises(InferenceInputError, match="origin_balance"):
        validate_inference_input(
            transaction_type="TRANSFER",
            amount=100.0,
            origin_balance=-0.01,
        )


@pytest.mark.parametrize(
    ("amount", "origin_balance"),
    [
        (float("nan"), 100.0),
        (100.0, float("nan")),
        (float("inf"), 100.0),
        (100.0, float("-inf")),
    ],
)
def test_nan_and_infinity_rejected(amount: float, origin_balance: float) -> None:
    with pytest.raises(InferenceInputError, match="finite"):
        validate_inference_input(
            transaction_type="TRANSFER",
            amount=amount,
            origin_balance=origin_balance,
        )


def test_public_fields_map_to_source_model_fields() -> None:
    frame = public_row_to_source_frame(
        transaction_type="TRANSFER",
        amount=8500.0,
        origin_balance=9000.0,
    )
    assert list(frame.columns) == ["type", "amount", "oldbalanceOrg"]
    assert frame.iloc[0]["type"] == "TRANSFER"
    assert frame.iloc[0]["amount"] == 8500.0
    assert frame.iloc[0]["oldbalanceOrg"] == 9000.0


def test_decision_review_when_probability_at_or_above_threshold() -> None:
    assert decide(0.044, 0.044) == "review"
    assert decide(0.05, 0.044) == "review"
    response = prediction_response(
        fraud_probability=0.044,
        threshold=0.044,
        model_version=DEPLOYMENT_MODEL_VERSION,
    )
    assert response["decision"] == "review"


def test_decision_pass_when_probability_below_threshold() -> None:
    assert decide(0.043999, 0.044) == "pass"
    response = prediction_response(
        fraud_probability=0.01,
        threshold=0.044,
        model_version=DEPLOYMENT_MODEL_VERSION,
    )
    assert response["decision"] == "pass"


def test_loaded_metadata_controls_threshold_and_model_version(tmp_path: Path) -> None:
    artifact = _tiny_trained_artifact()
    artifact_dir = save_artifact(artifact, tmp_path / "xgb-transformed-v1")
    predictor = FraudPredictor.load(artifact_dir)
    assert predictor.threshold == FROZEN_OPERATING_THRESHOLD
    assert predictor.model_version == DEPLOYMENT_MODEL_VERSION
    assert predictor.metadata["prediction_moment"] == PREDICTION_MOMENT
    assert predictor.metadata["allowed_transaction_types"] == list(MODELING_TYPES)
    assert predictor.metadata["source_features"] == list(BASELINE_FEATURES)


def test_artifact_save_load_round_trip_preserves_predictions(tmp_path: Path) -> None:
    artifact = _tiny_trained_artifact()
    artifact_dir = save_artifact(artifact, tmp_path / "pkg")

    # In-process reference score via the same path as inference.
    source = public_row_to_source_frame(
        transaction_type="TRANSFER",
        amount=8500.0,
        origin_balance=9000.0,
    )
    features = prepare_model_frame(source, XGB_TRANSFORMED_FEATURES)
    expected = float(
        artifact.calibrator.transform(predict_raw_margin(artifact.model, features))[0]
    )

    predictor = FraudPredictor.load(artifact_dir)
    result = predictor.predict_one(
        transaction_type="TRANSFER",
        amount=8500.0,
        origin_balance=9000.0,
    )
    assert result["model_version"] == DEPLOYMENT_MODEL_VERSION
    assert result["threshold"] == FROZEN_OPERATING_THRESHOLD
    assert result["decision"] in {"review", "pass"}
    np.testing.assert_allclose(result["fraud_probability"], expected, rtol=1e-10, atol=1e-12)

    # Fresh process not required here; second load still matches.
    again = FraudPredictor.load(artifact_dir).predict_one(
        transaction_type="TRANSFER",
        amount=8500.0,
        origin_balance=9000.0,
    )
    assert again == result


def test_inference_uses_shared_feature_engineering(tmp_path: Path) -> None:
    artifact = _tiny_trained_artifact()
    artifact_dir = save_artifact(artifact, tmp_path / "pkg")
    predictor = FraudPredictor.load(artifact_dir)

    with patch(
        "fraud_risk.inference.prepare_model_frame",
        wraps=prepare_model_frame,
    ) as mocked:
        predictor.predict_one(
            transaction_type="CASH_OUT",
            amount=120.0,
            origin_balance=500.0,
        )
    mocked.assert_called_once()
    source_df, feature_names = mocked.call_args.args
    assert feature_names == XGB_TRANSFORMED_FEATURES
    assert "type" in source_df.columns
    assert "oldbalanceOrg" in source_df.columns
    assert "transaction_type" not in source_df.columns
    assert "origin_balance" not in source_df.columns


def test_excluded_drain_features_not_in_deployed_feature_set() -> None:
    assert set(BALANCE_DRAIN_FEATURE_NAMES).isdisjoint(XGB_TRANSFORMED_FEATURES)
    artifact = _tiny_trained_artifact()
    assert set(BALANCE_DRAIN_FEATURE_NAMES).isdisjoint(
        artifact.metadata["engineered_features"]
    )
    assert set(BALANCE_DRAIN_FEATURE_NAMES).issubset(
        artifact.metadata["excluded_shortcut_features"]
    )


def test_incomplete_artifact_fails_clearly(tmp_path: Path) -> None:
    package = tmp_path / "broken"
    package.mkdir()
    (package / MODEL_FILENAME).write_bytes(b"not-a-real-model")
    with pytest.raises(ArtifactLoadError, match="Missing files"):
        FraudPredictor.load(package)

    artifact = _tiny_trained_artifact()
    good = save_artifact(artifact, tmp_path / "good")
    (good / METADATA_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="missing required fields"):
        FraudPredictor.load(good)


def test_train_xgb_transformed_v1_on_synthetic_raw_frame(tmp_path: Path) -> None:
    raw = _synthetic_paysim_like(n_steps=30, rows_per_step=20)
    raw.loc[raw["step"].isin(range(1, 31, 2)), "isFraud"] = 1
    trained = train_xgb_transformed_v1(raw)
    assert isinstance(trained.model, Pipeline)
    assert isinstance(trained.calibrator, PlattCalibrator)
    assert trained.metadata["model_version"] == DEPLOYMENT_MODEL_VERSION
    assert trained.metadata["threshold"] == FROZEN_OPERATING_THRESHOLD

    out = save_artifact(trained, tmp_path / DEPLOYMENT_MODEL_VERSION)
    assert (out / MODEL_FILENAME).is_file()
    assert (out / CALIBRATOR_FILENAME).is_file()
    meta = json.loads((out / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert "python_version" in meta
    assert "scikit_learn_version" in meta
    assert "xgboost_version" in meta
    # No absolute paths in metadata values.
    blob = json.dumps(meta)
    assert "/Users/" not in blob
    assert "C:\\\\" not in blob

    predictor = FraudPredictor.load(out)
    batch = predictor.predict_frame(
        pd.DataFrame(
            {
                "transaction_type": ["TRANSFER", "CASH_OUT"],
                "amount": [8500.0, 200.0],
                "origin_balance": [9000.0, 50.0],
            }
        )
    )
    assert len(batch) == 2
    assert set(batch["decision"]).issubset({"review", "pass"})


def test_frozen_threshold_constant() -> None:
    assert FROZEN_OPERATING_THRESHOLD == 0.044
