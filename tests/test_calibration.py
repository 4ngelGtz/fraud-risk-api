"""Tests for chronological calibration splits and Platt scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_risk.calibration import (
    DEPLOYMENT_MODEL_VERSION,
    PlattCalibrator,
    calibrate_scores,
    fit_platt_calibrator,
    predict_raw_margin,
    probability_metrics,
    reliability_table,
)
from fraud_risk.dataset import (
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
from fraud_risk.modeling import (
    DEFAULT_RECALL_FLOOR,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
    select_operating_threshold,
)


def _multi_step_train_frame(n_steps: int = 20, rows_per_step: int = 10) -> pd.DataFrame:
    """Synthetic chronological frame suitable for train→model-fit/calibration splits."""
    rows: list[dict[str, object]] = []
    for step in range(1, n_steps + 1):
        for i in range(rows_per_step):
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
                    "amount": float(10 * step + i),
                    "oldbalanceOrg": float(100 * step + i),
                    "isFraud": 1 if (i == 0 and step % 4 == 0) else 0,
                }
            )
    return pd.DataFrame(rows)


def test_model_fit_calibration_subsets_are_strictly_chronological() -> None:
    frame = _multi_step_train_frame()
    result = chronological_model_fit_calibration_split(frame)
    assert result.model_fit["step"].max() < result.calibration["step"].min()
    assert (
        result.boundaries.model_fit_max_step < result.boundaries.calibration_min_step
    )


def test_no_step_shared_between_model_fit_and_calibration() -> None:
    frame = _multi_step_train_frame()
    result = chronological_model_fit_calibration_split(frame)
    fit_steps = set(result.model_fit["step"])
    cal_steps = set(result.calibration["step"])
    assert fit_steps.isdisjoint(cal_steps)

    for step, group in frame.groupby("step"):
        in_fit = set(group.index).issubset(set(result.model_fit.index))
        in_cal = set(group.index).issubset(set(result.calibration.index))
        assert sum([in_fit, in_cal]) == 1


def test_calibration_rows_do_not_appear_in_validation_or_test() -> None:
    frame = select_baseline_frame(_multi_step_train_frame(n_steps=24, rows_per_step=8))
    split = chronological_train_val_test_split(frame)
    cal_split = chronological_model_fit_calibration_split(split.train)

    assert cal_split.calibration["step"].max() < split.validation["step"].min()
    assert split.validation["step"].max() < split.test["step"].min()

    cal_idx = set(cal_split.calibration.index)
    assert cal_idx.isdisjoint(set(split.validation.index))
    assert cal_idx.isdisjoint(set(split.test.index))
    assert set(cal_split.model_fit.index).isdisjoint(set(split.validation.index))
    assert set(cal_split.model_fit.index).isdisjoint(set(split.test.index))


def test_platt_calibrator_outputs_finite_probabilities_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=200)
    y = (raw > 0).astype(int)
    # Ensure both classes present with some noise
    y[::17] = 1 - y[::17]
    calibrator = fit_platt_calibrator(raw, y)
    proba = calibrate_scores(calibrator, raw)
    assert proba.shape == raw.shape
    assert np.isfinite(proba).all()
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_calibrated_scores_preserve_raw_margin_order() -> None:
    """Larger raw margins must map to larger (or equal) calibrated probabilities."""
    raw = np.linspace(-5.0, 5.0, 101)
    # Labels roughly aligned with score so the fitted slope is positive.
    y = (raw > 0).astype(int)
    y[0] = 0
    y[-1] = 1
    calibrator = PlattCalibrator().fit(raw, y)
    assert calibrator.coef_ > 0
    calibrated = calibrator.transform(raw)
    diffs = np.diff(calibrated)
    assert (diffs >= -1e-12).all()


def test_threshold_selection_satisfies_recall_floor_on_calibrated_scores() -> None:
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # Calibrated-like probabilities (already in [0, 1]).
    y_cal = np.array([0.92, 0.88, 0.81, 0.75, 0.15, 0.55, 0.08, 0.04, 0.02, 0.01])
    threshold, reason, table = select_operating_threshold(
        y_true, y_cal, recall_floor=DEFAULT_RECALL_FLOOR
    )
    selected = table.loc[np.isclose(table["threshold"], threshold)].iloc[0]
    assert selected["recall"] >= DEFAULT_RECALL_FLOOR
    assert "recall >=" in reason


def test_deployment_feature_set_excludes_drain_artifacts() -> None:
    assert set(BALANCE_DRAIN_FEATURE_NAMES).isdisjoint(XGB_TRANSFORMED_FEATURES)
    assert "amount_to_balance_ratio" not in XGB_TRANSFORMED_FEATURES
    assert "amount_exceeds_balance" not in XGB_TRANSFORMED_FEATURES
    assert DEPLOYMENT_MODEL_VERSION == "xgb-transformed-v1"


def test_predict_raw_margin_and_end_to_end_calibration_on_synthetic_xgb() -> None:
    """Model-fit-only weight + Platt fit on calibration margins; scores stay valid."""
    frame = _multi_step_train_frame(n_steps=24, rows_per_step=12)
    # Ensure enough fraud labels in early and late windows.
    frame.loc[frame["step"].isin([4, 8, 12, 16, 20, 24]), "isFraud"] = 1
    split = chronological_train_val_test_split(select_baseline_frame(frame))
    cal_split = chronological_model_fit_calibration_split(split.train)

    X_fit = prepare_model_frame(cal_split.model_fit, XGB_TRANSFORMED_FEATURES)
    y_fit = cal_split.model_fit[TARGET_COLUMN]
    X_cal = prepare_model_frame(cal_split.calibration, XGB_TRANSFORMED_FEATURES)
    y_cal = cal_split.calibration[TARGET_COLUMN]
    X_val = prepare_model_frame(split.validation, XGB_TRANSFORMED_FEATURES)

    weight = compute_scale_pos_weight(y_fit)
    model = build_xgboost_pipeline(
        scale_pos_weight=weight,
        feature_names=XGB_TRANSFORMED_FEATURES,
    )
    model.fit(X_fit, y_fit)

    cal_margin = predict_raw_margin(model, X_cal)
    calibrator = fit_platt_calibrator(cal_margin, y_cal)
    cal_proba = calibrator.transform(cal_margin)
    val_proba = calibrator.transform(predict_raw_margin(model, X_val))

    assert np.isfinite(cal_proba).all()
    assert ((cal_proba >= 0.0) & (cal_proba <= 1.0)).all()
    assert np.isfinite(val_proba).all()

    metrics = probability_metrics(y_cal, cal_proba)
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] >= 0.0

    table = reliability_table(y_cal, cal_proba, n_bins=5, strategy="quantile")
    assert "observed_fraud_rate" in table.columns
    assert set(table["status"]).issubset({"ok", "empty"})
