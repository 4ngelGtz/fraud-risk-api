"""Reproduce and persist the frozen ``xgb-transformed-v1`` deployment artifact.

Fits XGB Transformed on the chronological model-fit subset, fits Platt scaling on
the calibration subset, and writes versioned artifacts under ``artifacts/``.
Validation and test labels are never used to fit or change the persisted model.
The operating threshold is the Task 4 frozen value (not reselected here).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from fraud_risk.calibration import (
    DEPLOYMENT_MODEL_VERSION,
    FROZEN_OPERATING_THRESHOLD,
    PREDICTION_MOMENT,
    PlattCalibrator,
    fit_platt_calibrator,
    predict_raw_margin,
)
from fraud_risk.data import load_raw_data
from fraud_risk.dataset import (
    BASELINE_FEATURES,
    MODELING_TYPES,
    TARGET_COLUMN,
    TemporalSplitResult,
    TrainCalibrationSplitResult,
    build_modeling_dataset,
    chronological_model_fit_calibration_split,
)
from fraud_risk.features import (
    BALANCE_DRAIN_FEATURE_NAMES,
    XGB_TRANSFORMED_FEATURES,
    prepare_model_frame,
)
from fraud_risk.modeling import (
    RANDOM_STATE,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
)

MODEL_FILENAME: str = "model.joblib"
CALIBRATOR_FILENAME: str = "calibrator.joblib"
METADATA_FILENAME: str = "metadata.json"

REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    MODEL_FILENAME,
    CALIBRATOR_FILENAME,
    METADATA_FILENAME,
)

# Engineered predictors used by the frozen deployment feature set (no drain shortcuts).
DEPLOYMENT_ENGINEERED_FEATURES: tuple[str, ...] = tuple(
    name for name in XGB_TRANSFORMED_FEATURES if name not in BASELINE_FEATURES
)


def _default_artifacts_root() -> Path:
    """Resolve ``artifacts/`` relative to the repository root."""
    return Path(__file__).resolve().parents[2] / "artifacts"


def default_artifact_dir(
    *,
    artifacts_root: Path | str | None = None,
    model_version: str = DEPLOYMENT_MODEL_VERSION,
) -> Path:
    """Return ``artifacts/<model_version>/``."""
    root = Path(artifacts_root) if artifacts_root is not None else _default_artifacts_root()
    return root / model_version


def _step_range(steps: tuple[int, ...]) -> list[int]:
    return [int(min(steps)), int(max(steps))]


def build_metadata(
    *,
    model_version: str,
    threshold: float,
    temporal: TemporalSplitResult,
    cal_split: TrainCalibrationSplitResult,
) -> dict[str, Any]:
    """Build human-readable artifact metadata (no filesystem paths or raw data)."""
    return {
        "model_version": model_version,
        "threshold": float(threshold),
        "prediction_moment": PREDICTION_MOMENT,
        "allowed_transaction_types": list(MODELING_TYPES),
        "source_features": list(BASELINE_FEATURES),
        "engineered_features": list(DEPLOYMENT_ENGINEERED_FEATURES),
        "excluded_shortcut_features": list(BALANCE_DRAIN_FEATURE_NAMES),
        "model_fit_step_range": _step_range(cal_split.boundaries.model_fit_steps),
        "calibration_step_range": _step_range(cal_split.boundaries.calibration_steps),
        "validation_step_range": _step_range(temporal.boundaries.validation_steps),
        "test_step_range": _step_range(temporal.boundaries.test_steps),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }


@dataclass(frozen=True)
class TrainedArtifact:
    """Fitted deployment components ready for persistence or in-process use."""

    model: Pipeline
    calibrator: PlattCalibrator
    metadata: dict[str, Any]
    scale_pos_weight: float


def train_xgb_transformed_v1(
    raw_df: pd.DataFrame | None = None,
    *,
    random_state: int = RANDOM_STATE,
) -> TrainedArtifact:
    """Fit the frozen XGB Transformed + Platt package from a PaySim-like frame.

    Parameters
    ----------
    raw_df:
        Optional raw PaySim DataFrame. When omitted, loads from ``data/raw/``.
    """
    df = load_raw_data() if raw_df is None else raw_df
    temporal = build_modeling_dataset(df)
    cal_split = chronological_model_fit_calibration_split(temporal.train)

    model_fit = cal_split.model_fit
    calibration = cal_split.calibration
    y_fit = model_fit[TARGET_COLUMN]
    y_cal = calibration[TARGET_COLUMN]

    X_fit = prepare_model_frame(model_fit, XGB_TRANSFORMED_FEATURES)
    X_cal = prepare_model_frame(calibration, XGB_TRANSFORMED_FEATURES)

    scale_pos_weight = compute_scale_pos_weight(y_fit)
    model = build_xgboost_pipeline(
        scale_pos_weight=scale_pos_weight,
        feature_names=XGB_TRANSFORMED_FEATURES,
        random_state=random_state,
    )
    model.fit(X_fit, y_fit)

    cal_margin = predict_raw_margin(model, X_cal)
    calibrator = fit_platt_calibrator(cal_margin, y_cal, random_state=random_state)

    metadata = build_metadata(
        model_version=DEPLOYMENT_MODEL_VERSION,
        threshold=FROZEN_OPERATING_THRESHOLD,
        temporal=temporal,
        cal_split=cal_split,
    )
    return TrainedArtifact(
        model=model,
        calibrator=calibrator,
        metadata=metadata,
        scale_pos_weight=float(scale_pos_weight),
    )


def save_artifact(
    artifact: TrainedArtifact,
    artifact_dir: Path | str,
) -> Path:
    """Persist ``model.joblib``, ``calibrator.joblib``, and ``metadata.json``."""
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / MODEL_FILENAME
    calibrator_path = out_dir / CALIBRATOR_FILENAME
    metadata_path = out_dir / METADATA_FILENAME

    joblib.dump(artifact.model, model_path)
    joblib.dump(artifact.calibrator, calibrator_path)
    metadata_path.write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_dir


def train_and_save(
    *,
    artifact_dir: Path | str | None = None,
    raw_df: pd.DataFrame | None = None,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Train ``xgb-transformed-v1`` and save under ``artifacts/xgb-transformed-v1/``."""
    out = (
        Path(artifact_dir)
        if artifact_dir is not None
        else default_artifact_dir()
    )
    trained = train_xgb_transformed_v1(raw_df, random_state=random_state)
    return save_artifact(trained, out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: reproduce and save the frozen deployment artifact."""
    del argv  # reserved for future flags; no options required for Task 5
    out_dir = train_and_save()
    meta_path = out_dir / METADATA_FILENAME
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"Saved frozen model artifact: {out_dir}")
    print(f"  model_version: {metadata['model_version']}")
    print(f"  threshold: {metadata['threshold']}")
    print(f"  model_fit_step_range: {metadata['model_fit_step_range']}")
    print(f"  calibration_step_range: {metadata['calibration_step_range']}")
    print(f"  validation_step_range: {metadata['validation_step_range']}")
    print(f"  test_step_range: {metadata['test_step_range']}")
    print(f"  files: {MODEL_FILENAME}, {CALIBRATOR_FILENAME}, {METADATA_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
