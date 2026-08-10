"""Local inference for the frozen ``xgb-transformed-v1`` artifact.

Loads a fitted XGB Transformed pipeline and Platt calibrator, maps public request
fields to internal PaySim-style columns, applies shared feature engineering, and
returns calibrated fraud probabilities with the frozen review / pass policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_risk.calibration import (
    DEPLOYMENT_MODEL_VERSION,
    PlattCalibrator,
    predict_raw_margin,
)
from fraud_risk.dataset import MODELING_TYPES
from fraud_risk.features import (
    BALANCE_DRAIN_FEATURE_NAMES,
    XGB_TRANSFORMED_FEATURES,
    prepare_model_frame,
)
from fraud_risk.train_final import (
    CALIBRATOR_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    REQUIRED_ARTIFACT_FILES,
)

# Public inference fields (docs/inference_contract.md) → internal source columns.
PUBLIC_TO_INTERNAL: dict[str, str] = {
    "transaction_type": "type",
    "amount": "amount",
    "origin_balance": "oldbalanceOrg",
}

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "model_version",
    "threshold",
    "prediction_moment",
    "allowed_transaction_types",
    "source_features",
    "engineered_features",
    "excluded_shortcut_features",
    "model_fit_step_range",
    "calibration_step_range",
    "validation_step_range",
    "test_step_range",
    "python_version",
    "scikit_learn_version",
    "xgboost_version",
)

PUBLIC_BATCH_COLUMNS: tuple[str, ...] = ("transaction_type", "amount", "origin_balance")


class ArtifactLoadError(ValueError):
    """Raised when an on-disk artifact package is incomplete or invalid."""


class InferenceInputError(ValueError):
    """Raised when a public inference request fails validation."""


def validate_inference_input(
    *,
    transaction_type: str,
    amount: float,
    origin_balance: float,
    allowed_transaction_types: tuple[str, ...] | list[str] = MODELING_TYPES,
) -> None:
    """Reject invalid public inference inputs with clear Python exceptions."""
    if not isinstance(transaction_type, str):
        raise InferenceInputError(
            f"transaction_type must be a string; got {type(transaction_type).__name__}."
        )
    allowed = tuple(allowed_transaction_types)
    if transaction_type not in allowed:
        raise InferenceInputError(
            f"Unsupported transaction_type {transaction_type!r}. "
            f"Allowed: {list(allowed)}."
        )

    for name, value in (("amount", amount), ("origin_balance", origin_balance)):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InferenceInputError(f"{name} must be a finite number >= 0.") from exc
        if not math.isfinite(number):
            raise InferenceInputError(f"{name} must be finite (rejected NaN / Inf).")
        if number < 0.0:
            raise InferenceInputError(f"{name} must be >= 0; got {number}.")


def public_row_to_source_frame(
    *,
    transaction_type: str,
    amount: float,
    origin_balance: float,
) -> pd.DataFrame:
    """Map a single public request to a one-row internal source DataFrame."""
    return pd.DataFrame(
        {
            "type": [transaction_type],
            "amount": [float(amount)],
            "oldbalanceOrg": [float(origin_balance)],
        }
    )


def public_frame_to_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a public-column DataFrame to internal source columns."""
    missing = [col for col in PUBLIC_BATCH_COLUMNS if col not in df.columns]
    if missing:
        raise InferenceInputError(
            f"Batch frame missing required public columns: {missing}. "
            f"Expected: {list(PUBLIC_BATCH_COLUMNS)}."
        )
    return pd.DataFrame(
        {
            "type": df["transaction_type"].to_numpy(),
            "amount": df["amount"].astype(float).to_numpy(),
            "oldbalanceOrg": df["origin_balance"].astype(float).to_numpy(),
        }
    )


def decide(fraud_probability: float, threshold: float) -> str:
    """Apply the frozen review / pass policy."""
    return "review" if float(fraud_probability) >= float(threshold) else "pass"


def prediction_response(
    *,
    fraud_probability: float,
    threshold: float,
    model_version: str,
) -> dict[str, Any]:
    """Build an inference-contract response dict."""
    proba = float(fraud_probability)
    thr = float(threshold)
    return {
        "fraud_probability": proba,
        "decision": decide(proba, thr),
        "threshold": thr,
        "model_version": model_version,
    }


def validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Ensure artifact metadata contains the required Task 5 fields."""
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ArtifactLoadError(
            f"Artifact metadata.json is missing required fields: {missing}."
        )
    return metadata


def load_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load and validate ``metadata.json`` from an artifact directory."""
    path = artifact_dir / METADATA_FILENAME
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactLoadError(f"{path} must contain a JSON object.")
    return validate_metadata(metadata)


def ensure_artifact_files(artifact_dir: Path) -> None:
    """Fail clearly when required artifact files are missing."""
    if not artifact_dir.is_dir():
        raise ArtifactLoadError(f"Artifact directory not found: {artifact_dir}")
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise ArtifactLoadError(
            f"Incomplete artifact at {artifact_dir}. Missing files: {missing}. "
            f"Expected: {list(REQUIRED_ARTIFACT_FILES)}."
        )


@dataclass
class FraudPredictor:
    """Score transactions from a frozen local artifact package."""

    model: Pipeline
    calibrator: PlattCalibrator
    metadata: dict[str, Any]
    artifact_dir: Path | None = None

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def allowed_transaction_types(self) -> tuple[str, ...]:
        return tuple(self.metadata["allowed_transaction_types"])

    @classmethod
    def load(cls, artifact_dir: Path | str) -> FraudPredictor:
        """Load model, calibrator, and metadata from an artifact directory."""
        path = Path(artifact_dir)
        ensure_artifact_files(path)
        metadata = load_metadata(path)

        model = joblib.load(path / MODEL_FILENAME)
        calibrator = joblib.load(path / CALIBRATOR_FILENAME)
        if not isinstance(model, Pipeline):
            raise ArtifactLoadError(
                f"{MODEL_FILENAME} did not contain a scikit-learn Pipeline."
            )
        if not isinstance(calibrator, PlattCalibrator):
            raise ArtifactLoadError(
                f"{CALIBRATOR_FILENAME} did not contain a PlattCalibrator."
            )
        if not calibrator.is_fitted:
            raise ArtifactLoadError("Loaded calibrator is not fitted.")

        engineered = set(metadata.get("engineered_features", []))
        if engineered & set(BALANCE_DRAIN_FEATURE_NAMES):
            raise ArtifactLoadError(
                "Artifact metadata lists excluded drain-artifact features as engineered."
            )

        return cls(
            model=model,
            calibrator=calibrator,
            metadata=metadata,
            artifact_dir=path,
        )

    def _calibrated_proba_from_source(self, source_df: pd.DataFrame) -> np.ndarray:
        """Engineer features, score raw margins, and apply Platt calibration."""
        features = prepare_model_frame(source_df, XGB_TRANSFORMED_FEATURES)
        # Guard: deployment feature set must never include drain shortcuts.
        assert set(BALANCE_DRAIN_FEATURE_NAMES).isdisjoint(features.columns)
        margins = predict_raw_margin(self.model, features)
        return self.calibrator.transform(margins)

    def predict_one(
        self,
        *,
        transaction_type: str,
        amount: float,
        origin_balance: float,
    ) -> dict[str, Any]:
        """Score a single transaction; return the inference-contract response."""
        validate_inference_input(
            transaction_type=transaction_type,
            amount=amount,
            origin_balance=origin_balance,
            allowed_transaction_types=self.allowed_transaction_types,
        )
        source = public_row_to_source_frame(
            transaction_type=transaction_type,
            amount=amount,
            origin_balance=origin_balance,
        )
        proba = float(self._calibrated_proba_from_source(source)[0])
        return prediction_response(
            fraud_probability=proba,
            threshold=self.threshold,
            model_version=self.model_version,
        )

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-score a DataFrame of public columns; same logic as ``predict_one``."""
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "fraud_probability",
                    "decision",
                    "threshold",
                    "model_version",
                ]
            )

        source = public_frame_to_source_frame(df)
        for i in range(len(source)):
            validate_inference_input(
                transaction_type=str(source.iloc[i]["type"]),
                amount=float(source.iloc[i]["amount"]),
                origin_balance=float(source.iloc[i]["oldbalanceOrg"]),
                allowed_transaction_types=self.allowed_transaction_types,
            )

        probas = self._calibrated_proba_from_source(source)
        rows = [
            prediction_response(
                fraud_probability=float(p),
                threshold=self.threshold,
                model_version=self.model_version,
            )
            for p in probas
        ]
        return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one transaction with a local fraud-risk artifact.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts") / DEPLOYMENT_MODEL_VERSION,
        help="Path to the versioned artifact directory.",
    )
    parser.add_argument(
        "--transaction-type",
        required=True,
        help="TRANSFER or CASH_OUT",
    )
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--origin-balance", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    predictor = FraudPredictor.load(args.artifact_dir)
    result = predictor.predict_one(
        transaction_type=args.transaction_type,
        amount=args.amount,
        origin_balance=args.origin_balance,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
