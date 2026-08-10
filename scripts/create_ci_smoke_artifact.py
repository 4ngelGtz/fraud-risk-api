#!/usr/bin/env python3
"""Create a tiny CI-only smoke artifact for Docker / serving-path checks.

This artifact is only for CI serving-contract smoke tests. It is not the
portfolio fraud model and must never be used for evaluation or deployment.

The real ``xgb-transformed-v1`` binaries and PaySim are intentionally excluded
from Git. CI therefore writes a compatible package into the Docker-expected
path ``artifacts/xgb-transformed-v1/`` so the existing Dockerfile can bake it
into the image. Metadata ``model_version`` is ``ci-smoke-v1`` so the synthetic
package cannot be mistaken for the portfolio model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fraud_risk.calibration import (
    PREDICTION_MOMENT,
    fit_platt_calibrator,
    predict_raw_margin,
)
from fraud_risk.dataset import (
    TARGET_COLUMN,
    chronological_model_fit_calibration_split,
    chronological_train_val_test_split,
    select_baseline_frame,
)
from fraud_risk.features import XGB_TRANSFORMED_FEATURES, prepare_model_frame
from fraud_risk.modeling import build_xgboost_pipeline, compute_scale_pos_weight
from fraud_risk.train_final import (
    CALIBRATOR_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    REQUIRED_ARTIFACT_FILES,
    TrainedArtifact,
    build_metadata,
    save_artifact,
)

# Stable CI-only identifier — must never equal the portfolio model version.
CI_SMOKE_MODEL_VERSION: str = "ci-smoke-v1"

# Simple review threshold for the synthetic smoke package (not Task 4's 0.044).
CI_SMOKE_THRESHOLD: float = 0.5

# Directory name expected by the Dockerfile COPY / FRAUD_MODEL_DIR path.
DOCKER_ARTIFACT_DIRNAME: str = "xgb-transformed-v1"

# Tiny booster so artifact generation finishes in seconds.
_CI_XGB_PARAMS: dict[str, int | float] = {
    "n_estimators": 8,
    "max_depth": 2,
    "learning_rate": 0.3,
    "min_child_weight": 1,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_ci_artifact_dir() -> Path:
    """Return the Docker-expected artifact path under the repository root."""
    return _repo_root() / "artifacts" / DOCKER_ARTIFACT_DIRNAME


def build_synthetic_smoke_frame(
    *,
    n_steps: int = 24,
    rows_per_step: int = 8,
) -> pd.DataFrame:
    """Build a tiny deterministic chronological frame (no PaySim)."""
    rows: list[dict[str, object]] = []
    for step in range(1, n_steps + 1):
        for i in range(rows_per_step):
            # Sparse positives so both model-fit and calibration windows see both classes.
            is_fraud = 1 if (i == 0 and step % 4 == 0) else 0
            amount = float(40 * step + i + (3000 if is_fraud else 0))
            balance = float(80 * step + i + (amount if is_fraud else 150))
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
    frame = select_baseline_frame(pd.DataFrame(rows))
    # Guarantee positives in early and late chronological windows.
    frame.loc[frame["step"].isin([4, 8, 12, 16, 20, 24]), "isFraud"] = 1
    return frame


def train_ci_smoke_artifact(*, random_state: int = 42) -> TrainedArtifact:
    """Fit a minimal XGB Transformed + Platt package on synthetic data."""
    frame = build_synthetic_smoke_frame()
    temporal = chronological_train_val_test_split(frame)
    cal_split = chronological_model_fit_calibration_split(temporal.train)

    y_fit = cal_split.model_fit[TARGET_COLUMN]
    y_cal = cal_split.calibration[TARGET_COLUMN]
    X_fit = prepare_model_frame(cal_split.model_fit, XGB_TRANSFORMED_FEATURES)
    X_cal = prepare_model_frame(cal_split.calibration, XGB_TRANSFORMED_FEATURES)

    weight = compute_scale_pos_weight(y_fit)
    model = build_xgboost_pipeline(
        scale_pos_weight=weight,
        feature_names=XGB_TRANSFORMED_FEATURES,
        random_state=random_state,
        xgb_params=_CI_XGB_PARAMS,
    )
    model.fit(X_fit, y_fit)
    calibrator = fit_platt_calibrator(
        predict_raw_margin(model, X_cal),
        y_cal,
        random_state=random_state,
    )

    metadata = build_metadata(
        model_version=CI_SMOKE_MODEL_VERSION,
        threshold=CI_SMOKE_THRESHOLD,
        temporal=temporal,
        cal_split=cal_split,
    )
    metadata["prediction_moment"] = PREDICTION_MOMENT
    metadata["purpose"] = "ci-serving-smoke-only"
    metadata["warning"] = (
        "This artifact is only for CI serving-contract smoke tests. "
        "It is not the portfolio fraud model and must never be used for "
        "evaluation or deployment."
    )
    return TrainedArtifact(
        model=model,
        calibrator=calibrator,
        metadata=metadata,
        scale_pos_weight=float(weight),
    )


def create_ci_smoke_artifact(
    artifact_dir: Path | str | None = None,
    *,
    random_state: int = 42,
) -> Path:
    """Train and persist the CI smoke package; return the output directory."""
    out = Path(artifact_dir) if artifact_dir is not None else default_ci_artifact_dir()
    trained = train_ci_smoke_artifact(random_state=random_state)
    return save_artifact(trained, out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CI-only smoke artifact compatible with FraudPredictor "
            "(not the portfolio fraud model)."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: artifacts/xgb-transformed-v1 under the "
            "repository root — the path expected by the Dockerfile)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = create_ci_smoke_artifact(args.artifact_dir)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (out_dir / name).is_file()]
    if missing:
        print(f"ERROR: incomplete CI smoke artifact; missing {missing}", file=sys.stderr)
        return 1

    metadata = json.loads((out_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    print("Created CI smoke artifact (NOT the portfolio fraud model).")
    print(f"  path: {out_dir}")
    print(f"  model_version: {metadata['model_version']}")
    print(f"  threshold: {metadata['threshold']}")
    print(f"  files: {MODEL_FILENAME}, {CALIBRATOR_FILENAME}, {METADATA_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
