"""Leakage-aware modeling dataset construction and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# V1 row scope: PaySim labeled fraud appears only in these types (audit finding).
MODELING_TYPES: tuple[str, ...] = ("TRANSFER", "CASH_OUT")

# Conservative leakage-safe baseline predictors (prediction moment: pre-authorization).
BASELINE_FEATURES: tuple[str, ...] = ("type", "amount", "oldbalanceOrg")

TARGET_COLUMN: str = "isFraud"
TEMPORAL_COLUMN: str = "step"

# Explicitly excluded from baseline predictors (IDs, post-transaction, rule flag).
EXCLUDED_BASELINE_COLUMNS: tuple[str, ...] = (
    "nameOrig",
    "nameDest",
    "newbalanceOrig",
    "newbalanceDest",
    "oldbalanceDest",
    "isFlaggedFraud",
)


def restrict_to_modeling_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy restricted to V1 transaction types (TRANSFER, CASH_OUT).

    This scope is specific to PaySim: the Task 1 audit observed all positive fraud
    labels in TRANSFER and CASH_OUT. It must not be generalized to real systems.
    """
    if "type" not in df.columns:
        raise ValueError("DataFrame must contain a 'type' column to apply modeling scope.")
    return df.loc[df["type"].isin(MODELING_TYPES)].copy()


def select_baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select temporal key, baseline predictors, and target without mutating ``df``.

    Includes ``step`` for chronological splitting only — it is not a baseline predictor.
    """
    required = (TEMPORAL_COLUMN, *BASELINE_FEATURES, TARGET_COLUMN)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for baseline frame: {missing}.")
    return df.loc[:, list(required)].copy()


@dataclass(frozen=True)
class TemporalSplitBoundaries:
    """Step boundaries for a chronological train / validation / test split."""

    train_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    test_steps: tuple[int, ...]

    @property
    def train_max_step(self) -> int:
        return max(self.train_steps)

    @property
    def validation_min_step(self) -> int:
        return min(self.validation_steps)

    @property
    def validation_max_step(self) -> int:
        return max(self.validation_steps)

    @property
    def test_min_step(self) -> int:
        return min(self.test_steps)


@dataclass(frozen=True)
class TemporalSplitResult:
    """Chronological split frames plus the step boundaries used to create them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: TemporalSplitBoundaries


def _closest_cumulative_index(
    cumulative_rows: list[int],
    target_rows: float,
    *,
    lo: int,
    hi: int,
) -> int:
    """Return index in ``[lo, hi]`` whose cumulative count is closest to ``target_rows``."""
    if lo > hi:
        raise ValueError(f"Invalid boundary search window: lo={lo}, hi={hi}.")
    best_idx = lo
    best_distance = abs(cumulative_rows[lo] - target_rows)
    for idx in range(lo, hi + 1):
        distance = abs(cumulative_rows[idx] - target_rows)
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx


def chronological_train_val_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    step_column: str = TEMPORAL_COLUMN,
) -> TemporalSplitResult:
    """Split ``df`` into train / validation / test by chronological ``step`` ranges.

    Boundaries are chosen from cumulative row counts by ascending ``step`` so that
    each complete step lands in exactly one split and row shares are approximately
    ``train_ratio`` / ``validation_ratio`` / ``test_ratio`` (default 70% / 15% / 15%).

    Guarantees
    ----------
    ``max(train.step) < min(validation.step)`` and
    ``max(validation.step) < min(test.step)``.
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0.")
    if step_column not in df.columns:
        raise ValueError(f"DataFrame must contain temporal column '{step_column}'.")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    step_counts = df.groupby(step_column, sort=True).size()
    unique_steps = tuple(step_counts.index.tolist())
    n_steps = len(unique_steps)
    if n_steps < 3:
        raise ValueError(
            f"Need at least 3 unique step values for train/validation/test; found {n_steps}."
        )

    cumulative_rows = step_counts.cumsum().astype(int).tolist()
    total_rows = cumulative_rows[-1]
    train_target = train_ratio * total_rows
    train_val_target = (train_ratio + validation_ratio) * total_rows

    # Leave at least one step for validation and one for test.
    train_end = _closest_cumulative_index(
        cumulative_rows,
        train_target,
        lo=0,
        hi=n_steps - 3,
    )
    validation_end = _closest_cumulative_index(
        cumulative_rows,
        train_val_target,
        lo=train_end + 1,
        hi=n_steps - 2,
    )

    train_steps = unique_steps[: train_end + 1]
    validation_steps = unique_steps[train_end + 1 : validation_end + 1]
    test_steps = unique_steps[validation_end + 1 :]

    if not train_steps or not validation_steps or not test_steps:
        raise ValueError(
            "Failed to form non-empty step ranges for all three splits. "
            f"unique_steps={n_steps}, train_end={train_end}, "
            f"validation_end={validation_end}."
        )

    boundaries = TemporalSplitBoundaries(
        train_steps=train_steps,
        validation_steps=validation_steps,
        test_steps=test_steps,
    )

    train = df.loc[df[step_column].isin(train_steps)].copy()
    validation = df.loc[df[step_column].isin(validation_steps)].copy()
    test = df.loc[df[step_column].isin(test_steps)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more temporal splits is empty after applying step boundaries.")

    assert train[step_column].max() < validation[step_column].min()
    assert validation[step_column].max() < test[step_column].min()

    return TemporalSplitResult(
        train=train,
        validation=validation,
        test=test,
        boundaries=boundaries,
    )


@dataclass(frozen=True)
class TrainCalibrationBoundaries:
    """Step boundaries for a chronological model-fit / calibration split of train."""

    model_fit_steps: tuple[int, ...]
    calibration_steps: tuple[int, ...]

    @property
    def model_fit_max_step(self) -> int:
        return max(self.model_fit_steps)

    @property
    def calibration_min_step(self) -> int:
        return min(self.calibration_steps)

    @property
    def calibration_max_step(self) -> int:
        return max(self.calibration_steps)


@dataclass(frozen=True)
class TrainCalibrationSplitResult:
    """Chronological model-fit / calibration frames carved from the training split."""

    model_fit: pd.DataFrame
    calibration: pd.DataFrame
    boundaries: TrainCalibrationBoundaries


def chronological_model_fit_calibration_split(
    train_df: pd.DataFrame,
    *,
    model_fit_ratio: float = 0.85,
    step_column: str = TEMPORAL_COLUMN,
) -> TrainCalibrationSplitResult:
    """Split the training frame into earlier model-fit and later calibration subsets.

    Boundaries follow cumulative row counts by ascending ``step`` so each complete
    step lands in exactly one subset (~``model_fit_ratio`` / remainder).

    Guarantees
    ----------
    ``max(model_fit.step) < min(calibration.step)``.
    """
    if not 0.0 < model_fit_ratio < 1.0:
        raise ValueError(f"model_fit_ratio must be in (0, 1); got {model_fit_ratio}.")
    if step_column not in train_df.columns:
        raise ValueError(f"DataFrame must contain temporal column '{step_column}'.")
    if train_df.empty:
        raise ValueError("Cannot split an empty training DataFrame.")

    step_counts = train_df.groupby(step_column, sort=True).size()
    unique_steps = tuple(step_counts.index.tolist())
    n_steps = len(unique_steps)
    if n_steps < 2:
        raise ValueError(
            f"Need at least 2 unique step values for model-fit/calibration; found {n_steps}."
        )

    cumulative_rows = step_counts.cumsum().astype(int).tolist()
    total_rows = cumulative_rows[-1]
    model_fit_target = model_fit_ratio * total_rows

    # Leave at least one step for the calibration subset.
    model_fit_end = _closest_cumulative_index(
        cumulative_rows,
        model_fit_target,
        lo=0,
        hi=n_steps - 2,
    )

    model_fit_steps = unique_steps[: model_fit_end + 1]
    calibration_steps = unique_steps[model_fit_end + 1 :]

    if not model_fit_steps or not calibration_steps:
        raise ValueError(
            "Failed to form non-empty model-fit and calibration step ranges. "
            f"unique_steps={n_steps}, model_fit_end={model_fit_end}."
        )

    boundaries = TrainCalibrationBoundaries(
        model_fit_steps=model_fit_steps,
        calibration_steps=calibration_steps,
    )

    model_fit = train_df.loc[train_df[step_column].isin(model_fit_steps)].copy()
    calibration = train_df.loc[train_df[step_column].isin(calibration_steps)].copy()

    if model_fit.empty or calibration.empty:
        raise ValueError("model-fit or calibration subset is empty after applying step boundaries.")

    assert model_fit[step_column].max() < calibration[step_column].min()

    return TrainCalibrationSplitResult(
        model_fit=model_fit,
        calibration=calibration,
        boundaries=boundaries,
    )


def split_target_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    step_column: str = TEMPORAL_COLUMN,
    model_fit: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize row count/share, fraud count/rate, and step range for each split.

    When ``model_fit`` / ``calibration`` are provided they are summarized as
    additional rows; the train / validation / test totals still define ``pct_rows``
    denominators for the primary three splits. Model-fit and calibration shares
    are reported relative to the training row count.
    """
    total_rows = len(train) + len(validation) + len(test)
    train_rows = len(train)

    def _row(name: str, frame: pd.DataFrame, *, pct_base: int) -> dict[str, Any]:
        fraud_count = int(frame[target_column].sum())
        n_rows = len(frame)
        return {
            "split": name,
            "rows": n_rows,
            "pct_rows": n_rows / pct_base if pct_base else float("nan"),
            "fraud_cases": fraud_count,
            "fraud_rate": fraud_count / n_rows if n_rows else float("nan"),
            "min_step": int(frame[step_column].min()),
            "max_step": int(frame[step_column].max()),
        }

    rows = [
        _row("train", train, pct_base=total_rows),
        _row("validation", validation, pct_base=total_rows),
        _row("test", test, pct_base=total_rows),
    ]
    if model_fit is not None:
        rows.append(_row("model_fit", model_fit, pct_base=train_rows))
    if calibration is not None:
        rows.append(_row("calibration", calibration, pct_base=train_rows))
    return pd.DataFrame(rows)


def build_modeling_dataset(df: pd.DataFrame) -> TemporalSplitResult:
    """Restrict scope, select baseline columns, and chronologically split.

    Does not mutate the input DataFrame.
    """
    scoped = restrict_to_modeling_scope(df)
    baseline = select_baseline_frame(scoped)
    return chronological_train_val_test_split(baseline)
