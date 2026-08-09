"""Leakage-safe feature engineering from pre-authorization fields only.

All derived features use only ``type``, ``amount``, and ``oldbalanceOrg``.
No IDs, post-transaction balances, destination balances, or rule flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_risk.dataset import BASELINE_FEATURES, EXCLUDED_BASELINE_COLUMNS

# Zero-balance ratio definition (deterministic, finite, no NaN/inf):
#   if oldbalanceOrg > 0: amount / oldbalanceOrg
#   if oldbalanceOrg == 0: amount / 1.0  (treat zero balance as unit balance)
# Rationale: dividing by zero is undefined; using 1.0 keeps a finite magnitude
# proportional to the requested amount when the originator has no pre-balance.
ZERO_BALANCE_RATIO_DENOMINATOR: float = 1.0

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "log_amount",
    "log_origin_balance",
    "amount_to_balance_ratio",
    "origin_balance_zero",
    "amount_exceeds_balance",
)

# Model B / C predictors: original baseline columns plus engineered features.
ENGINEERED_MODEL_FEATURES: tuple[str, ...] = BASELINE_FEATURES + ENGINEERED_FEATURE_NAMES

ENGINEERED_CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
ENGINEERED_NUMERIC_FEATURES: tuple[str, ...] = tuple(
    f for f in ENGINEERED_MODEL_FEATURES if f != "type"
)

# Features that explicitly encode PaySim account-draining behavior.
BALANCE_DRAIN_FEATURE_NAMES: tuple[str, ...] = (
    "amount_to_balance_ratio",
    "amount_exceeds_balance",
)

# Task 3A ablation feature sets (XGBoost).
XGB_CORE_FEATURES: tuple[str, ...] = BASELINE_FEATURES
XGB_TRANSFORMED_FEATURES: tuple[str, ...] = (
    "type",
    "amount",
    "oldbalanceOrg",
    "log_amount",
    "log_origin_balance",
    "origin_balance_zero",
)
XGB_FULL_FEATURES: tuple[str, ...] = ENGINEERED_MODEL_FEATURES

# Optional LogReg diagnostic: Model B without explicit drain encodings.
LOGREG_TRANSFORMED_FEATURES: tuple[str, ...] = XGB_TRANSFORMED_FEATURES


def split_categorical_numeric(
    feature_names: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a feature list into categorical (``type``) and numeric columns."""
    names = tuple(feature_names)
    categorical = tuple(name for name in names if name == "type")
    numeric = tuple(name for name in names if name != "type")
    return categorical, numeric


def validate_safe_feature_names(feature_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Ensure feature names are a non-empty subset of approved engineered model features."""
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must be non-empty.")
    unknown = set(names) - set(ENGINEERED_MODEL_FEATURES)
    if unknown:
        raise ValueError(
            f"Unknown or disallowed features: {sorted(unknown)}. "
            f"Allowed: {list(ENGINEERED_MODEL_FEATURES)}."
        )
    forbidden = set(EXCLUDED_BASELINE_COLUMNS) & set(names)
    if forbidden:
        raise ValueError(f"Excluded / leakage columns requested as features: {sorted(forbidden)}.")
    return names


def amount_to_balance_ratio(
    amount: pd.Series | np.ndarray,
    oldbalance_org: pd.Series | np.ndarray,
) -> np.ndarray:
    """Compute amount / balance with a finite fallback when balance is zero.

    Definition
    ----------
    ``ratio = amount / oldbalanceOrg`` when ``oldbalanceOrg > 0``;
    ``ratio = amount / ZERO_BALANCE_RATIO_DENOMINATOR`` (``1.0``) when
    ``oldbalanceOrg == 0``.

    This avoids division by zero and never produces NaN or infinity for finite
    non-negative inputs.
    """
    amount_arr = np.asarray(amount, dtype=float)
    balance_arr = np.asarray(oldbalance_org, dtype=float)
    denominator = np.where(
        balance_arr > 0.0,
        balance_arr,
        ZERO_BALANCE_RATIO_DENOMINATOR,
    )
    return amount_arr / denominator


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with approved engineered columns appended.

    Requires baseline columns ``amount`` and ``oldbalanceOrg``. Does not mutate
    the input. Does not read or require any excluded / leakage columns.
    """
    required = ("amount", "oldbalanceOrg")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for feature engineering: {missing}.")

    leaked = [col for col in EXCLUDED_BASELINE_COLUMNS if col in ENGINEERED_FEATURE_NAMES]
    if leaked:
        raise RuntimeError(f"Engineered feature names collide with excluded columns: {leaked}.")

    out = df.copy()
    amount = out["amount"].astype(float)
    balance = out["oldbalanceOrg"].astype(float)

    out["log_amount"] = np.log1p(amount)
    out["log_origin_balance"] = np.log1p(balance)
    out["amount_to_balance_ratio"] = amount_to_balance_ratio(amount, balance)
    out["origin_balance_zero"] = (balance == 0.0).astype(np.int8)
    out["amount_exceeds_balance"] = (amount > balance).astype(np.int8)

    _assert_finite_engineered(out)
    return out


def select_model_feature_frame(
    df: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Select model feature columns only (no target, step, or excluded fields)."""
    names = list(feature_names)
    missing = [col for col in names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}.")
    forbidden = set(EXCLUDED_BASELINE_COLUMNS) & set(names)
    if forbidden:
        raise ValueError(f"Excluded / leakage columns requested as features: {sorted(forbidden)}.")
    return df.loc[:, names].copy()


def prepare_engineered_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and return the Model B/C feature matrix columns.

    Input must already contain baseline predictors (typically after
    ``select_baseline_frame`` / temporal split). Output contains only
    ``ENGINEERED_MODEL_FEATURES``.
    """
    enriched = add_engineered_features(df)
    return select_model_feature_frame(enriched, feature_names=ENGINEERED_MODEL_FEATURES)


def prepare_model_frame(
    df: pd.DataFrame,
    feature_names: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Add engineered features as needed and select ``feature_names``.

    Safe for ablation studies: any subset of ``ENGINEERED_MODEL_FEATURES``.
    """
    names = validate_safe_feature_names(feature_names)
    needs_engineering = any(name in ENGINEERED_FEATURE_NAMES for name in names)
    frame = add_engineered_features(df) if needs_engineering else df
    return select_model_feature_frame(frame, feature_names=names)


def _assert_finite_engineered(df: pd.DataFrame) -> None:
    """Raise if any engineered column contains NaN or non-finite values."""
    for name in ENGINEERED_FEATURE_NAMES:
        values = df[name].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Engineered feature '{name}' contains NaN or infinity.")
