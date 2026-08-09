"""PaySim simulator-artifact diagnostics (not temporal leakage checks).

These helpers quantify how strongly labels align with account-draining patterns
in the synthetic PaySim generator. Features derived from pre-authorization
fields are still valid at the prediction moment; the concern here is
**simulator artifact / synthetic shortcut**, not temporal target leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_risk.dataset import TARGET_COLUMN
from fraud_risk.features import amount_to_balance_ratio

# Documented tolerance for "amount approximately equals oldbalanceOrg".
# rtol=0, atol=1e-6 treats exact equality and tiny float noise as matches without
# treating materially different amounts as equal.
AMOUNT_EQUALS_BALANCE_RTOL: float = 0.0
AMOUNT_EQUALS_BALANCE_ATOL: float = 1e-6

RATIO_PERCENTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0)


def amount_equals_origin_balance(
    amount: pd.Series | np.ndarray,
    oldbalance_org: pd.Series | np.ndarray,
    *,
    rtol: float = AMOUNT_EQUALS_BALANCE_RTOL,
    atol: float = AMOUNT_EQUALS_BALANCE_ATOL,
) -> np.ndarray:
    """Return boolean mask where ``amount`` ≈ ``oldbalanceOrg`` via ``numpy.isclose``."""
    return np.isclose(
        np.asarray(amount, dtype=float),
        np.asarray(oldbalance_org, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def _group_pattern_row(
    frame: pd.DataFrame,
    *,
    label_name: str,
    split_name: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    amount = frame["amount"].to_numpy(dtype=float)
    balance = frame["oldbalanceOrg"].to_numpy(dtype=float)
    n = len(frame)
    eq_mask = amount_equals_origin_balance(amount, balance, rtol=rtol, atol=atol)
    exceeds = amount > balance
    zero_bal = balance == 0.0
    ratios = amount_to_balance_ratio(amount, balance)

    row: dict[str, Any] = {
        "split": split_name,
        "label": label_name,
        "count": n,
        "pct_amount_eq_balance": float(eq_mask.mean()) if n else float("nan"),
        "pct_amount_exceeds_balance": float(exceeds.mean()) if n else float("nan"),
        "pct_origin_balance_zero": float(zero_bal.mean()) if n else float("nan"),
    }
    if n:
        percentile_values = np.percentile(ratios, list(RATIO_PERCENTILES))
        for pct, value in zip(RATIO_PERCENTILES, percentile_values, strict=True):
            row[f"ratio_p{int(pct)}"] = float(value)
    else:
        for pct in RATIO_PERCENTILES:
            row[f"ratio_p{int(pct)}"] = float("nan")
    return row


def drain_pattern_summary(
    df: pd.DataFrame,
    *,
    split_name: str = "all",
    target_column: str = TARGET_COLUMN,
    rtol: float = AMOUNT_EQUALS_BALANCE_RTOL,
    atol: float = AMOUNT_EQUALS_BALANCE_ATOL,
) -> pd.DataFrame:
    """Summarize account-drain patterns for fraud vs legitimate rows in ``df``."""
    required = ("amount", "oldbalanceOrg", target_column)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for drain-pattern audit: {missing}.")

    rows = [
        _group_pattern_row(
            df.loc[df[target_column] == 1],
            label_name="fraud",
            split_name=split_name,
            rtol=rtol,
            atol=atol,
        ),
        _group_pattern_row(
            df.loc[df[target_column] == 0],
            label_name="legit",
            split_name=split_name,
            rtol=rtol,
            atol=atol,
        ),
    ]
    return pd.DataFrame(rows)


def drain_pattern_by_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    rtol: float = AMOUNT_EQUALS_BALANCE_RTOL,
    atol: float = AMOUNT_EQUALS_BALANCE_ATOL,
) -> pd.DataFrame:
    """Run ``drain_pattern_summary`` on each temporal split."""
    parts = [
        drain_pattern_summary(
            frame,
            split_name=name,
            target_column=target_column,
            rtol=rtol,
            atol=atol,
        )
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    ]
    return pd.concat(parts, ignore_index=True)
