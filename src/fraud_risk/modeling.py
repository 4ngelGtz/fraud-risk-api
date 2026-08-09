"""Logistic Regression baseline pipeline and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_risk.dataset import BASELINE_FEATURES

CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
NUMERIC_FEATURES: tuple[str, ...] = ("amount", "oldbalanceOrg")

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2).tolist())
DEFAULT_RECALL_FLOOR: float = 0.80
RANDOM_STATE: int = 42


def build_logistic_baseline_pipeline(
    *,
    categorical_features: tuple[str, ...] | list[str] = CATEGORICAL_FEATURES,
    numeric_features: tuple[str, ...] | list[str] = NUMERIC_FEATURES,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a deterministic Logistic Regression baseline Pipeline.

    Categorical ``type`` is one-hot encoded; numeric features are standardized.
    Class weighting is balanced for the strongly imbalanced fraud target.
    """
    if tuple(categorical_features) + tuple(numeric_features) != BASELINE_FEATURES:
        # Soft check: callers may reorder within groups, but the approved set must match.
        approved = set(BASELINE_FEATURES)
        provided = set(categorical_features) | set(numeric_features)
        if provided != approved:
            raise ValueError(
                f"Pipeline features {sorted(provided)} must match BASELINE_FEATURES "
                f"{list(BASELINE_FEATURES)}."
            )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            ),
            (
                "numeric",
                StandardScaler(),
                list(numeric_features),
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def predict_proba_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return fraud-class probabilities (column 1 of ``predict_proba``)."""
    return model.predict_proba(X)[:, 1]


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute ranking and thresholded classification metrics.

    Accuracy is intentionally omitted; it is misleading under severe imbalance.
    """
    y_true_arr = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true_arr, y_proba)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "predicted_positives": int(y_pred.sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    thresholds: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate precision / recall / F1 across thresholds (validation use)."""
    values = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    rows = [
        {
            "threshold": float(t),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "predicted_positives": metrics["predicted_positives"],
        }
        for t in values
        for metrics in [
            classification_metrics_at_threshold(y_true, y_proba, threshold=float(t))
        ]
    ]
    return pd.DataFrame(rows)


def select_provisional_threshold(
    threshold_table: pd.DataFrame,
    *,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> tuple[float, str]:
    """Select a simple provisional threshold from validation analysis only.

    Rule: among thresholds with recall >= ``recall_floor``, choose the one with
    highest precision (ties → higher threshold, then higher F1).

    If no threshold meets the recall floor, fall back to the threshold with
    maximum F1 and document that the preferred rule was infeasible.
    """
    if threshold_table.empty:
        raise ValueError("threshold_table is empty; cannot select a provisional threshold.")

    eligible = threshold_table.loc[threshold_table["recall"] >= recall_floor]
    if not eligible.empty:
        best = eligible.sort_values(
            by=["precision", "threshold", "f1"],
            ascending=[False, False, False],
        ).iloc[0]
        reason = (
            f"Highest-precision threshold among those with recall >= {recall_floor:.0%} "
            "(provisional portfolio-project policy; validation data only)."
        )
        return float(best["threshold"]), reason

    best = threshold_table.sort_values(
        by=["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    reason = (
        f"No threshold reached recall >= {recall_floor:.0%}; "
        "fell back to maximum-F1 threshold on validation "
        "(preferred high-recall rule was infeasible)."
    )
    return float(best["threshold"]), reason
