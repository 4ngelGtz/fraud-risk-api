"""Model pipelines and evaluation helpers for baseline and engineered models."""

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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_risk.dataset import BASELINE_FEATURES
from fraud_risk.features import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_MODEL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    split_categorical_numeric,
    validate_safe_feature_names,
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
NUMERIC_FEATURES: tuple[str, ...] = ("amount", "oldbalanceOrg")

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2).tolist())
DEFAULT_RECALL_FLOOR: float = 0.80
RANDOM_STATE: int = 42

# Fine grid for Task 3 operating-policy comparison (still validation-only).
FINE_THRESHOLD_GRID: tuple[float, ...] = tuple(
    np.round(np.arange(0.01, 1.0, 0.01), 2).tolist()
)

# Compact, reproducible XGBoost defaults — no hyperparameter search.
XGBOOST_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 1,
}


def build_logistic_baseline_pipeline(
    *,
    categorical_features: tuple[str, ...] | list[str] = CATEGORICAL_FEATURES,
    numeric_features: tuple[str, ...] | list[str] = NUMERIC_FEATURES,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a deterministic Logistic Regression baseline Pipeline (Model A).

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


def build_logistic_engineered_pipeline(
    *,
    categorical_features: tuple[str, ...] | list[str] = ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features: tuple[str, ...] | list[str] = ENGINEERED_NUMERIC_FEATURES,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build Logistic Regression on baseline + engineered features (Model B).

    Same preprocessing pattern as the baseline: one-hot ``type``, standardize
    numerics, ``class_weight='balanced'``. No hyperparameter tuning.
    """
    approved = set(ENGINEERED_MODEL_FEATURES)
    provided = set(categorical_features) | set(numeric_features)
    if provided != approved:
        raise ValueError(
            f"Engineered pipeline features {sorted(provided)} must match "
            f"ENGINEERED_MODEL_FEATURES {list(ENGINEERED_MODEL_FEATURES)}."
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


def build_logistic_feature_pipeline(
    feature_names: tuple[str, ...] | list[str],
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build class-weighted Logistic Regression for an approved feature subset.

    Used for diagnostics (e.g. Model B without explicit balance-drain features).
    Does not replace ``build_logistic_engineered_pipeline`` (Task 3 Model B).
    """
    names = validate_safe_feature_names(feature_names)
    categorical_features, numeric_features = split_categorical_numeric(names)
    if not categorical_features and not numeric_features:
        raise ValueError("No features remaining after categorical/numeric split.")

    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            )
        )
    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                list(numeric_features),
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
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


def compute_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Train-split-only imbalance weight for XGBoost: n_negative / n_positive.

    Must be computed exclusively from training labels — never validation or test.
    """
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight: training set has zero positives.")
    return float(n_neg) / float(n_pos)


def build_xgboost_pipeline(
    *,
    categorical_features: tuple[str, ...] | list[str] = ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features: tuple[str, ...] | list[str] = ENGINEERED_NUMERIC_FEATURES,
    scale_pos_weight: float,
    feature_names: tuple[str, ...] | list[str] | None = None,
    random_state: int = RANDOM_STATE,
    xgb_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build an XGBoost classifier pipeline.

    Default feature set is the Task 3 full engineered set (Model C). Pass
    ``feature_names`` for ablation subsets (must be approved safe features).

    ``scale_pos_weight`` must be supplied by the caller after computing it from
    the **training** labels only via ``compute_scale_pos_weight``.

    Scores from ``predict_proba`` are raw model outputs suitable for ranking and
    thresholding; they are **not** calibrated probabilities.
    """
    if feature_names is not None:
        names = validate_safe_feature_names(feature_names)
        categorical_features, numeric_features = split_categorical_numeric(names)
    else:
        approved = set(ENGINEERED_MODEL_FEATURES)
        provided = set(categorical_features) | set(numeric_features)
        if provided != approved:
            raise ValueError(
                f"XGBoost pipeline features {sorted(provided)} must match "
                f"ENGINEERED_MODEL_FEATURES {list(ENGINEERED_MODEL_FEATURES)} "
                "(or pass feature_names=... for an approved ablation subset)."
            )
    if scale_pos_weight <= 0:
        raise ValueError(f"scale_pos_weight must be positive; got {scale_pos_weight}.")

    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            )
        )
    if numeric_features:
        transformers.append(
            (
                "numeric",
                "passthrough",
                list(numeric_features),
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    params = {**XGBOOST_DEFAULT_PARAMS, **(xgb_params or {})}
    params["scale_pos_weight"] = float(scale_pos_weight)
    params["random_state"] = random_state

    classifier = XGBClassifier(**params)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def predict_proba_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores (column 1 of ``predict_proba``).

    For Logistic Regression these are class-weighted probabilities (not
    calibrated). For XGBoost these are raw booster scores mapped through the
    logistic link — also **not** calibrated probabilities.
    """
    return model.predict_proba(X)[:, 1]


def ranking_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Threshold-independent ranking metrics plus fraud prevalence."""
    y_true_arr = np.asarray(y_true)
    n = len(y_true_arr)
    prevalence = float(y_true_arr.mean()) if n else float("nan")
    return {
        "fraud_prevalence": prevalence,
        "average_precision": float(average_precision_score(y_true_arr, y_score)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score)),
    }


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute ranking and thresholded classification / operational metrics.

    Accuracy is intentionally omitted; it is misleading under severe imbalance.
    """
    y_true_arr = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    n = len(y_true_arr)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    predicted_positives = int(y_pred.sum())

    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true_arr, y_proba)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "predicted_positives": predicted_positives,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alert_rate": float(predicted_positives / n) if n else float("nan"),
    }


def candidate_operating_thresholds(
    y_proba: np.ndarray,
    *,
    grid: tuple[float, ...] | list[float] = FINE_THRESHOLD_GRID,
    include_pr_thresholds: bool = True,
) -> tuple[float, ...]:
    """Build a fine, deterministic candidate threshold set for operating policy.

    Combines a fixed fine grid (default step 0.01) with unique thresholds from
    the precision-recall curve on the provided scores (typically validation).
    """
    candidates: set[float] = {float(t) for t in grid}
    if include_pr_thresholds:
        # precision_recall_curve needs y_true; approximate unique score cutoffs instead
        # when only scores are available by using midpoints between sorted unique scores.
        scores = np.asarray(y_proba, dtype=float)
        unique = np.unique(scores)
        if unique.size >= 2:
            midpoints = ((unique[:-1] + unique[1:]) / 2.0).tolist()
            candidates.update(float(t) for t in midpoints)
        candidates.update(float(t) for t in unique.tolist())
    cleaned = sorted(t for t in candidates if 0.0 < t < 1.0)
    if not cleaned:
        return FINE_THRESHOLD_GRID
    return tuple(cleaned)


def candidate_thresholds_with_labels(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    grid: tuple[float, ...] | list[float] = FINE_THRESHOLD_GRID,
) -> tuple[float, ...]:
    """Fine grid plus rounded precision-recall curve thresholds (validation use).

    PR-curve cutoffs are rounded to 3 decimals so the candidate set stays fine
    enough for operating-policy selection without one threshold per distinct
    score on large datasets.
    """
    candidates: set[float] = {float(t) for t in grid}
    _precision, _recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    candidates.update(round(float(t), 3) for t in pr_thresholds.tolist())
    cleaned = sorted(t for t in candidates if 0.0 < t < 1.0)
    return tuple(cleaned) if cleaned else tuple(grid)


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    thresholds: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate precision / recall / F1 / alerts across thresholds (validation use).

    Uses a single sorted pass over scores so dense candidate grids stay practical
    on large validation sets.
    """
    values = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    n = len(y_true_arr)
    n_pos = int(y_true_arr.sum())
    n_neg = n - n_pos

    order = np.argsort(y_proba_arr)
    scores_sorted = y_proba_arr[order]
    labels_sorted = y_true_arr[order]

    # Cumulative fraud / non-fraud counts from the low-score end.
    cum_pos = np.concatenate(([0], np.cumsum(labels_sorted)))
    cum_neg = np.concatenate(([0], np.cumsum(1 - labels_sorted)))

    rows: list[dict[str, Any]] = []
    for t in values:
        # Predictions: score >= t. Index of first score >= t in ascending order.
        idx = int(np.searchsorted(scores_sorted, float(t), side="left"))
        fn = int(cum_pos[idx])
        tn = int(cum_neg[idx])
        tp = n_pos - fn
        fp = n_neg - tn
        predicted_positives = tp + fp
        precision = float(tp / predicted_positives) if predicted_positives else 0.0
        recall = float(tp / n_pos) if n_pos else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positives": int(predicted_positives),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "alert_rate": float(predicted_positives / n) if n else float("nan"),
            }
        )
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


def select_operating_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> tuple[float, str, pd.DataFrame]:
    """Validation-only operating threshold via fine + PR-curve candidates.

    Policy (documented recall constraint): highest precision subject to
    ``recall >= recall_floor`` (default 80%). Falls back to max-F1 if infeasible.
    """
    candidates = candidate_thresholds_with_labels(y_true, y_proba)
    table = threshold_analysis(y_true, y_proba, thresholds=candidates)
    threshold, reason = select_provisional_threshold(table, recall_floor=recall_floor)
    return threshold, reason, table


def relative_false_positive_reduction(
    baseline_false_positives: int,
    model_false_positives: int,
) -> float:
    """Fraction of baseline false positives removed: (FP_A - FP) / FP_A."""
    if baseline_false_positives <= 0:
        return float("nan")
    return float(baseline_false_positives - model_false_positives) / float(
        baseline_false_positives
    )
