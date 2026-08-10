"""Probability calibration via simple Platt (sigmoid) scaling.

The selected XGBoost deployment model is trained with class imbalance weighting,
so its raw scores are useful for ranking but are not automatically calibrated
probabilities. Platt scaling fits a one-dimensional Logistic Regression from the
model's raw margin to the binary fraud label on a chronological calibration
subset of training data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from fraud_risk.modeling import RANDOM_STATE

# Stable identifier for the calibrated XGB Transformed deployment package.
DEPLOYMENT_MODEL_VERSION: str = "xgb-transformed-v1"


def predict_raw_margin(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return XGBoost raw margin (pre-sigmoid) scores from a fitted pipeline.

    Applies the pipeline preprocessor, then calls the XGB classifier with
    ``output_margin=True``. These margins are the recommended input to Platt
    scaling.
    """
    if "preprocess" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("Expected a Pipeline with 'preprocess' and 'classifier' steps.")
    features = model.named_steps["preprocess"].transform(X)
    classifier = model.named_steps["classifier"]
    margins = classifier.predict(features, output_margin=True)
    return np.asarray(margins, dtype=float).reshape(-1)


@dataclass
class PlattCalibrator:
    """One-dimensional sigmoid / Platt calibrator.

    Maps a raw model score (typically an XGBoost margin) to an estimated
    probability via Logistic Regression fit on a held-out calibration set.
    """

    random_state: int = RANDOM_STATE
    _classifier: LogisticRegression | None = None

    def fit(
        self,
        raw_scores: np.ndarray | pd.Series,
        y_true: np.ndarray | pd.Series,
    ) -> PlattCalibrator:
        """Fit Platt scaling on calibration-subset scores and labels only."""
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        labels = np.asarray(y_true).reshape(-1)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(
                f"raw_scores length ({scores.shape[0]}) must match y_true ({labels.shape[0]})."
            )
        if scores.shape[0] == 0:
            raise ValueError("Cannot fit calibrator on an empty calibration set.")
        if len(np.unique(labels)) < 2:
            raise ValueError("Calibration labels must contain both classes.")

        # Unweighted LR: class weighting would re-distort probability estimates.
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        classifier.fit(scores.reshape(-1, 1), labels)
        self._classifier = classifier
        return self

    def transform(self, raw_scores: np.ndarray | pd.Series) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``."""
        if self._classifier is None:
            raise RuntimeError("PlattCalibrator must be fit before transform.")
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if scores.size == 0:
            return np.asarray([], dtype=float)
        proba = self._classifier.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.asarray(proba, dtype=float)

    def fit_transform(
        self,
        raw_scores: np.ndarray | pd.Series,
        y_true: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Fit on ``raw_scores`` / ``y_true`` and return calibrated probabilities."""
        return self.fit(raw_scores, y_true).transform(raw_scores)

    @property
    def is_fitted(self) -> bool:
        return self._classifier is not None

    @property
    def coef_(self) -> float:
        if self._classifier is None:
            raise RuntimeError("PlattCalibrator must be fit before reading coef_.")
        return float(self._classifier.coef_.ravel()[0])

    @property
    def intercept_(self) -> float:
        if self._classifier is None:
            raise RuntimeError("PlattCalibrator must be fit before reading intercept_.")
        return float(self._classifier.intercept_.ravel()[0])


def fit_platt_calibrator(
    raw_scores: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> PlattCalibrator:
    """Fit a :class:`PlattCalibrator` on calibration-subset scores and labels."""
    return PlattCalibrator(random_state=random_state).fit(raw_scores, y_true)


def calibrate_scores(
    calibrator: PlattCalibrator,
    raw_scores: np.ndarray | pd.Series,
) -> np.ndarray:
    """Transform raw scores into calibrated probabilities."""
    return calibrator.transform(raw_scores)


def probability_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Brier score and log loss for probability-quality evaluation."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_proba_arr = np.asarray(y_proba, dtype=float).reshape(-1)
    return {
        "brier_score": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr, labels=[0, 1])),
    }


def reliability_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Build a compact reliability table (predicted vs observed fraud rate).

    Default ``strategy='quantile'`` is preferred under severe class imbalance so
    that bins are not dominated by a single low-probability mass. Empty or
    collapsed bins are labeled explicitly.
    """
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    y_proba_arr = np.asarray(y_proba, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}.")

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(y_proba_arr, quantiles))
        if edges.size < 2:
            edges = np.array([float(y_proba_arr.min()), float(y_proba_arr.max()) + 1e-12])
    elif strategy == "uniform":
        lo = float(np.min(y_proba_arr))
        hi = float(np.max(y_proba_arr))
        if lo == hi:
            edges = np.array([lo, hi + 1e-12])
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
    else:
        raise ValueError(f"Unknown bin strategy: {strategy!r}. Use 'quantile' or 'uniform'.")

    # digitize with right-closed last bin
    bin_ids = np.digitize(y_proba_arr, edges[1:-1], right=False)

    rows: list[dict[str, Any]] = []
    n_actual_bins = len(edges) - 1
    for bin_id in range(n_actual_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        lo_edge = float(edges[bin_id])
        hi_edge = float(edges[bin_id + 1])
        if count == 0:
            rows.append(
                {
                    "bin": bin_id,
                    "bin_label": f"[{lo_edge:.6g}, {hi_edge:.6g})",
                    "count": 0,
                    "mean_predicted": float("nan"),
                    "observed_fraud_rate": float("nan"),
                    "status": "empty",
                }
            )
            continue
        pred = y_proba_arr[mask]
        obs = y_true_arr[mask]
        rows.append(
            {
                "bin": bin_id,
                "bin_label": f"[{lo_edge:.6g}, {hi_edge:.6g})",
                "count": count,
                "mean_predicted": float(pred.mean()),
                "observed_fraud_rate": float(obs.mean()),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    title: str = "Reliability diagram",
    ax: plt.Axes | None = None,
) -> tuple[plt.Axes, pd.DataFrame]:
    """Plot a reliability diagram with the compact calibration table overlaid as points."""
    table = reliability_table(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    usable = table.loc[table["status"] == "ok"]
    if not usable.empty:
        ax.plot(
            usable["mean_predicted"],
            usable["observed_fraud_rate"],
            marker="o",
            linestyle="-",
            label="Empirical",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    if created_fig:
        plt.tight_layout()
    return ax, table
