"""Evaluation and explainability.

Honest metrics or nothing. This module cross-validates the model zoo, reports
the standard regression and classification metrics, and explains what drove the
predictions via permutation importance. When the dataset is too small to
cross-validate responsibly, it says so out loud instead of manufacturing
impressive statistics from a handful of conversations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from .models import make_classifiers, make_regressors

__all__ = [
    "regression_metrics",
    "classification_metrics",
    "evaluate_regressors",
    "evaluate_classifiers",
    "prediction_drivers",
    "ModelResult",
    "EvaluationReport",
]


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R² for a regression prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

    ROC-AUC is ``NaN`` when only one class is present (undefined, not zero).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist()
    return metrics


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    """Cross-validated result for one model."""

    name: str
    metrics: dict
    predictions: np.ndarray
    scores: np.ndarray | None = None


@dataclass
class EvaluationReport:
    """A full evaluation over the model zoo."""

    task: str                      # "regression" | "classification"
    results: dict[str, ModelResult]
    n_samples: int
    n_splits: int
    cross_validated: bool
    notes: list[str] = field(default_factory=list)

    @property
    def best_model(self) -> str:
        """Best model name (highest R² / F1). Baselines can win — that's data."""
        if self.task == "regression":
            return max(self.results, key=lambda k: self.results[k].metrics["R2"])
        return max(self.results, key=lambda k: self.results[k].metrics["f1"])


def _safe_n_splits(n_samples: int, y=None, *, stratified: bool, requested: int) -> int:
    """Pick a fold count that won't blow up on a tiny dataset."""
    n = min(requested, n_samples)
    if stratified and y is not None:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        n = min(n, int(counts.min()))
    return max(n, 0)


def evaluate_regressors(X, y, models=None, *, n_splits: int = 5,
                        random_state: int = 42) -> EvaluationReport:
    """Cross-validate every regressor and report MAE/RMSE/R²."""
    models = models or make_regressors(random_state)
    y = np.asarray(y, dtype=float)
    n = len(y)
    notes: list[str] = []

    folds = _safe_n_splits(n, stratified=False, requested=n_splits)
    cross_validated = folds >= 2
    if not cross_validated:
        notes.append(
            f"Only {n} sessions — too few to cross-validate. Metrics below are "
            "IN-SAMPLE and optimistic. Treat them as vibes, not evidence."
        )

    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        if cross_validated:
            cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
            preds = cross_val_predict(model, X, y, cv=cv)
        else:
            preds = model.fit(X, y).predict(X)
        results[name] = ModelResult(name, regression_metrics(y, preds), preds)

    return EvaluationReport("regression", results, n, folds, cross_validated, notes)


def evaluate_classifiers(X, y, models=None, *, n_splits: int = 5,
                         random_state: int = 42) -> EvaluationReport:
    """Cross-validate every classifier and report the full metric suite."""
    models = models or make_classifiers(random_state)
    y = np.asarray(y, dtype=int)
    n = len(y)
    notes: list[str] = []

    folds = _safe_n_splits(n, y, stratified=True, requested=n_splits)
    cross_validated = folds >= 2
    if len(np.unique(y)) < 2:
        notes.append(
            "Only one engagement class present — classification is undefined. "
            "Adjust the HIGH-engagement percentile or feed me more variety."
        )
    if not cross_validated:
        notes.append(
            f"Only {n} sessions (smallest class limits folds) — metrics are "
            "IN-SAMPLE and optimistic. Do not put these in a slide deck."
        )

    results: dict[str, ModelResult] = {}
    for name, model in models.items():
        scores = None
        if cross_validated:
            cv = StratifiedKFold(n_splits=folds, shuffle=True,
                                 random_state=random_state)
            preds = cross_val_predict(model, X, y, cv=cv)
            try:
                proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
                scores = proba[:, 1]
            except (ValueError, AttributeError):
                scores = None
        else:
            model.fit(X, y)
            preds = model.predict(X)
            try:
                scores = model.predict_proba(X)[:, 1]
            except (ValueError, AttributeError, IndexError):
                scores = None
        results[name] = ModelResult(
            name, classification_metrics(y, preds, scores), preds, scores
        )

    return EvaluationReport("classification", results, n, folds, cross_validated, notes)


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
@dataclass
class Driver:
    """One prediction driver: a feature, its importance, and its direction."""

    feature: str
    importance: float
    direction: str  # "up" | "down" | "flat"


def prediction_drivers(model, X, y, *, task: str = "regression",
                       top_k: int = 8, random_state: int = 42) -> list[Driver]:
    """Explain what drove predictions via permutation importance.

    Importance is the drop in performance when a feature is shuffled. Direction
    is the sign of the feature's correlation with the target — purely a
    descriptive "moves with / against", NEVER a psychological claim.
    """
    import pandas as pd

    model = model.fit(X, y)
    scoring = "r2" if task == "regression" else "f1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=10,
            random_state=random_state,
        )

    cols = list(X.columns)
    y_arr = np.asarray(y, dtype=float)
    drivers: list[Driver] = []
    for i, col in enumerate(cols):
        imp = float(result.importances_mean[i])
        feat = pd.to_numeric(X[col], errors="coerce")
        if feat.notna().sum() > 1 and feat.std(skipna=True) > 0:
            corr = float(np.corrcoef(
                feat.fillna(feat.median()), y_arr
            )[0, 1])
        else:
            corr = 0.0
        direction = "up" if corr > 0.05 else "down" if corr < -0.05 else "flat"
        drivers.append(Driver(col, imp, direction))

    drivers.sort(key=lambda d: d.importance, reverse=True)
    return drivers[:top_k]
