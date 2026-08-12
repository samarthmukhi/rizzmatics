"""Cross-validated metrics with honest uncertainty.

The V1 evaluation reported a single pooled metric per model. That is fine for a
demo and dangerous for a claim. With only ~31 synthetic sessions, a single R²
is a point estimate with a large, invisible error bar. This module makes the
error bar visible: it computes each metric *per fold* (optionally repeated) and
reports mean ± standard deviation across folds, plus the number of estimates.

The whole pipeline (imputation, scaling, model) is cloned and refit inside every
fold via scikit-learn, so no preprocessing statistic ever leaks from a test fold
into training. Feature-group subsetting happens on the raw DataFrame *before*
the split, which only ever removes columns — never rows — so it cannot leak either.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from ..features import FEATURE_GROUPS

__all__ = [
    "CVMetrics",
    "cross_val_metrics",
    "subset_feature_groups",
    "small_data_caveat",
    "SMALL_DATA_THRESHOLD",
    "REGRESSION_METRICS",
    "CLASSIFICATION_METRICS",
]

# Below this many samples, treat every metric as a fragile point estimate.
SMALL_DATA_THRESHOLD = 100


def small_data_caveat(n_samples: int) -> str | None:
    """Return a warning string when the sample is too small to be stable.

    Returns ``None`` above the threshold. Used verbatim by the dashboard, the
    CLI, and the research report so the same honest caveat appears everywhere.
    """
    if n_samples >= SMALL_DATA_THRESHOLD:
        return None
    return (
        f"SMALL SAMPLE ({n_samples} sessions). Metrics are point estimates with "
        "wide error bars — read the ± std, not just the mean. Nothing here is a "
        "stable or definitive result; it demonstrates pipeline behavior, not "
        "real-world validity."
    )

REGRESSION_METRICS = ("R2", "MAE", "RMSE")
CLASSIFICATION_METRICS = ("accuracy", "f1", "roc_auc")


@dataclass
class CVMetrics:
    """Per-fold metric estimates and their summary statistics."""

    task: str                        # "regression" | "classification"
    n_samples: int
    n_splits: int
    n_repeats: int
    cross_validated: bool
    per_fold: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{metric: {mean, std, n}}`` ignoring NaN estimates."""
        out: dict[str, dict[str, float]] = {}
        for metric, vals in self.per_fold.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                out[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            else:
                out[metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n": int(arr.size),
                }
        return out

    def mean(self, metric: str) -> float:
        return self.summary().get(metric, {}).get("mean", float("nan"))

    def std(self, metric: str) -> float:
        return self.summary().get(metric, {}).get("std", float("nan"))

    def format_metric(self, metric: str) -> str:
        s = self.summary().get(metric)
        if not s or s["n"] == 0:
            return "n/a"
        return f"{s['mean']:.3f} ± {s['std']:.3f}"


def subset_feature_groups(X: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Return only the columns belonging to the named feature groups.

    Raises ``ValueError`` for an unknown group so a typo can't silently produce
    an empty (and misleadingly terrible) feature set.
    """
    cols: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group {g!r}. "
                f"Known groups: {sorted(FEATURE_GROUPS)}"
            )
        cols.extend(c for c in FEATURE_GROUPS[g] if c in X.columns)
    return X[cols].copy()


def _safe_splits(n_samples: int, y, *, stratified: bool, requested: int) -> int:
    n = min(requested, n_samples)
    if stratified and y is not None:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        n = min(n, int(counts.min()))
    return n


def cross_val_metrics(
    model,
    X: pd.DataFrame,
    y,
    *,
    task: str = "regression",
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
) -> CVMetrics:
    """Cross-validate ``model`` and return per-fold metrics + uncertainty.

    Args:
        model: An unfitted scikit-learn estimator/pipeline. It is cloned per fold.
        X: Feature DataFrame.
        y: Targets (float for regression, int {0,1} for classification).
        task: ``"regression"`` or ``"classification"``.
        n_splits: Requested CV folds (capped for small data / rare classes).
        n_repeats: Repeat the whole CV this many times with different shuffles
            for a more stable uncertainty estimate.
        random_state: Seed for reproducibility.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    n = len(y)
    notes: list[str] = []

    stratified = task == "classification"
    folds = _safe_splits(n, y if stratified else None,
                         stratified=stratified, requested=n_splits)

    if folds < 2:
        # Too small to cross-validate: fall back to honest in-sample.
        notes.append(
            f"Only {n} samples (folds={folds}) — cannot cross-validate. "
            "Reporting IN-SAMPLE metrics; treat as optimistic, not evidence."
        )
        model = clone(model).fit(X, y)
        pred = model.predict(X)
        per_fold = _score_fold(task, y, pred, _proba(model, X))
        per_fold = {k: [v] for k, v in per_fold.items()}
        return CVMetrics(task, n, folds, 1, False, per_fold, notes)

    if stratified:
        splitter = RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=n_repeats, random_state=random_state)
    else:
        splitter = RepeatedKFold(
            n_splits=folds, n_repeats=n_repeats, random_state=random_state)

    per_fold: dict[str, list[float]] = defaultdict(list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in splitter.split(X, y):
            est = clone(model).fit(X.iloc[train_idx], y[train_idx])
            pred = est.predict(X.iloc[test_idx])
            scores = _score_fold(
                task, y[test_idx], pred, _proba(est, X.iloc[test_idx])
            )
            for k, v in scores.items():
                per_fold[k].append(v)

    return CVMetrics(task, n, folds, n_repeats, True, dict(per_fold), notes)


def _proba(estimator, X):
    """Positive-class probabilities if available, else ``None``."""
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        p = estimator.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else None
    except (ValueError, AttributeError, IndexError):
        return None


def _score_fold(task: str, y_true, y_pred, y_score) -> dict[str, float]:
    if task == "regression":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return {
            "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
    return out
