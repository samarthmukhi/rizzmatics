"""Sanity experiments — does the benchmark behave like real ML should?

A high score is meaningless unless the pipeline also *fails* when it should:

* **Normal** — the honest baseline on the real (synthetic) targets.
* **Shuffled target** — permute the labels. A valid pipeline collapses toward
  chance, because there is no longer anything to learn. If it doesn't, the
  pipeline is leaking or the metric is lying.
* **Feature destruction** — randomize a signal family and watch performance
  drop. This shows whether the model actually uses the intended signals.

None of this is meant to make the benchmark look better. It is meant to prove
the benchmark is trustworthy enough to be worth reporting at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import FEATURE_GROUPS
from ..models import make_classifiers, make_regressors
from ..preprocessing import Dataset
from .experiment import CVMetrics, cross_val_metrics

__all__ = ["SanityResult", "run_normal", "run_shuffled_target", "run_feature_destruction"]


@dataclass
class SanityResult:
    """Paired regression + classification result for one sanity condition."""

    name: str
    regression: CVMetrics
    classification: CVMetrics

    def headline(self) -> dict[str, str]:
        return {
            "R2": self.regression.format_metric("R2"),
            "F1": self.classification.format_metric("f1"),
            "ROC-AUC": self.classification.format_metric("roc_auc"),
        }


def _evaluate(X, y_reg, y_clf, *, n_splits, n_repeats, seed) -> tuple[CVMetrics, CVMetrics]:
    reg = cross_val_metrics(
        make_regressors(seed)["Random Forest"], X, y_reg,
        task="regression", n_splits=n_splits, n_repeats=n_repeats, random_state=seed,
    )
    clf = cross_val_metrics(
        make_classifiers(seed)["Random Forest"], X, y_clf,
        task="classification", n_splits=n_splits, n_repeats=n_repeats, random_state=seed,
    )
    return reg, clf


def run_normal(ds: Dataset, *, n_splits: int = 5, n_repeats: int = 5,
               seed: int = 42) -> SanityResult:
    """Baseline: the real synthetic targets, untouched."""
    reg, clf = _evaluate(
        ds.X, ds.y_regression, ds.y_classification,
        n_splits=n_splits, n_repeats=n_repeats, seed=seed,
    )
    return SanityResult("normal", reg, clf)


def run_shuffled_target(ds: Dataset, *, n_splits: int = 5, n_repeats: int = 5,
                        seed: int = 42) -> SanityResult:
    """Permute the targets; features untouched. Performance should collapse."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds.X))
    y_reg = np.asarray(ds.y_regression)[perm]
    y_clf = np.asarray(ds.y_classification)[perm]
    reg, clf = _evaluate(
        ds.X, y_reg, y_clf,
        n_splits=n_splits, n_repeats=n_repeats, seed=seed,
    )
    return SanityResult("shuffled_target", reg, clf)


def run_feature_destruction(ds: Dataset, groups: list[str], *, n_splits: int = 5,
                            n_repeats: int = 5, seed: int = 42) -> SanityResult:
    """Randomize (within-column shuffle) the named feature groups, then re-run.

    Within-column shuffling preserves each feature's marginal distribution while
    destroying its relationship to the target and to the other features.
    """
    rng = np.random.default_rng(seed)
    X = ds.X.copy()
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group {g!r}")
        for col in FEATURE_GROUPS[g]:
            if col in X.columns:
                X[col] = rng.permutation(X[col].to_numpy())
    reg, clf = _evaluate(
        X, ds.y_regression, ds.y_classification,
        n_splits=n_splits, n_repeats=n_repeats, seed=seed,
    )
    return SanityResult(f"destroyed[{'+'.join(groups)}]", reg, clf)
