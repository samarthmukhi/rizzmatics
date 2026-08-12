"""Tests for evaluation metrics and explainability."""

import numpy as np
import pytest

from src.evaluation import (
    classification_metrics,
    evaluate_classifiers,
    evaluate_regressors,
    prediction_drivers,
    regression_metrics,
)
from src.models import make_regressors


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def test_regression_metrics_perfect_prediction():
    y = [0.1, 0.5, 0.9]
    m = regression_metrics(y, y)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["R2"] == pytest.approx(1.0)


def test_regression_metrics_values():
    m = regression_metrics([0.0, 0.0], [1.0, 1.0])
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(1.0)


def test_classification_metrics_perfect():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_single_class_auc_nan():
    m = classification_metrics([1, 1, 1], [1, 1, 1])
    assert np.isnan(m["roc_auc"])


# --------------------------------------------------------------------------- #
# Model evaluation on a learnable synthetic signal
# --------------------------------------------------------------------------- #
def _learnable_data(n=60, seed=0):
    """A dataset where feature 0 genuinely drives the target."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({
        "signal": x0,
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    y = 0.5 + 0.3 * x0 + noise  # engagement-like, roughly [0, 1]
    return X, pd.Series(y)


def test_regressor_beats_baseline_on_learnable_signal():
    X, y = _learnable_data()
    report = evaluate_regressors(X, y)
    baseline_r2 = report.results["Mean Predictor"].metrics["R2"]
    rf_r2 = report.results["Random Forest"].metrics["R2"]
    assert rf_r2 > baseline_r2
    assert report.cross_validated is True


def test_classifier_report_runs_and_reports_folds():
    X, y = _learnable_data()
    y_bin = (y >= y.median()).astype(int)
    report = evaluate_classifiers(X, y_bin)
    assert report.task == "classification"
    assert report.n_splits >= 2
    for r in report.results.values():
        assert 0.0 <= r.metrics["accuracy"] <= 1.0


def test_small_dataset_caps_folds_but_still_cross_validates():
    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.1, 0.2, 0.3, 0.4])
    report = evaluate_regressors(X, y, n_splits=5)
    # 4 samples can't do 5-fold; folds are capped to 4 (still real CV).
    assert report.n_splits == 4
    assert report.cross_validated is True


def test_single_sample_falls_back_to_in_sample_with_warning():
    import pandas as pd

    X = pd.DataFrame({"a": [1.0]})
    y = pd.Series([0.5])
    report = evaluate_regressors(X, y, n_splits=5)
    # One sample can't be cross-validated at all -> honest in-sample flag.
    assert report.cross_validated is False
    assert any("IN-SAMPLE" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_prediction_drivers_ranks_the_real_signal_first():
    X, y = _learnable_data(n=120)
    drivers = prediction_drivers(
        make_regressors()["Random Forest"], X, y, task="regression", top_k=3
    )
    assert drivers[0].feature == "signal"
    assert drivers[0].direction == "up"  # positive coefficient
    # Importance is sorted descending.
    assert drivers[0].importance >= drivers[-1].importance
