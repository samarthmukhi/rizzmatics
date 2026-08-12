"""Phase 15 — evaluation-methodology audit.

These tests assert the properties that make the cross-validation trustworthy:
preprocessing is refit per fold (no scaling leak), the passed model is never
mutated (proving we clone), each sample appears once, uncertainty is reported,
and the small-sample caveat fires.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models import make_classifiers, make_regressors
from src.parser import parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions
from src.research.experiment import (
    SMALL_DATA_THRESHOLD,
    cross_val_metrics,
    small_data_caveat,
)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(detect_sessions(parse_file("data/demo/demo_chat.txt")), prefix=10)


# --------------------------------------------------------------------------- #
# Preprocessing lives inside the pipeline (fit per fold, not globally).
# --------------------------------------------------------------------------- #
def test_models_carry_preprocessing_in_pipeline():
    for model in list(make_regressors().values()) + list(make_classifiers().values()):
        steps = dict(model.named_steps)
        assert "impute" in steps  # imputation is part of the estimator, not pre-done


def test_cross_val_does_not_mutate_the_passed_model(dataset):
    model = make_regressors()["Random Forest"]
    cross_val_metrics(model, dataset.X, dataset.y_regression, task="regression", n_repeats=1)
    # If we cloned properly, the ORIGINAL model is still unfitted.
    with pytest.raises(NotFittedError):
        model.predict(dataset.X)


# --------------------------------------------------------------------------- #
# No sample duplication; folds are disjoint by construction.
# --------------------------------------------------------------------------- #
def test_dataset_has_no_duplicate_sessions(dataset):
    assert len(set(dataset.session_ids)) == len(dataset.session_ids)
    assert not dataset.X.index.duplicated().any()


# --------------------------------------------------------------------------- #
# Uncertainty is actually reported.
# --------------------------------------------------------------------------- #
def test_cross_val_reports_multiple_estimates_and_std(dataset):
    cv = cross_val_metrics(make_regressors()["Random Forest"], dataset.X,
                           dataset.y_regression, task="regression",
                           n_splits=5, n_repeats=3)
    summary = cv.summary()
    assert summary["R2"]["n"] > 1                 # more than one fold estimate
    assert not np.isnan(summary["R2"]["std"])      # a real spread is reported
    assert "±" in cv.format_metric("R2")


def test_tiny_data_falls_back_to_in_sample_flag():
    X = pd.DataFrame({"a": [1.0]})
    cv = cross_val_metrics(
        make_regressors()["Mean Predictor"], X, [0.5], task="regression"
    )  # 1 sample -> cannot cross-validate
    assert cv.cross_validated is False
    assert any("IN-SAMPLE" in n for n in cv.notes)


# --------------------------------------------------------------------------- #
# Small-data caveat.
# --------------------------------------------------------------------------- #
def test_small_data_caveat_fires_below_threshold():
    assert small_data_caveat(31) is not None
    assert "SMALL SAMPLE" in small_data_caveat(31)


def test_small_data_caveat_silent_above_threshold():
    assert small_data_caveat(SMALL_DATA_THRESHOLD) is None
    assert small_data_caveat(500) is None
