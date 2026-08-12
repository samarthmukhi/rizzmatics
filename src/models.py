"""Model definitions.

We start dumb on purpose. The whole point of the experiment is to find out
whether *simple, interpretable* conversational signals carry any predictive
information — so we compare real models against a baseline that has no right to
be beaten, and we do NOT reach for a neural network to look impressive.

Every model is a scikit-learn ``Pipeline`` that:

1. median-imputes missing values (features are honestly NaN when undefined), and
2. standardizes features for the linear/logistic models (trees don't care).

The pipeline is what gets cross-validated, so imputation and scaling are refit
on each training fold — no statistics leak from validation folds into training.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["make_regressors", "make_classifiers", "BASELINE_REGRESSOR", "BASELINE_CLASSIFIER"]

BASELINE_REGRESSOR = "Mean Predictor"
BASELINE_CLASSIFIER = "Majority Class"


def _impute_scale(estimator) -> Pipeline:
    """Median-impute then standardize, then estimate (for linear models)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _impute_only(estimator) -> Pipeline:
    """Median-impute then estimate (for tree ensembles; no scaling needed)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def make_regressors(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the regression model zoo, keyed by human-readable name."""
    return {
        BASELINE_REGRESSOR: _impute_only(DummyRegressor(strategy="mean")),
        "Linear Regression": _impute_scale(LinearRegression()),
        "Random Forest": _impute_only(
            RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            )
        ),
        "Gradient Boosting": _impute_only(
            HistGradientBoostingRegressor(random_state=random_state)
        ),
    }


def make_classifiers(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the classification model zoo, keyed by human-readable name."""
    return {
        BASELINE_CLASSIFIER: _impute_only(
            DummyClassifier(strategy="most_frequent")
        ),
        "Logistic Regression": _impute_scale(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Random Forest": _impute_only(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": _impute_only(
            HistGradientBoostingClassifier(random_state=random_state)
        ),
    }
