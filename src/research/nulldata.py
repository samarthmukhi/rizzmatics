"""Null / negative-control dataset.

Features that are pure noise, a target that is independent of them. A correct
research pipeline MUST fail here: regression near baseline (R² ≈ 0 or below),
classification near chance (AUC ≈ 0.5), feature importance unstable and
uninformative.

This is the most important regression test in the whole project. If Rizzmatics
ever scores well on the null dataset, something is leaking — stop and investigate
before trusting any other number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engagement import label_high_engagement
from ..features import ALL_FEATURES
from ..preprocessing import Dataset

__all__ = ["make_null_dataset"]


def make_null_dataset(n_samples: int = 120, *, high_percentile: float = 75.0,
                      seed: int = 0) -> Dataset:
    """Build a Dataset whose features carry no information about the target.

    Feature columns reuse the real feature names (so the null control flows
    through the exact same code paths), but the values are random noise and the
    target is drawn independently. There is, by construction, nothing to learn.
    """
    rng = np.random.default_rng(seed)

    # Random features with the real column names and a mix of scales.
    data = {name: rng.normal(size=n_samples) for name in ALL_FEATURES}
    X = pd.DataFrame(data)
    X.index.name = "session_id"

    # A target that is independent of every feature.
    y_reg = pd.Series(rng.uniform(0.0, 1.0, size=n_samples),
                      index=X.index, name="engagement_index")
    y_clf, threshold = label_high_engagement(y_reg, percentile=high_percentile)

    return Dataset(
        X=X, y_regression=y_reg, y_classification=y_clf,
        high_threshold=threshold, prefix=-1, session_ids=list(X.index),
    )
