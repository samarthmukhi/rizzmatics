"""Phase 17 — null / negative-control regression test.

THE tripwire. If the pipeline ever scores well on a dataset with no signal, it
is leaking. This test must never be weakened to make anything else pass.
"""

import pytest

from src.research.nulldata import make_null_dataset
from src.research.sanity import run_normal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_null_regression_near_or_below_baseline(seed):
    ds = make_null_dataset(n_samples=120, seed=seed)
    r = run_normal(ds, n_repeats=2, seed=42)
    # Random features cannot explain a random target: R² must not be meaningfully positive.
    assert r.regression.mean("R2") < 0.1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_null_classification_near_chance(seed):
    ds = make_null_dataset(n_samples=120, seed=seed)
    r = run_normal(ds, n_repeats=2, seed=42)
    auc = r.classification.mean("roc_auc")
    assert 0.30 < auc < 0.65, f"AUC={auc} — suspiciously far from chance, suspect leakage"


def test_null_dataset_shape_and_independence():
    ds = make_null_dataset(n_samples=50, seed=0)
    assert len(ds) == 50
    assert ds.X.shape[1] == 30  # all real feature names, random values
    # Class labels exist for both classes at the 75th percentile.
    assert set(ds.y_classification.unique()) == {0, 1}
