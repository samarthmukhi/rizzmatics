"""Phase 11 — sanity checks.

These tests encode the promise that the benchmark behaves like valid ML:
performance must collapse when the target is shuffled or the features destroyed.
The shuffled-target guard is the important one — it catches a pipeline that
scores suspiciously well on noise (the classic sign of leakage).
"""

import numpy as np
import pytest

from src.parser import parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions
from src.research.sanity import (
    run_feature_destruction,
    run_normal,
    run_shuffled_target,
)

DEMO = "data/demo/demo_chat.txt"


@pytest.fixture(scope="module")
def dataset():
    sessions = detect_sessions(parse_file(DEMO))
    return build_dataset(sessions, prefix=10)


@pytest.fixture(scope="module")
def normal(dataset):
    return run_normal(dataset, n_repeats=2, seed=42)


@pytest.fixture(scope="module")
def shuffled(dataset):
    return run_shuffled_target(dataset, n_repeats=2, seed=42)


# --------------------------------------------------------------------------- #
def test_normal_shows_real_signal(normal):
    assert normal.regression.mean("R2") > 0.2
    assert normal.classification.mean("roc_auc") > 0.7


def test_shuffled_target_collapses_regression(shuffled):
    # With permuted labels there is nothing to learn: R² must fall to ~baseline.
    assert shuffled.regression.mean("R2") < 0.2


def test_shuffled_target_collapses_classification(shuffled):
    # AUC must fall toward chance (0.5). This is the "no suspicious skill on
    # noise" guard — if this ever fails, suspect leakage before celebrating.
    assert shuffled.classification.mean("roc_auc") < 0.65
    assert not np.isnan(shuffled.classification.mean("roc_auc"))


def test_normal_clearly_beats_shuffled(normal, shuffled):
    auc_gap = normal.classification.mean("roc_auc") - shuffled.classification.mean("roc_auc")
    assert auc_gap > 0.15
    assert normal.classification.mean("f1") > shuffled.classification.mean("f1")


def test_destroying_all_features_collapses(dataset):
    all_groups = ["volume", "participation", "response_latency", "linguistic", "temporal"]
    res = run_feature_destruction(dataset, all_groups, n_repeats=2, seed=42)
    assert res.regression.mean("R2") < 0.2
    assert res.classification.mean("roc_auc") < 0.65


def test_destroying_participation_degrades_performance(dataset, normal):
    res = run_feature_destruction(dataset, ["participation"], n_repeats=2, seed=42)
    # Participation is a core encoded signal; removing it should hurt.
    assert res.classification.mean("roc_auc") < normal.classification.mean("roc_auc")


def test_feature_destruction_rejects_unknown_group(dataset):
    with pytest.raises(ValueError):
        run_feature_destruction(dataset, ["nonsense"], n_repeats=1)
