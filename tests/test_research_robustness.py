"""Phase 14 — robustness tests.

The central honest finding these guard: participation features dominate. Removing
them must visibly hurt, while removing other families must not. We also check the
sensitivity sweeps are wired correctly (thresholds change class balance, etc.).
"""

import pytest

from src.parser import parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions
from src.research.robustness import (
    leave_one_group_out,
    sensitivity_class_threshold,
    sensitivity_inactivity,
    sensitivity_engagement_weights,
)

DEMO = "data/demo/demo_chat.txt"


@pytest.fixture(scope="module")
def messages():
    return parse_file(DEMO)


@pytest.fixture(scope="module")
def dataset(messages):
    return build_dataset(detect_sessions(messages), prefix=10)


@pytest.fixture(scope="module")
def logo(dataset):
    return leave_one_group_out(dataset, n_repeats=2, seed=42)


# --------------------------------------------------------------------------- #
def test_removing_participation_hurts_the_most(logo):
    by = {r.condition: r for r in logo}
    base_auc = by["all_features"].classification.mean("roc_auc")
    part_auc = by["minus_participation"].classification.mean("roc_auc")
    # Participation is the dominant family: dropping it must cause a real drop.
    assert base_auc - part_auc > 0.15
    assert by["minus_participation"].regression.mean("R2") < by["all_features"].regression.mean("R2")


def test_removing_a_noise_family_barely_hurts(logo):
    by = {r.condition: r for r in logo}
    # Linguistic carries ~no signal in the synthetic data; dropping it should
    # not collapse the model the way dropping participation does.
    assert (by["minus_linguistic"].classification.mean("roc_auc")
            > by["minus_participation"].classification.mean("roc_auc"))


def test_logo_has_one_row_per_group_plus_baseline(logo):
    from src.features import FEATURE_GROUPS
    assert len(logo) == len(FEATURE_GROUPS) + 1


def test_inactivity_sensitivity_runs_all_thresholds(messages):
    rows = sensitivity_inactivity(messages, thresholds=(3.0, 6.0, 12.0),
                                  n_repeats=2, seed=42)
    assert {r.condition for r in rows} == {"gap_3.0h", "gap_6.0h", "gap_12.0h"}
    for r in rows:
        assert r.extra["n_sessions"] > 0


def test_class_threshold_changes_class_balance(dataset):
    rows = sensitivity_class_threshold(dataset, percentiles=(65.0, 75.0, 85.0),
                                       n_repeats=2, seed=42)
    n_high = [r.extra["n_high"] for r in rows]
    # Higher percentile -> fewer HIGH-engagement positives.
    assert n_high[0] >= n_high[1] >= n_high[2]


def test_engagement_weight_sensitivity_runs(dataset):
    sessions = detect_sessions(parse_file(DEMO))
    rows = sensitivity_engagement_weights(sessions, n_repeats=2, seed=42)
    conds = {r.condition for r in rows}
    assert "weights_default" in conds
    assert all(r.n_samples > 0 for r in rows)
