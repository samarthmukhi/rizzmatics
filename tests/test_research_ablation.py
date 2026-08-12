"""Phase 12 — ablation study tests."""

import pytest

from src.features import ALL_FEATURES, FEATURE_GROUPS
from src.parser import parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions
from src.research.ablation import ablation_table, run_ablation
from src.research.experiment import subset_feature_groups


# --------------------------------------------------------------------------- #
# Feature-group hygiene: the groups must partition ALL_FEATURES exactly.
# --------------------------------------------------------------------------- #
def test_feature_groups_partition_all_features_exactly():
    covered = [f for group in FEATURE_GROUPS.values() for f in group]
    assert sorted(covered) == sorted(ALL_FEATURES)          # complete coverage
    assert len(covered) == len(set(covered))                 # no feature in two groups


def test_subset_isolates_only_named_group():
    import pandas as pd
    X = pd.DataFrame({f: [1.0, 2.0] for f in ALL_FEATURES})
    sub = subset_feature_groups(X, ["participation"])
    assert list(sub.columns) == FEATURE_GROUPS["participation"]


def test_subset_rejects_unknown_group():
    import pandas as pd
    X = pd.DataFrame({f: [1.0] for f in ALL_FEATURES})
    with pytest.raises(ValueError):
        subset_feature_groups(X, ["vibes"])


# --------------------------------------------------------------------------- #
# The study itself.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rows():
    ds = build_dataset(detect_sessions(parse_file("data/demo/demo_chat.txt")), prefix=10)
    return run_ablation(ds, n_repeats=2, seed=42)


def test_ablation_has_all_conditions(rows):
    names = {r.condition for r in rows}
    assert "all_features" in names
    assert "weak_minimal" in names
    for group in FEATURE_GROUPS:
        assert f"{group}_only" in names


def test_single_group_conditions_use_only_that_group(rows):
    by_name = {r.condition: r for r in rows}
    for group, feats in FEATURE_GROUPS.items():
        assert by_name[f"{group}_only"].n_features == len(feats)


def test_a_real_family_beats_the_weak_baseline(rows):
    by_name = {r.condition: r for r in rows}
    weak_auc = by_name["weak_minimal"].classification.mean("roc_auc")
    part_auc = by_name["participation_only"].classification.mean("roc_auc")
    assert part_auc > weak_auc


def test_all_features_beats_weak(rows):
    by_name = {r.condition: r for r in rows}
    assert (by_name["all_features"].classification.mean("roc_auc")
            > by_name["weak_minimal"].classification.mean("roc_auc"))


def test_ablation_table_shape(rows):
    df = ablation_table(rows)
    assert len(df) == len(rows)
    for col in ("R2_mean", "R2_std", "f1_mean", "roc_auc_mean"):
        assert col in df.columns
    disp = ablation_table(rows, formatted=True)
    assert "R2" in disp.columns and "ROC-AUC" in disp.columns
