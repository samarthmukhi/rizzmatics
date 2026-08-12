"""Phase 16 — synthetic generator audit tests.

The important test here (`test_metadata_predictions_match_ablation`) checks that
the generator's own claims about which feature families are informative actually
match how the model behaves. The metadata is a falsifiable prediction, and this
test is what falsifies it.
"""

import random

import pytest

from scripts.generate_demo_data import (
    GENERATOR_METADATA,
    _corrupt,
    generate_export,
)
from src.features import FEATURE_GROUPS
from src.models import make_classifiers
from src.parser import parse_chat, parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions
from src.research.experiment import cross_val_metrics, subset_feature_groups


# --------------------------------------------------------------------------- #
# Metadata structure.
# --------------------------------------------------------------------------- #
def test_metadata_covers_every_feature_group():
    documented = set(GENERATOR_METADATA["feature_dependencies"])
    assert documented == set(FEATURE_GROUPS)


def test_metadata_records_circularity_and_latent_variable():
    assert GENERATOR_METADATA["latent_variable"] == "archetype"
    assert "does not establish generalization" in GENERATOR_METADATA["circularity"]


# --------------------------------------------------------------------------- #
# The audit's central claim: metadata predictions match model behavior.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dataset():
    return build_dataset(detect_sessions(parse_file("data/demo/demo_chat.txt")), prefix=10)


def _group_auc(dataset, group):
    X = subset_feature_groups(dataset.X, [group])
    cv = cross_val_metrics(make_classifiers(42)["Random Forest"], X,
                           dataset.y_classification, task="classification",
                           n_splits=5, n_repeats=2, random_state=42)
    return cv.mean("roc_auc")


def test_metadata_predictions_match_ablation(dataset):
    deps = GENERATOR_METADATA["feature_dependencies"]
    informative = [g for g, d in deps.items() if d["expected_informative"]]
    uninformative = [g for g, d in deps.items() if not d["expected_informative"]]

    inf_auc = {g: _group_auc(dataset, g) for g in informative}
    uninf_auc = {g: _group_auc(dataset, g) for g in uninformative}

    # Every family the metadata calls informative must out-predict every family
    # it calls uninformative. If not, the generator audit is wrong.
    assert min(inf_auc.values()) > max(uninf_auc.values()), (inf_auc, uninf_auc)


# --------------------------------------------------------------------------- #
# Noisy mode degrades signal (but is not the null control).
# --------------------------------------------------------------------------- #
def test_noisy_dataset_is_weaker_than_structured(dataset):
    noisy = build_dataset(
        detect_sessions(parse_file("data/demo/demo_chat_noisy.txt")), prefix=10)
    struct_auc = cross_val_metrics(
        make_classifiers(42)["Random Forest"], dataset.X, dataset.y_classification,
        task="classification", n_repeats=2, random_state=42).mean("roc_auc")
    noisy_auc = cross_val_metrics(
        make_classifiers(42)["Random Forest"], noisy.X, noisy.y_classification,
        task="classification", n_repeats=2, random_state=42).mean("roc_auc")
    assert struct_auc > noisy_auc


# --------------------------------------------------------------------------- #
# Generator mechanics.
# --------------------------------------------------------------------------- #
def test_generate_export_rejects_bad_mode():
    with pytest.raises(ValueError):
        generate_export(mode="chaotic")


def test_generate_export_is_deterministic():
    a = generate_export(seed=7, days=20, mode="structured")
    b = generate_export(seed=7, days=20, mode="structured")
    assert a == b


def test_generated_export_parses_cleanly():
    text = generate_export(seed=1, days=15)
    msgs = parse_chat(text)
    assert len(msgs) > 0
    assert any(not m.is_system_message for m in msgs)


def test_corrupt_changes_structure():
    rng = random.Random(0)
    senders = ["A", "B", "A", "B", "A"]
    latencies = [10, 10, 10, 10, 10]
    s2, l2 = _corrupt(rng, senders, latencies)
    assert len(s2) == len(senders) and len(l2) == len(latencies)
    assert l2 != latencies  # latencies redrawn from a wide distribution
