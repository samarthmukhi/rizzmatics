"""Phase 18 — experiment registry tests."""

from pathlib import Path

import pytest

from scripts.run_experiment import run
from src.research.registry import (
    dataset_version,
    load_experiments,
    make_record,
    save_experiment,
)

DEMO = Path("data/demo/demo_chat.txt")


# --------------------------------------------------------------------------- #
# Registry primitives.
# --------------------------------------------------------------------------- #
def test_dataset_version_is_deterministic():
    a = dataset_version(DEMO)
    b = dataset_version(DEMO)
    assert a == b
    assert a["sha256_12"] and len(a["sha256_12"]) == 12


def test_save_and_load_roundtrip(tmp_path):
    rec = make_record(
        "unit_test", seed=1, dataset={"name": "x", "sha256_12": "abc", "n_bytes": 1},
        prefix=10, feature_groups=["all"], model="RandomForest",
        hyperparameters={"n_estimators": 300}, methodology={"cv": "kfold"},
        results=[{"condition": "normal"}], n_samples=42,
    )
    path = save_experiment(rec, tmp_path)
    assert path.exists()
    assert (tmp_path / "index.csv").exists()
    loaded = load_experiments(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["experiment_id"] == "unit_test"
    assert loaded[0]["n_samples"] == 42


def test_record_has_required_schema(tmp_path):
    rec = make_record(
        "schema", seed=7, dataset=dataset_version(DEMO), prefix=10,
        feature_groups=["all"], model="RandomForest", hyperparameters={},
        methodology={}, results=[], n_samples=0,
    )
    for key in ("experiment_id", "timestamp_utc", "seed", "dataset", "prefix",
                "feature_groups", "model", "hyperparameters", "methodology",
                "n_samples", "results"):
        assert key in rec


# --------------------------------------------------------------------------- #
# Reproducibility: same command, same metrics (timestamps aside).
# --------------------------------------------------------------------------- #
def test_experiment_is_reproducible(tmp_path):
    import json

    p1 = run("baseline", data=DEMO, prefix=10, seed=42, repeats=2, out_dir=tmp_path / "a")
    p2 = run("baseline", data=DEMO, prefix=10, seed=42, repeats=2, out_dir=tmp_path / "b")
    r1 = json.loads(p1.read_text())
    r2 = json.loads(p2.read_text())
    # Metrics must match exactly; only the timestamp may differ.
    assert r1["results"] == r2["results"]
    assert r1["n_samples"] == r2["n_samples"]
    assert r1["dataset"]["sha256_12"] == r2["dataset"]["sha256_12"]


def test_null_experiment_records_chance_performance(tmp_path):
    import json

    p = run("null", data=DEMO, prefix=10, seed=42, repeats=2, out_dir=tmp_path)
    rec = json.loads(p.read_text())
    auc = rec["results"][0]["classification"]["metrics"]["roc_auc"]["mean"]
    assert 0.30 < auc < 0.65  # the null control, faithfully recorded
