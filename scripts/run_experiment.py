#!/usr/bin/env python3
"""Reproducible experiment runner.

Runs a named research experiment with a fixed seed and records the result to
``experiments/results/<id>.json``. The same command reproduces the same metrics.

Usage:
    python scripts/run_experiment.py --experiment baseline
    python scripts/run_experiment.py --experiment all
    python scripts/run_experiment.py --experiment prefix --repeats 5 --seed 42

Experiments: baseline, shuffled, null, ablation, prefix, robustness, all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.models import make_classifiers  # noqa: E402
from src.parser import parse_file  # noqa: E402
from src.preprocessing import build_dataset  # noqa: E402
from src.sessions import detect_sessions  # noqa: E402
from src.research.ablation import run_ablation  # noqa: E402
from src.research.nulldata import make_null_dataset  # noqa: E402
from src.research.prefix import run_prefix_sweep  # noqa: E402
from src.research.registry import (  # noqa: E402
    cvmetrics_to_dict,
    dataset_version,
    make_record,
    save_experiment,
)
from src.research.robustness import (  # noqa: E402
    leave_one_group_out,
    sensitivity_class_threshold,
    sensitivity_engagement_weights,
    sensitivity_inactivity,
)
from src.research.sanity import run_normal, run_shuffled_target  # noqa: E402

DEFAULT_DATA = _REPO_ROOT / "data" / "demo" / "demo_chat.txt"
EXPERIMENTS = ["baseline", "shuffled", "null", "ablation", "prefix", "robustness"]

_RF_HYPERPARAMS = {"n_estimators": 300, "random_state": None}  # seed filled per run


def _methodology(n_splits: int, n_repeats: int) -> dict:
    return {
        "cv": "RepeatedStratifiedKFold (clf) / RepeatedKFold (reg)",
        "n_splits": n_splits, "n_repeats": n_repeats,
        "preprocessing": "median-impute (+scale for linear) inside pipeline, refit per fold",
        "leakage_control": "features from message prefix; target from full session",
    }


def _pair(name: str, sanity_result) -> dict:
    return {
        "condition": name,
        "regression": cvmetrics_to_dict(sanity_result.regression),
        "classification": cvmetrics_to_dict(sanity_result.classification),
    }


def _row(name: str, reg_cv, clf_cv) -> dict:
    return {
        "condition": name,
        "regression": cvmetrics_to_dict(reg_cv),
        "classification": cvmetrics_to_dict(clf_cv),
    }


def run(name: str, *, data: Path, prefix: int, seed: int, repeats: int,
        out_dir: Path | None = None) -> Path:
    """Run one named experiment and save its record. Returns the saved path."""
    ds_version = dataset_version(data)
    hyper = {**_RF_HYPERPARAMS, "random_state": seed}
    method = _methodology(5, repeats)
    common = dict(seed=seed, dataset=ds_version, prefix=prefix,
                  model="RandomForest", hyperparameters=hyper, methodology=method)

    sessions = detect_sessions(parse_file(data))

    if name == "baseline":
        ds = build_dataset(sessions, prefix=prefix)
        res = [_pair("normal", run_normal(ds, n_repeats=repeats, seed=seed))]
        rec = make_record("baseline", feature_groups=["all"], results=res,
                          n_samples=len(ds), **common)

    elif name == "shuffled":
        ds = build_dataset(sessions, prefix=prefix)
        res = [
            _pair("normal", run_normal(ds, n_repeats=repeats, seed=seed)),
            _pair("shuffled_target", run_shuffled_target(ds, n_repeats=repeats, seed=seed)),
        ]
        rec = make_record("shuffled", feature_groups=["all"], results=res,
                          n_samples=len(ds), **common)

    elif name == "null":
        nd = make_null_dataset(n_samples=120, seed=seed)
        res = [_pair("null_control", run_normal(nd, n_repeats=repeats, seed=seed))]
        rec = make_record("null", feature_groups=["random_noise"], results=res,
                          n_samples=len(nd), **{**common, "dataset": {"name": "null_control", "sha256_12": None, "n_bytes": 0}})

    elif name == "ablation":
        ds = build_dataset(sessions, prefix=prefix)
        rows = run_ablation(ds, n_repeats=repeats, seed=seed)
        res = [_row(r.condition, r.regression, r.classification) for r in rows]
        rec = make_record("ablation", feature_groups=["per-condition"], results=res,
                          n_samples=len(ds), **common)

    elif name == "prefix":
        rows = run_prefix_sweep(sessions, n_repeats=repeats, seed=seed)
        res = [{"condition": r.label, "window": r.spec, "n_samples": r.n_samples,
                "note": r.note,
                "regression": cvmetrics_to_dict(r.regression),
                "classification": cvmetrics_to_dict(r.classification)} for r in rows]
        rec = make_record("prefix", feature_groups=["all"], results=res,
                          n_samples=max((r.n_samples for r in rows), default=0), **common)

    elif name == "robustness":
        ds = build_dataset(sessions, prefix=prefix)
        rows = leave_one_group_out(ds, n_repeats=repeats, seed=seed)
        res = [_row(r.condition, r.regression, r.classification) for r in rows]
        for r in sensitivity_inactivity(parse_file(data), n_repeats=repeats, seed=seed):
            res.append({**_row(r.condition, r.regression, r.classification), "extra": r.extra})
        for r in sensitivity_engagement_weights(sessions, n_repeats=repeats, seed=seed):
            res.append(_row(r.condition, r.regression, r.classification))
        for r in sensitivity_class_threshold(ds, n_repeats=repeats, seed=seed):
            res.append({"condition": r.condition,
                        "classification": cvmetrics_to_dict(r.classification),
                        "extra": r.extra})
        rec = make_record("robustness", feature_groups=["various"], results=res,
                          n_samples=len(ds), **common)

    else:
        raise ValueError(f"Unknown experiment {name!r}. Choose from {EXPERIMENTS} or 'all'.")

    path = save_experiment(rec, out_dir) if out_dir else save_experiment(rec)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Rizzmatics research experiment.")
    parser.add_argument("--experiment", "-e", default="baseline",
                        help=f"One of: {', '.join(EXPERIMENTS)}, all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prefix", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    targets = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    for name in targets:
        path = run(name, data=args.data, prefix=args.prefix, seed=args.seed,
                   repeats=args.repeats)
        print(f"✓ {name:12s} -> {path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
