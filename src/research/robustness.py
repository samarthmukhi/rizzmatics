"""Robustness tests — is the result stable, or is one knob doing all the work?

Two families of check:

* **Leave-one-group-out** — remove a signal family and keep the rest. A large
  drop means that family is critical; a negligible drop means it was redundant.
  This is how we find out whether a single engineered variable carries the result.
* **Sensitivity** — vary the pipeline's free parameters (session inactivity
  threshold, engagement-index weights, classification percentile) and see whether
  the headline moves. A result that only holds at one specific setting is fragile,
  and we say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engagement import DEFAULT_WEIGHTS, EngagementConfig, label_high_engagement
from ..features import FEATURE_GROUPS
from ..models import make_classifiers, make_regressors
from ..parser import Message
from ..preprocessing import Dataset, build_dataset
from ..sessions import detect_sessions
from .experiment import CVMetrics, cross_val_metrics

__all__ = [
    "RobustnessRow",
    "leave_one_group_out",
    "sensitivity_inactivity",
    "sensitivity_engagement_weights",
    "sensitivity_class_threshold",
    "robustness_table",
]


@dataclass
class RobustnessRow:
    condition: str
    n_samples: int
    regression: CVMetrics | None
    classification: CVMetrics | None
    extra: dict = None


def _reg(X, y, seed, n_splits, n_repeats):
    return cross_val_metrics(make_regressors(seed)["Random Forest"], X, y,
                             task="regression", n_splits=n_splits,
                             n_repeats=n_repeats, random_state=seed)


def _clf(X, y, seed, n_splits, n_repeats):
    return cross_val_metrics(make_classifiers(seed)["Random Forest"], X, y,
                             task="classification", n_splits=n_splits,
                             n_repeats=n_repeats, random_state=seed)


def leave_one_group_out(ds: Dataset, *, n_splits: int = 5, n_repeats: int = 5,
                        seed: int = 42) -> list[RobustnessRow]:
    """Baseline (all features) plus one row per feature-family removed."""
    rows = [RobustnessRow(
        "all_features", len(ds),
        _reg(ds.X, ds.y_regression, seed, n_splits, n_repeats),
        _clf(ds.X, ds.y_classification, seed, n_splits, n_repeats),
    )]
    for group, cols in FEATURE_GROUPS.items():
        drop = [c for c in cols if c in ds.X.columns]
        X = ds.X.drop(columns=drop)
        rows.append(RobustnessRow(
            f"minus_{group}", len(ds),
            _reg(X, ds.y_regression, seed, n_splits, n_repeats),
            _clf(X, ds.y_classification, seed, n_splits, n_repeats),
        ))
    return rows


def sensitivity_inactivity(messages: list[Message], *, thresholds=(3.0, 6.0, 12.0),
                           prefix: int = 10, high_percentile: float = 75.0,
                           n_splits: int = 5, n_repeats: int = 5,
                           seed: int = 42) -> list[RobustnessRow]:
    """Rebuild sessions at different inactivity gaps and re-evaluate."""
    rows: list[RobustnessRow] = []
    for hours in thresholds:
        sessions = detect_sessions(messages, inactivity_hours=hours)
        try:
            ds = build_dataset(sessions, prefix=prefix, high_percentile=high_percentile)
        except ValueError as exc:
            rows.append(RobustnessRow(f"gap_{hours}h", 0, None, None, {"error": str(exc)}))
            continue
        rows.append(RobustnessRow(
            f"gap_{hours}h", len(ds),
            _reg(ds.X, ds.y_regression, seed, n_splits, n_repeats),
            _clf(ds.X, ds.y_classification, seed, n_splits, n_repeats),
            {"n_sessions": len(sessions)},
        ))
    return rows


def sensitivity_engagement_weights(sessions, *, weight_sets: dict | None = None,
                                   prefix: int = 10, high_percentile: float = 75.0,
                                   n_splits: int = 5, n_repeats: int = 5,
                                   seed: int = 42) -> list[RobustnessRow]:
    """Vary the engagement-index weights (which redefine the target)."""
    weight_sets = weight_sets or {
        "default": dict(DEFAULT_WEIGHTS),
        "volume_heavy": {"duration": 0.1, "volume": 0.6, "bidirectional": 0.1,
                         "balance": 0.1, "persistence": 0.1},
        "balance_heavy": {"duration": 0.1, "volume": 0.1, "bidirectional": 0.3,
                          "balance": 0.4, "persistence": 0.1},
    }
    rows: list[RobustnessRow] = []
    for name, weights in weight_sets.items():
        cfg = EngagementConfig(weights=weights)
        ds = build_dataset(sessions, prefix=prefix, high_percentile=high_percentile,
                           engagement_config=cfg)
        rows.append(RobustnessRow(
            f"weights_{name}", len(ds),
            _reg(ds.X, ds.y_regression, seed, n_splits, n_repeats),
            _clf(ds.X, ds.y_classification, seed, n_splits, n_repeats),
        ))
    return rows


def sensitivity_class_threshold(ds: Dataset, *, percentiles=(65.0, 75.0, 85.0),
                                n_splits: int = 5, n_repeats: int = 5,
                                seed: int = 42) -> list[RobustnessRow]:
    """Relabel HIGH/LOW at different percentiles; X and regression are unchanged."""
    rows: list[RobustnessRow] = []
    for pct in percentiles:
        y_clf, thr = label_high_engagement(ds.y_regression, percentile=pct)
        pos = int(y_clf.sum())
        rows.append(RobustnessRow(
            f"threshold_p{int(pct)}", len(ds), None,
            _clf(ds.X, y_clf, seed, n_splits, n_repeats),
            {"threshold": thr, "n_high": pos, "n_low": len(ds) - pos},
        ))
    return rows


def robustness_table(rows: list[RobustnessRow]) -> pd.DataFrame:
    """Tidy table with R²/F1/AUC means where available."""
    records = []
    for r in rows:
        rec = {"condition": r.condition, "n_samples": r.n_samples}
        if r.regression is not None:
            rec["R2_mean"] = r.regression.mean("R2")
            rec["R2_std"] = r.regression.std("R2")
        if r.classification is not None:
            rec["f1_mean"] = r.classification.mean("f1")
            rec["roc_auc_mean"] = r.classification.mean("roc_auc")
            rec["roc_auc_std"] = r.classification.std("roc_auc")
        if r.extra:
            rec.update(r.extra)
        records.append(rec)
    return pd.DataFrame.from_records(records)
