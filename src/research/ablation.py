"""Ablation study — which signal families carry predictive information?

We evaluate the model on the full feature set, on each feature family alone, and
on a deliberately weak set (calendar flags that should be near-orthogonal to
engagement). Same CV methodology for every condition, no cherry-picking: the
table reports whatever the data says, including the embarrassing rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features import FEATURE_GROUPS
from ..models import make_classifiers, make_regressors
from ..preprocessing import Dataset
from .experiment import CVMetrics, cross_val_metrics, subset_feature_groups

__all__ = ["AblationRow", "run_ablation", "ablation_table", "WEAK_FEATURES"]

# A deliberately weak baseline: calendar flags with no reason to track the
# conversational archetype. If a real family can't beat this, that's a finding.
WEAK_FEATURES = ["is_weekend", "is_late_night"]


@dataclass
class AblationRow:
    """One ablation condition's paired regression + classification result."""

    condition: str
    n_features: int
    regression: CVMetrics
    classification: CVMetrics


def _eval(X, ds: Dataset, *, n_splits, n_repeats, seed) -> tuple[CVMetrics, CVMetrics]:
    reg = cross_val_metrics(
        make_regressors(seed)["Random Forest"], X, ds.y_regression,
        task="regression", n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    clf = cross_val_metrics(
        make_classifiers(seed)["Random Forest"], X, ds.y_classification,
        task="classification", n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return reg, clf


def run_ablation(ds: Dataset, *, n_splits: int = 5, n_repeats: int = 5,
                 seed: int = 42) -> list[AblationRow]:
    """Run the full ablation and return one row per condition."""
    conditions: list[tuple[str, pd.DataFrame]] = [("all_features", ds.X)]
    for group in FEATURE_GROUPS:
        conditions.append((f"{group}_only", subset_feature_groups(ds.X, [group])))
    weak_cols = [c for c in WEAK_FEATURES if c in ds.X.columns]
    conditions.append(("weak_minimal", ds.X[weak_cols]))

    rows: list[AblationRow] = []
    for name, X in conditions:
        reg, clf = _eval(X, ds, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
        rows.append(AblationRow(name, X.shape[1], reg, clf))
    return rows


def ablation_table(rows: list[AblationRow], *, formatted: bool = False) -> pd.DataFrame:
    """Build the comparison table.

    With ``formatted=False`` (default) returns numeric mean/std columns for
    machine consumption. With ``formatted=True`` returns display strings
    ("0.53 ± 0.35") for humans.
    """
    records = []
    for r in rows:
        rec = {"condition": r.condition, "n_features": r.n_features}
        for m in ("MAE", "RMSE", "R2"):
            rec[f"{m}_mean"] = r.regression.mean(m)
            rec[f"{m}_std"] = r.regression.std(m)
        for m in ("f1", "roc_auc"):
            rec[f"{m}_mean"] = r.classification.mean(m)
            rec[f"{m}_std"] = r.classification.std(m)
        records.append(rec)
    df = pd.DataFrame.from_records(records)

    if not formatted:
        return df

    disp = pd.DataFrame({"condition": df["condition"], "n_features": df["n_features"]})
    disp["MAE"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df["MAE_mean"], df["MAE_std"])]
    disp["RMSE"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df["RMSE_mean"], df["RMSE_std"])]
    disp["R2"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df["R2_mean"], df["R2_std"])]
    disp["F1"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df["f1_mean"], df["f1_std"])]
    disp["ROC-AUC"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df["roc_auc_mean"], df["roc_auc_std"])]
    return disp
