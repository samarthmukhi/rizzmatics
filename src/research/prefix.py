"""Prefix-length experiment — how early does signal become useful?

We rebuild the leakage-safe dataset at several observation windows (first 3, 5,
10, 20, 30 messages, and first 50% of each session) and evaluate each with the
same CV methodology.

Important honesty caveat, reported alongside every curve: a larger prefix keeps
*fewer and different* sessions (only ones long enough to have a future beyond the
prefix survive). So the curve conflates "more context per session" with "a
different, longer-session cohort". ``n_samples`` is therefore reported for every
point and must be read together with the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engagement import EngagementConfig, compute_engagement, label_high_engagement
from ..features import build_feature_frame
from ..models import make_classifiers, make_regressors
from ..preprocessing import Dataset, build_dataset
from ..sessions import Session
from .experiment import CVMetrics, cross_val_metrics

__all__ = ["PrefixRow", "run_prefix_sweep", "prefix_table", "DEFAULT_PREFIXES"]

# (label, spec) where spec is an int (fixed messages) or a float in (0,1) (fraction).
DEFAULT_PREFIXES: list[tuple[str, object]] = [
    ("first_3", 3), ("first_5", 5), ("first_10", 10),
    ("first_20", 20), ("first_30", 30), ("first_50pct", 0.5),
]


@dataclass
class PrefixRow:
    label: str
    spec: str
    n_samples: int
    regression: CVMetrics | None
    classification: CVMetrics | None
    note: str = ""


def _fractional_dataset(sessions: list[Session], fraction: float, *,
                        cfg: EngagementConfig, high_pct: float) -> Dataset:
    """Build a leakage-safe dataset where each session's prefix is a fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    # Keep sessions with a genuine future when we look at only `fraction` of them.
    def cut(s: Session) -> int:
        return max(1, int(s.message_count * fraction))

    keep = [s for s in sessions if s.message_count >= 2 and cut(s) < s.message_count]
    if not keep:
        raise ValueError("No session long enough for a fractional-prefix future.")

    engagement = compute_engagement(sessions, cfg)
    X = build_feature_frame(keep, prefix=cut)
    ids = [s.session_id for s in keep]
    y_reg = engagement.loc[ids, "engagement_index"].copy()
    y_reg.name = "engagement_index"
    y_clf, thr = label_high_engagement(y_reg, percentile=high_pct)
    return Dataset(X, y_reg, y_clf, thr, prefix=-1, session_ids=ids)


def run_prefix_sweep(
    sessions: list[Session],
    *,
    conditions: list[tuple[str, object]] | None = None,
    engagement_config: EngagementConfig | None = None,
    high_percentile: float = 75.0,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> list[PrefixRow]:
    """Evaluate the model across observation windows."""
    conditions = conditions or DEFAULT_PREFIXES
    cfg = engagement_config or EngagementConfig()
    rows: list[PrefixRow] = []

    for label, spec in conditions:
        try:
            if isinstance(spec, float):
                ds = _fractional_dataset(sessions, spec, cfg=cfg, high_pct=high_percentile)
                spec_str = f"{int(spec * 100)}%"
            else:
                ds = build_dataset(sessions, prefix=int(spec),
                                   high_percentile=high_percentile,
                                   engagement_config=cfg)
                spec_str = f"{spec} msgs"
        except ValueError as exc:
            rows.append(PrefixRow(label, str(spec), 0, None, None, note=str(exc)))
            continue

        reg = cross_val_metrics(make_regressors(seed)["Random Forest"], ds.X,
                                ds.y_regression, task="regression",
                                n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        clf = cross_val_metrics(make_classifiers(seed)["Random Forest"], ds.X,
                                ds.y_classification, task="classification",
                                n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        note = ""
        if len(ds) < 20:
            note = f"Only {len(ds)} sessions — estimate is unstable."
        rows.append(PrefixRow(label, spec_str, len(ds), reg, clf, note))
    return rows


def prefix_table(rows: list[PrefixRow]) -> pd.DataFrame:
    """Tidy table: one row per prefix, with n_samples reported prominently."""
    records = []
    for r in rows:
        rec = {"prefix": r.label, "window": r.spec, "n_samples": r.n_samples}
        if r.regression is not None:
            rec["R2_mean"] = r.regression.mean("R2")
            rec["R2_std"] = r.regression.std("R2")
            rec["MAE_mean"] = r.regression.mean("MAE")
        if r.classification is not None:
            rec["f1_mean"] = r.classification.mean("f1")
            rec["roc_auc_mean"] = r.classification.mean("roc_auc")
            rec["roc_auc_std"] = r.classification.std("roc_auc")
        rec["note"] = r.note
        records.append(rec)
    return pd.DataFrame.from_records(records)
