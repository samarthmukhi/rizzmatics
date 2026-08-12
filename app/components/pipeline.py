"""Cached pipeline glue for the dashboard.

Everything expensive happens here, once, behind Streamlit's cache. Crucially,
this module only ever receives *raw text* and parameters — the raw chat is
parsed in-process and never written anywhere. Local by construction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make ``src`` importable whether the app is launched from the repo root or
# from inside ``app/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from src.engagement import EngagementConfig, compute_engagement  # noqa: E402
from src.evaluation import (  # noqa: E402
    EvaluationReport,
    evaluate_classifiers,
    evaluate_regressors,
    prediction_drivers,
)
from src.features import FEATURE_NAMES, build_feature_frame  # noqa: E402
from src.models import make_classifiers, make_regressors  # noqa: E402
from src.parser import parse_chat  # noqa: E402
from src.preprocessing import Dataset, build_dataset  # noqa: E402
from src.sessions import detect_sessions  # noqa: E402

DEMO_PATH = _REPO_ROOT / "data" / "demo" / "demo_chat.txt"


@dataclass
class RizzBundle:
    """The full computed state the dashboard renders."""

    messages: list
    sessions: list
    engagement: "object"          # DataFrame (full-session index + components)
    features_full: "object"       # DataFrame (features from full sessions)
    dataset: Dataset | None
    reg_report: EvaluationReport | None
    clf_report: EvaluationReport | None
    drivers: list
    error: str | None = None      # honest failure message, if any

    @property
    def n_messages(self) -> int:
        return len(self.messages)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


@st.cache_data(show_spinner=False)
def load_demo_text() -> str:
    """Load the bundled synthetic demo export."""
    if DEMO_PATH.exists():
        return DEMO_PATH.read_text(encoding="utf-8")
    return ""


@st.cache_data(show_spinner="Consulting predictive infrastructure...")
def run_pipeline(
    text: str,
    *,
    inactivity_hours: float,
    prefix: int,
    weights: tuple,          # (duration, volume, bidirectional, balance, persistence)
    high_percentile: float,
    dayfirst: bool | None,
) -> RizzBundle:
    """Run the whole pipeline on raw text and return a cached bundle.

    ``weights`` is passed as a tuple so the cache key is hashable.
    """
    messages = parse_chat(text, dayfirst=dayfirst)
    sessions = detect_sessions(messages, inactivity_hours=inactivity_hours)

    cfg = EngagementConfig(weights={
        "duration": weights[0], "volume": weights[1],
        "bidirectional": weights[2], "balance": weights[3],
        "persistence": weights[4],
    })
    engagement = compute_engagement(sessions, cfg)
    features_full = build_feature_frame(sessions) if sessions else _empty_frame()

    # Attach engagement to full-session features for the explorer/analytics.
    if not engagement.empty:
        features_full = features_full.join(engagement[["engagement_index"]])

    dataset = reg_report = clf_report = None
    drivers: list = []
    error = None
    try:
        dataset = build_dataset(
            sessions, prefix=prefix, high_percentile=high_percentile,
            engagement_config=cfg,
        )
        reg_report = evaluate_regressors(dataset.X, dataset.y_regression)
        clf_report = evaluate_classifiers(dataset.X, dataset.y_classification)
        # Explain the best regressor's drivers on the full dataset.
        best_name = reg_report.best_model
        drivers = prediction_drivers(
            make_regressors()[best_name], dataset.X, dataset.y_regression,
            task="regression", top_k=8,
        )
    except ValueError as exc:
        error = str(exc)

    return RizzBundle(
        messages=messages, sessions=sessions, engagement=engagement,
        features_full=features_full, dataset=dataset, reg_report=reg_report,
        clf_report=clf_report, drivers=drivers, error=error,
    )


def _empty_frame():
    import pandas as pd
    return pd.DataFrame(columns=FEATURE_NAMES)


@st.cache_data(show_spinner="Running the full research battery (~30s, real cross-validation)...")
def run_research(
    text: str,
    *,
    inactivity_hours: float,
    prefix: int,
    weights: tuple,
    high_percentile: float,
    dayfirst: bool | None,
    n_repeats: int = 3,
) -> dict:
    """Run the scientific-validation battery for the Research Lab page (cached).

    Returns plain DataFrames/dicts so the result is cache-friendly. Rebuilds the
    dataset from the same inputs as run_pipeline; kept separate so the expensive
    cross-validation only runs when the Research Lab is actually opened.
    """
    import numpy as np
    import pandas as pd

    from src.research.ablation import ablation_table, run_ablation
    from src.research.experiment import small_data_caveat
    from src.research.nulldata import make_null_dataset
    from src.research.prefix import prefix_table, run_prefix_sweep
    from src.research.robustness import leave_one_group_out, robustness_table
    from src.research.sanity import run_normal, run_shuffled_target

    messages = parse_chat(text, dayfirst=dayfirst)
    sessions = detect_sessions(messages, inactivity_hours=inactivity_hours)
    cfg = EngagementConfig(weights={
        "duration": weights[0], "volume": weights[1], "bidirectional": weights[2],
        "balance": weights[3], "persistence": weights[4]})

    try:
        ds = build_dataset(sessions, prefix=prefix, high_percentile=high_percentile,
                           engagement_config=cfg)
    except ValueError as exc:
        return {"error": str(exc)}

    # Sanity: normal / shuffled / null.
    normal = run_normal(ds, n_repeats=n_repeats)
    shuffled = run_shuffled_target(ds, n_repeats=n_repeats)
    null = run_normal(make_null_dataset(n_samples=120, seed=0), n_repeats=n_repeats)
    sanity = pd.DataFrame([
        {"condition": "Normal (real target)", "R2": normal.regression.mean("R2"),
         "F1": normal.classification.mean("f1"), "ROC_AUC": normal.classification.mean("roc_auc")},
        {"condition": "Shuffled target", "R2": shuffled.regression.mean("R2"),
         "F1": shuffled.classification.mean("f1"), "ROC_AUC": shuffled.classification.mean("roc_auc")},
        {"condition": "Null control (noise)", "R2": null.regression.mean("R2"),
         "F1": null.classification.mean("f1"), "ROC_AUC": null.classification.mean("roc_auc")},
    ])

    ablation_rows = run_ablation(ds, n_repeats=n_repeats)
    ablation_num = ablation_table(ablation_rows)
    ablation_disp = ablation_table(ablation_rows, formatted=True)

    prefix_rows = run_prefix_sweep(sessions, n_repeats=n_repeats)
    prefix_num = prefix_table(prefix_rows)

    logo_rows = leave_one_group_out(ds, n_repeats=n_repeats)
    logo_num = robustness_table(logo_rows)

    return {
        "n_samples": len(ds),
        "caveat": small_data_caveat(len(ds)),
        "sanity": sanity,
        "ablation_num": ablation_num,
        "ablation_disp": ablation_disp,
        "prefix_num": prefix_num,
        "logo_num": logo_num,
    }


def session_prefix_features(bundle: RizzBundle, session_id: int, prefix: int) -> dict:
    """Extract the prefix features for a single session (for the Oracle page)."""
    from src.features import extract_features

    session = next(s for s in bundle.sessions if s.session_id == session_id)
    return extract_features(session.messages[:prefix])


def session_full_features(bundle: RizzBundle, session_id: int) -> dict:
    """Extract full-session features (for the Rizz Coefficient readout)."""
    from src.features import extract_features

    session = next(s for s in bundle.sessions if s.session_id == session_id)
    return extract_features(session.messages)
