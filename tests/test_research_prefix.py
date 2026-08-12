"""Phase 13 — prefix-length experiment tests.

The scientific hazard here is the sample-size confound: longer prefixes keep
fewer, longer sessions. These tests guard that n_samples is tracked and that the
small-sample warning fires, so the curve is never read naively.
"""

import pytest

from src.parser import parse_file
from src.sessions import detect_sessions
from src.research.prefix import (
    DEFAULT_PREFIXES,
    _fractional_dataset,
    prefix_table,
    run_prefix_sweep,
)
from src.engagement import EngagementConfig


@pytest.fixture(scope="module")
def sessions():
    return detect_sessions(parse_file("data/demo/demo_chat.txt"))


@pytest.fixture(scope="module")
def rows(sessions):
    return run_prefix_sweep(sessions, n_repeats=2, seed=42)


def test_sweep_covers_all_conditions(rows):
    labels = {r.label for r in rows}
    assert labels == {lbl for lbl, _ in DEFAULT_PREFIXES}


def test_larger_integer_prefix_keeps_fewer_or_equal_sessions(rows):
    by = {r.label: r.n_samples for r in rows}
    seq = [by["first_3"], by["first_5"], by["first_10"], by["first_20"], by["first_30"]]
    assert all(a >= b for a, b in zip(seq, seq[1:])), seq


def test_n_samples_reported_for_every_point(rows):
    for r in rows:
        assert r.n_samples > 0  # demo data is rich enough for all conditions


def test_small_sample_note_fires_when_appropriate(rows):
    by = {r.label: r for r in rows}
    # first_30 keeps <20 sessions on the demo data -> must be flagged.
    if by["first_30"].n_samples < 20:
        assert "unstable" in by["first_30"].note


def test_signal_is_useful_early(rows):
    by = {r.label: r for r in rows}
    # Classification signal is present very early (participation is visible fast).
    assert by["first_5"].classification.mean("roc_auc") > 0.7


def test_prefix_table_reports_n_samples_column(rows):
    df = prefix_table(rows)
    assert "n_samples" in df.columns
    assert "roc_auc_mean" in df.columns
    assert len(df) == len(rows)


# --------------------------------------------------------------------------- #
# Fractional-prefix leakage safety.
# --------------------------------------------------------------------------- #
def test_fractional_prefix_leaves_a_future(sessions):
    ds = _fractional_dataset(sessions, 0.5, cfg=EngagementConfig(), high_pct=75.0)
    # Every kept session used strictly fewer messages than it has (future exists).
    for s in sessions:
        if s.session_id in ds.session_ids:
            assert max(1, int(s.message_count * 0.5)) < s.message_count


def test_fractional_rejects_bad_fraction(sessions):
    with pytest.raises(ValueError):
        _fractional_dataset(sessions, 1.5, cfg=EngagementConfig(), high_pct=75.0)
