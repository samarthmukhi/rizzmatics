"""Tests for the Conversational Engagement Index."""

from datetime import datetime, timedelta

import pytest

from src.parser import Message
from src.sessions import detect_sessions
from src.engagement import (
    DEFAULT_WEIGHTS,
    EngagementConfig,
    compute_engagement,
    label_high_engagement,
)


def _msg(dt, sender, text="hi"):
    return Message(timestamp=dt, sender=sender, message=text, is_system_message=False)


def _make_sessions():
    """Three sessions of deliberately different engagement."""
    base = datetime(2026, 8, 12, 12, 0)
    day = timedelta(days=1)

    # Session 0: long, balanced, chatty marathon.
    s0 = []
    for i in range(20):
        s0.append(_msg(base + timedelta(minutes=3 * i), "A" if i % 2 else "B"))

    # Session 1: short, one-sided, two messages from the same person.
    s1 = [
        _msg(base + day, "A"),
        _msg(base + day + timedelta(minutes=1), "A"),
    ]

    # Session 2: medium, balanced.
    s2 = []
    for i in range(8):
        s2.append(_msg(base + 2 * day + timedelta(minutes=5 * i), "A" if i % 2 else "B"))

    return detect_sessions(s0 + s1 + s2, inactivity_hours=6)


# --------------------------------------------------------------------------- #
def test_engagement_index_in_unit_range():
    sessions = _make_sessions()
    df = compute_engagement(sessions)
    assert (df["engagement_index"] >= 0).all()
    assert (df["engagement_index"] <= 1).all()


def test_marathon_beats_one_sided_pair():
    sessions = _make_sessions()
    df = compute_engagement(sessions)
    # Session 0 (long, balanced) should out-engage session 1 (short, one-sided).
    assert df.loc[0, "engagement_index"] > df.loc[1, "engagement_index"]


def test_components_present():
    sessions = _make_sessions()
    df = compute_engagement(sessions)
    for comp in ("duration", "volume", "bidirectional", "balance", "persistence"):
        assert comp in df.columns
        assert f"{comp}_norm" in df.columns


def test_weights_are_normalized_internally():
    # Doubling all weights must not change the resulting index.
    sessions = _make_sessions()
    a = compute_engagement(sessions, EngagementConfig(weights=dict(DEFAULT_WEIGHTS)))
    doubled = {k: v * 2 for k, v in DEFAULT_WEIGHTS.items()}
    b = compute_engagement(sessions, EngagementConfig(weights=doubled))
    assert a["engagement_index"].round(9).equals(b["engagement_index"].round(9))


def test_custom_weights_change_ranking_sensibly():
    sessions = _make_sessions()
    only_volume = compute_engagement(
        sessions,
        EngagementConfig(weights={"duration": 0, "volume": 1, "bidirectional": 0,
                                  "balance": 0, "persistence": 0}),
    )
    # Session 0 has the most messages -> highest volume-only engagement.
    assert only_volume["engagement_index"].idxmax() == 0


def test_zero_weight_sum_raises():
    with pytest.raises(ValueError):
        EngagementConfig(weights={c: 0 for c in DEFAULT_WEIGHTS}).normalized_weights()


def test_single_session_is_neutral_not_crash():
    base = datetime(2026, 8, 12, 12, 0)
    msgs = [_msg(base, "A"), _msg(base + timedelta(minutes=2), "B")]
    df = compute_engagement(detect_sessions(msgs))
    assert len(df) == 1
    assert 0.0 <= df.loc[0, "engagement_index"] <= 1.0


def test_empty_sessions_returns_empty_frame():
    assert compute_engagement([]).empty


# --------------------------------------------------------------------------- #
# Classification labels
# --------------------------------------------------------------------------- #
def test_label_high_engagement_threshold():
    sessions = _make_sessions()
    df = compute_engagement(sessions)
    labels, threshold = label_high_engagement(df["engagement_index"], percentile=75)
    assert set(labels.unique()).issubset({0, 1})
    # Values at/above threshold are labeled 1.
    assert ((df["engagement_index"] >= threshold) == (labels == 1)).all()


def test_label_high_engagement_marks_top_session():
    sessions = _make_sessions()
    df = compute_engagement(sessions)
    labels, _ = label_high_engagement(df["engagement_index"], percentile=75)
    assert labels.loc[df["engagement_index"].idxmax()] == 1
