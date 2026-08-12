"""Leakage prevention tests — the non-negotiable ones.

The central promise of Rizzmatics is: we never use information from the future
to predict the future. These tests make that promise executable. If any of them
fail, the science is broken and every downstream number is a lie.
"""

from datetime import datetime, timedelta

import pytest

from src.parser import Message
from src.sessions import detect_sessions
from src.features import build_feature_frame, FEATURE_NAMES
from src.engagement import compute_engagement
from src.preprocessing import build_dataset


def _msg(minute, sender, text="hi"):
    return Message(
        timestamp=datetime(2026, 8, 12, 12, 0) + timedelta(minutes=minute),
        sender=sender,
        message=text,
        is_system_message=False,
    )


def _long_session(tail_fast: bool):
    """A 10-message session with an identical 5-message prefix.

    The last 5 messages differ between variants (spacing + senders), so the
    full-session engagement differs, but the first-5-message prefix is byte-for
    -byte identical.
    """
    prefix = [_msg(i, "A" if i % 2 else "B", f"m{i}") for i in range(5)]
    if tail_fast:
        tail = [_msg(5 + i, "A" if i % 2 else "B", f"t{i}") for i in range(5)]
    else:
        # Same senders/texts but spread over hours (still one session < 6h gap),
        # which changes duration-based engagement without touching the prefix.
        tail = [_msg(5 + i * 30, "A" if i % 2 else "B", f"t{i}") for i in range(5)]
    return prefix + tail


# --------------------------------------------------------------------------- #
# 1. Prefix features ignore everything after the prefix.
# --------------------------------------------------------------------------- #
def test_prefix_features_invariant_to_future_messages():
    fast = detect_sessions(_long_session(tail_fast=True), inactivity_hours=6)
    slow = detect_sessions(_long_session(tail_fast=False), inactivity_hours=6)

    X_fast = build_feature_frame(fast, prefix=5)
    X_slow = build_feature_frame(slow, prefix=5)

    # Same prefix -> identical features, no matter what happened afterward.
    for col in FEATURE_NAMES:
        assert X_fast.iloc[0][col] == pytest.approx(X_slow.iloc[0][col], nan_ok=True), col


def test_future_messages_do_change_full_engagement():
    """Sanity check the mirror image: the future DOES change the target."""
    fast = detect_sessions(_long_session(tail_fast=True), inactivity_hours=6)
    slow = detect_sessions(_long_session(tail_fast=False), inactivity_hours=6)
    # Duration differs, so raw duration component differs between the variants.
    eng_fast = compute_engagement(fast)
    eng_slow = compute_engagement(slow)
    assert eng_fast.loc[0, "duration"] != eng_slow.loc[0, "duration"]


# --------------------------------------------------------------------------- #
# 2. The target must never appear inside the feature matrix.
# --------------------------------------------------------------------------- #
def test_engagement_not_a_feature_column():
    sessions = detect_sessions(_long_session(tail_fast=True), inactivity_hours=6)
    X = build_feature_frame(sessions, prefix=5)
    leaky = {"engagement_index", "engagement", "high_engagement", "target", "y"}
    assert leaky.isdisjoint(set(X.columns))


def test_dataset_X_excludes_target():
    ds = _two_session_dataset(prefix=5)
    assert "engagement_index" not in ds.X.columns
    assert "high_engagement" not in ds.X.columns


# --------------------------------------------------------------------------- #
# 3. Cross-session temporal context uses only the PAST.
# --------------------------------------------------------------------------- #
def _three_sessions(mutate_future: bool):
    day = 24 * 60
    s0 = [_msg(i, "A" if i % 2 else "B") for i in range(4)]
    s1 = [_msg(day + i, "A" if i % 2 else "B") for i in range(4)]
    if mutate_future:
        # Blow up session 2 into a giant marathon far in the future.
        s2 = [_msg(2 * day + i * 2, "A" if i % 2 else "B") for i in range(40)]
    else:
        s2 = [_msg(2 * day + i, "A" if i % 2 else "B") for i in range(4)]
    return detect_sessions(s0 + s1 + s2, inactivity_hours=6)


def test_context_features_ignore_future_sessions():
    base = build_feature_frame(_three_sessions(mutate_future=False))
    mutated = build_feature_frame(_three_sessions(mutate_future=True))
    ctx_cols = ["hours_since_prev_session", "rolling_msg_volume",
                "rolling_session_frequency_7d"]
    # Rows 0 and 1 must be untouched by whatever session 2 becomes.
    for row in (0, 1):
        for col in ctx_cols:
            assert base.iloc[row][col] == pytest.approx(
                mutated.iloc[row][col], nan_ok=True
            ), f"{col} at session {row} leaked from the future"


# --------------------------------------------------------------------------- #
# 4. Dataset keeps only sessions that actually have a future.
# --------------------------------------------------------------------------- #
def _two_session_dataset(prefix: int):
    day = 24 * 60
    s0 = [_msg(i, "A" if i % 2 else "B") for i in range(prefix + 6)]
    s1 = [_msg(day + i, "A" if i % 2 else "B") for i in range(prefix + 6)]
    sessions = detect_sessions(s0 + s1, inactivity_hours=6)
    return build_dataset(sessions, prefix=prefix)


def test_short_sessions_are_dropped():
    day = 24 * 60
    long_s = [_msg(i, "A" if i % 2 else "B") for i in range(12)]
    short_s = [_msg(day + i, "A" if i % 2 else "B") for i in range(3)]  # <= prefix
    sessions = detect_sessions(long_s + short_s, inactivity_hours=6)
    ds = build_dataset(sessions, prefix=5)
    assert len(ds) == 1  # only the long session survives


def test_all_short_raises_honestly():
    sessions = detect_sessions(
        [_msg(i, "A" if i % 2 else "B") for i in range(4)], inactivity_hours=6
    )
    with pytest.raises(ValueError):
        build_dataset(sessions, prefix=20)


def test_dataset_alignment():
    ds = _two_session_dataset(prefix=5)
    assert len(ds.X) == len(ds.y_regression) == len(ds.y_classification)
    assert list(ds.X.index) == ds.session_ids
