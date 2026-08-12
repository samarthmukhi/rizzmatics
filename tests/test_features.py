"""Tests for feature extraction."""

import math
from datetime import datetime, timedelta

import pytest

from src.parser import Message
from src.features import (
    FEATURE_NAMES,
    build_feature_frame,
    count_emojis,
    extract_features,
    is_media_message,
)
from src.sessions import detect_sessions


def _msg(minute, sender, text):
    return Message(
        timestamp=datetime(2026, 8, 12, 14, 0) + timedelta(minutes=minute),
        sender=sender,
        message=text,
        is_system_message=False,
    )


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def test_count_emojis():
    assert count_emojis("no emoji here") == 0
    assert count_emojis("hi 😄") == 1
    assert count_emojis("😄🔥🚀") == 3


def test_is_media_message():
    assert is_media_message("<Media omitted>")
    assert is_media_message("image omitted")
    assert is_media_message("This message was deleted")
    assert not is_media_message("just a normal message")


# --------------------------------------------------------------------------- #
# Empty / degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_messages_returns_all_features_with_zero_count():
    feats = extract_features([])
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert feats["n_messages"] == 0.0
    # Everything else undefined -> NaN, not a fabricated number.
    assert math.isnan(feats["mean_msg_len"])


def test_system_messages_are_ignored():
    msgs = [
        Message(datetime(2026, 8, 12, 14, 0), None, "encrypted", True),
        _msg(1, "Alice", "hello"),
    ]
    feats = extract_features(msgs)
    assert feats["n_messages"] == 1.0


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def test_volume_features():
    msgs = [_msg(0, "A", "hi"), _msg(1, "B", "hello!!")]  # len 2, len 7
    feats = extract_features(msgs)
    assert feats["n_messages"] == 2.0
    assert feats["total_chars"] == 9.0
    assert feats["mean_msg_len"] == pytest.approx(4.5)
    assert feats["median_msg_len"] == pytest.approx(4.5)
    assert feats["max_msg_len"] == 7.0


# --------------------------------------------------------------------------- #
# Participation
# --------------------------------------------------------------------------- #
def test_participation_balance_perfect_for_equal_two_party():
    msgs = [_msg(0, "A", "x"), _msg(1, "B", "y"), _msg(2, "A", "z"), _msg(3, "B", "w")]
    feats = extract_features(msgs)
    assert feats["n_participants"] == 2.0
    assert feats["participation_balance"] == pytest.approx(1.0)
    assert feats["max_participation_share"] == pytest.approx(0.5)


def test_participation_balance_zero_for_single_party():
    msgs = [_msg(0, "A", "x"), _msg(1, "A", "y"), _msg(2, "A", "z")]
    feats = extract_features(msgs)
    assert feats["n_participants"] == 1.0
    assert feats["participation_balance"] == 0.0
    assert feats["max_participation_share"] == 1.0


def test_runs_and_back_and_forth():
    # A A B A  -> senders change at idx1->2 and 2->3 => 3 runs, 2 turn changes
    msgs = [_msg(0, "A", "1"), _msg(1, "A", "2"), _msg(2, "B", "3"), _msg(3, "A", "4")]
    feats = extract_features(msgs)
    assert feats["n_runs"] == 3.0
    assert feats["mean_run_length"] == pytest.approx(4 / 3)
    assert feats["back_and_forth_rate"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Response latency
# --------------------------------------------------------------------------- #
def test_response_latency_only_counts_turn_changes():
    # A@0, A@2 (self, ignored), B@5 (reply, 3min=180s), A@6 (reply, 60s)
    msgs = [
        _msg(0, "A", "hey"),
        _msg(2, "A", "you there"),
        _msg(5, "B", "yeah"),
        _msg(6, "A", "cool"),
    ]
    feats = extract_features(msgs, latency_threshold_s=120)
    # latencies: 180s (A->B), 60s (B->A)
    assert feats["median_response_latency_s"] == pytest.approx(120.0)
    assert feats["mean_response_latency_s"] == pytest.approx(120.0)
    # within 120s threshold: only the 60s reply -> 1/2
    assert feats["frac_replies_within_threshold"] == pytest.approx(0.5)


def test_response_latency_nan_when_no_turn_changes():
    msgs = [_msg(0, "A", "1"), _msg(1, "A", "2")]
    feats = extract_features(msgs)
    assert math.isnan(feats["median_response_latency_s"])
    assert math.isnan(feats["frac_replies_within_threshold"])


# --------------------------------------------------------------------------- #
# Linguistic
# --------------------------------------------------------------------------- #
def test_linguistic_features():
    msgs = [
        _msg(0, "A", "how are you?"),         # question
        _msg(1, "B", "great!! 😄"),           # exclamation + emoji
        _msg(2, "A", "check https://x.com"),  # link
        _msg(3, "B", "<Media omitted>"),      # media
    ]
    feats = extract_features(msgs)
    assert feats["question_rate"] == pytest.approx(0.25)
    assert feats["exclamation_rate"] == pytest.approx(0.25)
    assert feats["emoji_count"] == 1.0
    assert feats["link_count"] == 1.0
    assert feats["media_count"] == 1.0
    assert 0.0 < feats["lexical_diversity"] <= 1.0


def test_lexical_diversity_bounds():
    msgs = [_msg(0, "A", "word word word")]
    feats = extract_features(msgs)
    assert feats["lexical_diversity"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Temporal
# --------------------------------------------------------------------------- #
def test_temporal_flags():
    # 2026-08-12 is a Wednesday, 14:00 -> not weekend, not late night
    feats = extract_features([_msg(0, "A", "hi")])
    assert feats["start_hour"] == 14.0
    assert feats["is_weekend"] == 0.0
    assert feats["is_late_night"] == 0.0

    late = Message(datetime(2026, 8, 15, 2, 30), "A", "up late", False)  # Sat 2:30am
    lf = extract_features([late])
    assert lf["is_weekend"] == 1.0
    assert lf["is_late_night"] == 1.0


# --------------------------------------------------------------------------- #
# Prefix behavior (the leakage-safe primitive)
# --------------------------------------------------------------------------- #
def test_prefix_uses_only_first_n_messages():
    msgs = [_msg(i, "A" if i % 2 else "B", f"m{i}") for i in range(10)]
    full = extract_features(msgs)
    prefix = extract_features(msgs[:3])
    assert full["n_messages"] == 10.0
    assert prefix["n_messages"] == 3.0


# --------------------------------------------------------------------------- #
# build_feature_frame + cross-session context
# --------------------------------------------------------------------------- #
def _session_msgs():
    # Two sessions separated by an 8h gap.
    s1 = [_msg(0, "A", "hi"), _msg(1, "B", "yo"), _msg(2, "A", "sup")]
    s2 = [
        Message(datetime(2026, 8, 12, 23, 0), "A", "back", False),
        Message(datetime(2026, 8, 12, 23, 5), "B", "hey", False),
    ]
    return s1 + s2


def test_build_feature_frame_columns_and_index():
    sessions = detect_sessions(_session_msgs(), inactivity_hours=6)
    df = build_feature_frame(sessions)
    assert df.index.name == "session_id"
    for name in FEATURE_NAMES:
        assert name in df.columns
    assert "hours_since_prev_session" in df.columns
    assert len(df) == 2


def test_first_session_has_nan_context():
    sessions = detect_sessions(_session_msgs(), inactivity_hours=6)
    df = build_feature_frame(sessions)
    assert math.isnan(df.iloc[0]["hours_since_prev_session"])
    assert df.iloc[1]["hours_since_prev_session"] > 0


def test_build_feature_frame_prefix_limits_messages():
    sessions = detect_sessions(_session_msgs(), inactivity_hours=6)
    df_full = build_feature_frame(sessions)
    df_prefix = build_feature_frame(sessions, prefix=2)
    assert df_full.iloc[0]["n_messages"] == 3.0
    assert df_prefix.iloc[0]["n_messages"] == 2.0
