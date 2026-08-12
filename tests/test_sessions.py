"""Tests for session detection."""

from datetime import datetime, timedelta

import pytest

from src.parser import Message
from src.sessions import Session, detect_sessions, sessions_to_dataframe


def _msg(dt: datetime, sender="Alice", text="hi", system=False) -> Message:
    return Message(timestamp=dt, sender=None if system else sender,
                   message=text, is_system_message=system)


def test_empty_input_returns_no_sessions():
    assert detect_sessions([]) == []


def test_single_session_when_all_within_threshold():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [_msg(base + timedelta(minutes=10 * i)) for i in range(5)]
    sessions = detect_sessions(msgs, inactivity_hours=6)
    assert len(sessions) == 1
    assert sessions[0].message_count == 5
    assert sessions[0].session_id == 0


def test_gap_larger_than_threshold_splits_sessions():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base),
        _msg(base + timedelta(minutes=5)),
        _msg(base + timedelta(hours=7)),        # 7h gap -> new session
        _msg(base + timedelta(hours=7, minutes=5)),
    ]
    sessions = detect_sessions(msgs, inactivity_hours=6)
    assert [s.message_count for s in sessions] == [2, 2]
    assert [s.session_id for s in sessions] == [0, 1]


def test_threshold_is_configurable():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [_msg(base), _msg(base + timedelta(hours=3))]
    assert len(detect_sessions(msgs, inactivity_hours=6)) == 1
    assert len(detect_sessions(msgs, inactivity_hours=2)) == 2


def test_gap_exactly_at_threshold_stays_same_session():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [_msg(base), _msg(base + timedelta(hours=6))]  # exactly 6h, not >6h
    assert len(detect_sessions(msgs, inactivity_hours=6)) == 1


def test_system_messages_excluded_by_default():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base, text="encrypted notice", system=True),
        _msg(base + timedelta(minutes=1), sender="Alice"),
        _msg(base + timedelta(minutes=2), sender="Bob"),
    ]
    sessions = detect_sessions(msgs)
    assert sessions[0].message_count == 2
    assert set(sessions[0].participants) == {"Alice", "Bob"}


def test_system_messages_included_when_requested():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base, text="encrypted notice", system=True),
        _msg(base + timedelta(minutes=1), sender="Alice"),
    ]
    sessions = detect_sessions(msgs, include_system=True)
    assert sessions[0].message_count == 2


def test_unsorted_input_is_sorted_before_sessionizing():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base + timedelta(minutes=2), text="third"),
        _msg(base, text="first"),
        _msg(base + timedelta(minutes=1), text="second"),
    ]
    sessions = detect_sessions(msgs)
    assert [m.message for m in sessions[0].messages] == ["first", "second", "third"]


def test_participants_in_first_appearance_order():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base, sender="Bob"),
        _msg(base + timedelta(minutes=1), sender="Alice"),
        _msg(base + timedelta(minutes=2), sender="Bob"),
    ]
    assert detect_sessions(msgs)[0].participants == ["Bob", "Alice"]


def test_session_duration_and_timing():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [_msg(base), _msg(base + timedelta(minutes=30))]
    s = detect_sessions(msgs)[0]
    assert s.start_time == base
    assert s.end_time == base + timedelta(minutes=30)
    assert s.duration == timedelta(minutes=30)
    assert s.duration_minutes == pytest.approx(30.0)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        detect_sessions([_msg(datetime(2026, 8, 12, 14, 0))], inactivity_hours=0)


def test_all_system_messages_excluded_yields_empty():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [_msg(base, text="notice", system=True)]
    assert detect_sessions(msgs) == []


def test_sessions_to_dataframe():
    base = datetime(2026, 8, 12, 14, 0)
    msgs = [
        _msg(base, sender="Alice"),
        _msg(base + timedelta(minutes=5), sender="Bob"),
        _msg(base + timedelta(hours=8), sender="Alice"),
    ]
    df = sessions_to_dataframe(detect_sessions(msgs))
    assert list(df.columns) == [
        "session_id", "start_time", "end_time", "duration_minutes",
        "n_participants", "participants", "message_count",
    ]
    assert len(df) == 2
    assert df.loc[0, "n_participants"] == 2
    assert df.loc[1, "message_count"] == 1
