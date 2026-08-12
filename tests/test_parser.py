"""Tests for the WhatsApp parser.

The parser is the foundation of Rizzmatics, so we beat on it: every date
format, iOS vs Android, multiline, system messages, media, AM/PM, and the
cursed invisible characters WhatsApp loves to sneak in.
"""

from datetime import datetime

import pytest

from src.parser import (
    Message,
    detect_dayfirst,
    parse_chat,
    to_dataframe,
)


# --------------------------------------------------------------------------- #
# Basic Android format
# --------------------------------------------------------------------------- #
def test_basic_android_two_messages():
    text = (
        "12/08/2026, 14:32 - Alice: Hey\n"
        "12/08/2026, 14:33 - Bob: What's up?\n"
    )
    msgs = parse_chat(text, dayfirst=True)
    assert len(msgs) == 2
    assert msgs[0] == Message(
        timestamp=datetime(2026, 8, 12, 14, 32),
        sender="Alice",
        message="Hey",
        is_system_message=False,
    )
    assert msgs[1].sender == "Bob"
    assert msgs[1].message == "What's up?"
    assert msgs[1].timestamp == datetime(2026, 8, 12, 14, 33)


def test_dayfirst_vs_monthfirst_changes_interpretation():
    text = "01/02/2026, 09:00 - A: hi\n"
    day = parse_chat(text, dayfirst=True)[0].timestamp
    month = parse_chat(text, dayfirst=False)[0].timestamp
    assert day == datetime(2026, 2, 1, 9, 0)     # 1 Feb
    assert month == datetime(2026, 1, 2, 9, 0)    # 2 Jan


# --------------------------------------------------------------------------- #
# 12-hour clock / AM-PM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "time_str, expected_hour",
    [
        ("2:32 PM", 14),
        ("2:32 AM", 2),
        ("12:00 AM", 0),   # midnight
        ("12:00 PM", 12),  # noon
        ("11:59 PM", 23),
    ],
)
def test_ampm_conversion(time_str, expected_hour):
    text = f"12/08/2026, {time_str} - Alice: hi\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.timestamp.hour == expected_hour


def test_narrow_no_break_space_before_ampm():
    # iOS uses U+202F (narrow no-break space) between the time and AM/PM.
    text = "12/08/2026, 2:32 PM - Alice: hi\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.timestamp.hour == 14


# --------------------------------------------------------------------------- #
# iOS bracketed format
# --------------------------------------------------------------------------- #
def test_ios_bracketed_with_seconds():
    text = "[12/08/2026, 14:32:05] Alice: Hey there\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.sender == "Alice"
    assert msg.message == "Hey there"
    assert msg.timestamp == datetime(2026, 8, 12, 14, 32, 5)


def test_ios_left_to_right_mark_stripped():
    # iOS prefixes some lines with U+200E; it must not break parsing.
    text = "‎[12/08/2026, 14:32:05] Alice: ‎image omitted\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.sender == "Alice"
    assert msg.message == "image omitted"


# --------------------------------------------------------------------------- #
# Date separators, ISO, and 2-digit years
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("12.08.2026", datetime(2026, 8, 12, 10, 0)),   # dot separator
        ("12-08-2026", datetime(2026, 8, 12, 10, 0)),   # dash separator
        ("12/08/26", datetime(2026, 8, 12, 10, 0)),     # 2-digit year
    ],
)
def test_date_separators_and_short_year(date_str, expected):
    text = f"{date_str}, 10:00 - A: hi\n"
    assert parse_chat(text, dayfirst=True)[0].timestamp == expected


def test_iso_date_format():
    text = "2026-08-12, 14:32 - Alice: hi\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.timestamp == datetime(2026, 8, 12, 14, 32)


# --------------------------------------------------------------------------- #
# Multiline messages
# --------------------------------------------------------------------------- #
def test_multiline_message_is_joined():
    text = (
        "12/08/2026, 14:32 - Alice: line one\n"
        "line two\n"
        "line three\n"
        "12/08/2026, 14:35 - Bob: reply\n"
    )
    msgs = parse_chat(text, dayfirst=True)
    assert len(msgs) == 2
    assert msgs[0].message == "line one\nline two\nline three"
    assert msgs[1].message == "reply"


def test_preamble_before_first_header_is_dropped():
    text = (
        "garbage line with no timestamp\n"
        "12/08/2026, 14:32 - Alice: real message\n"
    )
    msgs = parse_chat(text, dayfirst=True)
    assert len(msgs) == 1
    assert msgs[0].message == "real message"


# --------------------------------------------------------------------------- #
# System messages and media placeholders
# --------------------------------------------------------------------------- #
def test_system_message_has_no_sender():
    text = (
        "12/08/2026, 14:30 - Messages and calls are end-to-end encrypted.\n"
        "12/08/2026, 14:32 - Alice: hi\n"
    )
    msgs = parse_chat(text, dayfirst=True)
    assert msgs[0].is_system_message is True
    assert msgs[0].sender is None
    assert msgs[1].is_system_message is False


def test_media_placeholder_is_a_normal_message():
    text = "12/08/2026, 14:32 - Alice: <Media omitted>\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.sender == "Alice"
    assert msg.message == "<Media omitted>"
    assert msg.is_system_message is False


def test_message_containing_colon_keeps_full_text():
    text = "12/08/2026, 14:32 - Alice: link: https://example.com/x\n"
    msg = parse_chat(text, dayfirst=True)[0]
    assert msg.sender == "Alice"
    assert msg.message == "link: https://example.com/x"


# --------------------------------------------------------------------------- #
# Auto-detection of day-first
# --------------------------------------------------------------------------- #
def test_detect_dayfirst_true_when_day_exceeds_12():
    lines = ["25/01/2026, 10:00 - A: hi"]
    assert detect_dayfirst(lines) is True


def test_detect_dayfirst_false_when_second_field_exceeds_12():
    lines = ["01/25/2026, 10:00 - A: hi"]
    assert detect_dayfirst(lines) is False


def test_detect_dayfirst_defaults_true_when_ambiguous():
    lines = ["01/02/2026, 10:00 - A: hi"]  # could be either
    assert detect_dayfirst(lines) is True
    assert detect_dayfirst(lines, default=False) is False


def test_parse_chat_autodetects_month_first():
    # An unambiguous 13th-of-month in MM/DD forces month-first for the file.
    text = (
        "01/13/2026, 10:00 - A: hi\n"
        "01/02/2026, 11:00 - B: yo\n"
    )
    msgs = parse_chat(text)  # dayfirst=None -> auto
    assert msgs[0].timestamp == datetime(2026, 1, 13, 10, 0)
    assert msgs[1].timestamp == datetime(2026, 1, 2, 11, 0)


# --------------------------------------------------------------------------- #
# Chronological ordering & DataFrame export
# --------------------------------------------------------------------------- #
def test_messages_preserve_file_order():
    text = (
        "12/08/2026, 14:32 - Alice: 1\n"
        "12/08/2026, 14:33 - Bob: 2\n"
        "12/08/2026, 14:34 - Alice: 3\n"
    )
    msgs = parse_chat(text, dayfirst=True)
    assert [m.message for m in msgs] == ["1", "2", "3"]


def test_to_dataframe_shape_and_columns():
    text = (
        "12/08/2026, 14:32 - Alice: hi\n"
        "12/08/2026, 14:30 - System boot\n"
    )
    df = to_dataframe(parse_chat(text, dayfirst=True))
    assert list(df.columns) == [
        "timestamp",
        "sender",
        "message",
        "is_system_message",
    ]
    assert len(df) == 2
    assert df["is_system_message"].tolist() == [False, True]


def test_empty_input_returns_empty_list():
    assert parse_chat("") == []
    assert parse_chat("\n\n\n") == []
