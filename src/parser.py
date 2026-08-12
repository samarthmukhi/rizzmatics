"""WhatsApp ``.txt`` export parser.

The parser turns a raw WhatsApp export into a clean, typed list of
:class:`Message` objects. It is deliberately paranoid because it is the
foundation of the entire pipeline: if parsing is wrong, every downstream
statistic is confidently, precisely wrong.

Supported input variety
------------------------
* Android format:  ``12/08/2026, 14:32 - Alice: Hey``
* iOS format:      ``[12/08/2026, 14:32:05] Alice: Hey``
* Date orders:     ``DD/MM/YYYY`` and ``MM/DD/YYYY`` (auto-detected)
* Date separators: ``/`` ``.`` ``-`` and ISO ``YYYY-MM-DD``
* 2- or 4-digit years
* 24-hour and 12-hour (AM/PM) time, with optional seconds
* Multiline messages (continuation lines with no timestamp)
* Media placeholders (``<Media omitted>`` etc.)
* System / notification lines (no sender)
* The narrow no-break space (U+202F) iOS wedges before ``AM``/``PM``

Each message is normalized into exactly the fields the brief asks for:
``timestamp``, ``sender``, ``message``, ``is_system_message``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["Message", "parse_chat", "parse_file", "to_dataframe"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    """A single normalized message.

    Attributes:
        timestamp: When the message was sent.
        sender: Display name of the sender, or ``None`` for system lines.
        message: The message text (multiline messages joined with ``\\n``).
        is_system_message: ``True`` for WhatsApp notifications with no author
            (group creation, encryption notices, etc.).
    """

    timestamp: datetime
    sender: Optional[str]
    message: str
    is_system_message: bool = False

    # Convenience so the raw line is available for debugging without
    # bloating equality/repr in tests.
    raw: str = field(default="", repr=False, compare=False)


# --------------------------------------------------------------------------- #
# Invisible-character hygiene
# --------------------------------------------------------------------------- #
# WhatsApp exports are littered with formatting marks. If we don't strip
# these, regexes miss and "AM" refuses to match "AM". Ask me how I know.
_INVISIBLES = {
    "‎": "",   # LEFT-TO-RIGHT MARK (prefixes media lines)
    "‏": "",   # RIGHT-TO-LEFT MARK
    "‪": "",   # LEFT-TO-RIGHT EMBEDDING
    "‬": "",   # POP DIRECTIONAL FORMATTING
    " ": " ",  # NO-BREAK SPACE
    " ": " ",  # NARROW NO-BREAK SPACE (iOS before AM/PM)
}
_INVISIBLE_RE = re.compile("|".join(map(re.escape, _INVISIBLES)))


def _clean(line: str) -> str:
    """Strip invisible marks and normalize odd whitespace to plain spaces."""
    return _INVISIBLE_RE.sub(lambda m: _INVISIBLES[m.group()], line)


# --------------------------------------------------------------------------- #
# Line-header regex
# --------------------------------------------------------------------------- #
# Matches the "timestamp + separator" prefix of a message line for both the
# Android (``date, time - body``) and iOS (``[date, time] body``) layouts.
_HEADER_RE = re.compile(
    r"^\[?\s*"                                      # optional iOS opening bracket
    r"(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"    # date, any common order
    r",?\s+"                                        # comma/space between date & time
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)"           # HH:MM optionally :SS
    r"\s*(?P<ampm>[AaPp][Mm])?"                     # optional AM/PM
    r"\s*\]?"                                        # optional iOS closing bracket
    r"\s*[-–—]?\s*"                        # Android " - " separator (or none)
    r"(?P<body>.*)$"
)

# Splits a body into "Sender: text". Sender names never contain a colon in
# practice, so we split on the first ": ".
_BODY_RE = re.compile(r"^(?P<sender>[^:]{1,120}?):\s(?P<text>.*)$", re.DOTALL)


# --------------------------------------------------------------------------- #
# Date / time parsing
# --------------------------------------------------------------------------- #
def _split_date(date_str: str) -> tuple[int, int, int]:
    """Split a date string into (year, month, day) integers.

    Handles ISO ``YYYY-MM-DD`` explicitly. For the two remaining fields the
    caller-provided ``dayfirst`` decision is applied by :func:`_parse_datetime`.
    Returns ``(year, first_field, second_field)`` where the two fields still
    need day/month disambiguation — except ISO, already disambiguated.
    """
    parts = re.split(r"[./-]", date_str)
    if len(parts) != 3:
        raise ValueError(f"Unrecognized date: {date_str!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _looks_dayfirst(a: int, b: int) -> Optional[bool]:
    """Infer day/month order from a single (first, second) numeric pair.

    Returns ``True`` if the pair can only be day-first, ``False`` if it can
    only be month-first, and ``None`` if the pair is ambiguous.
    """
    if a > 12 and b <= 12:
        return True
    if b > 12 and a <= 12:
        return False
    return None


def detect_dayfirst(lines: Iterable[str], default: bool = True) -> bool:
    """Auto-detect whether dates are day-first by scanning all headers.

    We look for any unambiguous date (a field > 12 forces the order). If the
    whole file is ambiguous (every date has both fields <= 12), we fall back
    to ``default`` — day-first, because most of the world (and this author)
    lives outside ``MM/DD``.
    """
    for line in lines:
        m = _HEADER_RE.match(_clean(line))
        if not m:
            continue
        try:
            f0, f1, f2 = _split_date(m.group("date"))
        except ValueError:
            continue
        if len(m.group("date").split("-")[0]) == 4 or f0 > 31:
            # ISO year-first; date order is unambiguous, keep scanning.
            continue
        verdict = _looks_dayfirst(f0, f1)
        if verdict is not None:
            return verdict
    return default


def _parse_datetime(date_str: str, time_str: str, ampm: Optional[str],
                    dayfirst: bool) -> datetime:
    """Assemble a :class:`datetime` from parsed date/time components."""
    f0, f1, f2 = _split_date(date_str)

    # ISO layout: first field is a 4-digit year -> Y, M, D.
    if len(date_str.split("-")[0]) == 4 and "-" in date_str:
        year, month, day = f0, f1, f2
    else:
        year = f2
        if dayfirst:
            day, month = f0, f1
        else:
            month, day = f0, f1
        if year < 100:  # 2-digit year -> assume 21st century
            year += 2000

    hh, mm, *rest = (int(x) for x in time_str.split(":"))
    ss = rest[0] if rest else 0

    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hh != 12:
            hh += 12
        elif ampm == "am" and hh == 12:
            hh = 0

    return datetime(year, month, day, hh, mm, ss)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_chat(text: str, *, dayfirst: Optional[bool] = None) -> list[Message]:
    """Parse a full WhatsApp export string into a list of :class:`Message`.

    Args:
        text: The raw contents of a WhatsApp ``.txt`` export.
        dayfirst: Force day-first (``True``) or month-first (``False``) date
            interpretation. ``None`` (default) auto-detects from the file and
            falls back to day-first when the file is ambiguous.

    Returns:
        Messages in chronological order as they appear in the export. Lines
        before the first valid header are ignored (they cannot be timestamped).
    """
    lines = text.splitlines()
    if dayfirst is None:
        dayfirst = detect_dayfirst(lines)

    messages: list[Message] = []
    current: Optional[Message] = None

    for raw_line in lines:
        line = _clean(raw_line)
        header = _HEADER_RE.match(line)

        if header:
            # A new message begins. Flush the one we were building.
            if current is not None:
                messages.append(current)

            try:
                ts = _parse_datetime(
                    header.group("date"),
                    header.group("time"),
                    header.group("ampm"),
                    dayfirst,
                )
            except (ValueError, OverflowError):
                # Header-shaped but not a real date (e.g. "3/4-5 stars: ...").
                # Treat as a continuation of the previous message instead of
                # inventing a timestamp.
                if current is not None:
                    current.message += "\n" + line
                continue

            body = header.group("body")
            body_match = _BODY_RE.match(body)
            if body_match:
                current = Message(
                    timestamp=ts,
                    sender=body_match.group("sender").strip(),
                    message=body_match.group("text"),
                    is_system_message=False,
                    raw=raw_line,
                )
            else:
                # No "Sender: " prefix -> WhatsApp system/notification line.
                current = Message(
                    timestamp=ts,
                    sender=None,
                    message=body,
                    is_system_message=True,
                    raw=raw_line,
                )
        else:
            # Continuation line of a multiline message.
            if current is not None:
                current.message += "\n" + line
            # else: preamble before the first header; drop it.

    if current is not None:
        messages.append(current)

    return messages


def parse_file(path: str | Path, *, dayfirst: Optional[bool] = None,
               encoding: str = "utf-8") -> list[Message]:
    """Parse a WhatsApp export from a file path.

    Args:
        path: Path to the ``.txt`` export.
        dayfirst: See :func:`parse_chat`.
        encoding: File encoding (WhatsApp exports UTF-8).
    """
    text = Path(path).read_text(encoding=encoding)
    return parse_chat(text, dayfirst=dayfirst)


def to_dataframe(messages: list[Message]):
    """Convert messages to a pandas DataFrame (imported lazily).

    Columns: ``timestamp``, ``sender``, ``message``, ``is_system_message``.
    """
    import pandas as pd  # local import keeps the parser dependency-light

    return pd.DataFrame(
        {
            "timestamp": [m.timestamp for m in messages],
            "sender": [m.sender for m in messages],
            "message": [m.message for m in messages],
            "is_system_message": [m.is_system_message for m in messages],
        }
    )
