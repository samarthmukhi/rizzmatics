"""Session detection.

A WhatsApp export is not one conversation — it is months of them, glued
together by the export button. Treating the whole thing as a single unit
would blend a 3-hour late-night marathon with a "k" sent nine days later.

We split the message stream into *sessions* using an inactivity threshold:
whenever the gap between consecutive messages exceeds the threshold, a new
session begins. The default gap is 6 hours and is configurable, because
your definition of "the same conversation" is not our business.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .parser import Message

__all__ = ["Session", "detect_sessions", "sessions_to_dataframe"]

DEFAULT_INACTIVITY_HOURS = 6.0


@dataclass
class Session:
    """A single conversational session: a contiguous burst of messages.

    Attributes:
        session_id: Zero-based index of the session in the conversation.
        messages: The messages belonging to this session, in time order.
    """

    session_id: int
    messages: list[Message]

    @property
    def start_time(self) -> datetime:
        return self.messages[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self.messages[-1].timestamp

    @property
    def duration(self) -> timedelta:
        """Wall-clock span from first to last message."""
        return self.end_time - self.start_time

    @property
    def duration_minutes(self) -> float:
        return self.duration.total_seconds() / 60.0

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def participants(self) -> list[str]:
        """Distinct non-system senders, in order of first appearance."""
        seen: list[str] = []
        for m in self.messages:
            if m.sender is not None and m.sender not in seen:
                seen.append(m.sender)
        return seen

    def __len__(self) -> int:
        return len(self.messages)


def detect_sessions(
    messages: list[Message],
    inactivity_hours: float = DEFAULT_INACTIVITY_HOURS,
    *,
    include_system: bool = False,
) -> list[Session]:
    """Split messages into sessions by inactivity gaps.

    Args:
        messages: Parsed messages (any order; sorted internally by time).
        inactivity_hours: A gap larger than this many hours starts a new
            session. Must be positive.
        include_system: If ``False`` (default), WhatsApp system/notification
            lines are dropped before sessionizing so a lone "X was added"
            doesn't masquerade as a conversation.

    Returns:
        Sessions in chronological order, each with a zero-based ``session_id``.
        Empty input (or input that is all system messages, when excluded)
        returns an empty list.
    """
    if inactivity_hours <= 0:
        raise ValueError("inactivity_hours must be positive")

    pool = [
        m for m in messages if include_system or not m.is_system_message
    ]
    pool.sort(key=lambda m: m.timestamp)
    if not pool:
        return []

    threshold = timedelta(hours=inactivity_hours)
    sessions: list[Session] = []
    current: list[Message] = [pool[0]]

    for prev, msg in zip(pool, pool[1:]):
        if msg.timestamp - prev.timestamp > threshold:
            sessions.append(Session(len(sessions), current))
            current = []
        current.append(msg)

    sessions.append(Session(len(sessions), current))
    return sessions


def sessions_to_dataframe(sessions: list[Session]):
    """Session-level summary DataFrame (imported lazily).

    Columns: ``session_id``, ``start_time``, ``end_time``,
    ``duration_minutes``, ``n_participants``, ``participants``,
    ``message_count``.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "session_id": s.session_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_minutes": round(s.duration_minutes, 3),
                "n_participants": len(s.participants),
                "participants": ", ".join(s.participants),
                "message_count": s.message_count,
            }
            for s in sessions
        ]
    )
