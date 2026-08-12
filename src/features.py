"""Feature engineering — turning conversations into feature vectors.

This module extracts *interpretable behavioral signals* from messages. Every
feature answers "what observably happened", never "what did it mean". Response
latency is a stopwatch reading, not a diagnosis of someone's feelings.

Two layers:

* :func:`extract_features` — the pure primitive. Give it any list of messages
  (a whole session, or just its first N messages) and it returns a flat dict of
  numeric features. This is the piece that makes leakage-safe modeling possible:
  we feed it a *prefix* and predict the *whole*.
* :func:`build_feature_frame` — assembles a per-session DataFrame, optionally
  from a message prefix, and layers on cross-session temporal context computed
  strictly from *past* sessions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .parser import Message
from .sessions import Session

__all__ = [
    "extract_features",
    "build_feature_frame",
    "FEATURE_NAMES",
    "count_emojis",
    "is_media_message",
]

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LINK_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# A pragmatic emoji matcher covering the blocks that actually show up in chats.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs extended-A
    "\U00002600-\U000026FF"   # miscellaneous symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (stars, etc.)
    "\U00002190-\U000021FF"   # arrows
    "]",
    flags=re.UNICODE,
)

_MEDIA_MARKERS = (
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "this message was deleted",
    "you deleted this message",
)


def count_emojis(text: str) -> int:
    """Count emoji code points in a string."""
    return len(_EMOJI_RE.findall(text))


def is_media_message(text: str) -> bool:
    """True if the message is a WhatsApp media/omitted placeholder."""
    low = text.strip().lower()
    return any(marker in low for marker in _MEDIA_MARKERS)


# --------------------------------------------------------------------------- #
# Canonical feature list (order is stable for the model)
# --------------------------------------------------------------------------- #
FEATURE_NAMES: list[str] = [
    # volume
    "n_messages", "total_chars", "mean_msg_len", "median_msg_len", "max_msg_len",
    # participation
    "n_participants", "participation_balance", "max_participation_share",
    "n_runs", "mean_run_length", "back_and_forth_rate",
    # response behavior
    "median_response_latency_s", "mean_response_latency_s",
    "p90_response_latency_s", "frac_replies_within_threshold",
    # linguistic
    "total_words", "mean_words_per_msg", "question_rate", "exclamation_rate",
    "emoji_count", "emoji_density", "link_count", "media_count",
    "lexical_diversity",
    # temporal (within-session)
    "start_hour", "is_weekend", "is_late_night",
]

# Cross-session temporal features, added by build_feature_frame.
_CONTEXT_FEATURES = [
    "hours_since_prev_session",
    "rolling_msg_volume",
    "rolling_session_frequency_7d",
]

_NAN = float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return _NAN
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]). NaN if empty."""
    if not values:
        return _NAN
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def extract_features(
    messages: list[Message],
    *,
    latency_threshold_s: float = 300.0,
) -> dict[str, float]:
    """Extract behavioral features from a list of messages.

    The messages should be time-ordered and represent one conversational unit
    (a session, or the leading prefix of one). System messages are ignored.

    Args:
        messages: Time-ordered messages.
        latency_threshold_s: "Reply within X seconds" threshold for the
            ``frac_replies_within_threshold`` feature. Default 5 minutes.

    Returns:
        A flat ``{feature_name: value}`` dict. Undefined quantities (e.g. reply
        latency with no turn changes) are ``NaN`` and left for the pipeline's
        imputer — we do not fabricate zeros where there is no data.
    """
    msgs = [m for m in messages if not m.is_system_message]
    feats: dict[str, float] = {name: _NAN for name in FEATURE_NAMES}

    n = len(msgs)
    feats["n_messages"] = float(n)
    if n == 0:
        return feats

    texts = [m.message for m in msgs]
    senders = [m.sender for m in msgs]
    lengths = [len(t) for t in texts]

    # ---- volume -----------------------------------------------------------
    feats["total_chars"] = float(sum(lengths))
    feats["mean_msg_len"] = sum(lengths) / n
    feats["median_msg_len"] = _median([float(x) for x in lengths])
    feats["max_msg_len"] = float(max(lengths))

    # ---- participation ----------------------------------------------------
    counts = Counter(senders)
    n_participants = len(counts)
    feats["n_participants"] = float(n_participants)
    total = sum(counts.values())
    shares = [c / total for c in counts.values()]
    feats["max_participation_share"] = max(shares)
    if n_participants > 1:
        entropy = -sum(p * math.log(p) for p in shares if p > 0)
        feats["participation_balance"] = entropy / math.log(n_participants)
    else:
        feats["participation_balance"] = 0.0

    # consecutive-message runs & turn-taking
    runs = 1
    turn_changes = 0
    for a, b in zip(senders, senders[1:]):
        if a != b:
            runs += 1
            turn_changes += 1
    feats["n_runs"] = float(runs)
    feats["mean_run_length"] = n / runs
    feats["back_and_forth_rate"] = turn_changes / (n - 1) if n > 1 else 0.0

    # ---- response behavior ------------------------------------------------
    latencies: list[float] = []
    for prev, cur in zip(msgs, msgs[1:]):
        if prev.sender != cur.sender:  # a reply, not a self-continuation
            latencies.append((cur.timestamp - prev.timestamp).total_seconds())
    if latencies:
        feats["median_response_latency_s"] = _median(latencies)
        feats["mean_response_latency_s"] = sum(latencies) / len(latencies)
        feats["p90_response_latency_s"] = _percentile(latencies, 90)
        within = sum(1 for l in latencies if l <= latency_threshold_s)
        feats["frac_replies_within_threshold"] = within / len(latencies)

    # ---- linguistic -------------------------------------------------------
    all_words: list[str] = []
    questions = exclamations = emoji_count = link_count = media_count = 0
    for t in texts:
        words = _WORD_RE.findall(t.lower())
        all_words.extend(words)
        if "?" in t:
            questions += 1
        if "!" in t:
            exclamations += 1
        emoji_count += count_emojis(t)
        link_count += len(_LINK_RE.findall(t))
        if is_media_message(t):
            media_count += 1

    total_words = len(all_words)
    feats["total_words"] = float(total_words)
    feats["mean_words_per_msg"] = total_words / n
    feats["question_rate"] = questions / n
    feats["exclamation_rate"] = exclamations / n
    feats["emoji_count"] = float(emoji_count)
    feats["emoji_density"] = emoji_count / total_words if total_words else 0.0
    feats["link_count"] = float(link_count)
    feats["media_count"] = float(media_count)
    feats["lexical_diversity"] = (
        len(set(all_words)) / total_words if total_words else 0.0
    )

    # ---- temporal (within-session) ---------------------------------------
    start = msgs[0].timestamp
    feats["start_hour"] = float(start.hour)
    feats["is_weekend"] = 1.0 if start.weekday() >= 5 else 0.0
    feats["is_late_night"] = 1.0 if (start.hour >= 23 or start.hour < 6) else 0.0

    return feats


# --------------------------------------------------------------------------- #
# Session-sequence assembly
# --------------------------------------------------------------------------- #
def build_feature_frame(
    sessions: list[Session],
    *,
    prefix: Optional[int] = None,
    latency_threshold_s: float = 300.0,
    rolling_k: int = 3,
):
    """Build a per-session feature DataFrame (imported lazily).

    Args:
        sessions: Detected sessions, in chronological order.
        prefix: If given, features are extracted from only the first ``prefix``
            messages of each session (the leakage-safe "early portion"). If
            ``None``, the whole session is used.
        latency_threshold_s: Passed through to :func:`extract_features`.
        rolling_k: Window size for the rolling message-volume feature, computed
            over the ``rolling_k`` *previous* sessions.

    Returns:
        DataFrame indexed by ``session_id`` with all features in ``FEATURE_NAMES``
        plus cross-session temporal context. Context features are derived only
        from sessions strictly before the current one — never the future.
    """
    import pandas as pd

    rows = []
    for i, session in enumerate(sessions):
        msgs = session.messages
        if prefix is not None:
            msgs = msgs[:prefix]
        feats = extract_features(msgs, latency_threshold_s=latency_threshold_s)
        feats["session_id"] = session.session_id

        # ---- cross-session temporal context (past-only) ------------------
        if i == 0:
            feats["hours_since_prev_session"] = _NAN
            feats["rolling_msg_volume"] = _NAN
            feats["rolling_session_frequency_7d"] = 0.0
        else:
            prev = sessions[i - 1]
            gap = (session.start_time - prev.end_time).total_seconds() / 3600.0
            feats["hours_since_prev_session"] = gap
            window = sessions[max(0, i - rolling_k):i]
            feats["rolling_msg_volume"] = (
                sum(s.message_count for s in window) / len(window)
            )
            cutoff = session.start_time.timestamp() - 7 * 24 * 3600
            feats["rolling_session_frequency_7d"] = float(
                sum(1 for s in sessions[:i] if s.start_time.timestamp() >= cutoff)
            )
        rows.append(feats)

    cols = ["session_id"] + FEATURE_NAMES + _CONTEXT_FEATURES
    return pd.DataFrame(rows, columns=cols).set_index("session_id")
