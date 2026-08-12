"""Dataset assembly — leakage-safe by construction.

This is where the central methodological promise lives:

    Features are extracted from the FIRST ``prefix`` messages of a session.
    The target (engagement) is computed from the WHOLE session.

So the model only ever sees the *early portion* of a conversation and is asked
to predict how the *rest* of it turns out. We also keep only sessions that
actually have a future — i.e. more messages than the prefix — because
"predicting the future" of a conversation that already ended at message 4 is
not prediction, it's transcription.

See ``tests/test_leakage.py`` for the executable version of this promise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engagement import EngagementConfig, compute_engagement, label_high_engagement
from .features import build_feature_frame
from .sessions import Session

__all__ = ["Dataset", "build_dataset"]


@dataclass
class Dataset:
    """A leakage-safe supervised dataset.

    Attributes:
        X: Feature matrix (DataFrame), one row per session, extracted from each
            session's message *prefix* only.
        y_regression: Engagement index of each session's *full* conversation.
        y_classification: Binary HIGH/LOW engagement labels.
        high_threshold: The engagement cutoff used for the labels.
        prefix: The message prefix length used for features.
        session_ids: The session ids kept, aligned with the rows of ``X``.
    """

    X: "object"                 # pandas DataFrame
    y_regression: "object"      # pandas Series
    y_classification: "object"  # pandas Series
    high_threshold: float
    prefix: int
    session_ids: list[int]

    def __len__(self) -> int:
        return len(self.session_ids)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def build_dataset(
    sessions: list[Session],
    *,
    prefix: int = 20,
    min_future_messages: int = 1,
    high_percentile: float = 75.0,
    engagement_config: EngagementConfig | None = None,
    latency_threshold_s: float = 300.0,
) -> Dataset:
    """Assemble features (from prefixes) and targets (from full sessions).

    Args:
        sessions: Chronologically ordered sessions.
        prefix: Number of leading messages per session used for features.
        min_future_messages: A session is kept only if it has at least this
            many messages *beyond* the prefix, so there is genuine future to
            predict. Defaults to 1 (strictly more than the prefix).
        high_percentile: Percentile cutoff for the HIGH-engagement class.
        engagement_config: Weighting for the engagement index.
        latency_threshold_s: Response-latency threshold for features.

    Returns:
        A :class:`Dataset`. Raises ``ValueError`` if no session survives the
        ``min_future_messages`` filter (the honest failure mode when a chat is
        just too short to model).
    """
    # Target uses the FULL sessions.
    engagement = compute_engagement(sessions, engagement_config)

    # Keep only sessions with a real "future" beyond the prefix.
    keep = [s for s in sessions if s.message_count >= prefix + min_future_messages]
    if not keep:
        raise ValueError(
            f"No session has more than {prefix} messages "
            f"(+{min_future_messages} future). "
            "Lower the prefix, gather more data, or accept that some chats are "
            "simply too short to forecast. Bro, you gave me nothing to work with."
        )
    keep_ids = [s.session_id for s in keep]

    # Features use ONLY the prefix.
    X = build_feature_frame(
        keep, prefix=prefix, latency_threshold_s=latency_threshold_s
    )

    # Align targets to kept sessions.
    y_reg = engagement.loc[keep_ids, "engagement_index"].copy()
    y_reg.name = "engagement_index"
    y_clf, threshold = label_high_engagement(y_reg, percentile=high_percentile)

    return Dataset(
        X=X,
        y_regression=y_reg,
        y_classification=y_clf,
        high_threshold=threshold,
        prefix=prefix,
        session_ids=keep_ids,
    )
