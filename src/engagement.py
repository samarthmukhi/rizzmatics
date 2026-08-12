"""The Conversational Engagement Index™.

We need a *measurable target* to predict. This module defines one from purely
observable behavior, with fully configurable weights.

Read this and internalize it before you get any ideas:

    The Engagement Index is a research-defined behavioral proxy. It is NOT an
    objective measurement of human connection. It is not a love score, an
    attraction score, a chemistry score, or a compatibility score. It measures
    how *active and balanced* a conversation was — nothing about why, and
    nothing about anyone's feelings.

Components (all mapped to [0, 1], combined by configurable weights):

* ``duration``      — normalized session length in wall-clock minutes
* ``volume``        — normalized message count
* ``bidirectional`` — back-and-forth turn-taking rate (already 0..1)
* ``balance``       — participation balance / evenness (already 0..1)
* ``persistence``   — normalized number of sustained turn alternations
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import extract_features
from .sessions import Session

__all__ = [
    "EngagementConfig",
    "DEFAULT_WEIGHTS",
    "ENGAGEMENT_DISCLAIMER",
    "compute_engagement",
    "label_high_engagement",
]

ENGAGEMENT_DISCLAIMER = (
    "The Conversational Engagement Index is a research-defined behavioral "
    "proxy, not an objective measurement of human connection. It measures "
    "observable conversational activity and balance — not feelings, "
    "attraction, or meaning."
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "duration": 0.30,
    "volume": 0.25,
    "bidirectional": 0.20,
    "balance": 0.15,
    "persistence": 0.10,
}

_COMPONENTS = ("duration", "volume", "bidirectional", "balance", "persistence")
# These two are already on a [0, 1] scale, so we don't min-max them.
_ALREADY_NORMALIZED = {"bidirectional", "balance"}


@dataclass
class EngagementConfig:
    """Configuration for the engagement index.

    Attributes:
        weights: Component weights. Normalized internally to sum to 1 so the
            resulting index always lands in [0, 1]; the relative emphasis is
            what matters.
        clip_percentile: Robust min-max normalization clips raw components to
            this upper percentile before scaling, so one 11-hour marathon
            session doesn't flatten everything else to zero. Set to 100 to
            disable clipping.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    clip_percentile: float = 95.0

    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.weights.get(c, 0.0) for c in _COMPONENTS)
        if total <= 0:
            raise ValueError("Engagement weights must sum to a positive value.")
        return {c: self.weights.get(c, 0.0) / total for c in _COMPONENTS}


def _raw_components(session: Session) -> dict[str, float]:
    """Compute the raw (pre-normalization) component values for one session."""
    feats = extract_features(session.messages)
    n = session.message_count
    turn_changes = (feats["n_runs"] - 1.0) if n > 1 else 0.0
    return {
        "duration": session.duration_minutes,
        "volume": float(n),
        "bidirectional": feats["back_and_forth_rate"] if n > 1 else 0.0,
        "balance": feats["participation_balance"],
        "persistence": turn_changes,
    }


def compute_engagement(
    sessions: list[Session],
    config: EngagementConfig | None = None,
):
    """Compute the Engagement Index for each session (pandas imported lazily).

    Normalization is done across the provided ``sessions`` — the index is
    relative to *this* conversation's own range, which is the honest thing to
    do for a single person's chat history.

    Returns:
        DataFrame indexed by ``session_id`` with the raw component values, their
        normalized ``*_norm`` counterparts, and the final ``engagement_index``
        in [0, 1]. An empty input returns an empty DataFrame.
    """
    import numpy as np
    import pandas as pd

    if not sessions:
        return pd.DataFrame()

    cfg = config or EngagementConfig()
    weights = cfg.normalized_weights()

    raw = pd.DataFrame(
        [_raw_components(s) for s in sessions],
        index=[s.session_id for s in sessions],
    )
    raw.index.name = "session_id"

    norm = pd.DataFrame(index=raw.index)
    for comp in _COMPONENTS:
        col = raw[comp].astype(float)
        if comp in _ALREADY_NORMALIZED:
            norm[f"{comp}_norm"] = col.clip(0.0, 1.0)
            continue
        lo = float(col.min())
        hi = float(np.percentile(col, cfg.clip_percentile))
        if hi <= lo:
            # No spread (e.g. a single session): everything is "neutral".
            norm[f"{comp}_norm"] = 0.5
        else:
            norm[f"{comp}_norm"] = ((col - lo) / (hi - lo)).clip(0.0, 1.0)

    index = sum(weights[c] * norm[f"{c}_norm"] for c in _COMPONENTS)

    out = pd.concat([raw, norm], axis=1)
    out["engagement_index"] = index
    return out


def label_high_engagement(engagement_index, percentile: float = 75.0):
    """Binarize the engagement index into HIGH (1) / LOW (0).

    Args:
        engagement_index: A pandas Series (or array-like) of index values.
        percentile: Sessions at or above this percentile are HIGH engagement.

    Returns:
        ``(labels, threshold)`` where ``labels`` is an int Series/array (1 for
        HIGH, 0 for LOW) and ``threshold`` is the numeric cutoff used.
    """
    import numpy as np
    import pandas as pd

    values = np.asarray(engagement_index, dtype=float)
    threshold = float(np.percentile(values, percentile))
    labels = (values >= threshold).astype(int)
    if isinstance(engagement_index, pd.Series):
        labels = pd.Series(labels, index=engagement_index.index, name="high_engagement")
    return labels, threshold
