"""The Rizzmatics personality layer.

Everything in this module is deliberately unserious. The math elsewhere is
real; this file is where the project puts on sunglasses. The one rule it takes
completely seriously is the Human Override: when asked whether someone likes
you, Rizzmatics refuses, because it is software.

Nothing here should ever be mistaken for a scientific claim. The Rizz
Coefficient™ in particular is made up. We are telling you that on purpose.
"""

from __future__ import annotations

from textwrap import dedent

__all__ = [
    "TAGLINE", "SUPPORTING_LINES", "RIZZ_DISCLAIMER",
    "boot_screen", "rizz_engine_readout", "rizz_coefficient",
    "oracle_card", "confidence_translation", "model_outcome_verdict",
    "dataset_health", "human_override", "things_rizzmatics_cannot_tell_you",
    "is_relationship_question", "final_moment",
]

TAGLINE = "Applied mathematics for completely unnecessary flirting."

SUPPORTING_LINES = [
    "Quantifying the vibes since nobody asked.",
    "Enterprise-grade overthinking.",
    "Because apparently flirting needed statistics.",
    'Turning "what does this mean?" into a feature vector.',
    'We put the "analysis" in overanalysis.',
]

RIZZ_DISCLAIMER = (
    "The Rizz Coefficient™ is not a scientifically recognized metric. "
    "We made it up because it sounded funny. It combines real behavioral "
    "signals using entirely arbitrary weights and should be trusted "
    "approximately never."
)


# --------------------------------------------------------------------------- #
# Boot screen
# --------------------------------------------------------------------------- #
def boot_screen() -> str:
    """The ceremonial startup banner. Pure theater."""
    return dedent("""\
        ╔══════════════════════════════════════════════╗
        ║                R I Z Z M A T I C S           ║
        ║                                              ║
        ║  Applied mathematics for completely          ║
        ║  unnecessary flirting.                       ║
        ╚══════════════════════════════════════════════╝

        Initializing Rizzmatics Engine...

        ✓ Parsing conversations
        ✓ Extracting behavioral signals
        ✓ Engineering features
        ✓ Training predictive models
        ✓ Running statistical analysis
        ✓ Quantifying the vibes
        ✓ Overthinking the overthinking

        SYSTEM STATUS
        ──────────────────────────────────────────────
        RIZZ:                 ONLINE
        STATISTICAL RIGOR:    QUESTIONABLE
        OVERENGINEERING:      ENTERPRISE-GRADE
        ACTUAL NECESSITY:     0%
        EMOTIONAL MATURITY:   PLEASE CONSULT USER
        ──────────────────────────────────────────────

        FINAL RECOMMENDATION:

                     TALK TO THE HUMAN.
        """)


# --------------------------------------------------------------------------- #
# The Rizz Engine™
# --------------------------------------------------------------------------- #
def rizz_engine_readout(features: dict) -> dict[str, float]:
    """Surface the (real) signals the Rizz Engine™ pretends to brood over."""
    def g(k: str, default: float = 0.0) -> float:
        v = features.get(k, default)
        return default if v is None or (isinstance(v, float) and v != v) else v

    return {
        "Response Latency (s)": round(g("median_response_latency_s"), 1),
        "Message Momentum (msgs)": round(g("n_messages"), 1),
        "Conversation Persistence (runs)": round(g("n_runs"), 1),
        "Participation Balance": round(g("participation_balance"), 3),
        "Question Density": round(g("question_rate"), 3),
        "Back-and-Forth Rate": round(g("back_and_forth_rate"), 3),
        "Emoji Density": round(g("emoji_density"), 3),
        "Late-Night Indicator": g("is_late_night"),
    }


def rizz_coefficient(features: dict) -> float:
    """A deliberately fictional 0–100 score. See :data:`RIZZ_DISCLAIMER`.

    Built from real signals with made-up weights. It has no meaning. It is
    approximately as scientific as a horoscope with a p-value.
    """
    def g(k: str, default: float = 0.0) -> float:
        v = features.get(k, default)
        return default if v is None or (isinstance(v, float) and v != v) else v

    score = 50.0
    score += 20.0 * g("back_and_forth_rate")
    score += 15.0 * g("participation_balance")
    score += 10.0 * min(g("question_rate"), 1.0)
    score += 8.0 * min(g("emoji_density") * 5, 1.0)
    # A snappier median reply nudges the (meaningless) number up.
    latency = g("median_response_latency_s", 600.0)
    score += 12.0 * max(0.0, 1.0 - min(latency / 600.0, 1.0))
    # One-sided monologues tank it.
    score -= 15.0 * max(0.0, g("max_participation_share") - 0.5) * 2
    return round(max(0.0, min(100.0, score)), 1)


# --------------------------------------------------------------------------- #
# The Rizzmatics Oracle™
# --------------------------------------------------------------------------- #
def oracle_card(label: str, confidence: float, interpretation: str) -> str:
    """Render the Oracle's forecast as an absurd enterprise console box."""
    pct = max(0.0, min(1.0, confidence))
    filled = int(round(pct * 20))
    bar = "█" * filled + "░" * (20 - filled)
    return dedent(f"""\
        ┌─────────────────────────────────────────────┐
        │          RIZZMATICS ORACLE™                 │
        ├─────────────────────────────────────────────┤
        │ Consulting predictive infrastructure...     │
        │                                             │
        │ {bar} {int(pct * 100):>3d}%             │
        │                                             │
        │ FORECAST                                    │
        │   {label:<41.41s} │
        │                                             │
        │ Confidence: {pct:>4.2f}                          │
        │                                             │
        │ Interpretation:                             │
        │   {interpretation:<41.41s} │
        │                                             │
        │ ⚠ The Oracle is a Random Forest.            │
        │   Please do not worship it.                 │
        └─────────────────────────────────────────────┘""")


# --------------------------------------------------------------------------- #
# "How Sure Are We, Bro?™"
# --------------------------------------------------------------------------- #
def confidence_translation(p: float) -> str:
    """Translate a legitimate probability into an illegitimate feeling."""
    p = max(0.0, min(1.0, p))
    if p >= 0.9:
        vibe = "Extremely sure. Suspiciously sure, honestly."
    elif p >= 0.75:
        vibe = "Pretty sure."
    elif p >= 0.6:
        vibe = "Sure-ish. A coin that went to college."
    elif p >= 0.5:
        vibe = "We are guessing, but with extra steps."
    else:
        vibe = "No idea. None whatsoever."
    return (
        f"Prediction confidence: {int(round(p * 100))}%\n\n"
        f"TRANSLATION:\n\n{vibe}\n\n"
        "But statistically speaking, we have absolutely no idea."
    )


# --------------------------------------------------------------------------- #
# Fun model outcomes
# --------------------------------------------------------------------------- #
def model_outcome_verdict(predicted_high: bool, actual_high: bool) -> tuple[str, str]:
    """Return a (headline, subtitle) roast for a prediction vs. reality."""
    if predicted_high and not actual_high:
        return ("THE MODEL GOT RIZZ-LED",
                "Predicted high engagement. Conversation subsequently died. Embarrassing.")
    if not predicted_high and actual_high:
        return ("THE MODEL GOT GHOSTED",
                "Predicted low engagement. It went three hours. We have learned nothing.")
    if predicted_high and actual_high:
        return ("THE MODEL COOKED",
                "Predicted high engagement and the conversation delivered. Suspicious.")
    return ("AS FORECAST: NOTHING",
            "Predicted low engagement. It was, in fact, giving nothing. Anticlimactic.")


def dataset_health(n_sessions: int) -> tuple[str, str]:
    """Return a (status, message) verdict on how well-fed the machine is."""
    if n_sessions < 15:
        return ("MALNOURISHED",
                f"BRO, YOU GAVE ME {n_sessions} CONVERSATIONS. "
                "I am an ML model, not a fortune teller.")
    if n_sessions < 40:
        return ("EDIBLE", "The machine will eat, but it is not thrilled.")
    if n_sessions <= 150:
        return ("JUICY", "DATASET HEALTH: JUICY. The machine is sufficiently fed.")
    return ("OVERFED", "DATASET HEALTH: OBESE. The machine has had enough, thank you.")


# --------------------------------------------------------------------------- #
# The Human Override™ — the one serious feature wearing a joke's clothes
# --------------------------------------------------------------------------- #
def human_override() -> str:
    """The defining feature. When asked what data cannot answer, refuse."""
    return dedent("""\
        I have analyzed 47 conversational features.

        I have consulted 5 models.

        I have performed unnecessary mathematics.

        And I still cannot answer that.

        WHY?

        Because I am software.

        Please communicate with the human.
        """)


def things_rizzmatics_cannot_tell_you() -> list[str]:
    """The honesty section, delivered as a bit."""
    return [
        "Does someone like you?",
        "Are they attracted to you?",
        "Are they secretly mad at you?",
        "Should you text them?",
        "Why did they take four hours to reply?",
        "Are you compatible?",
        "Is this a situationship?",
        "Are you cooked?",
        "Are you cooking?",
        'What did "haha" actually mean?',
    ]


_RELATIONSHIP_TRIGGERS = (
    "like me", "likes me", "into me", "attracted", "have feelings",
    "does she", "does he", "do they like", "am i cooked", "situationship",
    "should i text", "secretly", "have a crush", "into each other",
    "does my crush", "will they date", "compatible",
)


def is_relationship_question(text: str) -> bool:
    """Detect the questions that trigger the Human Override."""
    low = text.lower()
    return any(trigger in low for trigger in _RELATIONSHIP_TRIGGERS)


# --------------------------------------------------------------------------- #
# The Final Rizzmatics Moment™
# --------------------------------------------------------------------------- #
def final_moment(stats: dict) -> str:
    """The closing screen after all the unnecessary computation."""
    return dedent(f"""\
        ╔══════════════════════════════════════════════╗
        ║            RIZZMATICS ANALYSIS               ║
        ╚══════════════════════════════════════════════╝

        Messages analyzed:       {stats.get('messages', 0):,}
        Sessions detected:       {stats.get('sessions', 0):,}
        Features engineered:     {stats.get('features', 0)}
        Models evaluated:        {stats.get('models', 0)}
        Predictions generated:   {stats.get('predictions', 0):,}

        OVERENGINEERING:        100%
        ACTUAL NECESSITY:         0%

        FINAL QUESTION:

        Can we determine what the other person actually feels?

        NO.

        FINAL RECOMMENDATION:

                     TALK TO THE HUMAN.
        """)
