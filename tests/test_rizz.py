"""Tests for the personality layer.

Yes, we unit-test the jokes. The Human Override is a real feature and the
Rizz Coefficient must stay inside its made-up bounds.
"""

import pytest

from src.rizz import (
    RIZZ_DISCLAIMER,
    boot_screen,
    confidence_translation,
    dataset_health,
    final_moment,
    human_override,
    is_relationship_question,
    model_outcome_verdict,
    oracle_card,
    rizz_coefficient,
    rizz_engine_readout,
    things_rizzmatics_cannot_tell_you,
)


def test_boot_screen_has_the_final_recommendation():
    assert "TALK TO THE HUMAN." in boot_screen()


def test_rizz_coefficient_stays_in_0_100():
    for feats in (
        {},
        {"back_and_forth_rate": 1.0, "participation_balance": 1.0,
         "question_rate": 1.0, "emoji_density": 1.0,
         "median_response_latency_s": 0.0, "max_participation_share": 0.5},
        {"max_participation_share": 1.0, "median_response_latency_s": 10000.0},
    ):
        score = rizz_coefficient(feats)
        assert 0.0 <= score <= 100.0


def test_rizz_coefficient_balanced_beats_one_sided():
    balanced = rizz_coefficient({
        "back_and_forth_rate": 0.9, "participation_balance": 1.0,
        "max_participation_share": 0.5, "median_response_latency_s": 60,
    })
    one_sided = rizz_coefficient({
        "back_and_forth_rate": 0.1, "participation_balance": 0.0,
        "max_participation_share": 0.95, "median_response_latency_s": 3000,
    })
    assert balanced > one_sided


def test_rizz_engine_readout_handles_nan_and_missing():
    readout = rizz_engine_readout({"n_messages": float("nan")})
    assert readout["Message Momentum (msgs)"] == 0.0
    assert "Participation Balance" in readout


def test_rizz_disclaimer_admits_its_made_up():
    assert "made" in RIZZ_DISCLAIMER.lower()


# --------------------------------------------------------------------------- #
# Human Override
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "does she like me?",
    "do they like me back",
    "am i cooked",
    "is this a situationship",
    "should i text her first",
    "are we compatible",
])
def test_relationship_questions_trigger_override(question):
    assert is_relationship_question(question) is True


@pytest.mark.parametrize("question", [
    "how many messages did we send",
    "what's the average response time",
    "which session was longest",
])
def test_analytical_questions_do_not_trigger_override(question):
    assert is_relationship_question(question) is False


def test_human_override_refuses_and_redirects():
    text = human_override()
    assert "Because I am software." in text
    assert "Please communicate with the human." in text


def test_things_we_cannot_tell_you_is_a_nonempty_list():
    items = things_rizzmatics_cannot_tell_you()
    assert len(items) >= 8
    assert any("like you" in i for i in items)


# --------------------------------------------------------------------------- #
# Verdicts, health, translations
# --------------------------------------------------------------------------- #
def test_model_outcome_verdicts():
    assert model_outcome_verdict(True, False)[0] == "THE MODEL GOT RIZZ-LED"
    assert model_outcome_verdict(False, True)[0] == "THE MODEL GOT GHOSTED"
    assert model_outcome_verdict(True, True)[0] == "THE MODEL COOKED"
    assert model_outcome_verdict(False, False)[0] == "AS FORECAST: NOTHING"


def test_dataset_health_tiers():
    assert dataset_health(10)[0] == "MALNOURISHED"
    assert dataset_health(25)[0] == "EDIBLE"
    assert dataset_health(80)[0] == "JUICY"
    assert dataset_health(500)[0] == "OVERFED"
    assert "14 CONVERSATIONS" in dataset_health(14)[1]


def test_confidence_translation_scales():
    assert "Extremely sure" in confidence_translation(0.95)
    assert "No idea" in confidence_translation(0.2)
    assert "absolutely no idea" in confidence_translation(0.8)


def test_oracle_card_renders_label_and_confidence():
    card = oracle_card("HIGH CONVERSATIONAL ENGAGEMENT", 0.73, "Something is happening.")
    assert "RIZZMATICS ORACLE" in card
    assert "73%" in card
    assert "Random Forest" in card


def test_final_moment_talks_to_the_human():
    text = final_moment({"messages": 8421, "sessions": 613, "features": 27,
                         "models": 5, "predictions": 184})
    assert "TALK TO THE HUMAN." in text
    assert "8,421" in text
