"""Phase 21 — Research Lab integration test (fast, no Streamlit runtime).

Exercises the cached research-runner the dashboard calls, asserting the whole
battery composes and that the sanity controls still collapse when driven through
the app layer (not just the src.research modules directly).
"""

import warnings

import pytest

warnings.filterwarnings("ignore")

from app.components.pipeline import load_demo_text, run_research

DEFAULT_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)


@pytest.fixture(scope="module")
def research():
    return run_research(
        load_demo_text(), inactivity_hours=6.0, prefix=10,
        weights=DEFAULT_WEIGHTS, high_percentile=75.0, dayfirst=None, n_repeats=1,
    )


def test_research_returns_all_sections(research):
    for key in ("sanity", "ablation_num", "ablation_disp", "prefix_num",
                "logo_num", "n_samples", "caveat"):
        assert key in research


def test_dashboard_sanity_controls_collapse(research):
    s = research["sanity"].set_index("condition")["ROC_AUC"]
    assert s["Normal (real target)"] > 0.7
    assert s["Shuffled target"] < 0.65
    assert s["Null control (noise)"] < 0.65


def test_dashboard_flags_small_sample(research):
    # 31 sessions is below the stability threshold; the caveat must be present.
    assert research["caveat"] is not None
    assert "SMALL SAMPLE" in research["caveat"]


def test_prefix_table_has_sample_sizes(research):
    assert "n_samples" in research["prefix_num"].columns
    assert (research["prefix_num"]["n_samples"] > 0).all()


def test_research_reports_insufficient_data_gracefully():
    # A tiny chat can't be modeled; the runner must return an honest error, not crash.
    tiny = (
        "12/08/2026, 14:00 - Alex: hi\n"
        "12/08/2026, 14:01 - Sam: yo\n"
    )
    out = run_research(tiny, inactivity_hours=6.0, prefix=20,
                       weights=DEFAULT_WEIGHTS, high_percentile=75.0,
                       dayfirst=None, n_repeats=1)
    assert "error" in out
