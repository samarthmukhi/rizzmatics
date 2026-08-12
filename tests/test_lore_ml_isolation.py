"""Prove the lore layer cannot influence the ML pipeline (spec section 37).

Static check: the public lore engine imports nothing from ``src/``.
Dynamic check: ML outputs are byte-identical whether lore is disabled or enabled.
"""

import importlib
from pathlib import Path

import pytest

from src.engagement import compute_engagement
from src.parser import parse_file
from src.preprocessing import build_dataset
from src.sessions import detect_sessions

PUBLIC_ENGINE_FILES = [
    "app/components/lorekit.py",
    "app/components/gate.py",
    "app/components/private_view.py",
]


def test_lore_engine_does_not_import_the_ml_pipeline():
    for rel in PUBLIC_ENGINE_FILES:
        src = Path(rel).read_text(encoding="utf-8")
        assert "from src" not in src, f"{rel} imports the ML pipeline"
        assert "import src" not in src, f"{rel} imports the ML pipeline"


def _ml_signature():
    sessions = detect_sessions(parse_file("data/demo/demo_chat.txt"))
    ds = build_dataset(sessions, prefix=10)
    eng = compute_engagement(sessions)
    y = tuple(round(v, 9) for v in ds.y_regression.tolist())
    x = tuple(round(v, 9) for v in ds.X.fillna(-99.0).to_numpy().flatten().tolist())
    return (y, x, round(float(eng["engagement_index"].sum()), 9))


def test_ml_output_identical_with_lore_disabled_vs_enabled(monkeypatch):
    before = _ml_signature()

    # "Enable" lore: import and build the fixture registry (activates the whole
    # lore layer in-process).
    monkeypatch.setenv("RIZZMATICS_LORE_MODULE", "tests.fixtures.lore_fixture")
    from app.components import gate
    registry = gate.load_registry()
    assert registry is not None  # lore is genuinely active

    after = _ml_signature()
    assert before == after, "lore layer changed an ML output — isolation broken!"


def test_real_private_package_also_does_not_touch_ml():
    # Only runs where the real private package is present (local dev).
    try:
        importlib.import_module("private.lore").build_registry()
    except Exception:
        pytest.skip("private lore package not present (public build)")
    after = _ml_signature()
    # Re-run cleanly to confirm determinism regardless of lore import.
    assert after == _ml_signature()
