"""Unit tests for the generic narrative state machine (public engine)."""

import pytest

from app.components.lorekit import LoreRegistry, NarrativeState, Node
from tests.fixtures.lore_fixture import build_registry


@pytest.fixture
def registry():
    return build_registry()


def test_registry_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        LoreRegistry([Node(id="x", title="A"), Node(id="x", title="B")])


def test_home_and_kinds(registry):
    assert registry.home_id() == "home"
    assert len(registry.kind("protocol")) == 2
    assert registry.kind("reveal")[0].id == "reveal"


def test_prerequisites_gate_availability(registry):
    state = NarrativeState()
    state.visit("home")
    # p1 has no prereq -> available; p2 requires p1 -> locked until p1 visited.
    assert state.is_available(registry.get("p1"), registry)
    assert not state.is_available(registry.get("p2"), registry)
    state.visit("p1")
    assert state.is_available(registry.get("p2"), registry)


def test_min_protocols_gate_blocks_audit_until_enough_visited(registry):
    state = NarrativeState()
    state.visit("home")
    assert not state.is_available(registry.get("audit"), registry)  # 0 protocols
    state.visit("p1")
    assert not state.is_available(registry.get("audit"), registry)  # 1 protocol
    state.visit("p2")
    assert state.is_available(registry.get("audit"), registry)      # 2 -> unlocked


def test_reveal_and_final_are_gated_in_sequence(registry):
    state = NarrativeState()
    for nid in ("home", "p1", "p2"):
        state.visit(nid)
    assert not state.is_available(registry.get("reveal"), registry)  # needs audit
    state.visit("audit")
    assert state.is_available(registry.get("reveal"), registry)
    assert not state.is_available(registry.get("final"), registry)   # needs reveal
    state.visit("reveal")
    assert state.is_available(registry.get("final"), registry)


def test_recognition_level_tracks_deepest_visited(registry):
    state = NarrativeState()
    state.visit("home")            # level 1
    assert state.recognition_level(registry) == 1
    state.visit("p2")             # level 3
    assert state.recognition_level(registry) == 3


def test_reveal_not_reachable_early(registry):
    # The payoff must never be available from the home screen.
    state = NarrativeState()
    state.visit("home")
    available = {n.id for n in state.available_nodes(registry)}
    assert "reveal" not in available
    assert "final" not in available
