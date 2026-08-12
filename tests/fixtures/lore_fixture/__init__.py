"""A fake lore registry for tests.

Deliberately uses nonsense vocabulary so it can be committed publicly and can
never trip the private-leak scanner. Its only job is to exercise the gate,
renderer, and state machine. The unique MARKER lets the security test assert
that node bodies are absent before auth and present after.
"""

from __future__ import annotations

from app.components.lorekit import LoreRegistry, Node

MARKER = "XYZZY_FIXTURE_MARKER"
REVEAL_MARKER = "XYZZY_REVEAL_MARKER"
CLUE = "fixture-clue: harmless-pre-auth-hint"


def build_registry() -> LoreRegistry:
    nodes = [
        Node(id="home", kind="home", title="FIXTURE HOME", recognition_level=1,
             status="ok", body=("welcome to the fixture",)),
        Node(id="p1", kind="protocol", title="ALPHA_PROTOCOL", recognition_level=2,
             body=(MARKER, "alpha body")),
        Node(id="p2", kind="protocol", title="BETA_PROTOCOL", recognition_level=3,
             prerequisite=("p1",), body=(MARKER, "beta body")),
        Node(id="audit", kind="audit", title="FIXTURE AUDIT", recognition_level=5,
             min_protocols=2, body=("audit ready",)),
        Node(id="reveal", kind="reveal", title="FIXTURE REVEAL", recognition_level=5,
             prerequisite=("audit",), body=(REVEAL_MARKER, "the payoff")),
        Node(id="final", kind="final", title="FIXTURE FINAL", recognition_level=5,
             prerequisite=("reveal",), body=("end",)),
    ]
    return LoreRegistry(nodes, clue=CLUE)
