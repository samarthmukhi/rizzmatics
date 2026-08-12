"""Lore engine — the generic narrative state machine.

This is public code and contains **no private content whatsoever**. It knows how
to hold a set of opaque "nodes", track which have been discovered, and decide
which become available next. The actual mythology (titles, copy, protocol names)
is injected at runtime by a separate, git-ignored package (or, in tests, a
fixture). The renderer draws whatever nodes it is handed.

Design contract: this module imports nothing from ``src/`` (the ML pipeline).
The lore layer can never touch features, labels, predictions, or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["Node", "LoreRegistry", "NarrativeState"]


@dataclass(frozen=True)
class Node:
    """One node in the discovery journey (a protocol, the reveal, etc.).

    Attributes:
        id: Stable identifier.
        title: Display title (private copy — supplied by the lore package).
        kind: One of ``home | protocol | issues | changelog | audit | reveal |
            final``. Drives how the renderer styles it; the engine treats all
            nodes uniformly except for the ``min_protocols`` gate.
        recognition_level: 0–5, the "how deep is this reference" tier. Used only
            to display progress; it does not gate access.
        status: Short system-status string shown in the node header.
        metadata: Ordered technical key/values rendered as a panel.
        body: Lines of renderable content.
        prerequisite: Node ids that must be visited before this one unlocks.
        min_protocols: If > 0, this node unlocks only after at least this many
            ``protocol`` nodes have been visited (used to gate the audit/reveal
            so the payoff never arrives early).
    """

    id: str
    title: str
    kind: str = "protocol"
    recognition_level: int = 0
    status: str = ""
    metadata: dict = field(default_factory=dict)
    body: tuple[str, ...] = ()
    prerequisite: tuple[str, ...] = ()
    min_protocols: int = 0


class LoreRegistry:
    """An ordered collection of nodes plus the door clue.

    The registry is built and returned by the private lore package; the public
    code only ever sees this generic container.
    """

    def __init__(self, nodes: Iterable[Node], *, clue: str = "") -> None:
        self._nodes: list[Node] = list(nodes)
        self._by_id: dict[str, Node] = {}
        for n in self._nodes:
            if n.id in self._by_id:
                raise ValueError(f"Duplicate node id: {n.id}")
            self._by_id[n.id] = n
        self._clue = clue

    @property
    def clue(self) -> str:
        """The pre-auth door hint. Deliberately cryptic; reveals no lore."""
        return self._clue

    def all(self) -> list[Node]:
        return list(self._nodes)

    def get(self, node_id: str) -> Node:
        return self._by_id[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._by_id

    def kind(self, kind: str) -> list[Node]:
        return [n for n in self._nodes if n.kind == kind]

    def home_id(self) -> str:
        homes = self.kind("home")
        return homes[0].id if homes else self._nodes[0].id


class NarrativeState:
    """Mutable progress through the journey (lives in st.session_state)."""

    def __init__(self, current: str | None = None) -> None:
        self.visited: set[str] = set()
        self.current: str | None = current

    # ---- progress ------------------------------------------------------- #
    def visit(self, node_id: str) -> None:
        self.visited.add(node_id)
        self.current = node_id

    def n_protocols_visited(self, registry: LoreRegistry) -> int:
        return sum(
            1 for nid in self.visited
            if nid in registry and registry.get(nid).kind == "protocol"
        )

    def recognition_level(self, registry: LoreRegistry) -> int:
        levels = [registry.get(nid).recognition_level
                  for nid in self.visited if nid in registry]
        return max(levels) if levels else 0

    # ---- availability --------------------------------------------------- #
    def is_available(self, node: Node, registry: LoreRegistry) -> bool:
        """A node unlocks once its prerequisites and protocol gate are met."""
        if not set(node.prerequisite) <= self.visited:
            return False
        if node.min_protocols and self.n_protocols_visited(registry) < node.min_protocols:
            return False
        return True

    def available_nodes(self, registry: LoreRegistry) -> list[Node]:
        return [n for n in registry.all() if self.is_available(n, registry)]

    def is_locked(self, node: Node, registry: LoreRegistry) -> bool:
        return not self.is_available(node, registry)
