# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The four Hive-Mind stores.

All stores are pure data holders — write routing and conflict
detection live in the StoreCoordinator adapter.  No framework
dependencies, no persistence logic here.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from datetime import datetime
from typing import Any

from nlght.core.hive_mind.models import (
    PromotionStatus,
    SessionResult,
    TurnSummary,
    WorkingAtom,
)

# ---------------------------------------------------------------------------
# 1. ConversationStore — WHERE AM I IN THE CONVERSATION?
# ---------------------------------------------------------------------------

class ConversationStore:
    """Chronological turn history for the session.  Append-only."""

    def __init__(self) -> None:
        self._turns: list[TurnSummary] = []
        self._entity_index: dict[str, list[int]] = defaultdict(list)

    def append(self, turn: TurnSummary) -> None:
        self._turns.append(turn)
        for entity in turn.entities:
            self._entity_index[entity.lower()].append(turn.turn_nr)

    def get_turn(self, turn_nr: int) -> TurnSummary | None:
        for t in self._turns:
            if t.turn_nr == turn_nr:
                return t
        return None

    def last_n(self, n: int) -> list[TurnSummary]:
        return self._turns[-n:] if n <= len(self._turns) else list(self._turns)

    def resolve_reference(self, ref: str) -> TurnSummary | None:
        candidates = self._entity_index.get(ref.lower(), [])
        if not candidates:
            return self._turns[-1] if self._turns else None
        return self.get_turn(max(candidates))

    def find_by_entities(self, entities: list[str], limit: int = 5) -> list[TurnSummary]:
        matched_nrs: set[int] = set()
        for entity in entities:
            matched_nrs.update(self._entity_index.get(entity.lower(), []))
        sorted_nrs = sorted(matched_nrs)[-limit:]
        return [t for nr in sorted_nrs if (t := self.get_turn(nr)) is not None]

    @property
    def current_turn_nr(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# 3. SessionResultStore — WHAT DO I KNOW?
# ---------------------------------------------------------------------------

class SessionResultStore:
    """Holds finished, validated facts and artefacts for the session."""

    def __init__(self) -> None:
        self._results: dict[str, SessionResult] = {}
        self._entity_index: dict[str, list[str]] = defaultdict(list)
        self._tag_index: dict[str, list[str]] = defaultdict(list)

    def store(self, result: SessionResult) -> None:
        self._results[result.id] = result
        for entity in result.entities:
            self._entity_index[entity.lower()].append(result.id)
        for tag in result.tags:
            self._tag_index[tag.lower()].append(result.id)

    def get(self, result_id: str) -> SessionResult | None:
        return self._results.get(result_id)

    def find_by_entities(self, entities: list[str]) -> list[SessionResult]:
        matched_ids: set[str] = set()
        for entity in entities:
            matched_ids.update(self._entity_index.get(entity.lower(), []))
        results = [self._results[i] for i in matched_ids if i in self._results]
        return sorted(results, key=lambda r: r.created_at)

    def find_by_tags(self, tags: list[str]) -> list[SessionResult]:
        if not tags:
            return []
        id_sets = [set(self._tag_index.get(t.lower(), [])) for t in tags]
        matched_ids = id_sets[0].intersection(*id_sets[1:]) if id_sets else set()
        results = [self._results[i] for i in matched_ids if i in self._results]
        return sorted(results, key=lambda r: r.created_at)

    def exists(self, entities: list[str], tags: list[str] | None = None) -> bool:
        candidates = self.find_by_entities(entities)
        if not candidates:
            return False
        if not tags:
            return True
        return any(all(t in r.tags for t in tags) for r in candidates)

    def update_status(self, result_id: str, status: PromotionStatus) -> bool:
        if result_id in self._results:
            self._results[result_id].status = status
            return True
        return False

    def all_final(self) -> list[SessionResult]:
        return [r for r in self._results.values() if r.status == PromotionStatus.FINAL]


# ---------------------------------------------------------------------------
# 4. WorkingMemory — WHAT AM I DOING RIGHT NOW?
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Branch:
    label: str
    atoms: list[WorkingAtom] = dataclasses.field(default_factory=list)


class WorkingMemory:
    """Cascading branch-stack for transient turn state.

    Branches are opened per OODA step and merged back into the parent branch.
    Atoms marked ``promote_to_parent=True`` bubble up on merge; all others are
    discarded.  The Turn-Branch is the root; its promoted atoms are written to
    ``SessionResultStore`` by the Pipeline-Runner at turn end.

    Always snapshot-capable: ``snapshot()`` serialises the full stack at any
    point in time; ``from_snapshot()`` restores it.
    """

    def __init__(self) -> None:
        self._stack: list[_Branch] = []

    # ------------------------------------------------------------------
    # Branch lifecycle
    # ------------------------------------------------------------------

    def branch(self, label: str) -> None:
        """Open a new branch level (e.g. 'observe:abc123:4')."""
        self._stack.append(_Branch(label=label))

    def merge(self, promote: bool = True) -> list[WorkingAtom]:
        """Close the current branch.

        When ``promote`` is false, the branch is dropped and all atoms are
        discarded.

        When ``promote`` is true, atoms with ``promote_to_parent=True`` are
        moved into the parent branch (or returned to the caller when the
        Turn-Branch is merged). All other atoms are discarded.
        """
        if not self._stack:
            return []
        current  = self._stack.pop()
        if not promote:
            return []
        promoted = [a for a in current.atoms if a.promote_to_parent]
        if self._stack:
            self._stack[-1].atoms.extend(promoted)
        return promoted

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------

    def write(self, atom: WorkingAtom) -> None:
        """Write into the current (top) branch."""
        if not self._stack:
            raise RuntimeError("Cannot write WorkingAtom without an active WorkingMemory branch")
        self._stack[-1].atoms.append(atom)

    def read_active(self, atom_type: str | None = None) -> list[WorkingAtom]:
        """Atoms in the current (top) branch only."""
        if not self._stack:
            return []
        atoms = self._stack[-1].atoms
        if atom_type:
            return [a for a in atoms if a.atom_type == atom_type]
        return list(atoms)

    def read_all(self, atom_type: str | None = None) -> list[WorkingAtom]:
        """All atoms across the entire stack (Turn-Branch + all sub-branches)."""
        result: list[WorkingAtom] = []
        for b in self._stack:
            for a in b.atoms:
                if atom_type is None or a.atom_type == atom_type:
                    result.append(a)
        return result

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """Serialisable snapshot of the full branch stack.

        Safe to call at any point — even mid-step.
        """
        return [
            {
                "label": b.label,
                "atoms": [dataclasses.asdict(a) for a in b.atoms],
            }
            for b in self._stack
        ]

    @classmethod
    def from_snapshot(cls, data: list[dict[str, Any]]) -> WorkingMemory:
        """Restore a WorkingMemory from a ``snapshot()`` dict."""
        wm = cls()
        for branch_data in data:
            b = _Branch(label=branch_data["label"])
            for ad in branch_data["atoms"]:
                ad = dict(ad)
                raw_ts = ad.get("created_at")
                if isinstance(raw_ts, str):
                    ad["created_at"] = datetime.fromisoformat(raw_ts)
                b.atoms.append(WorkingAtom(**ad))
            wm._stack.append(b)
        return wm

    # ------------------------------------------------------------------
    # Compatibility / introspection
    # ------------------------------------------------------------------

    @property
    def active_task_id(self) -> str | None:
        """Label of the top branch; None when the stack is empty."""
        return self._stack[-1].label if self._stack else None

    @property
    def depth(self) -> int:
        """Current stack depth (0 = no active branch)."""
        return len(self._stack)
