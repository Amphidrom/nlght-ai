# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""FrozenStoreCoordinator — read-only replay wrapper with ephemeral shadow writes.

Wraps any StoreCoordinator for debug replay:

* All reads delegate to the wrapped inner coordinator (snapshot state).
* All writes go to an in-memory shadow that is discarded when the
  replay ends — the real session state is never modified.
* Slots use shadow-first access: set_slot() writes to shadow, get_slot()
  returns the shadow value if present, then falls back to inner.

This allows the full OODA pipeline to run against a snapshot state and
produce output without polluting the original session.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from nlght.core.hive_mind.models import (
    AtomType,
    ConflictReport,
    PromotionStatus,
    SessionResult,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
)
from nlght.ports.outbound.store_coordinator import StoreCoordinator

T = TypeVar("T")
logger = logging.getLogger(__name__)


class FrozenStoreCoordinator(StoreCoordinator):
    """Implements StoreCoordinator as a read-only replay wrapper.

    Reads from snapshot state (inner), writes only to ephemeral shadow.
    """

    def __init__(self, inner: StoreCoordinator) -> None:
        self._inner                       = inner
        self._shadow_slots:    dict[str, Any]      = {}
        self._shadow_atoms:    list[WorkingAtom]   = []
        self._shadow_results:  list[SessionResult] = []
        self._shadow_branches: list[str]           = []
        self._shadow_task:     str | None          = None

    # ------------------------------------------------------------------
    def record_turn(self, turn: TurnSummary) -> None:
        pass

    def get_recent_turns(self, n: int) -> list[TurnSummary]:
        return self._inner.get_recent_turns(n)

    # ------------------------------------------------------------------
    # Working Memory — shadow only
    # ------------------------------------------------------------------

    def write(self, intent: WriteIntent) -> tuple[str, ConflictReport | None]:
        atom = WorkingAtom(
            atom_type = intent.atom_type,
            content   = intent.content,
            task_id   = intent.task_id,
            entities  = intent.entities,
            tags      = intent.tags,
            key       = intent.key,
            kind      = intent.kind,
            promote_to_parent=intent.promote_immediately,
        )
        self._shadow_atoms.append(atom)
        return atom.id, None

    def open_branch(self, label: str) -> None:
        self._shadow_branches.append(label)
        self._shadow_task = label

    def close_branch(self, promote: bool = False) -> list[WorkingAtom]:
        if not self._shadow_branches:
            return []
        label = self._shadow_branches.pop()
        current_atoms = [a for a in self._shadow_atoms if a.task_id == label]
        self._shadow_atoms = [a for a in self._shadow_atoms if a.task_id != label]
        promoted = [a for a in current_atoms if a.promote_to_parent] if promote else []
        if promote and self._shadow_branches:
            parent = self._shadow_branches[-1]
            for atom in promoted:
                atom.task_id = parent
            self._shadow_atoms.extend(promoted)
        self._shadow_task = self._shadow_branches[-1] if self._shadow_branches else None
        return promoted

    def store_promoted_atoms(
        self,
        atoms: list[WorkingAtom],
        *,
        turn_nr: int,
        entities: list[str] | None = None,
    ) -> list[SessionResult]:
        stored: list[SessionResult] = []
        for atom in atoms:
            if atom.atom_type not in {AtomType.RESULT, AtomType.EVAL}:
                continue
            result = SessionResult(
                content=atom.content,
                entities=list(entities or []),
                turn_nr=turn_nr,
                status=PromotionStatus.FINAL,
            )
            self._shadow_results.append(result)
            stored.append(result)
        return stored

    def start_task(self, task_id: str) -> None:
        self.open_branch(task_id)

    def finish_task(
        self,
        task_id:  str,
        entities: list[str],
        turn_nr:  int = 0,
    ) -> list[SessionResult]:
        if self._shadow_task == task_id:
            self.close_branch(promote=False)
        else:
            self._shadow_atoms = [a for a in self._shadow_atoms if a.task_id != task_id]
        return []

    def discard_task(self, task_id: str) -> None:
        if self._shadow_task == task_id:
            self.close_branch(promote=False)
        else:
            self._shadow_atoms = [a for a in self._shadow_atoms if a.task_id != task_id]

    def fail_task(self, task_id: str) -> None:
        if self._shadow_task == task_id:
            self.close_branch(promote=False)
        else:
            self._shadow_atoms = [a for a in self._shadow_atoms if a.task_id != task_id]

    # ------------------------------------------------------------------
    # Read interface — delegates to inner
    # ------------------------------------------------------------------

    def get_context(self, entities: list[str]) -> dict[str, Any]:
        return self._inner.get_context(entities)

    def is_known(
        self,
        entities: list[str],
        tags:     list[str] | None = None,
    ) -> bool:
        return self._inner.is_known(entities, tags)

    def resolve_reference(self, ref: str) -> TurnSummary | None:
        return self._inner.resolve_reference(ref)

    # ------------------------------------------------------------------
    # Payload — shadow first, then inner
    # ------------------------------------------------------------------

    def promote_payload(
        self,
        content:  str,
        entities: list[str],
        tags:     list[str],
        turn_nr:  int,
    ) -> tuple[str, bool]:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        payload_tags = list(set(tags) | {"payload", f"hash:{content_hash}"})
        result = SessionResult(
            content      = content,
            entities     = entities,
            tags         = payload_tags,
            turn_nr      = turn_nr,
            status       = PromotionStatus.FINAL,
            content_hash = content_hash,
        )
        self._shadow_results.append(result)
        return result.id, True

    def get_payload(self, entities: list[str]) -> SessionResult | None:
        entity_set = {e.lower() for e in entities}
        for r in reversed(self._shadow_results):
            if "payload" in r.tags and entity_set & {e.lower() for e in r.entities}:
                return r
        return self._inner.get_payload(entities)

    # ------------------------------------------------------------------
    # Slots — shadow-first
    # ------------------------------------------------------------------

    def get_slot(self, key: str, default_factory: Callable[[], T]) -> T:
        if key in self._shadow_slots:
            return cast(T, self._shadow_slots[key])
        return self._inner.get_slot(key, default_factory)

    def set_slot(self, key: str, value: object) -> None:
        self._shadow_slots[key] = value
