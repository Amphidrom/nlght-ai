# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""SimpleStoreCoordinator — in-memory session memory for the free tier.

No external dependencies.  Sessions are kept in a process-level dict
with TTL-based eviction.  On process restart all session state is lost.

For persistent session memory use HiveMindStoreCoordinator instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

from nlght.core.hive_mind.models import (
    AtomType,
    PromotionStatus,
    SessionResult,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
)
from nlght.ports.outbound.store_coordinator import (
    ArtifactContent,
    StoreCoordinator,
    StoreCoordinatorFactory,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

_PROMOTABLE = {AtomType.RESULT, AtomType.EVAL}


class SimpleStoreCoordinator(StoreCoordinator):
    """Flat in-memory implementation of the StoreCoordinator port.

    Intentionally minimal — no entity indexing, no relevance scoring,
    no conflict detection, no task-graph.  Designed to be fast and
    dependency-free.

    * ConversationStore: plain append-only list.
    * SessionResultStore: flat list, no indexing.
    * WorkingMemory: dict keyed by task_id.
    * ``get_context``: returns last 5 turns + last 10 results.
    """

    def __init__(self) -> None:
        self._turns: list[TurnSummary] = []
        self._results: list[SessionResult] = []
        self._working: dict[str, list[WorkingAtom]] = {}
        self._branch_stack: list[str] = []
        self._active_task: str | None = None
        self._slots: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def record_turn(self, turn: TurnSummary) -> None:
        self._turns.append(turn)

    def get_recent_turns(self, n: int) -> list[TurnSummary]:
        return self._turns[-n:] if n <= len(self._turns) else list(self._turns)

    def next_turn_nr(self) -> int:
        return len(self._turns) + 1

    # ------------------------------------------------------------------
    # Working Memory + Promotion
    # ------------------------------------------------------------------

    def write(self, intent: WriteIntent) -> tuple[str, None]:
        # Match HiveMind coordinator semantics while a branch stack is active:
        # promotable atoms remain in working memory and bubble to the parent
        # branch on merge. Only branch-less writes fall back to immediate
        # SessionResult storage because there is no later merge/store step.
        immediate = (
            intent.promote_immediately
            and intent.atom_type in _PROMOTABLE
            and not self._branch_stack
        )
        atom = WorkingAtom(
            atom_type         = intent.atom_type,
            content           = intent.content,
            task_id           = intent.task_id,
            entities          = intent.entities,
            tags              = intent.tags,
            key               = intent.key,
            kind              = intent.kind,
            promote_to_parent = intent.promote_immediately,
        )
        target_task = self._branch_stack[-1] if self._branch_stack else intent.task_id
        self._working.setdefault(target_task, []).append(atom)
        if immediate:
            self._results.append(SessionResult(
                id       = atom.id,
                content  = atom.content,
                entities = list(intent.entities),
                tags     = list(intent.tags),
                turn_nr  = intent.turn_nr,
                status   = PromotionStatus.FINAL,
            ))
        return atom.id, None

    def open_branch(self, label: str) -> None:
        self._branch_stack.append(label)
        self._working.setdefault(label, [])
        self._active_task = label

    def close_branch(self, promote: bool = False) -> list[WorkingAtom]:
        if not self._branch_stack:
            return []
        label = self._branch_stack.pop()
        current_atoms = self._working.pop(label, [])
        if promote and self._branch_stack:
            parent = self._branch_stack[-1]
            promoted = [atom for atom in current_atoms if atom.promote_to_parent]
            self._working.setdefault(parent, []).extend(promoted)
        else:
            promoted = [atom for atom in current_atoms if atom.promote_to_parent] if promote else []
        self._active_task = self._branch_stack[-1] if self._branch_stack else None
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
            if atom.atom_type not in _PROMOTABLE:
                continue
            result = SessionResult(
                content=atom.content,
                entities=list(entities or []),
                tags=list(atom.tags),
                turn_nr=turn_nr,
                status=PromotionStatus.FINAL,
            )
            self._results.append(result)
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
        atoms = self._working.get(task_id, [])
        promoted: list[SessionResult] = []
        for atom in atoms:
            if atom.atom_type in _PROMOTABLE:
                result = SessionResult(
                    content  = atom.content,
                    entities = entities,
                    turn_nr  = turn_nr,
                    status   = PromotionStatus.FINAL,
                )
                self._results.append(result)
                promoted.append(result)
        if self._active_task == task_id:
            self.close_branch(promote=False)
        return promoted

    def discard_task(self, task_id: str) -> None:
        if self._active_task == task_id:
            self.close_branch(promote=False)
        else:
            self._working.pop(task_id, None)

    def fail_task(self, task_id: str) -> None:
        if self._active_task == task_id:
            self.close_branch(promote=False)
        else:
            self._working.pop(task_id, None)

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def get_context(self, entities: list[str]) -> dict[str, Any]:
        active_atoms: list[WorkingAtom] = []
        for label in self._branch_stack:
            active_atoms.extend(self._working.get(label, []))
        # Empty entity list means "no filter" — return all stored results.
        if entities:
            entity_set = {e.lower() for e in entities}
            known = [
                r for r in self._results
                if any(en.lower() in entity_set for en in r.entities)
            ]
        else:
            known = list(self._results)
        return {
            "recent_turns":  self._turns[-5:],
            "known_results": known,
            "active_atoms":  active_atoms,
        }

    def is_known(
        self,
        entities: list[str],
        tags:     list[str] | None = None,
    ) -> bool:
        for result in self._results:
            entity_match = any(
                e.lower() in result.content.lower() for e in entities
            )
            if not entity_match:
                continue
            if not tags:
                return True
            if all(t in result.tags for t in tags):
                return True
        return False

    def resolve_reference(self, ref: str) -> TurnSummary | None:
        return None

    def promote_payload(
        self,
        content:  str,
        entities: list[str],
        tags:     list[str],
        turn_nr:  int,
    ) -> tuple[str, bool]:
        result = SessionResult(
            content  = content,
            entities = entities,
            tags     = list(set(tags) | {"payload"}),
            turn_nr  = turn_nr,
            status   = PromotionStatus.FINAL,
        )
        self._results.append(result)
        return result.id, True

    def get_payload(self, entities: list[str]) -> SessionResult | None:
        candidates = [
            r for r in self._results
            if "payload" in r.tags and r.status == PromotionStatus.FINAL
        ]
        return candidates[-1] if candidates else None

    def load_artifact(self, artifact_id: str) -> ArtifactContent | None:
        clean = artifact_id.strip()
        if not clean or "/" in clean or "\\" in clean or ".." in clean:
            logger.warning("simple_store.load_artifact.rejected | id=%r (unsafe)", clean)
            return None
        workspace = self._slots.get("workspace")
        if workspace is None:
            return None
        artifacts_dir: Path = workspace.root_path / "memory_artifacts"
        if not artifacts_dir.is_dir():
            return None
        matches = list(artifacts_dir.glob(f"{clean}*"))
        if not matches:
            return None
        path = matches[0]
        try:
            return ArtifactContent(content=path.read_text(encoding="utf-8"), extension=path.suffix)
        except OSError as exc:
            logger.warning("simple_store.load_artifact.failed | id=%s error=%s", clean, exc)
            return None

    # ------------------------------------------------------------------
    # Slot API
    # ------------------------------------------------------------------

    def _dump(self, checkpoint_name: str) -> None:
        logger.debug(
            "\n████████████████████████████████████████████████\n"
            "STORE COORDINATOR CHECKPOINT: %s\n"
            "████████████████████████████████████████████████",
            checkpoint_name,
        )

        recent_turns = self._turns[-4:]
        if recent_turns:
            logger.debug("CONVERSATION last %d turns:", len(recent_turns))
            for turn in recent_turns:
                logger.debug(
                    "  [%d] intent=%s entities=%s",
                    turn.turn_nr, turn.intent, turn.entities,
                )
        else:
            logger.debug("CONVERSATION: (empty)")

        final_results = [r for r in self._results if r.status == PromotionStatus.FINAL]
        if final_results:
            logger.debug("SESSION RESULTS (%d):", len(final_results))
            for r in final_results:
                logger.debug(
                    "  [%s] entities=%s tags=%s | %s",
                    r.id[:8], r.entities, r.tags,
                    r.content.replace("\n", " "),
                )
        else:
            logger.debug("SESSION RESULTS: (empty)")

        if self._active_task:
            atoms = self._working.get(self._active_task, [])
            logger.debug("WORKING MEMORY task=%s (%d atoms):", self._active_task, len(atoms))
            for atom in atoms:
                logger.debug(
                    "  [%s] %s | %s",
                    atom.atom_type, atom.id[:8],
                    atom.content.replace("\n", " "),
                )
        else:
            logger.debug("WORKING MEMORY: (no active task)")

        logger.debug("████████████████████████████████████████████████\n")

    def get_slot(self, key: str, default_factory: Callable[[], T]) -> T:
        if key not in self._slots:
            self._slots[key] = default_factory()
        return cast(T, self._slots[key])

    def set_slot(self, key: str, value: object) -> None:
        self._slots[key] = value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class SimpleStoreCoordinatorFactory(StoreCoordinatorFactory):
    """Process-local factory with TTL-based session eviction.

    Sessions are kept in memory until they exceed ``session_ttl_seconds``
    of inactivity.  On each ``get_or_create`` call, expired sessions are
    lazily evicted.

    ``save`` updates the last-accessed timestamp — no serialisation.
    """

    def __init__(self, session_ttl_seconds: int = 3600) -> None:
        self._ttl = timedelta(seconds=session_ttl_seconds)
        self._sessions: dict[str, tuple[SimpleStoreCoordinator, datetime]] = {}
        #: Owners, kept beside the sessions and evicted with them. This factory
        #: never touches a backend, so ownership recorded anywhere else would
        #: not exist for it.
        self._owners: dict[str, str] = {}
        logger.info(
            "simple_store.factory.init | ttl_seconds=%d", session_ttl_seconds,
        )

    def exists(self, session_id: str) -> bool:
        self._evict_expired()
        return session_id in self._sessions

    def owner_of(self, session_id: str) -> str | None:
        self._evict_expired()
        return self._owners.get(session_id) or None

    def claim_ownership(self, session_id: str, owner_principal_id: str) -> None:
        self._owners.setdefault(session_id, owner_principal_id)

    def get_or_create(self, session_id: str) -> SimpleStoreCoordinator:
        self._evict_expired()
        if session_id in self._sessions:
            coordinator, _ = self._sessions[session_id]
            self._sessions[session_id] = (coordinator, datetime.now(UTC))
            logger.debug("simple_store.session.reused | id=%s", session_id)
            return coordinator
        coordinator = SimpleStoreCoordinator()
        self._sessions[session_id] = (coordinator, datetime.now(UTC))
        logger.info("simple_store.session.created | id=%s", session_id)
        return coordinator

    # Narrower than StoreCoordinatorFactory.save(coordinator: StoreCoordinator) --
    # this factory only ever hands out and receives back its own coordinator
    # type (enforced by construction in bootstrap/wiring.py, one factory per
    # coordinator kind), never a StoreCoordinator of a different concrete type.
    def save(self, session_id: str, coordinator: SimpleStoreCoordinator) -> None:  # type: ignore[override]
        # In-memory: just refresh the last-accessed timestamp.
        self._sessions[session_id] = (coordinator, datetime.now(UTC))

    def _evict_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [
            k for k, (_, last) in self._sessions.items()
            if now - last > self._ttl
        ]
        for k in expired:
            del self._sessions[k]
            # The owner goes with the session. Leaving it behind would make an
            # evicted id un-creatable by anybody else, which is a denial of
            # service dressed as a security record.
            self._owners.pop(k, None)
            logger.debug("simple_store.session.evicted | id=%s", k)
