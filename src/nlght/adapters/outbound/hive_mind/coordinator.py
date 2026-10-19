# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Provide configured filesystem-backed session coordination.

Session snapshots use the configured backend.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from nlght.adapters.outbound.hive_mind.origin import LedgerStamp
from nlght.adapters.outbound.hive_mind.persistence import FileSystemBackend
from nlght.core.hive_mind.models import (
    AtomType,
    ConflictReport,
    MentalModelCache,
    PromotionStatus,
    SessionResult,
    SessionSnapshot,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
    _slots_to_json,
)
from nlght.core.hive_mind.stores import (
    ConversationStore,
    SessionResultStore,
    WorkingMemory,
)
from nlght.ports.outbound.session_backend import SessionBackend
from nlght.ports.outbound.store_coordinator import (
    ArtifactContent,
    StoreCoordinator,
    StoreCoordinatorFactory,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

_PROMOTABLE = {AtomType.RESULT, AtomType.EVAL}
_DISCARDED  = {AtomType.SPEC, AtomType.PLAN, AtomType.ERROR}


class HiveMindStoreCoordinator(LedgerStamp, StoreCoordinator):
    """Full StoreCoordinator with entity indexing, conflict detection,
    task lifecycle, and payload promotion.

    Implements StoreCoordinator as the sole writer into all four stores.
    Steps never write directly.
    """

    def __init__(
        self,
        on_invalidate: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.conversation        = ConversationStore()
        self.results             = SessionResultStore()
        self.working             = WorkingMemory()
        self._slots:             dict[str, Any] = {}
        self._on_invalidate      = on_invalidate or (lambda entities: None)
        self.mental_model_cache  = MentalModelCache()

    def record_turn(self, turn: TurnSummary) -> None:
        turn.origin = self._stamp(turn.origin)
        self.conversation.append(turn)
        logger.debug(
            "hive_mind.turn.recorded | turn_nr=%d intent=%s entities=%s",
            turn.turn_nr, turn.intent, turn.entities,
        )

    def get_recent_turns(self, n: int) -> list[TurnSummary]:
        return self.conversation.last_n(n)

    def next_turn_nr(self) -> int:
        return self.conversation.current_turn_nr + 1


    def write(self, intent: WriteIntent) -> tuple[str, ConflictReport | None]:
        atom = WorkingAtom(
            atom_type         = intent.atom_type,
            content           = intent.content,
            task_id           = intent.task_id,
            entities          = intent.entities,
            tags              = intent.tags,
            key               = intent.key,
            kind              = intent.kind,
            promote_to_parent = intent.promote_immediately,
            origin            = self._stamp(),
        )
        self.working.write(atom)
        return atom.id, None


    def open_branch(self, label: str) -> None:
        """Open a working-memory branch for a pipeline step.

        Args:
            label (str): Name used in branch diagnostics.

        Side Effects:
            Pushes a branch onto working memory and logs its depth.
        """
        self.working.branch(label)
        logger.debug(
            "hive_mind.branch.open | label=%s depth=%d", label, self.working.depth,
        )

    def close_branch(self, promote: bool = False) -> list[WorkingAtom]:
        """Merge or discard the current working-memory branch.

        Args:
            promote (bool): Whether marked atoms enter the parent branch.

        Returns:
            list[WorkingAtom]: Atoms promoted from the closed branch.

        Side Effects:
            Removes the current branch and logs its resulting depth.
        """
        promoted = self.working.merge(promote=promote)
        logger.debug(
            "hive_mind.branch.close | promote=%s promoted=%d depth=%d",
            promote,
            len(promoted), self.working.depth,
        )
        return promoted

    def store_promoted_atoms(
        self,
        atoms: list[WorkingAtom],
        *,
        turn_nr: int,
        entities: list[str] | None = None,
    ) -> list[SessionResult]:
        storable = {AtomType.RESULT, AtomType.EVAL}
        stored: list[SessionResult] = []
        for atom in atoms:
            if atom.atom_type not in storable:
                continue
            result = SessionResult(
                content=atom.content,
                entities=list(entities or []),
                tags=list(atom.tags),
                turn_nr=turn_nr,
                status=PromotionStatus.FINAL,
                origin=self._stamp(atom.origin),
            )
            self.results.store(result)
            stored.append(result)
        if stored:
            self._on_invalidate(list(entities or []))
        return stored


    def start_task(self, task_id: str) -> None:
        self.working.branch(task_id)
        logger.info("hive_mind.task.start | id=%s", task_id)

    def finish_task(
        self,
        task_id:  str,
        entities: list[str],
        turn_nr:  int = 0,
    ) -> list[SessionResult]:
        current_atoms = list(self.working.read_active())
        self.working.merge(promote=True)
        promoted: list[SessionResult] = []
        for atom in current_atoms:
            if atom.atom_type in _PROMOTABLE:
                result = SessionResult(
                    content  = atom.content,
                    entities = entities,
                    turn_nr  = turn_nr,
                    status   = PromotionStatus.FINAL,
                    origin   = self._stamp(atom.origin),
                )
                self.results.store(result)
                promoted.append(result)
                logger.info(
                    "hive_mind.atom.promoted | type=%s id=%s entities=%s",
                    atom.atom_type, result.id, entities,
                )
            else:
                logger.debug(
                    "hive_mind.atom.discarded | type=%s task=%s",
                    atom.atom_type, task_id,
                )
        if promoted:
            self._on_invalidate(entities)
        return promoted


    def get_context(self, entities: list[str]) -> dict[str, Any]:
        # Treat an empty entity list as no filter to include all FINAL results.
        # Entity-based retrieval only applies when the caller has specific entities.
        known = (
            self.results.all_final()
            if not entities
            else self.results.find_by_entities(entities)
        )
        return {
            "recent_turns":  self.conversation.last_n(5),
            "known_results": known,
            "active_atoms":  self.working.read_all(),
        }

    def is_known(
        self,
        entities: list[str],
        tags:     list[str] | None = None,
    ) -> bool:
        return self.results.exists(entities, tags)

    def discard_task(self, task_id: str) -> None:
        """Discard the current task branch without promotion.

        Args:
            task_id (str): Task ID used in the discard log.

        Side Effects:
            Closes the branch and logs its discarded atom count.
        """
        dropped_count = len(self.working.read_active())
        self.working.merge(promote=False)
        logger.debug(
            "hive_mind.task.discard | id=%s atoms_dropped=%d", task_id, dropped_count,
        )

    def fail_task(self, task_id: str) -> None:
        """Close a failed task branch without promotion.

        Args:
            task_id (str): Task ID used in the failure log.

        Side Effects:
            Closes the branch and logs its discarded atom count as a warning.
        """
        dropped_count = len(self.working.read_active())
        self.working.merge(promote=False)
        logger.warning(
            "hive_mind.task.fail | id=%s atoms_dropped=%d", task_id, dropped_count,
        )


    def get_slot(self, key: str, default_factory: Callable[[], T]) -> T:
        if key not in self._slots:
            self._slots[key] = default_factory()
        return cast(T, self._slots[key])

    def set_slot(self, key: str, value: object) -> None:
        self._slots[key] = value


    def promote_payload(
        self,
        content:  str,
        entities: list[str],
        tags:     list[str],
        turn_nr:  int,
    ) -> tuple[str, bool]:
        """Promote a payload into the session result store once per hash.

        Args:
            content (str): Payload text to store.
            entities (list[str]): Entities used to find an existing payload.
            tags (list[str]): Additional tags attached to a stored result.
            turn_nr (int): Conversation turn associated with the payload.

        Returns:
            tuple[str, bool]: Result ID and whether a new payload was stored.

        Side Effects:
            Stores and logs a new session result unless the latest payload for
            these entities has the same content hash.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        payload_tags = list(set(tags) | {"payload", f"hash:{content_hash}"})

        existing = [
            r for r in self.results.find_by_entities(entities)
            if "payload" in r.tags
        ]

        if existing:
            latest = sorted(existing, key=lambda r: r.created_at)[-1]
            if f"hash:{content_hash}" in latest.tags:
                logger.debug(
                    "hive_mind.promote_payload.skip | entities=%s id=%s",
                    entities, latest.id,
                )
                return latest.id, False
            self.results.update_status(latest.id, PromotionStatus.SUPERSEDED)
            logger.info(
                "hive_mind.promote_payload.supersede | entities=%s old=%s hash=%s",
                entities, latest.id, content_hash[:8],
            )

        result = SessionResult(
            content      = content,
            entities     = entities,
            tags         = payload_tags,
            turn_nr      = turn_nr,
            status       = PromotionStatus.FINAL,
            content_hash = content_hash,
            origin       = self._stamp(),
        )
        self.results.store(result)
        self._on_invalidate(entities)
        logger.info(
            "hive_mind.promote_payload.stored | id=%s entities=%s hash=%s",
            result.id, entities, content_hash[:8],
        )
        return result.id, True

    def get_payload(self, entities: list[str]) -> SessionResult | None:
        """Find the most recent final payload for a set of entities.

        Args:
            entities (list[str]): Entities used to search session results.

        Returns:
            SessionResult | None: Latest final payload, or ``None`` if absent.
        """
        candidates = [
            r for r in self.results.find_by_entities(entities)
            if "payload" in r.tags and r.status == PromotionStatus.FINAL
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: r.created_at)[-1]

    def load_artifact(self, artifact_id: str) -> ArtifactContent | None:
        clean = artifact_id.strip()
        if not clean or "/" in clean or "\\" in clean or ".." in clean:
            logger.warning("hive_mind.load_artifact.rejected | id=%r (unsafe)", clean)
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
            logger.warning("hive_mind.load_artifact.failed | id=%s error=%s", clean, exc)
            return None


    def resolve_reference(self, ref: str) -> TurnSummary | None:
        return self.conversation.resolve_reference(ref)


    def _dump(self, checkpoint_name: str) -> None:
        logger.debug("STORE COORDINATOR CHECKPOINT: %s", checkpoint_name)

        recent_turns = self.conversation.last_n(4)
        if recent_turns:
            logger.debug("CONVERSATION last %d turns:", len(recent_turns))
            for turn in recent_turns:
                logger.debug(
                    "  [%d] intent=%s entities=%s correction_of=%s",
                    turn.turn_nr, turn.intent, turn.entities, turn.correction_of,
                )
        else:
            logger.debug("CONVERSATION: (empty)")

        all_results = self.results.all_final()
        if all_results:
            logger.debug("SESSION RESULTS (%d):", len(all_results))
            for r in all_results:
                logger.debug(
                    "  [%s] entities=%s tags=%s | %s",
                    r.id[:8], r.entities, r.tags,
                    r.content.replace("\n", " "),
                )
        else:
            logger.debug("SESSION RESULTS: (empty)")

        active_task = self.working.active_task_id
        if active_task:
            atoms = self.working.read_all()
            logger.debug(
                "WORKING MEMORY top=%s depth=%d (%d atoms):",
                active_task, self.working.depth, len(atoms),
            )
            for atom in atoms:
                logger.debug(
                    "  [%s] %s | %s",
                    atom.atom_type, atom.id[:8],
                    atom.content.replace("\n", " "),
                )
        else:
            logger.debug("WORKING MEMORY: (empty stack)")


    def _check_conflict(
        self,
        entities: list[str],
        content:  str,
    ) -> ConflictReport | None:
        if not entities:
            return None
        for ex in self.results.find_by_entities(entities):
            if ex.content != content:
                report = ConflictReport(
                    existing_id          = ex.id,
                    existing_content     = ex.peek(),
                    new_content          = content,
                    conflicting_entities = entities,
                    resolution           = "UNRESOLVED",
                )
                logger.warning(
                    "hive_mind.conflict | entities=%s old=%r",
                    entities, ex.peek(),
                )
                return report
        return None


class HiveMindStoreCoordinatorFactory(StoreCoordinatorFactory):
    """Implements StoreCoordinatorFactory. Creates or restores HiveMindStoreCoordinators via a SessionBackend."""

    def __init__(
        self,
        backend:       SessionBackend | None = None,
        base_dir:      str              = "./sessions",
        on_invalidate: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._backend      = backend or FileSystemBackend(base_dir)
        self._on_invalidate = on_invalidate
        #: Owners claimed for sessions that have not been written yet. A session
        #: is created in memory and only persisted on `save`, so the claim has
        #: to survive that gap or the first turn of every new session would be
        #: ownerless.
        self._claimed: dict[str, str] = {}

    def exists(self, session_id: str) -> bool:
        return self._backend.exists(session_id)

    def owner_of(self, session_id: str) -> str | None:
        snapshot = self._backend.load(session_id)
        if snapshot is None:
            return None
        return snapshot.owner_principal_id or None

    def claim_ownership(self, session_id: str, owner_principal_id: str) -> None:
        """Record an owner for a new or previously unowned session.

        Args:
            session_id (str): Session key whose owner is recorded.
            owner_principal_id (str): Established principal's stable ID.

        Side Effects:
            Caches the owner for a new session or saves it on an unowned
            snapshot. An existing owner is never overwritten.
        """
        snapshot = self._backend.load(session_id)
        if snapshot is None:
            self._claimed[session_id] = owner_principal_id
            return
        if snapshot.owner_principal_id:
            return
        snapshot.owner_principal_id = owner_principal_id
        self._backend.save(snapshot)

    def get_or_create(self, session_id: str) -> HiveMindStoreCoordinator:
        if self._backend.exists(session_id):
            snapshot = self._backend.load(session_id)
            if snapshot:
                coordinator = self._restore(snapshot)
                logger.info(
                    "hive_mind.session.restored | id=%s turns=%d results=%d",
                    session_id,
                    len(snapshot.turns),
                    len(snapshot.results),
                )
                return coordinator

        coordinator = HiveMindStoreCoordinator(on_invalidate=self._on_invalidate)
        logger.info("hive_mind.session.created | id=%s", session_id)
        return coordinator

    # Narrower than StoreCoordinatorFactory.save(coordinator: StoreCoordinator) --
    # this factory only ever hands out and receives back its own coordinator
    # type (enforced by construction in bootstrap/wiring.py, one factory per
    # coordinator kind), never a StoreCoordinator of a different concrete type.
    def save(self, session_id: str, coordinator: HiveMindStoreCoordinator) -> None:  # type: ignore[override]
        interrupted: list[str] = []
        if coordinator.working.active_task_id:
            interrupted.append(coordinator.working.active_task_id)

        now = datetime.now(UTC)
        snapshot = SessionSnapshot(
            session_id           = session_id,
            turns                = coordinator.conversation.last_n(9999),
            results              = coordinator.results.all_final(),
            interrupted_task_ids = interrupted,
            slots                = _slots_to_json(coordinator._slots),
            created_at           = now,
            updated_at           = now,
        )
        existing = self._backend.load(session_id)
        if existing:
            snapshot.created_at = existing.created_at
            # An owner is never rewritten by a save: it was settled at creation.
            snapshot.owner_principal_id = existing.owner_principal_id
        snapshot.owner_principal_id = (
            snapshot.owner_principal_id or self._claimed.pop(session_id, "")
        )

        self._backend.save(snapshot)
        logger.info(
            "hive_mind.session.saved | id=%s turns=%d results=%d slots=%d",
            session_id, len(snapshot.turns), len(snapshot.results), len(snapshot.slots),
        )

    def _restore(self, snapshot: SessionSnapshot) -> HiveMindStoreCoordinator:
        coordinator = HiveMindStoreCoordinator(on_invalidate=self._on_invalidate)
        for turn in sorted(snapshot.turns, key=lambda t: t.turn_nr):
            coordinator.conversation.append(turn)
        for result in snapshot.results:
            coordinator.results.store(result)
        if snapshot.slots:
            coordinator._slots.update(snapshot.slots)
            logger.debug("hive_mind.session.slots_restored | count=%d", len(snapshot.slots))
        if snapshot.interrupted_task_ids:
            logger.warning(
                "hive_mind.session.interrupted_tasks | ids=%s",
                snapshot.interrupted_task_ids,
            )
        return coordinator
