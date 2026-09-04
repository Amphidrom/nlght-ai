# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from nlght.core.hive_mind.models import ConflictReport, SessionResult, TurnSummary, WorkingAtom, WriteIntent

T = TypeVar("T")


@dataclass(frozen=True)
class ArtifactContent:
    """Resolved artifact content together with its file extension.

    ``extension`` includes the leading dot (e.g. ``".py"``, ``".json"``).
    Empty string means the artifact was stored without an extension.
    """

    content: str
    extension: str


@runtime_checkable
class StoreCoordinator(Protocol):
    """Outbound port for session memory coordination.

    Steps interact with session memory exclusively through this port.
    Two implementations exist:

    * ``SimpleStoreCoordinator`` — in-memory, no indexing, free tier.
    * ``HiveMindStoreCoordinator`` — full entity-indexed, scored,
      persistent memory; requires ``[hive-mind]`` extra.

    Absence (``None`` on ``WorkflowStepContext``) is valid — the step
    machine runs without session memory.

    Coordinator-scoped extension state uses the generic ``get_slot`` and
    ``set_slot`` methods. Consumers own their namespaced keys and value types;
    persistence depends on the selected coordinator implementation.

    See: ADR-0017
    """

    def record_turn(self, turn: TurnSummary) -> None: ...

    def get_recent_turns(self, n: int) -> list[TurnSummary]: ...

    def next_turn_nr(self) -> int: ...

    def write(self, intent: WriteIntent) -> tuple[str, ConflictReport | None]: ...

    def start_task(self, task_id: str) -> None: ...

    def finish_task(
        self,
        task_id:  str,
        entities: list[str],
        turn_nr:  int = 0,
    ) -> list[SessionResult]: ...

    def discard_task(self, task_id: str) -> None: ...

    def fail_task(self, task_id: str) -> None: ...

    def get_context(self, entities: list[str]) -> dict[str, Any]: ...

    def is_known(
        self,
        entities: list[str],
        tags:     list[str] | None = None,
    ) -> bool: ...

    def resolve_reference(self, ref: str) -> TurnSummary | None: ...

    def promote_payload(
        self,
        content:  str,
        entities: list[str],
        tags:     list[str],
        turn_nr:  int,
    ) -> tuple[str, bool]: ...

    def get_payload(self, entities: list[str]) -> SessionResult | None: ...

    def load_artifact(self, artifact_id: str) -> ArtifactContent | None: ...

    # ------------------------------------------------------------------
    # Generic slot API — extension point for coordinator-scoped state.
    #
    # ``get_slot`` lazily initialises the slot via ``default_factory``
    # the first time a key is accessed. Consumers own their namespaced
    # keys and value types. Persistence is implementation-dependent:
    # the simple coordinator is process-local, while Hive Mind includes
    # JSON-serialisable values in its session snapshot.
    # ------------------------------------------------------------------

    def get_slot(self, key: str, default_factory: Callable[[], T]) -> T: ...

    def set_slot(self, key: str, value: object) -> None: ...

    def open_branch(self, label: str) -> None: ...

    def close_branch(self, promote: bool = False) -> list[WorkingAtom]: ...

    def store_promoted_atoms(
        self,
        atoms: list[WorkingAtom],
        *,
        turn_nr: int,
        entities: list[str] | None = None,
    ) -> list[SessionResult]: ...


@runtime_checkable
class StoreCoordinatorFactory(Protocol):
    """Factory that creates or restores a StoreCoordinator per session.

    ``get_or_create`` is called before the step machine starts;
    ``save`` is called after the terminal step completes.
    """

    def exists(self, session_id: str) -> bool: ...

    def get_or_create(self, session_id: str) -> StoreCoordinator: ...

    def owner_of(self, session_id: str) -> str | None:
        """Which principal owns this session, if it exists and has an owner.

        `None` for a session that does not exist **and** for one written before
        ownership was recorded. The two are told apart with `exists`, because
        they mean opposite things: the first may be created, the second may not
        be adopted (ADR-0061).

        On the factory rather than on `SessionBackend` deliberately. One factory
        keeps its sessions in memory and never reaches a backend at all, so a
        check placed underneath would be bypassed entirely by a cache hit —
        which is the exact path an attacker with a session id would take.
        """
        ...

    def claim_ownership(self, session_id: str, owner_principal_id: str) -> None:
        """Record who a newly created session belongs to.

        Only ever called for a session being created. An existing owner is never
        overwritten: a session changing hands is not something a request may do.
        """
        ...

    def save(self, session_id: str, coordinator: StoreCoordinator) -> None: ...
