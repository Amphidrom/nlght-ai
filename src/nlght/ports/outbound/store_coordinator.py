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

    * ``SimpleStoreCoordinator``: in-memory with no entity indexing.
    * ``HiveMindStoreCoordinator``: entity-indexed persistent memory.
      Install the ``[hive-mind]`` extra for this implementation.

    Absence (``None`` on ``WorkflowStepContext``) is valid. The step
    machine runs without session memory.

    Coordinator-scoped extension state uses the generic ``get_slot`` and
    ``set_slot`` methods. Consumers own their namespaced keys and value types.
    The selected coordinator implementation controls persistence.

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

    # Consumers own namespaced slot keys and types for coordinator-scoped state.
    #
    # ``get_slot`` lazily initialises the slot via ``default_factory``
    # the first time a key is accessed. Consumers own their namespaced
    # keys and value types. Persistence is implementation-dependent:
    # the simple coordinator is process-local, while Hive Mind includes
    # JSON-serialisable values in its session snapshot.

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

    ``get_or_create`` is called before the step machine starts.
    ``save`` is called after the terminal step completes.
    """

    def exists(self, session_id: str) -> bool: ...

    def get_or_create(self, session_id: str) -> StoreCoordinator: ...

    def owner_of(self, session_id: str) -> str | None:
        """Read ownership at the factory boundary for a session.

        Args:
            session_id (str): Session ID.

        Returns:
            str | None: Owner principal ID, or None for absent or unowned sessions.

        Side Effects:
            Reads session metadata and may evict expired in-memory state.
        """
        ...

    def claim_ownership(self, session_id: str, owner_principal_id: str) -> None:
        """Assign an owner when creating a session.

        Args:
            session_id (str): Session being created.
            owner_principal_id (str): Established principal that owns the session.

        Side Effects:
            Records ownership without replacing an existing owner.
        """
        ...

    def save(self, session_id: str, coordinator: StoreCoordinator) -> None: ...
