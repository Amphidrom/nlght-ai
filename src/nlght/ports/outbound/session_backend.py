# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nlght.core.hive_mind.models import SessionSnapshot


@runtime_checkable
class SessionBackend(Protocol):
    """Outbound port for session snapshot persistence.

    Used exclusively by ``HiveMindStoreCoordinatorFactory``.
    Implementations: ``FileSystemBackend`` (JSON files),
    future: ``PostgresSessionBackend``.

    ``SimpleStoreCoordinatorFactory`` does not use this port — it
    holds sessions in-memory with TTL eviction.
    """

    def save(self, snapshot: SessionSnapshot) -> None: ...

    def load(self, session_id: str) -> SessionSnapshot | None: ...

    def delete(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[str]: ...

    def exists(self, session_id: str) -> bool: ...
