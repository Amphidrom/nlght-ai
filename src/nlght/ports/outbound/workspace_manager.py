# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext


@runtime_checkable
class WorkspaceManager(Protocol):
    """Manages the lifecycle of workspaces.

    get_or_create:
        session_key=None  → ephemeral workspace (temp dir; automatically
                            deleted after the invocation)
        session_key=str   → session workspace (persistent, reused for
                            the same session_key across multiple requests)

    release:
        Ephemeral → deletes the directory.
        Session   → removed from the in-memory registry; directory remains.

    The protocol logic that extracts the session_key lives in the
    TriggerResolver — the WorkspaceManager only knows the key.
    """

    async def get_or_create(self, session_key: str | None) -> WorkspaceContext: ...

    async def release(self, workspace: WorkspaceContext) -> None: ...
