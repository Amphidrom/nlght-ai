# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext


@runtime_checkable
class WorkspaceManager(Protocol):
    """Manages the lifecycle of workspaces.

    ``get_or_create(None)`` creates an ephemeral directory that is deleted after
    the invocation. A session key reuses a persistent directory across requests.
    ``release()`` deletes ephemeral directories and removes session workspaces
    from the in-memory registry while retaining their directories.

    The protocol logic that extracts the session_key lives in the
    ``TriggerResolver``. The workspace manager receives the resolved key.
    """

    async def get_or_create(self, session_key: str | None) -> WorkspaceContext: ...

    async def release(self, workspace: WorkspaceContext) -> None: ...
