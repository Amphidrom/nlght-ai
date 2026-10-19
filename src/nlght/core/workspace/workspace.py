# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class WorkspaceStrategy(StrEnum):
    SESSION = "session"
    EPHEMERAL = "ephemeral"


@dataclass(frozen=True)
class WorkspaceContext:
    """Runtime representation of an active workspace.

    Provisioned once per invocation (or session) by the WorkspaceManager
    and stored in WorkflowStepContext.workspace.

    ``session_key`` is the client-side key extracted by the protocol for a
    session workspace, such as the OpenAI ``user`` field or x-session-id header.
    It is ``None`` for an ephemeral workspace.
    """

    workspace_id: str
    root_path: Path
    strategy: WorkspaceStrategy
    session_key: str | None
    created_at: datetime
