# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Protocol

from nlght.core.workflow.workflow import WorkflowDef, WorkflowVersionDef


class WorkflowRepository(Protocol):
    async def find_by_name(self, name: str) -> WorkflowDef | None: ...

    async def find_by_id(self, workflow_id: uuid.UUID) -> WorkflowDef | None: ...

    async def list_enabled(self) -> list[WorkflowDef]: ...

    async def find_active_version(
        self, workflow_id: uuid.UUID
    ) -> WorkflowVersionDef | None: ...

    async def find_version(
        self,
        workflow_id: uuid.UUID,
        workflow_version_id: uuid.UUID,
    ) -> WorkflowVersionDef | None: ...
