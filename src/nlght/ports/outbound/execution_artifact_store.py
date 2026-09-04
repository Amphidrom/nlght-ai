# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Any, Protocol

from nlght.core.execution import ExecutionArtifact


class ExecutionArtifactStore(Protocol):
    async def put(
        self,
        *,
        execution_id: uuid.UUID,
        name: str,
        content_type: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionArtifact: ...

    async def get(self, artifact_id: uuid.UUID) -> ExecutionArtifact | None: ...
