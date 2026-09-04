# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid

from nlght.core.execution import ExecutionRecord, ExecutionSubmission
from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
from nlght.ports.outbound.execution_repository import ExecutionRepository


class ExecutionDispatchService(ExecutionDispatcher):
    def __init__(self, repository: ExecutionRepository) -> None:
        self._repository = repository

    async def submit(self, submission: ExecutionSubmission) -> ExecutionRecord:
        return await self._repository.submit(submission)

    async def get_status(self, execution_id: uuid.UUID) -> ExecutionRecord | None:
        return await self._repository.get(execution_id)

    async def cancel(self, execution_id: uuid.UUID) -> ExecutionRecord | None:
        return await self._repository.cancel(execution_id)
