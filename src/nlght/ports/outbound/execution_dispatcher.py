# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Protocol

from nlght.core.execution import ExecutionRecord, ExecutionSubmission


class ExecutionDispatcher(Protocol):
    async def submit(self, submission: ExecutionSubmission) -> ExecutionRecord: ...

    async def get_status(self, execution_id: uuid.UUID) -> ExecutionRecord | None: ...

    async def cancel(self, execution_id: uuid.UUID) -> ExecutionRecord | None: ...
