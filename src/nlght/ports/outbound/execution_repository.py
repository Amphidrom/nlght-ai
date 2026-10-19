# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol

from nlght.core.execution import (
    ExecutionAttempt,
    ExecutionClaim,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionStatusCounts,
    ExecutionSubmission,
    FanOutSummary,
    WorkerRegistration,
    WorkerStatus,
)


class ExecutionRepository(Protocol):
    async def submit(self, submission: ExecutionSubmission) -> ExecutionRecord: ...

    async def get(self, execution_id: uuid.UUID) -> ExecutionRecord | None: ...

    async def fan_out_summary(self, parent_execution_id: uuid.UUID) -> FanOutSummary:
        """Count child executions by state without loading their records.

        Args:
            parent_execution_id (uuid.UUID): Parent execution ID.

        Returns:
            FanOutSummary: Fan-out summary for that parent.

        Side Effects:
            Reads child execution states.
        """
        ...

    async def cancel(self, execution_id: uuid.UUID) -> ExecutionRecord | None: ...

    # -- observation ---------------------------------------------------------
    #
    # Keep observation reads separate from dispatch so listing work cannot
    # affect the execution claim path.

    async def list_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ExecutionRecord, ...]:
        """Page top-level runs without flooding results with children.

        Args:
            status (ExecutionStatus | None): Optional execution-status filter.
            workflow_id (uuid.UUID | None): Optional workflow filter.
            limit (int): Maximum page size.
            offset (int): Number of top-level runs to skip.

        Returns:
            tuple[ExecutionRecord, ...]: Matching top-level executions in newest-first order.

        Side Effects:
            Reads execution records.
        """
        ...

    async def count_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
    ) -> int:
        """Count top-level runs for observation paging.

        Args:
            status (ExecutionStatus | None): Optional execution-status filter.
            workflow_id (uuid.UUID | None): Optional workflow filter.

        Returns:
            int: Number of matching runs.

        Side Effects:
            Reads execution records.
        """
        ...

    async def fan_out_summaries(
        self, parent_execution_ids: tuple[uuid.UUID, ...]
    ) -> dict[uuid.UUID, FanOutSummary]:
        """Summarize child states for multiple runs in one query.

        Args:
            parent_execution_ids (tuple[uuid.UUID, ...]): Parent execution IDs.

        Returns:
            dict[uuid.UUID, FanOutSummary]: Fan-out summary keyed by parent ID.

        Side Effects:
            Reads child execution states.
        """
        ...

    async def list_children(
        self, parent_execution_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[ExecutionRecord, ...]:
        """Page child executions in submission order.

        Args:
            parent_execution_id (uuid.UUID): Parent execution to inspect.
            limit (int): Maximum page size.
            offset (int): Number of child runs to skip.

        Returns:
            tuple[ExecutionRecord, ...]: Child executions in oldest-first order.

        Side Effects:
            Reads execution records.
        """
        ...

    async def attempts(self, execution_id: uuid.UUID) -> tuple[ExecutionAttempt, ...]:
        """List the attempts of one execution.

        Args:
            execution_id (uuid.UUID): Execution ID.

        Returns:
            tuple[ExecutionAttempt, ...]: Ordered attempts with worker attribution.

        Side Effects:
            Reads attempt records.
        """
        ...

    async def status_counts(self, *, since: datetime | None = None) -> ExecutionStatusCounts:
        """Count executions by status for observation.

        Args:
            since (datetime | None): Optional earliest creation time.

        Returns:
            ExecutionStatusCounts: Counts grouped by execution status.

        Side Effects:
            Reads execution records.
        """
        ...

    async def list_workers(self, *, now: datetime | None = None) -> tuple[WorkerStatus, ...]:
        """List worker capabilities and current load.

        Args:
            now (datetime | None): Optional clock value for consistent lease expiry checks.

        Returns:
            tuple[WorkerStatus, ...]: Registered worker status records.

        Side Effects:
            Reads worker registrations and leases.
        """
        ...

    async def register_worker(
        self,
        registration: WorkerRegistration,
        *,
        expires_at: datetime,
    ) -> None: ...

    async def heartbeat_worker(
        self,
        worker_id: uuid.UUID,
        boot_token: uuid.UUID,
        *,
        expires_at: datetime,
    ) -> bool: ...

    async def stop_worker(self, worker_id: uuid.UUID, boot_token: uuid.UUID) -> None: ...

    async def claim(
        self,
        registration: WorkerRegistration,
        *,
        now: datetime,
        lease_expires_at: datetime,
    ) -> ExecutionClaim | None: ...

    async def mark_running(
        self,
        claim: ExecutionClaim,
        *,
        lease_expires_at: datetime,
    ) -> bool: ...

    async def heartbeat_execution(
        self,
        claim: ExecutionClaim,
        *,
        lease_expires_at: datetime,
        progress: dict[str, Any] | None = None,
    ) -> bool: ...

    async def succeed(self, claim: ExecutionClaim, result: dict[str, Any]) -> bool: ...

    async def fail(
        self,
        claim: ExecutionClaim,
        diagnostics: dict[str, Any],
        *,
        retry_at: datetime | None,
    ) -> bool: ...
