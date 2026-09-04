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
        """Count the children of one fanned-out run by state.

        An aggregate rather than the records themselves: a run may fan out a
        document per child, and asking whether the corpus is done must not mean
        loading the corpus.
        """
        ...

    async def cancel(self, execution_id: uuid.UUID) -> ExecutionRecord | None: ...

    # -- observation ---------------------------------------------------------
    #
    # Everything below is read-only and exists so a queue that decides what runs
    # can be looked at. The dispatch path above never calls it.

    async def list_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ExecutionRecord, ...]:
        """Top-level runs, newest first — the ones no other execution fanned out.

        A fanned-out corpus is one run with hundreds of children; listing every
        execution would bury the hundred runs that matter under one. Children
        are reached through their parent.
        """
        ...

    async def count_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
    ) -> int:
        """How many top-level runs match, so a listing can be paged."""
        ...

    async def fan_out_summaries(
        self, parent_execution_ids: tuple[uuid.UUID, ...]
    ) -> dict[uuid.UUID, FanOutSummary]:
        """``fan_out_summary`` for a whole page of runs in one query.

        A listing shows the child state of every run on it; asking per row would
        be one query per run, which is the difference between a page that loads
        and one that crawls.
        """
        ...

    async def list_children(
        self, parent_execution_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[ExecutionRecord, ...]:
        """The children of one run, oldest first — the order they were handed out."""
        ...

    async def attempts(self, execution_id: uuid.UUID) -> tuple[ExecutionAttempt, ...]:
        """Every try at one execution, in order, with the worker that held it."""
        ...

    async def status_counts(self, *, since: datetime | None = None) -> ExecutionStatusCounts:
        """Executions per status, optionally only those created since a moment."""
        ...

    async def list_workers(self, *, now: datetime | None = None) -> tuple[WorkerStatus, ...]:
        """Registered workers with their capabilities and current load.

        ``now`` decides which leases count as expired; a caller that passes its
        own clock gets a consistent picture across several calls.
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
