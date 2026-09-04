# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from nlght.core.trigger.trigger import Trigger


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class ExecutionSubmission:
    workflow_id: uuid.UUID
    workflow_version_id: uuid.UUID
    trigger: Trigger
    idempotency_key: str
    required_capabilities: tuple[str, ...] = ()
    priority: int = 0
    artifact_refs: tuple[str, ...] = ()
    source_config_revision: str | None = None
    max_attempts: int = 3
    # Snapshot of the workflow's concurrency policy at submission. When true, at
    # most one execution of this workflow runs at a time — enforced at claim.
    exclusive: bool = False
    # The execution that fanned this one out, if any. A child is an ordinary
    # execution — claimed, leased, and retried like any other — and the link
    # exists so the work of one logical run remains findable once it has been
    # spread across workers.
    parent_execution_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        # Streaming triggers are allowed: a streaming execution runs on a worker
        # that publishes its signals to the execution stream broker, which the
        # gateway relays to the caller. The durable record still holds the
        # terminal result.
        normalized = tuple(sorted({item.strip() for item in self.required_capabilities if item.strip()}))
        object.__setattr__(self, "required_capabilities", normalized)


@dataclass(slots=True, frozen=True)
class ExecutionRecord:
    execution_id: uuid.UUID
    workflow_id: uuid.UUID
    workflow_version_id: uuid.UUID
    trigger: Trigger
    idempotency_key: str
    required_capabilities: tuple[str, ...]
    priority: int
    artifact_refs: tuple[str, ...]
    source_config_revision: str | None
    max_attempts: int
    attempt_count: int
    fencing_token: int
    status: ExecutionStatus
    owner_worker_id: uuid.UUID | None
    owner_boot_token: uuid.UUID | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime
    cancel_requested: bool
    progress: dict[str, Any]
    result: dict[str, Any] | None
    diagnostics: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    exclusive: bool = False
    parent_execution_id: uuid.UUID | None = None


@dataclass(slots=True, frozen=True)
class FanOutSummary:
    """How far the children of one fanned-out run have got.

    The run that fanned them out has long finished — it submits and ends rather
    than holding a worker while its children queue behind it — so this is how
    anyone asks whether the logical run is done.
    """

    parent_execution_id: uuid.UUID
    total: int
    pending: int
    """Children not yet in a terminal state: queued, leased, or running."""
    succeeded: int
    failed: int
    cancelled: int

    first_started_at: datetime | None = None
    """When the earliest child was picked up. None while none has been claimed."""
    last_completed_at: datetime | None = None
    """When the latest child to settle did. Only a finish once nothing is pending."""
    child_seconds: float = 0.0
    """Time the children spent between them, summed over each one's own run.

    Not the run's wall clock, and usually much larger than it: the children run
    on several workers at once, so the difference between the two is what the
    parallelism bought. Children still running contribute nothing yet.
    """

    @property
    def settled(self) -> bool:
        """Whether every child reached a terminal state."""
        return self.total > 0 and self.pending == 0

    @property
    def finished_at(self) -> datetime | None:
        """When the fanned-out work actually ended, or None while it has not.

        Deliberately not the maximum completion seen so far: with children still
        queued, the latest one to finish is not the last one, and showing it as
        a finish time is how a run half done reads as a run over.
        """
        return self.last_completed_at if self.settled else None

    @property
    def child_time(self) -> timedelta:
        return timedelta(seconds=self.child_seconds)


@dataclass(slots=True, frozen=True)
class ExecutionClaim:
    execution: ExecutionRecord
    attempt_id: uuid.UUID
    attempt_number: int
    worker_id: uuid.UUID
    boot_token: uuid.UUID
    fencing_token: int
    lease_expires_at: datetime


@dataclass(slots=True, frozen=True)
class WorkerRegistration:
    worker_id: uuid.UUID
    boot_token: uuid.UUID
    instance_name: str
    capabilities: tuple[str, ...] = ()
    capacity: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("worker capacity must be at least 1")
        normalized = tuple(sorted({item.strip() for item in self.capabilities if item.strip()}))
        object.__setattr__(self, "capabilities", normalized)


@dataclass(slots=True, frozen=True)
class ExecutionArtifact:
    artifact_id: uuid.UUID
    execution_id: uuid.UUID
    name: str
    content_type: str
    checksum: str
    content: bytes
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(slots=True, frozen=True)
class ExecutionAttempt:
    """One worker's try at one execution.

    The execution record carries only where a run ended up. This is how it got
    there: which worker held which attempt, under which fencing token, and what
    the failure said. A run that succeeded on its third try looks identical to
    one that succeeded on its first unless someone can read this.
    """

    attempt_id: uuid.UUID
    execution_id: uuid.UUID
    attempt_number: int
    worker_id: uuid.UUID
    instance_name: str
    fencing_token: int
    status: str
    lease_expires_at: datetime
    heartbeat_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    diagnostics: dict[str, Any] | None
    created_at: datetime


@dataclass(slots=True, frozen=True)
class WorkerStatus:
    """A registered worker as an observer sees it.

    ``expired`` is the interesting one: a worker that stopped without
    deregistering stays 'active' in its own row until its lease runs out, and
    the executions it held are only reclaimed once it does. Reading the two
    together is how "why is nothing progressing" gets answered.
    """

    worker_id: uuid.UUID
    instance_name: str
    status: str
    capacity: int
    capabilities: tuple[str, ...]
    running: int
    heartbeat_at: datetime
    expires_at: datetime
    expired: bool
    created_at: datetime


@dataclass(slots=True, frozen=True)
class ExecutionStatusCounts:
    """How many executions sit in each status, for an at-a-glance answer."""

    counts: dict[ExecutionStatus, int]

    def of(self, status: ExecutionStatus) -> int:
        return self.counts.get(status, 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def in_flight(self) -> int:
        return (
            self.of(ExecutionStatus.QUEUED)
            + self.of(ExecutionStatus.LEASED)
            + self.of(ExecutionStatus.RUNNING)
        )
