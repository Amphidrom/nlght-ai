# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import Float, and_, delete, exists, func, or_, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import aliased
from sqlalchemy.sql.functions import FunctionElement

from nlght.adapters.outbound.persistence.models import (
    WorkflowExecution,
    WorkflowExecutionArtifact,
    WorkflowExecutionAttempt,
    WorkflowExecutionRequirement,
    WorkflowWorker,
    WorkflowWorkerCapability,
)
from nlght.core.entry.context import PrincipalRef, RequestContext
from nlght.core.execution import (
    ExecutionArtifact,
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
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.ports.outbound.execution_artifact_store import ExecutionArtifactStore
from nlght.ports.outbound.execution_repository import ExecutionRepository


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(moment: datetime) -> datetime:
    """Restore the offset a backend without one dropped on the way out.

    The columns are timezone-aware and Postgres hands them back that way.
    SQLite has nowhere to keep the offset, so the same value returns naive —
    and comparing that against an aware ``now`` raises TypeError rather than
    being quietly wrong. Everything written here is UTC, so saying so is a
    restoration and not a guess.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


def _serialize_trigger(trigger: Trigger) -> dict[str, Any]:
    """The trigger as the queue stores it.

    Two fields of `RequestContext` are treated differently on purpose, and the
    difference is the whole security semantics of a durable execution.

    `principal_id` **is** written. The principal was established at a trusted
    entry, by a `PrincipalResolver`, from a source the worker no longer has — the
    gateway's headers are gone by the time work is claimed. So the established
    identity is delegated through the queue, and a worker adopts it rather than
    re-authenticating a request that has long since returned. What travels is a
    *reference to an established identity*, never a credential and never a claim
    somebody made: no tokens, no API keys, no header a caller controlled.

    The accepted contract is that a job runs as the principal established when it
    was submitted. Whether revoking a credential should also stop jobs already
    accepted is a revocation question, and answering it here by quietly dropping
    the principal would only mean they run as nobody instead.

    `workflow` is **not** written, and must not be. The executor knows which
    workflow it is actually running and stamps `invocation.workflow.name` onto
    the context before any policy reads it. A serialized copy would be a second
    answer to a question the runtime can answer authoritatively — and for a
    fan-out child it would be the *parent's* name.
    """
    return {
        "kind": trigger.kind.value,
        "protocol": trigger.protocol.value,
        "operation": trigger.operation,
        "payload": trigger.payload,
        "context": {
            "correlation_id": trigger.context.correlation_id,
            "request_id": trigger.context.request_id,
            "received_at": trigger.context.received_at.isoformat(),
            "path": trigger.context.path,
            "method": trigger.context.method,
            "headers": trigger.context.headers,
            "query_params": trigger.context.query_params,
            "client_host": trigger.context.client_host,
            "principal_id": (
                trigger.context.principal.id
                if trigger.context.principal is not None
                else None
            ),
        },
        "stream": trigger.stream,
        "session_key": trigger.session_key,
        "metadata": trigger.metadata,
    }


def _deserialize_trigger(value: dict[str, Any]) -> Trigger:
    context = dict(value["context"])
    return Trigger(
        kind=TriggerKind(str(value["kind"])),
        protocol=ProtocolKind(str(value["protocol"])),
        operation=str(value["operation"]),
        payload=dict(value.get("payload") or {}),
        context=RequestContext(
            correlation_id=str(context["correlation_id"]),
            request_id=str(context["request_id"]),
            received_at=datetime.fromisoformat(str(context["received_at"])),
            path=str(context["path"]),
            method=str(context["method"]),
            headers={str(k): str(v) for k, v in dict(context.get("headers") or {}).items()},
            query_params={
                str(k): str(v) for k, v in dict(context.get("query_params") or {}).items()
            },
            client_host=(
                str(context["client_host"])
                if context.get("client_host") is not None
                else None
            ),
            # A row enqueued before principals existed has no key here, and gets
            # `None` — no principal was established for it, which is the truth.
            # Under enforcement that is a refusal rather than a session opened
            # for nobody, which is the direction this has to fail in.
            principal=(
                PrincipalRef(str(context["principal_id"]))
                if context.get("principal_id")
                else None
            ),
            # `workflow` is deliberately absent: the executor establishes it from
            # the invocation it is actually running (see `_serialize_trigger`).
        ),
        stream=bool(value.get("stream", False)),
        session_key=(str(value["session_key"]) if value.get("session_key") is not None else None),
        metadata=dict(value.get("metadata") or {}),
    )


_TERMINAL_STATUSES = frozenset(
    {
        ExecutionStatus.SUCCEEDED.value,
        ExecutionStatus.FAILED.value,
        ExecutionStatus.CANCELLED.value,
    }
)


class _epoch_seconds(FunctionElement[float]):  # noqa: N801 (a SQL function, named as one)
    """Seconds between two timestamps, as a number the database can sum.

    There is no portable spelling for this. Postgres subtracts timestamps into
    an interval and pulls seconds out of it; SQLite has no interval type at all
    and goes the long way round through Julian days. Both spellings live here
    rather than as a dialect check in the middle of a query.

    NULL in either argument yields NULL, which sums to nothing — so a child that
    has not finished contributes no time, which is the truth about it.
    """

    inherit_cache = True
    type = Float()


@compiles(_epoch_seconds)
def _epoch_seconds_default(
    element: _epoch_seconds,
    compiler: Any,  # noqa: ANN401 (SQLAlchemy's compiler, whose shape is its own)
    **kw: Any,  # noqa: ANN401 (compiler flags, passed through untouched)
) -> str:
    later, earlier = tuple(element.clauses)
    return (
        f"EXTRACT(EPOCH FROM {compiler.process(later, **kw)}"
        f" - {compiler.process(earlier, **kw)})"
    )


@compiles(_epoch_seconds, "sqlite")
def _epoch_seconds_sqlite(
    element: _epoch_seconds,
    compiler: Any,  # noqa: ANN401 (SQLAlchemy's compiler, whose shape is its own)
    **kw: Any,  # noqa: ANN401 (compiler flags, passed through untouched)
) -> str:
    later, earlier = tuple(element.clauses)
    return (
        f"((julianday({compiler.process(later, **kw)})"
        f" - julianday({compiler.process(earlier, **kw)})) * 86400.0)"
    )


def _fan_out_timing() -> tuple[Any, Any, Any]:
    """The three timing aggregates, so both fan-out queries ask for the same ones."""
    return (
        func.min(WorkflowExecution.started_at),
        func.max(WorkflowExecution.completed_at),
        func.sum(
            _epoch_seconds(WorkflowExecution.completed_at, WorkflowExecution.started_at)
        ),
    )


def _summarize_fan_out(
    parent_id: uuid.UUID, rows: list[tuple[str, int, Any, Any, Any]]
) -> FanOutSummary:
    """Fold one parent's per-status rows into a single summary.

    The counts are grouped by status because that is what the status breakdown
    needs; the timing is not, so it is combined back across the groups here
    rather than asked for in a second query.
    """
    counts = {status: count for status, count, _, _, _ in rows}
    started = [row[2] for row in rows if row[2] is not None]
    completed = [row[3] for row in rows if row[3] is not None]
    return FanOutSummary(
        parent_execution_id=parent_id,
        total=sum(counts.values()),
        pending=sum(
            count for status, count in counts.items() if status not in _TERMINAL_STATUSES
        ),
        succeeded=counts.get(ExecutionStatus.SUCCEEDED.value, 0),
        failed=counts.get(ExecutionStatus.FAILED.value, 0),
        cancelled=counts.get(ExecutionStatus.CANCELLED.value, 0),
        first_started_at=min(started) if started else None,
        last_completed_at=max(completed) if completed else None,
        # Rounded because SQLite's route through Julian days does not land on
        # whole seconds: three children of a minute each summed to 179.99999,
        # which the duration format then reports as two minutes fifty-nine.
        # Milliseconds are more precision than a sum over hundreds of children
        # can mean anyway.
        child_seconds=round(
            sum(float(row[4]) for row in rows if row[4] is not None), 3
        ),
    )


def _record(row: WorkflowExecution, capabilities: tuple[str, ...]) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=row.execution_id,
        workflow_id=row.workflow_id,
        workflow_version_id=row.workflow_version_id,
        trigger=_deserialize_trigger(row.trigger),
        idempotency_key=row.idempotency_key,
        required_capabilities=capabilities,
        priority=row.priority,
        artifact_refs=tuple(row.artifact_refs),
        source_config_revision=row.source_config_revision,
        max_attempts=row.max_attempts,
        attempt_count=row.attempt_count,
        fencing_token=row.fencing_token,
        status=ExecutionStatus(row.status),
        owner_worker_id=row.owner_worker_id,
        owner_boot_token=row.owner_boot_token,
        lease_expires_at=row.lease_expires_at,
        next_attempt_at=row.next_attempt_at,
        cancel_requested=row.cancel_requested,
        progress=dict(row.progress),
        result=dict(row.result) if row.result is not None else None,
        diagnostics=dict(row.diagnostics) if row.diagnostics is not None else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        exclusive=row.exclusive,
        parent_execution_id=row.parent_execution_id,
    )


class SqlAlchemyExecutionRepository(ExecutionRepository):
    """PostgreSQL coordination adapter with short, fenced transactions.

    ``FOR UPDATE SKIP LOCKED`` is used for claims on PostgreSQL. Conditional
    owner/fencing predicates remain in every state transition as a second
    guard and make the repository deterministic under SQLite integration tests.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def _capabilities(
        self, session: AsyncSession, execution_id: uuid.UUID
    ) -> tuple[str, ...]:
        result = await session.execute(
            select(WorkflowExecutionRequirement.capability)
            .where(WorkflowExecutionRequirement.execution_id == execution_id)
            .order_by(WorkflowExecutionRequirement.capability)
        )
        return tuple(result.scalars())

    @staticmethod
    def _matches(existing: WorkflowExecution, submission: ExecutionSubmission) -> bool:
        return (
            existing.workflow_id == submission.workflow_id
            and existing.workflow_version_id == submission.workflow_version_id
            and existing.trigger == _serialize_trigger(submission.trigger)
            and existing.source_config_revision == submission.source_config_revision
            and existing.priority == submission.priority
            and tuple(existing.artifact_refs) == submission.artifact_refs
            and existing.max_attempts == submission.max_attempts
            and existing.parent_execution_id == submission.parent_execution_id
        )

    async def submit(self, submission: ExecutionSubmission) -> ExecutionRecord:
        now = _utcnow()
        execution_id = uuid.uuid4()
        try:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                async with session.begin():
                    row = WorkflowExecution(
                        execution_id=execution_id,
                        workflow_id=submission.workflow_id,
                        workflow_version_id=submission.workflow_version_id,
                        idempotency_key=submission.idempotency_key,
                        trigger=_serialize_trigger(submission.trigger),
                        source_config_revision=submission.source_config_revision,
                        priority=submission.priority,
                        artifact_refs=list(submission.artifact_refs),
                        max_attempts=submission.max_attempts,
                        exclusive=submission.exclusive,
                        parent_execution_id=submission.parent_execution_id,
                        status=ExecutionStatus.QUEUED.value,
                        next_attempt_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)
                    for capability in submission.required_capabilities:
                        session.add(
                            WorkflowExecutionRequirement(
                                execution_id=execution_id,
                                capability=capability,
                            )
                        )
                return _record(row, submission.required_capabilities)
        except IntegrityError:
            async with AsyncSession(self._engine) as session:
                result = await session.execute(
                    select(WorkflowExecution).where(
                        WorkflowExecution.idempotency_key == submission.idempotency_key
                    )
                )
                existing = result.scalar_one_or_none()
                if existing is None:
                    raise
                capabilities = await self._capabilities(session, existing.execution_id)
                if not self._matches(existing, submission) or capabilities != submission.required_capabilities:
                    raise ValueError(
                        "idempotency_key already belongs to a different execution submission"
                    ) from None
                return _record(existing, capabilities)

    async def get(self, execution_id: uuid.UUID) -> ExecutionRecord | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(WorkflowExecution, execution_id)
            if row is None:
                return None
            capabilities = await self._capabilities(session, execution_id)
            return _record(row, capabilities)

    async def fan_out_summary(self, parent_execution_id: uuid.UUID) -> FanOutSummary:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowExecution.status, func.count(), *_fan_out_timing())
                .where(WorkflowExecution.parent_execution_id == parent_execution_id)
                .group_by(WorkflowExecution.status)
            )
            rows = [
                (status, int(count), started, completed, seconds)
                for status, count, started, completed, seconds in result.all()
            ]
        return _summarize_fan_out(parent_execution_id, rows)

    async def cancel(self, execution_id: uuid.UUID) -> ExecutionRecord | None:
        now = _utcnow()
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            async with session.begin():
                result = await session.execute(
                    select(WorkflowExecution)
                    .where(WorkflowExecution.execution_id == execution_id)
                    .with_for_update()
                )
                row = result.scalar_one_or_none()
                if row is None:
                    return None
                if row.status not in {
                    ExecutionStatus.SUCCEEDED.value,
                    ExecutionStatus.FAILED.value,
                    ExecutionStatus.CANCELLED.value,
                }:
                    row.cancel_requested = True
                    row.status = ExecutionStatus.CANCELLED.value
                    row.completed_at = now
                    row.updated_at = now
                    await session.execute(
                        update(WorkflowExecutionAttempt)
                        .where(
                            WorkflowExecutionAttempt.execution_id == execution_id,
                            WorkflowExecutionAttempt.status.in_(("leased", "running")),
                        )
                        .values(status="cancelled", completed_at=now)
                    )
                capabilities = await self._capabilities(session, execution_id)
            return _record(row, capabilities)

    # ------------------------------------------------------------------
    # Observation — read-only, never on the dispatch path
    # ------------------------------------------------------------------

    async def _capabilities_of(
        self, session: AsyncSession, execution_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, tuple[str, ...]]:
        """Capabilities for a whole page in one query rather than one per row."""
        if not execution_ids:
            return {}
        result = await session.execute(
            select(
                WorkflowExecutionRequirement.execution_id,
                WorkflowExecutionRequirement.capability,
            )
            .where(WorkflowExecutionRequirement.execution_id.in_(execution_ids))
            .order_by(WorkflowExecutionRequirement.capability)
        )
        grouped: dict[uuid.UUID, list[str]] = {}
        for execution_id, capability in result.all():
            grouped.setdefault(execution_id, []).append(capability)
        return {key: tuple(value) for key, value in grouped.items()}

    async def _page(
        self,
        session: AsyncSession,
        statement: Any,  # noqa: ANN401 (a SQLAlchemy Select, whose generic parameters vary)
    ) -> tuple[ExecutionRecord, ...]:
        rows = list((await session.execute(statement)).scalars())
        capabilities = await self._capabilities_of(session, [row.execution_id for row in rows])
        return tuple(_record(row, capabilities.get(row.execution_id, ())) for row in rows)

    async def list_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ExecutionRecord, ...]:
        statement = select(WorkflowExecution).where(
            WorkflowExecution.parent_execution_id.is_(None)
        )
        if status is not None:
            statement = statement.where(WorkflowExecution.status == status.value)
        if workflow_id is not None:
            statement = statement.where(WorkflowExecution.workflow_id == workflow_id)
        statement = (
            statement.order_by(WorkflowExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with AsyncSession(self._engine) as session:
            return await self._page(session, statement)

    async def count_runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(WorkflowExecution)
            .where(WorkflowExecution.parent_execution_id.is_(None))
        )
        if status is not None:
            statement = statement.where(WorkflowExecution.status == status.value)
        if workflow_id is not None:
            statement = statement.where(WorkflowExecution.workflow_id == workflow_id)
        async with AsyncSession(self._engine) as session:
            return int((await session.execute(statement)).scalar_one())

    async def fan_out_summaries(
        self, parent_execution_ids: tuple[uuid.UUID, ...]
    ) -> dict[uuid.UUID, FanOutSummary]:
        if not parent_execution_ids:
            return {}
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(
                    WorkflowExecution.parent_execution_id,
                    WorkflowExecution.status,
                    func.count(),
                    *_fan_out_timing(),
                )
                .where(WorkflowExecution.parent_execution_id.in_(parent_execution_ids))
                .group_by(WorkflowExecution.parent_execution_id, WorkflowExecution.status)
            )
            rows = result.all()

        grouped: dict[uuid.UUID, list[tuple[str, int, Any, Any, Any]]] = {}
        for parent_id, status, count, started, completed, seconds in rows:
            grouped.setdefault(parent_id, []).append(
                (status, int(count), started, completed, seconds)
            )
        return {
            parent_id: _summarize_fan_out(parent_id, parent_rows)
            for parent_id, parent_rows in grouped.items()
        }

    async def list_children(
        self, parent_execution_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[ExecutionRecord, ...]:
        statement = (
            select(WorkflowExecution)
            .where(WorkflowExecution.parent_execution_id == parent_execution_id)
            # Oldest first: the order the parent handed the shares out, which is
            # what makes "it stopped after the 87th" legible.
            .order_by(WorkflowExecution.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        async with AsyncSession(self._engine) as session:
            return await self._page(session, statement)

    async def attempts(self, execution_id: uuid.UUID) -> tuple[ExecutionAttempt, ...]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowExecutionAttempt, WorkflowWorker.instance_name)
                .join(
                    WorkflowWorker,
                    WorkflowWorker.worker_id == WorkflowExecutionAttempt.worker_id,
                    isouter=True,
                )
                .where(WorkflowExecutionAttempt.execution_id == execution_id)
                .order_by(WorkflowExecutionAttempt.attempt_number.asc())
            )
            return tuple(
                ExecutionAttempt(
                    attempt_id=row.attempt_id,
                    execution_id=row.execution_id,
                    attempt_number=row.attempt_number,
                    worker_id=row.worker_id,
                    # A worker row can be gone while its attempts remain.
                    instance_name=instance_name or str(row.worker_id),
                    fencing_token=row.fencing_token,
                    status=row.status,
                    lease_expires_at=row.lease_expires_at,
                    heartbeat_at=row.heartbeat_at,
                    started_at=row.started_at,
                    completed_at=row.completed_at,
                    diagnostics=dict(row.diagnostics) if row.diagnostics is not None else None,
                    created_at=row.created_at,
                )
                for row, instance_name in result.all()
            )

    async def status_counts(self, *, since: datetime | None = None) -> ExecutionStatusCounts:
        statement = select(WorkflowExecution.status, func.count()).group_by(
            WorkflowExecution.status
        )
        if since is not None:
            statement = statement.where(WorkflowExecution.created_at >= since)
        async with AsyncSession(self._engine) as session:
            result = await session.execute(statement)
            return ExecutionStatusCounts(
                counts={
                    ExecutionStatus(status): int(count) for status, count in result.all()
                }
            )

    async def list_workers(self, *, now: datetime | None = None) -> tuple[WorkerStatus, ...]:
        moment = now or _utcnow()
        async with AsyncSession(self._engine) as session:
            workers = list(
                (
                    await session.execute(
                        select(WorkflowWorker).order_by(WorkflowWorker.instance_name.asc())
                    )
                ).scalars()
            )
            worker_ids = [worker.worker_id for worker in workers]

            capabilities: dict[uuid.UUID, list[str]] = {}
            if worker_ids:
                rows = await session.execute(
                    select(
                        WorkflowWorkerCapability.worker_id,
                        WorkflowWorkerCapability.capability,
                    )
                    .where(WorkflowWorkerCapability.worker_id.in_(worker_ids))
                    .order_by(WorkflowWorkerCapability.capability)
                )
                for worker_id, capability in rows.all():
                    capabilities.setdefault(worker_id, []).append(capability)

            running: dict[uuid.UUID, int] = {}
            if worker_ids:
                rows = await session.execute(
                    select(WorkflowExecution.owner_worker_id, func.count())
                    .where(
                        WorkflowExecution.owner_worker_id.in_(worker_ids),
                        WorkflowExecution.status.in_(
                            (ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value)
                        ),
                    )
                    .group_by(WorkflowExecution.owner_worker_id)
                )
                running = {worker_id: int(count) for worker_id, count in rows.all()}

        return tuple(
            WorkerStatus(
                worker_id=worker.worker_id,
                instance_name=worker.instance_name,
                status=worker.status,
                capacity=worker.capacity,
                capabilities=tuple(capabilities.get(worker.worker_id, ())),
                running=running.get(worker.worker_id, 0),
                heartbeat_at=_as_utc(worker.heartbeat_at),
                expires_at=_as_utc(worker.expires_at),
                # A worker that died without deregistering stays 'active' until
                # its lease runs out. Saying so is the point of this view.
                expired=_as_utc(worker.expires_at) <= _as_utc(moment),
                created_at=_as_utc(worker.created_at),
            )
            for worker in workers
        )

    async def register_worker(
        self,
        registration: WorkerRegistration,
        *,
        expires_at: datetime,
    ) -> None:
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                row = await session.get(WorkflowWorker, registration.worker_id)
                if row is None:
                    row = WorkflowWorker(
                        worker_id=registration.worker_id,
                        boot_token=registration.boot_token,
                        instance_name=registration.instance_name,
                        capacity=registration.capacity,
                        status="active",
                        worker_metadata=registration.metadata,
                        heartbeat_at=now,
                        expires_at=expires_at,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)
                else:
                    row.boot_token = registration.boot_token
                    row.instance_name = registration.instance_name
                    row.capacity = registration.capacity
                    row.status = "active"
                    row.worker_metadata = registration.metadata
                    row.heartbeat_at = now
                    row.expires_at = expires_at
                    row.updated_at = now
                    await session.execute(
                        delete(WorkflowWorkerCapability).where(
                            WorkflowWorkerCapability.worker_id == registration.worker_id
                        )
                    )
                for capability in registration.capabilities:
                    session.add(
                        WorkflowWorkerCapability(
                            worker_id=registration.worker_id,
                            capability=capability,
                        )
                    )

    async def heartbeat_worker(
        self,
        worker_id: uuid.UUID,
        boot_token: uuid.UUID,
        *,
        expires_at: datetime,
    ) -> bool:
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                result = await session.execute(
                    update(WorkflowWorker)
                    .where(
                        WorkflowWorker.worker_id == worker_id,
                        WorkflowWorker.boot_token == boot_token,
                        WorkflowWorker.status == "active",
                    )
                    .values(heartbeat_at=now, expires_at=expires_at, updated_at=now)
                )
                return bool(_rowcount(result))

    async def stop_worker(self, worker_id: uuid.UUID, boot_token: uuid.UUID) -> None:
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await session.execute(
                    update(WorkflowWorker)
                    .where(
                        WorkflowWorker.worker_id == worker_id,
                        WorkflowWorker.boot_token == boot_token,
                    )
                    .values(status="stopping", expires_at=now, updated_at=now)
                )

    async def claim(
        self,
        registration: WorkerRegistration,
        *,
        now: datetime,
        lease_expires_at: datetime,
    ) -> ExecutionClaim | None:
        # Capability matching. A worker that advertises capabilities may only
        # claim an execution whose every requirement it satisfies. A worker with
        # *no* declared capabilities is a wildcard — a single-node
        # gateway+worker has to run every workflow out of the box — so no
        # requirement excludes it, and capabilities become an opt-in for
        # heterogeneous worker pools.
        capability_clauses: list[Any] = []
        if registration.capabilities:
            missing_requirement = select(WorkflowExecutionRequirement.execution_id).where(
                WorkflowExecutionRequirement.execution_id == WorkflowExecution.execution_id,
                WorkflowExecutionRequirement.capability.not_in(registration.capabilities),
            )
            capability_clauses.append(~exists(missing_requirement))

        # Per-workflow concurrency: a blocking (exclusive) execution is only
        # claimable when no *other* execution of the same workflow is already
        # LEASED or RUNNING. A sibling whose lease has expired still counts as
        # active, so recovery of the crashed run is preferred over starting a
        # queued one — the two never run at once. Non-exclusive executions are
        # unconstrained here.
        sibling = aliased(WorkflowExecution)
        active_sibling = select(sibling.execution_id).where(
            sibling.workflow_id == WorkflowExecution.workflow_id,
            sibling.execution_id != WorkflowExecution.execution_id,
            sibling.status.in_(
                (ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value)
            ),
        )

        eligible = and_(
            WorkflowExecution.cancel_requested.is_(False),
            WorkflowExecution.attempt_count < WorkflowExecution.max_attempts,
            or_(
                and_(
                    WorkflowExecution.status == ExecutionStatus.QUEUED.value,
                    WorkflowExecution.next_attempt_at <= now,
                ),
                and_(
                    WorkflowExecution.status.in_(
                        (ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value)
                    ),
                    WorkflowExecution.lease_expires_at <= now,
                ),
            ),
            or_(WorkflowExecution.exclusive.is_(False), ~exists(active_sibling)),
            *capability_clauses,
        )

        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            async with session.begin():
                registered = await session.execute(
                    select(WorkflowWorker.worker_id).where(
                        WorkflowWorker.worker_id == registration.worker_id,
                        WorkflowWorker.boot_token == registration.boot_token,
                        WorkflowWorker.status == "active",
                        WorkflowWorker.expires_at > now,
                    )
                )
                if registered.scalar_one_or_none() is None:
                    return None

                exhausted_result = await session.execute(
                    select(WorkflowExecution)
                    .where(
                        WorkflowExecution.cancel_requested.is_(False),
                        WorkflowExecution.status.in_(
                            (ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value)
                        ),
                        WorkflowExecution.lease_expires_at <= now,
                        WorkflowExecution.attempt_count >= WorkflowExecution.max_attempts,
                    )
                    .order_by(
                        WorkflowExecution.lease_expires_at,
                        WorkflowExecution.execution_id,
                    )
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                exhausted = exhausted_result.scalar_one_or_none()
                if exhausted is not None:
                    diagnostics = {
                        "type": "AttemptsExhausted",
                        "message": "last worker lease expired after the maximum attempt count",
                    }
                    await session.execute(
                        update(WorkflowExecutionAttempt)
                        .where(
                            WorkflowExecutionAttempt.execution_id
                            == exhausted.execution_id,
                            WorkflowExecutionAttempt.status.in_(("leased", "running")),
                        )
                        .values(
                            status="expired",
                            diagnostics=diagnostics,
                            completed_at=now,
                        )
                    )
                    exhausted.status = ExecutionStatus.FAILED.value
                    exhausted.diagnostics = diagnostics
                    exhausted.owner_worker_id = None
                    exhausted.owner_boot_token = None
                    exhausted.lease_expires_at = None
                    exhausted.completed_at = now
                    exhausted.updated_at = now

                result = await session.execute(
                    select(WorkflowExecution)
                    .where(eligible)
                    .order_by(
                        WorkflowExecution.priority.desc(),
                        WorkflowExecution.created_at,
                        WorkflowExecution.execution_id,
                    )
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                row = result.scalar_one_or_none()
                if row is None:
                    return None

                old_status = row.status
                old_fencing = row.fencing_token
                attempt_number = row.attempt_count + 1
                fencing_token = old_fencing + 1
                attempt_id = uuid.uuid4()

                claimed = await session.execute(
                    update(WorkflowExecution)
                    .where(
                        WorkflowExecution.execution_id == row.execution_id,
                        WorkflowExecution.status == old_status,
                        WorkflowExecution.fencing_token == old_fencing,
                        eligible,
                    )
                    .values(
                        status=ExecutionStatus.LEASED.value,
                        attempt_count=attempt_number,
                        fencing_token=fencing_token,
                        owner_worker_id=registration.worker_id,
                        owner_boot_token=registration.boot_token,
                        lease_expires_at=lease_expires_at,
                        updated_at=now,
                    )
                )
                if _rowcount(claimed) != 1:
                    return None

                if old_status in {ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value}:
                    await session.execute(
                        update(WorkflowExecutionAttempt)
                        .where(
                            WorkflowExecutionAttempt.execution_id == row.execution_id,
                            WorkflowExecutionAttempt.status.in_(("leased", "running")),
                        )
                        .values(status="expired", completed_at=now)
                    )

                session.add(
                    WorkflowExecutionAttempt(
                        attempt_id=attempt_id,
                        execution_id=row.execution_id,
                        attempt_number=attempt_number,
                        worker_id=registration.worker_id,
                        boot_token=registration.boot_token,
                        fencing_token=fencing_token,
                        status="leased",
                        lease_expires_at=lease_expires_at,
                        heartbeat_at=now,
                        created_at=now,
                    )
                )
                await session.flush()
                await session.refresh(row)
                capabilities = await self._capabilities(session, row.execution_id)
                record = _record(row, capabilities)

            return ExecutionClaim(
                execution=record,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                worker_id=registration.worker_id,
                boot_token=registration.boot_token,
                fencing_token=fencing_token,
                lease_expires_at=lease_expires_at,
            )

    @staticmethod
    def _owned(claim: ExecutionClaim) -> tuple[Any, ...]:
        return (
            WorkflowExecution.execution_id == claim.execution.execution_id,
            WorkflowExecution.owner_worker_id == claim.worker_id,
            WorkflowExecution.owner_boot_token == claim.boot_token,
            WorkflowExecution.fencing_token == claim.fencing_token,
            WorkflowExecution.cancel_requested.is_(False),
        )

    async def mark_running(
        self,
        claim: ExecutionClaim,
        *,
        lease_expires_at: datetime,
    ) -> bool:
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                result = await session.execute(
                    update(WorkflowExecution)
                    .where(*self._owned(claim), WorkflowExecution.status == ExecutionStatus.LEASED.value)
                    .values(
                        status=ExecutionStatus.RUNNING.value,
                        started_at=now,
                        lease_expires_at=lease_expires_at,
                        updated_at=now,
                    )
                )
                if _rowcount(result) != 1:
                    return False
                await session.execute(
                    update(WorkflowExecutionAttempt)
                    .where(
                        WorkflowExecutionAttempt.attempt_id == claim.attempt_id,
                        WorkflowExecutionAttempt.fencing_token == claim.fencing_token,
                        WorkflowExecutionAttempt.status == "leased",
                    )
                    .values(
                        status="running",
                        started_at=now,
                        heartbeat_at=now,
                        lease_expires_at=lease_expires_at,
                    )
                )
                return True

    async def heartbeat_execution(
        self,
        claim: ExecutionClaim,
        *,
        lease_expires_at: datetime,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        now = _utcnow()
        values: dict[str, Any] = {
            "lease_expires_at": lease_expires_at,
            "updated_at": now,
        }
        if progress is not None:
            values["progress"] = progress
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                result = await session.execute(
                    update(WorkflowExecution)
                    .where(
                        *self._owned(claim),
                        WorkflowExecution.status == ExecutionStatus.RUNNING.value,
                    )
                    .values(**values)
                )
                if _rowcount(result) != 1:
                    return False
                await session.execute(
                    update(WorkflowExecutionAttempt)
                    .where(
                        WorkflowExecutionAttempt.attempt_id == claim.attempt_id,
                        WorkflowExecutionAttempt.fencing_token == claim.fencing_token,
                        WorkflowExecutionAttempt.status == "running",
                    )
                    .values(heartbeat_at=now, lease_expires_at=lease_expires_at)
                )
                return True

    async def succeed(self, claim: ExecutionClaim, result: dict[str, Any]) -> bool:
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            async with session.begin():
                completed = await session.execute(
                    update(WorkflowExecution)
                    .where(
                        *self._owned(claim),
                        WorkflowExecution.status == ExecutionStatus.RUNNING.value,
                    )
                    .values(
                        status=ExecutionStatus.SUCCEEDED.value,
                        result=result,
                        diagnostics=None,
                        completed_at=now,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                if _rowcount(completed) != 1:
                    return False
                await session.execute(
                    update(WorkflowExecutionAttempt)
                    .where(
                        WorkflowExecutionAttempt.attempt_id == claim.attempt_id,
                        WorkflowExecutionAttempt.fencing_token == claim.fencing_token,
                        WorkflowExecutionAttempt.status == "running",
                    )
                    .values(status="succeeded", result=result, completed_at=now)
                )
                return True

    async def fail(
        self,
        claim: ExecutionClaim,
        diagnostics: dict[str, Any],
        *,
        retry_at: datetime | None,
    ) -> bool:
        now = _utcnow()
        retry = retry_at is not None and claim.attempt_number < claim.execution.max_attempts
        execution_values: dict[str, Any] = {
            "diagnostics": diagnostics,
            "owner_worker_id": None,
            "owner_boot_token": None,
            "lease_expires_at": None,
            "updated_at": now,
        }
        if retry:
            execution_values.update(
                status=ExecutionStatus.QUEUED.value,
                next_attempt_at=retry_at,
            )
        else:
            execution_values.update(
                status=ExecutionStatus.FAILED.value,
                completed_at=now,
            )

        async with AsyncSession(self._engine) as session:
            async with session.begin():
                failed = await session.execute(
                    update(WorkflowExecution)
                    .where(
                        *self._owned(claim),
                        WorkflowExecution.status.in_(
                            (ExecutionStatus.LEASED.value, ExecutionStatus.RUNNING.value)
                        ),
                    )
                    .values(**execution_values)
                )
                if _rowcount(failed) != 1:
                    return False
                await session.execute(
                    update(WorkflowExecutionAttempt)
                    .where(
                        WorkflowExecutionAttempt.attempt_id == claim.attempt_id,
                        WorkflowExecutionAttempt.fencing_token == claim.fencing_token,
                        WorkflowExecutionAttempt.status.in_(("leased", "running")),
                    )
                    .values(status="failed", diagnostics=diagnostics, completed_at=now)
                )
                return True

class SqlAlchemyExecutionArtifactStore(ExecutionArtifactStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def put(
        self,
        *,
        execution_id: uuid.UUID,
        name: str,
        content_type: str,
        content: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionArtifact:
        checksum = hashlib.sha256(content).hexdigest()
        now = _utcnow()
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            async with session.begin():
                result = await session.execute(
                    select(WorkflowExecutionArtifact)
                    .where(
                        WorkflowExecutionArtifact.execution_id == execution_id,
                        WorkflowExecutionArtifact.name == name,
                    )
                    .with_for_update()
                )
                row = result.scalar_one_or_none()
                if row is None:
                    row = WorkflowExecutionArtifact(
                        artifact_id=uuid.uuid4(),
                        execution_id=execution_id,
                        name=name,
                        content_type=content_type,
                        checksum=checksum,
                        content=content,
                        artifact_metadata=metadata or {},
                        created_at=now,
                    )
                    session.add(row)
                elif row.checksum != checksum:
                    raise ValueError(
                        f"artifact name {name!r} already exists with different content"
                    )
            return ExecutionArtifact(
                artifact_id=row.artifact_id,
                execution_id=row.execution_id,
                name=row.name,
                content_type=row.content_type,
                checksum=row.checksum,
                content=row.content,
                metadata=dict(row.artifact_metadata),
                created_at=row.created_at,
            )

    async def get(self, artifact_id: uuid.UUID) -> ExecutionArtifact | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(WorkflowExecutionArtifact, artifact_id)
            if row is None:
                return None
            return ExecutionArtifact(
                artifact_id=row.artifact_id,
                execution_id=row.execution_id,
                name=row.name,
                content_type=row.content_type,
                checksum=row.checksum,
                content=row.content,
                metadata=dict(row.artifact_metadata),
                created_at=row.created_at,
            )
