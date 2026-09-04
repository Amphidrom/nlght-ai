# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from nlght.adapters.outbound.persistence.models import (
    AccessPolicyRule,
    Resource,
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)
from nlght.adapters.outbound.tools.registry import tool_registry
from nlght.adapters.outbound.workflow.registry import step_registry
from nlght.core.errors.errors import ResourceAddressAlreadyExists
from nlght.core.execution import (
    ExecutionRecord,
    ExecutionStatus,
    ExecutionStatusCounts,
    FanOutSummary,
    WorkerStatus,
)

#: How each dialect names the violated address constraint. PostgreSQL reports
#: the constraint by name; SQLite names the columns and never the constraint, so
#: matching on the name alone would translate the error on the deployment target
#: and not in any test. Both are listed rather than falling back to "any
#: IntegrityError is an address conflict", which would report a foreign-key or
#: not-null failure as a duplicate name.
_ADDRESS_CONSTRAINT = ("uq_resource_address", "resources.kind, resources.name")


def _address_conflict(
    error: IntegrityError, kind: str, name: str
) -> Exception:
    """The address conflict, or the original error if it was something else."""
    reported = str(getattr(error, "orig", error))
    if any(marker in reported for marker in _ADDRESS_CONSTRAINT):
        return ResourceAddressAlreadyExists(kind=kind, name=name)
    return error


class AdminService:
    """Configuration surface for the runtime: workflows, resources, policies.

    Deliberately does not reach into the knowledge database. Knowledge review is
    its own surface with its own backend (ADR-0032), and giving the runtime's
    configuration UI a second job would tie a reviewer's workflow to whether an
    operator's admin UI happens to be mounted.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    async def list_workflows(self) -> list[Workflow]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow)
                .options(selectinload(Workflow.versions))
                .order_by(Workflow.name)
            )
            return list(result.scalars().all())

    async def update_workflow(
        self,
        wid: uuid.UUID,
        *,
        name: str,
        description: str | None,
        enabled: bool,
        capabilities: list[str],
        concurrency: str,
        max_hops: int | None,
    ) -> Workflow | None:
        """Change a workflow's properties.

        The form for this was already written; there was no route behind it, so
        a workflow's name, capabilities, concurrency and now its hop budget
        could only ever be chosen when it was created. Versions and steps are
        edited elsewhere and are untouched here.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            workflow = await session.get(Workflow, wid)
            if workflow is None:
                return None
            workflow.name = name
            workflow.description = description or None
            workflow.enabled = enabled
            workflow.capabilities = capabilities
            workflow.concurrency = (
                concurrency if concurrency in ("blocking", "non-blocking") else "non-blocking"
            )
            workflow.max_hops = max_hops
            await session.flush()
            session.expunge(workflow)
            return workflow

    async def get_workflow(self, wid: uuid.UUID) -> Workflow | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow)
                .options(
                    selectinload(Workflow.versions).selectinload(WorkflowVersion.steps)
                )
                .where(Workflow.workflow_id == wid)
            )
            return result.scalar_one_or_none()

    async def create_workflow(
        self,
        name: str,
        description: str | None,
        enabled: bool,
        capabilities: list[str],
        concurrency: str = "non-blocking",
        max_hops: int | None = None,
    ) -> Workflow:
        async with AsyncSession(self._engine) as session:
            workflow = Workflow(
                workflow_id=uuid.uuid4(),
                name=name,
                description=description or None,
                enabled=enabled,
                capabilities=capabilities,
                concurrency=concurrency if concurrency in ("blocking", "non-blocking") else "non-blocking",
                max_hops=max_hops,
            )
            session.add(workflow)
            await session.flush()
            version = WorkflowVersion(
                workflow_version_id=uuid.uuid4(),
                workflow_id=workflow.workflow_id,
                version=1,
                status="draft",
            )
            session.add(version)
            await session.commit()
            await session.refresh(workflow)
            return workflow

    async def toggle_workflow(self, wid: uuid.UUID) -> Workflow:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow)
                .options(selectinload(Workflow.versions))
                .where(Workflow.workflow_id == wid)
            )
            workflow = result.scalar_one()
            workflow.enabled = not workflow.enabled
            await session.commit()
            await session.refresh(workflow)
            return workflow

    # ------------------------------------------------------------------
    # Workflow Versions
    # ------------------------------------------------------------------

    async def get_version(self, vid: uuid.UUID) -> WorkflowVersion | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowVersion)
                .options(
                    selectinload(WorkflowVersion.steps),
                    selectinload(WorkflowVersion.workflow),
                )
                .where(WorkflowVersion.workflow_version_id == vid)
            )
            return result.scalar_one_or_none()

    async def fork_latest_version(self, wid: uuid.UUID) -> WorkflowVersion:
        """Create a new draft version, copying steps from the highest existing version."""
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowVersion)
                .options(selectinload(WorkflowVersion.steps))
                .where(WorkflowVersion.workflow_id == wid)
                .order_by(WorkflowVersion.version.desc())
                .limit(1)
            )
            latest = result.scalar_one_or_none()
            new_version_nr = (latest.version + 1) if latest else 1

            new_version = WorkflowVersion(
                workflow_version_id=uuid.uuid4(),
                workflow_id=wid,
                version=new_version_nr,
                status="draft",
            )
            session.add(new_version)
            await session.flush()

            if latest:
                for step in latest.steps:
                    session.add(WorkflowStep(
                        workflow_step_id=uuid.uuid4(),
                        workflow_version_id=new_version.workflow_version_id,
                        position=step.position,
                        name=step.name,
                        type=step.type,
                        enabled=step.enabled,
                        config=dict(step.config),
                        transitions=dict(step.transitions),
                        is_terminal=step.is_terminal,
                        is_start=step.is_start,
                        is_resume=step.is_resume,
                    ))

            await session.commit()
            await session.refresh(new_version)
            return new_version

    async def activate_version(self, vid: uuid.UUID) -> None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowVersion).where(WorkflowVersion.workflow_version_id == vid)
            )
            version = result.scalar_one()
            await session.execute(
                update(WorkflowVersion)
                .where(WorkflowVersion.workflow_id == version.workflow_id)
                .values(status="draft")
            )
            version.status = "active"
            await session.commit()

    # ------------------------------------------------------------------
    # Workflow Steps
    # ------------------------------------------------------------------

    async def get_step(self, sid: uuid.UUID) -> WorkflowStep | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowStep).where(WorkflowStep.workflow_step_id == sid)
            )
            return result.scalar_one_or_none()

    async def create_step(
        self,
        vid: uuid.UUID,
        position: int,
        name: str,
        type_: str,
        enabled: bool,
        config: dict[str, Any],
        is_start: bool,
        is_terminal: bool,
        is_resume: bool,
    ) -> WorkflowStep:
        async with AsyncSession(self._engine) as session:
            step = WorkflowStep(
                workflow_step_id=uuid.uuid4(),
                workflow_version_id=vid,
                position=position,
                name=name,
                type=type_,
                enabled=enabled,
                config=config,
                transitions={},
                is_start=is_start,
                is_terminal=is_terminal,
                is_resume=is_resume,
            )
            session.add(step)
            await session.commit()
            await session.refresh(step)
            return step

    async def update_step(self, sid: uuid.UUID, **fields: object) -> WorkflowStep:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowStep).where(WorkflowStep.workflow_step_id == sid)
            )
            step = result.scalar_one()
            for key, value in fields.items():
                setattr(step, key, value)
            await session.commit()
            await session.refresh(step)
            return step

    async def delete_step(self, sid: uuid.UUID) -> None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowStep).where(WorkflowStep.workflow_step_id == sid)
            )
            step = result.scalar_one()
            await session.delete(step)
            await session.commit()

    async def save_graph(self, vid: uuid.UUID, transitions: dict[str, dict[str, str]]) -> None:
        """Overwrite transitions for all steps in a version from the graph editor output."""
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowStep).where(WorkflowStep.workflow_version_id == vid)
            )
            for step in result.scalars().all():
                step.transitions = transitions.get(str(step.workflow_step_id), {})
            await session.commit()

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    async def list_resources(self) -> list[Resource]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).order_by(Resource.kind, Resource.name)
            )
            return list(result.scalars().all())

    async def get_resource(self, rid: uuid.UUID) -> Resource | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(Resource.resource_id == rid)
            )
            return result.scalar_one_or_none()

    async def create_resource(
        self,
        name: str,
        kind: str,
        provider: str,
        config: dict[str, Any],
        enabled: bool,
    ) -> Resource:
        async with AsyncSession(self._engine) as session:
            resource = Resource(
                resource_id=uuid.uuid4(),
                name=name,
                kind=kind,
                provider=provider,
                config=config,
                enabled=enabled,
            )
            session.add(resource)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise _address_conflict(exc, kind, name) from exc
            await session.refresh(resource)
            return resource

    async def update_resource(self, rid: uuid.UUID, **fields: object) -> Resource:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(Resource.resource_id == rid)
            )
            resource = result.scalar_one()
            for key, value in fields.items():
                setattr(resource, key, value)
            # Read before the commit: a rollback expires the instance, and
            # reading an expired attribute afterwards is a lazy load in a place
            # that has no greenlet to run it in.
            attempted = (str(resource.kind), str(resource.name))
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise _address_conflict(exc, *attempted) from exc
            await session.refresh(resource)
            return resource

    async def toggle_resource(self, rid: uuid.UUID) -> Resource:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(Resource.resource_id == rid)
            )
            resource = result.scalar_one()
            resource.enabled = not resource.enabled
            await session.commit()
            await session.refresh(resource)
            return resource

    # ------------------------------------------------------------------
    # Access policies
    # ------------------------------------------------------------------

    async def list_policies(self) -> list[AccessPolicyRule]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).order_by(
                    AccessPolicyRule.subject_type,
                    AccessPolicyRule.priority.desc(),
                    AccessPolicyRule.subject,
                )
            )
            return list(result.scalars().all())

    async def get_policy(self, rule_id: uuid.UUID) -> AccessPolicyRule | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).where(AccessPolicyRule.rule_id == rule_id)
            )
            return result.scalar_one_or_none()

    async def create_policy(
        self,
        subject_type: str,
        subject: str,
        effect: str,
        conditions: dict[str, Any],
        priority: int,
        enabled: bool,
    ) -> AccessPolicyRule:
        async with AsyncSession(self._engine) as session:
            rule = AccessPolicyRule(
                rule_id=uuid.uuid4(),
                subject_type=subject_type,
                subject=subject,
                effect=effect,
                conditions=conditions,
                priority=priority,
                enabled=enabled,
            )
            session.add(rule)
            await session.commit()
            await session.refresh(rule)
            return rule

    async def update_policy(self, rule_id: uuid.UUID, **fields: object) -> AccessPolicyRule:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).where(AccessPolicyRule.rule_id == rule_id)
            )
            rule = result.scalar_one()
            for key, value in fields.items():
                setattr(rule, key, value)
            await session.commit()
            await session.refresh(rule)
            return rule

    async def toggle_policy(self, rule_id: uuid.UUID) -> AccessPolicyRule:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).where(AccessPolicyRule.rule_id == rule_id)
            )
            rule = result.scalar_one()
            rule.enabled = not rule.enabled
            await session.commit()
            await session.refresh(rule)
            return rule

    async def delete_policy(self, rule_id: uuid.UUID) -> None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).where(AccessPolicyRule.rule_id == rule_id)
            )
            rule = result.scalar_one()
            await session.delete(rule)
            await session.commit()

    # -- step type introspection ---------------------------------------------

    @staticmethod
    def step_types() -> list[dict[str, object]]:
        """Registered step types with the options each one understands.

        Lets the admin UI offer a typed form instead of asking an operator to
        recall a step's config keys and hand-write JSON.
        """
        types: list[dict[str, object]] = []
        for step_type, step_cls in sorted(step_registry._registry.items()):
            types.append(
                {
                    "type": step_type,
                    "doc": (step_cls.__doc__ or "").strip().split("\n")[0],
                    "options": [
                        {
                            "name": option.name,
                            "type": option.type,
                            "description": option.description,
                            "required": option.required,
                            "default": option.default,
                            "choices": option.choices,
                            "placeholder": option.placeholder,
                        }
                        for option in step_cls.options()
                    ],
                }
            )
        return types

    @staticmethod
    def resource_kinds() -> list[dict[str, object]]:
        """Registered tool kinds with the configuration each one understands.

        Same idea as ``step_types``: the resource form can then offer typed
        inputs instead of asking an operator to recall a tool's config keys.
        """
        kinds: list[dict[str, object]] = []
        seen: set[str] = set()
        for (kind, provider), tool_cls in sorted(tool_registry._registry.items()):
            if kind in seen:
                continue
            seen.add(kind)
            kinds.append(
                {
                    "kind": kind,
                    "provider": provider,
                    "doc": (tool_cls.__doc__ or "").strip().split("\n")[0],
                    "options": [
                        {
                            "name": option.name,
                            "type": option.type,
                            "description": option.description,
                            "required": option.required,
                            "default": option.default,
                            "choices": option.choices,
                            "placeholder": option.placeholder,
                            "secret": option.secret,
                        }
                        for option in tool_cls.options()
                    ],
                }
            )
        return kinds


class ExecutionObservationService:
    """Read-only view of the durable execution queue, for the admin UI.

    Separate from ``AdminService`` because it is a different job: that one edits
    configuration, this one only looks at what the runtime did with it. Nothing
    here writes.

    It builds its own repository on the **general** engine rather than borrowing
    the dispatcher's. The execution pool is deliberately tight — sized to the
    worker's concurrency, no overflow, a five-second timeout — because it exists
    to keep claims and heartbeats prompt. An operator holding a page open on a
    refresh interval would compete with the worker loop for those connections
    and could stall the very claims the page is there to display. Same database,
    different pool.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        from nlght.adapters.outbound.persistence.execution_repository import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            SqlAlchemyExecutionRepository,
        )

        self._engine = engine
        self._executions = SqlAlchemyExecutionRepository(engine)

    async def _workflow_names(self, workflow_ids: set[uuid.UUID]) -> dict[uuid.UUID, str]:
        """Names for a page of runs — an execution stores only the id."""
        if not workflow_ids:
            return {}
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow.workflow_id, Workflow.name).where(
                    Workflow.workflow_id.in_(workflow_ids)
                )
            )
            return {workflow_id: name for workflow_id, name in result.all()}

    async def list_workflows_for_filter(self) -> list[Workflow]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(select(Workflow).order_by(Workflow.name))
            return list(result.scalars().all())

    async def runs(
        self,
        *,
        status: ExecutionStatus | None = None,
        workflow_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        records = await self._executions.list_runs(
            status=status, workflow_id=workflow_id, limit=limit, offset=offset
        )
        total = await self._executions.count_runs(status=status, workflow_id=workflow_id)
        names = await self._workflow_names({record.workflow_id for record in records})
        summaries = await self._executions.fan_out_summaries(
            tuple(record.execution_id for record in records)
        )
        return {
            "runs": [
                {
                    "record": record,
                    "workflow_name": names.get(record.workflow_id, str(record.workflow_id)),
                    # Absent for a run that never fanned out, which is most of them.
                    "fan_out": summaries.get(record.execution_id),
                    # What the *run* is doing. A distribution flow's own execution
                    # succeeds as soon as it has handed the work out, so its
                    # status is not the run's.
                    "state": run_state(record, summaries.get(record.execution_id)),
                    # How long the *run* took, which for a fan-out ends with its
                    # last child rather than with its own execution.
                    "timing": run_timing(record, summaries.get(record.execution_id)),
                }
                for record in records
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def run(
        self, execution_id: uuid.UUID, *, child_limit: int = 100, child_offset: int = 0
    ) -> dict[str, Any] | None:
        record = await self._executions.get(execution_id)
        if record is None:
            return None
        names = await self._workflow_names({record.workflow_id})
        fan_out = await self._executions.fan_out_summary(execution_id)
        return {
            "record": record,
            "workflow_name": names.get(record.workflow_id, str(record.workflow_id)),
            "state": run_state(record, fan_out if fan_out.total else None),
            "timing": run_timing(record, fan_out if fan_out.total else None),
            # The history of how it got here: which worker held which attempt,
            # and what each failure said.
            "attempts": await self._executions.attempts(execution_id),
            "fan_out": fan_out if fan_out.total else None,
            "children": (
                await self._executions.list_children(
                    execution_id, limit=child_limit, offset=child_offset
                )
                if fan_out.total
                else ()
            ),
            "child_limit": child_limit,
            "child_offset": child_offset,
        }

    async def workers(self) -> tuple[WorkerStatus, ...]:
        return await self._executions.list_workers()

    async def counts(self, *, within: timedelta | None = None) -> ExecutionStatusCounts:
        since = datetime.now(UTC) - within if within is not None else None
        return await self._executions.status_counts(since=since)


@dataclass(slots=True, frozen=True)
class RunState:
    """What a *run* is doing, which is not always what its execution is doing.

    A distribution flow submits one child per document batch and ends — it must,
    because waiting would hold a worker for as long as the corpus takes. Its own
    execution therefore succeeds within a second or two while the work it handed
    out has barely started. Reporting that execution's status as the run's is how
    a knowledge ingestion comes to read `succeeded` with a hundred children still
    running.

    So a run that fanned out is described by its children, and the execution's
    own status stays available beside it rather than being replaced by this.
    """

    label: str
    badge: str
    detail: str


def run_state(record: ExecutionRecord, fan_out: FanOutSummary | None) -> RunState:
    """Derive what to show for one run from its execution and its children."""
    status = record.status
    if fan_out is None or fan_out.total == 0:
        return RunState(status.value, status.value, "")
    if status is not ExecutionStatus.SUCCEEDED:
        # The distribution itself is queued, running, failed or cancelled. That
        # is the run's state regardless of what any child is doing.
        return RunState(status.value, status.value, "the distribution itself")
    if fan_out.pending:
        return RunState(
            "in progress",
            "running",
            f"handed out {fan_out.total} children and ended; {fan_out.pending} still to finish",
        )
    if fan_out.failed:
        return RunState(
            "children failed",
            "failed",
            f"{fan_out.failed} of {fan_out.total} children failed",
        )
    if fan_out.cancelled:
        return RunState(
            "partly cancelled",
            "cancelled",
            f"{fan_out.cancelled} of {fan_out.total} children were cancelled",
        )
    return RunState("succeeded", "succeeded", f"all {fan_out.total} children succeeded")


def _aware(moment: datetime | None) -> datetime | None:
    """Read a stored timestamp as UTC when the backend dropped its offset.

    The column is timezone-aware and Postgres hands it back that way. SQLite has
    nowhere to keep the offset, so the same value returns naive — and comparing
    it against ``now`` would raise rather than be wrong, which at least is loud.
    Everything written here is UTC, so saying so is a restoration, not a guess.
    """
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


@dataclass(slots=True, frozen=True)
class RunTiming:
    """How long a run took, measured over the whole run rather than one execution.

    A distribution flow's execution starts and completes in a second or two; the
    corpus it handed out takes hours. Its ``completed_at`` is therefore not when
    the ingest finished, and reading it as such is the same mistake as reading
    its status as the run's — the run is over when the last child is.
    """

    started_at: datetime | None
    finished_at: datetime | None
    """When the whole run ended. None while anything is still to finish."""
    elapsed: timedelta | None
    """Start to finish — or to now, while it is still going."""
    running: bool
    child_time: timedelta | None
    """Time the children spent between them; None for a run that never fanned out."""

    @property
    def parallel_factor(self) -> float | None:
        """How much more work was done than time passed.

        The gap between the two durations is the whole point of fanning out, and
        stating it saves the reader dividing: forty minutes of child time inside
        a six-minute run is roughly seven workers' worth. Below 1.1 there is
        nothing to report — one worker did it, or the run spent its time waiting.
        """
        if self.child_time is None or self.elapsed is None:
            return None
        seconds = self.elapsed.total_seconds()
        if seconds <= 0:
            return None
        factor = self.child_time.total_seconds() / seconds
        return factor if factor >= 1.1 else None


def run_timing(
    record: ExecutionRecord, fan_out: FanOutSummary | None, *, now: datetime | None = None
) -> RunTiming:
    """Derive a run's timing from its execution and, if it fanned out, its children."""
    moment = now or datetime.now(UTC)
    # The parent is the honest beginning: no child can start before it did. It
    # falls back to submission time for a run still waiting to be claimed.
    started = _aware(record.started_at) or _aware(record.created_at)

    if fan_out is None or fan_out.total == 0:
        finished = _aware(record.completed_at)
        return RunTiming(
            started_at=started,
            finished_at=finished,
            elapsed=(finished or moment) - started if started else None,
            running=finished is None,
            child_time=None,
        )

    finished = _aware(fan_out.finished_at)
    # A child that outlives the parent decides the end; a parent that somehow
    # outlives every child decides it instead. Whichever is later is the finish.
    parent_completed = _aware(record.completed_at)
    if finished is not None and parent_completed is not None:
        finished = max(finished, parent_completed)
    return RunTiming(
        started_at=started,
        finished_at=finished,
        elapsed=(finished or moment) - started if started else None,
        running=finished is None,
        child_time=fan_out.child_time,
    )
