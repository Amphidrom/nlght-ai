# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from nlght.adapters.outbound.persistence.models import (
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)
from nlght.core.workflow.workflow import WorkflowDef, WorkflowStepDef, WorkflowVersionDef
from nlght.ports.outbound.workflow_repository import WorkflowRepository


def _to_step_def(step: WorkflowStep) -> WorkflowStepDef:
    return WorkflowStepDef(
        step_id=step.workflow_step_id,
        position=step.position,
        name=step.name,
        type=step.type,
        enabled=step.enabled,
        config=step.config,
        transitions=step.transitions,
        is_start=step.is_start,
        is_terminal=step.is_terminal,
        is_resume=step.is_resume,
    )


def _to_version_def(version: WorkflowVersion) -> WorkflowVersionDef:
    return WorkflowVersionDef(
        version_id=version.workflow_version_id,
        workflow_id=version.workflow_id,
        version=version.version,
        status=version.status,
        steps=[_to_step_def(s) for s in version.steps],
    )


def _to_workflow_def(workflow: Workflow) -> WorkflowDef:
    return WorkflowDef(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        enabled=workflow.enabled,
        capabilities=workflow.capabilities,
        max_hops=workflow.max_hops,
        description=workflow.description,
        concurrency=workflow.concurrency,
    )


class SqlAlchemyWorkflowRepository(WorkflowRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def find_by_name(self, name: str) -> WorkflowDef | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow).where(Workflow.name == name)
            )
            row = result.scalar_one_or_none()
            return _to_workflow_def(row) if row is not None else None

    async def find_by_id(self, workflow_id: uuid.UUID) -> WorkflowDef | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(Workflow, workflow_id)
            return _to_workflow_def(row) if row is not None else None

    async def list_enabled(self) -> list[WorkflowDef]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Workflow).where(Workflow.enabled.is_(True))
            )
            return [_to_workflow_def(row) for row in result.scalars()]

    async def find_active_version(
        self, workflow_id: uuid.UUID
    ) -> WorkflowVersionDef | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowVersion)
                .where(
                    WorkflowVersion.workflow_id == workflow_id,
                    WorkflowVersion.status == "active",
                )
                .options(selectinload(WorkflowVersion.steps))
            )
            row = result.scalar_one_or_none()
            return _to_version_def(row) if row is not None else None

    async def find_version(
        self,
        workflow_id: uuid.UUID,
        workflow_version_id: uuid.UUID,
    ) -> WorkflowVersionDef | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(WorkflowVersion)
                .where(
                    WorkflowVersion.workflow_id == workflow_id,
                    WorkflowVersion.workflow_version_id == workflow_version_id,
                )
                .options(selectinload(WorkflowVersion.steps))
            )
            row = result.scalar_one_or_none()
            return _to_version_def(row) if row is not None else None
