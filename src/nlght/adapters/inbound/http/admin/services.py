# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import selectinload

from nlght.adapters.outbound.persistence.models import (
    AccessPolicyRule,
    Resource,
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)


class AdminService:
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
    ) -> Workflow:
        async with AsyncSession(self._engine) as session:
            workflow = Workflow(
                workflow_id=uuid.uuid4(),
                name=name,
                description=description or None,
                enabled=enabled,
                capabilities=capabilities,
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
            await session.commit()
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
            await session.commit()
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
