# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from nlght.adapters.outbound.persistence.models import (
    Base,
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)
from nlght.adapters.outbound.persistence.workflow_repository import SqlAlchemyWorkflowRepository


@pytest.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def seeded_engine(engine):
    workflow_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with AsyncSession(engine) as session:
        async with session.begin():
            wf = Workflow(
                workflow_id=workflow_id,
                name="greet",
                enabled=True,
                capabilities=["greeting"],
            )
            session.add(wf)
            version = WorkflowVersion(
                workflow_version_id=version_id,
                workflow_id=workflow_id,
                version=1,
                status="active",
            )
            session.add(version)
            session.add(WorkflowStep(
                workflow_step_id=uuid.uuid4(),
                workflow_version_id=version_id,
                position=0,
                name="start",
                type="llm",
                enabled=True,
                config={"prompt": "Hello"},
                transitions={"done": "end"},
                is_start=True,
                is_terminal=False,
                is_resume=False,
            ))
            session.add(WorkflowStep(
                workflow_step_id=uuid.uuid4(),
                workflow_version_id=version_id,
                position=1,
                name="end",
                type="terminal",
                enabled=True,
                config={},
                transitions={},
                is_start=False,
                is_terminal=True,
                is_resume=False,
            ))
            session.add(Workflow(
                workflow_id=uuid.uuid4(),
                name="disabled",
                enabled=False,
                capabilities=[],
            ))

    return engine, workflow_id, version_id


async def test_find_by_name_returns_workflow(seeded_engine) -> None:
    engine, workflow_id, _ = seeded_engine
    repo = SqlAlchemyWorkflowRepository(engine)

    result = await repo.find_by_name("greet")

    assert result is not None
    assert result.name == "greet"
    assert result.workflow_id == workflow_id
    assert result.enabled is True


async def test_find_by_name_returns_none_for_unknown(seeded_engine) -> None:
    engine, _, _ = seeded_engine
    repo = SqlAlchemyWorkflowRepository(engine)

    assert await repo.find_by_name("nonexistent") is None


async def test_list_enabled_excludes_disabled(seeded_engine) -> None:
    engine, _, _ = seeded_engine
    repo = SqlAlchemyWorkflowRepository(engine)

    results = await repo.list_enabled()

    assert len(results) == 1
    assert results[0].name == "greet"


async def test_find_active_version_with_steps(seeded_engine) -> None:
    engine, workflow_id, version_id = seeded_engine
    repo = SqlAlchemyWorkflowRepository(engine)

    version = await repo.find_active_version(workflow_id)

    assert version is not None
    assert version.version_id == version_id
    assert version.status == "active"
    assert len(version.steps) == 2
    start_step = next(s for s in version.steps if s.is_start)
    assert start_step.name == "start"
    assert start_step.transitions == {"done": "end"}


async def test_find_active_version_returns_none_for_unknown(seeded_engine) -> None:
    engine, _, _ = seeded_engine
    repo = SqlAlchemyWorkflowRepository(engine)

    assert await repo.find_active_version(uuid.uuid4()) is None
