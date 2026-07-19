# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from nlght.adapters.outbound.persistence.models import Base, Resource
from nlght.adapters.outbound.persistence.resource_repository import SqlAlchemyResourceRepository


@pytest.fixture
async def seeded_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    tool_id = uuid.uuid4()
    disabled_id = uuid.uuid4()

    async with AsyncSession(engine) as session:
        async with session.begin():
            session.add(Resource(
                resource_id=tool_id,
                name="python-runner",
                kind="tool",
                provider="local",
                config={"command": "python"},
                enabled=True,
            ))
            session.add(Resource(
                resource_id=uuid.uuid4(),
                name="bash-runner",
                kind="tool",
                provider="local",
                config={},
                enabled=True,
            ))
            session.add(Resource(
                resource_id=disabled_id,
                name="old-tool",
                kind="tool",
                provider="local",
                config={},
                enabled=False,
            ))

    yield engine, tool_id, disabled_id
    await engine.dispose()


async def test_find_by_id_returns_resource(seeded_engine) -> None:
    engine, tool_id, _ = seeded_engine
    repo = SqlAlchemyResourceRepository(engine)

    result = await repo.find_by_id(tool_id)

    assert result is not None
    assert result.name == "python-runner"
    assert result.kind == "tool"
    assert result.config == {"command": "python"}


async def test_find_by_id_returns_none_for_unknown(seeded_engine) -> None:
    engine, _, _ = seeded_engine
    repo = SqlAlchemyResourceRepository(engine)

    assert await repo.find_by_id(uuid.uuid4()) is None


async def test_find_by_kind_excludes_disabled(seeded_engine) -> None:
    engine, _, _ = seeded_engine
    repo = SqlAlchemyResourceRepository(engine)

    results = await repo.find_by_kind("tool")

    assert len(results) == 2
    assert all(r.enabled for r in results)


async def test_list_enabled_excludes_disabled(seeded_engine) -> None:
    engine, _, disabled_id = seeded_engine
    repo = SqlAlchemyResourceRepository(engine)

    results = await repo.list_enabled()

    assert all(r.enabled for r in results)
    assert not any(r.resource_id == disabled_id for r in results)
