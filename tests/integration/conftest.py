# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Integration test fixtures.

All integration tests share a real SQLite in-memory database wired through
the full SQLAlchemy stack.  Helper functions (seed_workflow, build_test_container,
etc.) live in integration._helpers and are imported directly by test modules.
"""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nlght.adapters.outbound.persistence.models import Base
from nlght.adapters.outbound.persistence.resource_repository import SqlAlchemyResourceRepository
from nlght.adapters.outbound.persistence.workflow_repository import SqlAlchemyWorkflowRepository

# This module imports optional store SDKs and probes three live services during
# collection. Ordinary directory-based local and CI runs do not collect it. A
# maintainer can still select the file itself explicitly when the stack exists.
collect_ignore = ["test_corpus_authority_backends.py"]


@pytest.fixture
async def sqlite_engine():
    """In-memory SQLite engine with the full schema created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine):
    return async_sessionmaker(sqlite_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def workflow_repo(sqlite_engine):
    return SqlAlchemyWorkflowRepository(sqlite_engine)


@pytest.fixture
def resource_repo(sqlite_engine):
    return SqlAlchemyResourceRepository(sqlite_engine)
