# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from nlght.adapters.outbound.persistence.models import Resource
from nlght.core.runtime.resource import ResourceDef
from nlght.ports.outbound.resource_repository import ResourceRepository


def _to_resource_def(resource: Resource) -> ResourceDef:
    return ResourceDef(
        resource_id=resource.resource_id,
        name=resource.name,
        kind=resource.kind,
        provider=resource.provider,
        config=resource.config,
        enabled=resource.enabled,
    )


class SqlAlchemyResourceRepository(ResourceRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def find_by_id(self, resource_id: uuid.UUID) -> ResourceDef | None:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(Resource.resource_id == resource_id)
            )
            row = result.scalar_one_or_none()
            return _to_resource_def(row) if row is not None else None

    async def find_by_kind(self, kind: str) -> list[ResourceDef]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(
                    Resource.kind == kind,
                    Resource.enabled.is_(True),
                )
            )
            return [_to_resource_def(row) for row in result.scalars()]

    async def list_enabled(self) -> list[ResourceDef]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(Resource).where(Resource.enabled.is_(True))
            )
            return [_to_resource_def(row) for row in result.scalars()]
