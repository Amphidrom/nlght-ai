# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Callable

from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

import nlght
from nlght.core.config.runtime_context import PersistenceSubsystemRuntime

logger = logging.getLogger(__name__)

_EngineFactory = Callable[..., AsyncEngine]


def _default_engine_factory(url: str, **kwargs: object) -> AsyncEngine:
    return create_async_engine(url, **kwargs)


class PostgresPersistenceSubsystem:
    """Manages a SQLAlchemy async engine for the runtime metadata persistence backend."""

    def __init__(
        self,
        runtime: PersistenceSubsystemRuntime,
        engine_factory: _EngineFactory | None = None,
        *,
        engine_options: dict[str, object] | None = None,
        verify_migrations: bool = True,
    ) -> None:
        self._runtime = runtime
        self._engine_factory = engine_factory or _default_engine_factory
        self._engine_options = engine_options or {}
        self._verify_migrations = verify_migrations
        self._engine: AsyncEngine | None = None

    async def start(self) -> None:
        self._engine = self._engine_factory(self._runtime.url, **self._engine_options)
        if self._verify_migrations:
            await self._assert_migrations_applied()

    async def _assert_migrations_applied(self) -> None:
        from alembic.config import Config  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        from alembic.runtime.migration import MigrationContext  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        from alembic.script import ScriptDirectory  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        
        with importlib.resources.path(nlght, 'alembic.ini') as alembic_ini_path:
            cfg = Config(alembic_ini_path)
            script = ScriptDirectory.from_config(cfg)
            heads = {r.revision for r in script.get_revisions("heads")}

            def _current_heads(sync_conn: Connection) -> set[str]:
                context = MigrationContext.configure(sync_conn)
                return set(context.get_current_heads())

            async with self.engine.connect() as conn:
                current = await conn.run_sync(_current_heads)

            if current != heads:
                raise RuntimeError(
                    f"DB migration required — run 'nlght-ai migrate' or "
                    f"'PYTHONPATH=src python -m alembic upgrade head'. "
                    f"Current: {current or '(none)'}, Expected: {heads}"
                )

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("PostgresPersistenceSubsystem has not been started")
        return self._engine
