# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nlght.adapters.outbound.persistence import postgres
from nlght.adapters.outbound.persistence.postgres import PostgresPersistenceSubsystem
from nlght.core.config.runtime_context import PersistenceSubsystemRuntime


def _runtime(url: str = "postgresql+asyncpg://user:pass@localhost/db") -> PersistenceSubsystemRuntime:
    return PersistenceSubsystemRuntime(
        name="persistence",
        backend="postgres",
        url=url,
        runtime_metadata={},
        hive_mind_provider_name=None,
    )


class _StubEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


async def test_start_creates_engine() -> None:
    engine = _StubEngine()

    def fake_factory(url: str, **_) -> _StubEngine:
        assert url == "postgresql+asyncpg://user:pass@localhost/db"
        return engine

    subsystem = PostgresPersistenceSubsystem(_runtime(), engine_factory=fake_factory)
    with patch.object(PostgresPersistenceSubsystem, "_assert_migrations_applied", new=AsyncMock()):
        await subsystem.start()

    assert subsystem.engine is engine


async def test_stop_disposes_engine() -> None:
    engine = _StubEngine()

    def fake_factory(url: str, **_) -> _StubEngine:
        return engine

    subsystem = PostgresPersistenceSubsystem(_runtime(), engine_factory=fake_factory)
    with patch.object(PostgresPersistenceSubsystem, "_assert_migrations_applied", new=AsyncMock()):
        await subsystem.start()
    await subsystem.stop()

    assert engine.disposed
    with pytest.raises(RuntimeError, match="not been started"):
        _ = subsystem.engine


async def test_stop_is_safe_before_start() -> None:
    subsystem = PostgresPersistenceSubsystem(_runtime())
    await subsystem.stop()  # must not raise


async def test_engine_raises_before_start() -> None:
    subsystem = PostgresPersistenceSubsystem(_runtime())
    with pytest.raises(RuntimeError, match="not been started"):
        _ = subsystem.engine


class _PathContext:
    def __enter__(self) -> str:
        return "alembic.ini"

    def __exit__(self, *_: object) -> None:
        return None


class _ConnectionContext:
    def __init__(self, current_heads: set[str]) -> None:
        self.current_heads = current_heads

    async def __aenter__(self) -> _ConnectionContext:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def run_sync(self, fn):
        return fn(SimpleNamespace(current_heads=self.current_heads))


class _MigrationEngine:
    def __init__(self, current_heads: set[str]) -> None:
        self.current_heads = current_heads

    def connect(self) -> _ConnectionContext:
        return _ConnectionContext(self.current_heads)


async def test_assert_migrations_applied_accepts_matching_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postgres.importlib.resources, "path", lambda *_: _PathContext())

    class _Config:
        def __init__(self, path: str) -> None:
            self.path = path

    class _ScriptDirectory:
        @classmethod
        def from_config(cls, cfg: _Config) -> _ScriptDirectory:
            return cls()

        def get_revisions(self, value: str) -> list[object]:
            assert value == "heads"
            return [SimpleNamespace(revision="rev-1")]

    class _MigrationContext:
        @classmethod
        def configure(cls, sync_conn: object) -> object:
            return SimpleNamespace(get_current_heads=lambda: sync_conn.current_heads)

    monkeypatch.setitem(__import__("sys").modules, "alembic.config", SimpleNamespace(Config=_Config))
    monkeypatch.setitem(__import__("sys").modules, "alembic.script", SimpleNamespace(ScriptDirectory=_ScriptDirectory))
    monkeypatch.setitem(
        __import__("sys").modules,
        "alembic.runtime.migration",
        SimpleNamespace(MigrationContext=_MigrationContext),
    )

    subsystem = PostgresPersistenceSubsystem(_runtime())
    subsystem._engine = _MigrationEngine({"rev-1"})

    await subsystem._assert_migrations_applied()


async def test_assert_migrations_applied_rejects_stale_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postgres.importlib.resources, "path", lambda *_: _PathContext())

    class _Config:
        def __init__(self, path: str) -> None:
            self.path = path

    class _ScriptDirectory:
        @classmethod
        def from_config(cls, cfg: _Config) -> _ScriptDirectory:
            return cls()

        def get_revisions(self, value: str) -> list[object]:
            return [SimpleNamespace(revision="expected")]

    class _MigrationContext:
        @classmethod
        def configure(cls, sync_conn: object) -> object:
            return SimpleNamespace(get_current_heads=lambda: sync_conn.current_heads)

    monkeypatch.setitem(__import__("sys").modules, "alembic.config", SimpleNamespace(Config=_Config))
    monkeypatch.setitem(__import__("sys").modules, "alembic.script", SimpleNamespace(ScriptDirectory=_ScriptDirectory))
    monkeypatch.setitem(
        __import__("sys").modules,
        "alembic.runtime.migration",
        SimpleNamespace(MigrationContext=_MigrationContext),
    )

    subsystem = PostgresPersistenceSubsystem(_runtime())
    subsystem._engine = _MigrationEngine({"current"})

    with pytest.raises(RuntimeError, match="DB migration required"):
        await subsystem._assert_migrations_applied()
