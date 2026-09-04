"""Alembic migration environment — async SQLAlchemy setup.

The database URL is read from the same platform.yaml that the runtime uses.
No URL is stored in alembic.ini or in this file.

Config resolution order:
  1. NLGHT_CONFIG env var (path to the YAML file)
  2. Default: .config/platform.yaml  (relative to the project root)

Usage:
    alembic upgrade head
    alembic revision --autogenerate -m "add_my_column"
    alembic downgrade -1
    alembic upgrade head --sql        # offline SQL output without a live DB
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from pathlib import Path
from typing import Any

import yaml
from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

# ---------------------------------------------------------------------------
# Import the metadata Alembic uses for autogenerate
# ---------------------------------------------------------------------------
from nlght.adapters.outbound.persistence.models import Base

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Alembic config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ---------------------------------------------------------------------------
# Read DB URL from platform.yaml
# ---------------------------------------------------------------------------

def _load_db_url() -> str | None:
    """Find the persistence URL in platform.yaml (integrations.persistence.workflows.url)."""
    config_path_env = os.environ.get("NLGHT_CONFIG")
    if config_path_env:
        yaml_path = Path(config_path_env)
    else:
        # Walk up from the migrations/ directory to find the project root
        yaml_path = Path(__file__).parent.parent / ".config" / "platform.yaml"

    if not yaml_path.exists():
        return None

    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text()) or {}
    url = (
        (raw.get("integrations") or {})
        .get("persistence", {})
        .get("workflows", {})
        .get("url")
    )
    return str(url) if url else None


# `-x url=...` overrides the configured URL, so a database can be migrated
# without a platform.yaml — what a CI job or a one-off setup usually needs.
_x_args = context.get_x_argument(as_dictionary=True)
_db_url = _x_args.get("url") or _load_db_url()
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)
elif not context.is_offline_mode():
    _config_hint = os.environ.get("NLGHT_CONFIG", ".config/platform.yaml")
    raise RuntimeError(
        f"No persistence URL found.\n"
        f"Expected 'integrations.persistence.workflows.url' to be set\n"
        f"in: {_config_hint}\n"
        f"Override config path with: export NLGHT_CONFIG=/path/to/platform.yaml"
    )


# ---------------------------------------------------------------------------
# Offline mode: generate SQL without a live connection
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Emit migration SQL to stdout without connecting to the database.

    Run with: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode: connect and apply migrations
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations against a live database using an async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: each migration run gets a fresh connection —
        # avoids connection pool issues in short-lived CLI processes.
        poolclass=NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
