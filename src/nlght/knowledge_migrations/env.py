"""Alembic migration environment for the knowledge database — async setup.

The target database is always given on the command line. Nothing is read from
configuration and no URL is stored in the ini file or here, because there is no
single knowledge database to name: it is reached through the resource
activations that use it, and two of them may point at different servers.

Usage:
    nlght-ai migrate-knowledge -x url=postgresql+asyncpg://user:pass@host/db
    nlght-ai migrate-knowledge -x url=... downgrade -1
    nlght-ai migrate-knowledge -x url=... upgrade head --sql   # offline SQL
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

# ---------------------------------------------------------------------------
# Import the metadata Alembic uses for autogenerate
# ---------------------------------------------------------------------------
from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

target_metadata = KnowledgeBase.metadata

# ---------------------------------------------------------------------------
# Alembic config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ---------------------------------------------------------------------------
# The target database is always passed in
# ---------------------------------------------------------------------------
#
# There is no configured knowledge URL to fall back on, by design. The knowledge
# database is reached at runtime only through the resource activations that use
# it, each carrying its own connection, and two activations may well point at
# different databases. A single platform-wide URL here would name one of them as
# "the" knowledge database and quietly become the thing everything else is
# measured against.
#
# So the migration target is stated explicitly, every time:
#
#     nlght-ai migrate-knowledge -x url=postgresql+asyncpg://user:pass@host/db

_x_args = context.get_x_argument(as_dictionary=True)
_db_url = _x_args.get("url")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)
elif not context.is_offline_mode():
    raise RuntimeError(
        "No target database given.\n"
        "Pass it explicitly:\n"
        "  nlght-ai migrate-knowledge -x url=postgresql+asyncpg://user:pass@host/db\n"
        "The knowledge database has no configuration entry: it is reached through "
        "the resource activations that use it, so the migration target is stated "
        "per invocation rather than inferred."
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
