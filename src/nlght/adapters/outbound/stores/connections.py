# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Share store connection pools across activations in one process.

Pools open lazily and close with the runtime.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

ClientKey = tuple[object, ...]
"""What distinguishes one client from another: kind, endpoint, credentials, timeout."""


class StoreConnections:
    """The store connection pools of one runtime, keyed by configuration.

    Implements ``SubsystemLifecycle`` so the container disposes it on shutdown.
    Nothing is opened eagerly. A pool appears when an activation first asks for
    it, and a deployment pays only for the backends it actually uses.
    """

    def __init__(self) -> None:
        self._engines: dict[ClientKey, AsyncEngine] = {}
        self._clients: dict[ClientKey, Any] = {}
        self._closers: dict[ClientKey, Callable[[Any], None]] = {}
        self._bootstrapped: set[ClientKey] = set()

    # -- pools ---------------------------------------------------------------

    def engine(self, url: str, **options: Any) -> AsyncEngine:  # noqa: ANN401 (SQLAlchemy engine options are open)
        """Reuse an async engine for one URL and option set.

        Args:
            url (str): Database connection URL.
            options (Any): SQLAlchemy engine options included in the pool key.

        Returns:
            AsyncEngine: Shared engine for the exact configuration.

        Side Effects:
            Creates and records an engine on first use.
        """
        key: ClientKey = (url, tuple(sorted(options.items())))
        engine = self._engines.get(key)
        if engine is None:
            engine = create_async_engine(url, **options)
            self._engines[key] = engine
            logger.debug("store.connections.engine | pools=%d", len(self._engines))
        return engine

    def client(
        self,
        key: ClientKey,
        factory: Callable[[], Any],
        *,
        close: Callable[[Any], None] | None = None,
    ) -> Any:  # noqa: ANN401 (qdrant-client and opensearch-py are untyped)
        """Reuse a backend client identified by its configuration key.

        Args:
            key (ClientKey): Backend kind, endpoint, and connection options.
            factory (Callable[[], Any]): Client constructor called on first use.
            close (Callable[[Any], None] | None): Optional shutdown callback.

        Returns:
            Any: Shared client for this key.

        Side Effects:
            May construct a client and register its shutdown callback.
        """
        client = self._clients.get(key)
        if client is None:
            client = factory()
            self._clients[key] = client
            if close is not None:
                self._closers[key] = close
            logger.debug("store.connections.client | kind=%s clients=%d", key[0], len(self._clients))
        return client

    # -- one-time bootstrap --------------------------------------------------

    def needs_bootstrap(self, *scope: object) -> bool:
        """Check whether a collection or index needs process-local setup.

        Args:
            scope (object): Backend and target components of the setup key.

        Returns:
            bool: True until this scope has been marked ready.
        """
        return tuple(scope) not in self._bootstrapped

    def mark_bootstrapped(self, *scope: object) -> None:
        """Record successful process-local setup for a target.

        Args:
            scope (object): Backend and target components of the setup key.

        Side Effects:
            Marks the scope ready after the backend setup succeeds.
        """
        self._bootstrapped.add(tuple(scope))

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the lifecycle without eagerly opening store pools."""
        return None

    async def stop(self) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Dispose every backend client and database pool.

        Side Effects:
            Invokes registered client closers and disposes engines. Logs a
            failure for one resource while continuing with the others.
        """
        for key, client in self._clients.items():
            closer = self._closers.get(key)
            if closer is None:
                continue
            try:
                closer(client)
            except Exception:
                logger.debug("store.connections.close_failed | kind=%s", key[0], exc_info=True)
        self._clients.clear()
        self._closers.clear()

        for engine in self._engines.values():
            try:
                await engine.dispose()
            except Exception:
                logger.debug("store.connections.dispose_failed", exc_info=True)
        self._engines.clear()
        self._bootstrapped.clear()
