# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Connection pools shared by the store activations of one runtime.

A connection pool is a property of the process, not of the call that happens to
need it first. Store activations used to ignore that: every activation built its
own SQLAlchemy engine and its own Qdrant/OpenSearch client in ``__init__``, and
activations are constructed per catalog build *and* per step invocation — so an
``ingestion.write`` step that re-enters itself once per document batch left one
abandoned engine, with live asyncpg connections, behind per document.

Nothing disposed them. They sit in reference cycles, so only a generational
collection reclaims them, in bursts — and ``asyncpg.Connection.__del__`` calls
``terminate()``, which aborts the socket's pending overlapped read. On Windows
that finalizer, running in whichever thread triggered the collection, races
``IocpProactor._poll``: the future it is holding turns ``CANCELLED`` between the
``done()`` check and ``set_exception``, and the ``InvalidStateError`` takes the
whole event loop down. That is what killed a worker process mid-ingestion.

So pools live here, keyed by the configuration that defines them, and the
container disposes them when it stops. Two activations pointing at the same
database share one pool; two pointing at different ones do not. Sharing a pool
is not sharing state: an activation's index-state records are keyed by its
``target``, so which pool carried the query never enters into it.

An activation constructed without a registry makes a private one, which
reproduces the old per-instance behaviour for direct and test construction.
Nothing disposes a private registry, so it is not for runtime use.
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
    Nothing is opened eagerly — a pool appears when an activation first asks for
    it, and a deployment pays only for the backends it actually uses.
    """

    def __init__(self) -> None:
        self._engines: dict[ClientKey, AsyncEngine] = {}
        self._clients: dict[ClientKey, Any] = {}
        self._closers: dict[ClientKey, Callable[[Any], None]] = {}
        self._bootstrapped: set[ClientKey] = set()

    # -- pools ---------------------------------------------------------------

    def engine(self, url: str, **options: Any) -> AsyncEngine:  # noqa: ANN401 (SQLAlchemy engine options are open)
        """The async engine for ``url``, created once per configuration."""
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
        """The client for ``key``, built by ``factory`` on first use.

        ``close`` is how this client is shut down; without it the client is
        simply dropped, which is right for one that owns no socket.
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
        """Whether this collection/index still has to be created in this process."""
        return tuple(scope) not in self._bootstrapped

    def mark_bootstrapped(self, *scope: object) -> None:
        """Record that it exists — after the call that created it returned."""
        self._bootstrapped.add(tuple(scope))

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Nothing to do: pools are opened on first use, not at boot."""
        return None

    async def stop(self) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Dispose every pool. A failure to close one must not skip the rest."""
        for key, client in self._clients.items():
            closer = self._closers.get(key)
            if closer is None:
                continue
            try:
                closer(client)
            except Exception:  # noqa: BLE001 - closing a broken client is not news
                logger.debug("store.connections.close_failed | kind=%s", key[0], exc_info=True)
        self._clients.clear()
        self._closers.clear()

        for engine in self._engines.values():
            try:
                await engine.dispose()
            except Exception:  # noqa: BLE001 - same
                logger.debug("store.connections.dispose_failed", exc_info=True)
        self._engines.clear()
        self._bootstrapped.clear()
