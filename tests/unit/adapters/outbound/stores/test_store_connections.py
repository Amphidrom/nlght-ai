# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Pools belong to the process, not to the call that first needed one.

The defect these cover: every store activation built its own engine and clients
in ``__init__``, and activations are constructed per catalog build and per step
hop. An ``ingestion.write`` step that re-enters itself once per document batch
therefore abandoned one engine — with live asyncpg connections — per document.
Nothing disposed them, and their finalizers aborted socket reads from whichever
thread the collection ran in, which took a worker's event loop down.
"""

from __future__ import annotations

from typing import Any

from nlght.adapters.outbound.stores.connections import StoreConnections


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _close(client: Any) -> None:  # noqa: ANN401 (store clients are untyped)
    client.close()


def test_one_engine_per_url_however_often_it_is_asked_for() -> None:
    connections = StoreConnections()

    first = connections.engine("sqlite+aiosqlite:///:memory:", pool_pre_ping=True)
    second = connections.engine("sqlite+aiosqlite:///:memory:", pool_pre_ping=True)

    assert first is second


def test_a_different_database_gets_its_own_engine() -> None:
    # Sharing a pool must never mean sharing a database: two activations
    # pointing at different databases stay separate.
    connections = StoreConnections()

    first = connections.engine("sqlite+aiosqlite:///:memory:")
    second = connections.engine("sqlite+aiosqlite:///./other.db")

    assert first is not second


def test_a_client_is_built_once_and_shared() -> None:
    connections = StoreConnections()
    built = 0

    def factory() -> _FakeClient:
        nonlocal built
        built += 1
        return _FakeClient()

    first = connections.client(("qdrant", "http://q", None, 30), factory)
    second = connections.client(("qdrant", "http://q", None, 30), factory)

    assert first is second
    assert built == 1


def test_a_different_endpoint_is_a_different_client() -> None:
    connections = StoreConnections()

    first = connections.client(("qdrant", "http://a", None, 30), _FakeClient)
    second = connections.client(("qdrant", "http://b", None, 30), _FakeClient)

    assert first is not second


def test_bootstrap_is_remembered_across_activations() -> None:
    # The collection/index bootstrap claims to run once per process. It used to
    # live on the throwaway activation, so it ran again for every write batch.
    connections = StoreConnections()

    assert connections.needs_bootstrap("qdrant", "http://q", "data-main") is True
    connections.mark_bootstrapped("qdrant", "http://q", "data-main")

    assert connections.needs_bootstrap("qdrant", "http://q", "data-main") is False
    assert connections.needs_bootstrap("qdrant", "http://q", "data-main-identity") is True


async def test_stopping_disposes_every_pool() -> None:
    connections = StoreConnections()
    client = connections.client(("qdrant", "http://q", None, 30), _FakeClient, close=_close)
    engine = connections.engine("sqlite+aiosqlite:///:memory:")

    await connections.stop()

    assert client.closed is True
    # A disposed engine hands out a fresh pool rather than a closed connection.
    assert connections.engine("sqlite+aiosqlite:///:memory:") is not engine


async def test_one_failing_close_does_not_skip_the_others() -> None:
    connections = StoreConnections()

    def _explode(_client: Any) -> None:  # noqa: ANN401 (store clients are untyped)
        raise RuntimeError("broken client")

    connections.client(("qdrant", "http://q", None, 30), _FakeClient, close=_explode)
    survivor = connections.client(
        ("opensearch", "http://o", None, 30), _FakeClient, close=_close
    )

    await connections.stop()

    assert survivor.closed is True
