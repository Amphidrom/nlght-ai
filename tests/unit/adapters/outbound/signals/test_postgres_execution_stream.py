# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Delivery semantics of the distributed broker, over a substituted connection.

The fake bus below behaves like PostgreSQL's ``LISTEN``/``NOTIFY`` in the two
respects this adapter depends on: a notification reaches *every* listening
session, including one on the connection that published it, and payloads are
delivered in the order they were sent. Those two properties are the assumption
this file rests on, and nothing here proves them — that would need a real
database, which CI does not have. What *is* proven here is everything the
adapter does on top of them: routing, de-duplication of its own echo, and
reassembly of a payload too large for one notification.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from nlght.adapters.outbound.signals.postgres_execution_stream import (
    PostgresExecutionStreamBroker,
    asyncpg_dsn,
)
from nlght.core.signals.signal import Signal


class _Bus:
    """Stands in for one database's notification delivery."""

    def __init__(self) -> None:
        self.listeners: list[tuple[str, Any]] = []
        self.fail_publish = False

    def connect(self) -> Any:  # noqa: ANN401 - a test double for a driver connection
        return _FakeConnection(self)

    def notify(self, channel: str, payload: str) -> None:
        for listening_channel, callback in list(self.listeners):
            if listening_channel == channel:
                callback(None, 0, channel, payload)


class _FakeConnection:
    def __init__(self, bus: _Bus) -> None:
        self._bus = bus
        self._closed = False

    async def add_listener(self, channel: str, callback: Any) -> None:  # noqa: ANN401
        self._bus.listeners.append((channel, callback))

    async def execute(self, query: str, *args: Any) -> Any:  # noqa: ANN401
        if self._bus.fail_publish:
            raise RuntimeError("connection lost")
        assert "pg_notify" in query
        channel, payload = args
        self._bus.notify(channel, payload)

    async def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def add_termination_listener(self, callback: Any) -> None:  # noqa: ANN401
        return None


def _sig(content: str, kind: str = "token") -> Signal:
    return Signal(role="assistant", content=content, kind=kind)


async def _node(bus: _Bus, channel: str = "test_stream") -> PostgresExecutionStreamBroker:
    broker = PostgresExecutionStreamBroker(
        url="postgresql+asyncpg://unused",
        channel=channel,
        connect=lambda: _connected(bus),
    )
    await broker.start()
    await asyncio.sleep(0)  # let the listener register
    return broker


async def _connected(bus: _Bus) -> Any:  # noqa: ANN401
    return bus.connect()


async def _collect(
    broker: PostgresExecutionStreamBroker,
    execution_id: uuid.UUID,
) -> list[Signal]:
    return [signal async for signal in broker.subscribe(execution_id)]


async def test_a_signal_published_on_one_node_reaches_a_subscriber_on_another() -> None:
    bus = _Bus()
    publisher = await _node(bus)
    relay = await _node(bus)
    execution_id = uuid.uuid4()
    subscriber = asyncio.create_task(_collect(relay, execution_id))
    await asyncio.sleep(0.01)

    try:
        await publisher.publish(execution_id, _sig("a"))
        await publisher.publish(execution_id, _sig("b"))
        await publisher.close(execution_id)

        received = await asyncio.wait_for(subscriber, timeout=2)
        assert [signal.content for signal in received] == ["a", "b"]
    finally:
        await publisher.stop()
        await relay.stop()


async def test_the_publishing_node_delivers_locally_exactly_once() -> None:
    """Its own notification comes back from the database and must be ignored."""
    bus = _Bus()
    broker = await _node(bus)
    execution_id = uuid.uuid4()
    subscriber = asyncio.create_task(_collect(broker, execution_id))
    await asyncio.sleep(0.01)

    try:
        await broker.publish(execution_id, _sig("a"))
        await asyncio.sleep(0.01)  # give the echo a chance to be delivered twice
        await broker.close(execution_id)

        received = await asyncio.wait_for(subscriber, timeout=2)
        assert [signal.content for signal in received] == ["a"]
    finally:
        await broker.stop()


async def test_streams_on_other_channels_and_executions_do_not_cross() -> None:
    """A subscriber gets its own execution on its own channel, and nothing else.

    Both filters matter and they fail differently. A leak across executions
    hands one caller another caller's tokens; a leak across channels does the
    same between two deployments sharing a database. Neither is visible from a
    single-stream test, because with one channel and one execution every
    delivery is the right one.
    """
    bus = _Bus()
    publisher = await _node(bus)
    relay = await _node(bus)
    stranger = await _node(bus, channel="other_stream")
    wanted, unwanted = uuid.uuid4(), uuid.uuid4()
    subscriber = asyncio.create_task(_collect(relay, wanted))
    await asyncio.sleep(0.01)

    try:
        # Right execution, wrong channel.
        await stranger.publish(wanted, _sig("other-channel"))
        # Right channel, wrong execution.
        await publisher.publish(unwanted, _sig("other-execution"))
        await publisher.publish(wanted, _sig("mine"))
        await publisher.close(wanted)

        received = await asyncio.wait_for(subscriber, timeout=2)
        assert [signal.content for signal in received] == ["mine"]
    finally:
        await publisher.stop()
        await relay.stop()
        await stranger.stop()


async def test_a_signal_too_large_for_one_notification_arrives_whole() -> None:
    bus = _Bus()
    publisher = await _node(bus)
    relay = await _node(bus)
    execution_id = uuid.uuid4()
    content = "".join(f"line-{index:06d}\n" for index in range(5000))
    subscriber = asyncio.create_task(_collect(relay, execution_id))
    await asyncio.sleep(0.01)

    try:
        await publisher.publish(execution_id, _sig(content, kind="result"))
        await publisher.close(execution_id)

        received = await asyncio.wait_for(subscriber, timeout=5)
        assert [signal.content for signal in received] == [content]
        assert received[0].kind == "result"
    finally:
        await publisher.stop()
        await relay.stop()


async def test_a_broken_transport_does_not_fail_the_publishing_run() -> None:
    bus = _Bus()
    broker = await _node(bus)
    bus.fail_publish = True
    execution_id = uuid.uuid4()
    subscriber = asyncio.create_task(_collect(broker, execution_id))
    await asyncio.sleep(0.01)

    try:
        # The relay degrades; the execution producing the signals carries on, and
        # local subscribers still see everything.
        await broker.publish(execution_id, _sig("a"))
        await broker.close(execution_id)

        received = await asyncio.wait_for(subscriber, timeout=2)
        assert [signal.content for signal in received] == ["a"]
    finally:
        await broker.stop()


async def test_the_transport_declares_itself_distributed() -> None:
    assert PostgresExecutionStreamBroker.is_distributed is True


def test_the_sqlalchemy_url_is_translated_for_the_driver() -> None:
    dsn = asyncpg_dsn("postgresql+asyncpg://user:secret@db:5432/nlght")

    assert dsn.startswith("postgresql://")
    assert "+asyncpg" not in dsn
    assert "secret" in dsn  # the driver needs the password intact
    assert dsn.endswith("/nlght")
