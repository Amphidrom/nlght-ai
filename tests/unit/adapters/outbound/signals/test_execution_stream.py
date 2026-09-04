# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import uuid

from nlght.adapters.outbound.signals.execution_stream import (
    InProcessExecutionStreamBroker,
    PublishingSignalEmitter,
)
from nlght.core.signals.signal import Signal


def _sig(content: str, kind: str = "token") -> Signal:
    return Signal(role="assistant", content=content, kind=kind)


async def _collect(broker: InProcessExecutionStreamBroker, execution_id: uuid.UUID) -> list[Signal]:
    received: list[Signal] = []
    async for signal in broker.subscribe(execution_id):
        received.append(signal)
    return received


async def test_subscribe_then_publish_delivers_live() -> None:
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    task = asyncio.create_task(_collect(broker, eid))
    await asyncio.sleep(0.01)  # let the subscriber register

    await broker.publish(eid, _sig("a"))
    await broker.publish(eid, _sig("b"))
    await broker.close(eid)

    received = await task
    assert [s.content for s in received] == ["a", "b"]


async def test_publish_before_subscribe_is_replayed_from_the_backlog() -> None:
    # The race the gateway hits: the worker may start publishing before the
    # gateway attaches. A late subscriber still gets the earlier signals.
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    await broker.publish(eid, _sig("a"))
    await broker.publish(eid, _sig("b"))

    task = asyncio.create_task(_collect(broker, eid))
    await asyncio.sleep(0.01)
    await broker.publish(eid, _sig("c"))
    await broker.close(eid)

    received = await task
    assert [s.content for s in received] == ["a", "b", "c"]


async def test_two_subscribers_each_receive_every_signal() -> None:
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    first = asyncio.create_task(_collect(broker, eid))
    second = asyncio.create_task(_collect(broker, eid))
    await asyncio.sleep(0.01)

    await broker.publish(eid, _sig("x"))
    await broker.publish(eid, _sig("y"))
    await broker.close(eid)

    a = await first
    b = await second
    assert [s.content for s in a] == ["x", "y"]
    assert [s.content for s in b] == ["x", "y"]


async def test_subscribe_after_close_replays_backlog_then_ends() -> None:
    # A fast run can publish everything and close before the gateway subscribes.
    # The late subscriber must still replay the backlog and terminate, not hang
    # on a fresh empty channel.
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    await broker.publish(eid, _sig("a"))
    await broker.publish(eid, _sig("b"))
    await broker.close(eid)

    received = await asyncio.wait_for(_collect(broker, eid), timeout=1.0)
    assert [s.content for s in received] == ["a", "b"]


async def test_close_ends_the_subscriber_iterator() -> None:
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    task = asyncio.create_task(_collect(broker, eid))
    await asyncio.sleep(0.01)
    await broker.close(eid)

    received = await asyncio.wait_for(task, timeout=1.0)
    assert received == []


async def test_publishing_emitter_forwards_signals_and_close() -> None:
    broker = InProcessExecutionStreamBroker()
    eid = uuid.uuid4()
    emitter = PublishingSignalEmitter(broker, eid)
    task = asyncio.create_task(_collect(broker, eid))
    await asyncio.sleep(0.01)

    await emitter.emit(_sig("hi", kind="token"))
    await emitter.emit(_sig("done", kind="done"))
    await emitter.close()

    received = await task
    assert [(s.content, s.kind) for s in received] == [("hi", "token"), ("done", "done")]
