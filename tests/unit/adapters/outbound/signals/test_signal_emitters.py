# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.signals.streaming import QueuedSignalEmitter
from nlght.core.signals.signal import Signal

# ---------------------------------------------------------------------------
# BufferingSignalEmitter
# ---------------------------------------------------------------------------


async def test_buffering_emitter_collects_signals() -> None:
    emitter = BufferingSignalEmitter()
    s1 = Signal(role="assistant", content="hello", kind="token")
    s2 = Signal(role="assistant", content=" world", kind="token")

    await emitter.emit(s1)
    await emitter.emit(s2)

    assert emitter.collected() == [s1, s2]


async def test_buffering_emitter_collected_returns_copy() -> None:
    emitter = BufferingSignalEmitter()
    await emitter.emit(Signal(role="assistant", content="x", kind="token"))

    first = emitter.collected()
    second = emitter.collected()

    assert first == second
    assert first is not second


async def test_buffering_emitter_empty_by_default() -> None:
    emitter = BufferingSignalEmitter()
    assert emitter.collected() == []


# ---------------------------------------------------------------------------
# QueuedSignalEmitter
# ---------------------------------------------------------------------------


async def test_queued_emitter_yields_emitted_signals() -> None:
    emitter = QueuedSignalEmitter()
    signals = [
        Signal(role="assistant", content="a", kind="token"),
        Signal(role="assistant", content="b", kind="token"),
    ]

    async def produce() -> None:
        for s in signals:
            await emitter.emit(s)
        await emitter.close()

    asyncio.create_task(produce())

    received = []
    async for signal in emitter:
        received.append(signal)

    assert received == signals


async def test_queued_emitter_stops_at_close() -> None:
    emitter = QueuedSignalEmitter()

    async def produce() -> None:
        await emitter.emit(Signal(role="assistant", content="only one", kind="token"))
        await emitter.close()
        # nothing after close should reach the consumer
        await emitter.emit(Signal(role="assistant", content="ghost", kind="token"))

    asyncio.create_task(produce())

    received = []
    async for signal in emitter:
        received.append(signal)

    assert len(received) == 1
    assert received[0].content == "only one"
