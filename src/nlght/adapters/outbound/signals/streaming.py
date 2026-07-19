# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from nlght.core.signals.signal import Signal
from nlght.ports.outbound.signal_emitter import SignalEmitter

_SENTINEL = None


class QueuedSignalEmitter(SignalEmitter):
    """asyncio.Queue-based emitter for true incremental streaming.

    The step machine runs as an asyncio Task (producer).
    The HTTP response generator iterates via ``__aiter__`` (consumer).
    Both sides run concurrently inside the same event loop — signals are
    forwarded to the client as they are emitted, without waiting for the
    step machine to finish.

    Usage::

        emitter = QueuedSignalEmitter()
        task = asyncio.create_task(run_step_machine(..., emitter))
        async for signal in emitter:
            yield serialise_to_sse(signal)
        await task
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: asyncio.Queue[Signal | None] = asyncio.Queue(maxsize=maxsize)

    async def emit(self, signal: Signal) -> None:
        await self._queue.put(signal)

    async def close(self) -> None:
        await self._queue.put(_SENTINEL)

    def __aiter__(self) -> AsyncIterator[Signal]:
        return self._consume()

    async def _consume(self) -> AsyncIterator[Signal]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item
