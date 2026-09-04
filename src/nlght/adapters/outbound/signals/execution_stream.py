# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""In-process execution stream broker and its worker-side emitter.

Same-node ``gateway+worker`` streaming: the worker publishes a running
execution's signals into the broker; the gateway subscribes and relays them to
the caller. This is the concrete transport behind ``ExecutionStreamBroker`` for a
single process; ``postgres_execution_stream`` is the distributed one, selected by
``execution.stream.transport`` without either side noticing. It also reuses this
broker for local fan-out, so the delivery semantics below hold under both.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from nlght.core.signals.signal import Signal
from nlght.ports.outbound.execution_stream import ExecutionStreamBroker
from nlght.ports.outbound.signal_emitter import SignalEmitter

# How many recent signals a channel keeps for a subscriber that attaches after
# publishing has started. Bounds memory for a long stream with no live reader.
_MAX_BACKLOG = 1024

# Upper bound on retained channels. A closed channel is kept until its last
# subscriber drains it (so a subscriber that attaches after the run already
# finished still replays the backlog and ends, rather than hanging); this caps
# the retention of closed channels that are never read.
_MAX_CHANNELS = 4096


@dataclass
class _Channel:
    subscribers: list[asyncio.Queue[Signal | None]] = field(default_factory=list)
    backlog: list[Signal] = field(default_factory=list)
    closed: bool = False


class InProcessExecutionStreamBroker:
    """A per-execution fan-out of ``Signal``s, in one process.

    Ordering and the publish/subscribe race are handled without a lock by
    relying on the single-threaded event loop: ``publish`` and the registration
    half of ``subscribe`` each run to completion without awaiting, so a
    subscriber either sees a signal in its backlog snapshot (published before it
    registered) or on its queue (published after), never both and never neither.
    """

    is_distributed = False
    """One process only: a run claimed by a remote worker never reaches here."""

    def __init__(self) -> None:
        self._channels: dict[uuid.UUID, _Channel] = {}

    async def publish(self, execution_id: uuid.UUID, signal: Signal) -> None:
        channel = self._channel(execution_id)
        if channel.closed:
            return
        channel.backlog.append(signal)
        if len(channel.backlog) > _MAX_BACKLOG:
            del channel.backlog[0]
        for queue in channel.subscribers:
            queue.put_nowait(signal)

    async def close(self, execution_id: uuid.UUID) -> None:
        channel = self._channels.get(execution_id)
        if channel is None or channel.closed:
            return
        channel.closed = True
        for queue in channel.subscribers:
            queue.put_nowait(None)
        # Keep the closed channel (with its backlog) so a subscriber that attaches
        # after the run already finished — a fast run can complete before the
        # gateway subscribes — still replays it and then ends, instead of hanging
        # on a fresh empty channel. Dropped once its last subscriber drains it, or
        # evicted below if it is never read.
        if not channel.subscribers:
            self._evict_unread()

    async def subscribe(self, execution_id: uuid.UUID) -> AsyncIterator[Signal]:
        channel = self._channel(execution_id)
        queue: asyncio.Queue[Signal | None] = asyncio.Queue()
        # Atomic w.r.t. publish (no await between the snapshot and the append):
        # everything published so far is in `backlog`, everything after is routed
        # to `queue`.
        backlog = list(channel.backlog)
        channel.subscribers.append(queue)
        already_closed = channel.closed
        try:
            for signal in backlog:
                yield signal
            if already_closed:
                return
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            if queue in channel.subscribers:
                channel.subscribers.remove(queue)
            if channel.closed and not channel.subscribers:
                self._channels.pop(execution_id, None)

    def _channel(self, execution_id: uuid.UUID) -> _Channel:
        channel = self._channels.get(execution_id)
        if channel is None:
            channel = _Channel()
            self._channels[execution_id] = channel
            if len(self._channels) > _MAX_CHANNELS:
                self._evict_unread()
        return channel

    def _evict_unread(self) -> None:
        """Drop the oldest closed, subscriber-less channels over the cap."""
        for execution_id in list(self._channels):
            if len(self._channels) <= _MAX_CHANNELS:
                return
            channel = self._channels[execution_id]
            if channel.closed and not channel.subscribers:
                del self._channels[execution_id]


class PublishingSignalEmitter(SignalEmitter):
    """A ``SignalEmitter`` whose transport is the execution stream broker.

    The worker drives a streaming execution with this emitter so each step's
    output is forwarded to whoever is relaying the stream to the caller — the
    same ``SignalEmitter`` seam the buffering and queuing emitters implement,
    only its backend crosses to the gateway.
    """

    def __init__(self, broker: ExecutionStreamBroker, execution_id: uuid.UUID) -> None:
        self._broker = broker
        self._execution_id = execution_id

    async def emit(self, signal: Signal) -> None:
        await self._broker.publish(self._execution_id, signal)

    async def close(self) -> None:
        await self._broker.close(self._execution_id)
