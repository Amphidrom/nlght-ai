# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Relay live execution signals between nodes through PostgreSQL.

The adapter keeps listening connections and publishes notifications.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from nlght.adapters.outbound.signals.execution_stream import InProcessExecutionStreamBroker
from nlght.adapters.outbound.signals.execution_stream_wire import (
    StreamFrameAssembler,
    decode_frame,
    encode_close,
    encode_signal,
)
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.execution_stream import ExecutionStreamBroker

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL = "nlght_execution_stream"

# PostgreSQL caps a NOTIFY payload at 8000 bytes. Stay clear of the edge.
_PAYLOAD_BUDGET = 7800

# Backoff bounds for re-establishing a dropped listening connection.
_RECONNECT_BASE_SECONDS = 0.5
_RECONNECT_MAX_SECONDS = 30.0

# Frames buffered between the driver's callback (which cannot await) and the
# task that delivers them. A backlog this deep already means the consumer cannot
# keep up. Dropping the oldest frame beats unbounded growth.
_INBOX_SIZE = 4096


class _Connection(Protocol):
    """The slice of an asyncpg connection this adapter uses.

    Narrow on purpose: it is the seam the unit tests substitute, and it keeps the
    driver's full surface out of this module.
    """

    async def add_listener(self, channel: str, callback: Any) -> None: ...  # noqa: ANN401
    async def execute(self, query: str, *args: Any) -> Any: ...  # noqa: ANN401
    async def close(self) -> None: ...
    def is_closed(self) -> bool: ...
    def add_termination_listener(self, callback: Any) -> None: ...  # noqa: ANN401


ConnectionFactory = Callable[[], Awaitable[_Connection]]


class PostgresExecutionStreamBroker:
    """Distributed broker backed by PostgreSQL LISTEN and NOTIFY.

    Local subscribers receive signals immediately. Remote subscribers receive
    the live tail. The durable execution record supplies the terminal result.
    Notifications from this node are ignored after local delivery, and a lost
    listening connection is re-established with bounded backoff.
    """

    is_distributed = True

    def __init__(
        self,
        *,
        url: str,
        channel: str = DEFAULT_CHANNEL,
        connect: ConnectionFactory | None = None,
        local: ExecutionStreamBroker | None = None,
    ) -> None:
        self._url = url
        self._channel = channel
        self._connect = connect or _asyncpg_connector(url)
        self._local = local or InProcessExecutionStreamBroker()
        # Identifies this node's own notifications so the listener can skip them.
        self._origin = uuid.uuid4()
        self._assembler = StreamFrameAssembler()
        self._message_ids = itertools.count(1)
        self._inbox: asyncio.Queue[str] = asyncio.Queue(maxsize=_INBOX_SIZE)
        self._publish_connection: _Connection | None = None
        self._publish_lock = asyncio.Lock()
        self._listener_task: asyncio.Task[None] | None = None
        self._deliver_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._stopping.clear()
        self._listener_task = asyncio.create_task(self._listen_forever())
        self._deliver_task = asyncio.create_task(self._deliver_forever())

    async def stop(self) -> None:
        self._stopping.set()
        for task in (self._listener_task, self._deliver_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("execution.stream.stop_failed", exc_info=True)
        self._listener_task = None
        self._deliver_task = None
        async with self._publish_lock:
            if self._publish_connection is not None:
                await _close_quietly(self._publish_connection)
                self._publish_connection = None

    # -- ExecutionStreamBroker ----------------------------------------------

    async def publish(self, execution_id: uuid.UUID, signal: Signal) -> None:
        await self._local.publish(execution_id, signal)
        frames = encode_signal(
            self._origin,
            execution_id,
            signal,
            message_id=next(self._message_ids),
            budget=_PAYLOAD_BUDGET,
        )
        await self._notify(frames)

    async def close(self, execution_id: uuid.UUID) -> None:
        await self._local.close(execution_id)
        self._assembler.forget(execution_id)
        await self._notify([encode_close(self._origin, execution_id)])

    def subscribe(self, execution_id: uuid.UUID) -> AsyncIterator[Signal]:
        # Local delivery only: whatever arrives from another node has already been
        # published into the same in-process broker by the listener.
        return self._local.subscribe(execution_id)

    # -- publishing ----------------------------------------------------------

    async def _notify(self, payloads: list[str]) -> None:
        """Publish frames in order over one serialized connection.

        Args:
            payloads (list[str]): Encoded stream frames to publish.

        Side Effects:
            Sends PostgreSQL notifications. Logs transport failures without failing the execution.
        """
        async with self._publish_lock:
            try:
                connection = await self._publishing_connection()
                for payload in payloads:
                    await connection.execute("SELECT pg_notify($1, $2)", self._channel, payload)
            except Exception:
                logger.warning(
                    "execution.stream.publish_failed | channel=%s", self._channel, exc_info=True
                )
                if self._publish_connection is not None:
                    await _close_quietly(self._publish_connection)
                    self._publish_connection = None

    async def _publishing_connection(self) -> _Connection:
        if self._publish_connection is not None and not self._publish_connection.is_closed():
            return self._publish_connection
        self._publish_connection = await self._connect()
        return self._publish_connection

    # -- listening -----------------------------------------------------------

    async def _listen_forever(self) -> None:
        """Maintain a listening connection with bounded reconnect backoff.

        Side Effects:
            Opens and closes PostgreSQL connections until shutdown.
        """
        delay = _RECONNECT_BASE_SECONDS
        while not self._stopping.is_set():
            connection: _Connection | None = None
            try:
                connection = await self._connect()
                dead = asyncio.Event()
                connection.add_termination_listener(lambda _conn, event=dead: event.set())
                await connection.add_listener(self._channel, self._on_notify)
                delay = _RECONNECT_BASE_SECONDS
                logger.info("execution.stream.listening | channel=%s", self._channel)
                await _first_of(self._stopping.wait(), dead.wait())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "execution.stream.listen_failed | channel=%s retry_in=%.1fs",
                    self._channel, delay, exc_info=True,
                )
            finally:
                if connection is not None:
                    await _close_quietly(connection)
            if self._stopping.is_set():
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_MAX_SECONDS)

    def _on_notify(self, *args: Any) -> None:  # noqa: ANN401
        """Queue one driver notification for asynchronous delivery.

        Args:
            args (Any): Driver callback values with the payload last.

        Side Effects:
            Adds a valid string payload to the bounded inbox or logs overflow.
        """
        if not args:
            return
        payload = args[-1]
        if not isinstance(payload, str):
            return
        try:
            self._inbox.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("execution.stream.inbox_full | channel=%s", self._channel)

    async def _deliver_forever(self) -> None:
        """Deliver frames from other nodes to local subscribers.

        Side Effects:
            Reads the inbox and publishes decoded signals through the local broker.
        """
        while True:
            payload = await self._inbox.get()
            try:
                await self._deliver(payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("execution.stream.deliver_failed", exc_info=True)

    async def _deliver(self, payload: str) -> None:
        frame = decode_frame(payload)
        if frame is None or frame.origin == self._origin:
            # Our own notification, echoed back by PostgreSQL: already delivered
            # locally by `publish`.
            return
        complete = self._assembler.push(frame)
        if complete is None:
            return
        if complete.is_close:
            self._assembler.forget(complete.execution_id)
            await self._local.close(complete.execution_id)
            return
        if complete.signal is not None:
            await self._local.publish(complete.execution_id, complete.signal)


def _asyncpg_connector(url: str) -> ConnectionFactory:
    """Build a lazy asyncpg connection factory.

    Args:
        url (str): SQLAlchemy PostgreSQL URL.

    Returns:
        ConnectionFactory: Factory that opens a dedicated asyncpg connection.

    Side Effects:
        Imports and connects to asyncpg only when the factory is called.
    """

    async def connect() -> _Connection:
        # asyncpg ships no type information. The `_Connection` protocol above is
        # what this adapter actually depends on, so bind the driver to it here
        # and keep `Any` out of the rest of the module.
        import asyncpg  # type: ignore[import-untyped]  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        connection: _Connection = await asyncpg.connect(dsn=asyncpg_dsn(url))
        return connection

    return connect


def asyncpg_dsn(url: str) -> str:
    """Convert a SQLAlchemy PostgreSQL URL to an asyncpg DSN.

    Args:
        url (str): URL that may include the SQLAlchemy asyncpg driver suffix.

    Returns:
        str: Plain PostgreSQL DSN for a dedicated LISTEN connection.
    """
    from sqlalchemy.engine import make_url  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

    parsed = make_url(url)
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


async def _first_of(*awaitables: Awaitable[Any]) -> None:
    """Wait for the first awaitable and cancel the others.

    Args:
        awaitables (Awaitable[Any]): Operations whose first completion ends the wait.

    Side Effects:
        Schedules and cancels asyncio tasks.
    """
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()


async def _close_quietly(connection: _Connection) -> None:
    try:
        await connection.close()
    except Exception:
        logger.debug("execution.stream.close_failed", exc_info=True)
