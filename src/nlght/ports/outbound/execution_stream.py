# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Protocol

from nlght.core.signals.signal import Signal


class ExecutionStreamBroker(Protocol):
    """Carries a running execution's live ``Signal``s from the worker performing
    it to any subscriber relaying them onward, including the gateway holding the caller's
    HTTP stream.

    It is a live, best-effort relay, not a store: the durable execution record
    remains the source of truth for the terminal result. A subscriber that
    attaches after some signals were already published receives what a bounded
    backlog still holds, then the live tail.

    The seam is deliberately narrow so the in-process implementation (same-node
    ``gateway+worker``) can be swapped for a distributed transport (PostgreSQL
    ``LISTEN/NOTIFY``, Redis) without changing callers.
    """

    is_distributed: bool
    """Whether this transport carries signals *between* nodes.

    The one thing a caller legitimately needs to know about the transport, and
    it is a property of the seam rather than of any particular implementation: a
    gateway may relay a run claimed by a remote worker only when the answer is
    true. Callers ask this instead of inspecting the concrete adapter, so adding
    a transport never edits a caller.
    """

    async def publish(self, execution_id: uuid.UUID, signal: Signal) -> None:
        """Relay one execution signal to active subscribers.

        Args:
            execution_id (uuid.UUID): Execution whose signal is emitted.
            signal (Signal): Signal to publish.

        Side Effects:
            Publishes through the configured stream transport.
        """
        ...

    async def close(self, execution_id: uuid.UUID) -> None:
        """End an execution's live signal stream.

        Args:
            execution_id (uuid.UUID): Execution ID.

        Side Effects:
            Completes subscriber iterators through the transport.
        """
        ...

    def subscribe(self, execution_id: uuid.UUID) -> AsyncIterator[Signal]:
        """Receive backlog and live signals for an execution.

        Args:
            execution_id (uuid.UUID): Execution ID.

        Returns:
            AsyncIterator[Signal]: Signal iterator that ends after stream closure.

        Side Effects:
            Registers a transport subscriber during iteration.
        """
        ...
