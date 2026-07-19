# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nlght.core.workflow.workflow import WorkflowInvocation

if TYPE_CHECKING:
    from nlght.core.signals.signal import Signal


@runtime_checkable
class WorkflowExecutor(Protocol):
    """Outbound port for workflow execution.

    The caller selects the method based on ``invocation.trigger.stream``:
    - ``execute``        — full response collected, returns assembled dict
    - ``stream``         — SSE-formatted bytes yielded incrementally (OpenAI wire format)
    - ``stream_signals`` — raw Signal objects yielded; adapter serialises to wire format
    """

    async def execute(self, invocation: WorkflowInvocation) -> dict[str, Any]: ...

    # Not `async def` -- the implementation is an async generator (uses
    # `yield`), so calling it returns the iterator directly, no `await`
    # needed: `async for chunk in executor.stream(invocation):`. Matches
    # ModelClient.stream()'s calling convention (ports/outbound/model_client.py).
    def stream(self, invocation: WorkflowInvocation) -> AsyncIterator[bytes]: ...

    def stream_signals(self, invocation: WorkflowInvocation) -> AsyncIterator[Signal]: ...
