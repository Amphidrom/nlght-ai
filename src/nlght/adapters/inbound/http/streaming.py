# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Supply workflow signals to HTTP adapters and encode OpenAI SSE frames.

The source uses durable dispatch when a worker is reachable through the
configured broker. Otherwise it streams from the local executor.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from nlght.core.execution import ExecutionSubmission

if TYPE_CHECKING:
    from nlght.core.signals.signal import Signal
    from nlght.core.workflow.workflow import WorkflowInvocation
    from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
    from nlght.ports.outbound.execution_stream import ExecutionStreamBroker
    from nlght.ports.outbound.workflow_executor import WorkflowExecutor


class _StreamingServices(Protocol):
    """Services the HTTP streaming adapter reads from its runtime container."""

    @property
    def execution_dispatcher(self) -> ExecutionDispatcher | None: ...

    @property
    def execution_stream_broker(self) -> ExecutionStreamBroker | None: ...

    @property
    def execution_worker(self) -> object | None: ...

    @property
    def workflow_executor(self) -> WorkflowExecutor | None: ...


async def stream_execution_signals(
    container: _StreamingServices,
    invocation: WorkflowInvocation,
) -> AsyncIterator[Signal]:
    """Stream signals through durable dispatch or the local executor.

    Args:
        container (_StreamingServices): Runtime services and transport configuration.
        invocation (WorkflowInvocation): Resolved workflow request.

    Yields:
        Signal: Signals emitted by the claimed worker or local executor.

    Side Effects:
        Submit an execution when a dispatcher and reachable broker exist.
        Otherwise run the executor inline.
    """
    dispatcher = getattr(container, "execution_dispatcher", None)
    broker = getattr(container, "execution_stream_broker", None)
    worker = getattr(container, "execution_worker", None)
    executor = getattr(container, "workflow_executor", None)
    reachable = worker is not None or getattr(broker, "is_distributed", False)

    if dispatcher is not None and broker is not None and reachable:
        record = await dispatcher.submit(
            ExecutionSubmission(
                workflow_id=invocation.workflow.workflow_id,
                workflow_version_id=invocation.version.version_id,
                trigger=invocation.trigger,
                # A streaming caller rarely sends its own key. The request id
                # makes a retried connection idempotent.
                idempotency_key=invocation.trigger.context.request_id,
                required_capabilities=tuple(invocation.workflow.capabilities or ()),
                exclusive=invocation.workflow.is_blocking,
            )
        )
        async for signal in broker.subscribe(record.execution_id):
            yield signal
        return

    if executor is None:
        return
    async for signal in executor.stream_signals(invocation):
        yield signal


def _openai_sse_frame(signal: Signal) -> bytes:
    """Encode one workflow signal as an OpenAI-compatible SSE frame.

    Args:
        signal (Signal): Signal to send to the client.

    Returns:
        bytes: JSON delta frame, or the ``[DONE]`` marker.
    """
    if signal.kind == "done":
        return b"data: [DONE]\n\n"
    payload = {
        "choices": [
            {"delta": {"role": signal.role, "content": signal.content}, "finish_reason": None}
        ]
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def relay_openai_sse(signals: AsyncIterator[Signal]) -> AsyncIterator[bytes]:
    """Relay workflow signals as OpenAI SSE and terminate the stream.

    Args:
        signals (AsyncIterator[Signal]): Signals from the execution source.

    Yields:
        bytes: Encoded frames, ending with ``[DONE]`` if the source omitted it.
    """
    sent_done = False
    async for signal in signals:
        if signal.kind == "done":
            sent_done = True
        yield _openai_sse_frame(signal)
    if not sent_done:
        yield b"data: [DONE]\n\n"
