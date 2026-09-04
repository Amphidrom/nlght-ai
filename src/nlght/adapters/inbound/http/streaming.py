# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Shared, protocol-agnostic streaming source for the HTTP adapters.

Every streaming request draws its ``Signal``s from the same place: a worker,
when a durable queue and a local worker are available, otherwise the executor
inline in the request. The adapter keeps its own wire serialization (OpenAI SSE,
Ollama NDJSON); only the source of signals is shared.

Scope follows the configured transport. The in-process broker carries signals
from the *local* worker only, so with it the durable path requires a local worker
and anything else runs inline. A distributed transport (``is_distributed``)
carries them between nodes, so a gateway-only node can relay a run claimed by a
remote worker. The adapter asks the broker rather than inspecting it, so adding a
transport does not touch this module.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nlght.core.execution import ExecutionSubmission

if TYPE_CHECKING:
    from nlght.core.signals.signal import Signal
    from nlght.core.workflow.workflow import WorkflowInvocation


async def stream_execution_signals(
    container: Any,  # noqa: ANN401 - the bootstrap Container, untyped here by design
    invocation: WorkflowInvocation,
) -> AsyncIterator[Signal]:
    """Yield a streaming workflow's signals through a worker, or inline.

    Durable path (dispatcher + broker, and a worker this gateway can hear —
    a local one, or any worker when the transport is distributed): submit the run
    and subscribe to its stream, so the workflow executes on a worker and this
    gateway relays what it emits. Otherwise run the executor inline — the
    single-process fallback when persistence or a worker is not configured.
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
                # A streaming caller rarely sends its own key; the request id
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
    """One OpenAI-compatible SSE frame for a signal (matches the executor's)."""
    if signal.kind == "done":
        return b"data: [DONE]\n\n"
    payload = {
        "choices": [
            {"delta": {"role": signal.role, "content": signal.content}, "finish_reason": None}
        ]
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def relay_openai_sse(signals: AsyncIterator[Signal]) -> AsyncIterator[bytes]:
    """Serialize a signal stream to OpenAI SSE, terminating with ``[DONE]``."""
    sent_done = False
    async for signal in signals:
        if signal.kind == "done":
            sent_done = True
        yield _openai_sse_frame(signal)
    if not sent_done:
        yield b"data: [DONE]\n\n"
