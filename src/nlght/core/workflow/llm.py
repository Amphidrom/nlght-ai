# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Internal LLM call utility for workflow steps.

call_llm() captures the model response as a string without emitting to the
user-facing SignalEmitter.  Use it for classification, reasoning, and JSON
extraction where you need the text back rather than streamed output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlght.core.signals.signal import Signal

if TYPE_CHECKING:
    from nlght.core.workflow.step import WorkflowStepContext
    from nlght.ports.outbound.model_provider_backend import ModelProviderBackend

_logger = logging.getLogger(__name__)


class _BufferingCapture:
    """Minimal in-memory SignalEmitter implementation, local to core.

    Structurally identical to adapters.outbound.signals.buffering
    .BufferingSignalEmitter (emit() + collected()), but core must not
    import adapters -- this is pure, dependency-free buffering with no
    I/O, so it doesn't need to be "the same class", only the same shape.
    """

    def __init__(self) -> None:
        self._signals: list[Signal] = []

    async def emit(self, signal: Signal) -> None:
        self._signals.append(signal)

    def collected(self) -> list[Signal]:
        return list(self._signals)


async def call_llm(
    messages: list[dict[str, Any]],
    *,
    ctx:         WorkflowStepContext | None = None,
    backend:     ModelProviderBackend | None = None,
    model:       str | None = None,
    temperature: float | None = None,
) -> str:
    """Return the LLM response text without emitting to the user.

    Bypasses ``ctx.emitter`` by capturing output into a temporary
    ``BufferingSignalEmitter``.

    Supply either ``ctx`` (workflow step context) or ``backend`` directly.
    When both are supplied, ``backend`` takes precedence.
    """
    if backend is None and ctx is not None:
        if ctx.llm is None:
            raise RuntimeError("call_llm: no LLM configured on ctx (ctx.llm is None)")
        backend = getattr(ctx.llm, "_backend", None)
        if model is None:
            model = getattr(ctx.llm, "_model", None) or ctx.model

    if backend is None:
        raise RuntimeError(
            "call_llm: no LLM backend available — supply ctx= or backend="
        )

    capture = _BufferingCapture()
    bound = backend.bind(model=model, emitter=capture, stream=False)
    await bound.call(messages, temperature=temperature)

    parts = [sig.content for sig in capture.collected() if sig.content]
    return "".join(parts)
