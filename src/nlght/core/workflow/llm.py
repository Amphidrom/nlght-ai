# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Capture workflow model output without emitting it to the caller."""

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
) -> str:
    """Capture a model response without user-facing signal emission.

    Args:
        messages (list[dict[str, Any]]): Messages sent to the model.
        ctx (WorkflowStepContext | None): Optional workflow context with a bound model client.
        backend (ModelProviderBackend | None): Optional explicit provider backend.
        model (str | None): Optional model name for binding.

    Returns:
        str: Concatenated nonempty signal content.

    Side Effects:
        Binds and calls a provider backend. An explicit backend takes precedence over
            the context binding.
    """
    if backend is None and ctx is not None:
        if ctx.llm is None:
            raise RuntimeError("call_llm: no LLM configured on ctx (ctx.llm is None)")
        backend = getattr(ctx.llm, "_backend", None)
        if model is None:
            model = getattr(ctx.llm, "_model", None) or ctx.model

    if backend is None:
        raise RuntimeError(
            "call_llm: no LLM backend available. Supply ctx= or backend="
        )

    capture = _BufferingCapture()
    bound = backend.bind(model=model, emitter=capture, stream=False)
    await bound.call(messages)

    parts = [sig.content for sig in capture.collected() if sig.content]
    return "".join(parts)
