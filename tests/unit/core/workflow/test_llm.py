# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import pytest

from nlght.core.entry.context import RequestContext
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.llm import call_llm
from nlght.core.workflow.step import WorkflowStepContext


class _BoundClient:
    def __init__(self, emitter, outputs: list[str]) -> None:
        self._emitter = emitter
        self._outputs = outputs

    async def call(self, messages, *, temperature: float | None = None):
        for output in self._outputs:
            await self._emitter.emit(Signal(role="assistant", content=output, kind="result"))


class _Backend:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or ["hello", " world"]
        self.bind_calls: list[dict] = []

    def bind(self, *, model, emitter, stream):
        self.bind_calls.append({"model": model, "stream": stream})
        return _BoundClient(emitter, self.outputs)


class _Client:
    def __init__(self, backend: _Backend, model: str | None = None) -> None:
        self._backend = backend
        self._model = model


def _ctx(*, llm=None, model: str = "ctx-model") -> WorkflowStepContext:
    return WorkflowStepContext(
        correlation_id="cid",
        trigger=Trigger(
            kind=TriggerKind.MODEL_REQUEST,
            protocol=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
            operation="chat",
            payload={},
            context=RequestContext(
                correlation_id="cid",
                request_id="rid",
                received_at=__import__("datetime").datetime(2026, 1, 1),
                path="/",
                method="POST",
                headers={},
                query_params={},
                client_host=None,
            ),
        ),
        model=model,
        messages=[],
        stream=False,
        emitter=object(),
        llm=llm,
    )


async def test_call_llm_uses_backend_directly_and_captures_output() -> None:
    backend = _Backend(["a", "", "b"])

    text = await call_llm([{"role": "user", "content": "hi"}], backend=backend, model="m1")

    assert text == "ab"
    assert backend.bind_calls == [{"model": "m1", "stream": False}]


async def test_call_llm_resolves_backend_and_model_from_context() -> None:
    backend = _Backend(["ok"])
    ctx = _ctx(llm=_Client(backend, model="client-model"), model="ctx-model")

    text = await call_llm([], ctx=ctx)

    assert text == "ok"
    assert backend.bind_calls == [{"model": "client-model", "stream": False}]


async def test_call_llm_raises_without_context_llm() -> None:
    with pytest.raises(RuntimeError, match="no LLM configured"):
        await call_llm([], ctx=_ctx(llm=None))


async def test_call_llm_raises_without_backend() -> None:
    with pytest.raises(RuntimeError, match="no LLM backend available"):
        await call_llm([])
