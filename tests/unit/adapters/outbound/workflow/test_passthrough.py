# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from datetime import datetime

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.workflow.steps.passthrough import PassthroughStep
from nlght.core.entry.context import RequestContext
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Event:
    def __init__(self, kind: str, content: str = "") -> None:
        self.kind = kind
        self.content = content


class _Llm:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def call(self, messages):
        self.calls.append(messages)

    async def stream(self, messages):
        yield _Event("token", "A")
        yield _Event("token", "")
        yield _Event("token", "B")
        yield _Event("done")


def _ctx(*, stream: bool, llm) -> WorkflowStepContext:
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
                received_at=datetime(2026, 1, 1),
                path="/",
                method="POST",
                headers={},
                query_params={},
                client_host=None,
            ),
            stream=stream,
        ),
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        emitter=BufferingSignalEmitter(),
        llm=llm,
    )


async def test_passthrough_fails_without_llm() -> None:
    result = await PassthroughStep(config={}).run(_ctx(stream=False, llm=None))

    assert result.verdict == "failed"


async def test_passthrough_non_streaming_calls_llm_and_uses_configured_verdict() -> None:
    llm = _Llm()
    ctx = _ctx(stream=False, llm=llm)

    result = await PassthroughStep(config={"verdict": "next"}).run(ctx)

    assert result.verdict == "next"
    assert llm.calls == [ctx.messages]
    assert ctx.emitter.collected() == []


async def test_passthrough_streaming_emits_result_tokens() -> None:
    ctx = _ctx(stream=True, llm=_Llm())

    result = await PassthroughStep(config={}).run(ctx)

    assert result.verdict == "DEFAULT"
    assert [(signal.content, signal.kind) for signal in ctx.emitter.collected()] == [
        ("A", "result"),
        ("B", "result"),
    ]
