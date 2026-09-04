# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace

from nlght.adapters.inbound.http.streaming import (
    relay_openai_sse,
    stream_execution_signals,
)
from nlght.core.entry.context import RequestContext
from nlght.core.execution import ExecutionStatus
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef


def _sig(content: str, kind: str = "token") -> Signal:
    return Signal(role="assistant", content=content, kind=kind)


def _invocation() -> WorkflowInvocation:
    wid = uuid.uuid4()
    vid = uuid.uuid4()
    ctx = RequestContext(
        correlation_id="c", request_id="req-1", received_at=datetime.now(UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    trigger = Trigger(
        kind=TriggerKind.MODEL_REQUEST, protocol=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
        operation="chat", payload={}, context=ctx, stream=True,
    )
    return WorkflowInvocation(
        trigger=trigger,
        workflow=WorkflowDef(workflow_id=wid, name="w", enabled=True, capabilities=[]),
        version=WorkflowVersionDef(version_id=vid, workflow_id=wid, version=1, status="active", steps=[]),
    )


class _FakeDispatcher:
    def __init__(self, execution_id: uuid.UUID) -> None:
        self._id = execution_id
        self.submitted: list = []

    async def submit(self, submission) -> SimpleNamespace:
        self.submitted.append(submission)
        return SimpleNamespace(execution_id=self._id, status=ExecutionStatus.QUEUED)


class _FakeBroker:
    def __init__(self, signals: list[Signal], *, is_distributed: bool = False) -> None:
        self._signals = signals
        self.is_distributed = is_distributed
        self.subscribed: list[uuid.UUID] = []

    async def subscribe(self, execution_id: uuid.UUID) -> AsyncIterator[Signal]:
        self.subscribed.append(execution_id)
        for signal in self._signals:
            yield signal


class _FakeExecutor:
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = signals

    async def stream_signals(self, _invocation: WorkflowInvocation) -> AsyncIterator[Signal]:
        for signal in self._signals:
            yield signal


async def test_durable_path_submits_and_relays_the_broker_stream() -> None:
    execution_id = uuid.uuid4()
    dispatcher = _FakeDispatcher(execution_id)
    broker = _FakeBroker([_sig("a"), _sig("b")])
    container = SimpleNamespace(
        execution_dispatcher=dispatcher,
        execution_stream_broker=broker,
        execution_worker=object(),
        workflow_executor=None,
    )

    out = [s async for s in stream_execution_signals(container, _invocation())]

    assert [s.content for s in out] == ["a", "b"]
    assert len(dispatcher.submitted) == 1
    assert dispatcher.submitted[0].trigger.stream is True
    assert broker.subscribed == [execution_id]


async def test_a_distributed_transport_relays_without_a_local_worker() -> None:
    """A gateway-only node may relay a run some remote worker claims."""
    execution_id = uuid.uuid4()
    dispatcher = _FakeDispatcher(execution_id)
    broker = _FakeBroker([_sig("a")], is_distributed=True)
    container = SimpleNamespace(
        execution_dispatcher=dispatcher,
        execution_stream_broker=broker,
        execution_worker=None,
        workflow_executor=_FakeExecutor([_sig("inline")]),
    )

    out = [s async for s in stream_execution_signals(container, _invocation())]

    assert [s.content for s in out] == ["a"]
    assert broker.subscribed == [execution_id]


async def test_an_in_process_transport_runs_inline_without_a_local_worker() -> None:
    """Nothing would carry a remote worker's signals here, so do not submit."""
    dispatcher = _FakeDispatcher(uuid.uuid4())
    broker = _FakeBroker([_sig("a")])
    container = SimpleNamespace(
        execution_dispatcher=dispatcher,
        execution_stream_broker=broker,
        execution_worker=None,
        workflow_executor=_FakeExecutor([_sig("inline")]),
    )

    out = [s async for s in stream_execution_signals(container, _invocation())]

    assert [s.content for s in out] == ["inline"]
    assert dispatcher.submitted == []
    assert broker.subscribed == []


async def test_inline_fallback_uses_the_executor_when_no_worker() -> None:
    executor = _FakeExecutor([_sig("x")])
    container = SimpleNamespace(
        execution_dispatcher=None,
        execution_stream_broker=None,
        execution_worker=None,
        workflow_executor=executor,
    )

    out = [s async for s in stream_execution_signals(container, _invocation())]

    assert [s.content for s in out] == ["x"]


async def test_relay_openai_sse_serialises_and_terminates_on_done() -> None:
    async def _gen() -> AsyncIterator[Signal]:
        yield _sig("hi", "token")
        yield _sig("", "done")

    frames = [f async for f in relay_openai_sse(_gen())]

    assert b'"content": "hi"' in frames[0]
    assert frames[-1] == b"data: [DONE]\n\n"
    assert len(frames) == 2  # no extra [DONE] appended when one was emitted


async def test_relay_openai_sse_appends_done_when_stream_omits_it() -> None:
    async def _gen() -> AsyncIterator[Signal]:
        yield _sig("hi", "token")

    frames = [f async for f in relay_openai_sse(_gen())]

    assert frames[-1] == b"data: [DONE]\n\n"
    assert len(frames) == 2
