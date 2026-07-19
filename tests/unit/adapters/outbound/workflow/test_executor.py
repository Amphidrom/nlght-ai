# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for StepMachineWorkflowExecutor.

Each test drives the executor with a fake workflow built from WorkflowStepDef
fixtures. Real step classes are registered on a fresh StepLoader per test —
no shared global state.
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError, WorkflowExecutionError
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext
from nlght.core.workflow.workflow import (
    WorkflowDef,
    WorkflowInvocation,
    WorkflowStepDef,
    WorkflowVersionDef,
)
from nlght.ports.outbound.access_policy import ModelAccessPolicy

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _request_context() -> RequestContext:
    return RequestContext(
        correlation_id="test-cid",
        request_id="test-rid",
        received_at=datetime(2026, 1, 1),
        path="/v1/chat/completions",
        method="POST",
        headers={},
        query_params={},
        client_host="127.0.0.1",
    )


def _trigger(stream: bool = False) -> Trigger:
    return Trigger(
        kind=TriggerKind.MODEL_REQUEST,
        protocol=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
        operation="chat_completions",
        payload={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        context=_request_context(),
        stream=stream,
    )


def _step_def(
    type_: str,
    *,
    is_start: bool = False,
    is_terminal: bool = False,
    transitions: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    step_id: uuid.UUID | None = None,
    enabled: bool = True,
) -> WorkflowStepDef:
    return WorkflowStepDef(
        step_id=step_id or uuid.uuid4(),
        position=0,
        name=type_,
        type=type_,
        enabled=enabled,
        config=config or {},
        transitions=transitions or {},
        is_start=is_start,
        is_terminal=is_terminal,
        is_resume=False,
    )


def _invocation(steps: list[WorkflowStepDef], stream: bool = False) -> WorkflowInvocation:
    workflow_id = uuid.uuid4()
    version_id = uuid.uuid4()
    return WorkflowInvocation(
        trigger=_trigger(stream=stream),
        workflow=WorkflowDef(
            workflow_id=workflow_id,
            name="test-workflow",
            enabled=True,
            capabilities=[],
        ),
        version=WorkflowVersionDef(
            version_id=version_id,
            workflow_id=workflow_id,
            version=1,
            status="active",
            steps=steps,
        ),
    )


def _make_executor(*step_classes: type[StepBase]) -> StepMachineWorkflowExecutor:
    loader = StepLoader()
    for cls in step_classes:
        loader.register(cls)
    return StepMachineWorkflowExecutor(loader=loader)


# ---------------------------------------------------------------------------
# Step stubs
# ---------------------------------------------------------------------------


class _DoneStep(StepBase):
    TYPE = "emit_done"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="done")


class _FailedStep(StepBase):
    TYPE = "emit_failed"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="failed")


class _EmitAndDone(StepBase):
    """Emits one signal then returns done."""

    TYPE = "emit_and_done"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal(role="assistant", content="hello", kind="result"))
        return StepResult(ctx=ctx, verdict="done")


class _TerminalStep(StepBase):
    """is_terminal=True in the step def — any verdict ends the machine."""

    TYPE = "terminal_step"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict=None)


class _StepA(StepBase):
    TYPE = "step_a"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="go_b")


class _StepB(StepBase):
    TYPE = "step_b"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="done")


class _InfiniteStep(StepBase):
    """Returns a verdict that loops back to itself — triggers max_hops."""

    TYPE = "infinite"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="loop")


class _MetaWriterStep(StepBase):
    """Writes to ctx.metadata and passes it forward via DEFAULT transition."""

    TYPE = "meta_writer"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        ctx.metadata["written"] = True
        return StepResult(ctx=ctx, verdict=None)  # uses DEFAULT transition


class _RaisingStep(StepBase):
    TYPE = "raising"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        raise RuntimeError("step exploded")


# ---------------------------------------------------------------------------
# execute() tests
# ---------------------------------------------------------------------------


async def test_execute_single_done_step_returns_response() -> None:
    executor = _make_executor(_DoneStep)
    step = _step_def("emit_done", is_start=True)
    result = await executor.execute(_invocation([step]))

    assert "choices" in result


async def test_execute_collects_emitted_signals() -> None:
    executor = _make_executor(_EmitAndDone)
    step = _step_def("emit_and_done", is_start=True)
    result = await executor.execute(_invocation([step]))

    content = result["choices"][0]["message"]["content"]
    assert content == "hello"


async def test_execute_failed_verdict_raises() -> None:
    executor = _make_executor(_FailedStep)
    step = _step_def("emit_failed", is_start=True)

    with pytest.raises(WorkflowExecutionError):
        await executor.execute(_invocation([step]))


async def test_execute_transitions_between_steps() -> None:
    id_a = uuid.uuid4()
    id_b = uuid.uuid4()

    step_a = _step_def(
        "step_a",
        is_start=True,
        transitions={"go_b": id_b},
        step_id=id_a,
    )
    step_b = _step_def("step_b", is_terminal=True, step_id=id_b)

    executor = _make_executor(_StepA, _StepB)
    result = await executor.execute(_invocation([step_a, step_b]))

    assert "choices" in result


async def test_execute_default_transition_used_when_verdict_not_in_map() -> None:
    id_a = uuid.uuid4()
    id_b = uuid.uuid4()

    step_a = _step_def(
        "step_a",
        is_start=True,
        transitions={"DEFAULT": id_b},
        step_id=id_a,
    )
    step_b = _step_def("step_b", is_terminal=True, step_id=id_b)

    executor = _make_executor(_StepA, _StepB)
    result = await executor.execute(_invocation([step_a, step_b]))

    assert "choices" in result


async def test_execute_raises_when_no_transition_for_verdict() -> None:
    step_a = _step_def("step_a", is_start=True, transitions={})

    executor = _make_executor(_StepA)

    with pytest.raises(WorkflowConfigurationError, match="no transition"):
        await executor.execute(_invocation([step_a]))


async def test_execute_raises_when_no_steps() -> None:
    executor = _make_executor()

    with pytest.raises(WorkflowConfigurationError, match="no steps"):
        await executor.execute(_invocation([]))


async def test_execute_raises_when_no_start_step() -> None:
    step = _step_def("emit_done", is_start=False)

    executor = _make_executor(_DoneStep)

    with pytest.raises(WorkflowConfigurationError, match="no enabled start step"):
        await executor.execute(_invocation([step]))


async def test_execute_raises_for_unknown_step_type() -> None:
    step = _step_def("not_registered", is_start=True)
    executor = _make_executor()

    with pytest.raises(WorkflowConfigurationError, match="Unknown step type"):
        await executor.execute(_invocation([step]))


async def test_execute_terminal_step_def_stops_machine() -> None:
    """A step with is_terminal=True in the DB stops the machine regardless of verdict."""
    executor = _make_executor(_TerminalStep)
    step = _step_def("terminal_step", is_start=True, is_terminal=True)

    result = await executor.execute(_invocation([step]))

    assert "choices" in result


async def test_execute_max_hops_stops_machine() -> None:
    """Infinite loop hits _MAX_HOPS and the machine hard-stops."""
    id_inf = uuid.uuid4()
    step = _step_def(
        "infinite",
        is_start=True,
        transitions={"loop": id_inf},
        step_id=id_inf,
    )
    executor = _make_executor(_InfiniteStep)

    # Should not raise — max_hops causes a hard stop, not an exception
    result = await executor.execute(_invocation([step]))
    assert "choices" in result


async def test_execute_metadata_persists_across_steps() -> None:
    """ctx.metadata written by one step is visible in the next (via StepResult.ctx)."""
    id_a = uuid.uuid4()
    id_b = uuid.uuid4()

    captured: dict[str, Any] = {}

    class _ReadMeta(StepBase):
        TYPE = "read_meta"

        async def run(self, ctx: WorkflowStepContext) -> StepResult:
            captured.update(ctx.metadata)
            return StepResult(ctx=ctx, verdict="done")

    step_a = _step_def("meta_writer", is_start=True, transitions={"DEFAULT": id_b}, step_id=id_a)
    step_b = _step_def("read_meta", is_terminal=True, step_id=id_b)

    executor = _make_executor(_MetaWriterStep, _ReadMeta)
    await executor.execute(_invocation([step_a, step_b]))

    assert captured.get("written") is True


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


async def test_stream_yields_sse_bytes() -> None:
    executor = _make_executor(_EmitAndDone)
    step = _step_def("emit_and_done", is_start=True)

    chunks = []
    async for chunk in executor.stream(_invocation([step], stream=True)):
        chunks.append(chunk)

    assert any(b"hello" in c for c in chunks)
    assert chunks[-1] == b"data: [DONE]\n\n"


async def test_stream_failed_verdict_propagates_exception() -> None:
    executor = _make_executor(_FailedStep)
    step = _step_def("emit_failed", is_start=True)

    with pytest.raises(WorkflowExecutionError):
        async for _ in executor.stream(_invocation([step], stream=True)):
            pass


# ---------------------------------------------------------------------------
# max_hops — boundary cases
# ---------------------------------------------------------------------------


async def test_execute_max_hops_uses_configured_fallback_step() -> None:
    """When a step type "max_hops" is registered, the hard-stop routes to it instead of just stopping."""
    id_inf = uuid.uuid4()
    id_fallback = uuid.uuid4()

    class _MaxHopsFallback(StepBase):
        TYPE = "max_hops"

        async def run(self, ctx: WorkflowStepContext) -> StepResult:
            return StepResult(ctx=ctx, verdict="done")

    step_inf = _step_def("infinite", is_start=True, transitions={"loop": id_inf}, step_id=id_inf)
    step_fallback = _step_def("max_hops", is_terminal=True, step_id=id_fallback)

    executor = _make_executor(_InfiniteStep, _MaxHopsFallback)
    result = await executor.execute(_invocation([step_inf, step_fallback]))

    assert "choices" in result


async def test_execute_max_hops_fallback_step_disabled_hard_stops() -> None:
    """A registered but disabled "max_hops" step must not be routed to — hard stop instead."""
    id_inf = uuid.uuid4()
    id_fallback = uuid.uuid4()

    class _MaxHopsFallback(StepBase):
        TYPE = "max_hops"

        async def run(self, ctx: WorkflowStepContext) -> StepResult:
            return StepResult(ctx=ctx, verdict="done")

    step_inf = _step_def("infinite", is_start=True, transitions={"loop": id_inf}, step_id=id_inf)
    step_fallback = _step_def("max_hops", is_terminal=True, step_id=id_fallback, enabled=False)

    executor = _make_executor(_InfiniteStep, _MaxHopsFallback)

    # Disabled fallback isn't in type_index -> hard stop, must not raise.
    result = await executor.execute(_invocation([step_inf, step_fallback]))
    assert "choices" in result


async def test_execute_exactly_at_max_hops_boundary_does_not_overrun() -> None:
    """A workflow that terminates on exactly the last allowed hop must succeed normally."""
    # 20 steps in a chain (_MAX_HOPS = 20), each DEFAULT-transitioning to the next,
    # the last one terminal -- exercises the boundary without ever hard-stopping.
    ids = [uuid.uuid4() for _ in range(20)]
    steps = []
    for i, step_id in enumerate(ids):
        is_last = i == len(ids) - 1
        transitions = {} if is_last else {"DEFAULT": ids[i + 1]}
        steps.append(_step_def(
            "emit_done" if is_last else "meta_writer",
            is_start=(i == 0),
            is_terminal=is_last,
            transitions=transitions,
            step_id=step_id,
        ))

    executor = _make_executor(_DoneStep, _MetaWriterStep)
    result = await executor.execute(_invocation(steps))

    assert "choices" in result


# ---------------------------------------------------------------------------
# ModelAccessPolicy — denial paths
# ---------------------------------------------------------------------------


class _AllowAllModelPolicy:
    async def is_allowed(self, model_name: str, caller: RequestContext) -> bool:
        return True


class _DenyAllModelPolicy:
    async def is_allowed(self, model_name: str, caller: RequestContext) -> bool:
        return False


class _RaisingModelPolicy:
    async def is_allowed(self, model_name: str, caller: RequestContext) -> bool:
        raise RuntimeError("policy backend unreachable")


def _make_executor_with_model_policy(
    *step_classes: type[StepBase],
    model_access_policy: ModelAccessPolicy,
    model_clients: dict[str, Any],
) -> StepMachineWorkflowExecutor:
    loader = StepLoader()
    for cls in step_classes:
        loader.register(cls)
    return StepMachineWorkflowExecutor(
        loader=loader,
        model_clients=model_clients,
        model_access_policy=model_access_policy,
    )


class _StubBackend:
    def bind(self, **kwargs: object) -> _StubBoundClient:
        return _StubBoundClient()


class _StubBoundClient:
    async def call(self, messages: list[dict[str, Any]], *, temperature: float | None = None) -> None:
        return None


async def test_model_access_policy_denies_raises_configuration_error() -> None:
    step = _step_def("emit_done", is_start=True, config={"model_provider": "prov"})
    executor = _make_executor_with_model_policy(
        _DoneStep,
        model_access_policy=_DenyAllModelPolicy(),
        model_clients={"prov": _StubBackend()},
    )

    with pytest.raises(WorkflowConfigurationError, match="access denied"):
        await executor.execute(_invocation([step]))


async def test_model_access_policy_error_fails_closed() -> None:
    step = _step_def("emit_done", is_start=True, config={"model_provider": "prov"})
    executor = _make_executor_with_model_policy(
        _DoneStep,
        model_access_policy=_RaisingModelPolicy(),
        model_clients={"prov": _StubBackend()},
    )

    with pytest.raises(WorkflowConfigurationError, match="policy check failed"):
        await executor.execute(_invocation([step]))


async def test_model_access_policy_allows_proceeds_normally() -> None:
    step = _step_def("emit_done", is_start=True, config={"model_provider": "prov"})
    executor = _make_executor_with_model_policy(
        _DoneStep,
        model_access_policy=_AllowAllModelPolicy(),
        model_clients={"prov": _StubBackend()},
    )

    result = await executor.execute(_invocation([step]))

    assert "choices" in result


# ---------------------------------------------------------------------------
# Orchestration lifecycle and error isolation
# ---------------------------------------------------------------------------


async def test_execute_rejects_duplicate_correlation_id_and_recovers_after_unregister() -> None:
    executor = _make_executor(_DoneStep)
    invocation = _invocation([_step_def("emit_done", is_start=True)])
    assert executor._try_register("test-cid") is True
    with pytest.raises(WorkflowExecutionError, match="already running"):
        await executor.execute(invocation)
    executor._unregister("test-cid")
    assert "choices" in await executor.execute(invocation)


async def test_stream_signals_rejects_duplicate_correlation_id() -> None:
    executor = _make_executor(_DoneStep)
    invocation = _invocation([_step_def("emit_done", is_start=True)], stream=True)
    executor._try_register("test-cid")
    try:
        with pytest.raises(WorkflowExecutionError, match="already running"):
            async for _ in executor.stream_signals(invocation):
                pass
    finally:
        executor._unregister("test-cid")


async def test_execute_reports_transition_to_unknown_step_id() -> None:
    missing = uuid.uuid4()
    step = _step_def("step_a", is_start=True, transitions={"go_b": missing})
    with pytest.raises(WorkflowConfigurationError, match=str(missing)):
        await _make_executor(_StepA).execute(_invocation([step]))


def test_coordinator_provisioning_handles_ephemeral_resume_and_failure() -> None:
    factory = MagicMock()
    ephemeral = object()
    resumed = object()
    factory.get_or_create.side_effect = [ephemeral, resumed, RuntimeError("backend down")]
    factory.exists.return_value = True
    executor = StepMachineWorkflowExecutor(loader=StepLoader(), coordinator_factory=factory)

    assert executor._provision_coordinator(_trigger()) == (ephemeral, False)
    session_trigger = replace(_trigger(), session_key="session")
    assert executor._provision_coordinator(session_trigger) == (resumed, True)
    assert executor._provision_coordinator(session_trigger) == (None, False)


def test_coordinator_release_saves_sessions_but_not_ephemeral_and_is_fail_safe() -> None:
    factory = MagicMock()
    executor = StepMachineWorkflowExecutor(loader=StepLoader(), coordinator_factory=factory)
    coordinator = MagicMock()
    executor._release_coordinator(_trigger(), coordinator)
    factory.save.assert_not_called()

    session_trigger = replace(_trigger(), session_key="session")
    executor._release_coordinator(session_trigger, coordinator)
    factory.save.assert_called_once_with("session", coordinator)
    factory.save.side_effect = RuntimeError("save failed")
    executor._release_coordinator(session_trigger, coordinator)


async def test_step_failure_still_cleans_runtime_branch_and_records_error_metering() -> None:
    coordinator = MagicMock()
    coordinator.close_branch.return_value = [object()]
    coordinator_factory = MagicMock()
    coordinator_factory.get_or_create.return_value = coordinator
    runtime = MagicMock()
    runtime_factory = MagicMock()
    runtime_factory.bind.return_value = runtime
    runtime_factory.release_by_name = AsyncMock(side_effect=RuntimeError("cleanup unavailable"))
    metering = MagicMock()
    metering.record_step = AsyncMock(side_effect=RuntimeError("metrics unavailable"))
    metering.record_request = AsyncMock(side_effect=RuntimeError("metrics unavailable"))
    loader = StepLoader()
    loader.register(_RaisingStep)
    executor = StepMachineWorkflowExecutor(
        loader=loader,
        coordinator_factory=coordinator_factory,
        os_runtime=runtime_factory,
        metering_port=metering,
    )

    with pytest.raises(RuntimeError, match="step exploded"):
        await executor.execute(_invocation([_step_def("raising", is_start=True)]))

    runtime_factory.release_by_name.assert_awaited_once_with("test-cid")
    coordinator.close_branch.assert_called_with(promote=True)
    coordinator.store_promoted_atoms.assert_called_once()
    assert metering.record_step.await_args.kwargs["status"] == "error"
    assert metering.record_request.await_args.kwargs["status"] == "error"
    assert executor.is_running("test-cid") is False


async def test_turn_branch_merge_failure_does_not_change_successful_result() -> None:
    coordinator = MagicMock()
    coordinator.close_branch.side_effect = RuntimeError("merge failed")
    factory = MagicMock()
    factory.get_or_create.return_value = coordinator
    executor = _make_executor(_DoneStep)
    executor._coordinator_factory = factory

    result = await executor.execute(_invocation([_step_def("emit_done", is_start=True)]))
    assert "choices" in result
