# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""UC3 — Custom Steps

Tests the contracts documented in workflows/custom-steps:

  - Subclass StepBase, register, and the executor calls it.
  - ctx.messages contains the conversation history from the request.
  - ctx.model contains the model name from the request.
  - ctx.stream reflects the stream flag from the request.
  - ctx.metadata persists between steps when passed via dataclasses.replace.
  - ctx.store_coordinator is None when no session key is present.
  - Emitting a Signal(kind='result') produces content in the response.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
from fastapi.testclient import TestClient

from integration._helpers import (
    build_test_app,
    build_test_container,
    make_step_loader,
    seed_workflow,
)
from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

pytestmark = pytest.mark.integration

_MAPPING = {"chat_completions": "wf"}


# ---------------------------------------------------------------------------
# Inspector step — captures ctx fields for assertion
# ---------------------------------------------------------------------------

class InspectorStep(StepBase):
    """Captures ctx fields into a shared dict and terminates."""
    TYPE = "test_inspector"
    captured: dict = {}

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        InspectorStep.captured = {
            "messages": list(ctx.messages),
            "model": ctx.model,
            "stream": ctx.stream,
            "coordinator": ctx.store_coordinator,
            "metadata": dict(ctx.metadata),
        }
        return StepResult(ctx=ctx, verdict="done")


@pytest.fixture(autouse=True)
def _clear_inspector():
    InspectorStep.captured.clear()
    yield
    InspectorStep.captured.clear()


# ---------------------------------------------------------------------------
# Custom step registration and execution
# ---------------------------------------------------------------------------

async def test_custom_step_is_executed(
    session_factory, workflow_repo, resource_repo,
):
    """A registered custom step runs and its verdict is respected."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(InspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    assert InspectorStep.captured != {}, "InspectorStep.run() was never called"


async def test_ctx_messages_contains_request_conversation(
    session_factory, workflow_repo, resource_repo,
):
    """ctx.messages is populated with the messages from the HTTP request."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(InspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": messages},
        )

    assert InspectorStep.captured["messages"] == messages


async def test_ctx_model_reflects_request_model(
    session_factory, workflow_repo, resource_repo,
):
    """ctx.model contains the model name from the request body."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(InspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5", "messages": [{"role": "user", "content": "x"}]},
        )

    assert InspectorStep.captured["model"] == "qwen2.5"


async def test_ctx_stream_false_when_not_requested(
    session_factory, workflow_repo, resource_repo,
):
    """ctx.stream is False when the request does not set stream:true."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(InspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
        )

    assert InspectorStep.captured["stream"] is False


async def test_ctx_store_coordinator_is_ephemeral_without_session_key(
    session_factory, workflow_repo, resource_repo,
):
    """ctx.store_coordinator is an ephemeral coordinator when no session key is provided.

    Without a configured SessionKeyResolver (or when the resolver finds no matching
    header), the executor creates a per-request ephemeral coordinator using the
    correlation ID as the key.  The coordinator is available to steps but is not
    persisted after the request completes.
    """
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(InspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        # No session key header configured → ephemeral coordinator, not None
        client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
        )

    assert InspectorStep.captured["coordinator"] is not None


# ---------------------------------------------------------------------------
# ctx.metadata persists between steps via dataclasses.replace
# ---------------------------------------------------------------------------

_META_RECEIVED: dict = {}


class MetadataWriterStep(StepBase):
    TYPE = "test_meta_writer"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        updated = dataclasses.replace(ctx, metadata={**ctx.metadata, "key": "value"})
        return StepResult(ctx=updated, verdict="next")


class MetadataReaderStep(StepBase):
    TYPE = "test_meta_reader"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        _META_RECEIVED.update(ctx.metadata)
        return StepResult(ctx=ctx, verdict="done")


@pytest.fixture(autouse=True)
def _clear_meta():
    _META_RECEIVED.clear()
    yield
    _META_RECEIVED.clear()


async def test_metadata_propagates_between_steps(
    session_factory, workflow_repo, resource_repo,
):
    """ctx.metadata written in step A via dataclasses.replace is readable in step B."""
    reader_id = uuid.uuid4()
    writer_id = uuid.uuid4()

    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": writer_id,
            "type": "test_meta_writer",
            "is_start": True,
            "transitions": {"next": str(reader_id)},
        },
        {
            "id": reader_id,
            "type": "test_meta_reader",
            "is_terminal": True,
        },
    ])
    loader = make_step_loader(MetadataWriterStep, MetadataReaderStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
        )

    assert resp.status_code == 200
    assert _META_RECEIVED.get("key") == "value"


# ---------------------------------------------------------------------------
# Emitting a result signal produces content in the response
# ---------------------------------------------------------------------------

class EmittingStep(StepBase):
    TYPE = "test_emitter"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(
            Signal(kind="result", role="assistant", content="hello from step")
        )
        return StepResult(ctx=ctx, verdict="done")


async def test_emitted_result_signal_appears_in_response(
    session_factory, workflow_repo, resource_repo,
):
    """Content emitted via ctx.emitter.emit(Signal(kind='result')) is returned
    in the JSON response's choices[0].message.content."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_emitter", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(EmittingStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}]},
        )

    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "hello from step"
