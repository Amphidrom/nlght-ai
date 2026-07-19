# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""UC5 — Signal Emission and Response Format

Tests the contracts documented in concepts and custom-steps:

  Non-streaming (stream=false):
  - Multiple result signals are concatenated into choices[0].message.content.
  - Token signals are included in the concatenated content.
  - An empty emitter produces an empty content string.

  Streaming (stream=true):
  - The response is delivered as SSE (text/event-stream).
  - Each result/token signal becomes a data: {...} frame.
  - The stream ends with data: [DONE].
"""

from __future__ import annotations

import json

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
# Step helpers
# ---------------------------------------------------------------------------

class MultiSignalStep(StepBase):
    """Emits a mix of token and result signals."""
    TYPE = "test_multi_signal"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal(kind="token",  role="assistant", content="Hello"))
        await ctx.emitter.emit(Signal(kind="token",  role="assistant", content=" "))
        await ctx.emitter.emit(Signal(kind="result", role="assistant", content="World"))
        return StepResult(ctx=ctx, verdict="done")


class NoSignalStep(StepBase):
    """Emits nothing — tests the empty emitter path."""
    TYPE = "test_no_signal"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="done")


class SingleResultStep(StepBase):
    """Emits one result signal followed by the done sentinel."""
    TYPE = "test_single_result"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        await ctx.emitter.emit(Signal(kind="result", role="assistant", content="ping"))
        await ctx.emitter.emit(Signal(kind="done",   role="assistant", content=""))
        return StepResult(ctx=ctx, verdict="done")


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------

async def test_non_streaming_concatenates_all_signals(
    session_factory, workflow_repo, resource_repo,
):
    """stream=false: token + result signals are concatenated into response content."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_multi_signal", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(MultiSignalStep)
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
    assert content == "Hello World"


async def test_non_streaming_empty_emitter_returns_empty_content(
    session_factory, workflow_repo, resource_repo,
):
    """stream=false with no signals emitted → empty content string, not an error."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_no_signal", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(NoSignalStep)
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
    assert resp.json()["choices"][0]["message"]["content"] == ""


async def test_non_streaming_response_shape(
    session_factory, workflow_repo, resource_repo,
):
    """Non-streaming response always has choices[0].message.role == 'assistant'."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_single_result", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(SingleResultStep)
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

    body = resp.json()
    assert "choices" in body
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

async def test_streaming_response_content_type_is_event_stream(
    session_factory, workflow_repo, resource_repo,
):
    """stream=true → Content-Type is text/event-stream."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_single_result", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(SingleResultStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as resp:
            assert "text/event-stream" in resp.headers["content-type"]
            chunks = list(resp.iter_lines())

    assert len(chunks) > 0


async def test_streaming_response_ends_with_done(
    session_factory, workflow_repo, resource_repo,
):
    """The SSE stream ends with the `data: [DONE]` sentinel frame."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_single_result", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(SingleResultStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as resp:
            raw = resp.read().decode()

    assert "data: [DONE]" in raw


async def test_streaming_data_frames_carry_assistant_content(
    session_factory, workflow_repo, resource_repo,
):
    """SSE data frames contain the assistant content emitted by the step."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_single_result", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(SingleResultStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
    )
    with TestClient(build_test_app(container)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "x"}], "stream": True},
        ) as resp:
            raw = resp.read().decode()

    # Extract data frames, skip [DONE]
    frames = [
        line[len("data: "):]
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert len(frames) > 0
    payload = json.loads(frames[0])
    delta = payload["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta["content"] == "ping"
