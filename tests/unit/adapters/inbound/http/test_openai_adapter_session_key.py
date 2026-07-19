# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for session_key_resolver integration in OpenAIHttpProtocolAdapter.

Verifies that:
  - When a session_key_resolver is configured, its return value is forwarded as
    session_key to gateway_service.process().
  - When no resolver is configured, session_key=None is forwarded.
  - The resolver is only called on the workflow path (when executor + workflow_name
    are present); direct-inference calls bypass the gateway entirely.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("httpx2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.core.entry.context import RequestContext
from nlght.core.model.model_info import ModelInfo
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef

_MESSAGES = [{"role": "user", "content": "hi"}]

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _CapturingGatewayService:
    """Records every call to process() for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._invocation = _make_invocation()

    async def process(self, **kwargs: object) -> WorkflowInvocation:
        self.calls.append(kwargs)
        return self._invocation


class _FixedResolver:
    """Always returns the same session key."""

    def __init__(self, value: str | None) -> None:
        self._value = value

    async def resolve(self, **_: object) -> str | None:
        return self._value


class _FakeBoundClient:
    def __init__(self, emitter: object) -> None:
        self._emitter = emitter

    async def call(self, messages: list) -> None:
        await self._emitter.emit(Signal(role="assistant", content="ok", kind="result"))


class _FakeModelBackend:
    async def list_models(self) -> list[ModelInfo]:
        return []

    def bind(self, *, model: object, emitter: object, stream: bool, token_budget: object = None) -> _FakeBoundClient:
        return _FakeBoundClient(emitter)

    async def token_budget(self, model: object = None, *, client_max_tokens: object = None) -> None:
        pass


class _FakeExecutor:
    async def execute(self, invocation: WorkflowInvocation) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    def stream(self, invocation: WorkflowInvocation):
        return iter([])


class _FakeContainer:
    def __init__(self, gateway: _CapturingGatewayService) -> None:
        self.gateway_service = gateway
        self.workflow_executor = _FakeExecutor()


def _make_invocation() -> WorkflowInvocation:
    wf_id = uuid.uuid4()
    return WorkflowInvocation(
        trigger=Trigger(
            kind=TriggerKind.MODEL_REQUEST,
            protocol=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
            operation="chat_completions",
            payload={"messages": _MESSAGES, "stream": False},
            context=RequestContext(
                correlation_id="test",
                request_id="rid",
                received_at=datetime.now(UTC),
                path="/v1/chat/completions",
                method="POST",
                headers={},
                query_params={},
                client_host=None,
            ),
        ),
        workflow=WorkflowDef(
            workflow_id=wf_id,
            name="chat_completions",
            enabled=True,
            capabilities=["chat_completions"],
        ),
        version=WorkflowVersionDef(
            version_id=uuid.uuid4(),
            workflow_id=wf_id,
            version=1,
            status="active",
            steps=[],
        ),
    )


def _make_app(
    adapter: OpenAIHttpProtocolAdapter,
    gateway: _CapturingGatewayService,
) -> FastAPI:
    app = FastAPI()
    app.include_router(adapter.build_router())
    app.state.container = _FakeContainer(gateway)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_session_key_forwarded_to_gateway_when_resolver_configured() -> None:
    gateway = _CapturingGatewayService()
    adapter = OpenAIHttpProtocolAdapter(
        base_path="/v1",
        workflow_mapping={"chat_completions": "chat_completions"},
        session_key_resolver=_FixedResolver("sess-abc"),
    )
    app = _make_app(adapter, gateway)

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": _MESSAGES},
        )

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["session_key"] == "sess-abc"


def test_session_key_is_none_when_no_resolver_configured() -> None:
    gateway = _CapturingGatewayService()
    adapter = OpenAIHttpProtocolAdapter(
        base_path="/v1",
        workflow_mapping={"chat_completions": "chat_completions"},
        session_key_resolver=None,
    )
    app = _make_app(adapter, gateway)

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": _MESSAGES},
        )

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["session_key"] is None


def test_session_key_is_none_when_resolver_returns_none() -> None:
    gateway = _CapturingGatewayService()
    adapter = OpenAIHttpProtocolAdapter(
        base_path="/v1",
        workflow_mapping={"chat_completions": "chat_completions"},
        session_key_resolver=_FixedResolver(None),
    )
    app = _make_app(adapter, gateway)

    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": _MESSAGES},
        )

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["session_key"] is None


def test_direct_inference_bypasses_gateway() -> None:
    """Without workflow mapping, the adapter calls ModelClient directly — gateway not used."""
    gateway = _CapturingGatewayService()
    adapter = OpenAIHttpProtocolAdapter(
        base_path="/v1",
        model_backend=_FakeModelBackend(),
        session_key_resolver=_FixedResolver("should-not-appear"),
    )
    app = _make_app(adapter, gateway)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3", "messages": _MESSAGES},
        )

    assert resp.status_code == 200
    assert gateway.calls == []
