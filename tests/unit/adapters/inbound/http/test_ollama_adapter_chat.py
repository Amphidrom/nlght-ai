# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for OllamaHttpProtocolAdapter chat handling.

Verifies:
  - POST /api/chat without backend returns 503
  - POST /api/chat with backend → direct non-streaming inference returns NDJSON
  - POST /api/chat with backend + stream:true → streaming NDJSON response
  - POST /api/chat with workflow_mapping → calls gateway_service.process()
  - session_key_resolver result is forwarded to gateway_service.process()
  - session_key is None when no resolver configured
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("httpx2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.ollama_adapter import OllamaHttpProtocolAdapter
from nlght.core.entry.context import RequestContext
from nlght.core.model.model_info import ModelInfo
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef

_MESSAGES = [{"role": "user", "content": "hello"}]

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _CapturingGatewayService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def process(self, **kwargs: object) -> WorkflowInvocation:
        self.calls.append(kwargs)
        return _make_invocation(stream=False)


class _FixedResolver:
    def __init__(self, value: str | None) -> None:
        self._value = value

    async def resolve(self, **_: object) -> str | None:
        return self._value


class _FakeBoundClient:
    def __init__(self, emitter: object, content: str) -> None:
        self._emitter = emitter
        self._content = content

    async def call(self, messages: list) -> None:
        await self._emitter.emit(
            Signal(role="assistant", content=self._content, kind="result")
        )


class _FakeModelBackend:
    def __init__(self, content: str = "pong") -> None:
        self._content = content

    async def list_models(self) -> list[ModelInfo]:
        return []

    def bind(self, *, model: object, emitter: object, stream: bool, token_budget: object = None) -> _FakeBoundClient:
        return _FakeBoundClient(emitter, self._content)

    async def token_budget(self, model: object = None, *, client_max_tokens: object = None) -> None:
        pass


class _FakeExecutor:
    async def execute(self, invocation: WorkflowInvocation) -> dict:
        # Return OpenAI-format dict as the real executor does
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "model": invocation.trigger.payload.get("model", ""),
        }

    def stream(self, invocation: WorkflowInvocation):
        return iter([])

    def stream_signals(self, invocation: WorkflowInvocation):
        return iter([])


class _FakeContainer:
    def __init__(self, gateway: _CapturingGatewayService) -> None:
        self.gateway_service = gateway
        self.workflow_executor = _FakeExecutor()


def _make_invocation(stream: bool = False) -> WorkflowInvocation:
    wf_id = uuid.uuid4()
    return WorkflowInvocation(
        trigger=Trigger(
            kind=TriggerKind.MODEL_REQUEST,
            protocol=ProtocolKind.OLLAMA_CHAT,
            operation="chat",
            payload={"messages": _MESSAGES, "stream": stream, "model": "llama3"},
            stream=stream,
            context=RequestContext(
                correlation_id="test",
                request_id="rid",
                received_at=datetime.now(UTC),
                path="/api/chat",
                method="POST",
                headers={},
                query_params={},
                client_host=None,
            ),
        ),
        workflow=WorkflowDef(
            workflow_id=wf_id,
            name="chat",
            enabled=True,
            capabilities=["chat"],
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
    adapter: OllamaHttpProtocolAdapter,
    container: object = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(adapter.build_router())
    if container is not None:
        app.state.container = container
    return app


# ---------------------------------------------------------------------------
# Direct inference
# ---------------------------------------------------------------------------


def test_chat_returns_503_without_backend() -> None:
    adapter = OllamaHttpProtocolAdapter()
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"model": "llama3", "messages": _MESSAGES, "stream": False})
    assert resp.status_code == 503


def test_chat_direct_inference_non_streaming() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeModelBackend("pong"))
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.post(
            "/api/chat",
            json={"model": "llama3", "messages": _MESSAGES, "stream": False},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "pong"
    assert body["done"] is True


def test_chat_direct_inference_streaming_returns_ndjson() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeModelBackend("token"))
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.post(
            "/api/chat",
            json={"model": "llama3", "messages": _MESSAGES, "stream": True},
        )
    assert resp.status_code == 200
    assert "ndjson" in resp.headers.get("content-type", "")
    # At least one NDJSON line must be present
    lines = [line for line in resp.text.splitlines() if line.strip()]
    assert len(lines) >= 1
    parsed = json.loads(lines[0])
    assert "message" in parsed or "done" in parsed


# ---------------------------------------------------------------------------
# Workflow routing + session key
# ---------------------------------------------------------------------------


def test_chat_with_workflow_calls_gateway() -> None:
    gateway = _CapturingGatewayService()
    adapter = OllamaHttpProtocolAdapter(
        workflow_mapping={"chat": "chat"},
    )
    app = _make_app(adapter, _FakeContainer(gateway))
    with TestClient(app) as client:
        client.post(
            "/api/chat",
            json={"model": "llama3", "messages": _MESSAGES, "stream": False},
        )
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["workflow_name"] == "chat"


def test_chat_session_key_forwarded_when_resolver_configured() -> None:
    gateway = _CapturingGatewayService()
    adapter = OllamaHttpProtocolAdapter(
        workflow_mapping={"chat": "chat"},
        session_key_resolver=_FixedResolver("sess-xyz"),
    )
    app = _make_app(adapter, _FakeContainer(gateway))
    with TestClient(app) as client:
        client.post(
            "/api/chat",
            json={"model": "llama3", "messages": _MESSAGES, "stream": False},
        )
    assert gateway.calls[0]["session_key"] == "sess-xyz"


def test_chat_session_key_is_none_without_resolver() -> None:
    gateway = _CapturingGatewayService()
    adapter = OllamaHttpProtocolAdapter(
        workflow_mapping={"chat": "chat"},
        session_key_resolver=None,
    )
    app = _make_app(adapter, _FakeContainer(gateway))
    with TestClient(app) as client:
        client.post(
            "/api/chat",
            json={"model": "llama3", "messages": _MESSAGES, "stream": False},
        )
    assert gateway.calls[0]["session_key"] is None
