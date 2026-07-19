# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid
from datetime import UTC, datetime

import pytest

pytest.importorskip("httpx2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowNotFoundError
from nlght.core.model.model_info import ModelInfo
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_MESSAGES = [{"role": "user", "content": "Hello"}]


def _make_invocation(operation: str = "generic_json_request") -> WorkflowInvocation:
    wf_id = uuid.uuid4()
    return WorkflowInvocation(
        trigger=Trigger(
            kind=TriggerKind.HTTP_REQUEST,
            protocol=ProtocolKind.GENERIC_JSON,
            operation=operation,
            payload={},
            context=RequestContext(
                correlation_id="cid",
                request_id="rid",
                received_at=datetime.now(UTC),
                path="/anything",
                method="POST",
                headers={},
                query_params={},
                client_host="127.0.0.1",
            ),
        ),
        workflow=WorkflowDef(
            workflow_id=wf_id,
            name=operation,
            enabled=True,
            capabilities=[operation],
        ),
        version=WorkflowVersionDef(
            version_id=uuid.uuid4(),
            workflow_id=wf_id,
            version=1,
            status="active",
            steps=[],
        ),
    )


class StubGatewayService:
    def __init__(self, invocation: WorkflowInvocation) -> None:
        self._invocation = invocation

    async def process(self, **_) -> WorkflowInvocation:
        return self._invocation


class ErrorGatewayService:
    async def process(self, **_) -> WorkflowInvocation:
        raise WorkflowNotFoundError("no workflow")


class StubContainer:
    def __init__(self, gateway_service) -> None:
        self.gateway_service = gateway_service


def _make_app(*adapters) -> FastAPI:
    app = FastAPI()
    for adapter in adapters:
        app.include_router(adapter.build_router())
    return app


# ---------------------------------------------------------------------------
# Fake ModelProviderBackend
# ---------------------------------------------------------------------------

class _FakeBoundClient:
    def __init__(self, content: str, stream: bool) -> None:
        self._content = content
        self._stream = stream

    async def call(self, messages: list) -> None:
        if self._stream:
            await self._emitter.emit(Signal(role="assistant", content=self._content, kind="token"))
            await self._emitter.emit(Signal(role="assistant", content="", kind="done"))
        else:
            await self._emitter.emit(Signal(role="assistant", content=self._content, kind="result"))


class _FakeModelBackend:
    def __init__(self, models: list[ModelInfo], response_content: str = "Hi") -> None:
        self._models = models
        self._response_content = response_content

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    def bind(self, *, model, emitter, stream, token_budget=None) -> _FakeBoundClient:
        bound = _FakeBoundClient(self._response_content, stream)
        bound._emitter = emitter
        return bound

    async def token_budget(self, model=None, *, client_max_tokens=None):
        pass


# ---------------------------------------------------------------------------
# OpenAI — request validation
# ---------------------------------------------------------------------------

def test_openai_chat_completions_returns_422_without_messages() -> None:
    """Missing messages field → 422 Unprocessable Entity."""
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "qwen3"})
    assert response.status_code == 422


def test_openai_chat_completions_returns_422_with_empty_messages() -> None:
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "qwen3", "messages": []}
        )
    assert response.status_code == 422


def test_openai_chat_completions_returns_422_with_invalid_role() -> None:
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3", "messages": [{"role": "robot", "content": "hi"}]},
        )
    assert response.status_code == 422


def test_openai_chat_completions_returns_503_without_provider() -> None:
    """Valid request but no model_provider_base_url → 503."""
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3", "messages": _VALID_MESSAGES},
        )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# OpenAI — list_models proxy
# ---------------------------------------------------------------------------

def test_openai_models_returns_503_without_provider() -> None:
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1"))
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 503


def test_openai_models_lists_from_backend() -> None:
    """GET /v1/models returns models from the configured backend in OpenAI format."""
    backend = _FakeModelBackend(models=[ModelInfo(name="qwen3"), ModelInfo(name="llama3")])
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1", model_backend=backend))
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert [m["id"] for m in data] == ["qwen3", "llama3"]
    assert data[0]["object"] == "model"


# ---------------------------------------------------------------------------
# OpenAI — direct inference (no workflow)
# ---------------------------------------------------------------------------

def test_openai_chat_completions_direct_inference() -> None:
    """No workflow configured → direct ModelClient inference."""
    backend = _FakeModelBackend(models=[], response_content="Hello back")
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1", model_backend=backend))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3", "messages": _VALID_MESSAGES},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello back"


def test_openai_chat_completions_streaming_returns_sse() -> None:
    """stream:true → text/event-stream with [DONE] sentinel."""
    backend = _FakeModelBackend(models=[], response_content="chunk")
    app = _make_app(OpenAIHttpProtocolAdapter(base_path="/v1", model_backend=backend))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3", "messages": _VALID_MESSAGES, "stream": True},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text


# ---------------------------------------------------------------------------
# OpenAI routes take precedence over generic JSON catch-all
# ---------------------------------------------------------------------------

def test_openai_routes_take_precedence_over_generic_json() -> None:
    """OpenAI routes must be registered before the GenericJson catch-all.

    /v1/chat/completions with a valid body reaches the OpenAI adapter (503 — no
    provider) rather than the generic JSON catch-all (which would need a gateway
    service and would return a different error).
    """
    app = _make_app(
        OpenAIHttpProtocolAdapter(base_path="/v1"),
        GenericJsonHttpProtocolAdapter(),
    )
    app.state.container = StubContainer(ErrorGatewayService())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3", "messages": _VALID_MESSAGES},
        )

    # 503 = OpenAI adapter handled it (no provider), not 404 from generic JSON
    assert response.status_code == 503
    assert "model provider" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Generic JSON catch-all
# ---------------------------------------------------------------------------

def test_generic_json_catch_all_route() -> None:
    invocation = _make_invocation("generic_json_request")
    app = _make_app(GenericJsonHttpProtocolAdapter())
    app.state.container = StubContainer(StubGatewayService(invocation))

    with TestClient(app) as client:
        response = client.post("/anything/goes", json={"x": 1})

    assert response.status_code == 200
    assert response.json()["workflow"]["name"] == "generic_json_request"


def test_generic_json_returns_404_on_missing_workflow() -> None:
    app = _make_app(GenericJsonHttpProtocolAdapter())
    app.state.container = StubContainer(ErrorGatewayService())

    with TestClient(app) as client:
        response = client.post("/anything", json={})

    assert response.status_code == 404
