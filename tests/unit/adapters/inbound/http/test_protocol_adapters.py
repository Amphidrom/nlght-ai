# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowNotFoundError
from nlght.core.execution import ExecutionStatus
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
        self.calls: list[dict] = []

    async def process(self, **kwargs) -> WorkflowInvocation:
        self.calls.append(kwargs)
        return self._invocation


class ErrorGatewayService:
    async def process(self, **_) -> WorkflowInvocation:
        raise WorkflowNotFoundError("no workflow")


class StubExecutor:
    def __init__(self, result=None) -> None:
        self.executed: list[WorkflowInvocation] = []
        self._result = result if result is not None else {"ran": True}

    async def execute(self, invocation: WorkflowInvocation):
        self.executed.append(invocation)
        return self._result


class StubDispatcher:
    """Submits, and answers what became of it.

    The sync path holds the request open and polls `get_status` until the
    execution is terminal — a sync call is the same durable run as an async
    one, it just waits. A stub that only submits leaves that poll with nothing
    to ask.
    """

    def __init__(self, result: dict | None = None) -> None:
        self.submissions = []
        self._result = result if result is not None else {"ran": True}

    async def submit(self, submission):
        self.submissions.append(submission)
        return SimpleNamespace(
            execution_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            status=SimpleNamespace(value="queued"),
        )

    async def get_status(self, execution_id):
        return SimpleNamespace(
            execution_id=execution_id,
            status=ExecutionStatus.SUCCEEDED,
            result=self._result,
            diagnostics=None,
        )


class StubContainer:
    def __init__(self, gateway_service, workflow_executor=None, dispatcher=None) -> None:
        self.gateway_service = gateway_service
        if workflow_executor is not None:
            self.workflow_executor = workflow_executor
        if dispatcher is not None:
            self.execution_dispatcher = dispatcher


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


# ---------------------------------------------------------------------------
# Generic JSON as the configurable trigger surface
# ---------------------------------------------------------------------------


def test_a_configured_path_names_the_workflow_and_runs_it() -> None:
    # The one place a workflow that is not a completion can get an endpoint.
    invocation = _make_invocation("ingest-data")
    gateway = StubGatewayService(invocation)
    executor = StubExecutor({"documents": 12})
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/reindex-docs": "ingest-data"}
        )
    )
    app.state.container = StubContainer(gateway, workflow_executor=executor)

    with TestClient(app) as client:
        response = client.post("/hooks/reindex-docs", json={})

    # Resolution goes through the gateway's existing workflow_name override...
    assert gateway.calls[0]["workflow_name"] == "ingest-data"
    # ...and execution through the same executor the other adapters use.
    assert executor.executed == [invocation]
    assert response.status_code == 200
    assert response.json() == {"documents": 12}


def test_a_trailing_slash_names_the_same_endpoint() -> None:
    gateway = StubGatewayService(_make_invocation("ingest-data"))
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/reindex-docs/": "ingest-data"}
        )
    )
    app.state.container = StubContainer(gateway, workflow_executor=StubExecutor())

    with TestClient(app) as client:
        client.post("/hooks/reindex-docs", json={})

    assert gateway.calls[0]["workflow_name"] == "ingest-data"


def test_an_unmapped_path_still_falls_back_to_the_resolved_operation() -> None:
    gateway = StubGatewayService(_make_invocation("generic_json_request"))
    app = _make_app(
        GenericJsonHttpProtocolAdapter(workflow_mapping={"/hooks/x": "some-workflow"})
    )
    app.state.container = StubContainer(gateway, workflow_executor=StubExecutor())

    with TestClient(app) as client:
        client.post("/anything/else", json={})

    # No override: the gateway looks the workflow up by trigger.operation.
    assert gateway.calls[0]["workflow_name"] is None


def test_without_a_runtime_the_resolution_is_reported_rather_than_faked() -> None:
    # A container with no executor cannot run anything; saying so beats
    # answering as though the workflow had run.
    invocation = _make_invocation("ingest-data")
    app = _make_app(
        GenericJsonHttpProtocolAdapter(workflow_mapping={"/hooks/go": "ingest-data"})
    )
    app.state.container = StubContainer(StubGatewayService(invocation))

    with TestClient(app) as client:
        response = client.post("/hooks/go", json={})

    assert response.status_code == 200
    assert response.json()["workflow"]["name"] == "ingest-data"


# ---------------------------------------------------------------------------
# Sync vs async per endpoint
# ---------------------------------------------------------------------------


def test_an_async_endpoint_queues_the_run_and_answers_202() -> None:
    invocation = _make_invocation("ingest-data")
    dispatcher, executor = StubDispatcher(), StubExecutor()
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/reindex": {"workflow": "ingest-data", "mode": "async"}}
        )
    )
    app.state.container = StubContainer(
        StubGatewayService(invocation), workflow_executor=executor, dispatcher=dispatcher
    )

    with TestClient(app) as client:
        response = client.post("/hooks/reindex", json={})

    assert response.status_code == 202
    body = response.json()
    assert body["execution_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["status"] == "queued"
    assert body["workflow"] == "ingest-data"
    # Queued, not run here: the caller is free while a worker does the work.
    assert executor.executed == []

    submission = dispatcher.submissions[0]
    assert submission.workflow_id == invocation.workflow.workflow_id
    assert submission.workflow_version_id == invocation.version.version_id
    assert submission.trigger is invocation.trigger


def test_an_explicit_idempotency_key_is_used_for_the_submission() -> None:
    # A caller retrying a POST must not start the run a second time.
    dispatcher = StubDispatcher()
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/reindex": {"workflow": "ingest-data", "mode": "async"}}
        )
    )
    app.state.container = StubContainer(
        StubGatewayService(_make_invocation("ingest-data")), dispatcher=dispatcher
    )

    with TestClient(app) as client:
        client.post("/hooks/reindex", json={}, headers={"Idempotency-Key": "nightly-2026-08-22"})
        client.post("/hooks/reindex", json={})

    assert dispatcher.submissions[0].idempotency_key == "nightly-2026-08-22"
    # Without a supplied key the request id stands in, so it is never empty.
    assert dispatcher.submissions[1].idempotency_key


def test_the_adapter_default_mode_applies_to_short_form_entries() -> None:
    dispatcher = StubDispatcher()
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/a": "ingest-data"}, default_mode="async"
        )
    )
    app.state.container = StubContainer(
        StubGatewayService(_make_invocation("ingest-data")), dispatcher=dispatcher
    )

    with TestClient(app) as client:
        assert client.post("/hooks/a", json={}).status_code == 202
    assert len(dispatcher.submissions) == 1


def test_an_endpoint_may_override_the_default_mode() -> None:
    executor, dispatcher = StubExecutor({"ran": True}), StubDispatcher()
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/quick": {"workflow": "ingest-data", "mode": "sync"}},
            default_mode="async",
        )
    )
    app.state.container = StubContainer(
        StubGatewayService(_make_invocation("ingest-data")),
        workflow_executor=executor,
        dispatcher=dispatcher,
    )

    with TestClient(app) as client:
        response = client.post("/hooks/quick", json={})

    assert response.status_code == 200
    assert response.json() == {"ran": True}
    # Not zero submissions: with a durable dispatcher configured, sync commits
    # the run to the queue exactly as async does, so it is claimable, retried
    # and concurrency-governed either way. The mode decides only who waits —
    # async answers 202 with an id, sync holds the request open for the result.
    assert len(dispatcher.submissions) == 1


def test_an_async_endpoint_without_a_dispatcher_says_so() -> None:
    # Silently running it synchronously would hold a connection open for a
    # workflow the operator asked to be queued.
    app = _make_app(
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/hooks/a": {"workflow": "ingest-data", "mode": "async"}}
        )
    )
    app.state.container = StubContainer(
        StubGatewayService(_make_invocation("ingest-data")), workflow_executor=StubExecutor()
    )

    with TestClient(app) as client:
        response = client.post("/hooks/a", json={})

    assert response.status_code == 503
    assert "no execution dispatcher" in response.json()["detail"]


def test_a_malformed_endpoint_configuration_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        GenericJsonHttpProtocolAdapter(
            workflow_mapping={"/a": {"workflow": "w", "mode": "eventually"}}
        )
    with pytest.raises(ValueError, match="names no workflow"):
        GenericJsonHttpProtocolAdapter(workflow_mapping={"/a": {"mode": "async"}})
    with pytest.raises(ValueError, match="must map to a workflow name"):
        GenericJsonHttpProtocolAdapter(workflow_mapping={"/a": 42})
