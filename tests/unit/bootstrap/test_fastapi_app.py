# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import pytest

pytest.importorskip("httpx2")

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.bootstrap import fastapi_app
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import UnsupportedProtocolError, WorkflowNotFoundError
from nlght.core.execution import ExecutionStatus
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef


class StubConfigurationService:
    async def load_snapshot(self):
        class Snapshot:
            protocol_adapters = []
            gateways = []

        return Snapshot()


def _make_invocation() -> WorkflowInvocation:
    wf_id = uuid.uuid4()
    return WorkflowInvocation(
        trigger=Trigger(
            kind=TriggerKind.HTTP_REQUEST,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="generic_json_request",
            payload={"ok": True},
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
            name="generic_json_request",
            enabled=True,
            capabilities=["generic_json_request"],
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
    async def process(self, **_):
        return _make_invocation()


class ErrorGatewayService:
    async def process(self, **_):
        raise UnsupportedProtocolError("Unsupported or unknown protocol.")


class WorkflowNotFoundGatewayService:
    async def process(self, **_):
        raise WorkflowNotFoundError("No enabled workflow found for operation 'x'.")


class StubContainer:
    def __init__(self, gateway_service, http_protocol_adapters=None, engine=None) -> None:
        self.configuration_service = StubConfigurationService()
        self.gateway_service = gateway_service
        self.subsystems: list = []
        self.workflow_repository = None
        self.resource_repository = None
        self.http_protocol_adapters: list = http_protocol_adapters or []
        self.engine = engine


class _FakeConnection:
    async def execute(self, *_args, **_kwargs):
        return None


class _FakeConnectCM:
    def __init__(self, connection=None, error=None) -> None:
        self._connection = connection
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._connection

    async def __aexit__(self, *_exc) -> bool:
        return False


class WorkingEngine:
    def connect(self):
        return _FakeConnectCM(connection=_FakeConnection())


class UnreachableEngine:
    def connect(self):
        return _FakeConnectCM(error=ConnectionRefusedError("connection refused"))


async def _stub_build_container_ok(config_path: str) -> StubContainer:
    return StubContainer(StubGatewayService())


async def _stub_build_container_with_working_db(config_path: str) -> StubContainer:
    return StubContainer(StubGatewayService(), engine=WorkingEngine())


async def _stub_build_container_with_unreachable_db(config_path: str) -> StubContainer:
    return StubContainer(StubGatewayService(), engine=UnreachableEngine())


async def _stub_build_container_with_adapter(config_path: str) -> StubContainer:
    return StubContainer(StubGatewayService(), [GenericJsonHttpProtocolAdapter()])


class _FakeMetering:
    """A metrics endpoint mounted as an ASGI sub-app, like the real metering."""

    endpoint = "/metrics"

    def make_asgi_app(self):
        async def _app(scope, receive, send) -> None:
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({"type": "http.response.body", "body": b"metrics-body"})

        return _app


async def _stub_build_container_with_metering(config_path: str) -> StubContainer:
    # A catch-all adapter plus a metrics mount: the mount must win, not be
    # shadowed by the generic_json `/{full_path:path}` route.
    container = StubContainer(StubGatewayService(), [GenericJsonHttpProtocolAdapter()])
    container.metering = _FakeMetering()
    return container


_DISPATCH_EXEC_ID = uuid.uuid4()


class _FakeDispatcher:
    """A durable dispatcher whose submitted execution is already terminal.

    ``get_status`` returns the configured terminal record straight away, so the
    sync path resolves without waiting.
    """

    def __init__(self, terminal: SimpleNamespace) -> None:
        self._terminal = terminal
        self.submitted: list = []

    async def submit(self, submission) -> SimpleNamespace:
        self.submitted.append(submission)
        return SimpleNamespace(execution_id=_DISPATCH_EXEC_ID, status=ExecutionStatus.QUEUED)

    async def get_status(self, execution_id) -> SimpleNamespace:
        return self._terminal

    async def cancel(self, execution_id) -> None:
        return None


def _terminal(status: ExecutionStatus, *, result=None, diagnostics=None) -> SimpleNamespace:
    return SimpleNamespace(
        execution_id=_DISPATCH_EXEC_ID, status=status, result=result, diagnostics=diagnostics
    )


def _container_with_dispatcher(mode: str, terminal: SimpleNamespace) -> StubContainer:
    container = StubContainer(
        StubGatewayService(), [GenericJsonHttpProtocolAdapter(default_mode=mode)]
    )
    container.execution_dispatcher = _FakeDispatcher(terminal)
    return container


async def _stub_build_container_sync_dispatch(config_path: str) -> StubContainer:
    return _container_with_dispatcher(
        "sync", _terminal(ExecutionStatus.SUCCEEDED, result={"ok": True})
    )


async def _stub_build_container_async_dispatch(config_path: str) -> StubContainer:
    return _container_with_dispatcher("async", _terminal(ExecutionStatus.QUEUED))


async def _stub_build_container_failed_dispatch(config_path: str) -> StubContainer:
    return _container_with_dispatcher(
        "sync", _terminal(ExecutionStatus.FAILED, diagnostics={"type": "X", "message": "boom"})
    )


async def _stub_build_container_error(config_path: str) -> StubContainer:
    return StubContainer(ErrorGatewayService(), [GenericJsonHttpProtocolAdapter()])


async def _stub_build_container_not_found(config_path: str) -> StubContainer:
    return StubContainer(WorkflowNotFoundGatewayService(), [GenericJsonHttpProtocolAdapter()])


def test_healthz_is_public_but_config_does_not_exist(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_ok)
    monkeypatch.delenv("NLGHT_ADMIN", raising=False)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        health = client.get("/healthz")
        config = client.get("/config")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert config.status_code == 404


def test_readyz_reports_ok_when_no_database_configured(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_ok)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        ready = client.get("/readyz")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "database": "not configured"}


def test_readyz_reports_ok_when_database_reachable(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_with_working_db)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        ready = client.get("/readyz")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "database": "ok"}


def test_readyz_reports_503_when_database_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_with_unreachable_db)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        ready = client.get("/readyz")

    assert ready.status_code == 503
    assert ready.json() == {"status": "unavailable", "database": "unreachable"}


def test_config_route_does_not_exist_with_admin_mode(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_ok)
    monkeypatch.setenv("NLGHT_ADMIN", "1")
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        config = client.get("/config")

    assert config.status_code == 404


def test_ingress_returns_invocation_payload(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_with_adapter)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        response = client.post("/foo", json={"a": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["trigger"]["kind"] == "http.request"
    assert body["trigger"]["payload"] == {"ok": True}
    assert body["workflow"]["name"] == "generic_json_request"
    assert body["version"]["status"] == "active"


def test_catch_all_adapter_does_not_shadow_the_metrics_mount(monkeypatch) -> None:
    # Regression: the generic_json adapter is a catch-all, so /admin and /metrics
    # must be registered before it or Starlette routes them into the workflow
    # gateway instead. This exercises the mount; /admin uses the same ordering.
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_with_metering)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        metrics = client.get("/metrics")
        # The catch-all still serves everything else.
        other = client.post("/foo", json={"a": 1})

    assert metrics.status_code == 200
    assert metrics.text == "metrics-body"
    assert other.status_code == 200
    assert other.json()["workflow"]["name"] == "generic_json_request"


def test_bare_admin_redirects_to_dashboard_despite_catch_all(monkeypatch) -> None:
    # Regression: /admin (no trailing slash) must redirect to /admin/ even with
    # the generic_json catch-all present, which otherwise full-matches the bare
    # path and defeats Starlette's automatic trailing-slash redirect.
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_with_adapter)
    monkeypatch.setenv("NLGHT_ADMIN", "1")
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        resp = client.get("/admin", follow_redirects=False)

    assert resp.status_code == 307
    assert resp.headers["location"] == "/admin/"


def test_sync_endpoint_dispatches_and_returns_the_terminal_result(monkeypatch) -> None:
    # With a durable dispatcher, a sync endpoint submits the run and holds the
    # request open for the worker's result — the same path as async, but waiting.
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_sync_dispatch)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        resp = client.post("/foo", json={"a": 1})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_async_endpoint_returns_202_with_execution_id(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_async_dispatch)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        resp = client.post("/foo", json={"a": 1})

    assert resp.status_code == 202
    body = resp.json()
    assert body["workflow"] == "generic_json_request"
    assert body["execution_id"]


def test_sync_endpoint_maps_a_failed_execution_to_500(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_failed_dispatch)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        resp = client.post("/foo", json={"a": 1})

    assert resp.status_code == 500


def test_ingress_maps_runtime_errors_to_http_400(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_error)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        response = client.post("/foo", json={"a": 1})

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported or unknown protocol."


def test_ingress_maps_workflow_not_found_to_http_404(monkeypatch) -> None:
    monkeypatch.setattr(fastapi_app, "build_container", _stub_build_container_not_found)
    app = fastapi_app.create_app()

    with TestClient(app) as client:
        response = client.post("/foo", json={"a": 1})

    assert response.status_code == 404
    assert "No enabled workflow" in response.json()["detail"]
