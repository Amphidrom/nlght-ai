# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import pytest

pytest.importorskip("httpx2")

import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.bootstrap import fastapi_app
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import UnsupportedProtocolError, WorkflowNotFoundError
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
