# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid
from datetime import UTC, datetime

import pytest

from nlght.application.entry.gateway_service import GatewayService
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import UnsupportedProtocolError, WorkflowNotFoundError
from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowStepDef, WorkflowVersionDef


class StubProtocolDetector:
    def __init__(self, protocol: DetectedProtocol) -> None:
        self._protocol = protocol

    async def detect(self, **_: object) -> DetectedProtocol:
        return self._protocol


class StubTriggerResolver:
    def __init__(self, trigger: Trigger) -> None:
        self._trigger = trigger

    async def resolve(self, **_: object) -> Trigger:
        return self._trigger


class StubWorkflowRepository:
    def __init__(
        self,
        workflow: WorkflowDef | None,
        version: WorkflowVersionDef | None,
    ) -> None:
        self._workflow = workflow
        self._version = version

    async def find_by_name(self, name: str) -> WorkflowDef | None:
        return self._workflow

    async def list_enabled(self) -> list[WorkflowDef]:
        return [self._workflow] if self._workflow else []

    async def find_active_version(self, workflow_id: uuid.UUID) -> WorkflowVersionDef | None:
        return self._version


def _make_workflow(name: str = "generic_json_request") -> WorkflowDef:
    return WorkflowDef(
        workflow_id=uuid.uuid4(),
        name=name,
        enabled=True,
        capabilities=[name],
    )


def _make_version(workflow_id: uuid.UUID) -> WorkflowVersionDef:
    step_id = uuid.uuid4()
    return WorkflowVersionDef(
        version_id=uuid.uuid4(),
        workflow_id=workflow_id,
        version=1,
        status="active",
        steps=[
            WorkflowStepDef(
                step_id=step_id,
                position=0,
                name="start",
                type="llm",
                enabled=True,
                config={},
                transitions={},
                is_start=True,
                is_terminal=True,
                is_resume=False,
            )
        ],
    )


def _make_trigger(operation: str = "generic_json_request") -> Trigger:
    return Trigger(
        kind=TriggerKind.HTTP_REQUEST,
        protocol=ProtocolKind.GENERIC_JSON,
        operation=operation,
        payload={},
        context=RequestContext(
            correlation_id="cid",
            request_id="rid",
            received_at=datetime.now(UTC),
            path="/x",
            method="POST",
            headers={},
            query_params={},
            client_host=None,
        ),
    )


def _make_service(
    workflow: WorkflowDef | None = None,
    version: WorkflowVersionDef | None = None,
    protocol: ProtocolKind = ProtocolKind.GENERIC_JSON,
) -> GatewayService:
    wf = workflow or _make_workflow()
    ver = version or _make_version(wf.workflow_id)
    return GatewayService(
        protocol_detector=StubProtocolDetector(
            DetectedProtocol(kind=protocol, confidence=1.0, reason="test")
        ),
        trigger_resolver=StubTriggerResolver(_make_trigger()),
        workflow_repository=StubWorkflowRepository(wf, ver),
    )


async def test_process_returns_invocation() -> None:
    service = _make_service()

    result = await service.process(
        path="/ingress",
        method="POST",
        headers={"Content-Type": "application/json"},
        query_params={},
        raw_body=b"{}",
        client_host="127.0.0.1",
    )

    assert result.trigger.operation == "generic_json_request"
    assert result.workflow.name == "generic_json_request"
    assert result.version.status == "active"
    assert len(result.version.steps) == 1


async def test_process_raises_for_unknown_protocol() -> None:
    service = GatewayService(
        protocol_detector=StubProtocolDetector(
            DetectedProtocol(kind=ProtocolKind.UNKNOWN, confidence=0.0, reason="none")
        ),
        trigger_resolver=StubTriggerResolver(_make_trigger()),
    )

    with pytest.raises(UnsupportedProtocolError):
        await service.process(
            path="/ingress",
            method="POST",
            headers={},
            query_params={},
            raw_body=b"{}",
            client_host=None,
        )


async def test_process_raises_when_no_repository_configured() -> None:
    service = GatewayService(
        protocol_detector=StubProtocolDetector(
            DetectedProtocol(kind=ProtocolKind.GENERIC_JSON, confidence=1.0, reason="json")
        ),
        trigger_resolver=StubTriggerResolver(_make_trigger()),
        workflow_repository=None,
    )

    with pytest.raises(WorkflowNotFoundError, match="No workflow repository configured"):
        await service.process(
            path="/ingress",
            method="POST",
            headers={},
            query_params={},
            raw_body=b"{}",
            client_host=None,
        )


async def test_process_raises_when_workflow_not_found() -> None:
    service = GatewayService(
        protocol_detector=StubProtocolDetector(
            DetectedProtocol(kind=ProtocolKind.GENERIC_JSON, confidence=1.0, reason="json")
        ),
        trigger_resolver=StubTriggerResolver(_make_trigger()),
        workflow_repository=StubWorkflowRepository(workflow=None, version=None),
    )

    with pytest.raises(WorkflowNotFoundError, match="No enabled workflow"):
        await service.process(
            path="/ingress",
            method="POST",
            headers={},
            query_params={},
            raw_body=b"{}",
            client_host=None,
        )


async def test_process_raises_when_no_active_version() -> None:
    wf = _make_workflow()
    service = GatewayService(
        protocol_detector=StubProtocolDetector(
            DetectedProtocol(kind=ProtocolKind.GENERIC_JSON, confidence=1.0, reason="json")
        ),
        trigger_resolver=StubTriggerResolver(_make_trigger()),
        workflow_repository=StubWorkflowRepository(workflow=wf, version=None),
    )

    with pytest.raises(WorkflowNotFoundError, match="no active version"):
        await service.process(
            path="/ingress",
            method="POST",
            headers={},
            query_params={},
            raw_body=b"{}",
            client_host=None,
        )
