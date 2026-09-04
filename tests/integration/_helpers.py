# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Shared integration-test helpers.

This module contains builder functions and stub classes used across the
integration test suite.  It is NOT a pytest conftest — fixtures live in
conftest.py.  Tests import helpers from here directly:

    from integration._helpers import build_test_app, build_test_container, seed_workflow

(``tests/`` is on sys.path via pyproject.toml ``pythonpath``.)
"""
from __future__ import annotations

import uuid
from typing import Any, cast

from fastapi import FastAPI

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinatorFactory
from nlght.adapters.outbound.persistence.models import (
    AccessPolicyRule,
    Resource,
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)
from nlght.adapters.outbound.protocol.composite import CompositeProtocolDetector
from nlght.adapters.outbound.protocol.generic_json import GenericJsonProtocolDetector
from nlght.adapters.outbound.protocol.openai import OpenAIProtocolDetector
from nlght.adapters.outbound.trigger.default import DefaultTriggerResolver
from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.adapters.outbound.workflow.steps.done import DoneStep
from nlght.adapters.outbound.workflow.steps.failed import FailedStep
from nlght.adapters.outbound.workflow.steps.passthrough import PassthroughStep
from nlght.application.entry.gateway_service import GatewayService
from nlght.bootstrap.wiring import Container
from nlght.core.licensing.entitlements import HIVE_MIND_FILESYSTEM, PROVIDERS_MULTI
from nlght.core.licensing.license import License, Tier
from nlght.core.licensing.service import LicenseService
from nlght.core.session import SessionAccess
from nlght.core.signals.signal import Signal


class StubLicenseAdapter(LicenseService):
    """Full-access license for integration tests — all entitlements granted."""

    def __init__(self) -> None:
        super().__init__(License(
            customer_id="test",
            tier=Tier.ENTERPRISE,
            entitlements=frozenset({HIVE_MIND_FILESYSTEM, PROVIDERS_MULTI}),
        ))

    def context_metadata(self) -> dict:
        return {"customer_id": "test", "tier": "enterprise"}


# ---------------------------------------------------------------------------
# Stub model client
# ---------------------------------------------------------------------------

class StubBoundModelClient:
    """Returns a canned response for LLM calls — no real model needed."""

    def __init__(self, response: str, emitter: object) -> None:
        self._response = response
        self._emitter = emitter

    async def call(self, messages: list[dict], system: str | None = None) -> None:
        await self._emitter.emit(
            Signal(kind="result", role="assistant", content=self._response)
        )


class StubModelClient:
    """Stub model provider.  ``bind()`` returns a StubBoundModelClient."""

    PROVIDER_NAME = "test-provider"

    def __init__(self, response: str = "stub response") -> None:
        self.response = response

    def bind(self, *, model: str, emitter: object, stream: bool, token_budget: object = None):
        return StubBoundModelClient(response=self.response, emitter=emitter)


# ---------------------------------------------------------------------------
# DB seeding helpers
# ---------------------------------------------------------------------------

async def seed_workflow(
    session_factory,
    name: str,
    steps: list[dict],
) -> dict[str, Any]:
    """Insert a workflow + version + steps into SQLite.

    Each step dict keys:
        id (uuid.UUID, optional — generated if missing)
        name (str, optional)
        type (str, required)
        is_start (bool)
        is_terminal (bool)
        transitions (dict — values must be str(step_id) of the target step)
        config (dict, optional)
        enabled (bool, optional)

    Returns {workflow_id, version_id, step_ids}.
    """
    wf_id = uuid.uuid4()
    ver_id = uuid.uuid4()

    for step in steps:
        if "id" not in step:
            step["id"] = uuid.uuid4()

    async with session_factory() as session:
        async with session.begin():
            session.add(Workflow(
                workflow_id=wf_id,
                name=name,
                description=f"Integration test workflow: {name}",
                enabled=True,
                capabilities=[name],
            ))
            session.add(WorkflowVersion(
                workflow_version_id=ver_id,
                workflow_id=wf_id,
                version=1,
                status="active",
            ))
            for i, s in enumerate(steps):
                session.add(WorkflowStep(
                    workflow_step_id=s["id"],
                    workflow_version_id=ver_id,
                    position=i,
                    name=s.get("name", f"step_{i}"),
                    type=s["type"],
                    enabled=s.get("enabled", True),
                    config=s.get("config", {}),
                    transitions=s.get("transitions", {}),
                    is_start=s.get("is_start", False),
                    is_terminal=s.get("is_terminal", False),
                    is_resume=s.get("is_resume", False),
                ))

    return {
        "workflow_id": wf_id,
        "version_id": ver_id,
        "step_ids": [s["id"] for s in steps],
    }


async def seed_resource(
    session_factory,
    kind: str,
    name: str,
    config: dict | None = None,
    enabled: bool = True,
) -> uuid.UUID:
    """Insert a resource row (tool activation) into SQLite."""
    resource_id = uuid.uuid4()
    async with session_factory() as session:
        async with session.begin():
            session.add(Resource(
                resource_id=resource_id,
                name=name,
                kind=kind,
                provider="default",
                config=config or {},
                enabled=enabled,
            ))
    return resource_id


async def seed_policy(
    session_factory,
    subject_type: str,
    subject: str,
    effect: str = "allow",
    conditions: dict | None = None,
    priority: int = 0,
    enabled: bool = True,
) -> uuid.UUID:
    """Insert an access-policy rule into SQLite."""
    rule_id = uuid.uuid4()
    async with session_factory() as session:
        async with session.begin():
            session.add(AccessPolicyRule(
                rule_id=rule_id,
                subject_type=subject_type,
                subject=subject,
                effect=effect,
                conditions=conditions or {},
                priority=priority,
                enabled=enabled,
            ))
    return rule_id


# ---------------------------------------------------------------------------
# Step loader factory
# ---------------------------------------------------------------------------

def make_step_loader(*extra_step_classes) -> StepLoader:
    """Build a fresh StepLoader with built-ins + any test-specific step classes.

    Using a local loader (not the global step_registry) keeps tests isolated.
    """
    loader = StepLoader()
    loader.register(DoneStep)
    loader.register(FailedStep)
    loader.register(PassthroughStep)
    for cls in extra_step_classes:
        loader.register(cls)
    return loader


# ---------------------------------------------------------------------------
# Container + app builders
# ---------------------------------------------------------------------------

def _make_gateway_service(workflow_repo, resource_repo, *, with_generic_json_detector: bool = False):
    detectors = [OpenAIProtocolDetector(base_path="/v1")]
    if with_generic_json_detector:
        detectors.append(GenericJsonProtocolDetector())
    return GatewayService(
        protocol_detector=CompositeProtocolDetector(detectors),
        trigger_resolver=DefaultTriggerResolver(),
        workflow_repository=workflow_repo,
        resource_repository=resource_repo,
    )


def build_test_container(
    workflow_repo,
    resource_repo,
    *,
    openai_workflow_mapping: dict[str, str] | None = None,
    include_generic_json: bool = False,
    model_response: str = "stub response",
    step_loader: StepLoader | None = None,
    with_session_store: bool = False,
) -> Container:
    """Build a fully-wired Container for integration tests.

    - Uses SQLite-backed repositories (passed in as fixtures).
    - Uses StubModelClient for all LLM calls.
    - Never reads a YAML config file.
    """
    gateway = _make_gateway_service(
        workflow_repo, resource_repo,
        with_generic_json_detector=include_generic_json,
    )
    loader = step_loader or make_step_loader()
    stub_client = StubModelClient(response=model_response)

    coordinator_factory = SimpleStoreCoordinatorFactory() if with_session_store else None

    executor = StepMachineWorkflowExecutor(
        loader=loader,
        model_clients={StubModelClient.PROVIDER_NAME: stub_client},
        session_access=(
            SessionAccess(coordinator_factory, enforced=False)
            if coordinator_factory is not None else None
        ),
    )

    adapters = []
    if openai_workflow_mapping is not None:
        adapters.append(OpenAIHttpProtocolAdapter(
            base_path="/v1",
            model_backend=None,
            workflow_mapping=openai_workflow_mapping,
        ))
    if include_generic_json:
        adapters.append(GenericJsonHttpProtocolAdapter())

    return Container(
        configuration_service=cast(Any, None),
        composition_service=cast(Any, None),
        runtime_context=cast(Any, None),
        gateway_service=gateway,
        subsystems=[],
        workflow_repository=workflow_repo,
        resource_repository=resource_repo,
        http_protocol_adapters=adapters,
        workflow_executor=executor,
        license=StubLicenseAdapter(),
    )


def build_test_app(container: Container) -> FastAPI:
    """Create a plain FastAPI app with the container pre-set on app.state."""
    app = FastAPI(title="nlght-integration-test")
    app.state.container = container

    for adapter in container.http_protocol_adapters:
        app.include_router(adapter.build_router())

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app
