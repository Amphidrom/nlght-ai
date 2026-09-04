# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""End-to-end through the full stack: HTTP → adapter → submit → a real worker
claims and executes over the durable queue → broker → gateway relay → response.

A real ExecutionWorker polls concurrently with the request, so the database must
be genuinely shared across connections — a temp file, not ``:memory:`` (whose
per-connection database breaks once a background worker holds its own
connection). An ASGI transport keeps the gateway request, the worker, the broker
and the repository on one event loop, the way the runtime wires them, with a
tiny emitting step standing in for a model. Validates the unified execution path
(C1) and streaming through the worker (C2) that unit fakes cannot exercise.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, cast

import httpx2
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from integration._helpers import StubLicenseAdapter, make_step_loader, seed_workflow
from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.adapters.outbound.persistence.execution_repository import SqlAlchemyExecutionRepository
from nlght.adapters.outbound.persistence.models import Base
from nlght.adapters.outbound.persistence.resource_repository import SqlAlchemyResourceRepository
from nlght.adapters.outbound.persistence.workflow_repository import SqlAlchemyWorkflowRepository
from nlght.adapters.outbound.protocol.composite import CompositeProtocolDetector
from nlght.adapters.outbound.protocol.generic_json import GenericJsonProtocolDetector
from nlght.adapters.outbound.protocol.openai import OpenAIProtocolDetector
from nlght.adapters.outbound.signals.execution_stream import InProcessExecutionStreamBroker
from nlght.adapters.outbound.trigger.default import DefaultTriggerResolver
from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.application.entry.gateway_service import GatewayService
from nlght.application.execution.dispatch_service import ExecutionDispatchService
from nlght.application.execution.worker import ExecutionWorker, ExecutionWorkerSettings
from nlght.bootstrap.wiring import Container
from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

pytestmark = pytest.mark.integration


class _EmitStep(StepBase):
    """A start+terminal step that streams two tokens and a done — no model."""

    TYPE = "test.emit"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        for token in ("he", "llo"):
            await ctx.emitter.emit(Signal(role="assistant", content=token, kind="token"))
        await ctx.emitter.emit(Signal(role="assistant", content="", kind="done"))
        return StepResult(ctx=ctx, verdict="DEFAULT")


@pytest.fixture
async def file_engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
    try:
        os.remove(path)
    except OSError:
        pass


async def _build_app_with_worker(engine, adapters) -> tuple[FastAPI, ExecutionWorker]:
    workflow_repo = SqlAlchemyWorkflowRepository(engine)
    resource_repo = SqlAlchemyResourceRepository(engine)
    exec_repo = SqlAlchemyExecutionRepository(engine)
    broker = InProcessExecutionStreamBroker()
    executor = StepMachineWorkflowExecutor(loader=make_step_loader(_EmitStep))
    worker = ExecutionWorker(
        repository=exec_repo,
        workflow_repository=workflow_repo,
        executor=executor,
        stream_broker=broker,
        settings=ExecutionWorkerSettings(
            concurrency=1, poll_interval_seconds=0.02, lease_seconds=5, heartbeat_seconds=1
        ),
    )
    gateway = GatewayService(
        protocol_detector=CompositeProtocolDetector(
            [OpenAIProtocolDetector(base_path="/v1"), GenericJsonProtocolDetector()]
        ),
        trigger_resolver=DefaultTriggerResolver(),
        workflow_repository=workflow_repo,
        resource_repository=resource_repo,
    )
    container = Container(
        configuration_service=cast(Any, None),
        composition_service=cast(Any, None),
        runtime_context=cast(Any, None),
        gateway_service=gateway,
        workflow_repository=workflow_repo,
        resource_repository=resource_repo,
        http_protocol_adapters=adapters,
        workflow_executor=executor,
        license=StubLicenseAdapter(),
        execution_dispatcher=ExecutionDispatchService(exec_repo),
        execution_stream_broker=broker,
        execution_worker=worker,
    )
    app = FastAPI(title="nlght-e2e")
    app.state.container = container
    for adapter in adapters:
        app.include_router(adapter.build_router())
    await worker.start()
    return app, worker


def _client(app: FastAPI) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://e2e")


async def test_streaming_chat_completion_runs_on_a_worker_and_streams_back(file_engine) -> None:
    session_factory = async_sessionmaker(file_engine, class_=AsyncSession, expire_on_commit=False)
    await seed_workflow(
        session_factory=session_factory,
        name="stream-flow",
        steps=[{"type": "test.emit", "is_start": True, "is_terminal": True}],
    )
    adapter = OpenAIHttpProtocolAdapter(
        base_path="/v1", model_backend=None,
        workflow_mapping={"chat_completions": "stream-flow"},
    )
    app, worker = await _build_app_with_worker(file_engine, [adapter])
    try:
        async with _client(app) as client:
            resp = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ),
                timeout=10,
            )
        assert resp.status_code == 200
        body = resp.text
        assert "he" in body and "llo" in body
        assert "[DONE]" in body
    finally:
        await worker.stop()


async def test_sync_generic_json_runs_on_a_worker_and_returns_the_result(file_engine) -> None:
    session_factory = async_sessionmaker(file_engine, class_=AsyncSession, expire_on_commit=False)
    await seed_workflow(
        session_factory=session_factory,
        name="hook-flow",
        steps=[{"type": "test.emit", "is_start": True, "is_terminal": True}],
    )
    adapter = GenericJsonHttpProtocolAdapter(workflow_mapping={"/hooks/run": "hook-flow"})
    app, worker = await _build_app_with_worker(file_engine, [adapter])
    try:
        async with _client(app) as client:
            resp = await asyncio.wait_for(client.post("/hooks/run", json={}), timeout=10)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"
    finally:
        await worker.stop()
