# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Exercise session ownership and coordinator state across requests.

The tests cover session-key extraction, result promotion and discard, and
sharing a generic coordinator slot between workflow steps.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from integration._helpers import (
    build_test_app,
    build_test_container,
    make_step_loader,
    seed_workflow,
)
from nlght.core.hive_mind.models import AtomType, WriteIntent
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

pytestmark = pytest.mark.integration

_MAPPING = {"chat_completions": "wf"}
_MSG = [{"role": "user", "content": "hello"}]


_CAPTURED: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _clear_captured():
    _CAPTURED.clear()
    yield
    _CAPTURED.clear()


class CoordinatorInspectorStep(StepBase):
    """Records whether a coordinator is present."""
    TYPE = "test_coord_inspector"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        _CAPTURED["coordinator_present"] = ctx.store_coordinator is not None
        return StepResult(ctx=ctx, verdict="done")


class AtomWriterFinishStep(StepBase):
    """Writes a RESULT atom and calls finish_task (promotes)."""
    TYPE = "test_atom_writer_finish"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        coord = ctx.store_coordinator
        coord.start_task(ctx.correlation_id)
        coord.write(WriteIntent(
            atom_type=AtomType.RESULT,
            content="promoted content",
            task_id=ctx.correlation_id,
            tags=["test"],
        ))
        coord.finish_task(
            task_id=ctx.correlation_id,
            entities=[],
            turn_nr=0,
        )
        # Read back promoted results
        context = coord.get_context([])
        _CAPTURED["results"] = [r.content for r in context.get("known_results", [])]
        return StepResult(ctx=ctx, verdict="done")


class AtomWriterDiscardStep(StepBase):
    """Writes a RESULT atom and calls discard_task (does NOT promote)."""
    TYPE = "test_atom_writer_discard"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        coord = ctx.store_coordinator
        coord.start_task(ctx.correlation_id)
        coord.write(WriteIntent(
            atom_type=AtomType.RESULT,
            content="discarded content",
            task_id=ctx.correlation_id,
        ))
        coord.discard_task(ctx.correlation_id)
        context = coord.get_context([])
        _CAPTURED["results"] = [r.content for r in context.get("known_results", [])]
        return StepResult(ctx=ctx, verdict="done")


class SlotWriterStep(StepBase):
    """Writes a value into a slot."""
    TYPE = "test_slot_writer"
    SLOT = "test:shared_value"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        ctx.store_coordinator.set_slot(self.SLOT, {"answer": 42})
        return StepResult(ctx=ctx, verdict="next")


class SlotReaderStep(StepBase):
    """Reads the slot written by SlotWriterStep."""
    TYPE = "test_slot_reader"
    SLOT = "test:shared_value"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        _CAPTURED["slot_value"] = ctx.store_coordinator.get_slot(self.SLOT, lambda: None)
        return StepResult(ctx=ctx, verdict="done")


async def test_coordinator_provisioned_when_user_field_present(
    session_factory, workflow_repo, resource_repo,
):
    """Provision a session coordinator from the OpenAI user field."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_coord_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(CoordinatorInspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": _MSG, "user": "session-alice"},
        )

    assert resp.status_code == 200
    assert _CAPTURED.get("coordinator_present") is True


async def test_coordinator_is_ephemeral_without_user_field(
    session_factory, workflow_repo, resource_repo,
):
    """Use an ephemeral coordinator when the request has no user field."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_coord_inspector", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(CoordinatorInspectorStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            # Without a user field, the coordinator has no durable session key.
            json={"model": "test", "messages": _MSG},
        )

    assert resp.status_code == 200
    assert _CAPTURED.get("coordinator_present") is True


async def test_finish_task_promotes_result_atoms(
    session_factory, workflow_repo, resource_repo,
):
    """Promote result atoms when their task finishes."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_atom_writer_finish", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(AtomWriterFinishStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": _MSG, "user": "session-bob"},
        )

    assert "promoted content" in _CAPTURED.get("results", [])


async def test_discard_task_does_not_promote_atoms(
    session_factory, workflow_repo, resource_repo,
):
    """Keep discarded task atoms out of session results."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "test_atom_writer_discard", "is_start": True, "is_terminal": True},
    ])
    loader = make_step_loader(AtomWriterDiscardStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": _MSG, "user": "session-charlie"},
        )

    assert "discarded content" not in _CAPTURED.get("results", [])


async def test_slot_written_in_step_a_is_readable_in_step_b(
    session_factory, workflow_repo, resource_repo,
):
    """Share a generic coordinator slot across workflow steps."""
    reader_id = uuid.uuid4()
    writer_id = uuid.uuid4()

    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": writer_id,
            "type": "test_slot_writer",
            "is_start": True,
            "transitions": {"next": str(reader_id)},
        },
        {
            "id": reader_id,
            "type": "test_slot_reader",
            "is_terminal": True,
        },
    ])
    loader = make_step_loader(SlotWriterStep, SlotReaderStep)
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping=_MAPPING,
        step_loader=loader,
        with_session_store=True,
    )
    with TestClient(build_test_app(container)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": _MSG, "user": "session-dave"},
        )

    assert resp.status_code == 200
    assert _CAPTURED.get("slot_value") == {"answer": 42}
