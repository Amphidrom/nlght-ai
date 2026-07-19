# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""UC2 — Step Machine Behaviour

Tests the contracts documented in creating-workflows and built-in-steps:

  - `done` step terminates the workflow, produces no content.
  - `failed` step terminates with WorkflowExecutionError (→ 500 from adapter).
  - `passthrough → done` two-step workflow.
  - `passthrough` with `verdict` config override routes to the correct step.
  - `DEFAULT` transition fallback when no exact verdict match.
  - Hop limit (20) stops runaway workflows.
  - Missing transition → WorkflowConfigurationError (→ 500).
  - Unknown step type → WorkflowConfigurationError (→ 500).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from integration._helpers import (
    StubModelClient,
    build_test_app,
    build_test_container,
    seed_workflow,
)

pytestmark = pytest.mark.integration

_MAPPING = {"chat_completions": "wf"}
_MESSAGES = [{"role": "user", "content": "hello"}]
# The executor no longer falls back to the first-registered client; a step must
# resolve its provider explicitly (via config or payload).  See executor._bind_llm.
_PROVIDER = {"model_provider": StubModelClient.PROVIDER_NAME}


def _post(client: TestClient, *, stream: bool = False) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": _MESSAGES, "stream": stream},
    )
    return resp


# ---------------------------------------------------------------------------
# done — terminates with empty content
# ---------------------------------------------------------------------------

async def test_done_step_returns_200_with_empty_content(
    session_factory, workflow_repo, resource_repo,
):
    """The `done` built-in terminates the workflow; the response has empty content."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "done", "is_start": True, "is_terminal": True},
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container)) as client:
        resp = _post(client)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == ""


# ---------------------------------------------------------------------------
# failed — raises WorkflowExecutionError
# ---------------------------------------------------------------------------

async def test_failed_step_causes_500(
    session_factory, workflow_repo, resource_repo,
):
    """The `failed` built-in raises WorkflowExecutionError — the OpenAI adapter
    does not catch it, so FastAPI returns 500."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "failed", "is_start": True, "is_terminal": True},
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container), raise_server_exceptions=False) as client:
        resp = _post(client)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# passthrough → done — two-step workflow
# ---------------------------------------------------------------------------

async def test_passthrough_then_done_returns_200(
    session_factory, workflow_repo, resource_repo,
):
    """passthrough emits DEFAULT verdict, transitions to done step."""
    done_id = uuid.uuid4()
    passthrough_id = uuid.uuid4()

    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": passthrough_id,
            "type": "passthrough",
            "is_start": True,
            "config": dict(_PROVIDER),
            "transitions": {"DEFAULT": str(done_id)},
        },
        {
            "id": done_id,
            "type": "done",
            "is_terminal": True,
        },
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container)) as client:
        resp = _post(client)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# passthrough with custom verdict config
# ---------------------------------------------------------------------------

async def test_passthrough_custom_verdict_routes_to_correct_step(
    session_factory, workflow_repo, resource_repo,
):
    """passthrough with config.verdict='custom' routes to the step mapped under 'custom'."""
    target_id = uuid.uuid4()
    start_id = uuid.uuid4()

    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": start_id,
            "type": "passthrough",
            "is_start": True,
            "config": {"verdict": "custom", **_PROVIDER},
            "transitions": {"custom": str(target_id)},
        },
        {
            "id": target_id,
            "type": "done",
            "is_terminal": True,
        },
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container)) as client:
        resp = _post(client)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# DEFAULT fallback transition
# ---------------------------------------------------------------------------

async def test_default_transition_used_when_no_exact_match(
    session_factory, workflow_repo, resource_repo,
):
    """If the step returns a verdict not in transitions, DEFAULT is used."""
    fallback_id = uuid.uuid4()
    start_id = uuid.uuid4()

    # passthrough returns "DEFAULT" — which is not in the transitions dict
    # (only "other" is).  The executor should look for "DEFAULT" key instead.
    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": start_id,
            "type": "passthrough",
            "is_start": True,
            "config": dict(_PROVIDER),
            "transitions": {
                "other": "nonexistent",
                "DEFAULT": str(fallback_id),
            },
        },
        {
            "id": fallback_id,
            "type": "done",
            "is_terminal": True,
        },
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container)) as client:
        resp = _post(client)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Missing transition → WorkflowConfigurationError
# ---------------------------------------------------------------------------

async def test_missing_transition_causes_500(
    session_factory, workflow_repo, resource_repo,
):
    """A step that returns a verdict with no matching transition and no DEFAULT
    raises WorkflowConfigurationError."""
    await seed_workflow(session_factory, "wf", steps=[
        {
            "type": "passthrough",
            "is_start": True,
            "transitions": {},  # no DEFAULT, no exact match for "DEFAULT"
        },
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container), raise_server_exceptions=False) as client:
        resp = _post(client)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Unknown step type → WorkflowConfigurationError
# ---------------------------------------------------------------------------

async def test_unknown_step_type_causes_500(
    session_factory, workflow_repo, resource_repo,
):
    """A step whose type is not registered in the loader raises WorkflowConfigurationError."""
    await seed_workflow(session_factory, "wf", steps=[
        {"type": "nonexistent_step_type", "is_start": True, "is_terminal": True},
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container), raise_server_exceptions=False) as client:
        resp = _post(client)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Hop limit — 20 hops stops runaway workflows
# ---------------------------------------------------------------------------

async def test_hop_limit_stops_workflow_after_20_hops(
    session_factory, workflow_repo, resource_repo,
):
    """A workflow that loops forever is stopped at 20 hops.

    The executor logs a hard-stop warning and breaks out of the loop — it does
    NOT raise an exception, so the response is still 200.
    """
    # Single passthrough that loops back to itself (no target step → uses DEFAULT)
    loop_id = uuid.uuid4()
    await seed_workflow(session_factory, "wf", steps=[
        {
            "id": loop_id,
            "type": "passthrough",
            "is_start": True,
            "config": dict(_PROVIDER),
            "transitions": {"DEFAULT": str(loop_id)},  # loops to itself
        },
    ])
    container = build_test_container(
        workflow_repo, resource_repo, openai_workflow_mapping=_MAPPING,
    )
    with TestClient(build_test_app(container)) as client:
        resp = _post(client)

    # Executor hard-stops at 20 hops and returns normally
    assert resp.status_code == 200
