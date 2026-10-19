# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Route OpenAI and generic JSON requests through configured workflows.

The tests cover proxy fallback, invalid bodies, missing generic workflows,
workflow responses, and the health endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from integration._helpers import build_test_app, build_test_container, seed_workflow

pytestmark = pytest.mark.integration


async def test_openai_chat_completions_routes_through_workflow(
    session_factory, workflow_repo, resource_repo,
):
    """POST /v1/chat/completions reaches the step machine and returns 200."""
    await seed_workflow(session_factory, "my-agent", steps=[
        {"type": "done", "is_start": True, "is_terminal": True},
    ])

    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping={"chat_completions": "my-agent"},
    )
    app = build_test_app(container)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "choices" in body
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_openai_chat_completions_no_workflow_mapping_falls_back_to_proxy(
    workflow_repo, resource_repo,
):
    """Use the upstream proxy when no workflow mapping matches the request."""
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping={},  # An empty map routes to the proxy.
    )
    app = build_test_app(container)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )

    # 503 because no model_provider_base_url is set
    assert resp.status_code == 503


def test_openai_chat_completions_invalid_body_returns_422(
    workflow_repo, resource_repo,
):
    """Reject a malformed OpenAI request before workflow execution."""
    container = build_test_container(
        workflow_repo, resource_repo,
        openai_workflow_mapping={"chat_completions": "any"},
    )
    app = build_test_app(container)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 422


async def test_generic_json_workflow_not_found_returns_404(
    session_factory, workflow_repo, resource_repo,
):
    """When no matching workflow exists in the DB the gateway returns 404."""
    # No workflow is seeded, so "unknown-op" cannot resolve.
    container = build_test_container(
        workflow_repo, resource_repo,
        include_generic_json=True,
    )
    app = build_test_app(container)

    with TestClient(app) as client:
        resp = client.post("/unknown-op", json={})

    assert resp.status_code == 404
    assert "workflow" in resp.json()["detail"].lower() or "No enabled" in resp.json()["detail"]


async def test_generic_json_found_workflow_returns_200(
    session_factory, workflow_repo, resource_repo,
):
    """A seeded workflow reached via the generic-JSON adapter returns 200.

    The GenericJson trigger resolver always maps any path to the operation
    'generic_json_request', so the workflow must carry that name.
    """
    await seed_workflow(session_factory, "generic_json_request", steps=[
        {"type": "done", "is_start": True, "is_terminal": True},
    ])

    container = build_test_container(
        workflow_repo, resource_repo,
        include_generic_json=True,
    )
    app = build_test_app(container)

    with TestClient(app) as client:
        resp = client.post("/my-pipeline", json={})

    assert resp.status_code == 200


def test_healthz_always_returns_ok(workflow_repo, resource_repo):
    container = build_test_container(workflow_repo, resource_repo)
    app = build_test_app(container)

    with TestClient(app) as client:
        resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
