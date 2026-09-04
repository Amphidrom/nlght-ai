# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nlght.adapters.inbound.http.admin import router


class _Templates:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def TemplateResponse(self, request: object, template: str, context: dict) -> SimpleNamespace:
        self.calls.append((template, context))
        return SimpleNamespace(template=template, context=context, set_cookie=lambda *args, **kwargs: None)


class _Service:
    def __init__(self) -> None:
        self.workflow = SimpleNamespace(
            workflow_id=uuid.uuid4(),
            enabled=True,
            versions=[
                SimpleNamespace(version=1, workflow_version_id=uuid.uuid4(), steps=[]),
                SimpleNamespace(version=3, workflow_version_id=uuid.uuid4(), steps=[]),
            ],
        )
        self.version = SimpleNamespace(
            workflow_version_id=uuid.uuid4(),
            workflow=SimpleNamespace(name="wf"),
            steps=[
                SimpleNamespace(
                    workflow_step_id=uuid.uuid4(),
                    name="select",
                    type="phase",
                    position=2,
                    is_start=True,
                    is_terminal=False,
                    is_resume=False,
                    transitions={"ok": "fetch"},
                )
            ],
        )
        self.step = SimpleNamespace(
            workflow_step_id=uuid.uuid4(),
            workflow_version_id=uuid.uuid4(),
            position=4,
        )
        self.resource = SimpleNamespace(resource_id=uuid.uuid4(), enabled=True)
        self.policy = SimpleNamespace(rule_id=uuid.uuid4(), enabled=True)
        self.saved_graph: dict | None = None
        self.updated_step: dict | None = None
        self.updated_resource: dict | None = None
        self.updated_policy: dict | None = None
        self.deleted_policy: uuid.UUID | None = None

    async def list_workflows(self) -> list[object]:
        return [self.workflow]

    async def list_resources(self) -> list[object]:
        return [self.resource]

    async def create_workflow(self, **kwargs: object) -> object:
        self.created_workflow = kwargs
        return self.workflow

    async def get_workflow(self, wid: uuid.UUID) -> object | None:
        return self.workflow

    async def toggle_workflow(self, wid: uuid.UUID) -> object:
        self.workflow.enabled = not self.workflow.enabled
        return self.workflow

    async def fork_latest_version(self, wid: uuid.UUID) -> object:
        return self.version

    async def get_version(self, vid: uuid.UUID) -> object | None:
        return self.version

    async def activate_version(self, vid: uuid.UUID) -> None:
        self.activated = vid

    async def save_graph(self, vid: uuid.UUID, transitions: dict[str, dict[str, str]]) -> None:
        self.saved_graph = transitions

    async def create_step(self, **kwargs: object) -> object:
        self.created_step = kwargs
        return self.step

    async def get_step(self, sid: uuid.UUID) -> object | None:
        return self.step

    async def update_step(self, sid: uuid.UUID, **fields: object) -> object:
        self.updated_step = fields
        return self.step

    async def delete_step(self, sid: uuid.UUID) -> None:
        self.deleted_step = sid

    async def create_resource(self, **kwargs: object) -> object:
        self.created_resource = kwargs
        return self.resource

    async def get_resource(self, rid: uuid.UUID) -> object | None:
        return self.resource

    async def update_resource(self, rid: uuid.UUID, **fields: object) -> object:
        self.updated_resource = fields
        return self.resource

    async def toggle_resource(self, rid: uuid.UUID) -> object:
        self.resource.enabled = not self.resource.enabled
        return self.resource

    async def list_policies(self) -> list[object]:
        return [self.policy]

    async def create_policy(self, **kwargs: object) -> object:
        self.created_policy = kwargs
        return self.policy

    async def get_policy(self, rule_id: uuid.UUID) -> object | None:
        return self.policy

    async def update_policy(self, rule_id: uuid.UUID, **fields: object) -> object:
        self.updated_policy = fields
        return self.policy

    async def toggle_policy(self, rule_id: uuid.UUID) -> object:
        self.policy.enabled = not self.policy.enabled
        return self.policy

    async def delete_policy(self, rule_id: uuid.UUID) -> None:
        self.deleted_policy = rule_id


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(engine=object()))),
        cookies={},
        url=SimpleNamespace(scheme="http"),
    )


def _starlette_request(*, method: str = "GET", cookie: str = "", token: str = "") -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie:
        headers.append((b"cookie", f"{router._CSRF_COOKIE}={cookie}".encode()))
    if token:
        headers.append((b"x-csrf-token", token.encode()))
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "scheme": "http",
        "path": "/admin/test",
        "raw_path": b"/admin/test",
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1),
    }
    return Request(scope)


def test_router_helpers_parse_inputs_and_missing_engine() -> None:
    assert router._parse_json_field("", {"a": 1}) == {"a": 1}
    assert router._parse_json_field('{"a": 2}', {}) == {"a": 2}
    assert router._parse_capabilities(" web\n\ncode ") == ["web", "code"]
    assert router._redirect("/admin").status_code == 303

    with pytest.raises(HTTPException) as invalid:
        router._parse_json_field("{bad", {})
    assert invalid.value.status_code == 422

    assert router._parse_conditions("") == {}
    assert router._parse_conditions('{"model": "llama3"}') == {"model": ["llama3"]}
    assert router._parse_conditions('{"model": ["a", "b"]}') == {"model": ["a", "b"]}
    for bad in ('["not-an-object"]', '{"model": 3}', '{"model": [1]}'):
        with pytest.raises(HTTPException) as bad_conditions:
            router._parse_conditions(bad)
        assert bad_conditions.value.status_code == 422

    with pytest.raises(HTTPException) as bad_subject_type:
        router._validate_policy_fields("nonsense", "allow")
    assert bad_subject_type.value.status_code == 422
    with pytest.raises(HTTPException) as bad_effect:
        router._validate_policy_fields("tool", "nonsense")
    assert bad_effect.value.status_code == 422

    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace())))
    with pytest.raises(HTTPException) as missing:
        router._svc(req)
    assert missing.value.status_code == 503


async def test_csrf_verification_accepts_safe_methods_and_matching_header() -> None:
    await router._verify_csrf(_starlette_request())
    await router._verify_csrf(_starlette_request(method="POST", cookie="same", token="same"))

    for request in (
        _starlette_request(method="POST"),
        _starlette_request(method="POST", cookie="expected"),
        _starlette_request(method="POST", cookie="expected", token="wrong"),
    ):
        with pytest.raises(HTTPException) as rejected:
            await router._verify_csrf(request)
        assert rejected.value.status_code == 403


async def test_admin_static_serves_only_allowlisted_packaged_assets() -> None:
    response = await router.admin_static("htmx.min.js")
    assert str(response.path).endswith("htmx.min.js")
    with pytest.raises(HTTPException) as missing:
        await router.admin_static("../router.py")
    assert missing.value.status_code == 404


async def test_an_allowlisted_but_unvendored_asset_is_a_404_not_a_500() -> None:
    # The dagre pair is optional — the graph falls back to a built-in layout
    # without it — so a checkout that does not carry it must answer 404 rather
    # than raising on a missing file.
    absent = [
        asset for asset in ("dagre.min.js", "cytoscape-dagre.min.js")
        if not (router._STATIC_DIR / asset).is_file()
    ]
    if not absent:
        pytest.skip("both optional assets are vendored in this checkout")
    with pytest.raises(HTTPException) as missing:
        await router.admin_static(absent[0])
    assert missing.value.status_code == 404


async def test_router_dashboard_workflow_version_and_graph_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service()
    templates = _Templates()
    monkeypatch.setattr(router, "_svc", lambda request: svc)
    # The dashboard also reports what the runtime is doing, which reaches the
    # execution repository — the fake engine on the request would not survive it.
    monkeypatch.setattr(router, "_executions", lambda request: _Executions())
    monkeypatch.setattr(router, "_tmpl", lambda: templates)
    req = _request()

    dashboard = await router.dashboard(req)
    workflows = await router.list_workflows(req)
    detail = await router.workflow_detail(req, svc.workflow.workflow_id)
    toggled = await router.toggle_workflow(req, svc.workflow.workflow_id)
    created = await router.create_workflow(req, " name ", " desc ", "on", "web\ncode")
    forked = await router.fork_version(req, svc.workflow.workflow_id)
    version = await router.version_detail(req, svc.version.workflow_version_id)
    activated = await router.activate_version(req, svc.version.workflow_version_id)
    saved = await router.save_graph(req, svc.version.workflow_version_id, router.GraphSaveRequest(transitions={"a": {"ok": "b"}}))

    assert dashboard.template == "dashboard.html"
    assert workflows.template == "workflows/list.html"
    assert [v.version for v in detail.context["versions"]] == [3, 1]
    assert toggled.template == "workflows/_row.html"
    assert created.status_code == 303
    assert forked.status_code == 303
    assert version.context["steps_json"].startswith("[")
    assert activated.status_code == 303
    assert saved == {"ok": True}
    assert svc.saved_graph == {"a": {"ok": "b"}}
    assert svc.created_workflow == {
        "name": "name",
        "description": "desc",
        "enabled": True,
        "capabilities": ["web", "code"],
        "concurrency": "non-blocking",
        # Left empty on the form, which means the runtime's default rather than
        # a number — the three states are what makes 0 mean unlimited.
        "max_hops": None,
    }


async def test_router_step_and_resource_forms_and_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service()
    templates = _Templates()
    monkeypatch.setattr(router, "_svc", lambda request: svc)
    monkeypatch.setattr(router, "_tmpl", lambda: templates)
    req = _request()

    assert (await router.new_workflow_form(req)).template == "workflows/form.html"
    assert (await router.new_step_form(req, svc.version.workflow_version_id)).context["next_position"] == 3
    assert (await router.edit_step_form(req, svc.step.workflow_step_id)).context["step"] is svc.step
    assert (await router.create_step(req, svc.version.workflow_version_id, " s ", " phase ", 1, '{"x": 1}', "on", "on", "", "on")).status_code == 303
    assert svc.created_step["config"] == {"x": 1}
    assert (await router.update_step(req, svc.step.workflow_step_id, " s2 ", " phase2 ", 2, '{"y": 2}', "", "", "on", "")).status_code == 303
    assert svc.updated_step == {
        "name": "s2",
        "type": "phase2",
        "position": 2,
        "enabled": False,
        "config": {"y": 2},
        "is_start": False,
        "is_terminal": True,
        "is_resume": False,
    }
    assert (await router.delete_step(req, svc.step.workflow_step_id)).status_code == 303

    assert (await router.list_resources(req)).template == "resources/list.html"
    assert (await router.new_resource_form(req)).template == "resources/form.html"
    assert (await router.create_resource(req, " r ", " model ", " openai ", '{"m": 1}', "on")).status_code == 303
    assert svc.created_resource["config"] == {"m": 1}
    assert (await router.edit_resource_form(req, svc.resource.resource_id)).context["resource"] is svc.resource
    assert (await router.update_resource(req, svc.resource.resource_id, " r2 ", " tool ", " builtin ", '{"n": 2}', "")).status_code == 303
    assert svc.updated_resource["enabled"] is False
    assert svc.updated_resource["name"] == "r2"
    assert (await router.toggle_resource(req, svc.resource.resource_id)).template == "resources/_row.html"


async def test_router_policy_forms_and_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _Service()
    templates = _Templates()
    monkeypatch.setattr(router, "_svc", lambda request: svc)
    monkeypatch.setattr(router, "_tmpl", lambda: templates)
    req = _request()

    assert (await router.list_policies(req)).template == "policies/list.html"
    assert (await router.new_policy_form(req)).template == "policies/form.html"
    assert (await router.create_policy(
        req, " tool ", " web_* ", "deny", '{"model": ["llama3*"]}', 5, "on",
    )).status_code == 303
    assert svc.created_policy == {
        "subject_type": "tool",
        "subject": "web_*",
        "effect": "deny",
        "conditions": {"model": ["llama3*"]},
        "priority": 5,
        "enabled": True,
    }
    assert (await router.edit_policy_form(req, svc.policy.rule_id)).context["policy"] is svc.policy
    assert (await router.update_policy(
        req, svc.policy.rule_id, "model", "*", "allow", "{}", 0, "",
    )).status_code == 303
    assert svc.updated_policy == {
        "subject_type": "model",
        "subject": "*",
        "effect": "allow",
        "conditions": {},
        "priority": 0,
        "enabled": False,
    }
    assert (await router.toggle_policy(req, svc.policy.rule_id)).template == "policies/_row.html"
    assert (await router.delete_policy(req, svc.policy.rule_id)).status_code == 303
    assert svc.deleted_policy == svc.policy.rule_id

    with pytest.raises(HTTPException) as invalid:
        await router.create_policy(req, "bogus", "x", "allow", "{}", 0, "")
    assert invalid.value.status_code == 422


async def test_router_raises_404_for_missing_domain_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Missing(_Service):
        async def get_workflow(self, wid: uuid.UUID) -> None:
            return None

        async def get_version(self, vid: uuid.UUID) -> None:
            return None

        async def get_step(self, sid: uuid.UUID) -> None:
            return None

        async def get_resource(self, rid: uuid.UUID) -> None:
            return None

        async def get_policy(self, rule_id: uuid.UUID) -> None:
            return None

    svc = _Missing()
    monkeypatch.setattr(router, "_svc", lambda request: svc)
    monkeypatch.setattr(router, "_tmpl", lambda: _Templates())
    req = _request()

    for call in (
        router.workflow_detail(req, uuid.uuid4()),
        router.version_detail(req, uuid.uuid4()),
        router.activate_version(req, uuid.uuid4()),
        router.new_step_form(req, uuid.uuid4()),
        router.edit_step_form(req, uuid.uuid4()),
        router.update_step(req, uuid.uuid4(), "n", "t", 1),
        router.delete_step(req, uuid.uuid4()),
        router.edit_resource_form(req, uuid.uuid4()),
        router.edit_policy_form(req, uuid.uuid4()),
        router.update_policy(req, uuid.uuid4(), "tool", "x"),
        router.delete_policy(req, uuid.uuid4()),
    ):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Executions — read-only
# ---------------------------------------------------------------------------


class _Executions:
    def __init__(self, *, found: bool = True) -> None:
        self.found = found
        self.asked: dict[str, object] = {}

    async def runs(self, **kwargs: object) -> dict[str, object]:
        self.asked = kwargs
        return {"runs": [], "total": 0, "limit": kwargs["limit"], "offset": kwargs["offset"]}

    async def run(self, execution_id: uuid.UUID, **kwargs: object) -> dict[str, object] | None:
        return {"record": SimpleNamespace(execution_id=execution_id)} if self.found else None

    async def workers(self) -> tuple[object, ...]:
        return ()

    async def counts(self, **kwargs: object) -> object:
        return SimpleNamespace(in_flight=0, of=lambda _status: 0)

    async def list_workflows_for_filter(self) -> list[object]:
        return []


async def test_executions_listing_passes_its_filters_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions = _Executions()
    templates = _Templates()
    monkeypatch.setattr(router, "_executions", lambda request: executions)
    monkeypatch.setattr(router, "_tmpl", lambda: templates)
    workflow_id = uuid.uuid4()

    await router.list_executions(
        _request(), status="failed", workflow_id=str(workflow_id), offset=50
    )

    assert executions.asked["status"].value == "failed"
    assert executions.asked["workflow_id"] == workflow_id
    assert executions.asked["offset"] == 50
    assert templates.calls[0][0] == "executions/list.html"


async def test_an_unknown_status_is_refused_rather_than_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Silently listing everything would answer a question nobody asked.
    monkeypatch.setattr(router, "_executions", lambda request: _Executions())
    monkeypatch.setattr(router, "_tmpl", lambda: _Templates())

    with pytest.raises(HTTPException) as exc:
        await router.list_executions(_request(), status="exploded")
    assert exc.value.status_code == 400


async def test_the_page_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    executions = _Executions()
    monkeypatch.setattr(router, "_executions", lambda request: executions)
    monkeypatch.setattr(router, "_tmpl", lambda: _Templates())

    await router.list_executions(_request(), limit=100_000)

    assert executions.asked["limit"] == 200


async def test_a_missing_execution_is_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router, "_executions", lambda request: _Executions(found=False))
    monkeypatch.setattr(router, "_tmpl", lambda: _Templates())

    with pytest.raises(HTTPException) as exc:
        await router.execution_detail(_request(), uuid.uuid4())
    assert exc.value.status_code == 404


async def test_workers_render_their_own_page(monkeypatch: pytest.MonkeyPatch) -> None:
    templates = _Templates()
    monkeypatch.setattr(router, "_executions", lambda request: _Executions())
    monkeypatch.setattr(router, "_tmpl", lambda: templates)

    await router.list_workers(_request())

    assert templates.calls[0][0] == "executions/workers.html"


def test_a_duration_reads_as_time_not_as_two_timestamps() -> None:
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
    finished = SimpleNamespace(
        started_at=start, created_at=start, completed_at=start + timedelta(seconds=90)
    )
    assert router._duration(finished) == "1 m 30 s"

    # An unfinished run measures against now, so a stuck execution shows a
    # growing number rather than an empty cell.
    running = SimpleNamespace(started_at=start, created_at=start, completed_at=None)
    assert router._duration(running) != ""


def test_pretty_json_survives_a_payload_it_cannot_serialize() -> None:
    assert router._pretty_json({"a": 1}) == '{\n  "a": 1\n}'
    assert "object" in router._pretty_json({"a": object()})
