# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import hmac
import json
import secrets
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from nlght.adapters.inbound.http.admin.services import AdminService

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_ASSETS = {"pico.classless.min.css", "htmx.min.js", "cytoscape.min.js"}
_templates: Jinja2Templates | None = None


def _tmpl() -> Jinja2Templates:
    global _templates
    if _templates is None:
        from fastapi.templating import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            Jinja2Templates as _Jinja2Templates,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        )
        _templates = _Jinja2Templates(directory=str(_TEMPLATES_DIR))
    return _templates


_CSRF_COOKIE = "nlght_admin_csrf"
_CSRF_FIELD = "csrf_token"


def _csrf_token(request: Request) -> str:
    return request.cookies.get(_CSRF_COOKIE) or secrets.token_urlsafe(32)


async def _verify_csrf(request: Request) -> None:
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    expected = request.cookies.get(_CSRF_COOKIE, "")
    supplied = request.headers.get("x-csrf-token", "")
    if not supplied and "application/x-www-form-urlencoded" in request.headers.get("content-type", ""):
        supplied = str((await request.form()).get(_CSRF_FIELD, ""))
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")


admin_router = APIRouter(default_response_class=HTMLResponse, dependencies=[Depends(_verify_csrf)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(request: Request) -> AdminService:
    engine = getattr(request.app.state.container, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="No database engine configured.")
    return AdminService(engine)


def _parse_json_field(raw: str, default: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON-object form field; empty input returns ``default``."""
    raw = raw.strip()
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="Value must be a JSON object.")
    return parsed


def _parse_capabilities(raw: str) -> list[str]:
    return [c.strip() for c in raw.splitlines() if c.strip()]


_POLICY_SUBJECT_TYPES = {"tool", "model", "playbook"}
_POLICY_EFFECTS = {"allow", "deny"}


def _parse_conditions(raw: str) -> dict[str, list[str]]:
    """Parse the conditions JSON field: {key: pattern | [patterns, ...]}."""
    parsed = _parse_json_field(raw, {})
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="Conditions must be a JSON object.")
    conditions: dict[str, list[str]] = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            conditions[str(key)] = [value]
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            conditions[str(key)] = list(value)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Condition '{key}' must be a string or a list of strings.",
            )
    return conditions


def _validate_policy_fields(subject_type: str, effect: str) -> None:
    if subject_type not in _POLICY_SUBJECT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"subject_type must be one of {sorted(_POLICY_SUBJECT_TYPES)}.",
        )
    if effect not in _POLICY_EFFECTS:
        raise HTTPException(
            status_code=422,
            detail=f"effect must be one of {sorted(_POLICY_EFFECTS)}.",
        )


def _redirect(path: str) -> RedirectResponse:
    return RedirectResponse(url=path, status_code=303)


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    token = _csrf_token(request)
    response = _tmpl().TemplateResponse(request, template, {**context, _CSRF_FIELD: token})
    if _CSRF_COOKIE not in request.cookies:
        response.set_cookie(_CSRF_COOKIE, token, httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/admin")
    return response


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@admin_router.get("/static/{asset}", response_class=FileResponse)
async def admin_static(asset: str) -> FileResponse:
    if asset not in _STATIC_ASSETS:
        raise HTTPException(status_code=404, detail="Static asset not found.")
    return FileResponse(_STATIC_DIR / asset)


@admin_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    svc = _svc(request)
    workflows = await svc.list_workflows()
    resources = await svc.list_resources()
    return _render(request, "dashboard.html", {
        "workflow_count": len(workflows),
        "resource_count": len(resources),
        "recent_workflows": workflows[:5],
    })


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

@admin_router.get("/workflows", response_class=HTMLResponse)
async def list_workflows(request: Request) -> HTMLResponse:
    svc = _svc(request)
    workflows = await svc.list_workflows()
    return _render(request, "workflows/list.html", {
        "workflows": workflows,
    })


@admin_router.get("/workflows/new", response_class=HTMLResponse)
async def new_workflow_form(request: Request) -> HTMLResponse:
    return _render(request, "workflows/form.html", {
        "workflow": None,
        "error": None,
    })


@admin_router.post("/workflows/new")
async def create_workflow(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    enabled: Annotated[str, Form()] = "",
    capabilities: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    workflow = await svc.create_workflow(
        name=name.strip(),
        description=description.strip() or None,
        enabled=(enabled == "on"),
        capabilities=_parse_capabilities(capabilities),
    )
    return _redirect(f"/admin/workflows/{workflow.workflow_id}")


@admin_router.get("/workflows/{wid}", response_class=HTMLResponse)
async def workflow_detail(request: Request, wid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    workflow = await svc.get_workflow(wid)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    versions_sorted = sorted(workflow.versions, key=lambda v: v.version, reverse=True)
    return _render(request, "workflows/detail.html", {
        "workflow": workflow,
        "versions": versions_sorted,
    })


@admin_router.post("/workflows/{wid}/toggle", response_class=HTMLResponse)
async def toggle_workflow(request: Request, wid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    workflow = await svc.toggle_workflow(wid)
    return _render(request, "workflows/_row.html", {
        "wf": workflow,
    })


@admin_router.post("/workflows/{wid}/versions/new")
async def fork_version(request: Request, wid: uuid.UUID) -> RedirectResponse:
    svc = _svc(request)
    version = await svc.fork_latest_version(wid)
    return _redirect(f"/admin/versions/{version.workflow_version_id}")


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

@admin_router.get("/versions/{vid}", response_class=HTMLResponse)
async def version_detail(request: Request, vid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    version = await svc.get_version(vid)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found.")
    steps_json = json.dumps([
        {
            "id": str(s.workflow_step_id),
            "name": s.name,
            "type": s.type,
            "position": s.position,
            "is_start": s.is_start,
            "is_terminal": s.is_terminal,
            "is_resume": s.is_resume,
            "transitions": s.transitions,
        }
        for s in version.steps
    ])
    return _render(request, "versions/detail.html", {
        "version": version,
        "workflow": version.workflow,
        "steps": version.steps,
        "steps_json": steps_json,
    })


@admin_router.post("/versions/{vid}/activate")
async def activate_version(request: Request, vid: uuid.UUID) -> RedirectResponse:
    svc = _svc(request)
    version = await svc.get_version(vid)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found.")
    await svc.activate_version(vid)
    return _redirect(f"/admin/versions/{vid}")


class GraphSaveRequest(BaseModel):
    transitions: dict[str, dict[str, str]]


@admin_router.post("/versions/{vid}/graph", response_class=JSONResponse)
async def save_graph(request: Request, vid: uuid.UUID, body: GraphSaveRequest) -> dict[str, bool]:
    svc = _svc(request)
    await svc.save_graph(vid, body.transitions)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@admin_router.get("/versions/{vid}/steps/new", response_class=HTMLResponse)
async def new_step_form(request: Request, vid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    version = await svc.get_version(vid)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found.")
    next_position = max((s.position for s in version.steps), default=0) + 1
    return _render(request, "steps/form.html", {
        "step": None,
        "version_id": str(vid),
        "next_position": next_position,
        "error": None,
    })


@admin_router.post("/versions/{vid}/steps/new")
async def create_step(
    request: Request,
    vid: uuid.UUID,
    name: Annotated[str, Form()],
    type_: Annotated[str, Form(alias="type")],
    position: Annotated[int, Form()],
    config_json: Annotated[str, Form()] = "{}",
    enabled: Annotated[str, Form()] = "",
    is_start: Annotated[str, Form()] = "",
    is_terminal: Annotated[str, Form()] = "",
    is_resume: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    config = _parse_json_field(config_json, {})
    await svc.create_step(
        vid=vid,
        position=position,
        name=name.strip(),
        type_=type_.strip(),
        enabled=(enabled == "on"),
        config=config,
        is_start=(is_start == "on"),
        is_terminal=(is_terminal == "on"),
        is_resume=(is_resume == "on"),
    )
    return _redirect(f"/admin/versions/{vid}")


@admin_router.get("/steps/{sid}", response_class=HTMLResponse)
async def edit_step_form(request: Request, sid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    step = await svc.get_step(sid)
    if step is None:
        raise HTTPException(status_code=404, detail="Step not found.")
    return _render(request, "steps/form.html", {
        "step": step,
        "version_id": str(step.workflow_version_id),
        "next_position": step.position,
        "error": None,
    })


@admin_router.post("/steps/{sid}")
async def update_step(
    request: Request,
    sid: uuid.UUID,
    name: Annotated[str, Form()],
    type_: Annotated[str, Form(alias="type")],
    position: Annotated[int, Form()],
    config_json: Annotated[str, Form()] = "{}",
    enabled: Annotated[str, Form()] = "",
    is_start: Annotated[str, Form()] = "",
    is_terminal: Annotated[str, Form()] = "",
    is_resume: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    step = await svc.get_step(sid)
    if step is None:
        raise HTTPException(status_code=404, detail="Step not found.")
    config = _parse_json_field(config_json, {})
    await svc.update_step(
        sid,
        name=name.strip(),
        type=type_.strip(),
        position=position,
        enabled=(enabled == "on"),
        config=config,
        is_start=(is_start == "on"),
        is_terminal=(is_terminal == "on"),
        is_resume=(is_resume == "on"),
    )
    return _redirect(f"/admin/versions/{step.workflow_version_id}")


@admin_router.post("/steps/{sid}/delete")
async def delete_step(request: Request, sid: uuid.UUID) -> RedirectResponse:
    svc = _svc(request)
    step = await svc.get_step(sid)
    if step is None:
        raise HTTPException(status_code=404, detail="Step not found.")
    vid = step.workflow_version_id
    await svc.delete_step(sid)
    return _redirect(f"/admin/versions/{vid}")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@admin_router.get("/resources", response_class=HTMLResponse)
async def list_resources(request: Request) -> HTMLResponse:
    svc = _svc(request)
    resources = await svc.list_resources()
    return _render(request, "resources/list.html", {
        "resources": resources,
    })


@admin_router.get("/resources/new", response_class=HTMLResponse)
async def new_resource_form(request: Request) -> HTMLResponse:
    return _render(request, "resources/form.html", {
        "resource": None,
        "error": None,
    })


@admin_router.post("/resources/new")
async def create_resource(
    request: Request,
    name: Annotated[str, Form()],
    kind: Annotated[str, Form()],
    provider: Annotated[str, Form()],
    config_json: Annotated[str, Form()] = "{}",
    enabled: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    config = _parse_json_field(config_json, {})
    resource = await svc.create_resource(
        name=name.strip(),
        kind=kind.strip(),
        provider=provider.strip(),
        config=config,
        enabled=(enabled == "on"),
    )
    return _redirect(f"/admin/resources/{resource.resource_id}")


@admin_router.get("/resources/{rid}", response_class=HTMLResponse)
async def edit_resource_form(request: Request, rid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    resource = await svc.get_resource(rid)
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found.")
    return _render(request, "resources/form.html", {
        "resource": resource,
        "error": None,
    })


@admin_router.post("/resources/{rid}")
async def update_resource(
    request: Request,
    rid: uuid.UUID,
    name: Annotated[str, Form()],
    kind: Annotated[str, Form()],
    provider: Annotated[str, Form()],
    config_json: Annotated[str, Form()] = "{}",
    enabled: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    config = _parse_json_field(config_json, {})
    await svc.update_resource(
        rid,
        name=name.strip(),
        kind=kind.strip(),
        provider=provider.strip(),
        config=config,
        enabled=(enabled == "on"),
    )
    return _redirect(f"/admin/resources/{rid}")


@admin_router.post("/resources/{rid}/toggle", response_class=HTMLResponse)
async def toggle_resource(request: Request, rid: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    resource = await svc.toggle_resource(rid)
    return _render(request, "resources/_row.html", {
        "res": resource,
    })


# ---------------------------------------------------------------------------
# Access policies
# ---------------------------------------------------------------------------

@admin_router.get("/policies", response_class=HTMLResponse)
async def list_policies(request: Request) -> HTMLResponse:
    svc = _svc(request)
    policies = await svc.list_policies()
    return _render(request, "policies/list.html", {
        "policies": policies,
    })


@admin_router.get("/policies/new", response_class=HTMLResponse)
async def new_policy_form(request: Request) -> HTMLResponse:
    return _render(request, "policies/form.html", {
        "policy": None,
        "error": None,
    })


@admin_router.post("/policies/new")
async def create_policy(
    request: Request,
    subject_type: Annotated[str, Form()],
    subject: Annotated[str, Form()],
    effect: Annotated[str, Form()] = "allow",
    conditions_json: Annotated[str, Form()] = "{}",
    priority: Annotated[int, Form()] = 0,
    enabled: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    _validate_policy_fields(subject_type.strip(), effect.strip())
    rule = await svc.create_policy(
        subject_type=subject_type.strip(),
        subject=subject.strip(),
        effect=effect.strip(),
        conditions=_parse_conditions(conditions_json),
        priority=priority,
        enabled=(enabled == "on"),
    )
    return _redirect(f"/admin/policies/{rule.rule_id}")


@admin_router.get("/policies/{rule_id}", response_class=HTMLResponse)
async def edit_policy_form(request: Request, rule_id: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    policy = await svc.get_policy(rule_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found.")
    return _render(request, "policies/form.html", {
        "policy": policy,
        "error": None,
    })


@admin_router.post("/policies/{rule_id}")
async def update_policy(
    request: Request,
    rule_id: uuid.UUID,
    subject_type: Annotated[str, Form()],
    subject: Annotated[str, Form()],
    effect: Annotated[str, Form()] = "allow",
    conditions_json: Annotated[str, Form()] = "{}",
    priority: Annotated[int, Form()] = 0,
    enabled: Annotated[str, Form()] = "",
) -> RedirectResponse:
    svc = _svc(request)
    policy = await svc.get_policy(rule_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found.")
    _validate_policy_fields(subject_type.strip(), effect.strip())
    await svc.update_policy(
        rule_id,
        subject_type=subject_type.strip(),
        subject=subject.strip(),
        effect=effect.strip(),
        conditions=_parse_conditions(conditions_json),
        priority=priority,
        enabled=(enabled == "on"),
    )
    return _redirect(f"/admin/policies/{rule_id}")


@admin_router.post("/policies/{rule_id}/toggle", response_class=HTMLResponse)
async def toggle_policy(request: Request, rule_id: uuid.UUID) -> HTMLResponse:
    svc = _svc(request)
    policy = await svc.toggle_policy(rule_id)
    return _render(request, "policies/_row.html", {
        "policy": policy,
    })


@admin_router.post("/policies/{rule_id}/delete")
async def delete_policy(request: Request, rule_id: uuid.UUID) -> RedirectResponse:
    svc = _svc(request)
    policy = await svc.get_policy(rule_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found.")
    await svc.delete_policy(rule_id)
    return _redirect("/admin/policies")
