# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from nlght.adapters.inbound.http.admin.services import AdminService
from nlght.adapters.outbound.persistence.models import Base


async def _service() -> tuple[AdminService, object]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # SQLite cannot express the PostgreSQL-only partial unique index and
        # creates it as a plain unique index, which blocks draft version forks.
        await conn.execute(text("DROP INDEX IF EXISTS workflow_versions_one_active"))
    return AdminService(engine), engine


async def test_admin_service_manages_workflow_versions_steps_and_graph() -> None:
    svc, engine = await _service()
    try:
        workflow = await svc.create_workflow(
            name="review",
            description=" Review flow ",
            enabled=True,
            capabilities=["web"],
        )

        assert workflow.name == "review"
        assert workflow.description == " Review flow "
        assert workflow.enabled is True
        assert workflow.capabilities == ["web"]

        listed = await svc.list_workflows()
        assert [item.workflow_id for item in listed] == [workflow.workflow_id]

        loaded = await svc.get_workflow(workflow.workflow_id)
        assert loaded is not None
        assert len(loaded.versions) == 1
        original_version = loaded.versions[0]
        assert original_version.version == 1
        assert original_version.status == "draft"

        step = await svc.create_step(
            vid=original_version.workflow_version_id,
            position=1,
            name="select",
            type_="phase",
            enabled=True,
            config={"phase": "select"},
            is_start=True,
            is_terminal=False,
            is_resume=False,
        )
        updated = await svc.update_step(
            step.workflow_step_id,
            name="fetch",
            type="phase",
            position=2,
            enabled=False,
            config={"phase": "fetch"},
            is_start=False,
            is_terminal=True,
            is_resume=True,
        )
        assert updated.name == "fetch"
        assert updated.config == {"phase": "fetch"}
        assert updated.is_terminal is True

        await svc.save_graph(original_version.workflow_version_id, {str(step.workflow_step_id): {"ok": "done"}})
        graph_step = await svc.get_step(step.workflow_step_id)
        assert graph_step is not None
        assert graph_step.transitions == {"ok": "done"}

        forked = await svc.fork_latest_version(workflow.workflow_id)
        forked_loaded = await svc.get_version(forked.workflow_version_id)
        assert forked_loaded is not None
        assert forked_loaded.version == 2
        assert len(forked_loaded.steps) == 1
        assert forked_loaded.steps[0].transitions == {"ok": "done"}

        await svc.activate_version(forked.workflow_version_id)
        active = await svc.get_version(forked.workflow_version_id)
        assert active is not None
        assert active.status == "active"

        toggled = await svc.toggle_workflow(workflow.workflow_id)
        assert toggled.enabled is False

        await svc.delete_step(step.workflow_step_id)
        assert await svc.get_step(step.workflow_step_id) is None
    finally:
        await engine.dispose()


async def test_admin_service_creates_first_fork_and_manages_resources() -> None:
    svc, engine = await _service()
    try:
        wid = __import__("uuid").uuid4()
        first = await svc.fork_latest_version(wid)
        assert first.workflow_id == wid
        assert first.version == 1

        resource = await svc.create_resource(
            name="openai",
            kind="model_provider",
            provider="openai",
            config={"model": "gpt"},
            enabled=True,
        )
        assert resource.enabled is True

        listed = await svc.list_resources()
        assert [item.resource_id for item in listed] == [resource.resource_id]
        assert await svc.get_resource(resource.resource_id) is not None

        updated = await svc.update_resource(
            resource.resource_id,
            name="anthropic",
            kind="model_provider",
            provider="anthropic",
            config={"model": "claude"},
            enabled=False,
        )
        assert updated.provider == "anthropic"
        assert updated.config == {"model": "claude"}

        toggled = await svc.toggle_resource(resource.resource_id)
        assert toggled.enabled is True
    finally:
        await engine.dispose()


async def test_admin_service_access_policy_crud() -> None:
    svc, engine = await _service()
    try:
        rule = await svc.create_policy(
            subject_type="tool",
            subject="web_*",
            effect="deny",
            conditions={"model": ["llama3*"]},
            priority=5,
            enabled=True,
        )
        assert rule.effect == "deny"
        assert rule.conditions == {"model": ["llama3*"]}

        listed = await svc.list_policies()
        assert [p.rule_id for p in listed] == [rule.rule_id]
        assert await svc.get_policy(rule.rule_id) is not None

        updated = await svc.update_policy(
            rule.rule_id,
            subject_type="model",
            subject="*",
            effect="allow",
            conditions={},
            priority=0,
            enabled=True,
        )
        assert updated.subject_type == "model"
        assert updated.conditions == {}

        toggled = await svc.toggle_policy(rule.rule_id)
        assert toggled.enabled is False

        await svc.delete_policy(rule.rule_id)
        assert await svc.get_policy(rule.rule_id) is None
        assert await svc.list_policies() == []
    finally:
        await engine.dispose()
