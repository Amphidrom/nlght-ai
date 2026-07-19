# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Integration: access-policy rules flow from the DB through the rule engine
into tool/playbook catalogs and the model policy — the full SQLAlchemy stack
on a real (in-memory) database.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from integration._helpers import seed_policy, seed_resource
from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.adapters.outbound.persistence.access_rule_repository import SqlAlchemyAccessRuleRepository
from nlght.adapters.outbound.playbooks.catalog import PlaybookCatalogBuilder
from nlght.adapters.outbound.tools.catalog import ToolCatalogBuilder
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.playbooks.playbook import PlaybookDefinition
from nlght.core.tools.tool import ToolBase, ToolSignature

pytestmark = pytest.mark.integration


class _EchoTool(ToolBase):
    KIND = "echo"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [ToolSignature(name="echo", description="Echo.", method_name="run")]

    async def run(self) -> str:
        return "echo"


def _caller(headers: dict[str, str] | None = None) -> RequestContext:
    return RequestContext(
        correlation_id="cid-int",
        request_id="rid-int",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers=headers or {},
        query_params={},
        client_host="127.0.0.1",
    )


def _engine_for(sqlite_engine) -> AccessRuleEngine:
    return AccessRuleEngine(
        SqlAlchemyAccessRuleRepository(sqlite_engine),
        cache_ttl_seconds=0.0,  # no caching across seeding steps in tests
    )


def _tool_builder(sqlite_engine, resource_repo, engine: AccessRuleEngine) -> ToolCatalogBuilder:
    loader = ToolLoader()
    loader.register(_EchoTool)
    return ToolCatalogBuilder(
        loader=loader,
        resource_repository=resource_repo,
        access_policy=engine.for_tools(),
    )


async def test_tool_catalog_unrestricted_without_rules(sqlite_engine, session_factory, resource_repo) -> None:
    await seed_resource(session_factory, kind="echo", name="echo-tool")
    builder = _tool_builder(sqlite_engine, resource_repo, _engine_for(sqlite_engine))

    catalog = await builder.build(model="llama3", caller=_caller())

    assert "echo" in catalog.names()


async def test_tool_catalog_filters_by_model_rule(sqlite_engine, session_factory, resource_repo) -> None:
    await seed_resource(session_factory, kind="echo", name="echo-tool")
    await seed_policy(
        session_factory, "tool", "echo-tool",
        conditions={"model": ["llama3"]},
    )
    builder = _tool_builder(sqlite_engine, resource_repo, _engine_for(sqlite_engine))

    allowed = await builder.build(model="llama3", caller=_caller())
    denied = await builder.build(model="gpt-4", caller=_caller())

    assert "echo" in allowed.names()
    assert denied.names() == []


async def test_tool_catalog_deny_rule_wins_over_allow(sqlite_engine, session_factory, resource_repo) -> None:
    await seed_resource(session_factory, kind="echo", name="echo-tool")
    await seed_policy(session_factory, "tool", "*", effect="allow")
    await seed_policy(
        session_factory, "tool", "echo-*", effect="deny", priority=10,
        conditions={"header:x-org": ["blocked"]},
    )
    builder = _tool_builder(sqlite_engine, resource_repo, _engine_for(sqlite_engine))

    blocked = await builder.build(model="llama3", caller=_caller({"X-Org": "blocked"}))
    allowed = await builder.build(model="llama3", caller=_caller({"X-Org": "fine"}))

    assert blocked.names() == []
    assert "echo" in allowed.names()


async def test_disabled_rule_is_ignored(sqlite_engine, session_factory, resource_repo) -> None:
    await seed_resource(session_factory, kind="echo", name="echo-tool")
    await seed_policy(
        session_factory, "tool", "echo-tool", effect="deny", enabled=False,
    )
    builder = _tool_builder(sqlite_engine, resource_repo, _engine_for(sqlite_engine))

    catalog = await builder.build(model="llama3", caller=_caller())

    assert "echo" in catalog.names()


async def test_playbook_catalog_respects_rules(sqlite_engine, session_factory) -> None:
    defn = PlaybookDefinition(
        name="research",
        description_hint="hint",
        requires=[],
        optional=[],
        phases=[],
        constraints=[],
        fallback=None,
    )
    await seed_policy(
        session_factory, "playbook", "research",
        conditions={"model": ["llama*"]},
    )
    builder = PlaybookCatalogBuilder(
        {"research": defn},
        access_policy=_engine_for(sqlite_engine).for_playbooks(),
    )

    allowed = await builder.build(tool_catalog=None, model="llama3", caller=_caller())
    denied = await builder.build(tool_catalog=None, model="gpt-4", caller=_caller())

    assert allowed.get("research") is not None
    assert denied.get("research") is None


async def test_model_policy_denies_by_rule(sqlite_engine, session_factory) -> None:
    await seed_policy(session_factory, "model", "gpt-*", effect="deny")
    policy = _engine_for(sqlite_engine).for_models()

    assert await policy.is_allowed("gpt-4", _caller()) is False
    assert await policy.is_allowed("llama3", _caller()) is True
