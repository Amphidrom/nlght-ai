# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Exercise stored resource and tool rules across runtime boundaries.

A resource rule governs activation, a tool rule governs one signature, and
workflow conditions constrain either subject.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.integration._helpers import seed_policy, seed_resource

from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.adapters.outbound.persistence.access_rule_repository import (
    SqlAlchemyAccessRuleRepository,
)
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.adapters.outbound.tools.catalog import ToolCatalogBuilder
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import ResourceAccessDeniedError
from nlght.core.runtime.resource import ResourceDef
from nlght.core.tools.tool import ToolBase, ToolSignature

pytestmark = pytest.mark.integration


class _SearchTool(ToolBase):
    """A resource a model can call into."""

    KIND = "data_store"
    PROVIDER = "default"
    built = 0

    def __init__(self, *, name: str, config: dict[str, Any], **_: object) -> None:
        super().__init__(name=name, config=config)
        type(self).built += 1

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature("data-search", "hybrid", "search", action=READ_REQUEST),
            ToolSignature(
                "data-search-keyword", "keyword", "search", action=READ_REQUEST
            ),
        ]

    async def search(self, **_: object) -> str:
        return "found"


class _WriterTool(ToolBase):
    """A writer: used by steps, never offered to a model."""

    KIND = "data_index_writer"
    PROVIDER = "default"
    built = 0

    def __init__(self, *, name: str, config: dict[str, Any], **_: object) -> None:
        super().__init__(name=name, config=config)
        type(self).built += 1

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return []


class _Resources:
    """The resource repository, over the rows the test seeded."""

    def __init__(self, rows: list[ResourceDef]) -> None:
        self._rows = rows

    async def find_by_kind(self, kind: str) -> list[ResourceDef]:
        return [r for r in self._rows if r.kind == kind and r.enabled]

    async def find_by_id(self, resource_id: uuid.UUID) -> ResourceDef | None:
        return next((r for r in self._rows if r.resource_id == resource_id), None)

    async def list_enabled(self) -> list[ResourceDef]:
        return [r for r in self._rows if r.enabled]


def _caller(workflow: str | None) -> RequestContext:
    """Build a caller with the workflow identity resolved by the executor.

    Args:
        workflow (str | None): Workflow identity resolved for the caller.

    Returns:
        RequestContext: Caller context with the resolved workflow identity.
    """
    return RequestContext(
        correlation_id="cid", request_id="rid",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
        workflow=workflow,
    )


def _row(kind: str, name: str) -> ResourceDef:
    return ResourceDef(
        resource_id=uuid.uuid4(), name=name, kind=kind,
        provider="default", config={}, enabled=True,
    )


@pytest.fixture
def rows() -> list[ResourceDef]:
    return [
        _row("data_store", "data-main"),
        _row("data_store", "shared"),
        _row("data_index_writer", "writer-main"),
    ]


@pytest.fixture
def loader() -> ToolLoader:
    registry = ToolLoader()
    registry.register(_SearchTool)
    registry.register(_WriterTool)
    return registry


@pytest.fixture(autouse=True)
def _reset_counts() -> None:
    _SearchTool.built = 0
    _WriterTool.built = 0


def _engine(sqlite_engine) -> AccessRuleEngine:
    # No cache window: each test writes rules and reads them back immediately.
    return AccessRuleEngine(
        SqlAlchemyAccessRuleRepository(sqlite_engine), cache_ttl_seconds=0.0
    )


def _activator(rows, loader, sqlite_engine) -> ResourceActivator:
    return ResourceActivator(
        resources=_Resources(rows),
        loader=loader,
        access_policy=_engine(sqlite_engine).for_resources(),
    )


def _catalog(rows, loader, sqlite_engine) -> ToolCatalogBuilder:
    engine = _engine(sqlite_engine)
    return ToolCatalogBuilder(
        loader=loader,
        resource_repository=_Resources(rows),
        resource_access_policy=engine.for_resources(),
        tool_access_policy=engine.for_tools(),
    )


async def test_a_workflow_scoped_resource_rule_admits_one_workflow(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """Allow one workflow to activate the governed resource."""
    await seed_policy(
        session_factory, "resource", "data_store/data-main",
        effect="allow", conditions={"workflow": ["answer"]},
    )
    activator = _activator(rows, loader, sqlite_engine)

    instance = await activator.activate(
        kind="data_store", name="data-main", caller=_caller("answer")
    )
    assert isinstance(instance, _SearchTool)

    with pytest.raises(ResourceAccessDeniedError):
        await activator.activate(
            kind="data_store", name="data-main", caller=_caller("ingest")
        )
    assert _SearchTool.built == 1, "the denied activation constructed something"


async def test_a_caller_outside_any_workflow_is_denied_by_a_scoped_rule(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """No bypass for anything running outside a flow."""
    await seed_policy(
        session_factory, "resource", "data_store/data-main",
        effect="allow", conditions={"workflow": ["answer"]},
    )
    activator = _activator(rows, loader, sqlite_engine)

    with pytest.raises(ResourceAccessDeniedError):
        await activator.activate(
            kind="data_store", name="data-main", caller=_caller(None)
        )


async def test_an_unconditional_resource_rule_admits_every_workflow(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """Allow every workflow to activate an unconditionally allowed resource."""
    await seed_policy(session_factory, "resource", "data_store/shared", effect="allow")
    activator = _activator(rows, loader, sqlite_engine)

    for workflow in ("answer", "ingest", None):
        instance = await activator.activate(
            kind="data_store", name="shared", caller=_caller(workflow)
        )
        assert isinstance(instance, _SearchTool)


async def test_a_writer_is_governed_by_the_resource_rule_that_names_it(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """Apply the resource policy to a writer activated by a step."""
    await seed_policy(
        session_factory, "resource", "data_index_writer/writer-main",
        effect="allow", conditions={"workflow": ["ingest"]},
    )
    activator = _activator(rows, loader, sqlite_engine)

    instance = await activator.activate(
        kind="data_index_writer", name="writer-main", caller=_caller("ingest")
    )
    assert isinstance(instance, _WriterTool)

    with pytest.raises(ResourceAccessDeniedError):
        await activator.activate(
            kind="data_index_writer", name="writer-main", caller=_caller("answer")
        )


async def test_a_tool_rule_cannot_reinstate_a_denied_resource(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """resource DENY + tool ALLOW is DENY, and the resource is never built.

    The two levels are ordered, not weighed: a signature of a resource the
    caller may not use is not a thing that exists to be allowed.
    """
    await seed_policy(session_factory, "resource", "data_store/*", effect="deny")
    await seed_policy(
        session_factory, "tool", "data_store/data-main::data-search", effect="allow",
    )
    builder = _catalog(rows, loader, sqlite_engine)

    catalog = await builder.build(model="llama3", caller=_caller("answer"))

    assert catalog.names() == []
    assert _SearchTool.built == 0


async def test_a_tool_rule_narrows_within_a_resource_the_caller_may_use(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """Limit offered tools after granting access to their resource."""
    await seed_policy(
        session_factory, "tool", "data_store/*::data-search-keyword",
        effect="allow", conditions={"workflow": ["answer"]},
    )
    # One data store, so this is about the tool level. With both visible the
    # catalog would correctly refuse the build over `data-search`, and that
    # has its own test.
    only_main = [row for row in rows if row.name != "shared"]
    builder = _catalog(only_main, loader, sqlite_engine)

    catalog = await builder.build(model="llama3", caller=_caller("answer"))

    # `data-search` has no tool rule, so it remains ungoverned and offered.
    # `data-search-keyword` is governed and allowed for this workflow.
    assert sorted(catalog.names()) == ["data-search", "data-search-keyword"]

    other = await builder.build(model="llama3", caller=_caller("ingest"))
    assert other.names() == ["data-search"], (
        "a governed operation must not be offered to a workflow its rule excludes"
    )


async def test_the_catalog_filters_where_a_step_would_fail(
    sqlite_engine, session_factory, rows, loader
) -> None:
    """Filter denied tools from the catalog and reject required step access."""
    await seed_policy(
        session_factory, "resource", "data_store/data-main",
        effect="allow", conditions={"workflow": ["answer"]},
    )
    await seed_policy(session_factory, "resource", "data_store/shared", effect="allow")

    catalog = await _catalog(rows, loader, sqlite_engine).build(
        model="llama3", caller=_caller("ingest")
    )
    # Filtered, not raised: `shared` is allowed and answers, `data-main` is not.
    assert sorted(catalog.names()) == ["data-search", "data-search-keyword"]

    with pytest.raises(ResourceAccessDeniedError):
        await _activator(rows, loader, sqlite_engine).activate(
            kind="data_store", name="data-main", caller=_caller("ingest")
        )


async def test_no_rules_at_all_restricts_nothing(
    sqlite_engine, rows, loader
) -> None:
    """Leave resources unrestricted when the policy table is empty."""
    activator = _activator(rows, loader, sqlite_engine)

    instance = await activator.activate(
        kind="data_store", name="data-main", caller=_caller("anything")
    )

    assert isinstance(instance, _SearchTool)


async def test_seeding_a_resource_row_matches_the_address_a_rule_names(
    session_factory, sqlite_engine
) -> None:
    """Match a resource row against the rule for its kind and name."""
    from nlght.adapters.outbound.persistence.resource_repository import (
        SqlAlchemyResourceRepository,
    )

    await seed_resource(session_factory, kind="data_store", name="data-main")
    repository = SqlAlchemyResourceRepository(sqlite_engine)

    [stored] = await repository.find_by_kind("data_store")

    assert stored.address == "data_store/data-main"
