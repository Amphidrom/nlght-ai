# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for BoundToolContract, AdapterToolCatalog, and ToolCatalogBuilder."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nlght.adapters.outbound.tools.catalog import AdapterToolCatalog, ToolCatalogBuilder
from nlght.adapters.outbound.tools.contract import BoundToolContract
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.runtime.resource import ResourceDef
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GreetTool(ToolBase):
    KIND = "greeter"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="greet",
                description="Grüßt jemanden.",
                method_name="greet",
                parameters=[ToolParameter(name="name", type="string")],
            )
        ]

    async def greet(self, *, name: str) -> str:
        return f"Hallo, {name}!"


class _SyncTool(ToolBase):
    KIND = "sync_tool"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="sync_op",
                description="Sync operation.",
                method_name="sync_op",
            )
        ]

    def sync_op(self) -> str:
        return "sync_result"


class _WorkspaceAwareTool(ToolBase):
    KIND = "workspace_aware"

    def __init__(self, *, name: str, config: dict, workspace: object | None = None, **kwargs: object) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.workspace = workspace

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="workspace_path",
                description="Returns the injected workspace path.",
                method_name="workspace_path",
            )
        ]

    async def workspace_path(self) -> str:
        root_path = getattr(self.workspace, "root_path", None)
        return str(root_path) if root_path is not None else ""


def _resource(kind: str, name: str = "test-tool") -> ResourceDef:
    return ResourceDef(
        resource_id=uuid.uuid4(),
        name=name,
        kind=kind,
        provider="runtime",
        config={},
        enabled=True,
    )


def _make_repo(resources: list[ResourceDef]) -> AsyncMock:
    repo = AsyncMock()
    repo.list_enabled = AsyncMock(return_value=resources)
    return repo


# ---------------------------------------------------------------------------
# BoundToolContract
# ---------------------------------------------------------------------------


async def test_bound_tool_contract_execute_async_method() -> None:
    tool = _GreetTool(name="greeter", config={})
    contract = BoundToolContract(
        name="greet",
        description="Grüßt jemanden.",
        parameters=[],
        instance=tool,
        method_name="greet",
    )

    result = await contract.execute(name="Welt")

    assert result == "Hallo, Welt!"


async def test_bound_tool_contract_execute_sync_method() -> None:
    tool = _SyncTool(name="sync", config={})
    contract = BoundToolContract(
        name="sync_op",
        description="Sync.",
        parameters=[],
        instance=tool,
        method_name="sync_op",
    )

    result = await contract.execute()

    assert result == "sync_result"


def test_bound_tool_contract_properties() -> None:
    params = [ToolParameter(name="name", type="string")]
    tool = _GreetTool(name="greeter", config={})
    contract = BoundToolContract(
        name="greet",
        description="Grüßt jemanden.",
        parameters=params,
        instance=tool,
        method_name="greet",
    )

    assert contract.name == "greet"
    assert contract.description == "Grüßt jemanden."
    assert contract.parameters == params


# ---------------------------------------------------------------------------
# AdapterToolCatalog
# ---------------------------------------------------------------------------


def _catalog_with_greet() -> AdapterToolCatalog:
    tool = _GreetTool(name="greeter", config={})
    contract = BoundToolContract(
        name="greet",
        description="Grüßt.",
        parameters=[],
        instance=tool,
        method_name="greet",
    )
    return AdapterToolCatalog({"greet": contract})


def test_catalog_get_returns_contract() -> None:
    catalog = _catalog_with_greet()

    contract = catalog.get("greet")

    assert contract.name == "greet"


def test_catalog_get_raises_for_unknown_name() -> None:
    catalog = _catalog_with_greet()

    with pytest.raises(KeyError, match="greet_unknown"):
        catalog.get("greet_unknown")


def test_catalog_get_or_none_returns_none_for_unknown() -> None:
    catalog = _catalog_with_greet()

    assert catalog.get_or_none("missing") is None


def test_catalog_names_returns_all_names() -> None:
    catalog = _catalog_with_greet()

    assert catalog.names() == ["greet"]


def test_catalog_all_returns_all_contracts() -> None:
    catalog = _catalog_with_greet()

    assert len(catalog.all()) == 1


def test_catalog_bool_true_when_not_empty() -> None:
    assert bool(_catalog_with_greet()) is True


def test_catalog_bool_false_when_empty() -> None:
    assert bool(AdapterToolCatalog({})) is False


def test_catalog_len() -> None:
    assert len(_catalog_with_greet()) == 1


# ---------------------------------------------------------------------------
# ToolCatalogBuilder
# ---------------------------------------------------------------------------


async def test_builder_creates_contract_per_signature() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build()

    assert "greet" in catalog.names()


async def test_builder_skips_unknown_kinds() -> None:
    loader = ToolLoader()  # nothing registered
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build()

    assert catalog.names() == []


async def test_builder_empty_when_no_resources() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build()

    assert len(catalog) == 0


async def test_builder_builds_multiple_tools() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    loader.register(_SyncTool)
    repo = _make_repo([_resource("greeter", "g1"), _resource("sync_tool", "s1")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build()

    assert set(catalog.names()) == {"greet", "sync_op"}


async def test_builder_contract_is_executable() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build()
    result = await catalog.get("greet").execute(name="Test")

    assert result == "Hallo, Test!"


async def test_builder_injects_workspace_runtime_dependency() -> None:
    loader = ToolLoader()
    loader.register(_WorkspaceAwareTool)
    repo = _make_repo([_resource("workspace_aware")])
    workspace = type("Workspace", (), {"root_path": Path("workspaces/session-1")})()
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    catalog = await builder.build(workspace=workspace)
    result = await catalog.get("workspace_path").execute()

    assert result == str(Path("workspaces/session-1"))


async def test_builder_skips_failing_instantiation_gracefully() -> None:
    class _BrokenTool(ToolBase):
        KIND = "broken"

        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("cannot instantiate")

        @classmethod
        def signatures(cls) -> list[ToolSignature]:
            return []

    loader = ToolLoader()
    loader.register(_BrokenTool)
    repo = _make_repo([_resource("broken")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo)

    # must not raise — broken tools are logged and skipped
    catalog = await builder.build()

    assert len(catalog) == 0


# ---------------------------------------------------------------------------
# ToolAccessPolicy — denial paths
# ---------------------------------------------------------------------------


def _caller(cid: str = "cid-1") -> RequestContext:
    return RequestContext(
        correlation_id=cid,
        request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
    )


class _AllowAllPolicy:
    async def is_allowed(self, resource: ResourceDef, caller: RequestContext | None, model: str | None) -> bool:
        return True


class _DenyAllPolicy:
    async def is_allowed(self, resource: ResourceDef, caller: RequestContext | None, model: str | None) -> bool:
        return False


class _RaisingPolicy:
    async def is_allowed(self, resource: ResourceDef, caller: RequestContext | None, model: str | None) -> bool:
        raise RuntimeError("policy backend unreachable")


async def test_builder_access_policy_allows_when_permitted() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo, access_policy=_AllowAllPolicy())

    catalog = await builder.build(caller=_caller())

    assert "greet" in catalog.names()


async def test_builder_access_policy_denies_and_excludes_tool() -> None:
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo, access_policy=_DenyAllPolicy())

    catalog = await builder.build(caller=_caller())

    assert catalog.names() == []


async def test_builder_access_policy_error_fails_closed() -> None:
    # A policy that raises must deny the tool, not let it through.
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo, access_policy=_RaisingPolicy())

    catalog = await builder.build(caller=_caller())

    assert catalog.names() == []


async def test_builder_access_policy_consulted_without_caller() -> None:
    # The policy also governs internal invocations without a request context —
    # a deny must hold even when no caller is supplied.
    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo, access_policy=_DenyAllPolicy())

    catalog = await builder.build()  # no caller=

    assert catalog.names() == []


async def test_builder_access_policy_receives_model() -> None:
    seen: list[str | None] = []

    class _RecordingPolicy:
        async def is_allowed(self, resource: ResourceDef, caller: RequestContext | None, model: str | None) -> bool:
            seen.append(model)
            return True

    loader = ToolLoader()
    loader.register(_GreetTool)
    repo = _make_repo([_resource("greeter")])
    builder = ToolCatalogBuilder(loader=loader, resource_repository=repo, access_policy=_RecordingPolicy())

    await builder.build(model="llama3", caller=_caller())

    assert seen == ["llama3"]
