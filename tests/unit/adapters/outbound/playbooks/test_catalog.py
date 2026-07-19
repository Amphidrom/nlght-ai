# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for AdapterPlaybookCatalog and PlaybookCatalogBuilder."""
from __future__ import annotations

from nlght.adapters.outbound.playbooks.catalog import AdapterPlaybookCatalog, PlaybookCatalogBuilder
from nlght.core.playbooks.playbook import ActivePlaybook, Phase, PlaybookDefinition


def _active_playbook(name: str = "test_playbook") -> ActivePlaybook:
    return ActivePlaybook(name=name, description="# Playbook: test", tool_names=["shell"])


def _definition(
    name: str = "my_playbook",
    requires: list[list[str]] | None = None,
) -> PlaybookDefinition:
    return PlaybookDefinition(
        name=name,
        description_hint="A test playbook",
        requires=requires or [],
        optional=[],
        phases=[],
        constraints=[],
        fallback=None,
    )


# ---------------------------------------------------------------------------
# AdapterPlaybookCatalog
# ---------------------------------------------------------------------------

def test_catalog_get_existing() -> None:
    playbook   = _active_playbook("s1")
    catalog = AdapterPlaybookCatalog({"s1": playbook})
    assert catalog.get("s1") is playbook


def test_catalog_get_missing_returns_none() -> None:
    catalog = AdapterPlaybookCatalog({})
    assert catalog.get("missing") is None


def test_catalog_all() -> None:
    s1 = _active_playbook("a")
    s2 = _active_playbook("b")
    catalog = AdapterPlaybookCatalog({"a": s1, "b": s2})
    assert set(catalog.names()) == {"a", "b"}
    assert len(catalog.all()) == 2


def test_catalog_len() -> None:
    catalog = AdapterPlaybookCatalog({"a": _active_playbook("a")})
    assert len(catalog) == 1


def test_catalog_bool_true() -> None:
    assert bool(AdapterPlaybookCatalog({"a": _active_playbook()})) is True


def test_catalog_bool_false() -> None:
    assert bool(AdapterPlaybookCatalog({})) is False


def test_catalog_to_markdown_empty() -> None:
    catalog = AdapterPlaybookCatalog({})
    md = catalog.to_markdown()
    assert "No playbooks" in md


def test_catalog_to_markdown_with_playbook() -> None:
    playbook   = ActivePlaybook(name="s1", description="# Playbook: s1\ndesc", tool_names=[])
    catalog = AdapterPlaybookCatalog({"s1": playbook})
    md = catalog.to_markdown()
    assert "s1" in md


def test_catalog_playbook_markdown_existing() -> None:
    playbook   = ActivePlaybook(name="s1", description="# my description", tool_names=[])
    catalog = AdapterPlaybookCatalog({"s1": playbook})
    assert catalog.playbook_markdown("s1") == "# my description"


def test_catalog_playbook_markdown_missing() -> None:
    catalog = AdapterPlaybookCatalog({})
    assert catalog.playbook_markdown("gone") == ""


# ---------------------------------------------------------------------------
# PlaybookCatalogBuilder
# ---------------------------------------------------------------------------

class _FakeToolCatalog:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return self._names

    def get_or_none(self, name: str):
        return None


async def test_builder_includes_available_playbook() -> None:
    defn    = _definition("playbook_a")
    builder = PlaybookCatalogBuilder({"playbook_a": defn})
    catalog = await builder.build(tool_catalog=None)
    assert catalog.get("playbook_a") is not None


async def test_builder_excludes_playbook_with_missing_tools() -> None:
    defn    = _definition("playbook_b", requires=[["missing_tool"]])
    builder = PlaybookCatalogBuilder({"playbook_b": defn})
    catalog = await builder.build(tool_catalog=_FakeToolCatalog([]))
    assert catalog.get("playbook_b") is None


async def test_builder_includes_playbook_when_tools_available() -> None:
    defn    = _definition("playbook_c", requires=[["shell"]])
    builder = PlaybookCatalogBuilder({"playbook_c": defn})
    catalog = await builder.build(tool_catalog=_FakeToolCatalog(["shell"]))
    assert catalog.get("playbook_c") is not None


async def test_builder_access_policy_receives_model() -> None:
    from unittest.mock import AsyncMock

    policy = AsyncMock()
    policy.is_allowed = AsyncMock(return_value=True)

    defn    = _definition("playbook_d")
    builder = PlaybookCatalogBuilder({"playbook_d": defn}, access_policy=policy)
    await builder.build(tool_catalog=None, model="llama3")

    policy.is_allowed.assert_awaited_once_with("playbook_d", None, "llama3")


async def test_builder_access_policy_consulted_without_caller() -> None:
    from unittest.mock import AsyncMock

    policy = AsyncMock()
    policy.is_allowed = AsyncMock(return_value=False)

    defn    = _definition("playbook_e")
    builder = PlaybookCatalogBuilder({"playbook_e": defn}, access_policy=policy)
    catalog = await builder.build(tool_catalog=None)

    assert catalog.get("playbook_e") is None


async def test_builder_access_policy_deny() -> None:
    from unittest.mock import AsyncMock, MagicMock

    policy = AsyncMock()
    policy.is_allowed = AsyncMock(return_value=False)

    caller = MagicMock()
    caller.correlation_id = "cid"

    defn    = _definition("playbook_g")
    builder = PlaybookCatalogBuilder({"playbook_g": defn}, access_policy=policy)
    catalog = await builder.build(tool_catalog=None, caller=caller)
    assert catalog.get("playbook_g") is None


async def test_builder_access_policy_error_denies() -> None:
    from unittest.mock import AsyncMock, MagicMock

    policy = AsyncMock()
    policy.is_allowed = AsyncMock(side_effect=RuntimeError("boom"))

    caller = MagicMock()
    caller.correlation_id = "cid"

    defn    = _definition("playbook_h")
    builder = PlaybookCatalogBuilder({"playbook_h": defn}, access_policy=policy)
    catalog = await builder.build(tool_catalog=None, caller=caller)
    assert catalog.get("playbook_h") is None


# ---------------------------------------------------------------------------
# Markdown with phases / constraints / fallback
# ---------------------------------------------------------------------------

async def test_builder_playbook_with_phases() -> None:
    phases = [Phase(name="p1", goal="do x", tools=["shell"], role="executor",
                    guidance="be careful", exit_condition="done")]
    defn = PlaybookDefinition(
        name="phased",
        description_hint="hint",
        requires=[],
        optional=[],
        phases=phases,
        constraints=["no hallucinations"],
        fallback="ask user",
    )
    builder = PlaybookCatalogBuilder({"phased": defn})
    catalog = await builder.build(tool_catalog=None)
    md = catalog.playbook_markdown("phased")
    assert "p1" in md
    assert "no hallucinations" in md
    assert "ask user" in md
