# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for DirectiveManagementTool."""
from __future__ import annotations

import json

from nlght.adapters.outbound.hive_mind.coordinator import HiveMindStoreCoordinator
from nlght.adapters.outbound.tools.builtin.directive_management import DirectiveManagementTool


def _tool(store=None) -> DirectiveManagementTool:
    return DirectiveManagementTool(
        name="directive_management",
        config={},
        store_coordinator=store,
    )


def test_signatures_returns_directive_set() -> None:
    sigs = DirectiveManagementTool.signatures()
    assert any(s.name == "directive_set" for s in sigs)


async def test_set_directive_no_store_returns_error() -> None:
    tool   = _tool(store=None)
    result = await tool.set_directive(key="tone", value="formal")
    data   = json.loads(result)
    assert data["success"] is False
    assert "No session store" in data["error"]


async def test_set_directive_unknown_key_returns_error() -> None:
    store  = HiveMindStoreCoordinator()
    tool   = _tool(store=store)
    result = await tool.set_directive(key="unknown_key_xyz", value="val")
    data   = json.loads(result)
    assert data["success"] is False
    assert "Unknown directive key" in data["error"]
    assert "valid_keys" in data


async def test_set_directive_known_key_succeeds() -> None:
    store  = HiveMindStoreCoordinator()
    tool   = _tool(store=store)
    result = await tool.set_directive(key="tone", value="formal")
    data   = json.loads(result)
    assert data["success"] is True
    assert data["key"] == "tone"
    assert data["value"] == "formal"
    assert "formal" in data["prompt"]


async def test_set_directive_updates_store() -> None:
    store  = HiveMindStoreCoordinator()
    tool   = _tool(store=store)
    await tool.set_directive(key="conversation_language", value="fr")
    assert store.get_directives().get("conversation_language") == "fr"
