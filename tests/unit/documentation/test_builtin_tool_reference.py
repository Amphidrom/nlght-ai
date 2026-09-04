# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import re
from pathlib import Path

from nlght.adapters.outbound.tools.registry import tool_registry

REFERENCE = Path(__file__).parents[3] / "docs/documentation/tools/built-in-tools.mdx"
START = "{/* built-in-tool-registry:start */}"
END = "{/* built-in-tool-registry:end */}"
ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \| (.*?) \|$")


def _documented_inventory() -> dict[tuple[str, str], list[str]]:
    text = REFERENCE.read_text(encoding="utf-8")
    assert text.count(START) == 1
    assert text.count(END) == 1
    inventory = text.split(START, 1)[1].split(END, 1)[0]
    documented: dict[tuple[str, str], list[str]] = {}
    for line in inventory.splitlines():
        match = ROW.match(line)
        if match is None:
            continue
        kind, provider, operations_cell = match.groups()
        address = (kind, provider)
        assert address not in documented, f"duplicate built-in tool row: {kind}/{provider}"
        documented[address] = re.findall(r"`([^`]+)`", operations_cell)
    return documented


def test_builtin_tool_inventory_matches_runtime_registry() -> None:
    registered = {
        address: [signature.name for signature in implementation.signatures()]
        for address, implementation in tool_registry._registry.items()
    }

    assert _documented_inventory() == registered
