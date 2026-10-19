# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify stable tool ordering in playbook definitions."""

from nlght.core.playbooks.playbook import PlaybookDefinition


def test_active_tools_preserves_declaration_order_without_duplicates() -> None:
    """Preserve declaration order and remove duplicates from exposed tools."""
    definition = PlaybookDefinition(
        name="review",
        description_hint="Review evidence",
        requires=[["lookup", "read"], ["read", "search"]],
        phases=[],
        optional=["search", "write", "lookup"],
    )

    assert definition.active_tools({"search", "lookup", "read"}) == [
        "lookup", "read", "search",
    ]
