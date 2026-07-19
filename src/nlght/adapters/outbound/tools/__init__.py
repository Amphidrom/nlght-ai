# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tool adapter — catalog builder, loader, and built-in tool registry.

The ``tool_registry`` singleton is the extension point for registering custom tools:

    from nlght.adapters.outbound.tools import tool_registry
    from my_app.tools import MyTool

    tool_registry.register(MyTool)
"""

from nlght.adapters.outbound.tools.catalog import AdapterToolCatalog, ToolCatalogBuilder
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.adapters.outbound.tools.registry import tool_registry

__all__ = [
    "AdapterToolCatalog",
    "ToolCatalogBuilder",
    "ToolLoader",
    "tool_registry",
]
