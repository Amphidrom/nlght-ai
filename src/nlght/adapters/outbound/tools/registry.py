# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from nlght.adapters.outbound.tools.builtin.activate_playbook import ActivatePlaybookTool
from nlght.adapters.outbound.tools.builtin.directive_management import DirectiveManagementTool
from nlght.adapters.outbound.tools.builtin.fetch_url import FetchUrlTool
from nlght.adapters.outbound.tools.builtin.git import GitTool
from nlght.adapters.outbound.tools.builtin.http_probe import HttpProbeTool
from nlght.adapters.outbound.tools.builtin.memory_artifact import LoadMemoryArtifactTool
from nlght.adapters.outbound.tools.builtin.shell import ShellTool
from nlght.adapters.outbound.tools.builtin.web_search import WebSearchTool
from nlght.adapters.outbound.tools.loader import ToolLoader

# Global registry — extension point for users.
#
# Register before server start:
#
#   from nlght.adapters.outbound.tools.registry import tool_registry
#   from my_app.tools import MyCustomTool
#
#   tool_registry.register(MyCustomTool)
tool_registry: ToolLoader = ToolLoader()

# Built-in Tools
tool_registry.register(ActivatePlaybookTool)
tool_registry.register(DirectiveManagementTool)
tool_registry.register(ShellTool)
tool_registry.register(GitTool)
tool_registry.register(HttpProbeTool)
tool_registry.register(WebSearchTool)
tool_registry.register(FetchUrlTool)
tool_registry.register(LoadMemoryArtifactTool)

# Store Tools (optional — only registered when deps are installed)
try:
    from nlght.adapters.outbound.stores.lexical import LexicalStoreTool
    from nlght.adapters.outbound.stores.vector import VectorStoreTool
    tool_registry.register(LexicalStoreTool)
    tool_registry.register(VectorStoreTool)
except ImportError:
    pass
