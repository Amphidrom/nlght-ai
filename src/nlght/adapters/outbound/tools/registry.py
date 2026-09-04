# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from nlght.adapters.outbound.tools.builtin.activate_playbook import ActivatePlaybookTool
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
tool_registry.register(ShellTool)
tool_registry.register(GitTool)
tool_registry.register(HttpProbeTool)
tool_registry.register(WebSearchTool)
tool_registry.register(FetchUrlTool)
tool_registry.register(LoadMemoryArtifactTool)

# Data Store — one activation spanning Qdrant and OpenSearch, plus the corpus
# it answers from. There is no separate lexical or vector store kind: keyword
# and semantic discovery are two capabilities of one store over one corpus, and
# splitting them into a kind each made what the ingestion writes and what
# retrieval reads two configurations nothing kept aligned (ADR-0063).
try:
    from nlght.adapters.outbound.stores.data import DataStoreTool
    tool_registry.register(DataStoreTool)
except ImportError:
    pass

# Data index writer — a tool under a kind disjoint from data_store, declaring
# no signatures, so retrieval can never reach a write operation.
try:
    from nlght.adapters.outbound.stores.data_writer import DataIndexWriterTool
    tool_registry.register(DataIndexWriterTool)
except ImportError:
    pass

# Knowledge index writer — write-only, disjoint kind, no signatures.
try:
    from nlght.adapters.outbound.stores.knowledge_writer import KnowledgeIndexWriterTool
    tool_registry.register(KnowledgeIndexWriterTool)
except ImportError:
    pass

# Knowledge Store — one activation spanning OpenSearch and PostgreSQL.
try:
    from nlght.adapters.outbound.stores.knowledge import KnowledgeStoreTool
    tool_registry.register(KnowledgeStoreTool)
except ImportError:
    pass
