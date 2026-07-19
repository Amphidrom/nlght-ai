# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Model client adapters — one per supported provider."""

from nlght.adapters.outbound.model.anthropic import AnthropicModelClient
from nlght.adapters.outbound.model.google import GoogleModelClient
from nlght.adapters.outbound.model.ollama_cloud import OllamaCloudClient
from nlght.adapters.outbound.model.ollama_local import OllamaClient, _has_tool_protocol_messages
from nlght.adapters.outbound.model.openai_cloud import OpenAICloudModelClient

__all__ = [
    "AnthropicModelClient",
    "GoogleModelClient",
    "OllamaClient",
    "OllamaCloudClient",
    "OpenAICloudModelClient",
    "_has_tool_protocol_messages"
]
