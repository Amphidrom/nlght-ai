# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProtocolAdapterRegistry:
    items: list[Any] = field(default_factory=list)

    def all(self) -> list[Any]:
        return self.items

    def enabled(self) -> list[Any]:
        return [item for item in self.items if getattr(item, "enabled", False)]


@dataclass(slots=True)
class GatewayRegistry:
    """Registry of configured gateways (HTTP, gRPC, …)."""
    items: list[Any] = field(default_factory=list)

    def all(self) -> list[Any]:
        return self.items

    def enabled(self) -> list[Any]:
        return [item for item in self.items if getattr(item, "enabled", False)]


@dataclass(slots=True)
class ModelProviderRegistry:
    items: list[Any] = field(default_factory=list)

    def all(self) -> list[Any]:
        return self.items


@dataclass(slots=True)
class HiveMindProviderRegistry:
    items: list[Any] = field(default_factory=list)

    def all(self) -> list[Any]:
        return self.items

    def enabled(self) -> list[Any]:
        return [item for item in self.items if getattr(item, "enabled", False)]


@dataclass(slots=True)
class ProtocolDetectorRuntime:
    name: str
    kind: str
    enabled: bool
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HttpGatewaySubsystemRuntime:
    name: str
    host: str
    port: int


@dataclass(slots=True)
class PersistenceSubsystemRuntime:
    """Resolved persistence configuration — built from ``integrations.persistence.workflows``."""
    name: str
    backend: str
    url: str
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    hive_mind_provider_name: str | None = None


@dataclass(slots=True)
class LocalOsRuntimeSubsystemRuntime:
    name: str
    workdir: str = "."
    workspace_path: str = "./workspaces"


@dataclass(slots=True)
class DockerOsRuntimeSubsystemRuntime:
    name: str
    base_image: str
    workspace_path: str | None = None
    extra_hosts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenericRuntimeSubsystemRuntime:
    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OllamaModelProviderRuntime:
    """OpenAI-compatible Ollama provider (cloud / remote)."""
    name: str
    base_url: str
    default_model: str
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    request_timeout_s: float | None = None
    stream_connect_timeout_s: float | None = None
    stream_read_timeout_s: float | None = None


@dataclass(slots=True)
class OllamaLocalModelProviderRuntime:
    """Native local Ollama provider — tool calls via /api/chat NDJSON."""
    name: str
    base_url: str
    default_model: str
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    request_timeout_s: float | None = None
    stream_connect_timeout_s: float | None = None
    stream_read_timeout_s: float | None = None


@dataclass(slots=True)
class AnthropicModelProviderRuntime:
    name: str
    api_key: str  # empty string → SDK reads ANTHROPIC_API_KEY env var
    default_model: str


@dataclass(slots=True)
class OpenAICloudModelProviderRuntime:
    name: str
    api_key: str  # empty string → SDK reads OPENAI_API_KEY env var
    default_model: str
    base_url: str  # empty string → use default OpenAI API endpoint


@dataclass(slots=True)
class GoogleModelProviderRuntime:
    name: str
    api_key: str  # empty string → SDK reads GOOGLE_API_KEY / GEMINI_API_KEY env var
    default_model: str


@dataclass(slots=True)
class GenericModelProviderRuntime:
    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LicensingRuntime:
    license_key: str | None = None


@dataclass(slots=True)
class HiveMindProviderRuntime:
    name: str
    kind: str
    enabled: bool
    session_ttl_seconds: int = 3600
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeContext:
    protocol_adapter_registry: ProtocolAdapterRegistry
    gateway_registry: GatewayRegistry
    model_provider_registry: ModelProviderRegistry
    hive_mind_provider_registry: HiveMindProviderRegistry
    protocol_detector_runtimes: list[ProtocolDetectorRuntime]
    gateways: list[Any]
    os_runtime: LocalOsRuntimeSubsystemRuntime | DockerOsRuntimeSubsystemRuntime | None
    persistence: PersistenceSubsystemRuntime | None
    model_providers: list[Any]
    hive_mind_provider: HiveMindProviderRuntime | None
    licensing: LicensingRuntime = field(default_factory=LicensingRuntime)
