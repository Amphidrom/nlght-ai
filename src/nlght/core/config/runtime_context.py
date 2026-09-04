# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nlght.core.config.snapshot import WatcherConfig


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
    network_mode: str = "bridge"
    allow_runtime_env: bool = True


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


@dataclass(slots=True, frozen=True)
class ExecutionStreamRuntime:
    """Resolved transport for live execution signals — see ``ExecutionStreamConfig``."""
    transport: str = "in_process"
    url: str = ""
    channel: str = "nlght_execution_stream"


@dataclass(slots=True, frozen=True)
class ExecutionRuntime:
    role: str = "gateway+worker"
    instance_name: str = ""
    capabilities: tuple[str, ...] = ()
    concurrency: int = 1
    poll_interval_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0
    stream: ExecutionStreamRuntime = field(default_factory=ExecutionStreamRuntime)


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
    execution: ExecutionRuntime = field(default_factory=ExecutionRuntime)
    #: Configured watchers, passed through unchanged. Each names a source to
    #: observe and a workflow to run; the runtime builds them at startup and a
    #: broken one stops it.
    watchers: list[WatcherConfig] = field(default_factory=list)
