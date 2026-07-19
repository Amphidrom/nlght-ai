# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from nlght.application.config.composition_service import CompositionService
from nlght.core.config.runtime_context import (
    DockerOsRuntimeSubsystemRuntime,
    GenericModelProviderRuntime,
    GenericRuntimeSubsystemRuntime,
    HttpGatewaySubsystemRuntime,
    LocalOsRuntimeSubsystemRuntime,
    OllamaModelProviderRuntime,
    PersistenceSubsystemRuntime,
)
from nlght.core.config.snapshot import (
    GatewayConfig,
    HiveMindPersistenceConfig,
    HiveMindProviderConfig,
    IntegrationsSnapshot,
    ModelProviderConfig,
    OsRuntimeConfig,
    PersistenceIntegrationConfig,
    PlatformConfigSnapshot,
    ProtocolAdapterConfig,
    WorkflowsPersistenceConfig,
)


def test_compose_maps_registries_and_runtime_objects() -> None:
    snapshot = PlatformConfigSnapshot(
        protocol_adapters=[
            ProtocolAdapterConfig(name="openai", kind="openai", enabled=True, config={"base_path": "/v1"}),
            ProtocolAdapterConfig(name="disabled", kind="generic_json", enabled=False),
        ],
        gateways=[
            GatewayConfig(name="gateway", kind="http", enabled=True, config={"host": "127.0.0.1", "port": 9999}),
            GatewayConfig(name="custom", kind="custom", enabled=True, config={"k": "v"}),
            GatewayConfig(name="off", kind="custom", enabled=False),
        ],
        os_runtime=OsRuntimeConfig(kind="docker", enabled=True, config={"base_image": "python:3.12"}),
        integrations=IntegrationsSnapshot(
            model_providers=[
                ModelProviderConfig(
                    name="ollama-main",
                    kind="ollama",
                    enabled=True,
                    config={
                        "base_url": "http://localhost:11434",
                        "default_model": "llama3",
                        "api_key": "secret",
                        "headers": {"X-Test": "1"},
                        "request_timeout_s": 42,
                        "stream_connect_timeout_s": 7,
                        "stream_read_timeout_s": 90,
                    },
                ),
                ModelProviderConfig(name="generic", kind="custom", enabled=True, config={"x": 1}),
                ModelProviderConfig(name="disabled", kind="custom", enabled=False),
            ],
            persistence=PersistenceIntegrationConfig(
                workflows=WorkflowsPersistenceConfig(
                    backend="postgres",
                    url="postgresql://runtime",
                ),
                hive_mind=HiveMindPersistenceConfig(
                    provider=HiveMindProviderConfig(
                        name="hive",
                        kind="postgres",
                        enabled=True,
                        config={"dsn": "postgresql://hive"},
                    )
                ),
            ),
        ),
    )

    context = CompositionService().compose(snapshot)

    assert [item.name for item in context.protocol_detector_runtimes] == ["openai"]
    assert len(context.protocol_adapter_registry.all()) == 2
    assert len(context.protocol_adapter_registry.enabled()) == 1
    assert len(context.gateway_registry.all()) == 3
    assert len(context.gateway_registry.enabled()) == 2
    assert len(context.model_provider_registry.all()) == 3
    assert len(context.hive_mind_provider_registry.enabled()) == 1

    assert any(isinstance(item, HttpGatewaySubsystemRuntime) for item in context.gateways)
    assert any(isinstance(item, GenericRuntimeSubsystemRuntime) for item in context.gateways)

    assert isinstance(context.os_runtime, DockerOsRuntimeSubsystemRuntime)
    assert context.os_runtime.base_image == "python:3.12"

    assert context.persistence is not None
    assert isinstance(context.persistence, PersistenceSubsystemRuntime)
    assert context.persistence.url == "postgresql://runtime"

    assert any(isinstance(item, OllamaModelProviderRuntime) for item in context.model_providers)
    assert any(isinstance(item, GenericModelProviderRuntime) for item in context.model_providers)
    ollama = next(item for item in context.model_providers if isinstance(item, OllamaModelProviderRuntime))
    assert ollama.api_key == "secret"
    assert ollama.headers == {"X-Test": "1"}
    assert ollama.request_timeout_s == 42.0
    assert ollama.stream_connect_timeout_s == 7.0
    assert ollama.stream_read_timeout_s == 90.0
    assert context.hive_mind_provider is not None
    assert context.hive_mind_provider.name == "hive"


def test_compose_ollama_timeouts_remain_none_when_not_configured() -> None:
    snapshot = PlatformConfigSnapshot(
        integrations=IntegrationsSnapshot(
            model_providers=[
                ModelProviderConfig(
                    name="ollama-main",
                    kind="ollama",
                    enabled=True,
                    config={
                        "base_url": "http://localhost:11434",
                        "default_model": "llama3",
                    },
                ),
            ],
        ),
    )

    context = CompositionService().compose(snapshot)

    ollama = next(item for item in context.model_providers if isinstance(item, OllamaModelProviderRuntime))
    assert ollama.request_timeout_s is None
    assert ollama.stream_connect_timeout_s is None
    assert ollama.stream_read_timeout_s is None


def test_compose_local_os_runtime() -> None:
    snapshot = PlatformConfigSnapshot(
        os_runtime=OsRuntimeConfig(
            kind="local",
            enabled=True,
            config={"workdir": "/tmp", "workspace_path": "/tmp/ws"},
        ),
    )

    context = CompositionService().compose(snapshot)

    assert isinstance(context.os_runtime, LocalOsRuntimeSubsystemRuntime)
    assert context.os_runtime.workdir == "/tmp"
    assert context.os_runtime.workspace_path == "/tmp/ws"
