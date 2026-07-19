# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio

import nlght.bootstrap.diag as bootstrap_main
from nlght.core.config.runtime_context import (
    DockerOsRuntimeSubsystemRuntime,
    GatewayRegistry,
    GenericModelProviderRuntime,
    GenericRuntimeSubsystemRuntime,
    HiveMindProviderRegistry,
    HttpGatewaySubsystemRuntime,
    ModelProviderRegistry,
    OllamaModelProviderRuntime,
    PersistenceSubsystemRuntime,
    ProtocolAdapterRegistry,
    RuntimeContext,
)
from nlght.core.config.snapshot import IntegrationsSnapshot, PlatformConfigSnapshot


class StubConfigurationService:
    async def load_snapshot(self) -> PlatformConfigSnapshot:
        return PlatformConfigSnapshot(integrations=IntegrationsSnapshot())


class StubContainer:
    def __init__(self) -> None:
        self.configuration_service = StubConfigurationService()
        self.subsystems: list = []
        self.runtime_context = RuntimeContext(
            protocol_adapter_registry=ProtocolAdapterRegistry([]),
            gateway_registry=GatewayRegistry([]),
            model_provider_registry=ModelProviderRegistry([]),
            hive_mind_provider_registry=HiveMindProviderRegistry([]),
            protocol_detector_runtimes=[],
            gateways=[
                HttpGatewaySubsystemRuntime(name="gw", host="0.0.0.0", port=8000),
                GenericRuntimeSubsystemRuntime(name="generic", kind="custom", config={}),
            ],
            os_runtime=DockerOsRuntimeSubsystemRuntime(name="docker", base_image="python:3.12"),
            persistence=PersistenceSubsystemRuntime(name="db", backend="postgres", url="postgresql://db"),
            model_providers=[
                OllamaModelProviderRuntime(name="ollama", base_url="http://localhost:11434", default_model="llama3"),
                GenericModelProviderRuntime(name="generic", kind="custom", config={}),
            ],
            hive_mind_provider=None,
        )


async def _stub_build_container(config_path: str) -> StubContainer:
    return StubContainer()


def test_main_runs_and_prints_runtime_overview(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bootstrap_main, "build_container", _stub_build_container)

    asyncio.run(bootstrap_main.main())

    out = capsys.readouterr().out
    assert "Loaded config snapshot" in out
    assert "Initialized gateways:" in out
    assert "Initialized model providers:" in out
