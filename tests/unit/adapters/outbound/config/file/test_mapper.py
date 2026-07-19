# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from nlght.adapters.outbound.config.file.mapper import to_snapshot
from nlght.adapters.outbound.config.file.models import (
    GatewayModel,
    HiveMindProviderModel,
    ModelProviderModel,
    OsRuntimeModel,
    PlatformFileConfigModel,
    ProtocolAdapterModel,
    WorkflowsPersistenceModel,
)


def test_to_snapshot_maps_all_configuration_sections() -> None:
    model = PlatformFileConfigModel(
        catalogs_playbooks_definitions_path=".config/playbooks",
        protocol_adapters=[
            ProtocolAdapterModel(
                name="openai-http",
                kind="openai",
                enabled=True,
                config={"base_path": "/v1"},
            )
        ],
        gateways=[
            GatewayModel(
                name="gateway",
                kind="http",
                enabled=True,
                config={"port": 9000},
            )
        ],
        os_runtime=OsRuntimeModel(
            kind="local",
            enabled=True,
            config={"workdir": "."},
        ),
        model_providers=[
            ModelProviderModel(
                name="ollama-main",
                kind="ollama",
                enabled=True,
                config={"default_model": "llama3"},
            )
        ],
        persistence_workflows=WorkflowsPersistenceModel(
            backend="postgres",
            url="postgresql://db",
        ),
        persistence_hive_mind_provider=HiveMindProviderModel(
            name="hm",
            kind="postgres",
            enabled=True,
            config={"dsn": "postgresql://hm"},
        ),
    )

    snapshot = to_snapshot(model)

    assert snapshot.catalogs.playbooks.definitions_path == ".config/playbooks"
    assert snapshot.protocol_adapters[0].name == "openai-http"
    assert snapshot.gateways[0].kind == "http"
    assert snapshot.os_runtime is not None
    assert snapshot.os_runtime.kind == "local"
    assert snapshot.integrations.model_providers[0].kind == "ollama"
    assert snapshot.integrations.persistence.workflows.url == "postgresql://db"
    assert snapshot.integrations.persistence.hive_mind.provider is not None
    assert snapshot.integrations.persistence.hive_mind.provider.name == "hm"
