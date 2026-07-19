# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.adapters.outbound.config.file.models import PlatformFileConfigModel
from nlght.core.config.snapshot import (
    CatalogsSnapshot,
    GatewayConfig,
    HiveMindPersistenceConfig,
    HiveMindProviderConfig,
    IntegrationsSnapshot,
    LicensingConfig,
    ModelProviderConfig,
    OsRuntimeConfig,
    PersistenceIntegrationConfig,
    PlatformConfigSnapshot,
    PlaybooksCatalogConfig,
    ProtocolAdapterConfig,
    WorkflowsPersistenceConfig,
)


def to_snapshot(model: PlatformFileConfigModel) -> PlatformConfigSnapshot:
    return PlatformConfigSnapshot(
        licensing=LicensingConfig(
            license_key=model.licensing.license_key,
        ),
        catalogs=CatalogsSnapshot(
            playbooks=PlaybooksCatalogConfig(
                definitions_path=model.catalogs_playbooks_definitions_path,
            )
        ),
        protocol_adapters=[
            ProtocolAdapterConfig(
                name=item.name,
                kind=item.kind,
                enabled=item.enabled,
                config=item.config,
            )
            for item in model.protocol_adapters
        ],
        gateways=[
            GatewayConfig(
                name=item.name,
                kind=item.kind,
                enabled=item.enabled,
                config=item.config,
            )
            for item in model.gateways
        ],
        os_runtime=(
            OsRuntimeConfig(
                kind=model.os_runtime.kind,
                enabled=model.os_runtime.enabled,
                config=model.os_runtime.config,
            )
            if model.os_runtime is not None else None
        ),
        integrations=IntegrationsSnapshot(
            model_providers=[
                ModelProviderConfig(
                    name=item.name,
                    kind=item.kind,
                    enabled=item.enabled,
                    config=item.config,
                )
                for item in model.model_providers
            ],
            persistence=PersistenceIntegrationConfig(
                workflows=WorkflowsPersistenceConfig(
                    backend=model.persistence_workflows.backend,
                    url=model.persistence_workflows.url,
                ),
                hive_mind=HiveMindPersistenceConfig(
                    provider=(
                        HiveMindProviderConfig(
                            name=model.persistence_hive_mind_provider.name,
                            kind=model.persistence_hive_mind_provider.kind,
                            enabled=model.persistence_hive_mind_provider.enabled,
                            config=model.persistence_hive_mind_provider.config,
                        )
                        if model.persistence_hive_mind_provider is not None
                        else None
                    )
                ),
            ),
        ),
    )
