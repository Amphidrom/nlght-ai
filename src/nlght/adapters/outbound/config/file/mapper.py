# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.adapters.outbound.config.file.models import PlatformFileConfigModel
from nlght.core.config.snapshot import (
    CatalogsSnapshot,
    EmbeddingConfig,
    ExecutionConfig,
    ExecutionStreamConfig,
    ExecutionWorkerConfig,
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
    WatcherConfig,
    WorkflowsPersistenceConfig,
)


def to_snapshot(model: PlatformFileConfigModel) -> PlatformConfigSnapshot:
    return PlatformConfigSnapshot(
        principal=dict(model.principal or {}),
        licensing=LicensingConfig(
            license_key=model.licensing.license_key,
        ),
        execution=ExecutionConfig(
            role=model.execution.role,
            stream=ExecutionStreamConfig(
                transport=model.execution.stream.transport,
                url=model.execution.stream.url,
                channel=model.execution.stream.channel,
            ),
            worker=ExecutionWorkerConfig(
                instance_name=model.execution.worker.instance_name,
                capabilities=list(model.execution.worker.capabilities),
                concurrency=model.execution.worker.concurrency,
                poll_interval_seconds=model.execution.worker.poll_interval_seconds,
                lease_seconds=model.execution.worker.lease_seconds,
                heartbeat_seconds=model.execution.worker.heartbeat_seconds,
                retry_base_seconds=model.execution.worker.retry_base_seconds,
                retry_max_seconds=model.execution.worker.retry_max_seconds,
            ),
        ),
        embedding=EmbeddingConfig(
            provider=model.embedding.provider,
            model=model.embedding.model,
            batch_size=model.embedding.batch_size,
            normalize=model.embedding.normalize,
            device=model.embedding.device,
            cache_dir=model.embedding.cache_dir,
            offline=model.embedding.offline,
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
        watchers=[
            WatcherConfig(
                name=item.name,
                kind=item.kind,
                enabled=item.enabled,
                config=item.config,
            )
            for item in model.watchers
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
