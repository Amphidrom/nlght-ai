# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlaybooksCatalogConfig:
    definitions_path: str | None = None


@dataclass(slots=True)
class CatalogsSnapshot:
    playbooks: PlaybooksCatalogConfig = field(default_factory=PlaybooksCatalogConfig)


@dataclass(slots=True)
class NamedResourceConfig:
    """Shared shape for a named, toggleable resource with adapter-specific config.

    ``kind`` selects the adapter implementation; ``config`` is passed through
    to it unchanged. Subclassed (with no added fields) for each distinct
    resource kind below purely for self-documenting type names — they are
    otherwise interchangeable, and consumers duck-type on these four
    attributes rather than checking the concrete subclass.
    """
    name: str
    kind: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProtocolAdapterConfig(NamedResourceConfig):
    pass


@dataclass(slots=True)
class GatewayConfig(NamedResourceConfig):
    """Replaces the old ``RuntimeSubsystemConfig`` for gateway entries."""


@dataclass(slots=True)
class OsRuntimeConfig:
    """Singular OS runtime configuration — either ``local`` or ``docker``.

    Only exactly one runtime can be configured per deployment.
    """
    kind: str       # "local" | "docker"
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelProviderConfig(NamedResourceConfig):
    pass


@dataclass(slots=True)
class WorkflowsPersistenceConfig:
    """Persistence configuration for workflow metadata (DB URL etc.).

    Lives under ``integrations.persistence.workflows`` in the YAML.
    """
    backend: str = "postgres"
    url: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HiveMindProviderConfig(NamedResourceConfig):
    pass


@dataclass(slots=True)
class HiveMindPersistenceConfig:
    provider: HiveMindProviderConfig | None = None


@dataclass(slots=True)
class PersistenceIntegrationConfig:
    workflows: WorkflowsPersistenceConfig = field(default_factory=WorkflowsPersistenceConfig)
    hive_mind: HiveMindPersistenceConfig = field(default_factory=HiveMindPersistenceConfig)


@dataclass(slots=True)
class IntegrationsSnapshot:
    model_providers: list[ModelProviderConfig] = field(default_factory=list)
    persistence: PersistenceIntegrationConfig = field(default_factory=PersistenceIntegrationConfig)


@dataclass(slots=True)
class LicensingConfig:
    license_key: str | None = None


@dataclass(slots=True)
class LoggingConfig:
    """Spring-Boot-style per-logger level overrides.

    The special key ``root`` sets the root log level.  All other keys are
    logger names, e.g. ``nlght``, ``sqlalchemy.engine``, ``httpx2``.
    """
    level: dict[str, str] = field(default_factory=lambda: {"root": "INFO", "nlght": "DEBUG"})


@dataclass(slots=True)
class PlatformConfigSnapshot:
    catalogs: CatalogsSnapshot = field(default_factory=CatalogsSnapshot)
    protocol_adapters: list[ProtocolAdapterConfig] = field(default_factory=list)
    gateways: list[GatewayConfig] = field(default_factory=list)
    os_runtime: OsRuntimeConfig | None = None
    integrations: IntegrationsSnapshot = field(default_factory=IntegrationsSnapshot)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    licensing: LicensingConfig = field(default_factory=LicensingConfig)
