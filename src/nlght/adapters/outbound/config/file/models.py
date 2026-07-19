# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NamedResourceModel:
    """Shared shape for a named, toggleable YAML resource entry.

    Subclassed (with no added fields) per resource kind below purely for
    self-documenting type names — see ``core.config.snapshot.NamedResourceConfig``,
    which this maps onto.
    """
    name: str
    kind: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProtocolAdapterModel(NamedResourceModel):
    pass


@dataclass(slots=True)
class GatewayModel(NamedResourceModel):
    """Represents a single gateway entry from the YAML ``gateways`` block."""


@dataclass(slots=True)
class OsRuntimeModel:
    """Singular OS runtime from the YAML ``os_runtime`` block.

    ``kind``: ``"local"`` or ``"docker"``.
    """
    kind: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelProviderModel(NamedResourceModel):
    pass


@dataclass(slots=True)
class WorkflowsPersistenceModel:
    """Persistence configuration from ``integrations.persistence.workflows``."""
    backend: str = "postgres"
    url: str = ""


@dataclass(slots=True)
class HiveMindProviderModel(NamedResourceModel):
    pass


@dataclass(slots=True)
class LicensingConfigModel:
    license_key: str | None = None


@dataclass(slots=True)
class PlatformFileConfigModel:
    catalogs_playbooks_definitions_path: str | None = None
    protocol_adapters: list[ProtocolAdapterModel] = field(default_factory=list)
    gateways: list[GatewayModel] = field(default_factory=list)
    os_runtime: OsRuntimeModel | None = None
    model_providers: list[ModelProviderModel] = field(default_factory=list)
    default_provider: str = ""
    persistence_workflows: WorkflowsPersistenceModel = field(
        default_factory=WorkflowsPersistenceModel
    )
    persistence_hive_mind_provider: HiveMindProviderModel | None = None
    licensing: LicensingConfigModel = field(default_factory=LicensingConfigModel)
