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
class WatcherModel(NamedResourceModel):
    """Represents a single entry from the YAML ``watchers`` block."""


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
class ExecutionWorkerModel:
    instance_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    concurrency: int = 1
    poll_interval_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0


@dataclass(slots=True)
class ExecutionStreamModel:
    transport: str = "in_process"
    url: str = ""
    channel: str = "nlght_execution_stream"


@dataclass(slots=True)
class ExecutionModel:
    role: str = "gateway+worker"
    worker: ExecutionWorkerModel = field(default_factory=ExecutionWorkerModel)
    stream: ExecutionStreamModel = field(default_factory=ExecutionStreamModel)


@dataclass(slots=True)
class EmbeddingModel:
    provider: str = ""
    model: str = ""
    batch_size: int = 64
    normalize: bool = True
    device: str = "auto"
    cache_dir: str = ""
    offline: str = "auto"


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
    execution: ExecutionModel = field(default_factory=ExecutionModel)
    embedding: EmbeddingModel = field(default_factory=EmbeddingModel)
    #: The `principal:` block, passed through unparsed: `type` selects the
    #: resolver and each resolver needs different keys, so validating them here
    #: would mean this file knowing every authentication method.
    principal: dict[str, Any] = field(default_factory=dict)
    watchers: list[WatcherModel] = field(default_factory=list)
