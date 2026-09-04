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
class WatcherConfig(NamedResourceConfig):
    """One configured watcher: what it observes and what it runs.

    ``kind`` selects the source (`filesystem`, `confluence`), and ``config``
    carries that source's settings alongside the workflow to trigger and the
    timings. The source block is the same shape the `ingestion.source` step
    takes, so the two are described identically — deliberately separate values,
    though: a watcher may observe a narrower path than the run it triggers
    ingests.
    """


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
    # No knowledge entry: the knowledge database is reached only through the
    # resource activations that use it, each carrying its own pg_url, and is
    # migrated with an explicit `-x url=...`. A platform-wide URL here would
    # silently override where an activation reads and writes.
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
class ExecutionWorkerConfig:
    instance_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    concurrency: int = 1
    poll_interval_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0


@dataclass(slots=True)
class ExecutionStreamConfig:
    """Transport carrying a running execution's live signals to whoever relays them.

    ``in_process`` serves a single node: the gateway can only relay a run its own
    worker claimed. ``postgres`` carries signals between nodes, so any gateway can
    relay a run claimed by any worker. The default keeps single-node deployments
    exactly as they were.
    """

    transport: str = "in_process"
    url: str = ""
    """Database URL for a distributed transport; empty reuses the workflows URL."""
    channel: str = "nlght_execution_stream"
    """Channel name, so two deployments can share one database without crosstalk."""


@dataclass(slots=True)
class ExecutionConfig:
    role: str = "gateway+worker"
    worker: ExecutionWorkerConfig = field(default_factory=ExecutionWorkerConfig)
    stream: ExecutionStreamConfig = field(default_factory=ExecutionStreamConfig)


@dataclass(slots=True)
class EmbeddingConfig:
    """Text-to-vector provider for ingestion and vector retrieval.

    Deliberately separate from the chat model providers: the validated
    implementation is a local model, not a conversational backend. An empty
    ``provider`` means no embedding capability is configured, and vector
    retrieval is then unavailable rather than silently degraded.
    """

    provider: str = ""
    model: str = ""
    batch_size: int = 64
    normalize: bool = True

    #: Where the model runs. ``auto`` takes a GPU when one is usable and falls
    #: back to the CPU; a explicit ``cpu``/``cuda``/``cuda:1``/``mps`` overrides
    #: that, which is what a machine with a GPU reserved for something else
    #: needs.
    device: str = "auto"
    #: Where the model files live. Empty means the library default
    #: (``HF_HOME``, otherwise ``~/.cache/huggingface``). Setting it explicitly
    #: is what lets several workers, or a container, share one download.
    cache_dir: str = ""
    #: Whether loading may reach the network. ``auto`` uses the cache when the
    #: model is fully there and only downloads when it is not; ``always`` never
    #: reaches out and fails if the cache is incomplete; ``never`` revalidates
    #: every file against the hub on every start.
    offline: str = "auto"


@dataclass(slots=True)
class PlatformConfigSnapshot:
    catalogs: CatalogsSnapshot = field(default_factory=CatalogsSnapshot)
    protocol_adapters: list[ProtocolAdapterConfig] = field(default_factory=list)
    gateways: list[GatewayConfig] = field(default_factory=list)
    os_runtime: OsRuntimeConfig | None = None
    integrations: IntegrationsSnapshot = field(default_factory=IntegrationsSnapshot)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    licensing: LicensingConfig = field(default_factory=LicensingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    #: How a caller's identity is established, if at all. Passed through to the
    #: resolver factory the way an adapter-specific block is, because `type`
    #: selects the implementation and each one needs different keys.
    #:
    #: Platform-level rather than per protocol adapter: the trust boundary is a
    #: property of where this process runs, not of which endpoint a request
    #: arrived on. Empty means no principal is ever established, which is the
    #: default and what the platform did before this existed.
    principal: dict[str, Any] = field(default_factory=dict)
    watchers: list[WatcherConfig] = field(default_factory=list)
