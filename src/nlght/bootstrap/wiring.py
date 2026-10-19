# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from sqlalchemy.ext.asyncio import AsyncEngine

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.ollama_adapter import OllamaHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.adapters.inbound.ingestion import (
    DebouncedFilesystemWatcher,
    FilesystemWatcherSettings,
)
from nlght.adapters.inbound.registry import http_adapter_registry, inbound_adapter_registry
from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.adapters.outbound.config.file.source import FileConfigurationSource
from nlght.adapters.outbound.embedding.sentence_transformers import (
    PROVIDER as SENTENCE_TRANSFORMERS_PROVIDER,
)
from nlght.adapters.outbound.embedding.sentence_transformers import (
    SentenceTransformersEmbeddingClient,
)
from nlght.adapters.outbound.ingestion.factory import (
    SourceConfigurationError,
    build_source,
)
from nlght.adapters.outbound.licensing.keygen_license_adapter import KeygenLicenseAdapter
from nlght.adapters.outbound.model import OllamaClient, OllamaCloudClient
from nlght.adapters.outbound.os_runtime.local import LocalOsRuntime
from nlght.adapters.outbound.persistence.access_rule_repository import SqlAlchemyAccessRuleRepository
from nlght.adapters.outbound.persistence.execution_repository import (
    SqlAlchemyExecutionArtifactStore,
    SqlAlchemyExecutionRepository,
)
from nlght.adapters.outbound.persistence.postgres import PostgresPersistenceSubsystem
from nlght.adapters.outbound.persistence.resource_repository import SqlAlchemyResourceRepository
from nlght.adapters.outbound.persistence.workflow_repository import SqlAlchemyWorkflowRepository
from nlght.adapters.outbound.playbooks.catalog import PlaybookCatalogBuilder
from nlght.adapters.outbound.playbooks.loader import load_playbook_definitions
from nlght.adapters.outbound.principal import (
    CompositePrincipalResolver,
    TrustedGatewayPrincipalResolver,
)
from nlght.adapters.outbound.protocol.composite import CompositeProtocolDetector
from nlght.adapters.outbound.protocol.generic_json import GenericJsonProtocolDetector
from nlght.adapters.outbound.protocol.ollama import OllamaProtocolDetector
from nlght.adapters.outbound.protocol.openai import OpenAIProtocolDetector
from nlght.adapters.outbound.session.body import BodyParameterSessionKeyResolver
from nlght.adapters.outbound.session.composite import CompositeSessionKeyResolver
from nlght.adapters.outbound.session.header import HeaderSessionKeyResolver
from nlght.adapters.outbound.session.query import QueryParamSessionKeyResolver
from nlght.adapters.outbound.signals.execution_stream import InProcessExecutionStreamBroker
from nlght.adapters.outbound.signals.postgres_execution_stream import (
    PostgresExecutionStreamBroker,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.tools.action_registry import action_authorization_registry
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.catalog import ToolCatalogBuilder
from nlght.adapters.outbound.tools.registry import tool_registry
from nlght.adapters.outbound.trigger.default import DefaultTriggerResolver
from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.adapters.outbound.workflow.registry import step_registry
from nlght.application.config.composition_service import CompositionService
from nlght.application.config.configuration_service import ConfigurationService
from nlght.application.entry.gateway_service import GatewayService
from nlght.application.execution.dispatch_service import ExecutionDispatchService
from nlght.application.execution.worker import ExecutionWorker, ExecutionWorkerSettings
from nlght.core.config.runtime_context import (
    ActionAuthorizationRuntime,
    AnthropicModelProviderRuntime,
    DockerOsRuntimeSubsystemRuntime,
    GoogleModelProviderRuntime,
    HiveMindProviderRuntime,
    LocalOsRuntimeSubsystemRuntime,
    OllamaLocalModelProviderRuntime,
    OllamaModelProviderRuntime,
    OpenAICloudModelProviderRuntime,
    RuntimeContext,
)
from nlght.core.config.snapshot import PlatformConfigSnapshot
from nlght.core.errors.errors import ConfigurationError
from nlght.core.session import SessionAccess
from nlght.ports.inbound.http_protocol_adapter import HttpProtocolAdapter
from nlght.ports.inbound.inbound_adapter import InboundAdapter
from nlght.ports.outbound.access_rule_repository import AccessRuleRepository
from nlght.ports.outbound.embedding_client import EmbeddingClient
from nlght.ports.outbound.execution_artifact_store import ExecutionArtifactStore
from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
from nlght.ports.outbound.execution_stream import ExecutionStreamBroker
from nlght.ports.outbound.licensing import LicensingPort
from nlght.ports.outbound.metering import MeteringPort
from nlght.ports.outbound.os_runtime import OsRuntimeFactory
from nlght.ports.outbound.principal_resolver import PrincipalResolver
from nlght.ports.outbound.protocol_detection import ProtocolDetector
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.runtime_subsystem import SubsystemLifecycle
from nlght.ports.outbound.session_key_resolver import SessionKeyResolver
from nlght.ports.outbound.store_coordinator import StoreCoordinatorFactory
from nlght.ports.outbound.workflow_executor import WorkflowExecutor
from nlght.ports.outbound.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    configuration_service: ConfigurationService
    composition_service: CompositionService
    runtime_context: RuntimeContext
    gateway_service: GatewayService
    subsystems: list[SubsystemLifecycle] = field(default_factory=list)
    workflow_repository: WorkflowRepository | None = field(default=None)
    resource_repository: ResourceRepository | None = field(default=None)
    http_protocol_adapters: list[HttpProtocolAdapter] = field(default_factory=list)
    inbound_adapters: list[InboundAdapter] = field(default_factory=list)
    workflow_executor: WorkflowExecutor | None = field(default=None)
    coordinator_factory: StoreCoordinatorFactory | None = field(default=None)
    engine: AsyncEngine | None = field(default=None)
    # No zero-arg default is possible -- KeygenLicenseAdapter needs the
    # license_key from config. Matches the other fields built later in the
    # composition flow: optional here, always explicitly passed by build().
    license: LicensingPort | None = field(default=None)
    metering: MeteringPort | None = field(default=None)
    execution_dispatcher: ExecutionDispatcher | None = field(default=None)
    execution_artifact_store: ExecutionArtifactStore | None = field(default=None)
    execution_worker: ExecutionWorker | None = field(default=None)
    execution_stream_broker: ExecutionStreamBroker | None = field(default=None)


def _build_metering(runtime_context: RuntimeContext) -> MeteringPort | None:
    """Build the first enabled metering gateway.

    Args:
        runtime_context (RuntimeContext): Configured runtime gateways.

    Returns:
        MeteringPort | None: Metering adapter, or None when no usable gateway is configured.

    Side Effects:
        May import the Prometheus extra and log an unavailable or unknown backend.
    """
    from nlght.core.config.runtime_context import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        GenericRuntimeSubsystemRuntime,
    )

    for runtime in runtime_context.gateways:
        if not isinstance(runtime, GenericRuntimeSubsystemRuntime):
            continue
        if runtime.kind != "metering":
            continue
        backend = str(runtime.config.get("backend", "prometheus"))
        endpoint = str(runtime.config.get("endpoint", "/metrics"))
        if backend == "prometheus":
            try:
                from nlght.adapters.outbound.metering.prometheus import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    PrometheusMeteringAdapter,
                )
                adapter = PrometheusMeteringAdapter(endpoint=endpoint)
                logger.info("metering.prometheus.configured | endpoint=%s", endpoint)
                return adapter
            except ImportError:
                logger.warning(
                    "prometheus-client is not installed. Metering subsystem disabled. "
                    "Install with: pip install 'nlght-ai[metrics]'"
                )
        else:
            logger.warning("metering.unknown_backend | backend=%s: skipping", backend)
    return None


def _build_license_service(runtime_context: RuntimeContext) -> LicensingPort:
    """Require a configured license service at startup.

    Args:
        runtime_context (RuntimeContext): Runtime license configuration.

    Returns:
        LicensingPort: Keygen license adapter. The license does not gate individual capabilities.

    Raises:
        RuntimeError: No license key is configured.

    Side Effects:
        Logs the selected license adapter.
    """
    lic = runtime_context.licensing
    if lic and lic.license_key:
        adapter = KeygenLicenseAdapter(lic.license_key)
        logger.info("license.keygen | %s", adapter.context_metadata())
        return adapter
    raise RuntimeError(
        "No license configured. Set license_key in your config."
    )


def _build_os_runtime(
    runtime_context: RuntimeContext,
) -> tuple[OsRuntimeFactory | None, list[SubsystemLifecycle]]:
    runtime = runtime_context.os_runtime
    if runtime is None:
        return None, []
    if isinstance(runtime, LocalOsRuntimeSubsystemRuntime):
        return LocalOsRuntime(workdir=runtime.workdir), []
    if isinstance(runtime, DockerOsRuntimeSubsystemRuntime):
        try:
            from nlght.adapters.outbound.os_runtime.docker import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                DockerOsRuntimeFactory,
            )

            factory = DockerOsRuntimeFactory(
                base_image=runtime.base_image,
                workspace_path=runtime.workspace_path,
                extra_hosts=runtime.extra_hosts,
                network_mode=runtime.network_mode,
                allow_runtime_env=runtime.allow_runtime_env,
            )
            return factory, [factory]
        except ImportError:
            import logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

            logging.getLogger(__name__).warning(
                "docker package is not installed. DockerOsRuntimeFactory unavailable. "
                "Install with: pip install 'nlght-ai[docker]'"
            )
            return None, []
    return None, []


def _build_execution_stream_broker(
    runtime_context: RuntimeContext,
) -> tuple[ExecutionStreamBroker, SubsystemLifecycle | None]:
    """Select the live execution signal transport.

    Args:
        runtime_context (RuntimeContext): Execution-stream and persistence settings.

    Returns:
        tuple[ExecutionStreamBroker, SubsystemLifecycle | None]: Broker and its
            lifecycle, if the transport owns a connection.

    Raises:
        RuntimeError: PostgreSQL transport has no connection URL.

    Side Effects:
        Constructs an in-process or PostgreSQL broker.
    """
    stream = runtime_context.execution.stream
    if stream.transport == "in_process":
        return InProcessExecutionStreamBroker(), None

    persistence = runtime_context.persistence
    url = stream.url or (persistence.url if persistence is not None else "")
    if not url:
        raise RuntimeError(
            "execution.stream.transport=postgres requires execution.stream.url or "
            "integrations.persistence.workflows.url"
        )
    broker = PostgresExecutionStreamBroker(url=url, channel=stream.channel)
    return broker, broker


def _build_subsystems(
    runtime_context: RuntimeContext,
) -> list[SubsystemLifecycle]:
    persistence = runtime_context.persistence
    if persistence is not None and persistence.url:
        return [PostgresPersistenceSubsystem(persistence)]
    return []


def _build_http_adapters(
    runtime_context: RuntimeContext,
    model_clients: dict[str, Any],
) -> list[HttpProtocolAdapter]:
    """Build protocol adapters with specific routes before the catch-all route.

    Args:
        runtime_context (RuntimeContext): Configured protocol runtimes.
        model_clients (dict[str, Any]): Model clients keyed by provider name.

    Returns:
        list[HttpProtocolAdapter]: HTTP adapters in route precedence order.
    """
    adapters: list[HttpProtocolAdapter] = []
    generic_json: GenericJsonHttpProtocolAdapter | None = None

    for runtime in runtime_context.protocol_detector_runtimes:
        resolver = _build_single_resolver(runtime.config.get("session_key_resolver") or {})
        if runtime.kind == "openai":
            provider_name = str(runtime.config.get("model_provider", ""))
            model_backend = model_clients.get(provider_name) if provider_name else None
            workflows: dict[str, str] = {
                k: str(v)
                for k, v in (runtime.config.get("workflows") or {}).items()
            }
            adapters.append(
                OpenAIHttpProtocolAdapter(
                    base_path=str(runtime.config.get("base_path", "/v1")),
                    model_backend=model_backend,
                    workflow_mapping=workflows,
                    session_key_resolver=resolver,
                    model_provider_name=provider_name,
                )
            )
        elif runtime.kind == "ollama-api":
            provider_name = str(runtime.config.get("model_provider", ""))
            model_backend = model_clients.get(provider_name) if provider_name else None
            workflows = {
                k: str(v)
                for k, v in (runtime.config.get("workflows") or {}).items()
            }
            adapters.append(
                OllamaHttpProtocolAdapter(
                    model_backend=model_backend,
                    workflow_mapping=workflows,
                    session_key_resolver=resolver,
                    model_provider_name=provider_name,
                )
            )
        elif runtime.kind == "generic_json":
            # Same `workflows` key as the two protocol adapters above, keyed by
            # request path instead of model name: this is where a workflow that
            # is not a completion gets an endpoint. Values stay raw here as a
            # bare workflow name or a {workflow, mode} mapping because the
            # adapter owns that shape.
            generic_json = GenericJsonHttpProtocolAdapter(
                session_key_resolver=resolver,
                workflow_mapping=dict(runtime.config.get("workflows") or {}),
                default_mode=str(runtime.config.get("mode", "sync")),
            )

    if generic_json is not None:
        adapters.append(generic_json)

    adapters.extend(http_adapter_registry.all())

    return adapters


def _build_protocol_detectors(runtime_context: RuntimeContext) -> list[ProtocolDetector]:
    detectors: list[ProtocolDetector] = []
    for runtime in runtime_context.protocol_detector_runtimes:
        if runtime.kind == "openai":
            detectors.append(
                OpenAIProtocolDetector(base_path=str(runtime.config.get("base_path", "/v1")))
            )
        elif runtime.kind == "ollama-api":
            detectors.append(OllamaProtocolDetector())
        elif runtime.kind == "generic_json":
            detectors.append(GenericJsonProtocolDetector())
    return detectors


def _build_single_resolver(cfg: dict[str, Any]) -> SessionKeyResolver | None:
    """Build one session-key resolver from a protocol configuration.

    Args:
        cfg (dict[str, Any]): Header, query, body, or composite resolver configuration.

    Returns:
        SessionKeyResolver | None: Configured resolver, or None for an invalid or absent variant.
    """
    kind = cfg.get("resolver")
    if kind == "header":
        key = cfg.get("key")
        if key and isinstance(key, str):
            return HeaderSessionKeyResolver(key)
    elif kind == "query":
        parameter = cfg.get("parameter")
        if parameter and isinstance(parameter, str):
            return QueryParamSessionKeyResolver(parameter)
    elif kind == "body":
        key = cfg.get("key")
        if key and isinstance(key, str):
            return BodyParameterSessionKeyResolver(key)
    elif kind == "composite":
        sub_cfgs = cfg.get("resolvers")
        if isinstance(sub_cfgs, list):
            children = [
                r
                for sub in sub_cfgs
                if isinstance(sub, dict)
                for r in (_build_single_resolver(sub),)
                if r is not None
            ]
            if children:
                return CompositeSessionKeyResolver(children)
    return None



def _build_principal_resolver(cfg: dict[str, Any]) -> PrincipalResolver | None:
    """Build a deployment-wide principal resolver from trusted sources.

    Args:
        cfg (dict[str, Any]): Trusted-gateway or composite resolver configuration and
            its trust settings.

    Returns:
        PrincipalResolver | None: Principal resolver, or None when no type is configured.

    Raises:
        ConfigurationError: The configured resolver type is unknown or invalid.
    """
    kind = str(cfg.get("type", "")).strip()
    if kind == "trusted_gateway":
        return TrustedGatewayPrincipalResolver(
            header=str(cfg.get("header", "")),
            trusted_hosts=tuple(str(h) for h in (cfg.get("trusted_hosts") or [])),
            secret_header=str(cfg.get("secret_header", "")),
            secret=str(cfg.get("secret", "")),
        )
    if kind == "composite":
        children = [
            resolver
            for sub in (cfg.get("resolvers") or [])
            if isinstance(sub, dict)
            for resolver in (_build_principal_resolver(sub),)
            if resolver is not None
        ]
        if children:
            return CompositePrincipalResolver(children)
        return None
    if kind:
        raise ConfigurationError(
            f"unknown principal resolver type '{kind}'; expected 'trusted_gateway' "
            f"or 'composite'"
        )
    return None


def _build_embedding_client(snapshot: PlatformConfigSnapshot) -> EmbeddingClient | None:
    """Construct one reusable embedding client for the runtime.

    Args:
        snapshot (PlatformConfigSnapshot): Platform embedding provider settings.

    Returns:
        EmbeddingClient | None: Embedding client, or None when the provider is absent or
            unavailable.

    Side Effects:
        May load a local model and logs unavailable providers.
    """
    config = snapshot.embedding
    if not config.provider:
        return None
    if config.provider != SENTENCE_TRANSFORMERS_PROVIDER:
        logger.warning(
            "embedding.provider.unknown | provider=%s: embeddings disabled", config.provider
        )
        return None
    try:
        return SentenceTransformersEmbeddingClient(
            model=config.model,
            batch_size=config.batch_size,
            normalize=config.normalize,
            device=config.device,
            cache_dir=config.cache_dir,
            offline=config.offline,
        )
    except (ImportError, ValueError) as exc:
        logger.warning("embedding.provider.unavailable | %s: embeddings disabled", exc)
        return None


def _build_watchers(
    runtime_context: RuntimeContext,
    dispatcher: ExecutionDispatcher | None,
) -> list[InboundAdapter]:
    """Create enabled source watchers from the runtime configuration.

    Args:
        runtime_context (RuntimeContext): Configured watcher definitions.
        dispatcher (ExecutionDispatcher | None): Execution dispatcher used by each watcher.

    Returns:
        list[InboundAdapter]: Configured inbound watcher adapters.

    Raises:
        RuntimeError: An enabled watcher has an invalid source or settings.

    Side Effects:
        Constructs source observers and logs enabled or disabled watchers.
    """
    watchers: list[InboundAdapter] = []
    for configured in runtime_context.watchers:
        if not configured.enabled:
            logger.info("watch.disabled | name=%s", configured.name)
            continue
        where = f"Watcher '{configured.name}'"
        config = dict(configured.config)
        try:
            settings = FilesystemWatcherSettings(
                workflow=str(config.get("workflow", "")),
                poll_interval_seconds=float(config.get("poll_interval_seconds", 1.0)),
                debounce_seconds=float(config.get("debounce_seconds", 2.0)),
            )
            source = build_source(configured.kind, config, where=where)
        except (SourceConfigurationError, ValueError) as exc:
            raise RuntimeError(f"{where} is misconfigured: {exc}") from exc
        watchers.append(
            DebouncedFilesystemWatcher(
                source=source, settings=settings, dispatcher=dispatcher
            )
        )
        logger.info(
            "watch.configured | name=%s kind=%s workflow=%s",
            configured.name, configured.kind, settings.workflow,
        )
    return watchers


def _build_repositories(
    subsystems: list[SubsystemLifecycle],
) -> tuple[
    WorkflowRepository | None,
    ResourceRepository | None,
    AccessRuleRepository | None,
]:
    for s in subsystems:
        if isinstance(s, PostgresPersistenceSubsystem):
            return (
                SqlAlchemyWorkflowRepository(s.engine),
                SqlAlchemyResourceRepository(s.engine),
                SqlAlchemyAccessRuleRepository(s.engine),
            )
    return None, None, None


def _build_coordinator_factory(runtime_context: RuntimeContext) -> StoreCoordinatorFactory:
    """Select the configured store coordinator factory.

    Args:
        runtime_context (RuntimeContext): Hive Mind provider configuration.

    Returns:
        StoreCoordinatorFactory: Filesystem or in-memory coordinator factory.

    Side Effects:
        May import the filesystem extra and logs the chosen factory.
    """
    provider = runtime_context.hive_mind_provider
    if provider is None or not isinstance(provider, HiveMindProviderRuntime):
        from nlght.adapters.outbound.hive_mind.simple import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            SimpleStoreCoordinatorFactory,
        )
        return SimpleStoreCoordinatorFactory()

    if provider.kind == "filesystem":
        try:
            from nlght.adapters.outbound.hive_mind.coordinator import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                HiveMindStoreCoordinatorFactory,
            )
            base_dir = str(provider.config.get("base_dir", "./sessions"))
            logger.info("coordinator_factory.hive_mind | base_dir=%s", base_dir)
            return HiveMindStoreCoordinatorFactory(base_dir=base_dir)
        except ImportError:
            logger.warning(
                "hive-mind extra is not installed. Using SimpleStoreCoordinatorFactory. "
                "Install with: pip install 'nlght-ai[hive-mind]'"
            )

    from nlght.adapters.outbound.hive_mind.simple import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        SimpleStoreCoordinatorFactory,
    )
    logger.info("coordinator_factory.simple | ttl=%d", provider.session_ttl_seconds)
    return SimpleStoreCoordinatorFactory(session_ttl_seconds=provider.session_ttl_seconds)


def _log_action_authorization(settings: ActionAuthorizationRuntime) -> None:
    """Log the effective action policy and approval adapter.

    Args:
        settings (ActionAuthorizationRuntime): Action-authorization enablement and settings.

    Side Effects:
        Reads registered adapters and emits startup log messages.
    """
    policy = action_authorization_registry.policy()
    approval = action_authorization_registry.approval()
    if not settings.enabled:
        ignored = [type(port).__name__ for port in (policy, approval) if port is not None]
        logger.warning(
            "action_authorization | disabled: model tool calls execute without policy or "
            "approval. Argument validation and binding still apply%s",
            f". Ignored registered extensions: {', '.join(ignored)}" if ignored else "",
        )
        return
    logger.info(
        "action_authorization | policy=%s approval=%s policy_timeout_s=%s approval_timeout_s=%s",
        type(policy).__name__ if policy is not None else "DefaultActionPolicy",
        type(approval).__name__ if approval is not None else "none (confirmation denies)",
        settings.policy_timeout_seconds,
        settings.approval_timeout_seconds,
    )


async def build_container(config_path: str) -> Container:
    configuration_source = FileConfigurationSource(config_path)
    configuration_service = ConfigurationService(configuration_source)
    composition_service = CompositionService()

    snapshot = await configuration_service.load_snapshot()
    runtime_context = composition_service.compose(snapshot)

    license = _build_license_service(runtime_context)

    subsystems = _build_subsystems(runtime_context)
    for s in subsystems:
        await s.start()

    (
        workflow_repository,
        resource_repository,
        access_rule_repository,
    ) = _build_repositories(subsystems)

    _engine: AsyncEngine | None = None
    for _s in subsystems:
        if isinstance(_s, PostgresPersistenceSubsystem):
            _engine = _s.engine
            break
    execution_engine: AsyncEngine | None = None
    if runtime_context.persistence is not None and _engine is not None:
        execution_pool_size = max(2, runtime_context.execution.concurrency + 1)
        execution_persistence = PostgresPersistenceSubsystem(
            runtime_context.persistence,
            engine_options={
                "pool_size": execution_pool_size,
                "max_overflow": 0,
                "pool_timeout": 5,
                "connect_args": {
                    "server_settings": {
                        "lock_timeout": "5000ms",
                        "statement_timeout": "15000ms",
                    }
                },
            },
            verify_migrations=False,
        )
        await execution_persistence.start()
        subsystems.append(execution_persistence)
        execution_engine = execution_persistence.engine
    execution_repository = (
        SqlAlchemyExecutionRepository(execution_engine)
        if execution_engine is not None
        else None
    )
    execution_dispatcher = (
        ExecutionDispatchService(execution_repository)
        if execution_repository is not None
        else None
    )
    execution_artifact_store = (
        SqlAlchemyExecutionArtifactStore(execution_engine)
        if execution_engine is not None
        else None
    )

    # Rule-based access policies (ADR-0022) are active whenever persistence is
    # configured. An empty access_policies table restricts nothing.
    access_rule_engine = (
        AccessRuleEngine(access_rule_repository)
        if access_rule_repository is not None
        else None
    )

    protocol_detector = CompositeProtocolDetector(_build_protocol_detectors(runtime_context))
    trigger_resolver = DefaultTriggerResolver()

    principal_resolver = _build_principal_resolver(snapshot.principal)
    gateway_service = GatewayService(
        protocol_detector=protocol_detector,
        trigger_resolver=trigger_resolver,
        workflow_repository=workflow_repository,
        resource_repository=resource_repository,
        principal_resolver=principal_resolver,
    )

    http_client = httpx2.AsyncClient()
    model_clients: dict[str, Any] = {}
    for _p in runtime_context.model_providers:
        if isinstance(_p, (OllamaModelProviderRuntime, OllamaLocalModelProviderRuntime)) and _p.base_url:
            client_cls = OllamaClient if isinstance(_p, OllamaLocalModelProviderRuntime) else OllamaCloudClient
            model_clients[_p.name] = client_cls(
                http_client=http_client,
                base_url=_p.base_url,
                default_model=_p.default_model,
                api_key=_p.api_key,
                extra_headers=_p.headers,
                request_timeout_s=_p.request_timeout_s,
                stream_connect_timeout_s=_p.stream_connect_timeout_s,
                stream_read_timeout_s=_p.stream_read_timeout_s,
                temperature=_p.temperature,
            )
        elif isinstance(_p, AnthropicModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.anthropic import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    AnthropicModelClient,
                )
                model_clients[_p.name] = AnthropicModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                )
            except ImportError:
                logger.warning(
                    "anthropic package is not installed. Provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[anthropic]'",
                    _p.name,
                )
        elif isinstance(_p, OpenAICloudModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.openai_cloud import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    OpenAICloudModelClient,
                )
                model_clients[_p.name] = OpenAICloudModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                    base_url=_p.base_url,
                    temperature=_p.temperature,
                )
            except ImportError:
                logger.warning(
                    "openai package is not installed. Provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[openai]'",
                    _p.name,
                )
        elif isinstance(_p, GoogleModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.google import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    GoogleModelClient,
                )
                model_clients[_p.name] = GoogleModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                    temperature=_p.temperature,
                )
            except ImportError:
                logger.warning(
                    "google-genai package is not installed. Provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[google]'",
                    _p.name,
                )

    os_runtime, os_subsystems = _build_os_runtime(runtime_context)
    subsystems = subsystems + os_subsystems

    for s in os_subsystems:
        await s.start()

    embedding_client = _build_embedding_client(snapshot)

    # One set of store connection pools for this container. Store activations are
    # constructed per catalog build and per step hop. Without this each of those
    # would open its own pool and abandon it. Registered as a subsystem so the
    # pools are disposed on shutdown.
    store_connections = StoreConnections()
    subsystems.append(store_connections)

    tool_catalog_builder = (
        ToolCatalogBuilder(
            loader=tool_registry,
            resource_repository=resource_repository,
            os_runtime=os_runtime,
            resource_access_policy=(
                access_rule_engine.for_resources() if access_rule_engine else None
            ),
            tool_access_policy=(
                access_rule_engine.for_tools() if access_rule_engine else None
            ),
            embedding_client=embedding_client,
            store_connections=store_connections,
            action_policy=action_authorization_registry.policy(),
            action_approval=action_authorization_registry.approval(),
            action_authorization_enabled=runtime_context.action_authorization.enabled,
            policy_timeout_seconds=runtime_context.action_authorization.policy_timeout_seconds,
            approval_timeout_seconds=runtime_context.action_authorization.approval_timeout_seconds,
        )
        if resource_repository is not None
        else None
    )
    if tool_catalog_builder is not None:
        _log_action_authorization(runtime_context.action_authorization)

    playbook_catalog_builder: PlaybookCatalogBuilder | None = None
    playbooks_path = snapshot.catalogs.playbooks.definitions_path
    if playbooks_path:
        _playbook_defs = load_playbook_definitions(Path(playbooks_path))
        playbook_catalog_builder = PlaybookCatalogBuilder(
            _playbook_defs,
            access_policy=access_rule_engine.for_playbooks() if access_rule_engine else None,
        )
        logger.info(
            "playbook.catalog.configured | path=%s definitions=%d",
            playbooks_path, len(_playbook_defs),
        )

    coordinator_factory = _build_coordinator_factory(runtime_context)
    # Ownership is enforced exactly when the deployment establishes identity.
    # A configuration-level statement, never a per-request one: a fallback for a
    # request that happens to establish no principal would leave the hole open
    # to anybody who omits a header.
    session_access = SessionAccess(
        coordinator_factory, enforced=principal_resolver is not None
    )
    metering = _build_metering(runtime_context)

    workflow_executor = StepMachineWorkflowExecutor(
        loader=step_registry.bind(
            embedding_client=embedding_client,
            resource_repository=resource_repository,
            tool_loader=tool_registry,
            # Every step-internal resource activation goes through this, so a
            # step cannot reach a resource the caller may not use. Absent only
            # where there is no resource repository at all. A deployment with
            # no persistence has no resources to activate, and a step that names
            # one then fails with a configuration error rather than reaching a
            # resource nothing could have authorized.
            resource_activator=(
                ResourceActivator(
                    resources=resource_repository,
                    loader=tool_registry,
                    access_policy=(
                        access_rule_engine.for_resources() if access_rule_engine else None
                    ),
                    store_connections=store_connections,
                )
                if resource_repository is not None
                else None
            ),
            # A step that activates a store writer must reach the shared pools.
            store_connections=store_connections,
            # A step that fans work out resolves the child workflow it names.
            workflow_repository=workflow_repository,
        ),
        model_clients=model_clients,
        tool_catalog_builder=tool_catalog_builder,
        playbook_catalog_builder=playbook_catalog_builder,
        model_access_policy=access_rule_engine.for_models() if access_rule_engine else None,
        session_access=session_access,
        os_runtime=os_runtime,
        metering_port=metering,
        # Present only where a durable queue is configured. Without one a step
        # has nothing to fan work out to, and says so instead of degrading.
        dispatcher=execution_dispatcher,
    )

    # One broker shared by the worker (publisher) and the gateway (subscriber):
    # streaming execution signals pass through the broker to the gateway and caller.
    # Which transport carries them is configuration, not code (see
    # _build_execution_stream_broker).
    execution_stream_broker, execution_stream_subsystem = _build_execution_stream_broker(
        runtime_context
    )
    if execution_stream_subsystem is not None:
        await execution_stream_subsystem.start()
        subsystems.append(execution_stream_subsystem)

    execution_worker: ExecutionWorker | None = None
    if runtime_context.execution.role in {"worker", "gateway+worker"}:
        if execution_repository is not None and workflow_repository is not None:
            worker_runtime = runtime_context.execution
            execution_worker = ExecutionWorker(
                repository=execution_repository,
                workflow_repository=workflow_repository,
                executor=workflow_executor,
                stream_broker=execution_stream_broker,
                settings=ExecutionWorkerSettings(
                    instance_name=worker_runtime.instance_name,
                    capabilities=worker_runtime.capabilities,
                    concurrency=worker_runtime.concurrency,
                    poll_interval_seconds=worker_runtime.poll_interval_seconds,
                    lease_seconds=worker_runtime.lease_seconds,
                    heartbeat_seconds=worker_runtime.heartbeat_seconds,
                    retry_base_seconds=worker_runtime.retry_base_seconds,
                    retry_max_seconds=worker_runtime.retry_max_seconds,
                ),
            )
            await execution_worker.start()
            subsystems.append(execution_worker)
        elif runtime_context.execution.role == "worker":
            raise RuntimeError(
                "execution.role=worker requires integrations.persistence.workflows.url"
            )

    http_protocol_adapters = (
        _build_http_adapters(runtime_context, model_clients)
        if runtime_context.execution.role != "worker"
        else []
    )

    inbound_adapters = (
        # Registered watchers first, then whatever an embedder registered by
        # hand. A worker claims what it is given and observes nothing, so a
        # worker-only node starts none of them.
        _build_watchers(runtime_context, execution_dispatcher)
        + inbound_adapter_registry.all()
        if runtime_context.execution.role != "worker"
        else []
    )
    for adapter in inbound_adapters:
        await adapter.start(
            executor=workflow_executor,
            repository=workflow_repository,
        )

    return Container(
        configuration_service=configuration_service,
        composition_service=composition_service,
        runtime_context=runtime_context,
        gateway_service=gateway_service,
        subsystems=subsystems,
        workflow_repository=workflow_repository,
        resource_repository=resource_repository,
        http_protocol_adapters=http_protocol_adapters,
        inbound_adapters=inbound_adapters,
        workflow_executor=workflow_executor,
        coordinator_factory=coordinator_factory,
        engine=_engine,
        license=license,
        metering=metering,
        execution_dispatcher=execution_dispatcher,
        execution_artifact_store=execution_artifact_store,
        execution_worker=execution_worker,
        execution_stream_broker=execution_stream_broker,
    )
