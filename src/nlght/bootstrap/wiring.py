# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx2
from sqlalchemy.ext.asyncio import AsyncEngine

from nlght.adapters.inbound.http.generic_json_adapter import GenericJsonHttpProtocolAdapter
from nlght.adapters.inbound.http.ollama_adapter import OllamaHttpProtocolAdapter
from nlght.adapters.inbound.http.openai_adapter import OpenAIHttpProtocolAdapter
from nlght.adapters.inbound.registry import http_adapter_registry, inbound_adapter_registry
from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.adapters.outbound.config.file.source import FileConfigurationSource
from nlght.adapters.outbound.licensing.keygen_license_adapter import KeygenLicenseAdapter
from nlght.adapters.outbound.model import OllamaClient, OllamaCloudClient
from nlght.adapters.outbound.os_runtime.local import LocalOsRuntime
from nlght.adapters.outbound.persistence.access_rule_repository import SqlAlchemyAccessRuleRepository
from nlght.adapters.outbound.persistence.postgres import PostgresPersistenceSubsystem
from nlght.adapters.outbound.persistence.resource_repository import SqlAlchemyResourceRepository
from nlght.adapters.outbound.persistence.workflow_repository import SqlAlchemyWorkflowRepository
from nlght.adapters.outbound.playbooks.catalog import PlaybookCatalogBuilder
from nlght.adapters.outbound.playbooks.loader import load_playbook_definitions
from nlght.adapters.outbound.protocol.composite import CompositeProtocolDetector
from nlght.adapters.outbound.protocol.generic_json import GenericJsonProtocolDetector
from nlght.adapters.outbound.protocol.ollama import OllamaProtocolDetector
from nlght.adapters.outbound.protocol.openai import OpenAIProtocolDetector
from nlght.adapters.outbound.session.body import BodyParameterSessionKeyResolver
from nlght.adapters.outbound.session.composite import CompositeSessionKeyResolver
from nlght.adapters.outbound.session.header import HeaderSessionKeyResolver
from nlght.adapters.outbound.session.query import QueryParamSessionKeyResolver
from nlght.adapters.outbound.tools.catalog import ToolCatalogBuilder
from nlght.adapters.outbound.tools.registry import tool_registry
from nlght.adapters.outbound.trigger.default import DefaultTriggerResolver
from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.adapters.outbound.workflow.registry import step_registry
from nlght.application.config.composition_service import CompositionService
from nlght.application.config.configuration_service import ConfigurationService
from nlght.application.entry.gateway_service import GatewayService
from nlght.core.config.runtime_context import (
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
from nlght.ports.inbound.http_protocol_adapter import HttpProtocolAdapter
from nlght.ports.inbound.inbound_adapter import InboundAdapter
from nlght.ports.outbound.access_rule_repository import AccessRuleRepository
from nlght.ports.outbound.licensing import LicensingPort
from nlght.ports.outbound.metering import MeteringPort
from nlght.ports.outbound.os_runtime import OsRuntimeFactory
from nlght.ports.outbound.protocol_detection import ProtocolDetector
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.runtime_subsystem import SubsystemLifecycle
from nlght.ports.outbound.session_key_resolver import SessionKeyResolver
from nlght.ports.outbound.store_coordinator import StoreCoordinatorFactory
from nlght.ports.outbound.workflow_executor import WorkflowExecutor
from nlght.ports.outbound.workflow_repository import WorkflowRepository


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


def _build_metering(runtime_context: RuntimeContext) -> MeteringPort | None:
    """Instantiate a MeteringPort from the first enabled ``kind: metering`` gateway entry.

    Currently supports ``backend: prometheus`` which requires the [metrics] extra.
    Returns None if no metering gateway is configured or enabled.
    """
    import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    _log = _logging.getLogger(__name__)

    from nlght.core.config.runtime_context import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        GenericRuntimeSubsystemRuntime,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
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
                    PrometheusMeteringAdapter,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                )
                adapter = PrometheusMeteringAdapter(endpoint=endpoint)
                _log.info("metering.prometheus.configured | endpoint=%s", endpoint)
                return adapter
            except ImportError:
                _log.warning(
                    "prometheus-client not installed — metering subsystem disabled. "
                    "Install with: pip install 'nlght-ai[metrics]'"
                )
        else:
            _log.warning("metering.unknown_backend | backend=%s — skipping", backend)
    return None


def _build_license_service(runtime_context: RuntimeContext) -> LicensingPort:
    """Build the license service. A configured, valid license is required to start.

    The license is a compliance/activation requirement only — no capability
    check is derived from it. Feature gating hooks stay available for future
    enterprise features, but nothing is gated today.
    """
    import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    _log = _logging.getLogger(__name__)
    lic = runtime_context.licensing
    if lic and lic.license_key:
        adapter = KeygenLicenseAdapter(lic.license_key)
        _log.info("license.keygen | %s", adapter.context_metadata())
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
                DockerOsRuntimeFactory,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            )

            factory = DockerOsRuntimeFactory(
                base_image=runtime.base_image,
                workspace_path=runtime.workspace_path,
                extra_hosts=runtime.extra_hosts,
            )
            return factory, [factory]
        except ImportError:
            import logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

            logging.getLogger(__name__).warning(
                "docker package not installed — DockerOsRuntimeFactory unavailable. "
                "Install with: pip install 'nlght-ai[docker]'"
            )
            return None, []
    return None, []


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
    """Build HTTP protocol adapters in order: specific routes first, catch-all last."""
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
            generic_json = GenericJsonHttpProtocolAdapter(session_key_resolver=resolver)

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
    """Builds a single SessionKeyResolver from a config dict.

    Supported variants::

        resolver: header
        key: x-session-id

        resolver: query
        parameter: session_id

        resolver: body
        key: meta.session_id          # dot notation for nested fields

        resolver: composite
        resolvers:
          - resolver: header
            key: x-session-id
          - resolver: query
            parameter: session_id
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



def _build_repositories(
    subsystems: list[SubsystemLifecycle],
) -> tuple[WorkflowRepository | None, ResourceRepository | None, AccessRuleRepository | None]:
    for s in subsystems:
        if isinstance(s, PostgresPersistenceSubsystem):
            return (
                SqlAlchemyWorkflowRepository(s.engine),
                SqlAlchemyResourceRepository(s.engine),
                SqlAlchemyAccessRuleRepository(s.engine),
            )
    return None, None, None


def _build_coordinator_factory(runtime_context: RuntimeContext) -> StoreCoordinatorFactory:
    """Instantiate the correct StoreCoordinatorFactory based on hive_mind_provider kind.

    "simple" (or no provider configured) → SimpleStoreCoordinatorFactory (in-memory).
    "filesystem" → HiveMindStoreCoordinatorFactory (file-backed, requires the
    [hive-mind] extra). The choice depends only on configuration.
    """
    import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    _log = _logging.getLogger(__name__)

    provider = runtime_context.hive_mind_provider
    if provider is None or not isinstance(provider, HiveMindProviderRuntime):
        from nlght.adapters.outbound.hive_mind.simple import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            SimpleStoreCoordinatorFactory,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        )
        return SimpleStoreCoordinatorFactory()

    if provider.kind == "filesystem":
        try:
            from nlght.adapters.outbound.hive_mind.coordinator import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                HiveMindStoreCoordinatorFactory,
            )
            base_dir = str(provider.config.get("base_dir", "./sessions"))
            _log.info("coordinator_factory.hive_mind | base_dir=%s", base_dir)
            return HiveMindStoreCoordinatorFactory(base_dir=base_dir)
        except ImportError:
            _log.warning(
                "hive-mind extra not installed — falling back to SimpleStoreCoordinatorFactory. "
                "Install with: pip install 'nlght-ai[hive-mind]'"
            )

    from nlght.adapters.outbound.hive_mind.simple import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        SimpleStoreCoordinatorFactory,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    )
    _log.info("coordinator_factory.simple | ttl=%d", provider.session_ttl_seconds)
    return SimpleStoreCoordinatorFactory(session_ttl_seconds=provider.session_ttl_seconds)


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

    workflow_repository, resource_repository, access_rule_repository = _build_repositories(subsystems)

    # Rule-based access policies (ADR-0022) — active whenever persistence is
    # configured; an empty access_policies table restricts nothing.
    access_rule_engine = (
        AccessRuleEngine(access_rule_repository)
        if access_rule_repository is not None
        else None
    )

    protocol_detector = CompositeProtocolDetector(_build_protocol_detectors(runtime_context))
    trigger_resolver = DefaultTriggerResolver()

    gateway_service = GatewayService(
        protocol_detector=protocol_detector,
        trigger_resolver=trigger_resolver,
        workflow_repository=workflow_repository,
        resource_repository=resource_repository,
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
            )
        elif isinstance(_p, AnthropicModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.anthropic import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    AnthropicModelClient,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                )
                model_clients[_p.name] = AnthropicModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                )
            except ImportError:
                import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                _logging.getLogger(__name__).warning(
                    "anthropic package not installed — provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[anthropic]'",
                    _p.name,
                )
        elif isinstance(_p, OpenAICloudModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.openai_cloud import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    OpenAICloudModelClient,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                )
                model_clients[_p.name] = OpenAICloudModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                    base_url=_p.base_url,
                )
            except ImportError:
                import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                _logging.getLogger(__name__).warning(
                    "openai package not installed — provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[openai]'",
                    _p.name,
                )
        elif isinstance(_p, GoogleModelProviderRuntime):
            try:
                from nlght.adapters.outbound.model.google import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                    GoogleModelClient,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                )
                model_clients[_p.name] = GoogleModelClient(
                    api_key=_p.api_key,
                    default_model=_p.default_model,
                )
            except ImportError:
                import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                _logging.getLogger(__name__).warning(
                    "google-genai package not installed — provider '%s' skipped. "
                    "Install with: pip install 'nlght-ai[google]'",
                    _p.name,
                )

    os_runtime, os_subsystems = _build_os_runtime(runtime_context)
    subsystems = subsystems + os_subsystems

    for s in os_subsystems:
        await s.start()

    tool_catalog_builder = (
        ToolCatalogBuilder(
            loader=tool_registry,
            resource_repository=resource_repository,
            os_runtime=os_runtime,
            access_policy=access_rule_engine.for_tools() if access_rule_engine else None,
        )
        if resource_repository is not None
        else None
    )

    playbook_catalog_builder: PlaybookCatalogBuilder | None = None
    playbooks_path = snapshot.catalogs.playbooks.definitions_path
    if playbooks_path:
        import logging as _logging  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        from pathlib import Path as _Path  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        _playbook_defs = load_playbook_definitions(_Path(playbooks_path))
        playbook_catalog_builder = PlaybookCatalogBuilder(
            _playbook_defs,
            access_policy=access_rule_engine.for_playbooks() if access_rule_engine else None,
        )
        _logging.getLogger(__name__).info(
            "playbook.catalog.configured | path=%s definitions=%d",
            playbooks_path, len(_playbook_defs),
        )

    coordinator_factory = _build_coordinator_factory(runtime_context)
    metering = _build_metering(runtime_context)

    workflow_executor = StepMachineWorkflowExecutor(
        loader=step_registry,
        model_clients=model_clients,
        tool_catalog_builder=tool_catalog_builder,
        playbook_catalog_builder=playbook_catalog_builder,
        model_access_policy=access_rule_engine.for_models() if access_rule_engine else None,
        coordinator_factory=coordinator_factory,
        os_runtime=os_runtime,
        metering_port=metering,
    )

    http_protocol_adapters = _build_http_adapters(runtime_context, model_clients)

    inbound_adapters = inbound_adapter_registry.all()
    for adapter in inbound_adapters:
        await adapter.start(
            executor=workflow_executor,
            repository=workflow_repository,
        )

    _engine: AsyncEngine | None = None
    for _s in subsystems:
        if isinstance(_s, PostgresPersistenceSubsystem):
            _engine = _s.engine
            break

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
    )
