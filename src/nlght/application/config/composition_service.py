# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any

from nlght.core.config.runtime_context import (
    AnthropicModelProviderRuntime,
    DockerOsRuntimeSubsystemRuntime,
    GatewayRegistry,
    GenericModelProviderRuntime,
    GenericRuntimeSubsystemRuntime,
    GoogleModelProviderRuntime,
    HiveMindProviderRegistry,
    HiveMindProviderRuntime,
    HttpGatewaySubsystemRuntime,
    LicensingRuntime,
    LocalOsRuntimeSubsystemRuntime,
    ModelProviderRegistry,
    OllamaLocalModelProviderRuntime,
    OllamaModelProviderRuntime,
    OpenAICloudModelProviderRuntime,
    PersistenceSubsystemRuntime,
    ProtocolAdapterRegistry,
    ProtocolDetectorRuntime,
    RuntimeContext,
)
from nlght.core.config.snapshot import PlatformConfigSnapshot


class CompositionService:
    def compose(self, snapshot: PlatformConfigSnapshot) -> RuntimeContext:
        protocol_detector_runtimes = [
            ProtocolDetectorRuntime(
                name=adapter.name,
                kind=adapter.kind,
                enabled=adapter.enabled,
                config=adapter.config,
            )
            for adapter in snapshot.protocol_adapters
            if adapter.enabled
        ]

        hive_mind_provider_config = snapshot.integrations.persistence.hive_mind.provider
        hive_mind_provider = (
            HiveMindProviderRuntime(
                name=hive_mind_provider_config.name,
                kind=hive_mind_provider_config.kind,
                enabled=hive_mind_provider_config.enabled,
                session_ttl_seconds=int(
                    hive_mind_provider_config.config.get("session_ttl_seconds", 3600)
                ),
                config=hive_mind_provider_config.config,
            )
            if hive_mind_provider_config is not None
            else None
        )

        # ── OS Runtime (singular) ────────────────────────────────────────────
        os_runtime_cfg = snapshot.os_runtime
        os_runtime: LocalOsRuntimeSubsystemRuntime | DockerOsRuntimeSubsystemRuntime | None = None
        if os_runtime_cfg is not None and os_runtime_cfg.enabled:
            if os_runtime_cfg.kind == "local":
                os_runtime = LocalOsRuntimeSubsystemRuntime(
                    name="os-runtime",
                    workdir=str(os_runtime_cfg.config.get("workdir", ".")),
                    workspace_path=str(os_runtime_cfg.config.get("workspace_path", "./workspaces")),
                )
            elif os_runtime_cfg.kind == "docker":
                os_runtime = DockerOsRuntimeSubsystemRuntime(
                    name="os-runtime",
                    base_image=str(os_runtime_cfg.config.get("base_image", "python:3.12-slim")),
                    workspace_path=str(os_runtime_cfg.config["workspace_path"])
                    if os_runtime_cfg.config.get("workspace_path") else None,
                    extra_hosts=list(os_runtime_cfg.config.get("extra_hosts", [])),
                )

        # ── Persistence (from integrations.persistence.workflows) ────────────
        workflows_cfg = snapshot.integrations.persistence.workflows
        persistence: PersistenceSubsystemRuntime | None = None
        if workflows_cfg.url:
            persistence = PersistenceSubsystemRuntime(
                name="persistence",
                backend=workflows_cfg.backend,
                url=workflows_cfg.url,
                hive_mind_provider_name=(
                    hive_mind_provider.name if hive_mind_provider else None
                ),
            )

        # ── Gateways ─────────────────────────────────────────────────────────
        gateways: list[Any] = []
        for gw in snapshot.gateways:
            if not gw.enabled:
                continue
            if gw.kind == "http":
                gateways.append(
                    HttpGatewaySubsystemRuntime(
                        name=gw.name,
                        host=str(gw.config.get("host", "0.0.0.0")),
                        port=int(gw.config.get("port", 8000)),
                    )
                )
            else:
                gateways.append(
                    GenericRuntimeSubsystemRuntime(
                        name=gw.name,
                        kind=gw.kind,
                        config=gw.config,
                    )
                )

        # ── Model Providers ──────────────────────────────────────────────────
        model_providers: list[Any] = []
        for provider in snapshot.integrations.model_providers:
            if not provider.enabled:
                continue
            if provider.kind in ("ollama", "ollama-cloud", "ollama-local"):
                headers = provider.config.get("headers", {})
                if not isinstance(headers, dict):
                    headers = {}
                # Cloud kinds ("ollama"/"ollama-cloud") default to the public
                # Ollama Cloud endpoint so an API key alone is enough; only the
                # local daemon defaults to localhost. Any base_url in config wins.
                default_base = "http://localhost:11434" if provider.kind == "ollama-local" else "https://ollama.com"
                kwargs = dict(
                    name=provider.name,
                    base_url=str(provider.config.get("base_url", default_base)),
                    default_model=str(provider.config.get("default_model", "")),
                    api_key=str(provider.config.get("api_key", "")),
                    headers={str(k): str(v) for k, v in headers.items()},
                    request_timeout_s=(
                        float(provider.config["request_timeout_s"])
                        if "request_timeout_s" in provider.config
                        else None
                    ),
                    stream_connect_timeout_s=(
                        float(provider.config["stream_connect_timeout_s"])
                        if "stream_connect_timeout_s" in provider.config
                        else None
                    ),
                    stream_read_timeout_s=(
                        float(provider.config["stream_read_timeout_s"])
                        if "stream_read_timeout_s" in provider.config
                        else None
                    ),
                )
                runtime_cls = (
                    OllamaLocalModelProviderRuntime
                    if provider.kind == "ollama-local"
                    else OllamaModelProviderRuntime
                )
                # mypy can't decompose a dict[str, <union of value types>] into
                # per-field-typed kwargs against the dataclass constructor --
                # each value above is already correctly coerced (str/float/dict).
                model_providers.append(runtime_cls(**kwargs))  # type: ignore[arg-type]
            elif provider.kind == "anthropic":
                model_providers.append(
                    AnthropicModelProviderRuntime(
                        name=provider.name,
                        api_key=str(provider.config.get("api_key", "")),
                        default_model=str(provider.config.get("default_model", "claude-sonnet-4-6")),
                    )
                )
            elif provider.kind == "openai":
                model_providers.append(
                    OpenAICloudModelProviderRuntime(
                        name=provider.name,
                        api_key=str(provider.config.get("api_key", "")),
                        default_model=str(provider.config.get("default_model", "gpt-4o")),
                        base_url=str(provider.config.get("base_url", "")),
                    )
                )
            elif provider.kind == "google":
                model_providers.append(
                    GoogleModelProviderRuntime(
                        name=provider.name,
                        api_key=str(provider.config.get("api_key", "")),
                        default_model=str(provider.config.get("default_model", "gemini-3.5-flash")),
                    )
                )
            else:
                model_providers.append(
                    GenericModelProviderRuntime(
                        name=provider.name,
                        kind=provider.kind,
                        config=provider.config,
                    )
                )

        hive_mind_providers = [hive_mind_provider] if hive_mind_provider else []

        return RuntimeContext(
            protocol_adapter_registry=ProtocolAdapterRegistry(snapshot.protocol_adapters),
            gateway_registry=GatewayRegistry(snapshot.gateways),
            model_provider_registry=ModelProviderRegistry(
                snapshot.integrations.model_providers
            ),
            hive_mind_provider_registry=HiveMindProviderRegistry(hive_mind_providers),
            protocol_detector_runtimes=protocol_detector_runtimes,
            gateways=gateways,
            os_runtime=os_runtime,
            persistence=persistence,
            model_providers=model_providers,
            hive_mind_provider=hive_mind_provider,
            licensing=LicensingRuntime(
                license_key=snapshot.licensing.license_key,
            ),
        )
