# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nlght.adapters.outbound.config.file.mapper import to_snapshot
from nlght.adapters.outbound.config.file.models import (
    GatewayModel,
    HiveMindProviderModel,
    LicensingConfigModel,
    ModelProviderModel,
    OsRuntimeModel,
    PlatformFileConfigModel,
    ProtocolAdapterModel,
    WorkflowsPersistenceModel,
)
from nlght.core.config.snapshot import LoggingConfig, PlatformConfigSnapshot
from nlght.ports.outbound.configuration_source import ConfigurationSource


class FileConfigurationSource(ConfigurationSource):
    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)

    async def load(self) -> PlatformConfigSnapshot:
        if not self._config_path.exists():
            return to_snapshot(PlatformFileConfigModel())

        raw = yaml.safe_load(self._config_path.read_text()) or {}

        catalogs = raw.get("catalogs", {}) or {}
        integrations = raw.get("integrations", {}) or {}
        persistence = integrations.get("persistence", {}) or {}
        hive_mind = persistence.get("hive_mind", {}) or {}

        # ── Singular os_runtime block ────────────────────────────────────────
        os_runtime_raw = raw.get("os_runtime")
        os_runtime_model: OsRuntimeModel | None = None
        if os_runtime_raw and isinstance(os_runtime_raw, dict):
            os_runtime_model = OsRuntimeModel(
                kind=str(os_runtime_raw.get("kind", "local")),
                enabled=bool(os_runtime_raw.get("enabled", True)),
                config=dict(os_runtime_raw.get("config", {}) or {}),
            )

        # ── integrations.persistence.workflows ──────────────────────────────
        workflows_raw = persistence.get("workflows", {}) or {}
        persistence_workflows = WorkflowsPersistenceModel(
            backend=str(workflows_raw.get("backend", "postgres")),
            url=str(workflows_raw.get("url", "")),
        )

        hive_mind_provider = self._extract_single_hive_mind_provider(hive_mind)

        licensing_raw = raw.get("licensing", {}) or {}
        licensing = LicensingConfigModel(
            license_key=str(licensing_raw["license_key"]) if licensing_raw.get("license_key") else None,
        )

        logging_raw = raw.get("logging", {}) or {}
        logging_level: dict[str, str] = {
            str(k): str(v)
            for k, v in (logging_raw.get("level", {}) or {}).items()
        }

        playbooks_raw = catalogs.get("playbooks") or {}
        catalogs_playbooks_definitions_path = (
            str(playbooks_raw.get("definitions_path") or playbooks_raw.get("path")).strip()
            if isinstance(playbooks_raw, dict) and (playbooks_raw.get("definitions_path") or playbooks_raw.get("path"))
            else None
        )

        model = PlatformFileConfigModel(
            catalogs_playbooks_definitions_path=catalogs_playbooks_definitions_path,
            licensing=licensing,
            protocol_adapters=[
                self._protocol_adapter(item)
                for item in (raw.get("protocol_adapters", []) or [])
            ],
            gateways=[
                self._gateway(item)
                for item in (raw.get("gateways", []) or [])
            ],
            os_runtime=os_runtime_model,
            model_providers=[
                self._model_provider(item)
                for item in (integrations.get("model_providers", []) or [])
            ],
            persistence_workflows=persistence_workflows,
            persistence_hive_mind_provider=hive_mind_provider,
        )

        snapshot = to_snapshot(model)
        if logging_level:
            snapshot.logging = LoggingConfig(level=logging_level)
        return snapshot

    @staticmethod
    def _extract_single_hive_mind_provider(
        hive_mind: dict[str, Any],
    ) -> HiveMindProviderModel | None:
        if "provider" in hive_mind and hive_mind["provider"]:
            return FileConfigurationSource._hive_mind_provider(
                dict(hive_mind.get("provider", {}) or {})
            )

        providers = list(hive_mind.get("providers", []) or [])
        if len(providers) > 1:
            raise ValueError(
                "integrations.persistence.hive_mind supports only one provider. "
                "Use 'provider' (preferred) or a single-item legacy 'providers' list."
            )
        if len(providers) == 1:
            return FileConfigurationSource._hive_mind_provider(providers[0])
        return None

    @staticmethod
    def _protocol_adapter(raw: dict[str, Any]) -> ProtocolAdapterModel:
        return ProtocolAdapterModel(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            enabled=bool(raw.get("enabled", True)),
            config=dict(raw.get("config", {}) or {}),
        )

    @staticmethod
    def _gateway(raw: dict[str, Any]) -> GatewayModel:
        return GatewayModel(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            enabled=bool(raw.get("enabled", True)),
            config=dict(raw.get("config", {}) or {}),
        )

    @staticmethod
    def _model_provider(raw: dict[str, Any]) -> ModelProviderModel:
        return ModelProviderModel(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            enabled=bool(raw.get("enabled", True)),
            config=dict(raw.get("config", {}) or {}),
        )

    @staticmethod
    def _hive_mind_provider(raw: dict[str, Any]) -> HiveMindProviderModel:
        return HiveMindProviderModel(
            name=str(raw.get("name", "")),
            kind=str(raw.get("kind", "")),
            enabled=bool(raw.get("enabled", True)),
            config=dict(raw.get("config", {}) or {}),
        )
