# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from nlght.core.config.runtime_context import (
    GatewayRegistry,
    HiveMindProviderRegistry,
    ModelProviderRegistry,
    ProtocolAdapterRegistry,
)
from nlght.core.config.snapshot import PlatformConfigSnapshot


class Item:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


def test_registries_filter_enabled_items() -> None:
    protocol_registry = ProtocolAdapterRegistry([Item(True), Item(False)])
    gateway_registry = GatewayRegistry([Item(True), Item(False)])
    hive_registry = HiveMindProviderRegistry([Item(True), Item(False)])

    assert len(protocol_registry.all()) == 2
    assert len(protocol_registry.enabled()) == 1
    assert len(gateway_registry.enabled()) == 1
    assert len(hive_registry.enabled()) == 1


def test_model_provider_registry_returns_all_items() -> None:
    registry = ModelProviderRegistry([Item(True), Item(False)])

    assert len(registry.all()) == 2


def test_platform_snapshot_defaults_are_empty() -> None:
    snapshot = PlatformConfigSnapshot()

    assert snapshot.catalogs.playbooks.definitions_path is None
    assert snapshot.protocol_adapters == []
    assert snapshot.gateways == []
    assert snapshot.os_runtime is None
    assert snapshot.integrations.model_providers == []
