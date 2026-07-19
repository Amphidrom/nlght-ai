# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nlght.bootstrap import wiring
from nlght.core.config.runtime_context import (
    GatewayRegistry,
    GenericRuntimeSubsystemRuntime,
    HiveMindProviderRegistry,
    HiveMindProviderRuntime,
    LicensingRuntime,
    LocalOsRuntimeSubsystemRuntime,
    ModelProviderRegistry,
    PersistenceSubsystemRuntime,
    ProtocolAdapterRegistry,
    ProtocolDetectorRuntime,
    RuntimeContext,
)
from nlght.core.config.snapshot import PlatformConfigSnapshot


def test_build_protocol_detectors_maps_known_kinds() -> None:
    context = RuntimeContext(
        protocol_adapter_registry=cast(Any, None),
        gateway_registry=cast(Any, None),
        model_provider_registry=cast(Any, None),
        hive_mind_provider_registry=cast(Any, None),
        protocol_detector_runtimes=[
            ProtocolDetectorRuntime(name="a", kind="openai", enabled=True, config={"base_path": "/v1"}),
            ProtocolDetectorRuntime(name="b", kind="generic_json", enabled=True, config={}),
            ProtocolDetectorRuntime(name="c", kind="unknown", enabled=True, config={}),
        ],
        gateways=[],
        os_runtime=None,
        persistence=None,
        model_providers=[],
        hive_mind_provider=None,
    )

    detectors = wiring._build_protocol_detectors(context)

    assert [detector.__class__.__name__ for detector in detectors] == [
        "OpenAIProtocolDetector",
        "GenericJsonProtocolDetector",
    ]


def test_build_container_wires_services(monkeypatch) -> None:
    class StubConfigurationService:
        async def load_snapshot(self) -> PlatformConfigSnapshot:
            return PlatformConfigSnapshot()

    class StubCompositionService:
        def compose(self, snapshot: PlatformConfigSnapshot) -> RuntimeContext:
            assert isinstance(snapshot, PlatformConfigSnapshot)
            return RuntimeContext(
                protocol_adapter_registry=cast(Any, None),
                gateway_registry=cast(Any, None),
                model_provider_registry=cast(Any, None),
                hive_mind_provider_registry=cast(Any, None),
                protocol_detector_runtimes=[
                    ProtocolDetectorRuntime(name="openai", kind="openai", enabled=True, config={"base_path": "/v1"})
                ],
                gateways=[],
                os_runtime=None,
                persistence=None,
                model_providers=[],
                hive_mind_provider=None,
            )

    class StubFileConfigurationSource:
        def __init__(self, config_path: str) -> None:
            self.config_path = config_path

    from integration._helpers import StubLicenseAdapter

    monkeypatch.setattr(wiring, "FileConfigurationSource", StubFileConfigurationSource)
    monkeypatch.setattr(wiring, "ConfigurationService", lambda source: StubConfigurationService())
    monkeypatch.setattr(wiring, "CompositionService", lambda: StubCompositionService())
    monkeypatch.setattr(wiring, "_build_license_service", lambda _ctx: StubLicenseAdapter())

    container = asyncio.run(wiring.build_container("config/platform.yaml"))

    assert isinstance(container.gateway_service, wiring.GatewayService)


def _context(**kwargs: object) -> RuntimeContext:
    defaults = dict(
        protocol_adapter_registry=ProtocolAdapterRegistry(),
        gateway_registry=GatewayRegistry(),
        model_provider_registry=ModelProviderRegistry(),
        hive_mind_provider_registry=HiveMindProviderRegistry(),
        protocol_detector_runtimes=[],
        gateways=[],
        os_runtime=None,
        persistence=None,
        model_providers=[],
        hive_mind_provider=None,
        licensing=LicensingRuntime(),
    )
    defaults.update(kwargs)
    return RuntimeContext(**defaults)


def test_build_single_resolver_supports_all_configured_shapes() -> None:
    assert wiring._build_single_resolver({"resolver": "header", "key": "x-session"}).__class__.__name__ == (
        "HeaderSessionKeyResolver"
    )
    assert wiring._build_single_resolver({"resolver": "query", "parameter": "session"}).__class__.__name__ == (
        "QueryParamSessionKeyResolver"
    )
    assert wiring._build_single_resolver({"resolver": "body", "key": "meta.session"}).__class__.__name__ == (
        "BodyParameterSessionKeyResolver"
    )
    composite = wiring._build_single_resolver(
        {
            "resolver": "composite",
            "resolvers": [
                {"resolver": "header", "key": "x-session"},
                {"resolver": "query", "parameter": "session"},
                {"resolver": "unknown"},
            ],
        }
    )
    assert composite.__class__.__name__ == "CompositeSessionKeyResolver"
    assert wiring._build_single_resolver({"resolver": "header", "key": 123}) is None
    assert wiring._build_single_resolver({"resolver": "composite", "resolvers": [{"resolver": "unknown"}]}) is None


def test_build_http_adapters_orders_specific_routes_before_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wiring.http_adapter_registry, "all", lambda: [SimpleNamespace(kind="registered")])
    context = _context(
        protocol_detector_runtimes=[
            ProtocolDetectorRuntime(
                name="openai",
                kind="openai",
                enabled=True,
                config={"base_path": "/v1", "model_provider": "m", "workflows": {"gpt": "chat"}},
            ),
            ProtocolDetectorRuntime(
                name="ollama",
                kind="ollama-api",
                enabled=True,
                config={"model_provider": "m", "workflows": {"llama": "chat"}},
            ),
            ProtocolDetectorRuntime(name="json", kind="generic_json", enabled=True, config={}),
            ProtocolDetectorRuntime(name="unknown", kind="unknown", enabled=True, config={}),
        ],
    )

    adapters = wiring._build_http_adapters(context, {"m": object()})

    assert [adapter.__class__.__name__ for adapter in adapters[:3]] == [
        "OpenAIHttpProtocolAdapter",
        "OllamaHttpProtocolAdapter",
        "GenericJsonHttpProtocolAdapter",
    ]
    assert adapters[-1].kind == "registered"


def test_license_metering_and_subsystem_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Keygen:
        def __init__(self, key: str) -> None:
            self.key = key

        def context_metadata(self) -> dict[str, str]:
            return {"license": "ok"}

    class _Prometheus:
        def __init__(self, *, endpoint: str) -> None:
            self.endpoint = endpoint

    monkeypatch.setattr(wiring, "KeygenLicenseAdapter", _Keygen)
    module = SimpleNamespace(PrometheusMeteringAdapter=_Prometheus)
    monkeypatch.setitem(sys.modules, "nlght.adapters.outbound.metering.prometheus", module)

    license_ = wiring._build_license_service(
        _context(licensing=LicensingRuntime(license_key="key"))
    )
    assert license_.key == "key"
    with pytest.raises(RuntimeError, match="No license configured"):
        wiring._build_license_service(_context())

    metering = wiring._build_metering(
        _context(
            gateways=[
                GenericRuntimeSubsystemRuntime(name="wrong", kind="other", config={}),
                GenericRuntimeSubsystemRuntime(name="metrics", kind="metering", config={"endpoint": "/m"}),
            ]
        )
    )
    assert metering.endpoint == "/m"
    assert wiring._build_metering(
        _context(gateways=[GenericRuntimeSubsystemRuntime(name="metrics", kind="metering", config={"backend": "unknown"})])
    ) is None

    subsystems = wiring._build_subsystems(
        _context(
            persistence=PersistenceSubsystemRuntime(
                name="db",
                backend="postgres",
                url="postgresql+asyncpg://u:p@host/db",
            )
        )
    )
    assert [subsystem.__class__.__name__ for subsystem in subsystems] == ["PostgresPersistenceSubsystem"]


def test_os_runtime_and_coordinator_factory_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    local, subsystems = wiring._build_os_runtime(
        _context(os_runtime=LocalOsRuntimeSubsystemRuntime(name="local", workdir=".", workspace_path="./ws"))
    )
    assert local.__class__.__name__ == "LocalOsRuntime"
    assert subsystems == []
    assert wiring._build_os_runtime(_context()) == (None, [])

    factory = wiring._build_coordinator_factory(_context())
    assert factory.__class__.__name__ == "SimpleStoreCoordinatorFactory"

    class _FakeHiveMindFactory:
        def __init__(self, *, base_dir: str) -> None:
            self.base_dir = base_dir

    module = SimpleNamespace(HiveMindStoreCoordinatorFactory=_FakeHiveMindFactory)
    monkeypatch.setitem(sys.modules, "nlght.adapters.outbound.hive_mind.coordinator", module)
    hive_mind = wiring._build_coordinator_factory(
        _context(hive_mind_provider=HiveMindProviderRuntime(name="hm", kind="filesystem", enabled=True)),
    )
    assert isinstance(hive_mind, _FakeHiveMindFactory)
    assert hive_mind.base_dir == "./sessions"

    monkeypatch.setitem(sys.modules, "nlght.adapters.outbound.hive_mind.coordinator", None)
    fallback = wiring._build_coordinator_factory(
        _context(hive_mind_provider=HiveMindProviderRuntime(name="hm", kind="filesystem", enabled=True)),
    )
    assert fallback.__class__.__name__ == "SimpleStoreCoordinatorFactory"
