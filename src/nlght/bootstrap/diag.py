# Copyright (c) 2026 Amphidrom GmbH and Contributors. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio

from nlght.bootstrap.wiring import build_container
from nlght.core.config.runtime_context import (
    DockerOsRuntimeSubsystemRuntime,
    GenericModelProviderRuntime,
    GenericRuntimeSubsystemRuntime,
    HttpGatewaySubsystemRuntime,
    LocalOsRuntimeSubsystemRuntime,
    OllamaModelProviderRuntime,
)


async def main() -> None:
    container = await build_container(".config/platform.yaml")

    snapshot = await container.configuration_service.load_snapshot()
    runtime_context = container.runtime_context

    print("Loaded config snapshot")
    print(snapshot)
    print()

    print("Composed runtime context")
    print(
        f"Protocol adapters: configured={len(runtime_context.protocol_adapter_registry.all())} "
        f"enabled={len(runtime_context.protocol_adapter_registry.enabled())}"
    )
    print(
        f"Gateways: configured={len(runtime_context.gateway_registry.all())} "
        f"enabled={len(runtime_context.gateway_registry.enabled())}"
    )
    print(
        f"Model providers: configured={len(runtime_context.model_provider_registry.all())}"
    )
    print()

    print("Initialized gateways:")
    for gateway in runtime_context.gateways:
        if isinstance(gateway, HttpGatewaySubsystemRuntime):
            print(
                f"- HttpGatewaySubsystemRuntime(name={gateway.name}, host={gateway.host}, port={gateway.port})"
            )
        elif isinstance(gateway, GenericRuntimeSubsystemRuntime):
            print(
                f"- GenericRuntimeSubsystemRuntime(name={gateway.name}, kind={gateway.kind}, config={gateway.config})"
            )
        else:
            print(f"- {gateway}")
    print()

    os_rt = runtime_context.os_runtime
    if os_rt is not None:
        print("OS runtime:")
        if isinstance(os_rt, LocalOsRuntimeSubsystemRuntime):
            print(f"- LocalOsRuntimeSubsystemRuntime(name={os_rt.name}, workdir={os_rt.workdir})")
        elif isinstance(os_rt, DockerOsRuntimeSubsystemRuntime):
            print(f"- DockerOsRuntimeSubsystemRuntime(name={os_rt.name}, base_image={os_rt.base_image})")
        else:
            print(f"- {os_rt}")
        print()

    persistence = runtime_context.persistence
    if persistence is not None:
        print("Persistence:")
        print(
            f"- PersistenceSubsystemRuntime(name={persistence.name}, backend={persistence.backend}, url={persistence.url})"
        )
        print()

    print("Initialized model providers:")
    for provider in runtime_context.model_providers:
        if isinstance(provider, OllamaModelProviderRuntime):
            print(
                f"- OllamaModelProviderRuntime(name={provider.name}, base_url={provider.base_url}, default_model={provider.default_model})"
            )
        elif isinstance(provider, GenericModelProviderRuntime):
            print(
                f"- GenericModelProviderRuntime(name={provider.name}, kind={provider.kind}, config={provider.config})"
            )
        else:
            print(f"- {provider}")


if __name__ == "__main__":
    asyncio.run(main())