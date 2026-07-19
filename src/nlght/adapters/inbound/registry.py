# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging

from nlght.ports.inbound.http_protocol_adapter import HttpProtocolAdapter
from nlght.ports.inbound.inbound_adapter import InboundAdapter

logger = logging.getLogger(__name__)


class InboundAdapterRegistry:
    """Registry for non-HTTP inbound adapters.

    Adapters are registered as pre-configured instances (unlike step/tool
    registries which register classes). Call ``register()`` before ``serve()``.
    """

    def __init__(self) -> None:
        self._adapters: list[InboundAdapter] = []

    def register(self, adapter: InboundAdapter) -> None:
        self._adapters.append(adapter)
        logger.debug("inbound_adapter.registered | type=%s", type(adapter).__name__)

    def all(self) -> list[InboundAdapter]:
        return list(self._adapters)


class HttpAdapterRegistry:
    """Registry for custom HTTP protocol adapters.

    Use this to register additional FastAPI routers (custom trigger endpoints,
    webhooks, etc.) without forking ``create_app()``.
    Call ``register()`` before ``serve()``.
    """

    def __init__(self) -> None:
        self._adapters: list[HttpProtocolAdapter] = []

    def register(self, adapter: HttpProtocolAdapter) -> None:
        self._adapters.append(adapter)
        logger.debug("http_adapter.registered | type=%s", type(adapter).__name__)

    def all(self) -> list[HttpProtocolAdapter]:
        return list(self._adapters)


inbound_adapter_registry: InboundAdapterRegistry = InboundAdapterRegistry()
http_adapter_registry: HttpAdapterRegistry = HttpAdapterRegistry()
