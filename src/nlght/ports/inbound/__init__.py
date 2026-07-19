# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Inbound port protocols — interfaces implemented by HTTP/transport adapters."""

from nlght.ports.inbound.http_protocol_adapter import HttpProtocolAdapter
from nlght.ports.inbound.inbound_adapter import InboundAdapter

__all__ = ["HttpProtocolAdapter", "InboundAdapter"]
