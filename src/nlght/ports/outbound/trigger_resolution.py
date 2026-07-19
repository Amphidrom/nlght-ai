# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.entry.context import RequestContext
from nlght.core.protocol.protocol import DetectedProtocol
from nlght.core.trigger.trigger import Trigger


class TriggerResolver(Protocol):
    async def resolve(
        self,
        protocol: DetectedProtocol,
        context: RequestContext,
        raw_body: bytes,
        session_key: str | None = None,
    ) -> Trigger:
        raise NotImplementedError