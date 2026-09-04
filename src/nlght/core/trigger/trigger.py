# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nlght.core.entry.context import RequestContext
from nlght.core.protocol.protocol import ProtocolKind


class TriggerKind(StrEnum):
    HTTP_REQUEST = "http.request"
    #: Anything an InboundAdapter drives — a watcher, a poll loop, a queue
    #: consumer. Not tied to one kind of workload.
    INBOUND_EVENT = "inbound.event"
    MODEL_REQUEST = "model.request"
    MODELS_DISCOVERY = "models.discovery"


@dataclass(slots=True, frozen=True)
class Trigger:
    kind: TriggerKind
    protocol: ProtocolKind
    operation: str
    payload: dict[str, Any]
    context: RequestContext
    stream: bool = False
    session_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
