# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.ports.outbound.metering import MeteringPort


@dataclass(frozen=True)
class MeteringContext:
    """Per-request metering context carrying the port and label values.

    Created by the executor in ``_bind_llm()`` for each step and passed
    into the model backend. The backend is shared across requests, so the
    context is passed per call and is not stored on the backend.
    """

    port: MeteringPort
    caller: RequestContext
    session_key: str | None
    workflow: str
    step: str
    provider: str
