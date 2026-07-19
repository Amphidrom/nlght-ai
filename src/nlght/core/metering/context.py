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
    """Per-request metering context — carries the port and all label values.

    Created by the executor in ``_bind_llm()`` for each step and passed
    into the model backend.  Not stored on the backend (shared across
    requests) — always passed per-call.
    """

    port: MeteringPort
    caller: RequestContext
    session_key: str | None
    workflow: str
    step: str
    provider: str
