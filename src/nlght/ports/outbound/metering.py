# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext


@runtime_checkable
class MeteringPort(Protocol):
    """Outbound port for usage metering — Prometheus-compatible metrics.

    All four methods are always called (even when tokens=0) so counters increment
    on every request/step.  Implementations must be non-blocking; failures must
    not propagate to callers.

    Absence (None) is valid — no metering is performed.
    """

    async def record_tokens(
        self,
        *,
        caller: RequestContext,
        session_key: str | None,
        workflow: str,
        step: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None: ...

    async def record_request(
        self,
        *,
        caller: RequestContext,
        session_key: str | None,
        workflow: str,
        status: str,
        duration_ms: float,
    ) -> None: ...

    async def record_step(
        self,
        *,
        workflow: str,
        step: str,
        status: str,
        duration_ms: float,
    ) -> None: ...

    async def emit(
        self,
        *,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None: ...
