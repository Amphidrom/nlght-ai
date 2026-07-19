# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nlght.ports.outbound.workflow_executor import WorkflowExecutor
    from nlght.ports.outbound.workflow_repository import WorkflowRepository


class InboundAdapter(Protocol):
    """Inbound port for non-HTTP trigger sources.

    Implement this protocol to drive workflows from any external source —
    polling loops, message queues, cron schedulers, etc.

    The adapter receives ``executor`` and ``repository`` on ``start()``.
    It is responsible for managing its own background tasks and resources.
    ``stop()`` must cancel/clean up everything started in ``start()``.
    """

    async def start(
        self,
        executor: WorkflowExecutor,
        repository: WorkflowRepository | None,
    ) -> None: ...

    async def stop(self) -> None: ...
