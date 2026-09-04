# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Ports for deterministic action authorization and concrete approval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nlght.core.tools.action import (
    ActionDecisionResult,
    ActionExecutionContext,
    ApprovalResult,
    ConcreteAction,
    PendingAction,
)


@runtime_checkable
class ActionPolicy(Protocol):
    """Decides one concrete, already-bound action outside the model."""

    async def evaluate(
        self,
        action: ConcreteAction,
        context: ActionExecutionContext,
    ) -> ActionDecisionResult: ...


@runtime_checkable
class ActionApprovalPort(Protocol):
    """Asks a human or embedding application about one confirmable action.

    The port is called only after policy returned ``REQUIRE_CONFIRMATION``.  It
    is not an override mechanism and never receives a hard-denied action.
    """

    async def approve(self, action: PendingAction) -> ApprovalResult: ...
