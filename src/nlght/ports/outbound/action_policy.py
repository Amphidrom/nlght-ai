# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define action policy and human approval ports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nlght.core.tools.action import (
    ActionDecisionResult,
    ActionReview,
    ApprovalResult,
)


@runtime_checkable
class ActionPolicy(Protocol):
    """Decides one concrete, already-bound action.

    The decision logic is the deployment's: deterministic rules, a reviewing
    model, or both. The framework enforces only that a decision is made before
    execution, that it belongs to the reviewed snapshot, and that an exception,
    a timeout or a result that is not an ``ActionDecisionResult`` denies
    (ADR-0072).
    """

    async def evaluate(self, review: ActionReview) -> ActionDecisionResult: ...


@runtime_checkable
class ActionApprovalPort(Protocol):
    """Asks a human or embedding application about one confirmable action.

    The port is called only after policy returned ``REQUIRE_CONFIRMATION``.  It
    is not an override mechanism and never receives a hard-denied action.
    """

    async def approve(self, review: ActionReview) -> ApprovalResult: ...
