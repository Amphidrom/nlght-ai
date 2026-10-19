# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Built-in deterministic action policy.

The policy reads only typed action facts and runtime capabilities.  It never
parses command text, prompt text, tool descriptions, or claimed provenance.

Every risk dimension is evaluated independently and the authorization is the
strictest of the four results. That is deliberate: an ordered `if` chain gives
    the same answers today, but it makes the order itself load-bearing. The first
branch that returns hides every later one, so a new side effect could be
silently covered by an earlier `ALLOW` after an innocent-looking reorder. With a
maximum there is no order to get wrong (ADR-0069).
"""

from __future__ import annotations

from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionReview,
    ConcreteAction,
    DataEgressClass,
    ExecutionClass,
    FilesystemCapability,
    NetworkCapability,
    ProcessCapability,
    ScopeClass,
    SecretsCapability,
    SideEffectClass,
)


def _result(
    decision: ActionDecision,
    reason: ActionDecisionReason,
) -> ActionDecisionResult:
    return ActionDecisionResult(decision=decision, reason=reason)


#: Severity is written out rather than taken from the enum's values, so the
#: composition does not quietly depend on how the members happen to be defined.
_SEVERITY: dict[ActionDecision, int] = {
    ActionDecision.ALLOW: 0,
    ActionDecision.REQUIRE_CONFIRMATION: 1,
    ActionDecision.DENY: 2,
}


def strictest(*results: ActionDecisionResult) -> ActionDecisionResult:
    """Select the most restrictive independent action decision.

    Args:
        *results (ActionDecisionResult): Decisions in explanatory priority order.

    Returns:
        ActionDecisionResult: Strictest decision, keeping the first on ties.

    Raises:
        ValueError: No decisions are supplied.
    """

    return max(results, key=lambda result: _SEVERITY[result.decision])


def _evaluate_execution(action: ConcreteAction) -> ActionDecisionResult:
    if action.execution == ExecutionClass.NONE:
        return _result(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)

    capabilities = action.capabilities
    confined = (
        action.execution == ExecutionClass.CONFINED
        and capabilities.filesystem in (FilesystemCapability.WORKSPACE_ONLY, FilesystemCapability.SANDBOX)
        and capabilities.network == NetworkCapability.NONE
        and capabilities.process == ProcessCapability.SANDBOXED
        and capabilities.secrets == SecretsCapability.NONE
    )
    if not confined:
        return _result(ActionDecision.DENY, ActionDecisionReason.UNBOUNDED_CODE_EXECUTION)
    return _result(ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED)


def _evaluate_side_effect(action: ConcreteAction) -> ActionDecisionResult:
    if action.side_effect == SideEffectClass.PRIVILEGE_CHANGE:
        return _result(ActionDecision.DENY, ActionDecisionReason.PRIVILEGE_ESCALATION)
    if action.side_effect in (
        SideEffectClass.REVERSIBLE_WRITE,
        SideEffectClass.DESTRUCTIVE_WRITE,
        SideEffectClass.EXTERNAL_WRITE,
        # Running code is at least a write. Whether it may run at all is the
        # execution dimension's answer, and the strictest of the two wins.
        SideEffectClass.CODE_EXECUTION,
    ):
        return _result(ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED)
    if action.side_effect == SideEffectClass.NONE:
        return _result(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)
    # A member added to the enum and not to this policy is not "no effect".
    return _result(ActionDecision.DENY, ActionDecisionReason.POLICY_DENIED)


def _evaluate_egress(action: ConcreteAction) -> ActionDecisionResult:
    if action.egress == DataEgressClass.ARBITRARY_DESTINATION:
        return _result(ActionDecision.DENY, ActionDecisionReason.ARBITRARY_EGRESS)
    if action.egress == DataEgressClass.EXTERNAL_DESTINATION:
        return _result(ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED)
    if action.egress in (DataEgressClass.NONE, DataEgressClass.BOUNDED_DESTINATION):
        return _result(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)
    return _result(ActionDecision.DENY, ActionDecisionReason.POLICY_DENIED)


def _evaluate_scope(action: ConcreteAction) -> ActionDecisionResult:
    if action.scope == ScopeClass.HOST:
        return _result(ActionDecision.DENY, ActionDecisionReason.HOST_SCOPE)
    if action.scope in (
        ScopeClass.REQUEST, ScopeClass.SESSION, ScopeClass.WORKSPACE,
        ScopeClass.SANDBOX, ScopeClass.EXTERNAL,
    ):
        return _result(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)
    return _result(ActionDecision.DENY, ActionDecisionReason.POLICY_DENIED)


class DefaultActionPolicy:
    """Conservative platform policy for the action classes shipped in PI-3."""

    async def evaluate(self, review: ActionReview) -> ActionDecisionResult:
        # Only the typed action. Context, arguments and the destination's raw
        # target are there for policies that want them. This one rules on
        # classes alone and never on argument provenance.
        action = review.action
        return strictest(
            _evaluate_execution(action),
            _evaluate_side_effect(action),
            _evaluate_egress(action),
            _evaluate_scope(action),
        )
