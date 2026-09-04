# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""No risk dimension may be masked by another one's verdict.

The old policy was an ordered `if` chain: the first branch that returned hid
every later one. It gave the right answers, but correctness rested on the order,
and an order is exactly the kind of thing a later refactor moves without
noticing. The composition is now a maximum over four independent judgements, and
these tests hold that property over every combination rather than over the
handful the built-in tools happen to produce today (ADR-0069).
"""

from __future__ import annotations

import itertools

import pytest

from nlght.adapters.outbound.tools.action_policy import (
    DefaultActionPolicy,
    _evaluate_egress,
    _evaluate_execution,
    _evaluate_scope,
    _evaluate_side_effect,
    strictest,
)
from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionExecutionContext,
    ConcreteAction,
    DataEgressClass,
    ExecutionCapabilities,
    ExecutionClass,
    FilesystemCapability,
    NetworkCapability,
    ProcessCapability,
    ScopeClass,
    SecretsCapability,
    SideEffectClass,
)

CONFINED = ExecutionCapabilities(
    filesystem=FilesystemCapability.SANDBOX,
    network=NetworkCapability.NONE,
    process=ProcessCapability.SANDBOXED,
    secrets=SecretsCapability.NONE,
)
HOSTED = ExecutionCapabilities(
    filesystem=FilesystemCapability.HOST,
    network=NetworkCapability.ARBITRARY,
    process=ProcessCapability.HOST,
    secrets=SecretsCapability.MAY_READ,
)
CONTEXT = ActionExecutionContext(
    request_id="r", correlation_id="c", principal=None, workflow="w",
    session_id=None, resource_address="kind/name", signature_name="op",
    has_external_untrusted_input=True,
)


def _every_action():  # noqa: ANN202
    product = itertools.product(
        SideEffectClass, DataEgressClass, ScopeClass, ExecutionClass, (CONFINED, HOSTED),
    )
    for side_effect, egress, scope, execution, capabilities in product:
        yield ConcreteAction(
            side_effect=side_effect, egress=egress, scope=scope,
            execution=execution, capabilities=capabilities,
        )


async def test_no_dimension_is_ever_masked_by_another() -> None:
    """For every combination, the verdict is at least as strict as each dimension."""
    order = {ActionDecision.ALLOW: 0, ActionDecision.REQUIRE_CONFIRMATION: 1, ActionDecision.DENY: 2}
    policy = DefaultActionPolicy()
    checked = 0

    for action in _every_action():
        combined = await policy.evaluate(action, CONTEXT)
        dimensions = (
            _evaluate_execution(action),
            _evaluate_side_effect(action),
            _evaluate_egress(action),
            _evaluate_scope(action),
        )
        checked += 1
        for dimension in dimensions:
            assert order[combined.decision] >= order[dimension.decision], (
                f"{dimension.decision} on one dimension was softened to "
                f"{combined.decision} for {action}"
            )
        assert order[combined.decision] == max(order[d.decision] for d in dimensions)

    assert checked == 6 * 4 * 6 * 3 * 2


async def test_a_single_denying_dimension_denies_the_action() -> None:
    """The property that an ordered chain cannot promise: one veto is enough."""
    policy = DefaultActionPolicy()

    # Everything harmless except the destination.
    action = ConcreteAction(
        side_effect=SideEffectClass.NONE,
        egress=DataEgressClass.ARBITRARY_DESTINATION,
        scope=ScopeClass.REQUEST,
        execution=ExecutionClass.NONE,
        capabilities=CONFINED,
    )

    result = await policy.evaluate(action, CONTEXT)

    assert result.decision is ActionDecision.DENY
    assert result.reason is ActionDecisionReason.ARBITRARY_EGRESS


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (ActionDecision.ALLOW, ActionDecision.DENY, ActionDecision.DENY),
        (ActionDecision.DENY, ActionDecision.ALLOW, ActionDecision.DENY),
        (ActionDecision.ALLOW, ActionDecision.REQUIRE_CONFIRMATION, ActionDecision.REQUIRE_CONFIRMATION),
        (ActionDecision.REQUIRE_CONFIRMATION, ActionDecision.DENY, ActionDecision.DENY),
        (ActionDecision.ALLOW, ActionDecision.ALLOW, ActionDecision.ALLOW),
    ],
)
def test_strictest_is_independent_of_argument_order(
    first: ActionDecision,
    second: ActionDecision,
    expected: ActionDecision,
) -> None:
    reason = ActionDecisionReason.POLICY_ALLOWED
    a, b = ActionDecisionResult(first, reason), ActionDecisionResult(second, reason)

    assert strictest(a, b).decision is expected
    assert strictest(b, a).decision is expected
