# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import pytest

from nlght.adapters.outbound.tools.action_policy import DefaultActionPolicy
from nlght.adapters.outbound.tools.action_registry import ActionAuthorizationRegistry
from nlght.core.errors.errors import ConfigurationError
from nlght.core.tools.action import ActionReview, ApprovalDecision, ApprovalResult


class _Approval:
    async def approve(self, review: ActionReview) -> ApprovalResult:
        return ApprovalResult(ApprovalDecision.REJECTED, review.action_fingerprint)


def test_empty_registry_keeps_the_built_in_behaviour() -> None:
    registry = ActionAuthorizationRegistry()

    assert registry.policy() is None
    assert registry.approval() is None


def test_registered_policy_and_approval_are_returned_as_the_same_instances() -> None:
    registry = ActionAuthorizationRegistry()
    policy = DefaultActionPolicy()
    approval = _Approval()

    registry.register_policy(policy)
    registry.register_approval(approval)

    assert registry.policy() is policy
    assert registry.approval() is approval


def test_a_second_policy_is_refused_and_the_first_stays() -> None:
    registry = ActionAuthorizationRegistry()
    first = DefaultActionPolicy()
    registry.register_policy(first)

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_policy(DefaultActionPolicy())
    assert registry.policy() is first


def test_a_second_approval_is_refused_and_the_first_stays() -> None:
    registry = ActionAuthorizationRegistry()
    first = _Approval()
    registry.register_approval(first)

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_approval(_Approval())
    assert registry.approval() is first


def test_an_object_that_is_not_the_port_is_refused_at_registration() -> None:
    registry = ActionAuthorizationRegistry()

    with pytest.raises(TypeError, match="ActionPolicy"):
        registry.register_policy(_Approval())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ActionApprovalPort"):
        registry.register_approval(DefaultActionPolicy())  # type: ignore[arg-type]
    assert registry.policy() is None
    assert registry.approval() is None
