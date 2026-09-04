# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Whether a built-in write is reversible has to survive a policy that cares.

`DefaultActionPolicy` sends every write class to the same place, so it cannot
tell a truthful classification from an untruthful one. The natural first policy
somebody writes *can*: allow reversible writes without asking, confirm the
destructive ones. Under that policy a misclassified overwrite is silently
permitted, which is why the classification is asserted here and not through the
default (ADR-0069).
"""

from __future__ import annotations

import pytest

import nlght.adapters.outbound.tools.builtin  # noqa: F401  (fills the registry)
from nlght.adapters.outbound.tools.registry import tool_registry
from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionExecutionContext,
    ActionResolutionContext,
    ConcreteAction,
    DataEgressClass,
    ExecutionCapabilities,
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
ARGUMENTS = {
    "path": "notes.txt", "content": "x", "message": "m",
    "ref": "main", "create": False,
}


class AllowReversibleWritesPolicy:
    """Plausible deployment policy: reversible writes are not worth a prompt."""

    async def evaluate(
        self,
        action: ConcreteAction,
        context: ActionExecutionContext,
    ) -> ActionDecisionResult:
        del context
        if action.side_effect is SideEffectClass.REVERSIBLE_WRITE:
            return ActionDecisionResult(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)
        if action.side_effect is SideEffectClass.DESTRUCTIVE_WRITE:
            return ActionDecisionResult(
                ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED,
            )
        return ActionDecisionResult(ActionDecision.DENY, ActionDecisionReason.POLICY_ERROR)


def _side_effect(signature_name: str) -> SideEffectClass:
    for cls in tool_registry._registry.values():  # noqa: SLF001
        for signature in cls.signatures():
            if signature.name == signature_name:
                context = ActionResolutionContext(
                    resource_address="kind/name",
                    resource_config={},
                    runtime_capabilities=CONFINED,
                )
                return signature.action.resolve(ARGUMENTS, context).side_effect
    raise AssertionError(f"no built-in signature named {signature_name}")


async def _decide(signature_name: str) -> ActionDecision:
    action = ConcreteAction(
        side_effect=_side_effect(signature_name),
        egress=DataEgressClass.NONE,
        scope=ScopeClass.SANDBOX,
        capabilities=CONFINED,
    )
    result = await AllowReversibleWritesPolicy().evaluate(
        action,
        ActionExecutionContext(
            request_id="r", correlation_id="c", principal=None, workflow="w",
            session_id=None, resource_address="kind/name",
            signature_name=signature_name, has_external_untrusted_input=True,
        ),
    )
    return result.decision


async def test_overwriting_a_file_is_never_allowed_as_a_reversible_write() -> None:
    """`shell_write_text` truncates a caller-chosen path; nothing brings it back."""
    assert _side_effect("shell_write_text") is SideEffectClass.DESTRUCTIVE_WRITE
    assert await _decide("shell_write_text") is ActionDecision.REQUIRE_CONFIRMATION


@pytest.mark.parametrize("signature", ["git_commit", "git_checkout", "shell_make_dir"])
async def test_writes_with_a_way_back_stay_reversible(signature: str) -> None:
    """The other half of the contract, or the fix becomes "call everything destructive".

    These three earn the class: git tracks a commit and a checkout, and a created
    directory can be removed. A test that only forbade `REVERSIBLE_WRITE` would be
    satisfied by making every write destructive, which would make the distinction
    useless rather than correct.
    """
    assert _side_effect(signature) is SideEffectClass.REVERSIBLE_WRITE
    assert await _decide(signature) is ActionDecision.ALLOW
