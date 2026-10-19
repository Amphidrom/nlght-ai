# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

import pytest

from nlght.adapters.outbound.tools.action_gate import ActionGate, bind_tool_arguments
from nlght.adapters.outbound.tools.action_policy import DefaultActionPolicy
from nlght.adapters.outbound.tools.builtin.action_semantics import (
    EXECUTE_RUNTIME,
    HTTP_PROBE,
    READ_REQUEST,
    RelativePathBinder,
)
from nlght.adapters.outbound.tools.contract import CallbackToolContract
from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import ToolActionDeniedError, ToolArgumentsError
from nlght.core.model.trust import ContentExposure, ContentTrust, SourceRef
from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionExecutionContext,
    ActionReview,
    ApprovalDecision,
    ApprovalResult,
    ConcreteAction,
    DataEgressClass,
    ExecutionCapabilities,
    FilesystemCapability,
    NetworkCapability,
    ProcessCapability,
    ScopeClass,
    SecretsCapability,
    SideEffectClass,
    StaticActionSemantics,
)
from nlght.core.tools.tool import ToolParameter


def _context(*, external: bool = True) -> ActionExecutionContext:
    return ActionExecutionContext(
        request_id="request-7",
        correlation_id="correlation-8",
        principal=PrincipalRef("principal-9"),
        workflow="answer",
        session_id="session-10",
        resource_address="test/main",
        signature_name="operate",
        exposure=ContentExposure((SourceRef("web", "", "https://attacker.example", ContentTrust.UNTRUSTED),) if external else ()),
    )


def _contract(
    callback,
    *,
    action=READ_REQUEST,
    parameters: list[ToolParameter] | None = None,
    capabilities: ExecutionCapabilities | None = None,
) -> CallbackToolContract:
    return CallbackToolContract(
        name="operate",
        description="test",
        parameters=parameters or [],
        callback=callback,
        action=action,
        resource_address="test/main",
        runtime_capabilities=capabilities,
    )


class _FixedPolicy:
    def __init__(self, result: ActionDecisionResult) -> None:
        self.result = result
        self.calls = 0

    async def evaluate(self, review):
        self.calls += 1
        return self.result


@dataclass
class _Approval:
    decision: ApprovalDecision = ApprovalDecision.APPROVED
    wrong_fingerprint: bool = False
    raises: bool = False

    def __post_init__(self) -> None:
        self.pending = None

    async def approve(self, review):
        self.pending = review
        if self.raises:
            raise RuntimeError("approval backend failed")
        return ApprovalResult(
            decision=self.decision,
            action_fingerprint=("wrong" if self.wrong_fingerprint else review.action_fingerprint),
            approval_id="approval-11",
        )


def _review(action: ConcreteAction, context: ActionExecutionContext) -> ActionReview:
    return ActionReview(
        context=context, action=action, action_fingerprint="f", effective_arguments_json="{}",
    )


def test_argument_binding_validates_unknown_required_type_enum_and_defaults() -> None:
    parameters = [
        ToolParameter("required", "string"),
        ToolParameter("count", "integer", required=False, default=2),
        ToolParameter("mode", "string", enum=["safe"]),
    ]

    assert bind_tool_arguments(parameters, {"required": "yes", "mode": "safe"}) == {
        "required": "yes",
        "count": 2,
        "mode": "safe",
    }
    with pytest.raises(ToolArgumentsError, match="unknown"):
        bind_tool_arguments(parameters, {"required": "yes", "mode": "safe", "extra": 1})
    with pytest.raises(ToolArgumentsError, match="missing required"):
        bind_tool_arguments(parameters, {"mode": "safe"})
    with pytest.raises(ToolArgumentsError, match="JSON type integer"):
        bind_tool_arguments(parameters, {"required": "yes", "count": True, "mode": "safe"})
    with pytest.raises(ToolArgumentsError, match="outside its enum"):
        bind_tool_arguments(parameters, {"required": "yes", "mode": "unsafe"})


def test_argument_binding_validates_nested_array_item_schema() -> None:
    parameters = [
        ToolParameter(
            "records",
            "array",
            items={
                "type": "object",
                "required": ["id"],
                "additionalProperties": False,
                "properties": {"id": {"type": "integer"}},
            },
        )
    ]

    assert bind_tool_arguments(parameters, {"records": [{"id": 7}]}) == {
        "records": [{"id": 7}]
    }
    with pytest.raises(ToolArgumentsError, match=r"records\[0\].id.*integer"):
        bind_tool_arguments(parameters, {"records": [{"id": "7"}]})
    with pytest.raises(ToolArgumentsError, match="missing required property"):
        bind_tool_arguments(parameters, {"records": [{}]})
    with pytest.raises(ToolArgumentsError, match="unknown property"):
        bind_tool_arguments(parameters, {"records": [{"id": 7, "extra": True}]})


def test_action_fingerprint_binds_principal_and_untrusted_input_context() -> None:
    contract = _contract(lambda: "done")
    gate = ActionGate(policy=DefaultActionPolicy())

    baseline = gate.prepare(contract, {}, _context()).fingerprint
    other_principal = gate.prepare(
        contract,
        {},
        replace(_context(), principal=PrincipalRef("other-principal")),
    ).fingerprint
    trusted_only = gate.prepare(contract, {}, _context(external=False)).fingerprint

    assert baseline != other_principal
    assert baseline != trusted_only


async def test_host_scope_read_is_denied_with_machine_readable_reason() -> None:
    policy = DefaultActionPolicy()
    host_read = StaticActionSemantics(
        SideEffectClass.NONE,
        DataEgressClass.NONE,
        ScopeClass.HOST,
    ).resolve(
        {},
        type("Resolution", (), {"runtime_capabilities": ExecutionCapabilities.unconfined()})(),
    )

    assert await policy.evaluate(_review(host_read, _context())) == ActionDecisionResult(
        ActionDecision.DENY,
        ActionDecisionReason.HOST_SCOPE,
    )


async def test_allow_executes_the_exact_bound_arguments() -> None:
    seen: list[tuple[str, int]] = []

    def callback(*, text: str, count: int) -> str:
        seen.append((text, count))
        return "done"

    contract = _contract(
        callback,
        parameters=[
            ToolParameter("text", "string"),
            ToolParameter("count", "integer", required=False, default=2),
        ],
    )
    gate = ActionGate(policy=DefaultActionPolicy())
    prepared = gate.prepare(contract, {"text": "hello"}, _context())

    assert await gate.execute(prepared) == "done"
    assert seen == [("hello", 2)]


async def test_policy_and_tool_receive_bound_parameters_not_model_parameters() -> None:
    seen: list[str] = []
    contract = CallbackToolContract(
        name="operate",
        description="test",
        parameters=[ToolParameter("path", "string")],
        callback=lambda *, path: seen.append(path) or "done",
        action=READ_REQUEST,
        argument_binder=RelativePathBinder(fields=("path",)),
        resource_address="test/main",
    )
    gate = ActionGate(policy=DefaultActionPolicy())
    prepared = gate.prepare(contract, {"path": "a/./folder/file.txt"}, _context())

    assert '"path":"a/folder/file.txt"' in prepared.effective_arguments_json
    assert await gate.execute(prepared) == "done"
    assert seen == ["a/folder/file.txt"]

    with pytest.raises(ToolArgumentsError, match="escapes the repository scope"):
        gate.prepare(contract, {"path": "../../outside"}, _context())


async def test_hard_deny_never_calls_approval_or_tool() -> None:
    executed = False

    def callback() -> str:
        nonlocal executed
        executed = True
        return "unsafe"

    policy = _FixedPolicy(
        ActionDecisionResult(ActionDecision.DENY, ActionDecisionReason.POLICY_DENIED)
    )
    approval = _Approval()
    gate = ActionGate(policy=policy, approval=approval)
    prepared = gate.prepare(_contract(callback), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_denied"):
        await gate.execute(prepared)
    assert approval.pending is None
    assert executed is False


async def test_confirmation_without_port_fails_closed() -> None:
    executed = False

    def callback() -> str:
        nonlocal executed
        executed = True
        return "unsafe"

    gate = ActionGate(
        policy=_FixedPolicy(
            ActionDecisionResult(
                ActionDecision.REQUIRE_CONFIRMATION,
                ActionDecisionReason.APPROVAL_REQUIRED,
            )
        )
    )
    prepared = gate.prepare(_contract(callback), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="approval_unavailable"):
        await gate.execute(prepared)
    assert executed is False


@pytest.mark.parametrize(
    ("approval", "reason"),
    [
        (_Approval(decision=ApprovalDecision.REJECTED), "approval_rejected"),
        (_Approval(wrong_fingerprint=True), "approval_mismatch"),
        (_Approval(raises=True), "approval_error"),
    ],
)
async def test_rejected_mismatched_or_failing_approval_never_executes(
    approval: _Approval,
    reason: str,
) -> None:
    executed = False

    def callback() -> str:
        nonlocal executed
        executed = True
        return "unsafe"

    gate = ActionGate(
        policy=_FixedPolicy(
            ActionDecisionResult(
                ActionDecision.REQUIRE_CONFIRMATION,
                ActionDecisionReason.APPROVAL_REQUIRED,
            )
        ),
        approval=approval,
    )
    prepared = gate.prepare(_contract(callback), {}, _context())

    with pytest.raises(ToolActionDeniedError, match=reason):
        await gate.execute(prepared)
    assert executed is False


async def test_approval_is_bound_to_snapshot_not_mutable_proposed_arguments() -> None:
    seen: list[str] = []
    proposed = {"text": "approved"}
    approval = _Approval()
    gate = ActionGate(
        policy=_FixedPolicy(
            ActionDecisionResult(
                ActionDecision.REQUIRE_CONFIRMATION,
                ActionDecisionReason.APPROVAL_REQUIRED,
            )
        ),
        approval=approval,
    )
    contract = _contract(
        lambda *, text: seen.append(text) or "done",
        parameters=[ToolParameter("text", "string")],
    )
    prepared = gate.prepare(contract, proposed, _context())
    proposed["text"] = "changed after policy"

    assert await gate.execute(prepared) == "done"
    assert seen == ["approved"]
    assert approval.pending.action_fingerprint == prepared.fingerprint


async def test_policy_exception_fails_closed_without_execution() -> None:
    class _BrokenPolicy:
        async def evaluate(self, review):
            raise RuntimeError("broken")

    executed = False

    def callback() -> str:
        nonlocal executed
        executed = True
        return "unsafe"

    gate = ActionGate(policy=_BrokenPolicy())
    prepared = gate.prepare(_contract(callback), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_error"):
        await gate.execute(prepared)
    assert executed is False


async def test_confined_code_requires_confirmation_and_unconfined_code_is_denied() -> None:
    policy = DefaultActionPolicy()
    confined = EXECUTE_RUNTIME.resolve(
        {},
        type(
            "Resolution",
            (),
            {
                "runtime_capabilities": ExecutionCapabilities(
                    filesystem=FilesystemCapability.SANDBOX,
                    network=NetworkCapability.NONE,
                    process=ProcessCapability.SANDBOXED,
                    secrets=SecretsCapability.NONE,
                )
            },
        )(),
    )
    unconfined = EXECUTE_RUNTIME.resolve(
        {},
        type(
            "Resolution",
            (),
            {"runtime_capabilities": ExecutionCapabilities.unconfined()},
        )(),
    )

    assert (await policy.evaluate(_review(confined, _context()))).decision == ActionDecision.REQUIRE_CONFIRMATION
    denied = await policy.evaluate(_review(unconfined, _context()))
    assert denied == ActionDecisionResult(
        ActionDecision.DENY,
        ActionDecisionReason.UNBOUNDED_CODE_EXECUTION,
    )


def test_http_get_and_post_resolve_different_side_effects() -> None:
    resolution = type(
        "Resolution",
        (),
        {"runtime_capabilities": ExecutionCapabilities()},
    )()

    get = HTTP_PROBE.resolve({"url": "https://example.com", "method": "GET"}, resolution)
    post = HTTP_PROBE.resolve({"url": "https://example.com", "method": "POST"}, resolution)

    assert get.side_effect == SideEffectClass.NONE
    assert post.side_effect == SideEffectClass.EXTERNAL_WRITE
    assert get.egress == post.egress == DataEgressClass.ARBITRARY_DESTINATION


async def test_decision_log_omits_arguments_body_command_and_destination(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "SUPER-SECRET-COMMAND-AND-BODY"
    contract = _contract(
        lambda **_: "done",
        parameters=[ToolParameter("command", "string")],
    )
    gate = ActionGate(policy=DefaultActionPolicy())
    prepared = gate.prepare(contract, {"command": secret}, _context())

    with caplog.at_level(logging.INFO):
        await gate.execute(prepared)

    assert secret not in caplog.text
    assert "command" not in caplog.text
    assert "action.decision" in caplog.text


async def test_policy_and_approval_review_the_same_snapshot_that_executes() -> None:
    reviews = []

    class _RecordingPolicy:
        async def evaluate(self, review):
            reviews.append(review)
            # Mutating what the policy was given must not reach the tool.
            review.effective_arguments["value"] = "tampered"
            return ActionDecisionResult(
                ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED,
            )

    seen = []
    approval = _Approval()
    gate = ActionGate(policy=_RecordingPolicy(), approval=approval)
    prepared = gate.prepare(
        _contract(lambda *, value: seen.append(value) or "done",
                  parameters=[ToolParameter("value", "string")]),
        {"value": "bound"},
        _context(),
    )

    assert await gate.execute(prepared) == "done"
    assert approval.pending is reviews[0]
    assert reviews[0].action_fingerprint == prepared.fingerprint
    assert reviews[0].effective_arguments == {"value": "bound"}
    assert reviews[0].context.has_external_untrusted_input is True
    assert seen == ["bound"]


def test_a_network_destination_exposes_its_full_target_and_logs_only_its_origin_digest() -> None:
    resolution = type("Resolution", (), {"runtime_capabilities": ExecutionCapabilities()})()

    docs = HTTP_PROBE.resolve({"url": "https://example.com/docs?q=1", "method": "GET"}, resolution)
    admin = HTTP_PROBE.resolve({"url": "https://example.com/admin", "method": "GET"}, resolution)

    assert docs.destination is not None and admin.destination is not None
    # A policy can tell two paths apart ...
    assert docs.destination.target == "https://example.com/docs?q=1"
    assert admin.destination.target == "https://example.com/admin"
    # ... while correlation stays per origin, and the raw target never reaches a repr.
    assert docs.destination.identifier == admin.destination.identifier
    assert "docs" not in repr(docs) and "admin" not in repr(admin)


async def test_disabled_authorization_asks_nobody_and_says_so_in_the_audit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FixedPolicy(ActionDecisionResult(ActionDecision.DENY, ActionDecisionReason.POLICY_DENIED))
    approval = _Approval()
    seen = []
    gate = ActionGate(policy=policy, approval=approval, enabled=False)
    prepared = gate.prepare(
        _contract(lambda *, value: seen.append(value) or "done",
                  parameters=[ToolParameter("value", "string")]),
        {"value": "bound"},
        _context(),
    )

    with caplog.at_level(logging.INFO):
        assert await gate.execute(prepared) == "done"

    assert policy.calls == 0
    assert approval.pending is None
    assert seen == ["bound"]
    assert "reason=authorization_disabled" in caplog.text


def test_disabled_authorization_still_validates_and_binds_arguments() -> None:
    gate = ActionGate(policy=DefaultActionPolicy(), enabled=False)
    contract = _contract(lambda **_: "done", parameters=[ToolParameter("value", "string")])

    with pytest.raises(ToolArgumentsError):
        gate.prepare(contract, {"value": "x", "smuggled": "y"}, _context())


def _confirming_contract(executed: list[str]) -> CallbackToolContract:
    return _contract(lambda: executed.append("ran") or "done")


async def test_a_policy_past_its_deadline_denies_without_executing() -> None:
    class _SlowPolicy:
        async def evaluate(self, review):
            await asyncio.sleep(5)
            return ActionDecisionResult(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)

    executed: list[str] = []
    gate = ActionGate(policy=_SlowPolicy(), policy_timeout_seconds=0.01)
    prepared = gate.prepare(_confirming_contract(executed), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_timeout"):
        await gate.execute(prepared)
    assert executed == []


async def test_a_late_allow_is_void_even_when_the_policy_swallows_the_cancellation() -> None:
    class _StubbornPolicy:
        async def evaluate(self, review):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass  # ignores the deadline and answers anyway
            return ActionDecisionResult(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)

    executed: list[str] = []
    gate = ActionGate(policy=_StubbornPolicy(), policy_timeout_seconds=0.01)
    prepared = gate.prepare(_confirming_contract(executed), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_timeout"):
        await gate.execute(prepared)
    assert executed == []


async def test_a_policy_blocking_the_event_loop_past_its_deadline_is_void() -> None:
    class _BlockingPolicy:
        async def evaluate(self, review):
            time.sleep(0.05)  # synchronous: no deadline can interrupt this
            return ActionDecisionResult(ActionDecision.ALLOW, ActionDecisionReason.POLICY_ALLOWED)

    executed: list[str] = []
    gate = ActionGate(policy=_BlockingPolicy(), policy_timeout_seconds=0.01)
    prepared = gate.prepare(_confirming_contract(executed), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_timeout"):
        await gate.execute(prepared)
    assert executed == []


async def test_a_timeout_of_the_policys_own_before_the_deadline_is_a_policy_error() -> None:
    class _ClientTimeoutPolicy:
        async def evaluate(self, review):
            raise TimeoutError("upstream reviewer timed out")

    gate = ActionGate(policy=_ClientTimeoutPolicy(), policy_timeout_seconds=30)
    prepared = gate.prepare(_confirming_contract([]), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="policy_error"):
        await gate.execute(prepared)


async def test_an_approval_past_its_deadline_denies_without_executing() -> None:
    class _AbsentReviewer:
        async def approve(self, review):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass  # the person clicked "approve" after execution completed, which is too late.
            return ApprovalResult(ApprovalDecision.APPROVED, review.action_fingerprint, "late")

    executed: list[str] = []
    gate = ActionGate(
        policy=_FixedPolicy(
            ActionDecisionResult(ActionDecision.REQUIRE_CONFIRMATION, ActionDecisionReason.APPROVAL_REQUIRED)
        ),
        approval=_AbsentReviewer(),
        approval_timeout_seconds=0.01,
    )
    prepared = gate.prepare(_confirming_contract(executed), {}, _context())

    with pytest.raises(ToolActionDeniedError, match="approval_timeout"):
        await gate.execute(prepared)
    assert executed == []
