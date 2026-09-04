# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The one model-tool execution gate: bind, resolve, decide, approve, invoke."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from nlght.core.errors.errors import (
    ToolActionDeniedError,
    ToolActionSemanticsError,
    ToolArgumentsError,
)
from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionExecutionContext,
    ActionResolutionContext,
    ApprovalDecision,
    ConcreteAction,
    ExecutionCapabilities,
    JsonValue,
    PendingAction,
    ToolArgumentBinder,
    action_fingerprint,
)
from nlght.core.tools.tool import ToolParameter

if TYPE_CHECKING:
    from nlght.core.tools.action import ActionSemanticsResolver
    from nlght.ports.outbound.action_policy import ActionApprovalPort, ActionPolicy

logger = logging.getLogger(__name__)


class ExecutableToolContract(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> list[ToolParameter]: ...

    @property
    def action(self) -> ActionSemanticsResolver | None: ...

    @property
    def argument_binder(self) -> ToolArgumentBinder | None: ...

    @property
    def resource_address(self) -> str: ...

    @property
    def resource_config(self) -> Mapping[str, Any]: ...

    @property
    def runtime_capabilities(self) -> ExecutionCapabilities: ...

    async def _execute_bound(self, arguments: Mapping[str, JsonValue]) -> object: ...


@dataclass(frozen=True)
class PreparedToolInvocation:
    contract: ExecutableToolContract
    context: ActionExecutionContext
    action: ConcreteAction
    effective_arguments_json: str
    fingerprint: str


def _matches_type(value: object, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


def _validate_json_schema(value: object, schema: Mapping[str, Any], path: str) -> None:
    """Validate the deterministic JSON-Schema subset exposed by ToolParameter."""
    expected = schema.get("type")
    if expected is not None:
        if not isinstance(expected, str) or not _matches_type(value, expected):
            raise ToolArgumentsError(f"tool argument '{path}' must have JSON type {expected}")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ToolArgumentsError(f"tool argument '{path}' is outside its enum")

    items = schema.get("items")
    if isinstance(value, list) and items is not None:
        if not isinstance(items, Mapping):
            raise ToolArgumentsError(f"tool argument '{path}' has an invalid items schema")
        for index, item in enumerate(value):
            _validate_json_schema(item, items, f"{path}[{index}]")

    properties = schema.get("properties")
    if isinstance(value, dict) and properties is not None:
        if not isinstance(properties, Mapping):
            raise ToolArgumentsError(f"tool argument '{path}' has an invalid properties schema")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            raise ToolArgumentsError(f"tool argument '{path}' has an invalid required schema")
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolArgumentsError(
                f"tool argument '{path}' is missing required property: {missing[0]}"
            )
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                if not isinstance(child_schema, Mapping):
                    raise ToolArgumentsError(
                        f"tool argument '{path}.{name}' has an invalid property schema"
                    )
                _validate_json_schema(item, child_schema, f"{path}.{name}")
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                raise ToolArgumentsError(
                    f"tool argument '{path}' has unknown property: {name}"
                )
            if isinstance(additional, Mapping):
                _validate_json_schema(item, additional, f"{path}.{name}")


def bind_tool_arguments(
    parameters: list[ToolParameter],
    proposed: Mapping[str, Any],
) -> dict[str, JsonValue]:
    """Validate once and return the exact JSON arguments that may execute."""
    declared = {parameter.name: parameter for parameter in parameters}
    unknown = sorted(set(proposed) - set(declared))
    if unknown:
        raise ToolArgumentsError(f"unknown tool arguments: {', '.join(unknown)}")

    effective: dict[str, JsonValue] = {}
    for name, parameter in declared.items():
        if name in proposed:
            value = proposed[name]
        elif parameter.default is not None:
            value = copy.deepcopy(parameter.default)
        elif parameter.required:
            raise ToolArgumentsError(f"missing required tool argument: {name}")
        else:
            continue

        schema: dict[str, Any] = {"type": parameter.type}
        if parameter.enum is not None:
            schema["enum"] = parameter.enum
        if parameter.items is not None:
            schema["items"] = parameter.items
        _validate_json_schema(value, schema, name)
        effective[name] = copy.deepcopy(value)

    try:
        # JSON is both the immutability boundary and the representation used for
        # the approval fingerprint.  Non-JSON extension values never reach a tool.
        json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolArgumentsError("tool arguments must be finite JSON values") from exc
    return effective


class ActionGate:
    """Authorizes and invokes model-callable contracts through one final path."""

    def __init__(
        self,
        *,
        policy: ActionPolicy,
        approval: ActionApprovalPort | None = None,
    ) -> None:
        self._policy = policy
        self._approval = approval

    def prepare(
        self,
        contract: ExecutableToolContract,
        proposed_arguments: Mapping[str, Any],
        context: ActionExecutionContext,
    ) -> PreparedToolInvocation:
        resolver = contract.action
        if resolver is None:
            raise ToolActionSemanticsError(context.tool_subject)
        effective = bind_tool_arguments(contract.parameters, proposed_arguments)
        resolution_context = ActionResolutionContext(
            resource_address=contract.resource_address,
            resource_config=contract.resource_config,
            runtime_capabilities=contract.runtime_capabilities,
        )
        if contract.argument_binder is not None:
            try:
                effective = bind_tool_arguments(
                    contract.parameters,
                    contract.argument_binder.bind(effective, resolution_context),
                )
            except ToolArgumentsError:
                raise
            except Exception as exc:
                raise ToolArgumentsError("tool argument scope binding failed") from exc
        try:
            action = resolver.resolve(effective, resolution_context)
        except ToolArgumentsError:
            raise
        except Exception as exc:
            raise ToolActionDeniedError(
                context.tool_subject,
                ActionDecisionReason.ACTION_RESOLUTION_ERROR.value,
            ) from exc
        effective_json = json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        fingerprint = action_fingerprint(
            context=context,
            action=action,
            effective_arguments_json=effective_json,
        )
        return PreparedToolInvocation(
            contract=contract,
            context=context,
            action=action,
            effective_arguments_json=effective_json,
            fingerprint=fingerprint,
        )

    async def execute(self, prepared: PreparedToolInvocation) -> object:
        decision = await self._decide(prepared)
        approval_id: str | None = None

        if decision.decision == ActionDecision.DENY:
            self._log(prepared, decision)
            raise ToolActionDeniedError(
                prepared.context.tool_subject,
                decision.reason.value,
            )

        if decision.decision == ActionDecision.REQUIRE_CONFIRMATION:
            if self._approval is None:
                denied = replace(
                    decision,
                    decision=ActionDecision.DENY,
                    reason=ActionDecisionReason.APPROVAL_UNAVAILABLE,
                )
                self._log(prepared, denied)
                raise ToolActionDeniedError(
                    prepared.context.tool_subject,
                    denied.reason.value,
                )
            pending = PendingAction(
                context=prepared.context,
                action=prepared.action,
                action_fingerprint=prepared.fingerprint,
                effective_arguments_json=prepared.effective_arguments_json,
            )
            try:
                approval = await self._approval.approve(pending)
            except Exception as exc:
                denied = ActionDecisionResult(
                    ActionDecision.DENY,
                    ActionDecisionReason.APPROVAL_ERROR,
                )
                self._log(prepared, denied)
                raise ToolActionDeniedError(
                    prepared.context.tool_subject,
                    denied.reason.value,
                ) from exc
            approval_id = approval.approval_id
            if approval.decision != ApprovalDecision.APPROVED:
                denied = ActionDecisionResult(
                    ActionDecision.DENY,
                    ActionDecisionReason.APPROVAL_REJECTED,
                )
                self._log(prepared, denied, approval_id=approval_id)
                raise ToolActionDeniedError(
                    prepared.context.tool_subject,
                    denied.reason.value,
                )
            if approval.action_fingerprint != prepared.fingerprint:
                denied = ActionDecisionResult(
                    ActionDecision.DENY,
                    ActionDecisionReason.APPROVAL_MISMATCH,
                )
                self._log(prepared, denied, approval_id=approval_id)
                raise ToolActionDeniedError(
                    prepared.context.tool_subject,
                    denied.reason.value,
                )

        self._log(prepared, decision, approval_id=approval_id)
        arguments = json.loads(prepared.effective_arguments_json)
        return await prepared.contract._execute_bound(arguments)

    async def _decide(self, prepared: PreparedToolInvocation) -> ActionDecisionResult:
        try:
            result = await self._policy.evaluate(prepared.action, prepared.context)
        except Exception as exc:
            logger.warning(
                "action.policy_error | request_id=%s tool=%s error_type=%s",
                prepared.context.request_id,
                prepared.context.tool_subject,
                type(exc).__name__,
            )
            return ActionDecisionResult(ActionDecision.DENY, ActionDecisionReason.POLICY_ERROR)
        if not isinstance(result, ActionDecisionResult):
            return ActionDecisionResult(ActionDecision.DENY, ActionDecisionReason.POLICY_ERROR)
        return result

    @staticmethod
    def _log(
        prepared: PreparedToolInvocation,
        decision: ActionDecisionResult,
        *,
        approval_id: str | None = None,
    ) -> None:
        context = prepared.context
        action = prepared.action
        logger.info(
            "action.decision | request_id=%s correlation_id=%s workflow=%s principal=%s "
            "tool=%s side_effect=%s egress=%s scope=%s execution=%s decision=%s "
            "reason=%s action_id=%s approval_id=%s external_untrusted_input=%s",
            context.request_id,
            context.correlation_id,
            context.workflow,
            context.principal.id if context.principal else None,
            context.tool_subject,
            action.side_effect,
            action.egress,
            action.scope,
            action.execution,
            decision.decision,
            decision.reason,
            prepared.fingerprint,
            approval_id,
            context.has_external_untrusted_input,
        )
