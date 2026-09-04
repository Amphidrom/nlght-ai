# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Provider-neutral security semantics for model-proposed actions.

These types describe what the runtime can prove about an action.  They never
attempt to infer why the model proposed it or which sentence influenced an
argument.  Authorization consumes validated effective arguments and enforced
runtime capabilities only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.entry.context import PrincipalRef

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ActionDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class ActionDecisionReason(StrEnum):
    POLICY_ALLOWED = "policy_allowed"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIED = "policy_denied"
    UNKNOWN_ACTION_SEMANTICS = "unknown_action_semantics"
    ARBITRARY_EGRESS = "arbitrary_egress"
    HOST_SCOPE = "host_scope"
    UNBOUNDED_CODE_EXECUTION = "unbounded_code_execution"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_MISMATCH = "approval_mismatch"
    PARAMETER_SCOPE_VIOLATION = "parameter_scope_violation"
    INVALID_ARGUMENTS = "invalid_arguments"
    ACTION_RESOLUTION_ERROR = "action_resolution_error"
    POLICY_ERROR = "policy_error"
    APPROVAL_ERROR = "approval_error"


class SideEffectClass(StrEnum):
    NONE = "none"
    REVERSIBLE_WRITE = "reversible_write"
    DESTRUCTIVE_WRITE = "destructive_write"
    EXTERNAL_WRITE = "external_write"
    PRIVILEGE_CHANGE = "privilege_change"
    CODE_EXECUTION = "code_execution"


class DataEgressClass(StrEnum):
    NONE = "none"
    BOUNDED_DESTINATION = "bounded_destination"
    EXTERNAL_DESTINATION = "external_destination"
    ARBITRARY_DESTINATION = "arbitrary_destination"


class ScopeClass(StrEnum):
    REQUEST = "request"
    SESSION = "session"
    WORKSPACE = "workspace"
    SANDBOX = "sandbox"
    EXTERNAL = "external"
    HOST = "host"


class ExecutionClass(StrEnum):
    NONE = "none"
    CONFINED = "confined"
    UNCONFINED = "unconfined"


class FilesystemCapability(StrEnum):
    NONE = "none"
    WORKSPACE_ONLY = "workspace_only"
    SANDBOX = "sandbox"
    HOST = "host"


class NetworkCapability(StrEnum):
    NONE = "none"
    BOUNDED = "bounded"
    ARBITRARY = "arbitrary"


class ProcessCapability(StrEnum):
    NONE = "none"
    SANDBOXED = "sandboxed"
    HOST = "host"


class SecretsCapability(StrEnum):
    NONE = "none"
    MAY_READ = "may_read"


class DestinationKind(StrEnum):
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    GIT_REMOTE = "git_remote"
    SESSION = "session"
    RESOURCE = "resource"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionCapabilities:
    filesystem: FilesystemCapability = FilesystemCapability.NONE
    network: NetworkCapability = NetworkCapability.NONE
    process: ProcessCapability = ProcessCapability.NONE
    secrets: SecretsCapability = SecretsCapability.NONE

    @classmethod
    def unconfined(cls) -> ExecutionCapabilities:
        """Conservative answer for runtimes that expose no capability contract."""
        return cls(
            filesystem=FilesystemCapability.HOST,
            network=NetworkCapability.ARBITRARY,
            process=ProcessCapability.HOST,
            secrets=SecretsCapability.MAY_READ,
        )


@dataclass(frozen=True)
class BoundDestination:
    """A destination identity safe to correlate without logging its raw value."""

    kind: DestinationKind
    scope: ScopeClass
    identifier: str

    @classmethod
    def from_value(
        cls,
        *,
        kind: DestinationKind,
        scope: ScopeClass,
        value: str,
    ) -> BoundDestination:
        digest = hashlib.sha256(f"{kind.value}\0{value}".encode()).hexdigest()
        return cls(kind=kind, scope=scope, identifier=f"sha256:{digest}")


@dataclass(frozen=True)
class ConcreteAction:
    side_effect: SideEffectClass
    egress: DataEgressClass
    scope: ScopeClass
    execution: ExecutionClass = ExecutionClass.NONE
    capabilities: ExecutionCapabilities = field(default_factory=ExecutionCapabilities)
    destination: BoundDestination | None = None


@dataclass(frozen=True)
class ActionResolutionContext:
    """Trusted facts available while resolving one signature invocation."""

    resource_address: str
    resource_config: Mapping[str, Any]
    runtime_capabilities: ExecutionCapabilities


@runtime_checkable
class ActionSemanticsResolver(Protocol):
    """Deterministically turns bound JSON arguments into one concrete action."""

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction: ...


@runtime_checkable
class ToolArgumentBinder(Protocol):
    """Constrain schema-valid model arguments into their effective JSON form."""

    def bind(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True)
class StaticActionSemantics:
    """Action semantics that do not vary with individual arguments."""

    side_effect: SideEffectClass
    egress: DataEgressClass
    scope: ScopeClass
    execution: ExecutionClass = ExecutionClass.NONE

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        del arguments
        scope = self.scope
        if scope == ScopeClass.SANDBOX:
            filesystem = context.runtime_capabilities.filesystem
            if filesystem == FilesystemCapability.HOST:
                scope = ScopeClass.HOST
            elif filesystem == FilesystemCapability.WORKSPACE_ONLY:
                scope = ScopeClass.WORKSPACE
        execution = self.execution
        if execution != ExecutionClass.NONE:
            process = context.runtime_capabilities.process
            execution = (
                ExecutionClass.CONFINED
                if process == ProcessCapability.SANDBOXED
                else ExecutionClass.UNCONFINED
            )
        return ConcreteAction(
            side_effect=self.side_effect,
            egress=self.egress,
            scope=scope,
            execution=execution,
            capabilities=context.runtime_capabilities,
        )


@dataclass(frozen=True)
class ActionExecutionContext:
    request_id: str
    correlation_id: str
    principal: PrincipalRef | None
    workflow: str | None
    session_id: str | None
    resource_address: str
    signature_name: str
    has_external_untrusted_input: bool

    @property
    def tool_subject(self) -> str:
        return f"{self.resource_address}::{self.signature_name}"


@dataclass(frozen=True)
class ActionDecisionResult:
    decision: ActionDecision
    reason: ActionDecisionReason


@dataclass(frozen=True)
class PendingAction:
    """One immutable effective action presented for approval.

    ``effective_arguments_json`` may contain sensitive values and is for the
    approval adapter only.  It must never be copied into decision logs.
    """

    context: ActionExecutionContext
    action: ConcreteAction
    action_fingerprint: str
    effective_arguments_json: str


@dataclass(frozen=True)
class ApprovalResult:
    decision: ApprovalDecision
    action_fingerprint: str
    approval_id: str | None = None


def action_fingerprint(
    *,
    context: ActionExecutionContext,
    action: ConcreteAction,
    effective_arguments_json: str,
) -> str:
    """Bind approval to the tool, trusted context, action and exact arguments."""
    material = {
        "request_id": context.request_id,
        "correlation_id": context.correlation_id,
        "principal_id": context.principal.id if context.principal else None,
        "workflow": context.workflow,
        "session_id": context.session_id,
        "resource_address": context.resource_address,
        "signature_name": context.signature_name,
        "has_external_untrusted_input": context.has_external_untrusted_input,
        "action": asdict(action),
        "effective_arguments": json.loads(effective_arguments_json),
    }
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
